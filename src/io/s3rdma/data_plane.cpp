/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/io/rest/curl_handle.hpp>
#include <cucascade/io/s3rdma/cuobj_abi.h>
#include <cucascade/io/s3rdma/data_plane.hpp>
#include <cucascade/io/s3rdma/errors.hpp>
#include <cucascade/log/logging.hpp>

#include <curl/curl.h>
#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade::io::s3rdma {

namespace {

using rest::curl_easy_ptr;
using rest::curl_slist_ptr;

constexpr std::string_view token_header{"x-amz-rdma-token"};
constexpr std::string_view reply_header{"x-amz-rdma-reply"};

/// What the request callback returns for every failure.  The client repeats a
/// callback that fails with one of its retryable values; this one it does not.
constexpr std::int64_t refused = -ENOTSUP;

/// A response to the request of a GET carries no payload: the bytes travel
/// over RDMA.  A body beyond this is not read.
constexpr std::size_t body_limit = std::size_t{64} << 10;

char ascii_lower(char c) noexcept
{
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

std::string lowered(std::string_view text)
{
  std::string out{text};
  std::transform(out.begin(), out.end(), out.begin(), ascii_lower);
  return out;
}

error unavailable(std::string what) { return error{errc::backend_unavailable, std::move(what)}; }

extern "C" {
typedef cucascade_s3rdma_api const* (*binding_entry)(void);
}

//===----------------------------------------------------------------------===//
// Endpoint topology
//===----------------------------------------------------------------------===//

struct endpoint_parts {
  std::string scheme;
  std::string host;
  std::string port;
  std::string path;

  bool operator==(endpoint_parts const&) const = default;
};

endpoint_parts parts_of(std::string_view endpoint)
{
  endpoint_parts out;
  auto remainder = endpoint;
  if (auto const scheme_end = endpoint.find("://"); scheme_end != std::string_view::npos) {
    out.scheme = lowered(endpoint.substr(0, scheme_end));
    remainder  = endpoint.substr(scheme_end + 3);
  }
  auto const path_begin = remainder.find('/');
  auto const authority  = remainder.substr(0, path_begin);
  if (path_begin != std::string_view::npos) {
    auto path = remainder.substr(path_begin);
    if (path.back() == '/') { path.remove_suffix(1); }
    out.path.assign(path);
  }
  // A trailing ":digits" is the port.  Behind a bracketed IPv6 literal only a
  // colon that follows the closing bracket starts one: the colons of the
  // literal stay in the host.
  auto const colon    = authority.rfind(':');
  auto const bracket  = authority.find(']');
  bool const literal  = !authority.empty() && authority.front() == '[';
  bool const has_port = !authority.empty() && colon != std::string_view::npos &&
                        colon + 1 < authority.size() &&
                        (!literal || (bracket != std::string_view::npos && colon == bracket + 1)) &&
                        std::all_of(authority.begin() + static_cast<std::ptrdiff_t>(colon) + 1,
                                    authority.end(),
                                    [](char c) { return c >= '0' && c <= '9'; });
  out.host = lowered(has_port ? authority.substr(0, colon) : authority);
  if (has_port) { out.port.assign(authority.substr(colon + 1)); }
  if (out.port.empty()) {
    out.port = out.scheme == "http" ? "80" : out.scheme == "https" ? "443" : "";
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Binding library
//===----------------------------------------------------------------------===//

/**
 * The table of the binding library at @p path, after its handshake and probe.
 *
 * A library is loaded once and never unloaded, also when it is then refused:
 * it may have initialized an RDMA client, which does not survive being
 * unloaded.  Nothing but the given path is consulted.
 */
cucascade_s3rdma_api const& load_binding(std::string const& path, bool require_zero_io_retry)
{
  static std::mutex mutex;
  static std::map<std::string, cucascade_s3rdma_api const*> accepted;
  std::lock_guard lock{mutex};

  cucascade_s3rdma_api const* api = nullptr;
  if (auto const found = accepted.find(path); found != accepted.end()) {
    api = found->second;
  } else {
    void* const library = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      char const* const why = ::dlerror();
      throw unavailable(std::string{"s3rdma: the binding library could not be loaded: "} +
                        (why != nullptr ? why : "unknown error"));
    }
    void* const symbol = ::dlsym(library, CUCASCADE_S3RDMA_BINDING_SYMBOL);
    if (symbol == nullptr) {
      throw unavailable("s3rdma: the binding library does not export " +
                        std::string{CUCASCADE_S3RDMA_BINDING_SYMBOL});
    }
    // An object pointer does not convert to a function pointer; the bytes do.
    binding_entry entry = nullptr;
    static_assert(sizeof(entry) == sizeof(symbol));
    std::memcpy(&entry, &symbol, sizeof(entry));
    api = entry();
    if (api == nullptr || api->abi_version != CUCASCADE_S3RDMA_ABI_VERSION ||
        api->api_size != sizeof(cucascade_s3rdma_api)) {
      throw unavailable("s3rdma: the binding library was built for another interface version");
    }
    if (api->probe == nullptr || api->session_create == nullptr ||
        api->session_destroy == nullptr || api->register_memory == nullptr ||
        api->deregister_memory == nullptr || api->max_request_size == nullptr ||
        api->get == nullptr) {
      throw unavailable("s3rdma: the binding library's interface is incomplete");
    }
    accepted.emplace(path, api);
  }

  cucascade_s3rdma_probe probe{};
  probe.io_retry_count = -1;
  if (api->probe(&probe) != 0) { throw unavailable("s3rdma: the binding library's probe failed"); }
  if (probe.protocol != CUCASCADE_S3RDMA_PROTOCOL_DC_V1) {
    throw unavailable("s3rdma: the binding library uses a transfer protocol other than DC_V1");
  }
  if (probe.io_retry_count != 0) {
    if (require_zero_io_retry) {
      throw unavailable(
        "s3rdma: the RDMA client's I/O retry count is not zero (rdma_io_retry_count), or unknown");
    }
    CUCASCADE_LOG_WARN(
      "s3rdma: the RDMA client's I/O retry count is {}; zero is recommended "
      "(rdma_io_retry_count). A repeated request callback is refused either way.",
      probe.io_retry_count < 0 ? std::string{"unknown"} : std::to_string(probe.io_retry_count));
  }
  return *api;
}

//===----------------------------------------------------------------------===//
// The request of one GET
//===----------------------------------------------------------------------===//

/// The response to the request of a GET.  Each status line starts over, so
/// only the last response of the transfer is kept.
struct data_response {
  std::string_view bytes_name{};  ///< lowercase; empty when no count header is configured
  long status{0};
  std::string reply{};
  unsigned replies{0};
  std::string etag{};
  std::string bytes{};
  unsigned counts{0};
  std::size_t body{0};

  void take(std::string_view line)
  {
    constexpr std::string_view http{"http/"};
    if (line.size() >= http.size() &&
        std::equal(http.begin(), http.end(), line.begin(), [](char expected, char got) {
          return expected == ascii_lower(got);
        })) {
      *this            = data_response{bytes_name};
      auto const space = line.find(' ');
      for (auto i = space == std::string_view::npos ? line.size() : space + 1;
           i < line.size() && line[i] >= '0' && line[i] <= '9' && status < 100000;
           ++i) {
        status = status * 10 + (line[i] - '0');
      }
      return;
    }
    auto const colon = line.find(':');
    if (colon == std::string_view::npos) { return; }
    auto const name = lowered(line.substr(0, colon));
    auto value      = line.substr(colon + 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' ||
                              value.back() == '\t')) {
      value.remove_suffix(1);
    }
    if (name == reply_header) {
      reply.assign(value);
      ++replies;
    } else if (name == "etag") {
      etag.assign(value);
    } else if (!bytes_name.empty() && name == bytes_name) {
      bytes.assign(value);
      ++counts;
    }
  }
};

std::size_t on_data_header(char* buffer,
                           std::size_t size,
                           std::size_t count,
                           void* userdata) noexcept
{
  try {
    static_cast<data_response*>(userdata)->take(std::string_view{buffer, size * count});
    return size * count;
  } catch (...) {
    return 0;
  }
}

std::size_t on_data_body(char* /*data*/,
                         std::size_t size,
                         std::size_t count,
                         void* userdata) noexcept
{
  auto* response = static_cast<data_response*>(userdata);
  response->body += size * count;
  return response->body > body_limit ? 0 : size * count;
}

std::optional<std::size_t> decimal(std::string_view text) noexcept
{
  if (text.empty()) { return std::nullopt; }
  std::size_t value = 0;
  for (char const c : text) {
    if (c < '0' || c > '9') { return std::nullopt; }
    auto const digit = static_cast<std::size_t>(c - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) { return std::nullopt; }
    value = value * 10 + digit;
  }
  return value;
}

curl_slist_ptr header_list(std::vector<std::pair<std::string, std::string>> const& headers)
{
  curl_slist_ptr list;
  for (auto const& [name, value] : headers) {
    auto* next = curl_slist_append(list.get(), (name + ": " + value).c_str());
    if (next == nullptr) { throw std::bad_alloc(); }
    std::ignore = list.release();
    list.reset(next);
  }
  return list;
}

/// What is fixed for every GET of a factory's sessions.
struct data_plane {
  std::shared_ptr<rest::request_authorizer> authorizer;
  completion_rule rule;
  data_plane_config cfg;
  std::string bytes_name;  ///< @c cfg.bytes_header in lowercase
  cucascade_s3rdma_api const* api{nullptr};
};

/// A callback identity is carried where the client keeps a pointer.
static_assert(sizeof(void*) == sizeof(std::uint64_t));

/// How many bits of a callback identity number the GETs of one session; the
/// remaining ones number the sessions of the process.
constexpr unsigned sequence_bits      = 40;
constexpr std::uint64_t sequence_mask = (std::uint64_t{1} << sequence_bits) - 1;
constexpr std::uint64_t last_serial   = (std::uint64_t{1} << (64 - sequence_bits)) - 1;

/**
 * One GET as its request callback sees it, and what that callback observed.
 *
 * The client may run the callback more than once for one GET — a retry, a
 * continuation — or with another shape than asked.  Only the first run, with
 * offset zero and the whole length, sends a request; every other run is
 * refused without sending anything.
 *
 * What the client hands back to a callback is a number, not an address: the
 * serial of the session and the sequence of the GET within it.  Neither is
 * used twice.  A callback of a GET that has ended can therefore never be
 * taken for a later GET, however late it runs: it is refused, sends nothing,
 * touches nothing the ended GET borrowed from its caller, and fails the GET
 * that is in progress or comes next.  A callback of a session that is gone
 * finds no session.
 *
 * The gate is held for a whole callback and while a GET starts and ends.  A
 * GET therefore does not end while its request is on the way.
 *
 * Nothing here is logged and no message carries a header value: the token is
 * a capability.
 */
struct get_state {
  get_state()                            = default;
  get_state(get_state const&)            = delete;
  get_state& operator=(get_state const&) = delete;

  std::mutex gate;

  std::uint64_t serial{0};    ///< of the session; set once, before its first GET
  std::uint64_t sequence{0};  ///< of the GET in progress; zero between GETs
  std::uint64_t issued{0};    ///< sequences handed out

  data_plane const* plane{nullptr};
  CURL* handle{nullptr};
  object_ref const* object{nullptr};  ///< the caller's; valid until the GET ends
  std::size_t object_offset{0};
  std::size_t size{0};
  std::string_view condition;  ///< the caller's; valid until the GET ends

  std::uint32_t runs{0};    ///< callbacks run for the GET in progress
  std::uint32_t stale{0};   ///< callbacks run, since a GET last ended, for a GET that had ended
  bool handed_over{false};  ///< the request was given to the transport
  bool answered{false};     ///< a whole response arrived
  long status{0};
  std::string reply;
  std::string etag;
  std::size_t delivered{0};
  std::string failure;  ///< a local or transport failure; empty otherwise

  /// Starts a GET.  Returns what the client hands back to its callbacks, or
  /// zero when the session has used every sequence.
  [[nodiscard]] std::uint64_t begin(data_plane const& owner,
                                    CURL* easy,
                                    object_ref const& target,
                                    std::size_t offset,
                                    std::size_t length,
                                    std::string_view if_match)
  {
    std::lock_guard lock{gate};
    if (issued == sequence_mask) { return 0; }
    sequence      = ++issued;
    plane         = &owner;
    handle        = easy;
    object        = &target;
    object_offset = offset;
    size          = length;
    condition     = if_match;
    runs          = 0;
    handed_over   = false;
    answered      = false;
    status        = 0;
    reply.clear();
    etag.clear();
    delivered = 0;
    failure.clear();
    return (serial << sequence_bits) | sequence;
  }

  /// Ends the GET in progress.  Returns the callbacks run for it plus those
  /// run, since a GET last ended, for a GET that had already ended: the
  /// client did something this backend does not account for, and the GET is
  /// reported as failed.
  [[nodiscard]] std::uint32_t end()
  {
    std::lock_guard lock{gate};
    sequence         = 0;
    auto const total = std::uint64_t{runs} + std::exchange(stale, 0U);
    if (total != runs && failure.empty()) {
      failure = "the client ran a request callback of a GET that had ended";
    }
    return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(total, std::numeric_limits<std::uint32_t>::max()));
  }
};

/**
 * The data sessions of the process by serial.  A callback finds its session
 * here and nowhere else, so one that runs after the session is gone finds
 * nothing.  Never destroyed: the client may run a callback while the process
 * exits.
 */
class session_registry {
 public:
  /// Registers @p state under a serial of its own; false when the process has
  /// used every serial.
  [[nodiscard]] bool enter(std::shared_ptr<get_state> const& state)
  {
    std::lock_guard lock{_mutex};
    if (_issued == last_serial) { return false; }
    state->serial = ++_issued;
    _sessions.emplace(state->serial, state);
    return true;
  }

  void leave(std::uint64_t serial) noexcept
  {
    try {
      std::lock_guard lock{_mutex};
      _sessions.erase(serial);
    } catch (...) {
      // The state stays registered and takes no further GET.
    }
  }

  [[nodiscard]] std::shared_ptr<get_state> find(std::uint64_t serial)
  {
    std::lock_guard lock{_mutex};
    auto const found = _sessions.find(serial);
    return found == _sessions.end() ? nullptr : found->second;
  }

 private:
  std::mutex _mutex;
  std::uint64_t _issued{0};
  std::unordered_map<std::uint64_t, std::shared_ptr<get_state>> _sessions;
};

session_registry& sessions()
{
  static auto* const registry = new session_registry;
  return *registry;
}

/// Send the one request of @p state and judge its response.  The byte count
/// on success; @c refused otherwise, with @p state saying why.
std::int64_t send_request(get_state& state, cucascade_s3rdma_request const& request)
{
  auto const& plane = *state.plane;
  if (request.descriptor == nullptr || request.offset != 0 || request.size != state.size) {
    state.failure = "the client ran the request callback with an unexpected offset or size";
    return refused;
  }
  std::string_view token{request.descriptor, request.descriptor_length};
  while (!token.empty() && token.back() == '\0') {
    token.remove_suffix(1);
  }
  // Visible ASCII only: anything else has no single form as a header value.
  if (token.empty() || !std::all_of(token.begin(), token.end(), [](char c) {
        return static_cast<unsigned char>(c) > 0x20 && static_cast<unsigned char>(c) < 0x7f;
      })) {
    state.failure = "the RDMA descriptor cannot be sent as a header value";
    return refused;
  }

  // Token, range and condition are part of what is signed: a request that
  // differs in any of them does not verify.
  std::vector<std::pair<std::string, std::string>> to_sign;
  to_sign.emplace_back(std::string{token_header}, std::string{token});
  to_sign.emplace_back("range",
                       "bytes=" + std::to_string(state.object_offset) + "-" +
                         std::to_string(state.object_offset + state.size - 1));
  if (!state.condition.empty()) { to_sign.emplace_back("if-match", std::string{state.condition}); }
  rest::authorized_request authorized;
  try {
    authorized = plane.authorizer->authorize_with_headers(
      *state.object,
      rest::request_method::GET,
      to_sign,
      std::chrono::ceil<std::chrono::seconds>(plane.cfg.request_deadline) +
        std::chrono::seconds{60});
  } catch (...) {
    state.failure = "the request could not be authorized";
    return refused;
  }
  auto const headers = header_list(authorized.headers);

  // Exactly one request on the wire.  The transport sends a request again by
  // itself in two cases, and both are excluded: on a reused connection that
  // turns out to be dead, so the connection is new and is not kept; and when
  // an HTTP/2 server refuses the stream, so the request is HTTP/1.1.  No
  // redirect is followed and no proxy is used, the environment's included.
  CURL* const easy = state.handle;
  curl_easy_reset(easy);
  rest::configure_easy_handle(easy, rest::global_curl_context::instance().share_handle());
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_FRESH_CONNECT, 1L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_FORBID_REUSE, 1L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 0L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_PROXY, ""));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_NOPROXY, "*"));
  // The path is sent as it was signed: "." and ".." are legal in an object
  // key, and the transport would otherwise remove such segments.
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_PATH_AS_IS, 1L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(
    easy, CURLOPT_TIMEOUT_MS, static_cast<long>(plane.cfg.request_deadline.count())));
  if (!plane.cfg.ca_bundle_path.empty()) {
    CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_CAINFO, plane.cfg.ca_bundle_path.c_str()));
  }
  if (!plane.cfg.tls_verify) {
    CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, 0L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_SSL_VERIFYHOST, 0L));
  }
  data_response response{plane.bytes_name};
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_URL, authorized.url.c_str()));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_HTTPGET, 1L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers.get()));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &on_data_body));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_WRITEDATA, &response));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, &on_data_header));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(easy, CURLOPT_HEADERDATA, &response));

  state.handed_over = true;
  auto const rc     = curl_easy_perform(easy);
  state.status      = response.status;
  state.etag        = std::move(response.etag);
  if (response.replies == 1) { state.reply = std::move(response.reply); }
  if (rc != CURLE_OK) {
    state.failure = curl_easy_strerror(rc);
    return refused;
  }
  state.answered = true;

  // The count is reported to the client only for a response that passes every
  // check the backend applies afterwards; anything else is a failure here.
  auto const& rule = plane.rule;
  bool const accepted =
    std::find(rule.accepted_statuses.begin(), rule.accepted_statuses.end(), state.status) !=
      rule.accepted_statuses.end() &&
    !state.reply.empty() && rule.reply_ok(state.reply, state.status) &&
    (state.condition.empty() ||
     (state.etag.empty() ? !rule.require_validator_echo : state.etag == state.condition));

  // The count the response states: its count header when one is configured —
  // missing, repeated or malformed states none — and otherwise the requested
  // count for a response that is accepted.
  if (!plane.bytes_name.empty()) {
    state.delivered = response.counts == 1 ? decimal(response.bytes).value_or(0) : 0;
  } else {
    state.delivered = accepted ? state.size : 0;
  }
  if (!accepted || state.delivered != state.size) { return refused; }
  return static_cast<std::int64_t>(state.size);
}

/// The request callback handed to the binding.  Nothing may leave it through
/// the client's frames.
std::int64_t on_request(void* context, cucascade_s3rdma_request const* request) noexcept
{
  try {
    auto const identity = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(context));
    auto const found    = sessions().find(identity >> sequence_bits);
    if (found == nullptr) { return refused; }
    auto& state = *found;
    std::lock_guard lock{state.gate};
    if (state.sequence == 0 || (identity & sequence_mask) != state.sequence) {
      if (state.stale != std::numeric_limits<std::uint32_t>::max()) { ++state.stale; }
      return refused;
    }
    if (state.runs != std::numeric_limits<std::uint32_t>::max()) { ++state.runs; }
    if (state.runs != 1) { return refused; }
    try {
      if (request == nullptr) {
        state.failure = "the client ran the request callback without a request";
        return refused;
      }
      return send_request(state, *request);
    } catch (...) {
      try {
        state.failure = "the request could not be prepared";
      } catch (...) {
        // The count of runs and the missing byte count already fail the GET.
      }
      return refused;
    }
  } catch (...) {
    // The session could not be looked up or its gate not taken: nothing was
    // sent.
    return refused;
  }
}

//===----------------------------------------------------------------------===//
// Sessions
//===----------------------------------------------------------------------===//

class binding_session final : public rdma_data_session {
 public:
  explicit binding_session(std::shared_ptr<data_plane const> plane)
    : _plane(std::move(plane)), _easy(curl_easy_init())
  {
    if (_easy == nullptr) { throw unavailable("s3rdma: curl_easy_init failed"); }
    if (!sessions().enter(_state)) {
      throw unavailable("s3rdma: the process has used every data-session serial");
    }
    if (_plane->api->session_create(&_session) != 0 || _session == nullptr) {
      _session = nullptr;
      sessions().leave(_state->serial);
      throw unavailable("s3rdma: the binding library could not create a data session");
    }
  }

  ~binding_session() override
  {
    _plane->api->session_destroy(_session);
    sessions().leave(_state->serial);
  }

  binding_session(binding_session const&)            = delete;
  binding_session& operator=(binding_session const&) = delete;

  void register_memory(std::uint8_t* base, std::size_t bytes) override
  {
    if (_plane->api->register_memory(_session, base, bytes) != 0) {
      throw unavailable("s3rdma: the landing memory could not be registered");
    }
  }

  bool deregister_memory(std::uint8_t* base) noexcept override
  {
    return _plane->api->deregister_memory(_session, base) == 0;
  }

  std::int64_t max_request_callback_size(std::uint8_t* registered_base) noexcept override
  {
    return _plane->api->max_request_size(_session, registered_base);
  }

  data_get_result get(object_ref const& object,
                      std::size_t object_offset,
                      std::size_t size,
                      std::uint8_t* registered_base,
                      std::size_t dest_offset) override
  {
    return get_if_match(object, object_offset, size, registered_base, dest_offset, {});
  }

  data_get_result get_if_match(object_ref const& object,
                               std::size_t object_offset,
                               std::size_t size,
                               std::uint8_t* registered_base,
                               std::size_t dest_offset,
                               std::string_view if_match) override
  {
    if (registered_base == nullptr || size == 0 ||
        size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        size > std::numeric_limits<std::size_t>::max() - object_offset) {
      throw std::invalid_argument("s3rdma: invalid range or destination of a data GET");
    }
    if (!if_match.empty() && !is_strong_validator(if_match)) {
      throw std::invalid_argument("s3rdma: a GET can be conditioned on a strong validator only");
    }
    auto const identity =
      _state->begin(*_plane, _easy.get(), object, object_offset, size, if_match);
    if (identity == 0) {
      data_get_result unissued;
      unissued.transport_error = "the session has used every GET sequence";
      return unissued;
    }
    auto const returned =
      _plane->api->get(_session,
                       registered_base,
                       size,
                       dest_offset,
                       &on_request,
                       reinterpret_cast<void*>(static_cast<std::uintptr_t>(identity)));
    auto const invocations = _state->end();

    data_get_result result;
    result.sdk_return           = returned;
    result.callback_invocations = invocations;
    result.commit               = !_state->handed_over ? data_commit_state::not_sent
                                  : _state->answered   ? data_commit_state::completed
                                                       : data_commit_state::sent_unknown;
    result.delivered_bytes      = _state->delivered;
    result.http_status          = _state->status;
    result.reply_tag            = std::move(_state->reply);
    result.validation_tag       = std::move(_state->etag);
    result.transport_error      = std::move(_state->failure);
    return result;
  }

 private:
  std::shared_ptr<data_plane const> _plane;
  curl_easy_ptr _easy;
  cucascade_s3rdma_session* _session{nullptr};
  // Shared with the registry of sessions: a callback that is running when
  // the session goes away keeps it alive.
  std::shared_ptr<get_state> _state{std::make_shared<get_state>()};
};

class binding_session_factory final : public rdma_data_session_factory {
 public:
  explicit binding_session_factory(std::shared_ptr<data_plane const> plane)
    : _plane(std::move(plane))
  {
  }

  std::unique_ptr<rdma_data_session> acquire() override
  {
    return std::make_unique<binding_session>(_plane);
  }

 private:
  std::shared_ptr<data_plane const> _plane;
};

}  // namespace

endpoint_topology detect_endpoint_topology(std::string_view control_endpoint,
                                           std::string_view data_endpoint) noexcept
{
  try {
    auto const control = parts_of(control_endpoint);
    auto const data    = parts_of(data_endpoint);
    // A side without a host proves nothing.
    if (control.host.empty() || data.host.empty()) { return endpoint_topology::split; }
    return control == data ? endpoint_topology::same_address : endpoint_topology::split;
  } catch (...) {
    return endpoint_topology::split;
  }
}

std::shared_ptr<rdma_data_session_factory> make_data_session_factory(
  std::shared_ptr<rest::request_authorizer> authorizer, completion_rule rule, data_plane_config cfg)
{
  if (authorizer == nullptr) {
    throw std::invalid_argument("s3rdma: the data plane needs a request authorizer");
  }
  if (rule.accepted_statuses.empty() || rule.reply_ok == nullptr) {
    throw std::invalid_argument("s3rdma: the completion rule needs statuses and a reply check");
  }
  if (cfg.library_path.empty() || cfg.library_path.front() != '/') {
    throw std::invalid_argument("s3rdma: the binding library needs an absolute path");
  }
  if (cfg.request_deadline.count() <= 0) {
    throw std::invalid_argument("s3rdma: the data-plane request deadline must be positive");
  }
  // The request of a GET carries its token inside the signature.  An
  // authorizer that cannot sign a header is found here, before anything is
  // loaded.
  try {
    std::ignore = authorizer->authorize_with_headers(object_ref{"bucket", "key"},
                                                     rest::request_method::GET,
                                                     {{"range", "bytes=0-0"}},
                                                     std::chrono::seconds{60});
  } catch (std::exception const&) {
    throw unavailable("s3rdma: the data plane needs an authorizer that signs request headers");
  }

  auto plane        = std::make_shared<data_plane>();
  plane->authorizer = std::move(authorizer);
  plane->rule       = std::move(rule);
  plane->bytes_name = lowered(cfg.bytes_header);
  plane->api        = &load_binding(cfg.library_path, cfg.require_zero_io_retry);
  plane->cfg        = std::move(cfg);
  return std::make_shared<binding_session_factory>(std::move(plane));
}

transport_clients make_transport_clients(deployment where)
{
  if (where.control_endpoint.empty() || where.data_endpoint.empty()) {
    throw std::invalid_argument("s3rdma: the control and the data endpoint are both required");
  }
  transport_clients clients;
  clients.topology = detect_endpoint_topology(where.control_endpoint, where.data_endpoint);
  clients.control =
    make_control_client(std::move(where.control_authorizer), std::move(where.control));
  clients.rule          = where.rule;
  clients.data_sessions = make_data_session_factory(
    std::move(where.data_authorizer), std::move(where.rule), std::move(where.data));

  if (clients.topology == endpoint_topology::same_address) {
    CUCASCADE_LOG_INFO(
      "s3rdma: one address serves the control and the data plane; an object is "
      "readable once the store reports it");
    if (where.distinct_data_credentials) {
      CUCASCADE_LOG_WARN(
        "s3rdma: one address serves both planes, but the data plane has credentials of its own");
    }
  } else {
    CUCASCADE_LOG_INFO(
      "s3rdma: the control and the data plane are separate addresses; an object "
      "must be visible at both before it is read");
  }
  return clients;
}

}  // namespace cucascade::io::s3rdma
