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
#include <cucascade/io/s3rdma/control_client.hpp>

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::s3rdma {

namespace {

using rest::curl_easy_ptr;
using rest::curl_slist_ptr;

char ascii_lower(char c) noexcept
{
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

bool is_status_line(std::string_view line) noexcept
{
  constexpr std::string_view prefix{"http/"};
  if (line.size() < prefix.size()) { return false; }
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    if (ascii_lower(line[i]) != prefix[i]) { return false; }
  }
  return true;
}

/// The status code of a status line; zero when it has none.
long status_of(std::string_view line) noexcept
{
  auto const space = line.find(' ');
  if (space == std::string_view::npos) { return 0; }
  long code = 0;
  for (std::size_t i = space + 1; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
    if (code > 99999) { return 0; }
    code = code * 10 + (line[i] - '0');
  }
  return code;
}

/// The trimmed value of header line @p line when its name is @p name
/// (lowercase), compared without regard to case.
std::optional<std::string_view> header_value(std::string_view line, std::string_view name) noexcept
{
  if (line.size() <= name.size() || line[name.size()] != ':') { return std::nullopt; }
  for (std::size_t i = 0; i < name.size(); ++i) {
    if (ascii_lower(line[i]) != name[i]) { return std::nullopt; }
  }
  auto value = line.substr(name.size() + 1);
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' ||
                            value.back() == '\t')) {
    value.remove_suffix(1);
  }
  return value;
}

/// What one response says about itself.  A transfer can carry several
/// responses (an interim one, a proxy's); each status line starts over, so
/// only the last one is kept.
struct response_head {
  long status{0};
  std::string content_range;
  std::string etag;
  std::vector<std::string> content_lengths;

  void restart(long code)
  {
    status = code;
    content_range.clear();
    etag.clear();
    content_lengths.clear();
  }

  void take(std::string_view line)
  {
    if (is_status_line(line)) {
      restart(status_of(line));
    } else if (auto const value = header_value(line, "content-range")) {
      content_range.assign(*value);
    } else if (auto const tag = header_value(line, "etag")) {
      etag.assign(*tag);
    } else if (auto const length = header_value(line, "content-length")) {
      content_lengths.emplace_back(*length);
    }
  }
};

/// libcurl callbacks must not let an exception cross into C: a failure to
/// record a header ends the transfer instead.
std::size_t on_header(char* buffer, std::size_t size, std::size_t count, void* userdata) noexcept
{
  try {
    static_cast<response_head*>(userdata)->take(std::string_view{buffer, size * count});
    return size * count;
  } catch (...) {
    return 0;
  }
}

/// The object size a HEAD response states: exactly one decimal
/// @c Content-Length that fits, or the same one repeated.
std::optional<std::size_t> stated_length(std::vector<std::string> const& values) noexcept
{
  if (values.empty()) { return std::nullopt; }
  std::string_view const text{values.front()};
  if (text.empty() ||
      !std::all_of(values.begin(), values.end(), [&](std::string const& v) { return v == text; })) {
    return std::nullopt;
  }
  std::size_t length = 0;
  for (char const c : text) {
    if (c < '0' || c > '9') { return std::nullopt; }
    auto const digit = static_cast<std::size_t>(c - '0');
    if (length > (std::numeric_limits<std::size_t>::max() - digit) / 10) { return std::nullopt; }
    length = length * 10 + digit;
  }
  return length;
}

/// Destination of a range GET.  Only a 200 or 206 body is payload; any other
/// body is a diagnostic and is dropped.  A payload longer than the range is
/// not read to its end.
struct range_sink {
  response_head head;
  std::uint8_t* destination{nullptr};
  std::size_t capacity{0};
  std::size_t written{0};
  bool longer{false};
};

std::size_t on_range_header(char* buffer,
                            std::size_t size,
                            std::size_t count,
                            void* userdata) noexcept
{
  return on_header(buffer, size, count, &static_cast<range_sink*>(userdata)->head);
}

std::size_t on_range_body(char* data, std::size_t size, std::size_t count, void* userdata) noexcept
{
  auto* sink       = static_cast<range_sink*>(userdata);
  auto const bytes = size * count;
  if (sink->head.status != 200 && sink->head.status != 206) { return bytes; }
  auto const room = sink->capacity - sink->written;
  std::memcpy(sink->destination + sink->written, data, std::min(room, bytes));
  sink->written += std::min(room, bytes);
  if (bytes > room) {
    sink->longer = true;
    return 0;
  }
  return bytes;
}

/// Destination of a LIST response, bounded.
struct text_sink {
  std::string body;
  std::size_t limit{0};
  bool over{false};
};

std::size_t on_text_body(char* data, std::size_t size, std::size_t count, void* userdata) noexcept
{
  auto* sink       = static_cast<text_sink*>(userdata);
  auto const bytes = size * count;
  if (bytes > sink->limit - sink->body.size()) {
    sink->over = true;
    return 0;
  }
  try {
    sink->body.append(data, bytes);
    return bytes;
  } catch (...) {
    return 0;
  }
}

std::size_t drop_body(char* /*data*/, std::size_t size, std::size_t count, void* /*userdata*/)
{
  return size * count;
}

/// The header list of a request.  A request is never sent with part of it.
curl_slist_ptr header_list(std::vector<std::pair<std::string, std::string>> const& headers,
                           std::vector<std::string> const& extra)
{
  curl_slist_ptr list;
  auto const append = [&list](std::string const& line) {
    auto* next = curl_slist_append(list.get(), line.c_str());
    if (next == nullptr) { throw std::bad_alloc(); }
    std::ignore = list.release();
    list.reset(next);
  };
  for (auto const& [name, value] : headers) {
    append(name + ": " + value);
  }
  for (auto const& line : extra) {
    append(line);
  }
  return list;
}

class curl_control_client final : public s3_control_client {
 public:
  curl_control_client(std::shared_ptr<rest::request_authorizer> authorizer,
                      control_client_config cfg)
    : _authorizer(std::move(authorizer)), _cfg(std::move(cfg))
  {
    if (_authorizer == nullptr) {
      throw std::invalid_argument("s3rdma: the control client needs a request authorizer");
    }
    if (_cfg.request_timeout.count() <= 0) {
      throw std::invalid_argument("s3rdma: the control client's request timeout must be positive");
    }
  }

  head_result head(object_ref const& object) override
  {
    auto const authorized = _authorizer->authorize(object, rest::request_method::HEAD, url_ttl());
    auto const headers    = header_list(authorized.headers, {});
    lease handle{*this};
    prepare(handle.get(), authorized.url, headers.get(), /*bounded=*/true);
    response_head response;
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_NOBODY, 1L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &drop_body));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HEADERFUNCTION, &on_header));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HEADERDATA, &response));

    auto const rc = curl_easy_perform(handle.get());
    head_result result;
    result.outcome.http_status = response.status;
    if (rc != CURLE_OK) {
      result.outcome.transport_error = curl_easy_strerror(rc);
      return result;
    }
    if (response.status != 200) { return result; }
    auto const length = stated_length(response.content_lengths);
    if (!length.has_value()) {
      result.outcome.transport_error = "the response carries no usable Content-Length";
      return result;
    }
    result.object_size    = *length;
    result.validation_tag = std::move(response.etag);
    return result;
  }

  range_get_result range_get(object_ref const& object,
                             std::size_t offset,
                             std::size_t size,
                             std::uint8_t* host_dst) override
  {
    return range_get_if_match(object, offset, size, host_dst, {});
  }

  range_get_result range_get_if_match(object_ref const& object,
                                      std::size_t offset,
                                      std::size_t size,
                                      std::uint8_t* host_dst,
                                      std::string_view if_match) override
  {
    // The largest size is refused as well: size + 1 reports a longer body.
    if (size == 0 || host_dst == nullptr || size == std::numeric_limits<std::size_t>::max() ||
        size > std::numeric_limits<std::size_t>::max() - offset) {
      throw std::invalid_argument("s3rdma: invalid range or destination of a control read");
    }
    if (!if_match.empty() && !is_strong_validator(if_match)) {
      throw std::invalid_argument("s3rdma: a read can be conditioned on a strong validator only");
    }
    auto const authorized = _authorizer->authorize(object, rest::request_method::GET, url_ttl());
    std::vector<std::string> extra;
    extra.push_back("Range: bytes=" + std::to_string(offset) + "-" +
                    std::to_string(offset + size - 1));
    if (!if_match.empty()) { extra.push_back("If-Match: " + std::string{if_match}); }
    auto const headers = header_list(authorized.headers, extra);
    lease handle{*this};
    prepare(handle.get(), authorized.url, headers.get(), /*bounded=*/false);
    range_sink sink;
    sink.destination = host_dst;
    sink.capacity    = size;
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HTTPGET, 1L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &on_range_body));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &sink));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HEADERFUNCTION, &on_range_header));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HEADERDATA, &sink));

    auto const rc = curl_easy_perform(handle.get());
    range_get_result result;
    result.outcome.http_status = sink.head.status;
    result.delivered_bytes     = sink.longer ? size + 1 : sink.written;
    result.content_range       = std::move(sink.head.content_range);
    result.validation_tag      = std::move(sink.head.etag);
    // A body that was longer was cut off here, not by the transport.
    if (rc != CURLE_OK && !sink.longer) { result.outcome.transport_error = curl_easy_strerror(rc); }
    return result;
  }

  list_page_result list_page(std::string_view bucket, std::string_view canonical_query) override
  {
    auto const authorized = _authorizer->authorize_list(bucket, canonical_query, url_ttl());
    auto const headers    = header_list(authorized.headers, {});
    lease handle{*this};
    prepare(handle.get(), authorized.url, headers.get(), /*bounded=*/true);
    response_head response;
    text_sink sink;
    sink.limit = _cfg.max_list_body_bytes;
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HTTPGET, 1L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &on_text_body));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &sink));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HEADERFUNCTION, &on_header));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HEADERDATA, &response));

    auto const rc = curl_easy_perform(handle.get());
    list_page_result result;
    result.outcome.http_status = response.status;
    if (sink.over) {
      result.outcome.transport_error = "the response body exceeds the configured limit";
    } else if (rc != CURLE_OK) {
      result.outcome.transport_error = curl_easy_strerror(rc);
    } else {
      result.body = std::move(sink.body);
    }
    return result;
  }

 private:
  /// One easy handle taken from the pool for one request.  Handles are kept
  /// so that their connections are reused.
  class lease {
   public:
    explicit lease(curl_control_client& owner) : _owner(owner), _handle(owner.take()) {}
    ~lease() { _owner.give_back(std::move(_handle)); }
    lease(lease const&)            = delete;
    lease& operator=(lease const&) = delete;

    [[nodiscard]] CURL* get() const noexcept { return _handle.get(); }

   private:
    curl_control_client& _owner;
    curl_easy_ptr _handle;
  };

  [[nodiscard]] curl_easy_ptr take()
  {
    {
      std::lock_guard lock{_mutex};
      if (!_idle.empty()) {
        auto handle = std::move(_idle.back());
        _idle.pop_back();
        return handle;
      }
    }
    curl_easy_ptr handle{curl_easy_init()};
    if (handle == nullptr) { throw std::runtime_error("s3rdma: curl_easy_init failed"); }
    return handle;
  }

  void give_back(curl_easy_ptr handle) noexcept
  {
    // Everything a request set, pointers into its frame included, is dropped;
    // the handle's connections stay.
    curl_easy_reset(handle.get());
    try {
      std::lock_guard lock{_mutex};
      if (_idle.size() < max_idle) { _idle.push_back(std::move(handle)); }
    } catch (...) {
      // The handle is closed instead of kept.
    }
  }

  /// Options every request shares.  @p bounded limits the whole request;
  /// otherwise only a stalled transfer ends.
  void prepare(CURL* handle, std::string const& url, curl_slist* headers, bool bounded) const
  {
    rest::configure_easy_handle(handle, rest::global_curl_context::instance().share_handle());
    if (!_cfg.ca_bundle_path.empty()) {
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_CAINFO, _cfg.ca_bundle_path.c_str()));
    }
    if (!_cfg.tls_verify) {
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 0L));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 0L));
    }
    if (bounded) {
      CUCASCADE_CURL_CHECK(
        curl_easy_setopt(handle, CURLOPT_TIMEOUT, static_cast<long>(_cfg.request_timeout.count())));
    } else {
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TIMEOUT, 0L));
      CUCASCADE_CURL_CHECK(
        curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, _cfg.stall_bytes_per_second));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(
        handle, CURLOPT_LOW_SPEED_TIME, static_cast<long>(_cfg.stall_time.count())));
    }
    // The path is sent as it was authorized: "." and ".." are legal in an
    // object key, and the transport would otherwise remove such segments.
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_PATH_AS_IS, 1L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_URL, url.c_str()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers));
  }

  /// Lifetime of a query-signed URL: one request plus clock skew.
  [[nodiscard]] std::chrono::seconds url_ttl() const noexcept
  {
    return _cfg.request_timeout + std::chrono::seconds{60};
  }

  static constexpr std::size_t max_idle = 8;

  std::shared_ptr<rest::request_authorizer> _authorizer;
  control_client_config _cfg;
  std::mutex _mutex;
  std::vector<curl_easy_ptr> _idle;
};

}  // namespace

std::shared_ptr<s3_control_client> make_control_client(
  std::shared_ptr<rest::request_authorizer> authorizer, control_client_config cfg)
{
  return std::make_shared<curl_control_client>(std::move(authorizer), std::move(cfg));
}

}  // namespace cucascade::io::s3rdma
