/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"
#include "sigv4_reference.hpp"
#include "transport_test_support.hpp"

#include <curl/curl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <tuple>

using namespace cucascade::test::s3rdma;
namespace io = cucascade::io;

namespace {

struct data_fixture {
  explicit data_fixture(rdma::data_plane_config config = data_config(),
                        rdma::completion_rule rule     = accepted_rule())
    : auth(std::make_shared<wire_authorizer>(server.endpoint())),
      factory(rdma::make_data_session_factory(auth, rule, config)),
      module(config.library_path)
  {
    module.reset();
    session = factory->acquire();
    destination.fill(0xcc);
    session->register_memory(destination.data(), destination.size());
    CUCASCADE_LOG_TRACE("data fixture log capture control");
  }
  ~data_fixture()
  {
    if (session) { (void)session->deregister_memory(destination.data()); }
  }
  rdma::data_get_result get(std::string_view condition = "\"version-a\"")
  {
    return session->get_if_match({"test", "object"}, 17, 4, destination.data(), 3, condition);
  }
  deadline bound;
  log_capture logs;
  http_fixture server;
  std::array<std::uint8_t, 32> destination{};
  std::shared_ptr<wire_authorizer> auth;
  std::shared_ptr<rdma::rdma_data_session_factory> factory;
  module_view module;
  std::unique_ptr<rdma::rdma_data_session> session;
};

struct environment_value {
  environment_value(char const* key, std::string const& value) : key(key)
  {
    if (auto const* old = std::getenv(key)) { saved = old; }
    if (::setenv(key, value.c_str(), 1) != 0) { throw std::runtime_error("setenv failed"); }
  }
  ~environment_value()
  {
    if (saved) {
      (void)::setenv(key, saved->c_str(), 1);
    } else {
      (void)::unsetenv(key);
    }
  }
  char const* key;
  std::optional<std::string> saved;
};

std::size_t discard_body(char*, std::size_t size, std::size_t count, void*) { return size * count; }

struct temporary_library {
  temporary_library()
  {
    int const fd = ::mkstemp(path.data());
    if (fd < 0) { throw std::runtime_error("mkstemp failed"); }
    (void)::write(fd, "not an ELF library", 18);
    (void)::close(fd);
  }
  ~temporary_library() { (void)::unlink(path.data()); }
  std::array<char, 48> path{"/tmp/cucascade-binding-XXXXXX"};
};

}  // namespace

TEST_CASE("S3RDMA topology compares endpoint text without resolving names", "[s3rdma][topology]")
{
  for (auto const& [control, data, same] : std::vector<std::tuple<std::string, std::string, bool>>{
         {"HTTP://Example.COM", "http://example.com:80/", true},
         {"https://example.com", "https://EXAMPLE.com:443", true},
         {"http://example.com/path/", "http://example.com/path", true},
         {"http://example.com/Path", "http://example.com/path", false},
         {"http://example.com", "https://example.com", false},
         {"http://example.com:81", "http://example.com", false},
         {"http://[::1]", "HTTP://[::1]:80/", true},
         {"https://[2001:db8::1]:443/a/", "https://[2001:db8::1]/a", true},
         {"http://127.0.0.1", "http://localhost", false},
         {"http:///path", "http:///path", false},
         {"", "", false},
         {"http://example.com//", "http://example.com/", false}}) {
    CAPTURE(control, data);
    CHECK(rdma::detect_endpoint_topology(control, data) ==
          (same ? rdma::endpoint_topology::same_address : rdma::endpoint_topology::split));
  }
  CHECK_NOTHROW(rdma::detect_endpoint_topology("garbage", "garbage"));
}

TEST_CASE("S3RDMA deployment logs the topology once and exposes it through the ioctx",
          "[s3rdma][topology]")
{
  for (bool same : {true, false}) {
    deadline bound;
    log_capture logs;
    rdma::deployment deployment;
    deployment.control_endpoint   = "http://control.example";
    deployment.data_endpoint      = same ? deployment.control_endpoint : "http://data.example";
    deployment.control_authorizer = signer(deployment.control_endpoint);
    deployment.data_authorizer    = signer(deployment.data_endpoint, "data-fixture", "data-secret");
    deployment.distinct_data_credentials = true;
    deployment.data                      = data_config();
    deployment.rule                      = accepted_rule();
    auto clients                         = rdma::make_transport_clients(deployment);
    auto controls                        = std::make_shared<runtime_controls>();
    rdma::s3rdma_ioctx context(runtime_config(), clients, controls->delivery(), controls->policy());
    CHECK(context.topology() == clients.topology);
    CHECK(context.topology() ==
          (same ? rdma::endpoint_topology::same_address : rdma::endpoint_topology::split));
    auto const records = logs.snapshot();
    REQUIRE(records.size() == (same ? 2 : 1));
    CHECK(records[0].first == cucascade::log::level::info);
    CHECK(records[0].second ==
          (same ? "s3rdma: one address serves the control and the data plane; an object is "
                  "readable once the store reports it"
                : "s3rdma: the control and the data plane are separate addresses; an object "
                  "must be visible at both before it is read"));
    if (same) {
      CHECK(records[1].first == cucascade::log::level::warn);
      CHECK(
        records[1].second ==
        "s3rdma: one address serves both planes, but the data plane has credentials of its own");
    }
    deployment.control_endpoint.clear();
    CHECK_THROWS_AS(rdma::make_transport_clients(deployment), std::invalid_argument);
    deployment.control_endpoint = "http://control.example";
    deployment.data_endpoint.clear();
    CHECK_THROWS_AS(rdma::make_transport_clients(deployment), std::invalid_argument);
  }
}

TEST_CASE("S3RDMA loader checks configuration and signing before loading a library",
          "[s3rdma][binding]")
{
  deadline bound;
  auto auth   = std::make_shared<wire_authorizer>("http://127.0.0.1:9");
  auto config = data_config("auth_first");
  CHECK_FALSE(is_loaded(config.library_path));
  CHECK_THROWS_AS(rdma::make_data_session_factory(nullptr, accepted_rule(), config),
                  std::invalid_argument);
  auto rule = accepted_rule();
  rule.accepted_statuses.clear();
  CHECK_THROWS_AS(rdma::make_data_session_factory(auth, rule, config), std::invalid_argument);
  rule          = accepted_rule();
  rule.reply_ok = nullptr;
  CHECK_THROWS_AS(rdma::make_data_session_factory(auth, rule, config), std::invalid_argument);
  for (std::string const path : {"", "relative.so"}) {
    auto bad         = config;
    bad.library_path = path;
    CHECK_THROWS_AS(rdma::make_data_session_factory(auth, accepted_rule(), bad),
                    std::invalid_argument);
  }
  for (auto timeout : {0ms, -1ms}) {
    auto bad             = config;
    bad.request_deadline = timeout;
    CHECK_THROWS_AS(rdma::make_data_session_factory(auth, accepted_rule(), bad),
                    std::invalid_argument);
  }
  auth->signs_headers = false;
  require_code([&] { (void)rdma::make_data_session_factory(auth, accepted_rule(), config); },
               rdma::errc::backend_unavailable);
  CHECK_FALSE(is_loaded(config.library_path));
}

TEST_CASE("S3RDMA loader refuses every incompatible table without creating a session",
          "[s3rdma][binding]")
{
  deadline bound;
  auto auth = std::make_shared<wire_authorizer>("http://127.0.0.1:9");
  for (auto variant : {"missing_symbol",
                       "null_table",
                       "wrong_version",
                       "wrong_size",
                       "missing_probe",
                       "missing_create",
                       "missing_destroy",
                       "missing_register",
                       "missing_deregister",
                       "missing_capacity",
                       "missing_get",
                       "probe_fails",
                       "wrong_protocol"}) {
    CAPTURE(variant);
    auto config = data_config(variant);
    require_code([&] { (void)rdma::make_data_session_factory(auth, accepted_rule(), config); },
                 rdma::errc::backend_unavailable);
    REQUIRE(is_loaded(config.library_path));
    module_view module(config.library_path);
    CHECK(module.snapshot().creates == 0);
    CHECK_FALSE(module.snapshot().overflow);
  }
  auto missing = data_config("does_not_exist");
  require_code([&] { (void)rdma::make_data_session_factory(auth, accepted_rule(), missing); },
               rdma::errc::backend_unavailable);
  CHECK_FALSE(is_loaded(missing.library_path));
  temporary_library invalid;
  missing.library_path = invalid.path.data();
  require_code([&] { (void)rdma::make_data_session_factory(auth, accepted_rule(), missing); },
               rdma::errc::backend_unavailable);
  CHECK_FALSE(is_loaded(missing.library_path));
}

TEST_CASE("S3RDMA accepted tables are cached and probes repeat without a new entry lookup",
          "[s3rdma][binding]")
{
  deadline bound;
  auto const path = binding_path("unretained");
  CHECK_FALSE(is_loaded(path));
  {
    module_view control(path, true);
  }
  CHECK_FALSE(is_loaded(path));
  auto auth   = std::make_shared<wire_authorizer>("http://127.0.0.1:9");
  auto config = data_config("cache_probe");
  {
    auto factory = rdma::make_data_session_factory(auth, accepted_rule(), config);
  }
  REQUIRE(is_loaded(config.library_path));
  module_view module(config.library_path);
  auto first = module.snapshot();
  CHECK(first.entries == 1);
  CHECK(first.probes == 1);
  CHECK(first.creates == 0);
  {
    auto factory = rdma::make_data_session_factory(auth, accepted_rule(), config);
  }
  CHECK(module.snapshot().entries == first.entries);
  CHECK(module.snapshot().probes == first.probes + 1);
  CHECK(module.snapshot().creates == 0);
  CHECK(::dlsym(RTLD_DEFAULT, "test_binding_snapshot") == nullptr);
}

TEST_CASE("S3RDMA retry counts warn or refuse before a session is created", "[s3rdma][binding]")
{
  deadline bound;
  auto auth = std::make_shared<wire_authorizer>("http://127.0.0.1:9");
  for (auto variant : {"retry_unknown", "retry_nonzero"}) {
    log_capture logs;
    auto config = data_config(variant);
    {
      auto factory = rdma::make_data_session_factory(auth, accepted_rule(), config);
    }
    auto records = logs.snapshot();
    REQUIRE(records.size() == 1);
    CHECK(records[0].first == cucascade::log::level::warn);
    CHECK(records[0].second.find("rdma_io_retry_count") != std::string::npos);
    module_view module(config.library_path);
    CHECK(module.snapshot().creates == 0);
    config.require_zero_io_retry = true;
    require_code([&] { (void)rdma::make_data_session_factory(auth, accepted_rule(), config); },
                 rdma::errc::backend_unavailable);
    CHECK(module.snapshot().creates == 0);
    CHECK(logs.snapshot().size() == 1);
  }
}

TEST_CASE("S3RDMA session operations preserve results and registration lifetime",
          "[s3rdma][binding]")
{
  deadline bound;
  auto auth    = std::make_shared<wire_authorizer>("http://127.0.0.1:9");
  auto factory = rdma::make_data_session_factory(auth, accepted_rule(), data_config());
  module_view module(binding_path("good"));
  module.reset();
  binding_options options;
  options.create_result = 9;
  module.configure(options);
  require_code([&] { (void)factory->acquire(); }, rdma::errc::backend_unavailable);
  options.create_result = 0;
  options.create_null   = true;
  module.configure(options);
  require_code([&] { (void)factory->acquire(); }, rdma::errc::backend_unavailable);
  options.create_null = false;
  module.configure(options);
  auto session = factory->acquire();
  std::array<std::uint8_t, 32> base{};
  options.register_result = 7;
  module.configure(options);
  CHECK_THROWS(session->register_memory(base.data(), base.size()));
  options.register_result = 0;
  module.configure(options);
  CHECK_NOTHROW(session->register_memory(base.data(), base.size()));
  for (std::int64_t capacity : {std::int64_t{-1}, std::int64_t{8192}}) {
    options.capacity = capacity;
    module.configure(options);
    CHECK(session->max_request_callback_size(base.data()) == capacity);
  }
  options.deregister_result = 2;
  module.configure(options);
  CHECK_FALSE(session->deregister_memory(base.data()));
  options.deregister_result = 0;
  module.configure(options);
  CHECK(session->deregister_memory(base.data()));
  session.reset();
  auto const snapshot = module.snapshot();
  CHECK(snapshot.creates == 3);
  CHECK(snapshot.destroys == 1);
  CHECK(snapshot.registrations == 2);
  CHECK(snapshot.deregistrations == 2);
  REQUIRE(snapshot.event_count >= 2);
  CHECK(snapshot.events[snapshot.event_count - 2].operation == binding_operation::deregister);
  CHECK(snapshot.events[snapshot.event_count - 1].operation == binding_operation::destroy);
  CHECK(snapshot.events[snapshot.event_count - 2].session ==
        snapshot.events[snapshot.event_count - 1].session);

  module.reset();
  auto controls         = std::make_shared<runtime_controls>();
  auto clients          = controls->clients();
  clients.data_sessions = factory;
  {
    rdma::s3rdma_ioctx context(runtime_config(), clients, controls->delivery(), controls->policy());
    context.start();
    context.shutdown();
  }
  auto const released = module.snapshot();
  auto first          = released.events.begin();
  auto end            = first + static_cast<std::ptrdiff_t>(released.event_count);
  auto registration   = std::find_if(first, end, [](auto const& event) {
    return event.operation == binding_operation::register_memory;
  });
  REQUIRE(registration != end);
  auto deregistration = std::find_if(registration, end, [&](auto const& event) {
    return event.operation == binding_operation::deregister &&
           event.session == registration->session;
  });
  REQUIRE(deregistration != end);
  auto destruction = std::find_if(deregistration, end, [&](auto const& event) {
    return event.operation == binding_operation::destroy && event.session == registration->session;
  });
  CHECK(destruction != end);
  CHECK_FALSE(released.overflow);
}

TEST_CASE("S3RDMA GET signs the data request and closes each HTTP1 connection",
          "[s3rdma][data-plane]")
{
  deadline bound;
  log_capture logs;
  http_fixture control;
  http_fixture data;
  control.respond({.headers = {{"Content-Length", "32"}}});
  data.respond(accepted_response());
  data.respond(accepted_response());
  rdma::deployment deployment;
  deployment.control_endpoint   = control.endpoint();
  deployment.data_endpoint      = data.endpoint();
  deployment.control_authorizer = signer(control.endpoint(), "control-fixture", "control-secret");
  deployment.data_authorizer    = signer(data.endpoint(), "data-fixture", "data-secret");
  deployment.data               = data_config();
  deployment.rule               = accepted_rule();
  auto clients                  = rdma::make_transport_clients(deployment);
  CHECK(clients.control->head({"test", "object"}).object_size == 32);
  module_view module(binding_path("good"));
  module.reset();
  auto session = clients.data_sessions->acquire();
  std::array<std::uint8_t, 32> bytes{};
  session->register_memory(bytes.data(), bytes.size());
  for (unsigned i = 0; i < 2; ++i) {
    auto const result =
      session->get_if_match({"test", "object"}, 17, 4, bytes.data(), 3, "\"version-a\"");
    CHECK(result.commit == rdma::data_commit_state::completed);
    CHECK(result.sdk_return == 4);
    CHECK(result.callback_invocations == 1);
    CHECK(result.delivered_bytes == 4);
    CHECK(result.http_status == 206);
    CHECK(result.reply_tag == "accepted");
    CHECK(result.validation_tag == "\"version-a\"");
    CHECK(result.transport_error.empty());
    no_token(logs, result.transport_error);
  }
  REQUIRE(data.wait_for_peer_closes(2));
  auto const requests = data.requests();
  REQUIRE(requests.size() == 2);
  CHECK(requests[0].connection != requests[1].connection);
  for (auto const& request : requests) {
    CHECK(request.method == "GET");
    CHECK(request.target == "/test/object");
    CHECK(request.version == "HTTP/1.1");
    CHECK(request.value("upgrade").empty());
    CHECK(request.value("x-amz-rdma-token") == "CAPABILITY-NOT-FOR-LOGS");
    CHECK(request.value("range") == "bytes=17-20");
    CHECK(request.value("if-match") == "\"version-a\"");
    auto const names = "host;if-match;range;x-amz-content-sha256;x-amz-date;x-amz-rdma-token";
    auto const auth  = request.value("authorization");
    CHECK(auth.find("Credential=data-fixture/") != std::string::npos);
    CHECK(auth.find(std::string{"SignedHeaders="} + names + ",") != std::string::npos);
    CHECK(auth.substr(auth.find("Signature=") + 10) ==
          reference_sigv4(request.headers, "GET", "/test/object", names, "data-secret"));
  }
  REQUIRE(control.requests().size() == 1);
  CHECK(control.requests()[0].method == "HEAD");
  CHECK(session->deregister_memory(bytes.data()));
}

TEST_CASE("S3RDMA data requests refuse redirects and bypass an effective environment proxy",
          "[s3rdma][data-plane]")
{
  data_fixture f;
  http_fixture redirected;
  http_fixture proxy;
  f.server.respond({.status = 307, .headers = {{"Location", redirected.endpoint() + "/moved"}}});
  auto const redirect = f.get();
  CHECK(redirect.http_status == 307);
  CHECK(redirect.commit == rdma::data_commit_state::completed);
  CHECK(redirect.sdk_return == -ENOTSUP);
  CHECK(redirected.requests().empty());
  no_token(f.logs, redirect.transport_error);

  environment_value lower_proxy("http_proxy", proxy.endpoint());
  environment_value upper_proxy("HTTP_PROXY", proxy.endpoint());
  environment_value all_proxy("all_proxy", proxy.endpoint());
  environment_value upper_all("ALL_PROXY", proxy.endpoint());
  environment_value lower_no("no_proxy", "");
  environment_value upper_no("NO_PROXY", "");
  proxy.respond({});
  auto* raw = curl_easy_init();
  REQUIRE(raw != nullptr);
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> easy(raw, &curl_easy_cleanup);
  auto const url = f.server.endpoint() + "/proxy-control";
  REQUIRE(curl_easy_setopt(raw, CURLOPT_URL, url.c_str()) == CURLE_OK);
  REQUIRE(curl_easy_setopt(raw, CURLOPT_TIMEOUT_MS, 1000L) == CURLE_OK);
  REQUIRE(curl_easy_setopt(raw, CURLOPT_WRITEFUNCTION, &discard_body) == CURLE_OK);
  CHECK(curl_easy_perform(raw) == CURLE_OK);
  REQUIRE(proxy.requests().size() == 1);
  CHECK(proxy.requests()[0].target == url);
  f.server.respond(accepted_response());
  auto const direct = f.get();
  CHECK(direct.sdk_return == 4);
  CHECK(f.server.requests().size() == 2);
  CHECK(proxy.requests().size() == 1);
  no_token(f.logs, direct.transport_error);
}

TEST_CASE("S3RDMA incomplete and oversized data responses are sent with unknown completion",
          "[s3rdma][data-plane]")
{
  bool const oversized    = GENERATE(false, true);
  auto config             = data_config();
  config.request_deadline = 200ms;
  data_fixture f(config);
  auto response = accepted_response();
  if (oversized) {
    response.body.assign(65537, 'x');
  } else {
    response.body              = "incomplete";
    response.chunk_bytes       = 1;
    response.pause_after_bytes = 1;
    response.body_pause        = 20s;
  }
  f.server.respond(response);
  auto const start  = std::chrono::steady_clock::now();
  auto const result = f.get();
  CHECK(result.sdk_return == -ENOTSUP);
  CHECK(result.commit == rdma::data_commit_state::sent_unknown);
  CHECK(result.callback_invocations == 1);
  CHECK(result.http_status == 206);
  CHECK(result.delivered_bytes == 0);
  CHECK_FALSE(result.transport_error.empty());
  if (!oversized) { CHECK(std::chrono::steady_clock::now() - start >= 100ms); }
  CHECK(std::chrono::steady_clock::now() - start < 3s);
  CHECK(f.server.requests().size() == 1);
  no_token(f.logs, result.transport_error);
}

TEST_CASE("S3RDMA data completion refuses every status reply and validator mismatch",
          "[s3rdma][data-plane]")
{
  for (auto fault : {"status",
                     "absent-reply",
                     "empty-reply",
                     "two-replies",
                     "bad-reply",
                     "absent-validator",
                     "other-validator",
                     "none"}) {
    CAPTURE(fault);
    data_fixture f;
    auto response = accepted_response();
    std::string const kind{fault};
    if (kind == "status") { response.status = 500; }
    if (kind == "absent-reply") { response.headers.erase(response.headers.begin()); }
    if (kind == "empty-reply") { response.headers[0].second.clear(); }
    if (kind == "two-replies") { response.headers.emplace_back("x-amz-rdma-reply", "accepted"); }
    if (kind == "bad-reply") { response.headers[0].second = "refused"; }
    if (kind == "absent-validator") { response.headers.pop_back(); }
    if (kind == "other-validator") { response.headers[1].second = "\"version-b\""; }
    f.server.respond(response);
    auto const result   = f.get();
    bool const accepted = kind == "none";
    CHECK(result.sdk_return == (accepted ? 4 : -ENOTSUP));
    CHECK(result.delivered_bytes == (accepted ? 4 : 0));
    CHECK(result.http_status == response.status);
    CHECK(result.commit == rdma::data_commit_state::completed);
    CHECK(result.callback_invocations == 1);
    CHECK(result.transport_error.empty());
    if (kind == "two-replies" || kind == "absent-reply" || kind == "empty-reply") {
      CHECK(result.reply_tag.empty());
    }
    CHECK(f.server.requests().size() == 1);
    auto const observed = f.module.snapshot();
    REQUIRE(observed.result_count == 1);
    CHECK(observed.results[0] == (accepted ? 4 : -ENOTSUP));
    REQUIRE_FALSE(f.logs.snapshot().empty());
    no_token(f.logs, result.transport_error);
  }
}

TEST_CASE("S3RDMA the data byte count must be singular decimal and exact", "[s3rdma][data-plane]")
{
  for (auto const& [values, count] :
       std::vector<std::pair<std::vector<std::string>, std::size_t>>{{{}, 0},
                                                                     {{"4", "4"}, 0},
                                                                     {{"4x"}, 0},
                                                                     {{"-1"}, 0},
                                                                     {{"184467440737095516160"}, 0},
                                                                     {{"3"}, 3},
                                                                     {{"5"}, 5},
                                                                     {{"4"}, 4},
                                                                     {{"04"}, 4}}) {
    CAPTURE(values);
    auto config         = data_config();
    config.bytes_header = "X-Transfer-Bytes";
    data_fixture f(config);
    auto response = accepted_response();
    for (auto const& value : values) {
      response.headers.emplace_back("x-transfer-bytes", value);
    }
    f.server.respond(response);
    auto const result = f.get();
    CHECK(result.delivered_bytes == count);
    CHECK(result.sdk_return == (count == 4 ? 4 : -ENOTSUP));
    CHECK(result.callback_invocations == 1);
    CHECK(result.commit == rdma::data_commit_state::completed);
    CHECK(result.http_status == 206);
    CHECK(result.reply_tag == "accepted");
    CHECK(result.validation_tag == "\"version-a\"");
    CHECK(result.transport_error.empty());
    no_token(f.logs, result.transport_error);
  }
}

TEST_CASE("S3RDMA missing validator echo is accepted only by explicit policy",
          "[s3rdma][data-plane]")
{
  for (bool require_echo : {true, false}) {
    auto rule                   = accepted_rule();
    rule.require_validator_echo = require_echo;
    data_fixture f(data_config(), rule);
    f.server.respond(accepted_response(""));
    auto const result = f.get();
    CHECK(result.sdk_return == (require_echo ? -ENOTSUP : 4));
    CHECK(result.validation_tag.empty());
    CHECK(result.commit == rdma::data_commit_state::completed);
    no_token(f.logs, result.transport_error);
  }
}

TEST_CASE("S3RDMA callback shape failures send nothing and redact the token",
          "[s3rdma][data-plane]")
{
  for (auto mode : {callback_mode::wrong_offset,
                    callback_mode::wrong_size,
                    callback_mode::null_descriptor,
                    callback_mode::empty_descriptor,
                    callback_mode::non_ascii_descriptor,
                    callback_mode::null_request}) {
    CAPTURE(static_cast<int>(mode));
    data_fixture f;
    binding_options options;
    options.mode = mode;
    f.module.configure(options);
    auto const result = f.get();
    CHECK(result.sdk_return == -ENOTSUP);
    CHECK(result.callback_invocations == 1);
    CHECK(result.commit == rdma::data_commit_state::not_sent);
    CHECK(result.http_status == 0);
    CHECK(result.delivered_bytes == 0);
    CHECK(result.reply_tag.empty());
    CHECK_FALSE(result.transport_error.empty());
    CHECK(f.server.requests().empty());
    no_token(f.logs, result.transport_error);
  }
}

TEST_CASE("S3RDMA callbacks from another thread and tokens with terminal NULs are served",
          "[s3rdma][data-plane]")
{
  for (auto mode : {callback_mode::threaded, callback_mode::trailing_nuls}) {
    data_fixture f;
    binding_options options;
    options.mode = mode;
    f.module.configure(options);
    f.server.respond(accepted_response());
    auto const result = f.get();
    CHECK(result.sdk_return == 4);
    CHECK(result.callback_invocations == 1);
    CHECK(result.commit == rdma::data_commit_state::completed);
    REQUIRE(f.server.requests().size() == 1);
    CHECK(f.server.requests()[0].value("x-amz-rdma-token") == "CAPABILITY-NOT-FOR-LOGS");
    no_token(f.logs, result.transport_error);
  }
}

TEST_CASE("S3RDMA repeated callbacks cannot issue a second request", "[s3rdma][data-plane]")
{
  data_fixture f;
  binding_options options;
  options.mode = callback_mode::twice;
  f.module.configure(options);
  f.server.respond(accepted_response());
  auto const result = f.get();
  CHECK(result.sdk_return == -ENOTSUP);
  CHECK(result.callback_invocations == 2);
  CHECK(result.commit == rdma::data_commit_state::completed);
  CHECK(result.delivered_bytes == 4);
  CHECK(f.server.requests().size() == 1);
  auto const observed = f.module.snapshot();
  REQUIRE(observed.result_count == 2);
  CHECK(observed.results[0] == 4);
  CHECK(observed.results[1] == -ENOTSUP);
  no_token(f.logs, result.transport_error);
}

TEST_CASE("S3RDMA late callbacks in the next seven GETs cannot use an ended request",
          "[s3rdma][data-plane]")
{
  for (bool first_ran : {false, true}) {
    for (unsigned age = 1; age <= 7; ++age) {
      for (bool between_gets : {false, true}) {
        CAPTURE(first_ran, age, between_gets);
        data_fixture f;
        binding_options options;
        options.mode = first_ran ? callback_mode::normal : callback_mode::absent;
        f.module.configure(options);
        if (first_ran) { f.server.respond(accepted_response()); }
        {
          rdma::object_ref old{"test", std::string(128, 'o')};
          std::string condition = "\"version-a\"";
          auto const result =
            f.session->get_if_match(old, 0, 4, f.destination.data(), 0, condition);
          CHECK(result.callback_invocations == (first_ran ? 1 : 0));
        }
        options.mode = callback_mode::normal;
        f.module.configure(options);
        for (unsigned n = 1; n < age; ++n) {
          f.server.respond(accepted_response());
          CHECK(f.get().sdk_return == 4);
        }
        auto const before = f.server.requests().size();
        if (between_gets) {
          CHECK(f.module.fire(0) == -ENOTSUP);
          CHECK(f.server.requests().size() == before);
        } else {
          options.late_before = 0;
          f.module.configure(options);
        }
        f.server.respond(accepted_response());
        auto const next = f.get();
        CHECK(next.callback_invocations == 2);
        CHECK(next.sdk_return == 4);
        CHECK(next.commit == rdma::data_commit_state::completed);
        CHECK(f.server.requests().size() == before + 1);
        auto const observations = f.module.snapshot();
        REQUIRE(observations.result_count >= 2);
        CHECK(observations.results[observations.result_count - 2] == -ENOTSUP);
        CHECK(observations.results[observations.result_count - 1] == 4);
        no_token(f.logs, next.transport_error);
      }
    }
  }
}

TEST_CASE("S3RDMA the ninth GET rejects the first identity and fails its receipt",
          "[s3rdma][data-plane]")
{
  data_fixture f;
  for (unsigned n = 0; n < 8; ++n) {
    f.server.respond(accepted_response());
    CHECK(f.get().sdk_return == 4);
  }
  binding_options options;
  options.late_before = 0;
  f.module.configure(options);
  f.server.respond(accepted_response());
  auto const result =
    f.session->get_if_match({"test", "ninth"}, 19, 4, f.destination.data(), 0, "\"version-a\"");
  CHECK(result.callback_invocations == 2);
  CHECK(result.sdk_return == 4);
  CHECK_FALSE(result.transport_error.empty());
  CHECK(result.commit == rdma::data_commit_state::completed);
  auto const requests = f.server.requests();
  REQUIRE(requests.size() == 9);
  CHECK(requests.back().target == "/test/ninth");
  CHECK(requests.back().value("range") == "bytes=19-22");
  auto const observations = f.module.snapshot();
  REQUIRE(observations.result_count == 10);
  CHECK(observations.results[8] == -ENOTSUP);
  CHECK(observations.results[9] == 4);
  no_token(f.logs, result.transport_error);
}

TEST_CASE("S3RDMA preserves binding results including no result", "[s3rdma][data-plane]")
{
  for (std::int64_t value : {std::int64_t{-77}, std::int64_t{123}, CUCASCADE_S3RDMA_NO_RESULT}) {
    data_fixture f;
    binding_options options;
    options.override_result = true;
    options.sdk_result      = value;
    f.module.configure(options);
    f.server.respond(accepted_response());
    auto const result = f.get();
    CHECK(result.sdk_return == value);
    CHECK(result.callback_invocations == 1);
    CHECK(result.commit == rdma::data_commit_state::completed);
    no_token(f.logs, result.transport_error);
  }
}

TEST_CASE("S3RDMA authorization failure and invalid GET arguments precede issue",
          "[s3rdma][data-plane]")
{
  data_fixture f;
  rdma::object_ref object{"test", "object"};
  CHECK_THROWS_AS(f.session->get(object, 0, 0, f.destination.data(), 0), std::invalid_argument);
  CHECK_THROWS_AS(f.session->get(object, 0, 4, nullptr, 0), std::invalid_argument);
  CHECK_THROWS_AS(
    f.session->get(object, std::numeric_limits<std::size_t>::max(), 4, f.destination.data(), 0),
    std::invalid_argument);
  CHECK_THROWS_AS(f.session->get_if_match(object, 0, 4, f.destination.data(), 0, "W/\"weak\""),
                  std::invalid_argument);
  CHECK(f.module.snapshot().gets == 0);
  f.auth->reject    = true;
  f.auth->message   = "CAPABILITY-NOT-FOR-LOGS";
  auto const result = f.get();
  CHECK(result.sdk_return == -ENOTSUP);
  CHECK(result.commit == rdma::data_commit_state::not_sent);
  CHECK(result.callback_invocations == 1);
  CHECK_FALSE(result.transport_error.empty());
  CHECK(f.server.requests().empty());
  no_token(f.logs, result.transport_error);
}

TEST_CASE("S3RDMA a representable final-byte data range reaches the wire", "[s3rdma][data-plane]")
{
  data_fixture f;
  f.server.respond(accepted_response());
  auto const offset = std::numeric_limits<std::size_t>::max() - 1;
  auto const result = f.session->get({"test", "object"}, offset, 1, f.destination.data(), 0);
  CHECK(result.sdk_return == 1);
  REQUIRE(f.server.requests().size() == 1);
  CHECK(f.server.requests()[0].value("range") ==
        "bytes=" + std::to_string(offset) + "-" + std::to_string(offset));
}

namespace {

struct callback_record {
  unsigned calls{0};
  std::size_t size{0};
  std::int64_t offset{0};
  std::size_t descriptor_length{0};
  std::array<char, 128> descriptor{};
  bool null_descriptor{false};
  std::int64_t result{-37};
};

std::int64_t capture_request(void* context, cucascade_s3rdma_request const* request) noexcept
{
  if (context == nullptr || request == nullptr) { return -333; }
  auto& record = *static_cast<callback_record*>(context);
  ++record.calls;
  record.size              = request->size;
  record.offset            = request->offset;
  record.descriptor_length = request->descriptor_length;
  record.null_descriptor   = request->descriptor == nullptr;
  if (!record.null_descriptor) {
    std::memcpy(record.descriptor.data(),
                request->descriptor,
                std::min(record.descriptor.size(), request->descriptor_length));
  }
  return record.result;
}

struct cuobj_fixture {
  cuobj_fixture() : module(cuobj_test_path(), true)
  {
    module.reset();
    using entry = cucascade_s3rdma_api const* (*)();
    api         = module.symbol<entry>("cucascade_s3rdma_binding")();
    if (api == nullptr || api->session_create(&session) != 0 || session == nullptr) {
      throw std::runtime_error("stand-in binding session could not be created");
    }
  }
  ~cuobj_fixture() { api->session_destroy(session); }
  deadline bound;
  module_view module;
  cucascade_s3rdma_api const* api{nullptr};
  cucascade_s3rdma_session* session{nullptr};
  std::array<std::uint8_t, 32> base{};
};

}  // namespace

TEST_CASE("S3RDMA cuObject binding preserves its protocol callback table and base-offset GET",
          "[s3rdma][cuobj-binding]")
{
  cuobj_fixture f;
  cucascade_s3rdma_probe probe{};
  CHECK(f.api->probe(&probe) == 0);
  CHECK(probe.protocol == CUCASCADE_S3RDMA_PROTOCOL_DC_V1);
  CHECK(probe.io_retry_count == -1);
  auto state = f.module.snapshot();
  CHECK(state.creates == 1);
  CHECK(state.protocol == 1001);
  CHECK(state.has_get);
  CHECK_FALSE(state.has_put);
  CHECK(f.api->register_memory(f.session, f.base.data(), f.base.size()) == 0);
  binding_options options;
  options.capacity        = 12345;
  options.override_result = true;
  options.sdk_result      = 27;
  f.module.configure(options);
  CHECK(f.api->max_request_size(f.session, f.base.data()) == 12345);
  callback_record record;
  CHECK(f.api->get(f.session, f.base.data(), 4, 9, &capture_request, &record) == 27);
  CHECK(record.calls == 1);
  CHECK(record.size == 4);
  CHECK(record.offset == 0);
  CHECK(record.descriptor_length == 23);
  CHECK(std::string(record.descriptor.data(), record.descriptor_length) ==
        "CAPABILITY-NOT-FOR-LOGS");
  state = f.module.snapshot();
  REQUIRE(state.result_count == 1);
  CHECK(state.results[0] == record.result);
  CHECK(state.last_offset == 0);
  auto const& get = state.events[state.event_count - 1];
  CHECK(get.operation == binding_operation::get);
  CHECK(get.base == reinterpret_cast<std::uintptr_t>(f.base.data()));
  CHECK(get.bytes == 4);
  CHECK(get.destination == 9);
  CHECK(f.api->deregister_memory(f.session, f.base.data()) == 0);
  f.api->session_destroy(f.session);
  f.session = nullptr;
  CHECK(f.module.snapshot().destroys == 1);
  CHECK(f.module.snapshot().ops_alive_at_destroy);
}

TEST_CASE("S3RDMA cuObject binding contains client exceptions and forwards failure results",
          "[s3rdma][cuobj-binding]")
{
  cuobj_fixture f;
  callback_record record;
  binding_options options;
  options.throw_on = 1;
  f.module.configure(options);
  auto* refused = reinterpret_cast<cucascade_s3rdma_session*>(std::uintptr_t{1});
  CHECK(f.api->session_create(&refused) != 0);
  CHECK(refused == nullptr);
  for (int throwing : {2, 3, 4, 5}) {
    options.throw_on = throwing;
    f.module.configure(options);
    if (throwing == 2) { CHECK(f.api->register_memory(f.session, f.base.data(), 32) != 0); }
    if (throwing == 3) { CHECK(f.api->deregister_memory(f.session, f.base.data()) != 0); }
    if (throwing == 4) { CHECK(f.api->max_request_size(f.session, f.base.data()) == -1); }
    if (throwing == 5) {
      CHECK(f.api->get(f.session, f.base.data(), 4, 0, &capture_request, &record) ==
            CUCASCADE_S3RDMA_NO_RESULT);
    }
  }
  CHECK(record.calls == 0);
  options.throw_on          = 0;
  options.register_result   = 1;
  options.deregister_result = 1;
  options.capacity          = -43;
  f.module.configure(options);
  CHECK(f.api->register_memory(f.session, f.base.data(), 32) != 0);
  CHECK(f.api->deregister_memory(f.session, f.base.data()) != 0);
  CHECK(f.api->max_request_size(f.session, f.base.data()) == -43);
  for (std::int64_t returned : {std::int64_t{-91}, CUCASCADE_S3RDMA_NO_RESULT}) {
    options.override_result = true;
    options.sdk_result      = returned;
    f.module.configure(options);
    CHECK(f.api->get(f.session, f.base.data(), 4, 0, &capture_request, &record) == returned);
  }
}

TEST_CASE("S3RDMA cuObject binding checks null inputs before calling the client",
          "[s3rdma][cuobj-binding]")
{
  cuobj_fixture f;
  callback_record record;
  CHECK(f.api->probe(nullptr) != 0);
  CHECK(f.api->session_create(nullptr) != 0);
  CHECK_NOTHROW(f.api->session_destroy(nullptr));
  CHECK(f.api->register_memory(nullptr, f.base.data(), 32) != 0);
  CHECK(f.api->register_memory(f.session, nullptr, 32) != 0);
  CHECK(f.api->register_memory(f.session, f.base.data(), 0) != 0);
  CHECK(f.api->deregister_memory(nullptr, f.base.data()) != 0);
  CHECK(f.api->deregister_memory(f.session, nullptr) != 0);
  CHECK(f.api->max_request_size(nullptr, f.base.data()) == -1);
  CHECK(f.api->max_request_size(f.session, nullptr) == -1);
  CHECK(f.api->get(nullptr, f.base.data(), 4, 0, &capture_request, &record) ==
        CUCASCADE_S3RDMA_NO_RESULT);
  CHECK(f.api->get(f.session, nullptr, 4, 0, &capture_request, &record) ==
        CUCASCADE_S3RDMA_NO_RESULT);
  CHECK(f.api->get(f.session, f.base.data(), 4, 0, nullptr, &record) == CUCASCADE_S3RDMA_NO_RESULT);
  auto const state = f.module.snapshot();
  CHECK(state.registrations == 0);
  CHECK(state.deregistrations == 0);
  CHECK(state.capacities == 0);
  CHECK(state.gets == 0);
}

TEST_CASE("S3RDMA cuObject binding refuses a null GET context before calling the client",
          "[s3rdma][cuobj-binding]")
{
  cuobj_fixture f;
  CHECK(f.api->get(f.session, f.base.data(), 4, 0, &capture_request, nullptr) ==
        CUCASCADE_S3RDMA_NO_RESULT);
  CHECK(f.module.snapshot().gets == 0);
}

TEST_CASE("S3RDMA cuObject callbacks preserve size offset null RDMA info and results",
          "[s3rdma][cuobj-binding]")
{
  for (auto mode : {callback_mode::threaded,
                    callback_mode::wrong_size,
                    callback_mode::wrong_offset,
                    callback_mode::null_request,
                    callback_mode::null_descriptor}) {
    cuobj_fixture f;
    binding_options options;
    options.mode = mode;
    f.module.configure(options);
    callback_record record;
    CHECK(f.api->get(f.session, f.base.data(), 4, 7, &capture_request, &record) == record.result);
    CHECK(record.calls == 1);
    CHECK(record.size == (mode == callback_mode::wrong_size ? 5 : 4));
    CHECK(record.offset == (mode == callback_mode::wrong_offset ? 17 : 0));
    CHECK(record.null_descriptor ==
          (mode == callback_mode::null_request || mode == callback_mode::null_descriptor));
    CHECK(record.descriptor_length == (mode == callback_mode::null_request ? 0 : 23));
  }
}

TEST_CASE("S3RDMA cuObject ended GETs are refused before and after the ninth call",
          "[s3rdma][cuobj-binding]")
{
  for (bool first_ran : {false, true}) {
    cuobj_fixture f;
    std::array<callback_record, 9> records{};
    binding_options options;
    options.mode = first_ran ? callback_mode::normal : callback_mode::absent;
    f.module.configure(options);
    (void)f.api->get(f.session, f.base.data(), 4, 0, &capture_request, &records[0]);
    CHECK(records[0].calls == (first_ran ? 1 : 0));
    options.mode = callback_mode::normal;
    f.module.configure(options);
    for (std::size_t index = 1; index < 8; ++index) {
      CHECK(f.api->get(f.session, f.base.data(), 4, index, &capture_request, &records[index]) ==
            -37);
      auto const before = records[0].calls;
      CHECK(f.module.fire(0) == -ENOTSUP);
      CHECK(records[0].calls == before);
      CHECK(records[index].calls == 1);
    }
    CHECK(f.api->get(f.session, f.base.data(), 4, 8, &capture_request, &records[8]) == -37);
    auto const first_count = records[0].calls;
    CHECK(f.module.fire(0) == -ENOTSUP);
    CHECK(records[0].calls == first_count);
    CHECK(records[8].calls == 1);
  }
}

TEST_CASE("S3RDMA a stale first callback cannot complete an otherwise absent ninth GET",
          "[s3rdma][data-plane][callback-witness]")
{
  data_fixture f;
  for (unsigned n = 0; n < 8; ++n) {
    f.server.respond(accepted_response());
    auto const result = f.get();
    REQUIRE(result.sdk_return == 4);
    REQUIRE(result.transport_error.empty());
    REQUIRE(result.callback_invocations == 1);
    REQUIRE(f.destination[3] == 0x5a);
  }
  binding_options options;
  options.mode            = callback_mode::absent;
  options.late_before     = 0;
  options.override_result = true;
  options.sdk_result      = 4;
  options.poison_output   = true;
  f.module.configure(options);
  f.server.respond(accepted_response());
  auto const result = f.get();
  CHECK(result.sdk_return == 4);
  CHECK(result.callback_invocations == 1);
  CHECK(result.commit == rdma::data_commit_state::not_sent);
  CHECK_FALSE(result.transport_error.empty());
  CHECK(result.delivered_bytes == 0);
  CHECK(f.server.requests().size() == 8);
  auto const observed = f.module.snapshot();
  REQUIRE(observed.result_count == 9);
  CHECK(observed.results[8] == -ENOTSUP);
  CHECK(std::all_of(
    f.destination.begin() + 3, f.destination.begin() + 7, [](auto byte) { return byte == 0xe1; }));
  CHECK_FALSE(observed.overflow);
}

TEST_CASE("S3RDMA old identities are rejected at varied ages during and between GETs",
          "[s3rdma][data-plane][callback-identity]")
{
  for (unsigned age : {1U, 7U, 8U, 9U, 16U, 17U, 129U}) {
    for (bool between : {false, true}) {
      CAPTURE(age, between);
      data_fixture f;
      std::set<std::uintptr_t> identities;
      for (unsigned n = 0; n < age; ++n) {
        f.server.respond(accepted_response());
        auto const result = f.get();
        REQUIRE(result.sdk_return == 4);
        REQUIRE(result.transport_error.empty());
        auto const identity = f.module.snapshot().last_context;
        CHECK(identity != 0);
        CHECK(identities.insert(identity).second);
      }
      auto const before = f.server.requests().size();
      binding_options options;
      if (between) {
        CHECK(f.module.fire(0) == -ENOTSUP);
        CHECK(f.server.requests().size() == before);
      } else {
        options.late_before = 0;
      }
      f.module.configure(options);
      f.server.respond(accepted_response());
      auto const failed = f.get();
      CHECK(failed.sdk_return == 4);
      CHECK(failed.callback_invocations == 2);
      CHECK_FALSE(failed.transport_error.empty());
      CHECK(f.server.requests().size() == before + 1);
      auto const observed = f.module.snapshot();
      REQUIRE(observed.result_count >= 2);
      CHECK(observed.results[observed.result_count - 2] == -ENOTSUP);
      CHECK(observed.results[observed.result_count - 1] == 4);
      CHECK(identities.insert(observed.last_context).second);
      CHECK_FALSE(observed.overflow);
      f.module.configure({});
      f.server.respond(accepted_response());
      auto const fresh = f.get();
      CHECK(fresh.sdk_return == 4);
      CHECK(fresh.callback_invocations == 1);
      CHECK(fresh.transport_error.empty());
      CHECK(identities.insert(f.module.snapshot().last_context).second);
    }
  }
}

TEST_CASE("S3RDMA a first callback after return fails the next GET without borrowing old inputs",
          "[s3rdma][data-plane][callback-identity]")
{
  data_fixture f;
  binding_options options;
  options.mode            = callback_mode::absent;
  options.override_result = true;
  options.sdk_result      = 4;
  f.module.configure(options);
  {
    rdma::object_ref old{"test", std::string(512, 'o')};
    std::string condition = "\"" + std::string(512, 'v') + "\"";
    auto const result     = f.session->get_if_match(old, 0, 4, f.destination.data(), 0, condition);
    CHECK(result.sdk_return == 4);
    CHECK(result.callback_invocations == 0);
    CHECK(result.commit == rdma::data_commit_state::not_sent);
    CHECK(result.delivered_bytes == 0);
  }
  CHECK(f.module.fire(0) == -ENOTSUP);
  CHECK(f.server.requests().empty());
  f.module.configure({});
  f.server.respond(accepted_response());
  auto const next = f.get();
  CHECK(next.sdk_return == 4);
  CHECK(next.callback_invocations == 2);
  CHECK_FALSE(next.transport_error.empty());
  CHECK(f.server.requests().size() == 1);
  f.server.respond(accepted_response());
  auto const fresh = f.get();
  CHECK(fresh.callback_invocations == 1);
  CHECK(fresh.sdk_return == 4);
  CHECK(fresh.transport_error.empty());
}

TEST_CASE("S3RDMA a destroyed session identity cannot reach its successor",
          "[s3rdma][data-plane][callback-identity]")
{
  data_fixture f;
  f.server.respond(accepted_response());
  REQUIRE(f.get().sdk_return == 4);
  auto const old_identity = f.module.snapshot().last_context;
  REQUIRE(f.session->deregister_memory(f.destination.data()));
  f.session.reset();
  CHECK(f.module.fire(0) == -ENOTSUP);
  CHECK(f.server.requests().size() == 1);
  f.session = f.factory->acquire();
  f.session->register_memory(f.destination.data(), f.destination.size());
  f.server.respond(accepted_response());
  auto const fresh = f.get();
  CHECK(fresh.sdk_return == 4);
  CHECK(fresh.callback_invocations == 1);
  CHECK(fresh.transport_error.empty());
  CHECK(f.module.snapshot().last_context != old_identity);
  CHECK(f.module.fire(0) == -ENOTSUP);
  CHECK(f.server.requests().size() == 2);
  f.server.respond(accepted_response());
  CHECK(f.get().transport_error.empty());
}

namespace {

class receipt_factory final : public rdma::rdma_data_session_factory {
  class session final : public rdma::rdma_data_session {
   public:
    session(receipt_factory& owner, std::unique_ptr<rdma::rdma_data_session> inner)
      : owner(owner), inner(std::move(inner))
    {
    }
    ~session() override
    {
      std::lock_guard lock{owner.mutex};
      owner.live.erase(this);
    }
    void register_memory(std::uint8_t* base, std::size_t bytes) override
    {
      inner->register_memory(base, bytes);
    }
    bool deregister_memory(std::uint8_t* base) noexcept override
    {
      return inner->deregister_memory(base);
    }
    std::int64_t max_request_callback_size(std::uint8_t* base) noexcept override
    {
      return inner->max_request_callback_size(base);
    }
    rdma::data_get_result get(rdma::object_ref const& object,
                              std::size_t offset,
                              std::size_t bytes,
                              std::uint8_t* base,
                              std::size_t destination) override
    {
      return get_if_match(object, offset, bytes, base, destination, {});
    }
    rdma::data_get_result get_if_match(rdma::object_ref const& object,
                                       std::size_t offset,
                                       std::size_t bytes,
                                       std::uint8_t* base,
                                       std::size_t destination,
                                       std::string_view condition) override
    {
      auto result = inner->get_if_match(object, offset, bytes, base, destination, condition);
      std::lock_guard lock{owner.mutex};
      owner.receipts.push_back(result);
      return result;
    }
    receipt_factory& owner;
    std::unique_ptr<rdma::rdma_data_session> inner;
  };

 public:
  explicit receipt_factory(std::shared_ptr<rdma::rdma_data_session_factory> inner)
    : inner(std::move(inner))
  {
  }
  std::unique_ptr<rdma::rdma_data_session> acquire() override
  {
    auto result = std::make_unique<session>(*this, inner->acquire());
    std::lock_guard lock{mutex};
    live.insert(result.get());
    return result;
  }
  void reclaim()
  {
    for (;;) {
      rdma::rdma_data_session* held;
      {
        std::lock_guard lock{mutex};
        if (live.empty()) { return; }
        held = *live.begin();
        live.erase(live.begin());
      }
      delete held;
    }
  }
  std::vector<rdma::data_get_result> snapshot()
  {
    std::lock_guard lock{mutex};
    return receipts;
  }

 private:
  std::shared_ptr<rdma::rdma_data_session_factory> inner;
  std::mutex mutex;
  std::set<rdma::rdma_data_session*> live;
  std::vector<rdma::data_get_result> receipts;
};

struct wire_runtime {
  wire_runtime()
  {
    auto clients = fixture.controls->clients();
    factory      = std::make_shared<receipt_factory>(
      rdma::make_data_session_factory(signer(server.endpoint()), accepted_rule(), data_config()));
    module = std::make_unique<module_view>(binding_path("good"));
    module->reset();
    clients.data_sessions = factory;
    clients.rule          = accepted_rule();
    fixture.object        = std::make_shared<rdma::s3rdma_io_object>(
      "s3://test/object", rdma::object_ref{"test", "object"}, 64, "\"version-a\"");
    fixture.context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(
      fixture.cfg, clients, fixture.controls->delivery(), fixture.controls->policy());
    fixture.reactor = std::make_unique<rdma::s3rdma_reactor>(fixture.context);
    fixture.start();
  }
  ~wire_runtime()
  {
    fixture.shutdown();
    fixture.reactor.reset();
    fixture.context.reset();
    factory->reclaim();
  }
  runtime_fixture fixture;
  http_fixture server;
  std::shared_ptr<receipt_factory> factory;
  std::unique_ptr<module_view> module;
};

std::int64_t other_request(void* context, cucascade_s3rdma_request const* request) noexcept
{
  return -capture_request(context, request);
}

struct descriptor_observation {
  std::shared_ptr<rdma::s3_control_client> control;
  callback_record record;
  bool failed{false};
  long status{0};
};

std::int64_t observe_descriptor(void* context, cucascade_s3rdma_request const* request) noexcept
{
  auto& observed = *static_cast<descriptor_observation*>(context);
  ++observed.record.calls;
  observed.record.descriptor_length = request->descriptor_length;
  observed.record.null_descriptor   = request->descriptor == nullptr;
  if (request->descriptor != nullptr &&
      request->descriptor_length <= observed.record.descriptor.size()) {
    std::memcpy(observed.record.descriptor.data(), request->descriptor, request->descriptor_length);
  }
  try {
    observed.status = observed.control->head({"test", "descriptor"}).outcome.http_status;
    return 4;
  } catch (...) {
    observed.failed = true;
    return -ENOTSUP;
  }
}

}  // namespace

TEST_CASE("S3RDMA the reactor refuses the stale callback witness and preserves the destination",
          "[s3rdma][data-plane][callback-witness]")
{
  wire_runtime test;
  auto& f = test.fixture;
  for (unsigned n = 0; n < 8; ++n) {
    test.server.respond(accepted_response());
    auto pending = f.submit(0, 4);
    REQUIRE(wait_until([&] { return pending.future.is_ready(); }));
    REQUIRE(std::move(pending.future).get(2s) == 4);
    REQUIRE(wait_until([&] { return pending.completion->calls.load() == 1; }));
    REQUIRE(pending.completion->success.load());
    REQUIRE(std::all_of(
      f.destination.begin(), f.destination.begin() + 4, [](auto byte) { return byte == 0x5a; }));
    REQUIRE_FALSE(f.context->status()->terminal());
  }
  f.destination.fill(0xcc);
  binding_options options;
  options.mode            = callback_mode::absent;
  options.late_before     = 0;
  options.override_result = true;
  options.sdk_result      = 4;
  options.poison_output   = true;
  test.module->configure(options);
  test.server.respond(accepted_response());
  auto pending = f.submit(0, 4);
  REQUIRE(wait_until([&] { return pending.future.is_ready(); }));
  auto const error = error_from(pending.future);
  CHECK(has_code(error, rdma::errc::transport_failed));
  CHECK(f.context->status()->terminal());
  REQUIRE(wait_until([&] { return pending.completion->calls.load() == 1; }));
  CHECK_FALSE(pending.completion->success.load());
  CHECK(std::all_of(
    f.destination.begin(), f.destination.end(), [](auto byte) { return byte == 0xcc; }));
  CHECK(f.controls->log.count(operation::copy) == 8);
  CHECK(test.server.requests().size() == 8);
  auto const receipts = test.factory->snapshot();
  REQUIRE(receipts.size() == 9);
  CHECK(receipts.back().sdk_return == 4);
  CHECK(receipts.back().callback_invocations == 1);
  CHECK_FALSE(receipts.back().transport_error.empty());
  auto const observations = test.module->snapshot();
  REQUIRE(observations.result_count == 9);
  CHECK(observations.results[8] == -ENOTSUP);
}

TEST_CASE("S3RDMA cuObject late callbacks cannot cross GET or session identities",
          "[s3rdma][cuobj-binding][callback-identity]")
{
  cuobj_fixture f;
  callback_record original;
  original.result = 4;
  REQUIRE(f.api->get(f.session, f.base.data(), 4, 0, &capture_request, &original) == 4);
  REQUIRE(original.calls == 1);
  CHECK(f.module.fire(0) == -ENOTSUP);
  CHECK(original.calls == 1);
  auto const first = f.module.snapshot().last_context;
  std::set<std::uintptr_t> identities{first};
  CHECK(first != 0);
  cucascade_s3rdma_session* raw = nullptr;
  REQUIRE(f.api->session_create(&raw) == 0);
  std::unique_ptr<cucascade_s3rdma_session, decltype(f.api->session_destroy)> second(
    raw, f.api->session_destroy);
  REQUIRE(second != nullptr);
  std::array<callback_record, 34> records{};
  for (std::size_t n = 1; n <= 33; ++n) {
    CAPTURE(n);
    binding_options options;
    options.late_before = 0;
    f.module.configure(options);
    records[n].result = 4;
    auto* session     = n % 2 == 0 ? second.get() : f.session;
    CHECK(f.api->get(session, f.base.data(), 4, 0, &capture_request, &records[n]) == 4);
    CHECK(records[n].calls == 1);
    CHECK(original.calls == 1);
    CHECK(f.module.fire(0) == -ENOTSUP);
    CHECK(original.calls == 1);
    auto const state = f.module.snapshot();
    CHECK(state.last_context != 0);
    CHECK(identities.insert(state.last_context).second);
    REQUIRE(state.result_count >= 3);
    CHECK(state.results[state.result_count - 3] == -ENOTSUP);
    CHECK(state.results[state.result_count - 2] == 4);
    CHECK(state.results[state.result_count - 1] == -ENOTSUP);
    CHECK_FALSE(state.overflow);
  }
}

TEST_CASE("S3RDMA cuObject concurrent GETs retain their own function and context",
          "[s3rdma][cuobj-binding][callback-identity]")
{
  cuobj_fixture f;
  cucascade_s3rdma_session* raw = nullptr;
  REQUIRE(f.api->session_create(&raw) == 0);
  std::unique_ptr<cucascade_s3rdma_session, decltype(f.api->session_destroy)> second(
    raw, f.api->session_destroy);
  REQUIRE(second != nullptr);
  auto gate = std::make_shared<callback_gate>();
  binding_options options;
  options.before_get         = &hold_callback;
  options.before_get_context = gate.get();
  f.module.configure(options);
  std::array<callback_record, 2> records{};
  records[0].result = 41;
  records[1].result = 73;
  workers threads;
  release_callbacks release{gate};
  auto first = threads.launch(
    [&] { return f.api->get(f.session, f.base.data(), 4, 0, &capture_request, &records[0]); });
  auto other = threads.launch(
    [&] { return f.api->get(second.get(), f.base.data(), 7, 0, &other_request, &records[1]); });
  REQUIRE(gate->wait_for_entries(2));
  auto const state = f.module.snapshot();
  REQUIRE(state.gets == 2);
  CHECK(state.contexts[0] != 0);
  CHECK(state.contexts[1] != 0);
  CHECK(state.contexts[0] != state.contexts[1]);
  gate->release();
  CHECK(ready(first) == 41);
  CHECK(ready(other) == -73);
  CHECK(records[0].calls == 1);
  CHECK(records[1].calls == 1);
  CHECK(records[0].size == 4);
  CHECK(records[1].size == 7);
  CHECK_FALSE(gate->timed_out());
}

TEST_CASE("S3RDMA cuObject refuses negative descriptor lengths before forwarding or HTTP",
          "[s3rdma][cuobj-binding][descriptor-witness]")
{
  for (auto mode : {callback_mode::negative_descriptor, callback_mode::minimum_descriptor}) {
    CAPTURE(static_cast<int>(mode));
    cuobj_fixture f;
    http_fixture server;
    descriptor_observation observed{rdma::make_control_client(signer(server.endpoint())), {}};
    binding_options options;
    options.mode = mode;
    f.module.configure(options);
    server.respond({.headers = {{"Content-Length", "4"}}});
    auto const result = f.api->get(f.session, f.base.data(), 4, 0, &observe_descriptor, &observed);
    CHECK(result == -ENOTSUP);
    CHECK(observed.record.calls == 0);
    CHECK(server.requests().empty());
    auto const state = f.module.snapshot();
    REQUIRE(state.result_count == 1);
    CHECK(state.results[0] == -ENOTSUP);
    CAPTURE(observed.record.descriptor_length);
    f.module.configure({});
    server.respond({.headers = {{"Content-Length", "4"}}});
    CHECK(f.api->get(f.session, f.base.data(), 4, 0, &observe_descriptor, &observed) == 4);
    CHECK(observed.record.calls == 1);
    CHECK_FALSE(observed.record.null_descriptor);
    CHECK(observed.record.descriptor_length == 23);
    CHECK(std::string(observed.record.descriptor.data(), 23) == "CAPABILITY-NOT-FOR-LOGS");
    CHECK(observed.status == 200);
    CHECK_FALSE(observed.failed);
    CHECK(server.requests().size() == 1);
    REQUIRE_FALSE(server.requests().empty());
    CHECK(server.requests()[0].method == "HEAD");
    CHECK(server.requests()[0].target == "/test/descriptor");
  }
}

TEST_CASE("S3RDMA overlapping callbacks and client return preserve the borrowed GET",
          "[s3rdma][data-plane][callback-overlap]")
{
  for (std::string const path : {"duplicate", "early-return", "complete", "deadline"}) {
    CAPTURE(path);
    auto config             = data_config();
    config.request_deadline = path == "deadline" ? 200ms : 2000ms;
    data_fixture f(config);
    auto client_gate        = std::make_shared<callback_gate>();
    auto http_gate          = std::make_shared<callback_gate>();
    auto const tag          = "\"" + std::string(256, 'v') + "\"";
    auto response           = accepted_response(tag);
    response.before_headers = http_gate;
    f.server.respond(std::move(response));
    binding_options options;
    options.mode               = callback_mode::absent;
    options.override_result    = true;
    options.sdk_result         = 4;
    options.before_get         = &hold_callback;
    options.before_get_context = client_gate.get();
    f.module.configure(options);
    workers threads;
    release_callbacks release{client_gate, http_gate};
    auto get = threads.launch([&] {
      rdma::object_ref borrowed{"test", std::string(256, 'o')};
      std::string condition = tag;
      return f.session->get_if_match(borrowed, 17, 4, f.destination.data(), 3, condition);
    });
    REQUIRE(client_gate->wait_for_entries());
    auto first = threads.launch([&] { return f.module.fire(0); });
    REQUIRE(http_gate->wait_for_entries());
    REQUIRE(f.server.wait_for_requests(1));
    std::future<std::int64_t> second;
    if (path == "duplicate") {
      second = threads.launch([&] { return f.module.fire(0); });
      REQUIRE(wait_until([&] { return f.module.snapshot().callback_starts == 2; }));
    }
    if (path == "early-return" || path == "deadline") {
      client_gate->release();
      REQUIRE(wait_until([&] { return f.module.snapshot().returns == 1; }));
      if (path == "early-return") {
        CHECK(get.wait_for(100ms) == std::future_status::timeout);
        CHECK(first.wait_for(0ms) == std::future_status::timeout);
      }
    }
    if (path != "deadline") { http_gate->release(); }
    CHECK(ready(first) == (path == "deadline" ? -ENOTSUP : 4));
    if (second.valid()) { CHECK(ready(second) == -ENOTSUP); }
    client_gate->release();
    auto const result = ready(get);
    CHECK(result.sdk_return == 4);
    CHECK(result.callback_invocations == (path == "duplicate" ? 2 : 1));
    CHECK(result.commit == (path == "deadline" ? rdma::data_commit_state::sent_unknown
                                               : rdma::data_commit_state::completed));
    CHECK(result.delivered_bytes == (path == "deadline" ? 0 : 4));
    if (path == "deadline") { CHECK_FALSE(result.transport_error.empty()); }
    if (path == "complete" || path == "early-return") { CHECK(result.transport_error.empty()); }
    auto const state = f.module.snapshot();
    REQUIRE(state.result_count == (path == "duplicate" ? 2 : 1));
    if (path == "duplicate") {
      CHECK(std::count(state.results.begin(), state.results.begin() + 2, -ENOTSUP) == 1);
      CHECK(std::count(state.results.begin(), state.results.begin() + 2, 4) == 1);
    } else {
      CHECK(state.results[0] == (path == "deadline" ? -ENOTSUP : 4));
    }
    REQUIRE(f.server.requests().size() == 1);
    CHECK(f.server.requests()[0].target == "/test/" + std::string(256, 'o'));
    CHECK(f.server.requests()[0].value("if-match") == tag);
    CHECK_FALSE(client_gate->timed_out());
    CHECK_FALSE(http_gate->timed_out());
    http_gate->release();
  }
}

TEST_CASE("S3RDMA data GET preserves period segments and signs the observed target",
          "[s3rdma][data-plane][path-witness]")
{
  deadline bound;
  http_fixture server;
  auto factory = rdma::make_data_session_factory(
    signer(server.endpoint(), "data-fixture", "data-secret"), accepted_rule(), data_config());
  module_view module(binding_path("good"));
  module.reset();
  auto session = factory->acquire();
  std::array<std::uint8_t, 32> destination{};
  session->register_memory(destination.data(), destination.size());
  std::size_t count = 0;
  for (std::string const key : {"object", "dir/./object", "dir/../object"}) {
    CAPTURE(key);
    server.respond(accepted_response());
    auto const result =
      session->get_if_match({"test", key}, 0, 4, destination.data(), 0, "\"version-a\"");
    CHECK(result.sdk_return == 4);
    CHECK(result.transport_error.empty());
    auto const requests = server.requests();
    ++count;
    REQUIRE(requests.size() == count);
    auto const& request = requests.back();
    CHECK(request.method == "GET");
    CHECK(request.target == "/test/" + key);
    CHECK(request.value("range") == "bytes=0-3");
    CHECK(request.value("if-match") == "\"version-a\"");
    CHECK(request.value("x-amz-rdma-token") == "CAPABILITY-NOT-FOR-LOGS");
    auto const auth      = request.value("authorization");
    auto const signature = auth.find("Signature=");
    REQUIRE(signature != std::string::npos);
    auto const names = "host;if-match;range;x-amz-content-sha256;x-amz-date;x-amz-rdma-token";
    CHECK(auth.substr(signature + 10) ==
          reference_sigv4(request.headers, "GET", request.target, names, "data-secret"));
  }
  CHECK(session->deregister_memory(destination.data()));
}
