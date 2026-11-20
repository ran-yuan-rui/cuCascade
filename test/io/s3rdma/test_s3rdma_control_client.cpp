/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "http_fixture.hpp"
#include "test_support.hpp"
#include "transport_test_support.hpp"

#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/s3rdma/control_client.hpp>

#include <array>
#include <limits>
#include <memory>

using namespace cucascade::test::s3rdma;
namespace rest = cucascade::io::rest;

TEST_CASE("S3RDMA HEAD and control ranges preserve period-only key segments",
          "[s3rdma][control][path-witness]")
{
  deadline bound;
  http_fixture server;
  auto client       = rdma::make_control_client(signer(server.endpoint()));
  std::size_t count = 0;
  for (std::string const key : {"object", "dir/./object", "dir/../object"}) {
    CAPTURE(key);
    server.respond({.headers = {{"Content-Length", "4"}}});
    server.respond({.status = 206, .headers = {{"Content-Range", "bytes 0-3/4"}}, .body = "data"});
    rdma::object_ref object{"test", key};
    auto const head = client->head(object);
    CHECK(head.outcome.http_status == 200);
    CHECK(head.outcome.transport_error.empty());
    CHECK(head.object_size == 4);
    std::array<std::uint8_t, 4> destination{};
    auto const get = client->range_get(object, 0, destination.size(), destination.data());
    CHECK(get.outcome.http_status == 206);
    CHECK(get.outcome.transport_error.empty());
    CHECK(get.delivered_bytes == 4);
    CHECK(std::string(destination.begin(), destination.end()) == "data");
    auto const requests = server.requests();
    REQUIRE(requests.size() == count + 2);
    CHECK(requests[count].method == "HEAD");
    CHECK(requests[count].target == "/test/" + key);
    CHECK(requests[count + 1].method == "GET");
    CHECK(requests[count + 1].target == "/test/" + key);
    CHECK(requests[count + 1].value("range") == "bytes=0-3");
    count += 2;
  }
}

TEST_CASE("S3RDMA control requests use the HTTP fixture despite an h2c offer", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  server.respond({.headers = {{"Content-Length", "12"}, {"ETag", "\"version-a\""}}});
  server.respond({.status  = 206,
                  .headers = {{"Content-Range", "bytes 4-7/12"}, {"ETag", "\"version-a\""}},
                  .body    = "data"});
  rest::s3::static_credentials credentials;
  credentials.access_key_id     = "u3-control";
  credentials.secret_access_key = "u3-control-secret";
  auto authorizer               = std::make_shared<rest::s3::sigv4_header_authorizer>(
    credentials, "us-east-1", server.endpoint());
  auto client = rdma::make_control_client(authorizer);
  rdma::object_ref object{"test", "object"};
  auto const head = client->head(object);
  CHECK(head.outcome.http_status == 200);
  CHECK(head.outcome.transport_error.empty());
  CHECK(head.object_size == 12);
  CHECK(head.validation_tag == "\"version-a\"");
  std::array<std::uint8_t, 6> destination{};
  destination.fill(0xcc);
  auto const get =
    client->range_get_if_match(object, 4, 4, destination.data() + 1, "\"version-a\"");
  CHECK(get.outcome.http_status == 206);
  CHECK(get.outcome.transport_error.empty());
  CHECK(get.delivered_bytes == 4);
  CHECK(get.content_range == "bytes 4-7/12");
  CHECK(get.validation_tag == "\"version-a\"");
  CHECK(destination.front() == 0xcc);
  CHECK(destination.back() == 0xcc);
  CHECK(std::string(destination.begin() + 1, destination.end() - 1) == "data");
  REQUIRE(server.wait_for_requests(2));
  auto const requests = server.requests();
  REQUIRE(requests.size() == 2);
  CHECK(server.failure().empty());
  CHECK(requests[0].method == "HEAD");
  CHECK(requests[1].method == "GET");
  CHECK(requests[0].target == "/test/object");
  CHECK(requests[1].target == "/test/object");
  CHECK(requests[0].connection != requests[1].connection);
  CHECK(requests[1].value("range") == "bytes=4-7");
  CHECK(requests[1].value("if-match") == "\"version-a\"");
  auto const auth = requests[1].value("authorization");
  CHECK(auth.find("Credential=u3-control/") != std::string::npos);
  CHECK(auth.find("SignedHeaders=host;x-amz-content-sha256;x-amz-date") != std::string::npos);
  CHECK(auth.find("if-match") == std::string::npos);
  CHECK(auth.find("range") == std::string::npos);
}

TEST_CASE("S3RDMA HEAD distinguishes unusable lengths from a real empty object",
          "[s3rdma][control]")
{
  struct example {
    std::vector<std::string> lengths;
    bool usable;
    std::size_t bytes;
  };
  for (auto const& value : std::vector<example>{{{}, false, 0},
                                                {{"0"}, true, 0},
                                                {{"17"}, true, 17},
                                                {{"17", "17"}, true, 17},
                                                {{"17", "18"}, false, 0},
                                                {{"-1"}, false, 0},
                                                {{"no-number"}, false, 0},
                                                {{"184467440737095516160"}, false, 0}}) {
    deadline bound;
    http_fixture server;
    http_response response;
    response.add_content_length = false;
    response.headers.emplace_back("ETag", "W/\"raw-tag\"");
    for (auto const& length : value.lengths) {
      response.headers.emplace_back("Content-Length", length);
    }
    server.respond(std::move(response));
    auto client       = rdma::make_control_client(signer(server.endpoint()));
    auto const result = client->head({"test", "object"});
    CAPTURE(value.lengths);
    CHECK(result.outcome.http_status == 200);
    CHECK(result.outcome.transport_error.empty() == value.usable);
    if (value.usable) {
      CHECK(result.object_size == value.bytes);
      CHECK(result.validation_tag == "W/\"raw-tag\"");
    }
    CHECK(server.requests().size() == 1);
    CHECK(server.failure().empty());
  }
}

TEST_CASE("S3RDMA control failures preserve status and do not follow a redirect",
          "[s3rdma][control]")
{
  deadline bound;
  http_fixture origin;
  http_fixture redirected;
  origin.respond({.status = 307, .headers = {{"Location", redirected.endpoint() + "/moved"}}});
  origin.respond({.status = 404});
  origin.respond({.headers = {{"Content-Length", "10"}}, .body = "short"});
  auto client = rdma::make_control_client(signer(origin.endpoint()));
  CHECK(client->head({"test", "object"}).outcome.http_status == 307);
  CHECK(client->head({"test", "object"}).outcome.http_status == 404);
  auto const partial = client->list_page("test", "list-type=2");
  CHECK(partial.outcome.http_status == 200);
  CHECK_FALSE(partial.outcome.transport_error.empty());
  CHECK(partial.body.empty());
  CHECK(origin.requests().size() == 3);
  CHECK(redirected.requests().empty());
  CHECK(origin.failure().empty());
}

TEST_CASE("S3RDMA control authorization and input refusals send no request", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  auto auth = std::make_shared<wire_authorizer>(server.endpoint());
  CHECK_THROWS_AS(rdma::make_control_client(nullptr), std::invalid_argument);
  for (auto timeout : {0s, -1s}) {
    rdma::control_client_config config;
    config.request_timeout = timeout;
    CHECK_THROWS_AS(rdma::make_control_client(auth, config), std::invalid_argument);
  }
  auto client = rdma::make_control_client(auth);
  std::array<std::uint8_t, 8> bytes{};
  rdma::object_ref object{"test", "object"};
  CHECK_THROWS_AS(client->range_get(object, 0, 0, bytes.data()), std::invalid_argument);
  CHECK_THROWS_AS(client->range_get(object, 0, 4, nullptr), std::invalid_argument);
  CHECK_THROWS_AS(
    client->range_get(object, std::numeric_limits<std::size_t>::max(), 2, bytes.data()),
    std::invalid_argument);
  CHECK_THROWS_AS(client->range_get_if_match(object, 0, 4, bytes.data(), "W/\"weak\""),
                  std::invalid_argument);
  CHECK(auth->calls.load() == 0);
  auth->reject = true;
  CHECK_THROWS_AS(client->head(object), cucascade::io::credential_error);
  CHECK_THROWS_AS(client->range_get(object, 0, 4, bytes.data()), cucascade::io::credential_error);
  CHECK_THROWS_AS(client->list_page("test", "list-type=2"), cucascade::io::credential_error);
  CHECK(auth->calls.load() == 3);
  CHECK(server.requests().empty());
}

TEST_CASE("S3RDMA a representable final-byte control range reaches the wire", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  server.respond({.status = 206, .body = "x"});
  auto client = rdma::make_control_client(signer(server.endpoint()));
  std::array<std::uint8_t, 3> bytes{0xcc, 0xcc, 0xcc};
  auto const offset = std::numeric_limits<std::size_t>::max() - 1;
  auto const result = client->range_get({"test", "object"}, offset, 1, bytes.data() + 1);
  CHECK(result.delivered_bytes == 1);
  CHECK(result.outcome.transport_error.empty());
  CHECK(bytes == std::array<std::uint8_t, 3>{0xcc, 'x', 0xcc});
  auto const requests = server.requests();
  REQUIRE(requests.size() == 1);
  CHECK(requests[0].value("range") ==
        "bytes=" + std::to_string(offset) + "-" + std::to_string(offset));
}

TEST_CASE("S3RDMA a longer control body stops at the destination bound", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  http_response response;
  response.status = 206;
  response.body.assign(65536, 'x');
  response.chunk_bytes       = 8;
  response.pause_after_bytes = 8;
  response.body_pause        = 20s;
  server.respond(std::move(response));
  auto client = rdma::make_control_client(signer(server.endpoint()));
  std::array<std::uint8_t, 6> bytes{};
  bytes.fill(0xcc);
  auto const result = client->range_get({"test", "object"}, 0, 4, bytes.data() + 1);
  CHECK(result.outcome.http_status == 206);
  CHECK(result.outcome.transport_error.empty());
  CHECK(result.delivered_bytes == 5);
  CHECK(bytes.front() == 0xcc);
  CHECK(bytes.back() == 0xcc);
  CHECK(std::all_of(bytes.begin() + 1, bytes.end() - 1, [](auto b) { return b == 'x'; }));
  auto const requests = server.requests();
  REQUIRE(requests.size() == 1);
  CHECK_FALSE(requests[0].response_finished);
  CHECK(requests[0].body_sent < 65536);
}

TEST_CASE("S3RDMA unsuccessful control responses never become payload", "[s3rdma][control]")
{
  for (int const status : {403, 412, 500}) {
    deadline bound;
    http_fixture server;
    server.respond({.status = status, .body = "error body"});
    auto client = rdma::make_control_client(signer(server.endpoint()));
    std::array<std::uint8_t, 6> bytes{};
    bytes.fill(0xcc);
    auto const result = client->range_get({"test", "object"}, 0, 4, bytes.data() + 1);
    CHECK(result.outcome.http_status == status);
    CHECK(result.delivered_bytes == 0);
    CHECK(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == 0xcc; }));
    CHECK(server.requests().size() == 1);
  }
}

TEST_CASE("S3RDMA a stalled control body ends through the low-speed bound", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  http_response response;
  response.body              = "data";
  response.chunk_bytes       = 1;
  response.pause_after_bytes = 1;
  response.body_pause        = 20s;
  server.respond(response);
  rdma::control_client_config config;
  config.request_timeout        = 10s;
  config.stall_time             = 1s;
  config.stall_bytes_per_second = 1024;
  auto client                   = rdma::make_control_client(signer(server.endpoint()), config);
  std::array<std::uint8_t, 6> bytes{};
  bytes.fill(0xcc);
  auto const start   = std::chrono::steady_clock::now();
  auto const result  = client->range_get({"test", "object"}, 0, 4, bytes.data() + 1);
  auto const elapsed = std::chrono::steady_clock::now() - start;
  CHECK_FALSE(result.outcome.transport_error.empty());
  CHECK(result.delivered_bytes == 1);
  CHECK(elapsed >= 500ms);
  CHECK(elapsed < 4s);
  CHECK(bytes.front() == 0xcc);
  CHECK(bytes.back() == 0xcc);
}

TEST_CASE("S3RDMA a control GET can outlive the HEAD and LIST total timeout", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  server.respond({.body = "data", .chunk_bytes = 1, .pause_after_bytes = 1, .body_pause = 1500ms});
  rdma::control_client_config config;
  config.request_timeout = 1s;
  config.stall_time      = 3s;
  auto client            = rdma::make_control_client(signer(server.endpoint()), config);
  std::array<std::uint8_t, 4> bytes{};
  auto const start  = std::chrono::steady_clock::now();
  auto const result = client->range_get({"test", "object"}, 0, 4, bytes.data());
  CHECK(std::chrono::steady_clock::now() - start >= 1s);
  CHECK(result.outcome.transport_error.empty());
  CHECK(result.delivered_bytes == 4);
  CHECK(std::string(bytes.begin(), bytes.end()) == "data");
}

TEST_CASE("S3RDMA HEAD and LIST have a total deadline", "[s3rdma][control]")
{
  bool const head = GENERATE(true, false);
  deadline bound;
  http_fixture server;
  server.respond({.header_delay = 20s});
  rdma::control_client_config config;
  config.request_timeout = 1s;
  auto client            = rdma::make_control_client(signer(server.endpoint()), config);
  auto const start       = std::chrono::steady_clock::now();
  auto const outcome     = head ? client->head({"test", "object"}).outcome
                                : client->list_page("test", "list-type=2").outcome;
  CHECK_FALSE(outcome.transport_error.empty());
  CHECK(std::chrono::steady_clock::now() - start >= 500ms);
  CHECK(std::chrono::steady_clock::now() - start < 4s);
  CHECK(server.requests().size() == 1);
}

TEST_CASE("S3RDMA LIST returns only a complete bounded response", "[s3rdma][control]")
{
  deadline bound;
  http_fixture server;
  server.respond({.body = "123456789"});
  server.respond({.body = "12345678"});
  rdma::control_client_config config;
  config.max_list_body_bytes = 8;
  auto client                = rdma::make_control_client(signer(server.endpoint()), config);
  auto const long_body       = client->list_page("test", "list-type=2&prefix=p%2F");
  CHECK_FALSE(long_body.outcome.transport_error.empty());
  CHECK(long_body.body.empty());
  auto const exact = client->list_page("test", "list-type=2&prefix=p%2F");
  CHECK(exact.outcome.transport_error.empty());
  CHECK(exact.body == "12345678");
  auto const requests = server.requests();
  REQUIRE(requests.size() == 2);
  CHECK(requests[0].method == "GET");
  CHECK(requests[0].target == "/test?list-type=2&prefix=p%2F");
  CHECK(requests[0].connection != requests[1].connection);
}
