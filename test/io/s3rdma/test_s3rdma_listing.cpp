/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"
#include "transport_test_support.hpp"

using namespace cucascade::test::s3rdma;

namespace {

std::string page(std::vector<std::string> const& keys, std::string_view continuation = {})
{
  std::string xml = "<ListBucketResult><IsTruncated>";
  xml += continuation.empty() ? "false" : "true";
  xml += "</IsTruncated>";
  for (auto const& key : keys) {
    xml += "<Contents><Key>" + key + "</Key><Size>4</Size><ETag>&quot;v1&quot;</ETag></Contents>";
  }
  if (!continuation.empty()) {
    xml += "<NextContinuationToken>" + std::string{continuation} + "</NextContinuationToken>";
  }
  return xml + "</ListBucketResult>";
}

struct listing_context {
  explicit listing_context(std::shared_ptr<rdma::s3_control_client> control,
                           rdma::config config = runtime_config())
  {
    auto clients         = controls->clients();
    clients.control      = std::move(control);
    auto delivery        = controls->delivery();
    delivery.before_call = [&](rdma::backend_call call) {
      if (call == rdma::backend_call::control_list) { ++calls; }
    };
    context = std::make_unique<rdma::s3rdma_ioctx>(config, clients, delivery, controls->policy());
  }
  ~listing_context()
  {
    context->shutdown();
    context.reset();
    controls->reclaim_sessions();
  }
  deadline bound;
  std::shared_ptr<runtime_controls> controls{std::make_shared<runtime_controls>()};
  std::atomic<unsigned> calls{0};
  std::unique_ptr<rdma::s3rdma_ioctx> context;
};

class held_control final : public rdma::s3_control_client {
 public:
  held_control(std::shared_ptr<rdma::s3_control_client> inner, rendezvous& point)
    : inner(std::move(inner)), point(point)
  {
  }
  rdma::head_result head(rdma::object_ref const& /*object*/) override
  {
    ++head_probes;
    return {{404, {}}, 0, {}};
  }
  rdma::range_get_result range_get(rdma::object_ref const& object,
                                   std::size_t offset,
                                   std::size_t size,
                                   std::uint8_t* dst) override
  {
    return inner->range_get(object, offset, size, dst);
  }
  rdma::list_page_result list_page(std::string_view bucket, std::string_view query) override
  {
    auto result = inner->list_page(bucket, query);
    point.arrive_and_wait();
    return result;
  }
  std::shared_ptr<rdma::s3_control_client> inner;
  rendezvous& point;
  std::atomic<unsigned> head_probes{0};
};

}  // namespace

TEST_CASE("S3RDMA listing sends one control request and entry per page", "[s3rdma][listing]")
{
  http_fixture server;
  server.respond({.body = page({"p/a"}, "next+token")});
  server.respond({.body = page({"p/b"})});
  listing_context test(rdma::make_control_client(signer(server.endpoint())));
  test.context->start();
  auto const entries = test.context->list_objects("test", "p/", 1);
  REQUIRE(entries.size() == 2);
  CHECK(entries[0].key == "p/a");
  CHECK(entries[1].key == "p/b");
  CHECK(entries[0].size == 4);
  CHECK(entries[0].etag == "\"v1\"");
  CHECK(test.calls.load() == 2);
  auto const requests = server.requests();
  REQUIRE(requests.size() == 2);
  CHECK(requests[0].target == "/test?list-type=2&max-keys=1&prefix=p%2F");
  CHECK(requests[1].target.find("continuation-token=next%2Btoken") != std::string::npos);
  CHECK(requests[1].target.find("prefix=p%2F") != std::string::npos);
}

TEST_CASE("S3RDMA listing is refused before start and after shutdown", "[s3rdma][listing]")
{
  http_fixture server;
  listing_context test(rdma::make_control_client(signer(server.endpoint())));
  require_code([&] { (void)test.context->list_objects("test", ""); },
               rdma::errc::backend_unavailable);
  CHECK(test.calls.load() == 0);
  CHECK(server.requests().empty());
  test.context->start();
  server.respond({.body = page({"a"})});
  CHECK(test.context->list_objects("test", "").size() == 1);
  test.context->shutdown();
  require_code([&] { (void)test.context->list_objects("test", ""); }, rdma::errc::transport_closed);
  CHECK(test.calls.load() == 1);
  CHECK(server.requests().size() == 1);
}

TEST_CASE("S3RDMA listing fails on status or incomplete transport without retry",
          "[s3rdma][listing]")
{
  for (bool transport : {false, true}) {
    http_fixture server;
    if (transport) {
      server.respond({.headers = {{"Content-Length", "100"}}, .body = "short"});
    } else {
      server.respond({.status = 503, .body = "unavailable"});
    }
    listing_context test(rdma::make_control_client(signer(server.endpoint())));
    test.context->start();
    require_code([&] { (void)test.context->list_objects("test", ""); }, rdma::errc::control_failed);
    CHECK(test.calls.load() == 1);
    CHECK(server.requests().size() == 1);
    CHECK_FALSE(test.context->status()->terminal());
  }
}

TEST_CASE("S3RDMA both listing caps fail rather than truncate", "[s3rdma][listing]")
{
  for (bool scanned : {false, true}) {
    http_fixture server;
    server.respond({.body = page({"a", "b"})});
    auto config = runtime_config();
    if (scanned) {
      config.list_max_scanned = 1;
    } else {
      config.list_max_matches = 1;
    }
    listing_context test(rdma::make_control_client(signer(server.endpoint())), config);
    test.context->start();
    CHECK(test.context->list_max_matches() == config.list_max_matches);
    CHECK_THROWS_AS(test.context->list_objects("test", ""), std::runtime_error);
    CHECK(server.requests().size() == 1);
  }
  auto mock = std::make_shared<mock_rdma_client>();
  for (bool scanned : {false, true}) {
    auto config = runtime_config();
    if (scanned) {
      config.list_max_scanned = 0;
    } else {
      config.list_max_matches = 0;
    }
    CHECK_THROWS_AS(listing_context(mock, config), std::invalid_argument);
  }
}

TEST_CASE("S3RDMA listing releases its permit before invoking the page consumer",
          "[s3rdma][listing]")
{
  http_fixture server;
  server.respond({.body = page({"a"}, "next")});
  listing_context test(rdma::make_control_client(signer(server.endpoint())));
  test.context->start();
  std::size_t seen = 0;
  test.context->list_objects_paged("test", "", 1, [&](auto const& result) {
    seen += result.entries.size();
    test.context->shutdown();
    return false;
  });
  CHECK(seen == 1);
  CHECK(server.requests().size() == 1);
  CHECK(test.calls.load() == 1);
}

TEST_CASE("S3RDMA shutdown waits for an entered listing page", "[s3rdma][listing]")
{
  http_fixture server;
  server.respond({.body = page({"a"})});
  rendezvous point;
  auto control =
    std::make_shared<held_control>(rdma::make_control_client(signer(server.endpoint())), point);
  listing_context test(control);
  test.context->start();
  require_code([&] { (void)test.context->open_io_object("s3://test/stop-probe"); },
               rdma::errc::control_failed);
  CHECK(control->head_probes.load() == 1);
  workers tasks;
  struct release_guard {
    rendezvous& point;
    ~release_guard() { point.release(); }
  } release{point};
  auto listing = tasks.launch([&] { return test.context->list_objects("test", ""); });
  REQUIRE(wait_until([&] { return point.entered(); }));
  auto closing = tasks.launch([&] { test.context->shutdown(); });
  REQUIRE(wait_until([&] {
    try {
      (void)test.context->open_io_object("s3://test/stop-probe");
    } catch (rdma::error const& error) {
      return error.code() == rdma::errc::transport_closed;
    }
    return false;
  }));
  CHECK(closing.wait_for(0ms) != std::future_status::ready);
  point.release();
  CHECK(ready(listing).size() == 1);
  ready(closing);
  CHECK_FALSE(point.timed_out());
  CHECK(test.calls.load() == 1);
}

TEST_CASE("S3RDMA a client without listing refuses explicitly", "[s3rdma][listing]")
{
  auto mock = std::make_shared<mock_rdma_client>();
  listing_context test(mock);
  test.context->start();
  require_code([&] { (void)test.context->list_objects("test", ""); },
               rdma::errc::backend_unavailable);
  CHECK(test.calls.load() == 1);
  CHECK(mock->counts().heads == 0);
  CHECK(mock->counts().ranges == 0);
}
