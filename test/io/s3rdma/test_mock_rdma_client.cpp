/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mock_rdma_client.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <limits>
#include <set>

using namespace cucascade::test::s3rdma;

namespace {

struct fixture {
  std::shared_ptr<mock_rdma_client> mock{std::make_shared<mock_rdma_client>()};
  rdma::object_ref ref{"test", "object"};
  std::vector<std::uint8_t> source{2, 4, 6, 8, 10, 12, 14, 16};
  std::array<std::uint8_t, 24> destination;
  std::unique_ptr<rdma::rdma_data_session> session{mock->acquire()};
  fixture()
  {
    destination.fill(0xa5);
    mock->put(ref, source);
    session->register_memory(destination.data() + 4, 16);
  }
  ~fixture() { session->deregister_memory(destination.data() + 4); }
  rdma::data_get_result get(std::size_t size = 4, std::size_t landing = 3)
  {
    return session->get(ref, 1, size, destination.data() + 4, landing);
  }
};

}  // namespace

TEST_CASE("S3RDMA mock registrations serve distinct sessions", "[s3rdma][mock]")
{
  fixture f;
  auto worker = f.mock->acquire();
  CHECK(worker.get() != f.session.get());
  CHECK(worker->max_request_callback_size(f.destination.data() + 4) == 16);
  auto result = worker->get(f.ref, 0, 2, f.destination.data() + 4, 0);
  CHECK(result.delivered_bytes == 2);
  CHECK(f.destination[4] == 2);
  f.session->deregister_memory(f.destination.data() + 4);
  auto const before = f.destination;
  require_code([&] { (void)worker->get(f.ref, 0, 2, f.destination.data() + 4, 0); },
               rdma::errc::unsupported_request);
  CHECK(f.destination == before);
  CHECK(f.mock->counts().sessions == 2);
  CHECK(f.mock->counts().registrations == 1);
  CHECK(f.mock->counts().deregistrations == 1);
}

TEST_CASE("S3RDMA mock domain refusals never write", "[s3rdma][mock]")
{
  fixture f;
  auto const before = f.destination;
  auto* base        = f.destination.data() + 4;
  auto landing      = std::size_t{0};
  auto length       = std::size_t{4};
  SECTION("unregistered base") { base = f.destination.data(); }
  SECTION("interior pointer is not a registration") { ++base; }
  SECTION("landing past the registration") { landing = 17; }
  SECTION("landing plus length exceeds registration") { landing = 14; }
  SECTION("overflowing landing") { landing = std::numeric_limits<std::size_t>::max(); }
  SECTION("overflowing length") { length = std::numeric_limits<std::size_t>::max(); }
  require_code([&] { (void)f.session->get(f.ref, 0, length, base, landing); },
               rdma::errc::unsupported_request);
  CHECK(f.destination == before);
  CHECK(f.mock->counts().issued_requests == 0);
}

TEST_CASE("S3RDMA mock capacity lookup uses each registration base", "[s3rdma][mock]")
{
  fixture f;
  CHECK(f.session->max_request_callback_size(f.destination.data() + 4) == 16);
  CHECK(f.session->max_request_callback_size(f.destination.data() + 5) < 0);
  f.mock->callback_limit(3);
  CHECK(f.session->max_request_callback_size(f.destination.data() + 4) == 3);
  f.mock->callback_limit(-1);
  CHECK(f.session->max_request_callback_size(f.destination.data() + 4) == -1);
  f.mock->callback_limit({});
  CHECK(f.session->max_request_callback_size(f.destination.data() + 4) == 16);
  CHECK_THROWS_AS(f.session->register_memory(nullptr, 8), std::invalid_argument);
  CHECK_THROWS_AS(f.session->register_memory(f.destination.data(), 0), std::invalid_argument);
  CHECK_THROWS_AS(f.session->register_memory(f.destination.data() + 4, 1), std::invalid_argument);
  CHECK(f.session->max_request_callback_size(f.destination.data() + 4) == 16);
}

TEST_CASE("S3RDMA mock completed GET writes only the requested window", "[s3rdma][mock]")
{
  fixture f;
  auto landing = std::size_t{3};
  SECTION("interior landing") {}
  SECTION("landing ends at the registration boundary") { landing = 12; }
  auto expected = f.destination;
  std::copy_n(f.source.data() + 1, 4, expected.data() + 4 + landing);
  auto result = f.get(4, landing);
  CHECK(f.destination == expected);
  CHECK(result.commit == rdma::data_commit_state::completed);
  CHECK(result.sdk_return == 0);
  CHECK(result.callback_invocations == 1);
  CHECK(result.delivered_bytes == 4);
  CHECK(result.http_status == 206);
  CHECK(result.reply_tag == "mock-rdma");
  CHECK(result.transport_error.empty());
  CHECK(f.mock->counts().issued_requests == 1);
}

TEST_CASE("S3RDMA mock scripts distinguish delivery evidence", "[s3rdma][mock]")
{
  fixture f;
  auto behavior = get_behavior::completed;
  SECTION("not sent") { behavior = get_behavior::not_sent; }
  SECTION("sent unknown") { behavior = get_behavior::sent_unknown; }
  SECTION("completed") { behavior = get_behavior::completed; }
  SECTION("completed short") { behavior = get_behavior::completed_short; }
  SECTION("no callback") { behavior = get_behavior::no_callback; }
  SECTION("callback reentered") { behavior = get_behavior::reentered_callback; }
  f.mock->script(behavior);
  auto expected     = f.destination;
  auto const result = f.get();
  bool const issued = behavior != get_behavior::not_sent && behavior != get_behavior::no_callback;
  auto const bytes  = issued ? (behavior == get_behavior::completed_short ? 3U : 4U) : 0U;
  std::copy_n(f.source.data() + 1, bytes, expected.data() + 7);
  CHECK(f.destination == expected);
  CHECK(result.delivered_bytes == bytes);
  CHECK(result.callback_invocations ==
        (issued ? (behavior == get_behavior::reentered_callback ? 2U : 1U) : 0U));
  CHECK(result.commit ==
        (issued ? (behavior == get_behavior::sent_unknown ? rdma::data_commit_state::sent_unknown
                                                          : rdma::data_commit_state::completed)
                : rdma::data_commit_state::not_sent));
  CHECK(result.http_status == (issued ? 206 : 0));
  CHECK(result.transport_error.empty() == (behavior != get_behavior::sent_unknown));
  CHECK(f.mock->counts().issued_requests == (issued ? 1 : 0));
}

TEST_CASE("S3RDMA mock resets every GET receipt and preserves SDK return", "[s3rdma][mock]")
{
  fixture f;
  f.mock->script(get_behavior::reentered_callback, -37);
  auto first = f.get();
  CHECK(first.sdk_return == -37);
  CHECK(first.callback_invocations == 2);
  f.mock->script(get_behavior::no_callback, 91);
  auto second = f.get();
  CHECK(second.sdk_return == 91);
  CHECK(second.callback_invocations == 0);
  CHECK(second.delivered_bytes == 0);
  CHECK(second.http_status == 0);
  CHECK(second.reply_tag.empty());
  CHECK(second.transport_error.empty());
  CHECK(f.get().callback_invocations == 1);
  CHECK(f.mock->counts().gets == 3);
  CHECK(f.mock->counts().issued_requests == 2);
}

TEST_CASE("S3RDMA mock local failure can occur after issuance", "[s3rdma][mock]")
{
  fixture f;
  f.mock->script(get_behavior::throw_error);
  require_code([&] { (void)f.get(); }, rdma::errc::internal_error);
  CHECK(f.mock->counts().issued_requests == 1);
  CHECK(f.destination[7] == f.source[1]);
  CHECK(f.get().callback_invocations == 1);
  CHECK(f.mock->counts().issued_requests == 2);
}

TEST_CASE("S3RDMA mock control results and bounds are independent of RDMA", "[s3rdma][mock]")
{
  fixture f;
  auto const head = f.mock->head(f.ref);
  CHECK(head.outcome.http_status == 200);
  CHECK(head.outcome.transport_ok());
  CHECK(head.object_size == 8);
  CHECK(f.mock->head({"test", "absent"}).outcome.http_status == 404);
  auto expected = f.destination;
  std::copy_n(f.source.data() + 2, 3, expected.data() + 1);
  auto range = f.mock->range_get(f.ref, 2, 3, f.destination.data() + 1);
  CHECK(range.outcome.http_status == 206);
  CHECK(range.delivered_bytes == 3);
  CHECK(range.content_range == "bytes 2-4/8");
  CHECK(f.destination == expected);
  CHECK(f.mock->range_get(f.ref, 7, 2, f.destination.data()).outcome.http_status == 416);
  CHECK(f.mock->range_get(f.ref, std::numeric_limits<std::size_t>::max(), 1, f.destination.data())
          .outcome.http_status == 416);
  CHECK(f.mock->range_get({"test", "absent"}, 0, 1, f.destination.data()).outcome.http_status ==
        404);
  CHECK(f.destination == expected);
  CHECK(f.mock->counts().gets == 0);
  rdma::control_outcome outcome{206, "truncated"};
  CHECK_FALSE(outcome.transport_ok());
  CHECK(outcome.http_status == 206);
}

TEST_CASE("S3RDMA mock factories and control calls support concurrent users", "[s3rdma][mock]")
{
  deadline bound;
  fixture f;
  workers threads;
  using sessions = std::vector<std::unique_ptr<rdma::rdma_data_session>>;
  std::vector<std::future<sessions>> results;
  for (int i = 0; i < 4; ++i) {
    results.push_back(threads.launch([&] {
      sessions owned;
      for (int j = 0; j < 8; ++j) {
        owned.push_back(f.mock->acquire());
        if (f.mock->head(f.ref).object_size != 8) { throw std::runtime_error("wrong HEAD"); }
      }
      return owned;
    }));
  }
  std::vector<sessions> retained;
  std::set<rdma::rdma_data_session*> addresses;
  for (auto& result : results) {
    retained.push_back(ready(result));
    for (auto const& session : retained.back()) {
      CHECK(addresses.insert(session.get()).second);
    }
  }
  CHECK(addresses.size() == 32);
  CHECK(f.mock->counts().sessions == 33);
  CHECK(f.mock->counts().heads == 32);
}
