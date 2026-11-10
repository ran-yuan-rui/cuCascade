/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Keep the required gate header ahead of unrelated declarations.
// clang-format off
#include <cucascade/io/s3rdma/admission_gate.hpp>
// clang-format on

#include "test_support.hpp"
#include "utils/allocation_counter.hpp"

#include <cucascade/io/s3rdma/config.hpp>

#include <atomic>
#include <barrier>
#include <limits>
#include <optional>

using namespace cucascade::test::s3rdma;
using cucascade::test::scoped_allocation_counter;
using gate_type = rdma::admission_gate;

namespace {

void drain(gate_type& gate, gate_type::drain_batch batch, std::exception_ptr error)
{
  batch.error_complete_all(error);
  if (batch.has_token()) { gate.complete_drain(std::move(batch).take_token()); }
}

void close(gate_type& gate)
{
  drain(gate, gate.begin_close(), failure(rdma::errc::transport_closed));
  gate.await_closed();
}

}  // namespace

TEST_CASE("S3RDMA gate validates geometry and overflow before allocating the ring",
          "[s3rdma][gate]")
{
  rdma::config cfg;
  cfg.max_inflight    = 2;
  cfg.arena_slot_size = 1;
  CHECK(rdma::sanitized(cfg).queue_cap == 8);
  cfg.queue_cap = 3;
  CHECK(rdma::sanitized(cfg).queue_cap == 3);
  CHECK(cfg.min_alignment_requirement() == 1);
  CHECK(cfg.merge_gap_size() == 0);
  SECTION("zero workers")
  {
    cfg.max_inflight = 0;
    CHECK_THROWS_AS(rdma::sanitized(cfg), std::invalid_argument);
  }
  SECTION("zero slot")
  {
    cfg.arena_slot_size = 0;
    CHECK_THROWS_AS(rdma::sanitized(cfg), std::invalid_argument);
  }
  SECTION("zero cap")
  {
    cfg.queue_cap = 0;
    CHECK_THROWS_AS(rdma::sanitized(cfg), std::invalid_argument);
    CHECK_THROWS_AS((gate_type{0, 1}), std::invalid_argument);
    CHECK_THROWS_AS((gate_type{1, 0}), std::invalid_argument);
  }
  SECTION("derived cap overflow")
  {
    cfg.queue_cap.reset();
    cfg.max_inflight = std::numeric_limits<std::size_t>::max() / 4 + 1;
    CHECK_THROWS_AS(rdma::sanitized(cfg), std::overflow_error);
  }
  SECTION("arena overflow before ring construction")
  {
    cfg.queue_cap.reset();
    cfg.max_inflight    = std::numeric_limits<std::size_t>::max() / 4;
    cfg.arena_slot_size = 8;
    CHECK_THROWS_AS(rdma::sanitized(cfg), std::overflow_error);
  }
  SECTION("explicit cap cannot wrap ring storage")
  {
    cfg.max_inflight = 1;
    cfg.queue_cap    = std::numeric_limits<std::size_t>::max();
    auto valid       = rdma::sanitized(cfg);
    CHECK_THROWS_AS((gate_type{*valid.queue_cap, valid.arena_slot_size}), std::overflow_error);
  }
}

TEST_CASE("S3RDMA gate rejects invalid requests without taking ownership", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{2, 4};
  request_fixture fixture;
  auto request = fixture.request();
  SECTION("null request")
  {
    request.reset();
    CHECK(has_code(gate.submit(request), rdma::errc::internal_error));
    CHECK(request == nullptr);
    close(gate);
    return;
  }
  SECTION("empty request")
  {
    request = cucascade::io::grouped_io_request::create(fixture.source, {});
  }
  SECTION("zero length") { request->front().rng.size = 0; }
  SECTION("host only")
  {
    request->front().h_buffer = cucascade::io::host_buffer{fixture.destination.data()};
    request->front().d_buffer = {};
  }
  SECTION("cache fragments")
  {
    request->front().h_buffer =
      cucascade::io::host_buffer{std::vector<cucascade::io::cache::cached_chunk*>{nullptr}};
    request->front().d_buffer = {};
  }
  SECTION("both destinations")
  {
    request->front().h_buffer = cucascade::io::host_buffer{fixture.destination.data()};
  }
  SECTION("neither destination") { request->front().d_buffer = {}; }
  SECTION("invalid second slice")
  {
    request = cucascade::io::grouped_io_request::create(
      fixture.source, {fixture.slice(0, 1), cucascade::io::prepared_io_slice{}});
  }
  auto* original   = request.get();
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  bool const empty = request->empty();
  auto error       = gate.submit(request);
  CHECK(has_code(error, rdma::errc::internal_error));
  REQUIRE(request.get() == original);
  CHECK(future.is_ready() == empty);
  CHECK_FALSE(gate.terminal());
  CHECK(gate.envelope_depth_peak() == 0);
  request->cancel_remaining(error);
  CHECK(coordinator->tasks_remaining() == 0);
  if (empty) {
    CHECK(std::move(future).get(2s) == 0);
  } else {
    require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::internal_error);
  }
  close(gate);
}

TEST_CASE("S3RDMA dropping an unsettled claim publishes internal error in Release",
          "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request     = fixture.request();
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  auto chunk = gate.claim();
  REQUIRE(chunk);
  CHECK_FALSE(chunk->admitted());
  CHECK_FALSE(future.is_ready());
  chunk.reset();
  CHECK(future.is_ready());
  CHECK(coordinator->tasks_remaining() == 0);
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::internal_error);
  close(gate);
}

TEST_CASE("S3RDMA incremental credits retain the partial slice and untouched slice",
          "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request = cucascade::io::grouped_io_request::create(
    fixture.source, {fixture.slice(0, 12), fixture.slice(32, 4, 32)});
  std::array<unsigned, 2> callbacks{};
  std::array<bool, 2> successes{true, true};
  for (std::size_t i = 0; i < 2; ++i) {
    request->slices[i].on_complete =
      std::make_shared<cucascade::io::prepared_io_completion>([&, i](auto, bool success) noexcept {
        ++callbacks[i];
        successes[i] = success;
      });
  }
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  CHECK(coordinator->tasks_remaining() == 2);
  REQUIRE(gate.submit(request) == nullptr);
  auto chunk = gate.claim();
  REQUIRE(chunk);
  CHECK(coordinator->tasks_remaining() == 3);
  chunk->complete();
  CHECK(coordinator->tasks_remaining() == 2);
  CHECK_FALSE(future.is_ready());
  auto batch = gate.begin_close();
  CHECK(batch.size() == 1);
  drain(gate, std::move(batch), failure(rdma::errc::transport_closed));
  CHECK(coordinator->tasks_remaining() == 0);
  CHECK(callbacks == std::array<unsigned, 2>{1, 1});
  CHECK(successes == std::array<bool, 2>{false, false});
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::transport_closed);
  chunk.reset();
  gate.await_closed();
}

TEST_CASE("S3RDMA final chunks transfer credits and preserve every chunk field", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request = cucascade::io::grouped_io_request::create(
    fixture.source, {fixture.slice(3, 10), fixture.slice(50, 3, 40)});
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  std::array<std::size_t, 4> const offsets{3, 7, 11, 50};
  std::array<std::size_t, 4> const sizes{4, 4, 2, 3};
  std::array<std::size_t, 4> const landings{0, 4, 8, 40};
  std::array<std::size_t, 4> const credits{3, 3, 2, 1};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    auto chunk = gate.claim();
    REQUIRE(chunk);
    CHECK(chunk->chunk().object == fixture.source);
    CHECK(chunk->chunk().object_offset == offsets[i]);
    CHECK(chunk->chunk().size == sizes[i]);
    CHECK(chunk->chunk().device_dst == fixture.destination.data() + landings[i]);
    CHECK(chunk->chunk().device_id == 7);
    CHECK(chunk->chunk().stream.get() == cudaStream_t{nullptr});
    CHECK(coordinator->tasks_remaining() == credits[i]);
    CHECK_FALSE(future.is_ready());
    gate.acquire_get(*chunk);
    CHECK(chunk->admitted());
    chunk->complete();
    CHECK(coordinator->tasks_remaining() == credits[i] - 1);
    CHECK(future.is_ready() == (i == 3));
  }
  CHECK(std::move(future).get(2s) == 13);
  close(gate);
}

TEST_CASE("S3RDMA draining a partial slice cannot settle its outstanding claim", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request     = fixture.request(12);
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  auto chunk = gate.claim();
  REQUIRE(chunk);
  CHECK(coordinator->tasks_remaining() == 2);
  auto fatal = failure(rdma::errc::backend_unavailable);
  drain(gate, gate.fail_stop(fatal), fatal);
  CHECK(coordinator->tasks_remaining() == 1);
  CHECK_FALSE(future.is_ready());
  chunk->complete();
  CHECK(coordinator->tasks_remaining() == 0);
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
  chunk.reset();
  close(gate);
}

TEST_CASE("S3RDMA claim cancels failed remainders and reaches a live sentinel", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{2, 4};
  request_fixture fixture;
  auto request         = fixture.request(12);
  auto coordinator     = request->coordinator;
  auto future          = coordinator->get_future();
  auto sentinel        = fixture.request(1);
  auto sentinel_future = sentinel->coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  REQUIRE(gate.submit(sentinel) == nullptr);
  auto first = gate.claim();
  REQUIRE(first);
  first->report_error(failure(rdma::errc::backend_unavailable));
  CHECK_FALSE(future.is_ready());
  auto next = gate.claim();
  REQUIRE(next);
  CHECK(next->chunk().size == 1);
  CHECK(coordinator->tasks_remaining() == 0);
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
  next->complete();
  CHECK(std::move(sentinel_future).get(2s) == 1);
  first.reset();
  next.reset();
  close(gate);
}

TEST_CASE("S3RDMA GET admission rejects a failed request without admitting its other claim",
          "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request     = fixture.request(8);
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  auto a = gate.claim();
  auto b = gate.claim();
  REQUIRE(a);
  REQUIRE(b);
  a->report_error(failure(rdma::errc::backend_unavailable));
  require_code([&] { gate.acquire_get(*b); }, rdma::errc::request_canceled);
  CHECK_FALSE(b->admitted());
  CHECK(coordinator->tasks_remaining() == 1);
  CHECK_FALSE(future.is_ready());
  b->report_error(failure(rdma::errc::request_canceled));
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
  CHECK_FALSE(gate.terminal());
  a.reset();
  b.reset();
  close(gate);
}

TEST_CASE("S3RDMA queue cap blocks a submitter until one logical envelope leaves", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{2, 4};
  request_fixture fixture;
  auto a  = fixture.request();
  auto b  = fixture.request();
  auto c  = fixture.request();
  auto fa = a->coordinator->get_future();
  auto fb = b->coordinator->get_future();
  auto fc = c->coordinator->get_future();
  REQUIRE(gate.submit(a) == nullptr);
  REQUIRE(gate.submit(b) == nullptr);
  std::promise<void> entered;
  auto started = entered.get_future();
  workers threads;
  auto submitted = threads.launch([&] {
    entered.set_value();
    return gate.submit(c);
  });
  ready(started);
  CHECK(submitted.wait_for(50ms) == std::future_status::timeout);
  auto chunk = gate.claim();
  REQUIRE(chunk);
  CHECK(ready(submitted) == nullptr);
  CHECK(c == nullptr);
  CHECK(gate.envelope_depth_peak() == 2);
  CHECK(gate.envelope_wait_total() == 1);
  CHECK(gate.envelope_wait_ns_total() > 0);
  chunk->complete();
  chunk.reset();
  close(gate);
  CHECK(std::move(fa).get(2s) == 1);
  require_code([&] { (void)std::move(fb).get(2s); }, rdma::errc::transport_closed);
  require_code([&] { (void)std::move(fc).get(2s); }, rdma::errc::transport_closed);
}

TEST_CASE("S3RDMA terminal transitions wake refused submitters with ownership intact",
          "[s3rdma][gate]")
{
  bool const fatal_mode = GENERATE(false, true);
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto queued        = fixture.request();
  auto waiting       = fixture.request();
  auto* original     = waiting.get();
  auto coordinator   = waiting->coordinator;
  auto future        = coordinator->get_future();
  auto queued_future = queued->coordinator->get_future();
  REQUIRE(gate.submit(queued) == nullptr);
  std::promise<void> entered;
  auto started = entered.get_future();
  workers threads;
  auto submitted = threads.launch([&] {
    entered.set_value();
    return gate.submit(waiting);
  });
  ready(started);
  CHECK(submitted.wait_for(50ms) == std::future_status::timeout);
  auto code    = fatal_mode ? rdma::errc::backend_unavailable : rdma::errc::transport_closed;
  auto error   = failure(code);
  auto batch   = fatal_mode ? gate.fail_stop(error) : gate.begin_close();
  auto refused = ready(submitted);
  CHECK(has_code(refused, code));
  REQUIRE(waiting.get() == original);
  CHECK_FALSE(future.is_ready());
  waiting->cancel_remaining(refused);
  require_code([&] { (void)std::move(future).get(2s); }, code);
  CHECK(gate.envelope_depth_peak() == 1);
  CHECK(gate.envelope_wait_total() == 1);
  CHECK(gate.envelope_wait_ns_total() > 0);
  drain(gate, std::move(batch), error);
  require_code([&] { (void)std::move(queued_future).get(2s); }, code);
  close(gate);
}

TEST_CASE("S3RDMA close observes settlement before claim retirement on every path",
          "[s3rdma][gate]")
{
  auto const mode = GENERATE(0, 1, 2, 3, 4);
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request     = fixture.request();
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  auto chunk = gate.claim();
  REQUIRE(chunk);
  if (mode == 2) {
    gate.acquire_get(*chunk);
    REQUIRE(chunk->admitted());
  }
  if (mode == 3) { chunk->complete(); }
  drain(gate, gate.begin_close(), failure(rdma::errc::transport_closed));
  std::promise<void> entered;
  auto started = entered.get_future();
  workers threads;
  auto closed = threads.launch([&] {
    entered.set_value();
    gate.await_closed();
    return future.is_ready() && coordinator->tasks_remaining() == 0;
  });
  ready(started);
  CHECK(closed.wait_for(50ms) == std::future_status::timeout);
  if (mode == 0) {
    chunk->report_error(failure(rdma::errc::transport_closed));
    CHECK(future.is_ready());
    CHECK(closed.wait_for(50ms) == std::future_status::timeout);
  } else if (mode == 1) {
    require_code([&] { gate.acquire_get(*chunk); }, rdma::errc::transport_closed);
    CHECK_FALSE(chunk->admitted());
    CHECK_FALSE(future.is_ready());
    CHECK(closed.wait_for(50ms) == std::future_status::timeout);
  } else if (mode == 2) {
    require_code(
      [&] {
        auto unwinding = std::move(*chunk);
        chunk.reset();
        throw rdma::error{rdma::errc::backend_unavailable, "after admission"};
      },
      rdma::errc::backend_unavailable);
  } else if (mode == 4) {
    require_code(
      [&] {
        auto unwinding = std::move(*chunk);
        chunk.reset();
        gate.acquire_get(unwinding);
      },
      rdma::errc::transport_closed);
  } else {
    CHECK(future.is_ready());
  }
  chunk.reset();
  CHECK(ready(closed));
  if (mode == 3) {
    CHECK(std::move(future).get(2s) == 1);
  } else {
    require_code([&] { (void)std::move(future).get(2s); },
                 mode == 0 ? rdma::errc::transport_closed : rdma::errc::internal_error);
  }
}

TEST_CASE("S3RDMA control holder starts close without blocking itself or losing waiters",
          "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  std::optional<gate_type::admission_permit> permit{gate.acquire_control()};
  std::optional<gate_type::admission_permit> last_permit{gate.acquire_control()};
  auto batch = gate.begin_close();
  CHECK(batch.empty());
  CHECK_FALSE(batch.has_token());
  std::array<std::promise<void>, 2> entered;
  workers threads;
  std::vector<std::future<void>> waiters;
  for (auto& signal : entered) {
    auto started = signal.get_future();
    waiters.push_back(threads.launch([&gate, &signal] {
      signal.set_value();
      gate.await_closed();
    }));
    ready(started);
  }
  for (auto& waiter : waiters) {
    CHECK(waiter.wait_for(50ms) == std::future_status::timeout);
  }
  permit.reset();
  for (auto& waiter : waiters) {
    CHECK(waiter.wait_for(50ms) == std::future_status::timeout);
  }
  last_permit.reset();
  for (auto& waiter : waiters) {
    ready(waiter);
  }
}

TEST_CASE("S3RDMA concurrent terminal transitions grant one nonempty drain token", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{2, 4};
  request_fixture fixture;
  auto request = fixture.request();
  auto future  = request->coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  auto fatal = failure(rdma::errc::backend_unavailable);
  std::barrier rendezvous{3};
  workers threads;
  auto first  = threads.launch([&] {
    rendezvous.arrive_and_wait();
    return gate.fail_stop(fatal);
  });
  auto second = threads.launch([&] {
    rendezvous.arrive_and_wait();
    return gate.begin_close();
  });
  rendezvous.arrive_and_wait();
  auto a = ready(first);
  auto b = ready(second);
  CHECK(a.size() + b.size() == 1);
  CHECK(static_cast<int>(a.has_token()) + static_cast<int>(b.has_token()) == 1);
  CHECK(gate.first_fatal() == fatal);
  CHECK(gate.fail_stop_total() == 1);
  a.error_complete_all(fatal);
  b.error_complete_all(fatal);
  CHECK(future.is_ready());
  std::promise<void> entered;
  auto started = entered.get_future();
  auto closed  = threads.launch([&] {
    entered.set_value();
    gate.await_closed();
  });
  ready(started);
  CHECK(closed.wait_for(50ms) == std::future_status::timeout);
  if (a.has_token()) { gate.complete_drain(std::move(a).take_token()); }
  if (b.has_token()) { gate.complete_drain(std::move(b).take_token()); }
  ready(closed);
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
}

TEST_CASE("S3RDMA first fatal after close stays visible on both admission planes", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  request_fixture fixture;
  auto request = fixture.request();
  auto future  = request->coordinator->get_future();
  REQUIRE(gate.submit(request) == nullptr);
  auto chunk = gate.claim();
  REQUIRE(chunk);
  auto closed_error = failure(rdma::errc::transport_closed);
  drain(gate, gate.begin_close(), closed_error);
  require_code([&] { (void)gate.acquire_control(); }, rdma::errc::transport_closed);
  auto fatal = failure(rdma::errc::backend_unavailable);
  drain(gate, gate.fail_stop(fatal), fatal);
  auto later = failure(rdma::errc::internal_error);
  drain(gate, gate.fail_stop(later), later);
  CHECK(gate.first_fatal() == fatal);
  CHECK(gate.fail_stop_total() == 1);
  require_code([&] { (void)gate.acquire_control(); }, rdma::errc::backend_unavailable);
  require_code([&] { (void)gate.enter_creation(); }, rdma::errc::backend_unavailable);
  require_code([&] { gate.acquire_get(*chunk); }, rdma::errc::backend_unavailable);
  CHECK_FALSE(chunk->admitted());
  auto refused        = fixture.request();
  auto refused_future = refused->coordinator->get_future();
  CHECK(gate.submit(refused) == fatal);
  REQUIRE(refused);
  CHECK_FALSE(refused_future.is_ready());
  refused->cancel_remaining(fatal);
  require_code([&] { (void)std::move(refused_future).get(2s); }, rdma::errc::backend_unavailable);
  chunk->report_error(fatal);
  chunk.reset();
  gate.await_closed();
  CHECK_FALSE(gate.claim());
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
}

TEST_CASE("S3RDMA creation scope serializes terminal publication", "[s3rdma][gate]")
{
  deadline bound;
  gate_type gate{1, 4};
  std::optional<gate_type::creation_scope> creation{gate.enter_creation()};
  std::promise<void> entered;
  auto started = entered.get_future();
  workers threads;
  auto closing = threads.launch([&] {
    entered.set_value();
    return gate.begin_close();
  });
  ready(started);
  CHECK(closing.wait_for(50ms) == std::future_status::timeout);
  creation.reset();
  drain(gate, ready(closing), failure(rdma::errc::transport_closed));
  require_code([&] { (void)gate.enter_creation(); }, rdma::errc::transport_closed);
  gate.await_closed();
}

TEST_CASE("S3RDMA claim moves dispose old ownership and leave moved-from guards inert",
          "[s3rdma][gate]")
{
  deadline bound;
  static_assert(!std::is_copy_constructible_v<gate_type::claimed_chunk>);
  static_assert(std::is_nothrow_move_constructible_v<gate_type::claimed_chunk>);
  static_assert(std::is_nothrow_move_assignable_v<gate_type::claimed_chunk>);
  gate_type gate{2, 4};
  request_fixture fixture;
  auto a  = fixture.request();
  auto b  = fixture.request();
  auto fa = a->coordinator->get_future();
  auto fb = b->coordinator->get_future();
  REQUIRE(gate.submit(a) == nullptr);
  REQUIRE(gate.submit(b) == nullptr);
  auto first  = gate.claim();
  auto second = gate.claim();
  REQUIRE(first);
  REQUIRE(second);
  *first = std::move(*second);
  CHECK(fa.is_ready());
  CHECK_FALSE(fb.is_ready());
  second.reset();
  CHECK_FALSE(fb.is_ready());
  auto last = std::move(*first);
  first.reset();
  CHECK_FALSE(fb.is_ready());
  last.complete();
  require_code([&] { (void)std::move(fa).get(2s); }, rdma::errc::internal_error);
  CHECK(std::move(fb).get(2s) == 1);
  // last still owns the count; its destruction precedes the gate's destruction.
}

TEST_CASE("S3RDMA gate operations allocate nothing after construction", "[s3rdma][gate]")
{
  deadline bound;
  int* volatile escaped = nullptr;
  std::size_t positive  = 0;
  {
    scoped_allocation_counter counter;
    escaped = new int{7};
    delete escaped;
    positive = counter.count();
  }
  REQUIRE(positive > 0);
  gate_type gate{1, 4};
  gate_type failing{1, 4};
  gate_type closing{1, 4};
  request_fixture fixture;
  auto a                  = fixture.request(8);
  auto b                  = fixture.request();
  auto c                  = fixture.request();
  auto fa                 = a->coordinator->get_future();
  auto fb                 = b->coordinator->get_future();
  auto fc                 = c->coordinator->get_future();
  auto fatal              = failure(rdma::errc::backend_unavailable);
  auto closed             = failure(rdma::errc::transport_closed);
  bool ok                 = true;
  std::size_t allocations = 0;
  {
    scoped_allocation_counter counter;
    ok = gate.submit(a) == nullptr && failing.submit(b) == nullptr && closing.submit(c) == nullptr;
    auto chunk = gate.claim();
    if (chunk) {
      gate.acquire_get(*chunk);
      {
        auto permit = gate.acquire_control();
      }
      chunk->complete();
      chunk.reset();
      chunk = gate.claim();
      if (chunk) {
        gate.acquire_get(*chunk);
        chunk->complete();
        chunk.reset();
      } else {
        ok = false;
      }
    } else {
      ok = false;
    }
    auto failed = failing.fail_stop(fatal);
    ok          = ok && failed.has_token();
    drain(failing, std::move(failed), fatal);
    auto batch = closing.begin_close();
    ok         = ok && batch.has_token();
    drain(closing, std::move(batch), closed);
    auto again  = failing.begin_close();
    ok          = ok && !again.has_token();
    auto empty  = gate.begin_close();
    ok          = ok && empty.empty() && !empty.has_token();
    allocations = counter.count();
  }
  CHECK(ok);
  CHECK(allocations == 0);
  gate.await_closed();
  failing.await_closed();
  closing.await_closed();
  CHECK(std::move(fa).get(2s) == 8);
  require_code([&] { (void)std::move(fb).get(2s); }, rdma::errc::backend_unavailable);
  require_code([&] { (void)std::move(fc).get(2s); }, rdma::errc::transport_closed);
}
