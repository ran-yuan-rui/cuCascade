/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

using namespace cucascade::test::s3rdma;
namespace io = cucascade::io;

namespace {

action failed(cudaError_t error)
{
  action step;
  step.result = error;
  return step;
}

action held(rendezvous& point, cudaError_t result = cudaSuccess)
{
  auto step   = failed(result);
  step.before = [&](auto const&) { point.arrive_and_wait(); };
  return step;
}

void delivered(pending_read& pending, std::size_t bytes, unsigned callbacks = 1)
{
  REQUIRE(wait_until([&] { return pending.future.is_ready(); }));
  CHECK(std::move(pending.future).get(2s) == bytes);
  CHECK(pending.coordinator->tasks_remaining() == 0);
  CHECK(pending.completion->calls.load() == callbacks);
  CHECK(pending.completion->success.load());
  CHECK(pending.completion->fragments.load() == 0);
}

void no_calls_after_sticky(journal const& log)
{
  auto entries = log.snapshot();
  auto first   = std::find_if(entries.begin(), entries.end(), [](auto const& e) {
    return e.returned && rdma::is_context_fatal(e.result);
  });
  REQUIRE(first != entries.end());
  for (auto it = first + 1; it != entries.end(); ++it) {
    if (it->returned) { continue; }
    CAPTURE(static_cast<int>(it->op), it->returned);
    CHECK((it->op == operation::policy || it->op == operation::worker_exit ||
           it->op == operation::completion));
  }
  CHECK_FALSE(log.lost_record());
}

std::vector<observation> calls(journal const& log, operation op, bool returned = false)
{
  auto result = log.snapshot();
  std::erase_if(result, [&](auto const& e) { return e.op != op || e.returned != returned; });
  return result;
}

}  // namespace

TEST_CASE("S3RDMA device chunks deliver exact bytes in protocol order", "[s3rdma][runtime]")
{
  auto const length = GENERATE(std::size_t{1}, std::size_t{4}, std::size_t{6}, std::size_t{12});
  runtime_fixture f;
  f.start();
  auto read = f.submit(3, length, 5);
  delivered(read, length);
  f.shutdown();
  auto expected = std::array<std::uint8_t, 128>{};
  expected.fill(0xcc);
  std::copy_n(f.source.data() + 3, length, expected.data() + 5);
  CHECK(f.destination == expected);
  auto gets = calls(f.controls->log, operation::get);
  CHECK(gets.size() == (length + 3) / 4);
  std::size_t offset = 3;
  for (auto const& get : gets) {
    CHECK(get.object_offset == offset);
    CHECK(get.size <= 4);
    offset += get.size;
  }
  std::vector<operation> actual;
  for (auto const& e : f.controls->log.snapshot()) {
    if (!e.returned &&
        (e.op == operation::get || e.op == operation::flush || e.op == operation::copy ||
         e.op == operation::record || e.op == operation::wait)) {
      actual.push_back(e.op);
    }
  }
  std::vector<operation> expected_order;
  for (std::size_t i = 0; i < gets.size(); ++i) {
    expected_order.insert(
      expected_order.end(),
      {operation::get, operation::flush, operation::copy, operation::record, operation::wait});
  }
  CHECK(actual == expected_order);
  CHECK(f.reactor->perf().bytes_total == length);
  CHECK(f.reactor->perf().retries_total == 0);
}

TEST_CASE("S3RDMA several slices and concurrent envelopes settle independently",
          "[s3rdma][runtime]")
{
  rendezvous first_wait;
  runtime_fixture f{runtime_config(2)};
  f.controls->script(operation::wait, held(first_wait));
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(first_wait.entered());
  auto second =
    f.submit({f.slice(8, 3, f.destination.data() + 8), f.slice(20, 5, f.destination.data() + 20)});
  delivered(second, 8, 2);
  CHECK_FALSE(first.future.is_ready());
  first_wait.release();
  delivered(first, 4);
  f.shutdown();
  CHECK(std::equal(f.source.begin() + 8, f.source.begin() + 11, f.destination.begin() + 8));
  CHECK(std::equal(f.source.begin() + 20, f.source.begin() + 25, f.destination.begin() + 20));
  CHECK(f.reactor->perf().requests_total == 2);
  CHECK_FALSE(first_wait.timed_out());
}

TEST_CASE("S3RDMA a ready worker reuses its slot while another event is held", "[s3rdma][runtime]")
{
  rendezvous first_wait;
  runtime_fixture f{runtime_config(2)};
  f.controls->script(operation::wait, held(first_wait));
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(first_wait.entered());
  auto second = f.submit(8, 4, 8);
  delivered(second, 4);
  auto third = f.submit(16, 4, 16);
  delivered(third, 4);
  auto before_release = calls(f.controls->log, operation::get);
  REQUIRE(before_release.size() == 3);
  CHECK(before_release[0].address == before_release[1].address);
  CHECK(before_release[0].offset != before_release[1].offset);
  CHECK(before_release[1].offset == before_release[2].offset);
  CHECK(before_release[1].thread == before_release[2].thread);
  CHECK_FALSE(first.future.is_ready());
  first_wait.release();
  delivered(first, 4);
  f.shutdown();
  CHECK(f.reactor->perf().inflight_peak <= 2);
  CHECK(f.reactor->perf().slots_in_use_peak == 2);
  CHECK_FALSE(first_wait.timed_out());
}

TEST_CASE("S3RDMA every nonexact GET closes and settles all admission paths", "[s3rdma][runtime]")
{
  auto const leg = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  CAPTURE(leg);
  rendezvous inside_get;
  runtime_fixture f{runtime_config()};
  f.cfg.queue_cap     = 1;
  auto step           = held(inside_get);
  step.change_receipt = [leg](rdma::data_get_result& result) {
    switch (leg) {
      case 0: result.reply_tag.clear(); break;
      case 1: result.reply_tag = "rejected"; break;
      case 2: result.http_status = 500; break;
      case 3: --result.delivered_bytes; break;
      case 4: result.sdk_return = -17; break;
      case 5: --result.sdk_return; break;
      case 6: result.transport_error = "completion uncertain"; break;
      case 7: result.callback_invocations = 0; break;
      case 8: result.callback_invocations = 0; break;
      case 9: result.callback_invocations = 2; break;
      case 10:
        result.commit               = rdma::data_commit_state::not_sent;
        result.callback_invocations = 0;
        break;
      case 11: result.commit = rdma::data_commit_state::sent_unknown; break;
      default: break;
    }
  };
  if (leg == 12) { step.behavior = get_behavior::throw_error; }
  f.controls->script(operation::get, std::move(step));
  f.start();
  auto const bytes = leg == 8 ? std::size_t{1} : std::size_t{4};
  auto first       = f.submit(0, bytes);
  REQUIRE(inside_get.entered());
  auto queued = f.submit(8, bytes, 8);
  workers threads;
  auto submitting = threads.launch([&] { return f.submit(16, bytes, 16); });
  REQUIRE(wait_until([&] { return f.context->gate().envelope_wait_total() > 0; }));
  CHECK(submitting.wait_for(0ms) != std::future_status::ready);
  inside_get.release();
  auto blocked = ready(submitting);
  auto fatal   = error_from(first.future);
  REQUIRE(fatal != nullptr);
  CHECK(error_from(queued.future) == fatal);
  CHECK(error_from(blocked.future) == fatal);
  CHECK(f.context->gate().terminal_error() == fatal);
  for (auto* read : {&first, &queued, &blocked}) {
    CHECK(read->coordinator->tasks_remaining() == 0);
    CHECK(read->completion->calls.load() == 1);
    CHECK_FALSE(read->completion->success.load());
  }
  auto later = f.submit(24, bytes, 24);
  CHECK(error_from(later.future) == fatal);
  std::exception_ptr host_error;
  try {
    (void)f.reactor->host_read(*f.object, 0, 1, f.destination.data());
  } catch (...) {
    host_error = std::current_exception();
  }
  CHECK(host_error == fatal);
  f.shutdown();
  CHECK(f.reactor->perf().fail_stop_total == 1);
  CHECK(f.reactor->perf().requests_total == 2);
  CHECK(f.reactor->perf().error_total > 0);
  CHECK(f.reactor->perf().arena_leak_total > 0);
  if (leg == 3) { CHECK(f.reactor->perf().short_read_total == 1); }
  CHECK(f.reactor->perf().retries_total == 0);
  CHECK(f.controls->log.count(operation::get) == 1);
  CHECK(f.controls->log.count(operation::range) == 0);
  CHECK(f.controls->log.count(operation::copy) == 0);
  CHECK(f.controls->log.count(operation::deregister_memory) == 0);
  CHECK(f.controls->log.count(operation::free) == 0);
  CHECK(f.reactor->status()->terminal());
  CHECK(f.reactor->status()->arena_retained());
  CHECK(f.reactor->status()->torn_down());
  CHECK(std::all_of(
    f.destination.begin(), f.destination.end(), [](auto byte) { return byte == 0xcc; }));
  CHECK_FALSE(inside_get.timed_out());
}

TEST_CASE("S3RDMA an absent callback cannot reuse a preceding successful receipt",
          "[s3rdma][runtime]")
{
  runtime_fixture f;
  f.start();
  auto first = f.submit(0, 4);
  delivered(first, 4);
  action missing;
  missing.behavior = get_behavior::no_callback;
  f.controls->script(operation::get, missing);
  auto second = f.submit(8, 4, 8);
  REQUIRE(error_from(second.future) != nullptr);
  f.shutdown();
  CHECK(f.controls->log.count(operation::get) == 2);
  CHECK(f.controls->log.count(operation::copy) == 1);
  CHECK(f.destination[8] == 0xcc);
  CHECK(f.reactor->perf().fail_stop_total == 1);
  CHECK(f.reactor->perf().bytes_total == 4);
  CHECK(f.reactor->perf().retries_total == 0);
}

TEST_CASE("S3RDMA an injected reply rule cannot relax independent completion vetoes",
          "[s3rdma][runtime]")
{
  auto const bad_status = GENERATE(false, true);
  runtime_fixture f;
  auto clients          = f.controls->clients();
  clients.rule.reply_ok = +[](std::string_view, long) noexcept { return true; };
  action bad;
  bad.change_receipt = [bad_status](auto& result) {
    if (bad_status) {
      result.http_status = 500;
    } else {
      result.callback_invocations = 0;
    }
  };
  f.controls->script(operation::get, bad);
  f.context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(
    f.cfg, clients, f.controls->delivery(), f.controls->policy());
  f.reactor = std::make_unique<rdma::s3rdma_reactor>(f.context);
  f.start();
  auto read = f.submit(0, 4);
  REQUIRE(error_from(read.future) != nullptr);
  f.shutdown();
  CHECK(f.reactor->perf().fail_stop_total == 1);
  CHECK(f.controls->log.count(operation::copy) == 0);
}

TEST_CASE("S3RDMA a failed sibling refuses GET admission without poisoning the runtime",
          "[s3rdma][runtime]")
{
  rendezvous before_permit;
  runtime_fixture f{runtime_config(2)};
  f.controls->script(operation::capture, held(before_permit));
  f.controls->script(operation::capture, failed(cudaErrorInvalidValue));
  f.start();
  auto read = f.submit(0, 8);
  REQUIRE(before_permit.entered());
  REQUIRE(wait_until([&] { return !read.coordinator->should_continue(); }));
  before_permit.release();
  REQUIRE(error_from(read.future) != nullptr);
  CHECK(read.coordinator->tasks_remaining() == 0);
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK_FALSE(f.context->gate().terminal());
  auto next = f.submit(16, 4, 16);
  delivered(next, 4);
  f.shutdown();
  CHECK(f.reactor->perf().fail_stop_total == 0);
  CHECK(f.reactor->perf().retries_total == 0);
  CHECK_FALSE(before_permit.timed_out());
}

TEST_CASE("S3RDMA recoverable delivery preparation errors leave the worker usable",
          "[s3rdma][runtime]")
{
  auto const leg = GENERATE(0, 1, 2);
  runtime_fixture f;
  if (leg == 0) { f.controls->script(operation::flush, failed(cudaErrorInvalidValue)); }
  if (leg == 1) {
    action step;
    step.throws = true;
    f.controls->script(operation::create_event, step);
  }
  if (leg == 2) { f.controls->capture_status = cudaStreamCaptureStatusActive; }
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(error_from(first.future) != nullptr);
  CHECK(f.controls->log.count(operation::get) == (leg == 0 ? 1 : 0));
  CHECK(f.controls->log.count(operation::copy) == 0);
  CHECK_FALSE(f.context->gate().terminal());
  f.controls->capture_status = cudaStreamCaptureStatusNone;
  auto next                  = f.submit(8, 4, 8);
  delivered(next, 4);
  f.shutdown();
  CHECK(f.reactor->perf().fail_stop_total == 0);
  CHECK(f.controls->policies.load() == 0);
}

TEST_CASE("S3RDMA sticky preparation errors retain resources and prohibit further calls",
          "[s3rdma][runtime]")
{
  auto const point        = GENERATE(operation::create_event, operation::flush, operation::capture);
  auto const empty_policy = GENERATE(false, true);
  std::atomic<bool> observed_terminal{false};
  runtime_fixture f;
  f.controls->policy_observer = [&](auto const&) {
    observed_terminal.store(f.context->status()->terminal());
  };
  f.construct(empty_policy ? rdma::context_health_policy{} : f.controls->policy());
  f.start();
  f.controls->script(point, failed(cudaErrorIllegalAddress));
  auto read = f.submit(0, 4);
  REQUIRE(error_from(read.future) != nullptr);
  f.shutdown();
  CHECK(f.reactor->status()->terminal());
  CHECK(f.reactor->status()->arena_retained());
  CHECK(f.reactor->status()->torn_down());
  CHECK(f.controls->policies.load() == (empty_policy ? 0 : 1));
  if (!empty_policy) { CHECK(observed_terminal.load()); }
  f.reactor.reset();
  f.context.reset();
  no_calls_after_sticky(f.controls->log);
}

TEST_CASE("S3RDMA no-touch precedes a held drain callback and another worker exit",
          "[s3rdma][runtime]")
{
  rendezvous failing_create;
  rendezvous other_capture;
  rendezvous draining;
  runtime_fixture f{runtime_config(2)};
  f.controls->script(operation::create_event, held(failing_create, cudaErrorIllegalAddress));
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(failing_create.entered());
  f.controls->script(operation::capture, held(other_capture));
  auto second = f.submit(8, 4, 8);
  REQUIRE(other_capture.entered());
  auto completion   = std::make_shared<completion_record>();
  completion->after = [&] { draining.arrive_and_wait(); };
  auto queued       = f.submit({f.slice(16, 4, f.destination.data() + 16)}, completion);
  failing_create.release();
  REQUIRE(draining.entered());
  other_capture.release();
  REQUIRE(wait_until([&] { return f.controls->log.count(operation::worker_exit) >= 1; }));
  CHECK(f.controls->policies.load() == 0);
  CHECK(f.context->status()->terminal());
  CHECK(f.controls->log.count(operation::release_session) == 0);
  // The capture was entered before no-touch; its return is not a subsequent call.
  auto entries = f.controls->log.snapshot();
  auto fatal   = std::find_if(entries.begin(), entries.end(), [](auto const& e) {
    return e.returned && e.result == cudaErrorIllegalAddress;
  });
  REQUIRE(fatal != entries.end());
  for (auto it = fatal + 1; it != entries.end(); ++it) {
    if (!it->returned) {
      CHECK((it->op == operation::worker_exit || it->op == operation::completion));
    }
  }
  draining.release();
  REQUIRE(error_from(first.future) != nullptr);
  REQUIRE(error_from(second.future) != nullptr);
  REQUIRE(error_from(queued.future) != nullptr);
  f.shutdown();
  CHECK(f.controls->policies.load() == 1);
  CHECK(completion->calls.load() == 1);
  f.reactor.reset();
  f.context.reset();
  auto after        = f.controls->log.snapshot();
  auto first_sticky = std::find_if(after.begin(), after.end(), [](auto const& e) {
    return e.returned && e.result == cudaErrorIllegalAddress;
  });
  REQUIRE(first_sticky != after.end());
  for (auto it = first_sticky + 1; it != after.end(); ++it) {
    if (!it->returned) {
      CHECK((it->op == operation::policy || it->op == operation::worker_exit ||
             it->op == operation::completion));
    }
  }
  CHECK_FALSE(failing_create.timed_out());
  CHECK_FALSE(other_capture.timed_out());
  CHECK_FALSE(draining.timed_out());
}

TEST_CASE("S3RDMA a sticky report after an RDMA fail-stop still establishes no-touch",
          "[s3rdma][runtime]")
{
  rendezvous uncertain_get;
  rendezvous sticky_create;
  runtime_fixture f{runtime_config(2)};
  auto step     = held(uncertain_get);
  step.behavior = get_behavior::sent_unknown;
  f.controls->script(operation::get, step);
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(uncertain_get.entered());
  f.controls->script(operation::create_event, held(sticky_create, cudaErrorIllegalAddress));
  auto second = f.submit(8, 4, 8);
  REQUIRE(sticky_create.entered());
  uncertain_get.release();
  REQUIRE(error_from(first.future) != nullptr);
  CHECK(f.controls->policies.load() == 0);
  sticky_create.release();
  REQUIRE(error_from(second.future) != nullptr);
  f.shutdown();
  CHECK(f.controls->policies.load() == 1);
  CHECK(f.controls->policy_code.load() == cudaErrorIllegalAddress);
  CHECK(f.reactor->perf().fail_stop_total == 1);
  f.reactor.reset();
  f.context.reset();
  no_calls_after_sticky(f.controls->log);
}

TEST_CASE("S3RDMA a transfer past the copy boundary finishes after another worker poisons",
          "[s3rdma][runtime]")
{
  rendezvous copying;
  runtime_fixture f{runtime_config(2)};
  f.controls->script(operation::copy, held(copying));
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(copying.entered());
  f.controls->script(operation::create_event, failed(cudaErrorIllegalAddress));
  auto second = f.submit(8, 4, 8);
  REQUIRE(error_from(second.future) != nullptr);
  REQUIRE(wait_until([&] { return f.controls->policies.load() == 1; }));
  CHECK_FALSE(first.future.is_ready());
  copying.release();
  delivered(first, 4);
  f.shutdown();
  CHECK(std::equal(f.source.begin(), f.source.begin() + 4, f.destination.begin()));
  CHECK(f.reactor->status()->arena_retained());
  f.reactor.reset();
  f.context.reset();
  auto entries = f.controls->log.snapshot();
  auto fault   = std::find_if(entries.begin(), entries.end(), [](auto const& e) {
    return e.returned && e.result == cudaErrorIllegalAddress;
  });
  REQUIRE(fault != entries.end());
  std::vector<operation> completion;
  for (auto it = fault + 1; it != entries.end(); ++it) {
    if (it->returned) { continue; }
    CHECK((it->op == operation::record || it->op == operation::wait ||
           it->op == operation::policy || it->op == operation::worker_exit));
    if (it->op == operation::record || it->op == operation::wait) { completion.push_back(it->op); }
  }
  CHECK(completion == std::vector<operation>{operation::record, operation::wait});
  CHECK_FALSE(copying.timed_out());
}

TEST_CASE("S3RDMA write ordering selects flush only for exact completions", "[s3rdma][runtime]")
{
  auto const ordering = GENERATE(
    rdma::write_ordering::owner, rdma::write_ordering::all_devices, rdma::write_ordering::none);
  runtime_fixture f;
  f.controls->write_probe = {ordering, ordering == rdma::write_ordering::none};
  f.start();
  CHECK(f.controls->log.count(operation::probe) == 1);
  CHECK(f.controls->log.count(operation::get) == 0);
  auto read = f.submit(0, 6);
  delivered(read, 6);
  auto expected = ordering == rdma::write_ordering::none ? 2U : 0U;
  CHECK(f.controls->log.count(operation::flush) == expected);
  action bad;
  bad.change_receipt = [](auto& result) { --result.delivered_bytes; };
  f.controls->script(operation::get, bad);
  auto rejected = f.submit(8, 4, 8);
  REQUIRE(error_from(rejected.future) != nullptr);
  f.shutdown();
  CHECK(f.controls->log.count(operation::flush) == expected);
  CHECK(f.reactor->perf().flush_total == expected);
}

TEST_CASE("S3RDMA unsupported write probes fail before any GET and clean up", "[s3rdma][runtime]")
{
  auto const leg = GENERATE(0, 1, 2);
  runtime_fixture f;
  if (leg == 0) { f.controls->write_probe = {rdma::write_ordering::none, false}; }
  if (leg == 1) { f.controls->write_probe = {rdma::write_ordering::unknown, true}; }
  if (leg == 2) { f.controls->script(operation::probe, failed(cudaErrorInvalidValue)); }
  CHECK_THROWS(f.start());
  CHECK(f.reactor->status()->torn_down());
  CHECK_FALSE(f.reactor->status()->terminal());
  CHECK_FALSE(f.reactor->status()->arena_retained());
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(f.controls->policies.load() == 0);
}

TEST_CASE("S3RDMA arenas use one registration and every bounded slot per device",
          "[s3rdma][runtime]")
{
  std::array<rendezvous, 2> waits;
  runtime_fixture f{runtime_config(2, 4, {7, 9})};
  f.start();
  auto registrations = calls(f.controls->log, operation::register_memory);
  REQUIRE(registrations.size() == 2);
  CHECK(registrations[0].size == 8);
  CHECK(registrations[1].size == 8);
  CHECK(registrations[0].address != registrations[1].address);
  CHECK(std::set<int>{registrations[0].device, registrations[1].device} == std::set<int>{7, 9});
  std::size_t next_wait = 0;
  for (int device : {7, 9}) {
    auto& wait = waits[next_wait++];
    f.controls->script(operation::wait, held(wait));
    auto first = f.submit({f.slice(0, 4, f.destination.data(), device)});
    REQUIRE(wait.entered());
    auto second = f.submit({f.slice(4, 4, f.destination.data() + 4, device)});
    delivered(second, 4);
    wait.release();
    delivered(first, 4);
  }
  f.shutdown();
  auto gets = calls(f.controls->log, operation::get);
  REQUIRE(gets.size() == 4);
  std::map<void*, std::set<std::size_t>> offsets;
  std::size_t get_index = 0;
  for (auto const& get : gets) {
    CHECK(get.device == (get_index++ < 2 ? 7 : 9));
    CHECK(get.offset + get.size <= 8);
    CHECK((get.offset == 0 || get.offset == 4));
    CHECK(std::any_of(registrations.begin(), registrations.end(), [&](auto const& r) {
      return r.address == get.address && r.device == get.device;
    }));
    offsets[get.address].insert(get.offset);
  }
  REQUIRE(offsets.size() == 2);
  for (auto const& [base, values] : offsets) {
    (void)base;
    CHECK(values == std::set<std::size_t>{0, 4});
  }
  CHECK(f.controls->log.count(operation::deregister_memory) == 2);
  CHECK(f.controls->log.count(operation::free) == 2);
}

TEST_CASE("S3RDMA unarmed enqueue refuses without touching a client or caller CUDA",
          "[s3rdma][runtime]")
{
  runtime_fixture f;
  f.start();
  auto before = f.controls->log.snapshot();
  auto read   = f.submit({f.slice(0, 4, f.destination.data(), 8)});
  CHECK(has_code(error_from(read.future), rdma::errc::unsupported_request));
  auto after = f.controls->log.snapshot();
  for (auto it = after.begin() + static_cast<std::ptrdiff_t>(before.size()); it != after.end();
       ++it) {
    CHECK(it->thread != f.controls->caller);
  }
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(read.completion->calls.load() == 1);
  CHECK_FALSE(read.completion->success.load());
  auto valid = f.submit(8, 4, 8);
  delivered(valid, 4);
  f.shutdown();
  CHECK(f.reactor->perf().requests_total == 1);
}

TEST_CASE("S3RDMA shutdown waits for the final worker use before releasing the arena",
          "[s3rdma][runtime]")
{
  auto const point = GENERATE(operation::get, operation::wait);
  rendezvous active;
  runtime_fixture f;
  f.controls->script(point, held(active));
  f.start();
  auto read = f.submit(0, 4);
  REQUIRE(active.entered());
  workers threads;
  auto one = threads.launch([&] { f.shutdown(); });
  REQUIRE(wait_until([&] { return f.context->gate().terminal(); }));
  auto two = threads.launch([&] { f.shutdown(); });
  CHECK_FALSE(f.reactor->status()->torn_down());
  CHECK(f.controls->log.count(operation::deregister_memory) == 0);
  CHECK(f.controls->log.count(operation::free) == 0);
  active.release();
  ready(one);
  ready(two);
  REQUIRE(wait_until([&] { return read.future.is_ready(); }));
  CHECK(read.coordinator->tasks_remaining() == 0);
  auto entries = f.controls->log.snapshot();
  auto release = std::find_if(entries.begin(), entries.end(), [](auto const& e) {
    return e.op == operation::deregister_memory;
  });
  REQUIRE(release != entries.end());
  for (auto it = release; it != entries.end(); ++it) {
    CHECK((it->op != operation::get && it->op != operation::copy && it->op != operation::wait &&
           it->op != operation::worker_exit));
  }
  CHECK(f.controls->log.count(operation::deregister_memory) == 1);
  CHECK(f.controls->log.count(operation::free) == 1);
  CHECK(f.reactor->status()->torn_down());
  CHECK_FALSE(f.reactor->status()->arena_retained());
  f.shutdown();
  CHECK(f.controls->log.count(operation::free) == 1);
}

TEST_CASE("S3RDMA failed release retains residency and publishes teardown last",
          "[s3rdma][runtime]")
{
  auto const point = GENERATE(operation::deregister_memory, operation::free);
  rendezvous release;
  runtime_fixture f;
  f.start();
  if (point == operation::deregister_memory) { f.controls->mock->script_deregister(false); }
  f.controls->script(point, held(release, cudaErrorInvalidValue));
  workers threads;
  auto closing = threads.launch([&] { f.shutdown(); });
  REQUIRE(release.entered());
  CHECK_FALSE(f.reactor->status()->torn_down());
  release.release();
  ready(closing);
  CHECK(f.reactor->status()->torn_down());
  CHECK(f.reactor->status()->terminal());
  CHECK(f.reactor->status()->arena_retained());
  CHECK(f.controls->log.count(operation::free) == (point == operation::free ? 1 : 0));
  auto count = f.controls->log.snapshot().size();
  f.shutdown();
  CHECK(f.controls->log.snapshot().size() == count);
}

TEST_CASE("S3RDMA a sticky free latches no-touch before registrar destruction", "[s3rdma][runtime]")
{
  runtime_fixture f;
  f.start();
  f.controls->script(operation::free, failed(cudaErrorIllegalAddress));
  f.shutdown();
  CHECK(f.reactor->status()->terminal());
  CHECK(f.reactor->status()->arena_retained());
  CHECK(f.reactor->status()->torn_down());
  CHECK(f.controls->policies.load() == 1);
  f.reactor.reset();
  f.context.reset();
  no_calls_after_sticky(f.controls->log);
}

TEST_CASE("S3RDMA close between claim and GET issues no request", "[s3rdma][runtime]")
{
  rendezvous claimed;
  runtime_fixture f;
  f.controls->script(operation::capture, held(claimed));
  f.start();
  auto read = f.submit(0, 4);
  REQUIRE(claimed.entered());
  finish_close(f.context->gate(), f.context->gate().begin_close());
  claimed.release();
  CHECK(has_code(error_from(read.future), rdma::errc::transport_closed));
  f.shutdown();
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(read.coordinator->tasks_remaining() == 0);
}

TEST_CASE("S3RDMA a permit holder can mark close without waiting for itself", "[s3rdma][runtime]")
{
  runtime_fixture f;
  f.start();
  action close;
  close.before = [&](auto const&) {
    finish_close(f.context->gate(), f.context->gate().begin_close());
  };
  f.controls->script(operation::get, close);
  auto read = f.submit(0, 4);
  REQUIRE(wait_until([&] { return read.future.is_ready(); }));
  f.shutdown();
  CHECK(read.coordinator->tasks_remaining() == 0);
  CHECK(f.controls->log.count(operation::get) == 1);
  CHECK(f.reactor->status()->torn_down());
}

TEST_CASE("S3RDMA workers make progress with a full ring and blocked submitter",
          "[s3rdma][runtime]")
{
  rendezvous active;
  runtime_fixture f;
  f.cfg.queue_cap = 1;
  f.controls->script(operation::wait, held(active));
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(active.entered());
  auto queued = f.submit(8, 4, 8);
  workers threads;
  auto submitting = threads.launch([&] { return f.submit(16, 4, 16); });
  REQUIRE(wait_until([&] { return f.context->gate().envelope_wait_total() > 0; }));
  active.release();
  auto last = ready(submitting);
  delivered(first, 4);
  delivered(queued, 4);
  delivered(last, 4);
  f.shutdown();
  auto perf = f.reactor->perf();
  CHECK(perf.requests_total == 3);
  CHECK(perf.envelope_depth_peak == 1);
  CHECK(perf.envelope_wait_total == 1);
  CHECK(perf.envelope_wait_ns_total > 0);
  CHECK(perf.slot_wait_total == 0);
  CHECK(perf.retries_total == 0);
}

TEST_CASE("S3RDMA normal close wakes a blocked submitter and returns one drain token",
          "[s3rdma][runtime]")
{
  rendezvous active;
  runtime_fixture f;
  f.cfg.queue_cap = 1;
  f.controls->script(operation::get, held(active));
  f.start();
  auto first = f.submit(0, 4);
  REQUIRE(active.entered());
  auto queued = f.submit(8, 4, 8);
  workers threads;
  auto submitting = threads.launch([&] { return f.submit(16, 4, 16); });
  REQUIRE(wait_until([&] { return f.context->gate().envelope_wait_total() > 0; }));
  auto& gate = f.context->gate();
  auto batch = gate.begin_close();
  CHECK(batch.has_token());
  auto second = gate.begin_close();
  CHECK_FALSE(second.has_token());
  finish_close(gate, std::move(second));
  auto fatal_error = failure(rdma::errc::internal_error);
  auto fatal       = gate.fail_stop(fatal_error);
  CHECK_FALSE(fatal.has_token());
  finish_close(gate, std::move(fatal));
  finish_close(gate, std::move(batch));
  auto blocked = ready(submitting);
  REQUIRE(error_from(blocked.future) != nullptr);
  CHECK(gate.terminal_error() == fatal_error);
  CHECK(error_from(queued.future) == fatal_error);
  active.release();
  REQUIRE(error_from(first.future) != nullptr);
  f.shutdown();
  CHECK(f.reactor->perf().fail_stop_total == 1);
  CHECK(first.coordinator->tasks_remaining() == 0);
}

TEST_CASE("S3RDMA an early final-chunk callback is not the slice completion barrier",
          "[s3rdma][runtime]")
{
  rendezvous earlier_get;
  runtime_fixture f{runtime_config(2)};
  auto bad           = held(earlier_get);
  bad.get_offset     = 0;
  bad.change_receipt = [](auto& result) { result.callback_invocations = 0; };
  f.controls->script(operation::get, bad);
  f.start();
  auto read = f.submit(0, 8);
  REQUIRE(earlier_get.entered());
  REQUIRE(wait_until([&] { return read.completion->calls.load() == 1; }));
  CHECK(read.completion->success.load());
  CHECK(read.completion->fragments.load() == 0);
  CHECK_FALSE(read.future.is_ready());
  CHECK(read.coordinator->tasks_remaining() > 0);
  earlier_get.release();
  REQUIRE(error_from(read.future) != nullptr);
  f.shutdown();
  CHECK(read.completion->calls.load() == 1);
  CHECK(read.coordinator->tasks_remaining() == 0);
  CHECK(f.reactor->perf().fail_stop_total == 1);
}

TEST_CASE("S3RDMA failed start rolls back sessions and cannot be restarted", "[s3rdma][runtime]")
{
  auto const leg = GENERATE(0, 1, 2, 3, 4);
  runtime_fixture f{runtime_config(2)};
  action throwing;
  throwing.throws = true;
  if (leg == 0) { f.controls->script(operation::acquire_session, throwing); }
  if (leg == 1) {
    f.controls->script(operation::acquire_session, {});
    f.controls->script(operation::acquire_session, {});
    f.controls->script(operation::acquire_session, throwing);
  }
  if (leg == 2) { f.controls->script(operation::register_memory, throwing); }
  if (leg == 3) { f.controls->mock->callback_limit(-1); }
  if (leg == 4) { f.controls->mock->callback_limit(3); }
  CHECK_THROWS(f.start());
  auto status = f.reactor->status();
  CHECK(status->torn_down());
  CHECK_FALSE(status->terminal());
  CHECK_FALSE(status->arena_retained());
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(f.controls->log.count(operation::allocate, true) ==
        f.controls->log.count(operation::free, true));
  CHECK(f.controls->mock->counts().registrations == f.controls->mock->counts().deregistrations);
  CHECK(f.controls->log.count(operation::release_session) == f.controls->mock->counts().sessions);
  auto before = f.controls->log.snapshot().size();
  CHECK_THROWS(f.start());
  CHECK(f.controls->log.snapshot().size() == before);
  runtime_fixture next;
  next.controls = f.controls;
  next.controls->mock->callback_limit({});
  CHECK_NOTHROW(next.start());
  auto read = next.submit(0, 4);
  delivered(read, 4);
}

TEST_CASE("S3RDMA failed startup publishes the decision before joining waiting workers",
          "[s3rdma][runtime]")
{
  rendezvous last_session;
  runtime_fixture f{runtime_config(2)};
  f.construct(f.controls->policy());
  f.controls->script(operation::acquire_session, {});
  f.controls->script(operation::acquire_session, {});
  auto late   = held(last_session);
  late.throws = true;
  f.controls->script(operation::acquire_session, late);
  std::atomic<bool> submitting{false};
  workers threads;
  auto starting = threads.launch([&] {
    try {
      f.start();
    } catch (...) {
      return true;
    }
    return false;
  });
  REQUIRE(last_session.entered());
  auto request = threads.launch([&] {
    submitting.store(true);
    return f.submit(0, 4);
  });
  REQUIRE(wait_until([&] { return submitting.load(); }));
  CHECK(f.controls->log.count(operation::get) == 0);
  last_session.release();
  CHECK(ready(starting));
  auto early = ready(request);
  REQUIRE(error_from(early.future) != nullptr);
  CHECK(early.coordinator->tasks_remaining() == 0);
  CHECK(f.reactor->status()->torn_down());
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK_FALSE(last_session.timed_out());
}

TEST_CASE("S3RDMA rollback retains failed deregistration or failed free", "[s3rdma][runtime]")
{
  auto const release_failure = GENERATE(false, true);
  runtime_fixture f;
  f.controls->mock->callback_limit(-1);
  if (release_failure) {
    f.controls->mock->script_deregister(false);
  } else {
    f.controls->script(operation::free, failed(cudaErrorInvalidValue));
  }
  CHECK_THROWS(f.start());
  CHECK(f.reactor->status()->terminal());
  CHECK(f.reactor->status()->arena_retained());
  CHECK(f.reactor->status()->torn_down());
  CHECK(f.controls->log.count(operation::release_session) == 0);
  auto read = f.submit(0, 4);
  CHECK(error_from(read.future) == f.context->gate().terminal_error());
  auto before = f.controls->log.snapshot().size();
  CHECK_THROWS(f.start());
  f.shutdown();
  CHECK(f.controls->log.snapshot().size() == before);
  CHECK(f.controls->log.count(operation::free) == (release_failure ? 0 : 1));
}

TEST_CASE("S3RDMA sticky startup distinguishes health from actual arena residue",
          "[s3rdma][runtime]")
{
  auto const point         = GENERATE(operation::probe, operation::allocate, operation::set_device);
  auto const second_device = GENERATE(false, true);
  runtime_fixture f{runtime_config(1, 4, {7, 9})};
  if (second_device) { f.controls->script(point, {}); }
  f.controls->script(point, failed(cudaErrorIllegalAddress));
  CHECK_THROWS(f.start());
  CHECK(f.reactor->status()->terminal());
  CHECK(f.reactor->status()->arena_retained() == second_device);
  CHECK(f.reactor->status()->torn_down());
  CHECK(f.controls->policies.load() == 1);
  CHECK(f.controls->log.count(operation::get) == 0);
  f.reactor.reset();
  f.context.reset();
  no_calls_after_sticky(f.controls->log);
}

TEST_CASE("S3RDMA configuration validates devices operations and checked arena footprint",
          "[s3rdma][runtime]")
{
  auto cfg = runtime_config();
  CHECK(rdma::arena_bytes(cfg) == 4);
  SECTION("empty device set") { cfg.devices.clear(); }
  SECTION("negative device") { cfg.devices = {-1}; }
  SECTION("duplicate device") { cfg.devices = {7, 7}; }
  CHECK_THROWS_AS(rdma::sanitized(cfg), std::invalid_argument);
  cfg              = runtime_config();
  cfg.max_inflight = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(rdma::arena_bytes(cfg), std::invalid_argument);
  auto controls = std::make_shared<runtime_controls>();
  auto ops      = controls->delivery();
  CHECK_NOTHROW(rdma::validate(ops));
  using operations = rdma::cuda_delivery_ops;
  auto members     = std::tuple{&operations::get_device,
                            &operations::set_device,
                            &operations::malloc_device,
                            &operations::free_device,
                            &operations::probe_writes,
                            &operations::flush,
                            &operations::stream_capture_query,
                            &operations::event_create,
                            &operations::memcpy_async,
                            &operations::event_record,
                            &operations::event_synchronize,
                            &operations::event_destroy,
                            &operations::fatal_hook};
  std::apply(
    [&](auto... member) {
      (
        [&] {
          auto incomplete    = ops;
          incomplete.*member = {};
          CHECK_THROWS_AS(rdma::validate(incomplete), std::invalid_argument);
        }(),
        ...);
    },
    members);
  CHECK(controls->log.snapshot().empty());
}

TEST_CASE("S3RDMA performance snapshots cover every field after the barrier", "[s3rdma][runtime]")
{
  runtime_fixture f;
  f.start();
  auto read = f.submit(0, 6);
  delivered(read, 6);
  auto status = f.reactor->status();
  f.shutdown();
  auto perf = f.reactor->perf();
  CHECK(perf.bytes_total == 6);
  CHECK(perf.requests_total == 1);
  CHECK(perf.error_total == 0);
  CHECK(perf.short_read_total == 0);
  CHECK(perf.retries_total == 0);
  CHECK(perf.flush_total == 2);
  CHECK(perf.slot_wait_total == 0);
  CHECK(perf.inflight_peak == 1);
  CHECK(perf.slots_in_use_peak == 1);
  CHECK(perf.arena_leak_total == 0);
  CHECK(perf.fail_stop_total == 0);
  CHECK(perf.envelope_wait_total == 0);
  CHECK(perf.envelope_wait_ns_total == 0);
  CHECK(perf.envelope_depth_peak == 1);
  f.reactor.reset();
  f.context.reset();
  CHECK(status->torn_down());
  CHECK_FALSE(status->terminal());
  CHECK_FALSE(status->arena_retained());
}
