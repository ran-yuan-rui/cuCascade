/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"

#include <algorithm>
#include <iostream>

using namespace cucascade::test::s3rdma;
namespace io = cucascade::io;

namespace {

action response(long status,
                std::string error         = {},
                std::size_t bytes         = 4,
                std::string content_range = "bytes 2-5/64")
{
  action step;
  step.range = rdma::range_get_result{{status, std::move(error)}, bytes, std::move(content_range)};
  return step;
}

std::exception_ptr caught(std::function<void()> const& function)
{
  try {
    function();
  } catch (...) {
    return std::current_exception();
  }
  return {};
}

rdma::s3rdma_reactor& selected(host_fixture& f)
{
  auto result = f.ctx->next_reactor(static_cast<rdma::s3rdma_io_object const&>(*f.object),
                                    1,
                                    rdma::s3rdma_ioctx::io_op_type::host,
                                    7);
  REQUIRE(result.size() == 1);
  return *result.front();
}

}  // namespace

TEST_CASE("S3RDMA host range identification requires the entire header grammar",
          "[s3rdma][host][range-grammar]")
{
  auto const header = GENERATE("bytes 2-",
                               "bytes 2-garbage",
                               "bytes 2-5",
                               "bytes 2-5/",
                               "bytes 2-5/64x",
                               "bytes 2-5/64 ",
                               "bytes 2-5/*x",
                               "bytes 5-2/64",
                               "bytes 2-64/64",
                               "bytes 2-5/5",
                               "bytes 2-5/0",
                               " bytes 2-5/64",
                               "Bytes 2-5/64",
                               "bytes=2-5/64",
                               "bytes  2-5/64",
                               "bytes */64",
                               "bytes 2-5/-1",
                               "bytes 184467440737095516160-5/64",
                               "bytes 2-184467440737095516160/64",
                               "bytes 2-5/184467440737095516160");
  CAPTURE(header);
  host_fixture f;
  f.start();
  f.controls->script(operation::range, response(206, {}, 4, header));
  auto result = f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  CHECK(has_code(error_from(result), rdma::errc::control_failed));
  CHECK(f.controls->log.count(operation::range) == 1);
  CHECK_FALSE(f.ctx->status()->terminal());
}

TEST_CASE("S3RDMA valid range grammar accepts known and unknown lengths",
          "[s3rdma][host][range-grammar]")
{
  auto const header = GENERATE("bytes 2-5/64", "bytes 2-5/*");
  host_fixture f;
  f.start();
  f.controls->script(operation::range, response(206, {}, 4, header));
  CHECK(f.ctx->host_read(*f.object, 2, 4, f.destination.data()) == 4);
  CHECK(f.controls->log.count(operation::range) == 1);
  CHECK(std::equal(f.source.begin() + 2, f.source.begin() + 6, f.destination.begin()));
}

TEST_CASE("S3RDMA malformed range identification prevents short-body retry",
          "[s3rdma][host][range-grammar]")
{
  host_fixture f;
  f.start();
  f.controls->script(operation::range, response(206, {}, 3, "bytes 2-"));
  auto result = f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  CHECK(has_code(error_from(result), rdma::errc::control_failed));
  CHECK(f.controls->log.count(operation::range) == 1);
}

TEST_CASE("S3RDMA public host entries read exact scalar and vector ranges", "[s3rdma][host]")
{
  host_fixture f;
  f.start();
  CHECK(f.ctx->host_read(*f.object, 2, 4, f.destination.data()) == 4);
  CHECK(std::equal(f.source.begin() + 2, f.source.begin() + 6, f.destination.begin()));
  auto async = f.ctx->host_read_async(*f.object, 8, 3, f.destination.data() + 8);
  CHECK(async.is_ready());
  CHECK(std::move(async).get(2s) == 3);
  std::array<io::slice, 2> ranges{io::slice{12, 4, f.destination.data() + 12},
                                  io::slice{25, 7, f.destination.data() + 25}};
  auto vector = f.ctx->host_readv_async_io(*f.object, ranges, io::io_priority::automatic);
  CHECK(vector.is_ready());
  CHECK(std::move(vector).get(2s) == 11);
  for (auto const& range : ranges) {
    CHECK(std::equal(f.source.data() + range.offset(),
                     f.source.data() + range.offset() + range.size(),
                     range.dst));
  }
  auto& reactor = selected(f);
  f.ctx->shutdown();
  CHECK(f.controls->log.count(operation::range) == 4);
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(reactor.perf().retries_total == 0);
}

TEST_CASE("S3RDMA public host entries clip EOF and reject nonempty null destinations",
          "[s3rdma][host]")
{
  auto const entry = GENERATE(0, 1, 2);
  host_fixture f;
  f.start();
  auto read = [&](std::size_t offset, std::size_t size, std::uint8_t* dst) {
    if (entry == 0) { return f.ctx->host_read(*f.object, offset, size, dst); }
    if (entry == 1) {
      auto result = f.ctx->host_read_async(*f.object, offset, size, dst);
      return std::move(result).get(2s);
    }
    io::slice range;
    range.rng = {offset, size};
    range.dst = dst;
    auto result =
      f.ctx->host_readv_async_io(*f.object, std::span{&range, 1}, io::io_priority::automatic);
    return std::move(result).get(2s);
  };
  CHECK(read(0, 0, nullptr) == 0);
  CHECK(read(f.source.size(), 4, f.destination.data()) == 0);
  CHECK(f.controls->log.count(operation::range) == 0);
  CHECK_THROWS(read(0, 4, nullptr));
  CHECK(f.controls->log.count(operation::range) == 0);
  CHECK(read(62, 8, f.destination.data() + 1) == 2);
  CHECK(f.destination[0] == 0xcc);
  CHECK(f.destination[1] == f.source[62]);
  CHECK(f.destination[2] == f.source[63]);
  CHECK(f.destination[3] == 0xcc);
  CHECK(f.controls->log.count(operation::range) == 1);
}

TEST_CASE("S3RDMA host retries are bounded and do not increment the RDMA retry counter",
          "[s3rdma][host]")
{
  auto const fault   = GENERATE(0, 1, 2);
  auto const exhaust = GENERATE(false, true);
  host_fixture f;
  f.start();
  auto bad = fault == 0   ? response(206, "connection reset")
             : fault == 1 ? response(503)
                          : response(206, {}, 3);
  f.controls->script(operation::range, bad);
  f.controls->script(operation::range, bad);
  if (exhaust) { f.controls->script(operation::range, bad); }
  auto result = f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  CHECK(result.is_ready());
  if (exhaust) {
    REQUIRE(error_from(result) != nullptr);
  } else {
    CHECK(std::move(result).get(2s) == 4);
    CHECK(std::equal(f.source.begin() + 2, f.source.begin() + 6, f.destination.begin()));
  }
  auto& reactor = selected(f);
  f.ctx->shutdown();
  CHECK(f.controls->log.count(operation::range) == 3);
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(reactor.perf().retries_total == 0);
  CHECK_FALSE(f.ctx->status()->terminal());
}

TEST_CASE("S3RDMA host 4xx responses take precedence over transport errors", "[s3rdma][host]")
{
  auto const status          = GENERATE(400L, 403L, 404L, 412L, 416L);
  auto const transport_fault = GENERATE(false, true);
  host_fixture f;
  f.start();
  f.controls->script(operation::range, response(status, transport_fault ? "truncated" : ""));
  auto result = f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  REQUIRE(error_from(result) != nullptr);
  CHECK(f.controls->log.count(operation::range) == 1);
  auto next = f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  CHECK(std::move(next).get(2s) == 4);
  CHECK_FALSE(f.ctx->status()->terminal());
}

TEST_CASE("S3RDMA successful HTTP status must identify the requested host range", "[s3rdma][host]")
{
  auto const leg = GENERATE(0, 1, 2, 3);
  host_fixture f;
  f.start();
  auto bad = response(206);
  if (leg == 0) { bad.range->content_range.clear(); }
  if (leg == 1) { bad.range->content_range = "unparsable"; }
  if (leg == 2) { bad.range->content_range = "bytes 1-4/64"; }
  if (leg == 3) { bad.range->outcome.http_status = 200; }
  f.controls->script(operation::range, bad);
  auto result = f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  REQUIRE(error_from(result) != nullptr);
  CHECK(f.controls->log.count(operation::range) == 1);
}

TEST_CASE("S3RDMA a whole-object host response may use status 200", "[s3rdma][host]")
{
  host_fixture f;
  f.start();
  f.controls->script(operation::range, response(200, {}, f.source.size(), {}));
  CHECK(f.ctx->host_read(*f.object, 0, f.source.size(), f.destination.data()) == f.source.size());
  CHECK(std::equal(f.source.begin(), f.source.end(), f.destination.begin()));
  CHECK(f.controls->log.count(operation::range) == 1);
}

TEST_CASE("S3RDMA close prevents the next host attempt or vector range", "[s3rdma][host]")
{
  auto const vector_read = GENERATE(false, true);
  rendezvous active;
  host_fixture f;
  f.start();
  action step;
  step.before = [&](auto const&) { active.arrive_and_wait(); };
  if (!vector_read) { step.range = rdma::range_get_result{{503, {}}, 4, "bytes 2-5/64"}; }
  f.controls->script(operation::range, step);
  std::array<io::slice, 2> ranges{io::slice{2, 4, f.destination.data()},
                                  io::slice{12, 4, f.destination.data() + 12}};
  workers threads;
  auto reading = threads.launch([&] {
    return vector_read ? f.ctx->host_readv_async_io(*f.object, ranges, io::io_priority::automatic)
                       : f.ctx->host_read_async(*f.object, 2, 4, f.destination.data());
  });
  REQUIRE(active.entered());
  auto closing = threads.launch([&] { f.ctx->shutdown(); });
  // Observe the public refusal instead of assuming a shutdown thread has run.
  std::exception_ptr closed;
  std::size_t probe = 0;
  REQUIRE(wait_until([&] {
    auto error = caught(
      [&] { (void)f.ctx->open_io_object("s3://test/close-probe-" + std::to_string(probe++)); });
    if (has_code(error, rdma::errc::transport_closed)) {
      closed = error;
      return true;
    }
    return false;
  }));
  active.release();
  auto result = ready(reading);
  CHECK(error_from(result) == closed);
  ready(closing);
  CHECK(f.controls->log.count(operation::range) == 1);
  auto attempts = f.controls->log.count(operation::range);
  CHECK(caught([&] { (void)f.ctx->host_read(*f.object, 0, 4, f.destination.data()); }) == closed);
  auto later = f.ctx->host_read_async(*f.object, 0, 4, f.destination.data());
  CHECK(error_from(later) == closed);
  CHECK(f.controls->log.count(operation::range) == attempts);
  CHECK_FALSE(active.timed_out());
}

TEST_CASE("S3RDMA public opens always use one HEAD rather than the supplied size", "[s3rdma][host]")
{
  auto const form  = GENERATE(0, 1, 2);
  auto const empty = GENERATE(false, true);
  host_fixture f;
  if (empty) { f.controls->mock->put({"test", "object"}, {}); }
  f.start();
  std::shared_ptr<io::io_object> object;
  if (form == 0) { object = f.ctx->open_io_object("s3://test/object"); }
  if (form == 1) {
    object = f.ctx->open_io_object("s3://test/object", io::open_hint::parquet_footer_probe);
  }
  if (form == 2) { object = f.ctx->open_io_object("s3://test/object", std::uint64_t{9999}); }
  REQUIRE(object != nullptr);
  CHECK(object->size() == (empty ? 0 : f.source.size()));
  CHECK(object->object_path() == "s3://test/object");
  CHECK(f.controls->log.count(operation::head) == 1);
  CHECK(f.controls->log.count(operation::range) == 0);
}

TEST_CASE("S3RDMA public open rejects failed or unusable HEAD metadata without retry",
          "[s3rdma][host]")
{
  auto const leg = GENERATE(0, 1, 2, 3);
  host_fixture f;
  f.start();
  action step;
  if (leg == 0) { step.head = rdma::head_result{{404, {}}, 64}; }
  if (leg == 1) { step.head = rdma::head_result{{503, {}}, 64}; }
  if (leg == 2) { step.head = rdma::head_result{{200, "connection reset"}, 64}; }
  // Metadata parsing belongs to the control client; the value surface reports its failure here.
  if (leg == 3) { step.head = rdma::head_result{{200, "invalid Content-Length"}, 0}; }
  f.controls->script(operation::head, step);
  CHECK_THROWS(f.ctx->open_io_object("s3://test/object"));
  CHECK(f.controls->log.count(operation::head) == 1);
  CHECK_FALSE(f.ctx->status()->terminal());
}

TEST_CASE("S3RDMA public reads and opens acquire nothing before successful start", "[s3rdma][host]")
{
  host_fixture f;
  auto result = f.ctx->host_read_async(*f.object, 0, 4, f.destination.data());
  CHECK(has_code(error_from(result), rdma::errc::backend_unavailable));
  CHECK(has_code(caught([&] { (void)f.ctx->host_read(*f.object, 0, 4, f.destination.data()); }),
                 rdma::errc::backend_unavailable));
  CHECK(has_code(caught([&] { (void)f.ctx->open_io_object("s3://test/object"); }),
                 rdma::errc::backend_unavailable));
  CHECK(f.controls->log.snapshot().empty());
  f.start();
  CHECK(f.ctx->host_read(*f.object, 0, 4, f.destination.data()) == 4);
}

TEST_CASE("S3RDMA terminal public mixed entry preserves lifecycle and domain precedence",
          "[s3rdma][host]")
{
  host_fixture f;
  action sticky;
  sticky.result = cudaErrorIllegalAddress;
  f.controls->script(operation::probe, sticky);
  CHECK_THROWS(f.start());
  CHECK(f.ctx->status()->terminal());
  CHECK_FALSE(f.ctx->status()->arena_retained());
  auto error = caught([&] { (void)f.ctx->open_io_object("s3://test/object"); });
  REQUIRE(error != nullptr);
  auto before   = f.controls->log.snapshot().size();
  auto callback = std::make_shared<completion_record>();
  io::prepared_io_slice slice{
    io::range{0, 4},
    io::device_buffer{f.destination.data(), ::cuda::stream_ref{cudaStream_t{nullptr}}, 7}};
  slice.on_complete = callback->callback();
  auto device       = f.ctx->mixed_readv_async_io(*f.object, {slice});
  CHECK(error_from(device) == error);
  CHECK(callback->calls.load() == 1);
  CHECK_FALSE(callback->success.load());
  auto empty = f.ctx->mixed_readv_async_io(*f.object, {});
  CHECK(std::move(empty).get(2s) == 0);
  slice.d_buffer = {};
  slice.h_buffer = io::host_buffer{f.destination.data()};
  auto host      = f.ctx->mixed_readv_async_io(*f.object, {slice});
  CHECK(has_code(error_from(host), rdma::errc::unsupported_request));
  CHECK(callback->calls.load() == 2);
  CHECK(f.controls->log.snapshot().size() == before);
  CHECK(f.controls->policies.load() == 1);
}

TEST_CASE("S3RDMA public device and host entries share the same RDMA terminal error",
          "[s3rdma][host]")
{
  host_fixture f;
  f.start();
  action bad;
  bad.change_receipt = [](auto& result) { result.callback_invocations = 0; };
  f.controls->script(operation::get, bad);
  io::prepared_io_slice slice{
    io::range{0, 4},
    io::device_buffer{f.destination.data(), ::cuda::stream_ref{cudaStream_t{nullptr}}, 7}};
  auto request = io::grouped_io_request::create(f.object, {slice});
  auto result  = request->coordinator->get_future();
  selected(f).enqueue(std::move(request));
  auto fatal = error_from(result);
  REQUIRE(fatal != nullptr);
  auto device = f.ctx->mixed_readv_async_io(*f.object, {slice});
  CHECK(error_from(device) == fatal);
  auto host = f.ctx->host_read_async(*f.object, 0, 4, f.destination.data());
  CHECK(error_from(host) == fatal);
  CHECK(caught([&] { (void)f.ctx->open_io_object("s3://test/object"); }) == fatal);
  f.ctx->shutdown();
  CHECK(f.controls->log.count(operation::get) == 1);
  CHECK(f.controls->log.count(operation::range) == 0);
  CHECK(f.controls->log.count(operation::head) == 0);
  CHECK(f.ctx->status()->arena_retained());
}

TEST_CASE("S3RDMA public host latency reports the injected control-client delay",
          "[s3rdma][host][measurement]")
{
  host_fixture f;
  f.start();
  action delay;
  delay.before = [](auto const&) { std::this_thread::sleep_for(1ms); };
  for (int i = 0; i < 4; ++i) {
    f.controls->script(operation::range, delay);
  }
  auto start      = std::chrono::steady_clock::now();
  auto scalar     = f.ctx->host_read_async(*f.object, 0, 4, f.destination.data());
  auto scalar_end = std::chrono::steady_clock::now();
  CHECK(scalar.is_ready());
  CHECK(std::move(scalar).get(2s) == 4);
  std::array<io::slice, 3> ranges{io::slice{8, 4, f.destination.data() + 8},
                                  io::slice{16, 4, f.destination.data() + 16},
                                  io::slice{24, 4, f.destination.data() + 24}};
  auto vector_start = std::chrono::steady_clock::now();
  auto vector       = f.ctx->host_readv_async_io(*f.object, ranges, io::io_priority::automatic);
  auto vector_end   = std::chrono::steady_clock::now();
  CHECK(vector.is_ready());
  CHECK(std::move(vector).get(2s) == 12);
  CHECK(f.controls->log.count(operation::range) == 4);
  std::cout
    << "S3RDMA_HOST_MEASUREMENT control=scripted delay_us=1000 scalar_us="
    << std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - start).count()
    << " vector_ranges=3 vector_us="
    << std::chrono::duration_cast<std::chrono::microseconds>(vector_end - vector_start).count()
    << '\n';
}
