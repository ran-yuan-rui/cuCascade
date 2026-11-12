/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"
#include "utils/allocation_counter.hpp"

#include <algorithm>

using namespace cucascade::test::s3rdma;
namespace io = cucascade::io;

namespace {

void cuda_ok(cudaError_t result)
{
  if (result != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(result)); }
}

struct gpu_fixture {
  explicit gpu_fixture(std::function<void(rdma::cuda_delivery_ops&,
                                          std::shared_ptr<runtime_controls> const&)> decorate = {})
  {
    cuda_ok(cudaSetDevice(0));
    source.resize(64);
    for (std::size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<std::uint8_t>(11 + i * 3);
    }
    controls->mock->put({"test", "object"}, source);
    object = std::make_shared<rdma::s3rdma_io_object>(
      "s3://test/object", rdma::object_ref{"test", "object"}, source.size());
    auto ops = controls->delivery();
    if (decorate) { decorate(ops, controls); }
    ctx = std::make_unique<rdma::s3rdma_ioctx>(
      runtime_config(2, 4, {0}), controls->clients(), std::move(ops), controls->policy());
    controls->policy_observer = [this](auto const&) {
      policy_saw_terminal.store(ctx->status()->terminal());
    };
    try {
      void* pointer = nullptr;
      cuda_ok(cudaMalloc(&pointer, 128));
      destination = static_cast<std::uint8_t*>(pointer);
      cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cuda_ok(cudaMemsetAsync(destination, 0xcc, 128, stream));
      cuda_ok(cudaStreamSynchronize(stream));
    } catch (...) {
      cleanup();
      throw;
    }
  }
  ~gpu_fixture()
  {
    if (ctx) {
      ctx->shutdown();
      ctx.reset();
    }
    controls->reclaim_sessions();
    cleanup();
  }
  void cleanup() noexcept
  {
    if (stream) { (void)cudaStreamDestroy(std::exchange(stream, nullptr)); }
    if (destination) { (void)cudaFree(std::exchange(destination, nullptr)); }
  }
  io::prepared_io_slice slice(std::size_t offset, std::size_t size, std::size_t landing = 0)
  {
    return io::prepared_io_slice{
      io::range{offset, size},
      io::device_buffer{destination + landing, ::cuda::stream_ref{stream}, 0}};
  }
  std::array<std::uint8_t, 128> bytes()
  {
    std::array<std::uint8_t, 128> result;
    cuda_ok(cudaMemcpy(result.data(), destination, result.size(), cudaMemcpyDeviceToHost));
    return result;
  }
  deadline bound;
  std::shared_ptr<runtime_controls> controls{std::make_shared<runtime_controls>(true)};
  std::vector<std::uint8_t> source;
  std::shared_ptr<rdma::s3rdma_io_object> object;
  std::unique_ptr<rdma::s3rdma_ioctx> ctx;
  std::atomic<bool> policy_saw_terminal{false};
  std::uint8_t* destination{nullptr};
  cudaStream_t stream{nullptr};
};

struct capture_scope {
  explicit capture_scope(cudaStream_t value) : stream(value)
  {
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  }
  ~capture_scope()
  {
    if (active) { (void)finish(); }
  }
  cudaError_t finish()
  {
    cudaGraph_t graph = nullptr;
    auto rc           = cudaStreamEndCapture(stream, &graph);
    active            = false;
    if (graph) { (void)cudaGraphDestroy(graph); }
    return rc;
  }
  cudaStream_t stream;
  bool active{true};
};

}  // namespace

TEST_CASE("S3RDMA public device entry performs real D2D and clips prepared ranges", "[s3rdma][gpu]")
{
  gpu_fixture f;
  f.ctx->start();
  std::array<std::shared_ptr<completion_record>, 4> callbacks;
  std::vector<io::prepared_io_slice> slices{
    f.slice(1, 6, 4), f.slice(62, 8, 16), f.slice(0, 0, 24), f.slice(64, 4, 32)};
  for (std::size_t i = 0; i < slices.size(); ++i) {
    callbacks[i]          = std::make_shared<completion_record>();
    slices[i].on_complete = callbacks[i]->callback();
  }
  auto result = f.ctx->mixed_readv_async_io(*f.object, std::move(slices));
  REQUIRE(wait_until([&] { return result.is_ready(); }));
  CHECK(std::move(result).get(2s) == 8);
  auto expected = std::array<std::uint8_t, 128>{};
  expected.fill(0xcc);
  std::copy_n(f.source.data() + 1, 6, expected.data() + 4);
  std::copy_n(f.source.data() + 62, 2, expected.data() + 16);
  CHECK(f.bytes() == expected);
  for (std::size_t i = 0; i < callbacks.size(); ++i) {
    CHECK(callbacks[i]->calls.load() == 1);
    CHECK(callbacks[i]->success.load() == (i < 2));
    CHECK(callbacks[i]->fragments.load() == 0);
  }
  f.ctx->shutdown();
  CHECK(f.controls->log.count(operation::get) == 3);
  CHECK(f.controls->log.count(operation::copy) == 3);
  CHECK(f.controls->log.count(operation::wait, true) == 3);
  CHECK(f.controls->log.count(operation::range) == 0);
  for (auto const& e : f.controls->log.snapshot()) {
    if (e.op == operation::copy) { CHECK(e.kind == cudaMemcpyDeviceToDevice); }
  }
}

TEST_CASE("S3RDMA public captured-stream refusal issues no GET and permits a later read",
          "[s3rdma][gpu]")
{
  gpu_fixture f;
  f.ctx->start();
  auto completion   = std::make_shared<completion_record>();
  auto slice        = f.slice(0, 4);
  slice.on_complete = completion->callback();
  capture_scope capture{f.stream};
  auto rejected   = f.ctx->mixed_readv_async_io(*f.object, {slice});
  auto error      = error_from(rejected);
  auto end_result = capture.finish();
  REQUIRE(error != nullptr);
  CHECK(end_result == cudaSuccess);
  CHECK(f.controls->log.count(operation::get) == 0);
  CHECK(f.controls->log.count(operation::copy) == 0);
  CHECK(completion->calls.load() == 1);
  CHECK_FALSE(completion->success.load());
  CHECK_FALSE(f.ctx->status()->terminal());
  auto next = f.ctx->mixed_readv_async_io(*f.object, {f.slice(8, 6, 8)});
  REQUIRE(wait_until([&] { return next.is_ready(); }));
  CHECK(std::move(next).get(2s) == 6);
  auto bytes = f.bytes();
  CHECK(std::equal(f.source.begin() + 8, f.source.begin() + 14, bytes.begin() + 8));
}

TEST_CASE("S3RDMA framework dispatch failure invokes the context-health probe", "[s3rdma][gpu]")
{
  auto const sticky = GENERATE(false, true);
  bool witnessed    = false;
  // Locate a real dispatch allocation; do not bind the test to an allocation index.
  for (std::size_t index = 1; index <= 32 && !witnessed; ++index) {
    auto armed  = std::make_shared<std::atomic<bool>>(false);
    auto probes = std::make_shared<std::atomic<unsigned>>(0);
    auto caller = std::this_thread::get_id();
    gpu_fixture f{
      [&](rdma::cuda_delivery_ops& ops, std::shared_ptr<runtime_controls> const& controls) {
        auto ordinary  = ops.get_device;
        ops.get_device = [=](int* device) {
          if (armed->load() && std::this_thread::get_id() == caller) {
            probes->fetch_add(1);
            auto code = sticky ? cudaErrorIllegalAddress : cudaErrorInvalidValue;
            controls->log.append({operation::get_device, nullptr, 0, 0, -1, nullptr, code, true});
            return code;
          }
          return ordinary(device);
        };
      }};
    f.ctx->start();
    auto completion   = std::make_shared<completion_record>();
    auto slice        = f.slice(0, 4);
    slice.on_complete = completion->callback();
    std::vector<io::prepared_io_slice> slices{slice};
    future_type result;
    armed->store(true);
    {
      cucascade::test::scoped_allocation_counter allocation;
      allocation.fail_at(index);
      result = f.ctx->mixed_readv_async_io(*f.object, std::move(slices));
    }
    armed->store(false);
    REQUIRE(wait_until([&] { return result.is_ready(); }));
    if (probes->load() == 0) {
      (void)error_from(result);
      continue;
    }
    witnessed = true;
    CHECK_THROWS_AS(std::move(result).get(2s), std::bad_alloc);
    CHECK(completion->calls.load() == 1);
    CHECK_FALSE(completion->success.load());
    CHECK(f.controls->log.count(operation::get) == 0);
    CHECK(f.ctx->status()->terminal() == sticky);
    if (sticky) {
      CHECK(f.controls->policies.load() == 1);
      CHECK(f.policy_saw_terminal.load());
      auto later = f.ctx->mixed_readv_async_io(*f.object, {f.slice(8, 4, 8)});
      REQUIRE(error_from(later) != nullptr);
      f.ctx->shutdown();
      CHECK(f.ctx->status()->arena_retained());
      CHECK(f.ctx->status()->torn_down());
      f.ctx.reset();
      auto after = f.controls->log.snapshot();
      auto fault = std::find_if(after.begin(), after.end(), [](auto const& e) {
        return e.returned && e.result == cudaErrorIllegalAddress;
      });
      REQUIRE(fault != after.end());
      for (auto it = fault + 1; it != after.end(); ++it) {
        if (!it->returned) {
          CHECK((it->op == operation::worker_exit || it->op == operation::policy));
        }
      }
    } else {
      CHECK(f.controls->policies.load() == 0);
      auto later = f.ctx->mixed_readv_async_io(*f.object, {f.slice(8, 4, 8)});
      CHECK(std::move(later).get(2s) == 4);
    }
  }
  REQUIRE(witnessed);
}
