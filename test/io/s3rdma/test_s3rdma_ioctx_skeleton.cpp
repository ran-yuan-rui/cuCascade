/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Include the reactor before the context that instantiates it.
// clang-format off
#include <cucascade/io/s3rdma/s3rdma_reactor.hpp>
// clang-format on
#include "mock_rdma_client.hpp"
#include "test_support.hpp"
#include "utils/allocation_counter.hpp"

#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/s3rdma/s3rdma_ioctx.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/null_device_memory_resource.hpp>

#include <atomic>
#include <limits>

using namespace cucascade::test::s3rdma;
namespace io     = cucascade::io;
namespace memory = cucascade::memory;

namespace {

rdma::config geometry()
{
  rdma::config cfg;
  cfg.max_inflight           = 2;
  cfg.arena_slot_size        = 4;
  cfg.n_max_concurrent_scans = 5;
  return cfg;
}

struct cpu_memory {
  static std::vector<memory::memory_space_config> configs()
  {
    memory::host_memory_space_config host;
    host.numa_id              = 0;
    host.memory_capacity      = 4096;
    host.block_size           = 256;
    host.pool_size            = 4;
    host.initial_number_pools = 0;
    host.mr_factory_fn        = [](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        memory::null_device_memory_resource{}};
    };
    return {host};
  }
  memory::memory_reservation_manager manager{configs()};
};

class observed_object final : public io::io_object {
 public:
  std::string const& raw_file_cache_id() const noexcept override { return _path; }
  std::string const& object_path() const noexcept override { return _path; }
  std::size_t size() const noexcept override
  {
    ++size_queries;
    return 0;
  }
  mutable std::atomic<unsigned> size_queries{0};

 private:
  std::string _path{"s3://test/object"};
};

struct completions {
  std::array<unsigned, 2> calls{};
  std::array<bool, 2> success{true, true};
  std::array<std::size_t, 2> fragments{};

  void attach(std::vector<io::prepared_io_slice>& slices)
  {
    for (std::size_t i = 0; i < slices.size(); ++i) {
      slices[i].on_complete =
        std::make_shared<io::prepared_io_completion>([this, i](auto chunks, bool ok) noexcept {
          ++calls[i];
          success[i]   = ok;
          fragments[i] = chunks.size();
        });
    }
  }
  void check() const
  {
    CHECK(calls == std::array<unsigned, 2>{1, 1});
    CHECK(success == std::array<bool, 2>{false, false});
  }
};

void no_transport(mock_rdma_client const& mock)
{
  auto const counts = mock.counts();
  CHECK(counts.heads == 0);
  CHECK(counts.ranges == 0);
  CHECK(counts.gets == 0);
  CHECK(counts.registrations == 0);
}

}  // namespace

TEST_CASE("S3RDMA reactor traits and context expose the device-only profile", "[s3rdma][ioctx]")
{
  deadline bound;
  static_assert(io::io_reactor_c<rdma::s3rdma_reactor>);
  using traits = io::reactor_traits<rdma::s3rdma_reactor>;
  static_assert(traits::supports_device_read);
  static_assert(traits::supports_device_range_read);
  static_assert(!traits::supports_host_to_device_read);
  static_assert(!traits::supports_vector_host_read);
  static_assert(traits::prefers_bulk_io);
  auto mock = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  CHECK(ctx.supports_device_read());
  CHECK(ctx.supports_device_range_read());
  CHECK_FALSE(ctx.supports_host_to_device_read());
  CHECK_FALSE(ctx.supports_vector_host_read());
  CHECK(ctx.prefers_bulk_io());
  CHECK(ctx.min_alignment_requirement() == 1);
  CHECK(ctx.merge_gap_size() == 0);
  CHECK(ctx.staging_block_size() == 0);
  CHECK(ctx.n_max_concurrent_scans() == 5);
  auto context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(geometry(), clients(mock));
  CHECK(context->cfg().queue_cap == 8);
  CHECK_FALSE(context->gate().terminal());
  no_transport(*mock);
}

TEST_CASE("S3RDMA cache initialization is a no-op with a CPU-only memory fixture",
          "[s3rdma][ioctx]")
{
  deadline bound;
  cpu_memory fixture;
  auto mock = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  ctx.initialize_cache(fixture.manager, io::cache::config{}, nullptr);
  CHECK_FALSE(ctx.uses_fs_cache());
  ctx.shutdown_cache();
  CHECK_FALSE(ctx.uses_fs_cache());
  no_transport(*mock);
}

TEST_CASE("S3RDMA type and path checker select only the S3 scheme", "[s3rdma][ioctx]")
{
  deadline bound;
  auto mock = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  CHECK(ctx.type() == io::io_context_type::s3rdma);
  CHECK(rdma::supports("s3://test/object"));
  CHECK_FALSE(rdma::supports("https://example.invalid/object"));
  CHECK_FALSE(rdma::supports("file:///tmp/object"));
  CHECK_FALSE(rdma::supports("rdma://test/object"));
  CHECK_FALSE(rdma::supports(""));
}

TEST_CASE("S3RDMA empty input returns ready zero without looking at the object", "[s3rdma][ioctx]")
{
  deadline bound;
  auto mock = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  observed_object object;
  auto future = ctx.mixed_readv_async_io(object, {});
  CHECK(future.is_ready());
  CHECK(std::move(future).get(2s) == 0);
  CHECK(object.size_queries == 0);
  no_transport(*mock);
}

TEST_CASE("S3RDMA mixed entry refuses every host destination shape exactly once", "[s3rdma][ioctx]")
{
  deadline bound;
  auto const shape            = GENERATE(0, 1, 2, 3);
  auto const invalid_position = GENERATE(std::size_t{0}, std::size_t{1});
  auto mock                   = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  observed_object object;
  request_fixture fixture;
  std::vector<io::prepared_io_slice> slices{fixture.slice(0, 1), fixture.slice(8, 4, 8)};
  auto& bad = slices[invalid_position];
  if (shape != 2) { bad.d_buffer = {}; }
  if (shape == 0 || shape == 2) {
    bad.h_buffer = io::host_buffer{fixture.destination.data()};
  } else if (shape == 1) {
    bad.h_buffer = io::host_buffer{std::vector<io::cache::cached_chunk*>{nullptr}};
  }
  completions observed;
  observed.attach(slices);
  auto future = ctx.mixed_readv_async_io(object, std::move(slices));
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::unsupported_request);
  observed.check();
  CHECK(observed.fragments[invalid_position] == (shape == 1 ? 1 : 0));
  CHECK(object.size_queries == 0);
  no_transport(*mock);
}

TEST_CASE("S3RDMA device entry fails unavailable without metadata or CUDA probes",
          "[s3rdma][ioctx]")
{
  deadline bound;
  auto mock = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  observed_object object;
  request_fixture fixture;
  std::vector<io::prepared_io_slice> slices{fixture.slice(0, 1), fixture.slice(8, 4, 8)};
  completions observed;
  observed.attach(slices);
  auto future = ctx.mixed_readv_async_io(object, std::move(slices));
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
  observed.check();
  CHECK(object.size_queries == 0);
  no_transport(*mock);
}

TEST_CASE("S3RDMA every open form reports unavailable instead of inventing a size",
          "[s3rdma][ioctx]")
{
  deadline bound;
  auto mock = std::make_shared<mock_rdma_client>();
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  require_code([&] { (void)ctx.open_io_object("s3://test/object"); },
               rdma::errc::backend_unavailable);
  require_code(
    [&] { (void)ctx.open_io_object("s3://test/object", io::open_hint::parquet_footer_probe); },
    rdma::errc::backend_unavailable);
  require_code([&] { (void)ctx.open_io_object("s3://test/object", std::uint64_t{97}); },
               rdma::errc::backend_unavailable);
  no_transport(*mock);
}

TEST_CASE("S3RDMA direct reactor stub settles every known-size request slice", "[s3rdma][ioctx]")
{
  deadline bound;
  auto mock    = std::make_shared<mock_rdma_client>();
  auto context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(geometry(), clients(mock));
  rdma::s3rdma_reactor reactor{context};
  auto object = std::make_shared<rdma::s3rdma_io_object>(
    "s3://test/object", rdma::object_ref{"test", "object"}, 128);
  CHECK(object->size() == 128);
  CHECK(object->ref().bucket == "test");
  CHECK(object->ref().key == "object");
  request_fixture fixture;
  std::vector<io::prepared_io_slice> slices{fixture.slice(0, 1), fixture.slice(8, 4, 8)};
  completions observed;
  observed.attach(slices);
  auto request     = io::grouped_io_request::create(object, std::move(slices));
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  reactor.start();
  reactor.enqueue(std::move(request));
  CHECK(coordinator->tasks_remaining() == 0);
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::backend_unavailable);
  observed.check();
  CHECK(reactor.queued_bytes() == 0);
  CHECK(reactor.staging_block_size() == 0);
  reactor.interrupt();
  reactor.shutdown();
  no_transport(*mock);
}

TEST_CASE("S3RDMA reactor host read is unavailable and inherited async host entry is refused",
          "[s3rdma][ioctx]")
{
  deadline bound;
  auto mock    = std::make_shared<mock_rdma_client>();
  auto context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(geometry(), clients(mock));
  rdma::s3rdma_reactor reactor{context};
  auto object = std::make_shared<rdma::s3rdma_io_object>(
    "s3://test/object", rdma::object_ref{"test", "object"}, 128);
  std::array<std::uint8_t, 8> destination{};
  require_code([&] { (void)reactor.host_read(*object, 0, 4, destination.data()); },
               rdma::errc::backend_unavailable);
  rdma::s3rdma_ioctx ctx{geometry(), clients(mock)};
  auto future = ctx.host_read_async_io(*object, 0, 4, destination.data());
  require_code([&] { (void)std::move(future).get(2s); }, rdma::errc::unsupported_request);
  no_transport(*mock);
}

TEST_CASE("S3RDMA registry replacement routes S3 and constructs the injected context",
          "[s3rdma][ioctx]")
{
  deadline bound;
  cpu_memory fixture;
  io::io_context_registry registry{io::io_config{}, fixture.manager};
  auto mock = std::make_shared<mock_rdma_client>();
  registry.replace_ioctx(io::io_context_type::restful,
                         io::io_context_type::s3rdma,
                         &rdma::supports,
                         rdma::make_ioctx_factory(geometry(), clients(mock)));
  CHECK(registry.lookup_path("s3://test/object") == io::io_context_type::s3rdma);
  CHECK(registry.make_ioctx(io::io_context_type::restful) == nullptr);
  auto ctx = registry.make_ioctx(io::io_context_type::s3rdma);
  REQUIRE(ctx);
  CHECK(dynamic_cast<rdma::s3rdma_ioctx*>(ctx.get()) != nullptr);
  CHECK(ctx->type() == io::io_context_type::s3rdma);
}

TEST_CASE("S3RDMA registry factories contain invalid input and allocation failures",
          "[s3rdma][ioctx]")
{
  auto const fault = GENERATE(0, 1, 2, 3, 4, 5, 6);
  deadline bound;
  cpu_memory fixture;
  io::io_context_registry registry{io::io_config{}, fixture.manager};
  auto mock      = std::make_shared<mock_rdma_client>();
  auto cfg       = geometry();
  auto transport = clients(mock);
  if (fault == 0) { transport.control.reset(); }
  if (fault == 1) { transport.data_sessions.reset(); }
  if (fault == 2) { cfg.max_inflight = 0; }
  if (fault == 3) { cfg.arena_slot_size = 0; }
  if (fault == 4) { cfg.queue_cap = 0; }
  if (fault == 5) {
    cfg.max_inflight    = 1;
    cfg.arena_slot_size = 1;
    cfg.queue_cap       = std::numeric_limits<std::size_t>::max();
  }
  registry.replace_ioctx(io::io_context_type::restful,
                         io::io_context_type::s3rdma,
                         &rdma::supports,
                         rdma::make_ioctx_factory(cfg, transport));
  std::shared_ptr<io::ioctx> result;
  if (fault == 6) {
    std::size_t allocations = 0;
    {
      cucascade::test::scoped_allocation_counter counter;
      counter.fail_at(1);
      result      = registry.make_ioctx(io::io_context_type::s3rdma);
      allocations = counter.count();
    }
    CHECK(allocations > 0);
  } else {
    CHECK_NOTHROW(result = registry.make_ioctx(io::io_context_type::s3rdma));
  }
  CHECK(result == nullptr);
  CHECK(registry.make_ioctx(io::io_context_type::restful) == nullptr);
}
