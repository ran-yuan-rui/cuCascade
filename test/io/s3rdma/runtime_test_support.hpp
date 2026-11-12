/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mock_rdma_client.hpp"
#include "test_support.hpp"

#include <cucascade/io/s3rdma/cuda_delivery.hpp>
#include <cucascade/io/s3rdma/s3rdma_ioctx.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string_view>

namespace cucascade::test::s3rdma {

using future_type = exec::semi_future<std::size_t>;

enum class operation {
  get_device,
  set_device,
  allocate,
  free,
  probe,
  flush,
  capture,
  create_event,
  copy,
  record,
  wait,
  destroy_event,
  acquire_session,
  release_session,
  register_memory,
  deregister_memory,
  capacity,
  get,
  head,
  range,
  worker_exit,
  policy,
  completion
};

struct observation {
  operation op;
  void* address{nullptr};
  std::size_t size{0};
  std::size_t offset{0};
  int device{-1};
  cudaEvent_t event{nullptr};
  cudaError_t result{cudaSuccess};
  bool returned{false};
  std::thread::id thread{std::this_thread::get_id()};
  void const* source{nullptr};
  std::size_t object_offset{0};
  cudaMemcpyKind kind{cudaMemcpyDefault};
};

class journal {
 public:
  std::function<void(observation const&)> observer;
  void append(observation value) noexcept;
  [[nodiscard]] std::vector<observation> snapshot() const;
  [[nodiscard]] std::size_t count(operation op, bool returned = false) const;
  [[nodiscard]] bool lost_record() const noexcept { return _lost.load(); }

 private:
  mutable std::mutex _mutex;
  std::vector<observation> _entries;
  std::atomic<bool> _lost{false};
};

// Release can precede entry. A forgotten release fails under the enclosing deadline.
class rendezvous {
 public:
  void arrive_and_wait();
  bool entered();
  void release() noexcept;
  [[nodiscard]] bool timed_out() const noexcept { return _timed_out.load(); }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
  std::atomic<bool> _timed_out{false};
};

struct action {
  cudaError_t result{cudaSuccess};
  bool throws{false};
  std::function<void(observation const&)> before;
  std::function<void(rdma::data_get_result&)> change_receipt;
  get_behavior behavior{get_behavior::completed};
  std::optional<rdma::head_result> head;
  std::optional<rdma::range_get_result> range;
  // When set, only a GET at this object offset consumes the step.
  std::optional<std::size_t> get_offset;
};

class scripted_client;

// Owns fake resources even when the runtime abandons its handles after a fatal result.
struct runtime_controls : std::enable_shared_from_this<runtime_controls> {
  explicit runtime_controls(bool use_cuda = false);
  ~runtime_controls();
  runtime_controls(runtime_controls const&)            = delete;
  runtime_controls& operator=(runtime_controls const&) = delete;

  void script(operation op, action step);
  action enter(observation value);
  cudaError_t leave(observation value, cudaError_t result);
  rdma::cuda_delivery_ops delivery();
  rdma::transport_clients clients();
  rdma::context_health_policy policy();
  void reclaim_sessions();
  void observe_thread_exit();

  journal log;
  std::shared_ptr<mock_rdma_client> mock{std::make_shared<mock_rdma_client>()};
  std::atomic<unsigned> policies{0};
  std::atomic<cudaError_t> policy_code{cudaSuccess};
  std::function<void(rdma::context_fatal_report const&)> policy_observer;
  rdma::write_probe_result write_probe{rdma::write_ordering::none, true};
  cudaStreamCaptureStatus capture_status{cudaStreamCaptureStatusNone};
  bool const native_cuda;
  std::thread::id const caller{std::this_thread::get_id()};

 private:
  friend class scripted_client;
  mutable std::mutex _mutex;
  std::mutex _get_mutex;
  std::map<operation, std::deque<action>> _actions;
  std::map<std::thread::id, int> _devices;
  std::map<void*, std::unique_ptr<std::uint8_t[]>> _blocks;
  std::set<void*> _cuda_blocks;
  std::set<rdma::rdma_data_session*> _sessions;
  std::uintptr_t _next_event{1};
  bool _reclaiming{false};
};

rdma::config runtime_config(std::size_t workers      = 1,
                            std::size_t slot         = 4,
                            std::vector<int> devices = {7});

struct completion_record : std::enable_shared_from_this<completion_record> {
  std::atomic<unsigned> calls{0};
  std::atomic<bool> success{false};
  std::atomic<std::size_t> fragments{0};
  // Must not throw; blocking hooks still use the fixture's bounded waits.
  std::function<void()> after;
  std::shared_ptr<io::prepared_io_completion> callback();
};

struct pending_read {
  std::shared_ptr<io::grouped_coordinator> coordinator;
  future_type future;
  std::shared_ptr<completion_record> completion;
};

// Its watchdog is constructed first and destroyed after all runtime resources.
struct runtime_fixture {
  explicit runtime_fixture(rdma::config cfg = runtime_config(), bool native = false);
  ~runtime_fixture();
  runtime_fixture(runtime_fixture const&)            = delete;
  runtime_fixture& operator=(runtime_fixture const&) = delete;
  void construct(rdma::context_health_policy policy);
  void start();
  io::prepared_io_slice slice(std::size_t offset,
                              std::size_t size,
                              std::uint8_t* dst,
                              int device          = 7,
                              cudaStream_t stream = nullptr);
  pending_read submit(std::vector<io::prepared_io_slice> slices,
                      std::shared_ptr<completion_record> completion = {});
  pending_read submit(std::size_t offset, std::size_t size, std::size_t destination = 0);
  void shutdown();

  deadline bound;
  rdma::config cfg;
  std::shared_ptr<runtime_controls> controls;
  std::vector<std::uint8_t> source;
  std::array<std::uint8_t, 128> destination{};
  std::shared_ptr<rdma::s3rdma_io_object> object;
  std::shared_ptr<rdma::s3rdma_reactor::reactor_context> context;
  std::unique_ptr<rdma::s3rdma_reactor> reactor;
};

struct host_fixture {
  host_fixture();
  ~host_fixture();
  deadline bound;
  std::shared_ptr<runtime_controls> controls{std::make_shared<runtime_controls>()};
  std::vector<std::uint8_t> source;
  std::unique_ptr<rdma::s3rdma_ioctx> ctx;
  std::shared_ptr<io::io_object> object;
  std::array<std::uint8_t, 128> destination{};
  void start();
};

inline std::exception_ptr error_from(future_type& future)
{
  REQUIRE(wait_until([&] { return future.is_ready(); }));
  try {
    (void)std::move(future).get(2s);
  } catch (...) {
    return std::current_exception();
  }
  return {};
}

inline void finish_close(rdma::admission_gate& gate, rdma::admission_gate::drain_batch batch)
{
  batch.error_complete_all(gate.terminal_error());
  if (batch.has_token()) { gate.complete_drain(std::move(batch).take_token()); }
}

}  // namespace cucascade::test::s3rdma
