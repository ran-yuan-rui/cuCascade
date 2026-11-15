/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/s3rdma/admission_gate.hpp>
#include <cucascade/io/s3rdma/client.hpp>
#include <cucascade/io/s3rdma/config.hpp>
#include <cucascade/io/s3rdma/cuda_delivery.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace cucascade::io::s3rdma {

/**
 * @brief An S3 object as the S3-over-RDMA backend addresses it.
 */
class s3rdma_io_object final : public io_object {
 public:
  /// @param size Size of the object in bytes, as established by the caller.
  s3rdma_io_object(std::string path, object_ref ref, std::size_t size);

  [[nodiscard]] object_ref const& ref() const noexcept { return _ref; }
  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  object_ref _ref;
  std::size_t _size;
};

/**
 * @brief Counters of one backend instance.  Totals only grow; peaks are
 * high-water marks.
 *
 * @c short_read_total counts only a transfer that completed and reported
 * fewer bytes than asked.  It is not the number of transfers that did not
 * complete exactly: the first of those stops the backend, and
 * @c fail_stop_total counts that stop like one for any other failure.
 * @c flush_total counts flushes that succeeded.  @c inflight_peak covers a
 * chunk from the moment a worker takes it until it is settled, not only
 * while its transfer is on the wire.
 */
struct perf_snapshot {
  std::uint64_t bytes_total{0};             ///< bytes copied to caller device memory
  std::uint64_t requests_total{0};          ///< device requests admitted
  std::uint64_t error_total{0};             ///< chunks that failed
  std::uint64_t short_read_total{0};        ///< completed transfers that reported too few bytes
  std::uint64_t retries_total{0};           ///< data-plane transfers issued again; always zero
  std::uint64_t flush_total{0};             ///< GPUDirect RDMA write flushes that succeeded
  std::uint64_t slot_wait_total{0};         ///< waits for a landing slot; always zero
  std::uint64_t inflight_peak{0};           ///< chunks being worked on at once
  std::uint64_t slots_in_use_peak{0};       ///< landing slots holding a transfer at once
  std::uint64_t arena_leak_total{0};        ///< landing arenas kept allocated at close
  std::uint64_t fail_stop_total{0};         ///< times the backend stopped after a failure
  std::uint64_t envelope_wait_total{0};     ///< submissions that waited for queue room
  std::uint64_t envelope_wait_ns_total{0};  ///< time those submissions waited
  std::uint64_t envelope_depth_peak{0};     ///< queued requests at once
};

/**
 * @brief How one backend instance ended, readable after the instance is gone.
 *
 * @c terminal and @c arena_retained can become true at any time.
 * @c torn_down is published last: once it is true the other two are final.
 */
class runtime_status {
 public:
  /// The backend stopped after a failure and admits no further work.
  [[nodiscard]] bool terminal() const noexcept { return _terminal.load(std::memory_order_acquire); }

  /// Landing memory is still allocated or registered and will stay so.
  [[nodiscard]] bool arena_retained() const noexcept
  {
    return _arena_retained.load(std::memory_order_acquire);
  }

  /// The backend finished closing, or finished undoing a failed start.
  [[nodiscard]] bool torn_down() const noexcept
  {
    return _torn_down.load(std::memory_order_acquire);
  }

 private:
  friend class s3rdma_reactor;
  std::atomic<bool> _terminal{false};
  std::atomic<bool> _arena_retained{false};
  std::atomic<bool> _torn_down{false};
};

/**
 * @brief Reactor of the S3-over-RDMA backend.
 *
 * @ref start allocates and registers one landing arena per configured device
 * and starts @c config::max_inflight workers, each with its own data session.
 * A worker takes one chunk at a time: it issues one GET into its own slot of
 * the arena, checks the response, and copies the bytes to the caller's device
 * memory on the caller's stream, waiting for that copy before it takes the
 * next chunk.
 *
 * A GET that does not complete exactly stops the backend: nothing further is
 * admitted and the landing memory is kept, because a transfer of unknown
 * outcome may still write to it.  A CUDA error that leaves the context
 * unusable stops it the same way and additionally ends every CUDA and
 * transport call of the backend; see @ref context_health_policy.
 *
 * Host reads and object open go through the control client on the calling
 * thread.
 */
class s3rdma_reactor {
  struct runtime;

 public:
  using io_object_type      = s3rdma_io_object;
  using reactor_config_type = config;

  /**
   * @brief State shared by the reactor and its I/O context: configuration,
   * clients, CUDA operations, admission gate and status.  A context serves
   * one reactor.
   */
  class reactor_context {
   public:
    /**
     * @throws std::invalid_argument a client is null, the completion rule is
     *         incomplete, a member of @p delivery is not callable, or @p cfg
     *         is rejected by @ref sanitized.
     * @throws std::overflow_error from @ref sanitized, or the storage of the
     *         admission queue overflows.
     */
    reactor_context(config cfg,
                    transport_clients clients,
                    cuda_delivery_ops delivery   = default_delivery_ops(),
                    context_health_policy policy = {});
    ~reactor_context();
    reactor_context(reactor_context const&)            = delete;
    reactor_context& operator=(reactor_context const&) = delete;

    /// The configuration after @ref sanitized: @c queue_cap is populated.
    [[nodiscard]] config const& cfg() const noexcept { return _cfg; }
    [[nodiscard]] transport_clients const& clients() const noexcept { return _clients; }
    [[nodiscard]] admission_gate& gate() noexcept { return _gate; }
    [[nodiscard]] cuda_delivery_ops const& delivery_ops() const noexcept { return _delivery; }
    [[nodiscard]] std::shared_ptr<runtime_status const> status() const noexcept { return _status; }

   private:
    friend class s3rdma_reactor;
    transport_clients _clients;
    config _cfg;
    cuda_delivery_ops _delivery;
    context_health_policy _policy;
    std::shared_ptr<runtime_status> _status;
    admission_gate _gate;
    std::unique_ptr<runtime> _runtime;
  };

  /// @throws std::invalid_argument @p ctx is null.
  /// @throws std::logic_error @p ctx already serves a reactor.
  explicit s3rdma_reactor(std::shared_ptr<reactor_context> ctx);
  /// Closes the reactor if @ref shutdown has not run.
  ~s3rdma_reactor();
  s3rdma_reactor(s3rdma_reactor const&)            = delete;
  s3rdma_reactor& operator=(s3rdma_reactor const&) = delete;

  [[nodiscard]] config const& get_config() const noexcept { return _ctx->cfg(); }

  /**
   * @brief Allocate and register the landing arenas and start the workers.
   *
   * Calls CUDA on the calling thread and then tries to restore that thread's
   * current device.  A non-sticky restore failure fails an otherwise successful
   * start and is only logged if start already failed.  A sticky restore failure
   * takes the context-fatal path and is returned as the start error.  A second
   * call after success does nothing.
   *
   * @throws std::exception the backend did not start.  Everything acquired
   *         was released again unless a release failed or a CUDA error left
   *         the context unusable; @ref status tells which.  A reactor that
   *         failed to start cannot be started again.
   */
  void start();

  /**
   * @brief Stop admitting work, wait for work in progress, release everything.
   *
   * Queued requests fail.  Work in progress includes a device request that
   * the I/O context is still handing to the reactor, with the check of the
   * CUDA context it makes when that fails.  Landing memory that may still be
   * written to, or whose release fails, is kept allocated and @ref status
   * reports it.  The calling thread's CUDA device is put back afterwards if
   * possible; a failure to do so is logged.  Idempotent; concurrent callers
   * wait for the first.  Must not be called from a worker, a completion
   * callback or the health policy.
   */
  void shutdown() noexcept;

  void interrupt() noexcept {}

  /**
   * @brief Queue @p request.  Blocks while the queue is full.
   *
   * A request that cannot be queued is failed here: with the error the
   * backend stopped with once it stopped, with @c errc::backend_unavailable
   * before a successful @ref start, and with @c errc::unsupported_request
   * when a slice is not a device read for one of the configured devices or
   * the object belongs to another backend.
   */
  void enqueue(std::unique_ptr<grouped_io_request> request) noexcept;

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  /**
   * @brief Read into host memory through the control client.
   *
   * The range is clipped to the object.  Up to three attempts are made for a
   * transport error, a 5xx status or a short body; any other failure ends the
   * read at once.
   *
   * @return Bytes read; zero for an empty or out-of-object range.
   * @throws std::invalid_argument @p destination is null for a non-empty read.
   * @throws std::exception the read failed, or the backend is stopped or not
   *         started.
   */
  std::size_t host_read(io_object_type const& object,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* destination);

  [[nodiscard]] perf_snapshot perf() const noexcept;
  [[nodiscard]] std::shared_ptr<runtime_status const> status() const noexcept
  {
    return _ctx->status();
  }

  /// @throws error with @c errc::backend_unavailable: an object is opened
  ///         through the I/O context, which has the control client.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// True for `s3://` paths.
  [[nodiscard]] static bool supports(std::string_view path) noexcept;

  /// Sorts and coalesces @p ranges.  An @p alignment of at least 1 is honored
  /// as a lower bound.  Returns no ranges if it cannot allocate.
  [[nodiscard]] static std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept;

 private:
  friend class s3rdma_ioctx;

  [[nodiscard]] runtime& state() const noexcept;
  static void mark_poisoned(void* opaque) noexcept;

  [[nodiscard]] std::exception_ptr bring_up() noexcept;
  void finish_teardown() noexcept;
  void release_arenas() noexcept;
  void worker_main(std::size_t index) noexcept;
  void process(std::size_t index,
               rdma_data_session& session,
               int& current_device,
               admission_gate::claimed_chunk claimed) noexcept;
  void context_fatal(int device, cudaError_t code, char const* what) noexcept;
  void fail_closed(std::exception_ptr fatal) noexcept;

  /// One device request on its way from the I/O context to the reactor.
  /// @ref shutdown waits until it is destroyed.
  class dispatch_scope {
   public:
    dispatch_scope(dispatch_scope&& other) noexcept;
    dispatch_scope(dispatch_scope const&)            = delete;
    dispatch_scope& operator=(dispatch_scope const&) = delete;
    dispatch_scope& operator=(dispatch_scope&&)      = delete;
    ~dispatch_scope();

   private:
    friend class s3rdma_reactor;
    dispatch_scope(admission_gate::admission_permit permit, s3rdma_reactor* owner) noexcept;
    admission_gate::admission_permit _permit;
    s3rdma_reactor* _owner;
  };

  /// Null when a device request may be queued; otherwise why not.
  [[nodiscard]] std::exception_ptr admission_error() const noexcept;
  /// The error the backend stopped with.
  [[nodiscard]] std::exception_ptr stop_error() const noexcept;
  /// Admit the I/O context to hand one device request to the reactor.  No
  /// value, with @p refusal set, when it may not.
  [[nodiscard]] std::optional<dispatch_scope> admit_dispatch(std::exception_ptr& refusal) noexcept;
  [[nodiscard]] std::shared_ptr<io_object> open(std::string path);
  void check_device_health() noexcept;

  std::shared_ptr<reactor_context> _ctx;
  std::vector<std::thread> _threads;
};

}  // namespace cucascade::io::s3rdma

namespace cucascade::io {

/// Capability profile of the S3-over-RDMA reactor: device reads only.
template <>
struct reactor_traits<s3rdma::s3rdma_reactor> {
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = false;
  static constexpr bool supports_vector_host_read    = false;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = true;
};

static_assert(io_reactor_c<s3rdma::s3rdma_reactor>);

}  // namespace cucascade::io
