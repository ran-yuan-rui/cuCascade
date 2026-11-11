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

#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>

namespace cucascade::io::s3rdma {

/**
 * @brief One GET-sized piece of a device read, bound to its final destination.
 */
struct rx_chunk {
  std::shared_ptr<const io_object> object;
  std::size_t object_offset{0};
  std::size_t size{0};  ///< 0 < size <= the gate's slot size
  std::uint8_t* device_dst{nullptr};
  ::cuda::stream_ref stream{cudaStream_t{nullptr}};
  int device_id{-1};
};

/**
 * @brief Request intake, terminal state and close completion of the S3-over-RDMA reactor.
 *
 * One mutex is the single linearization domain for all three.  The gate owns a
 * bounded queue of logical requests; @ref claim hands out one chunk of at most
 * the slot size at a time, so the cap bounds requests, not chunks.
 *
 * Accepted input: a non-empty request whose remaining slices are all
 * device-only and non-empty.
 *
 * Credits.  A slice's own credit on its coordinator stands for its unclaimed
 * remainder.  Claiming a chunk that leaves a remainder adds one credit; the
 * chunk that takes the last piece inherits the slice's credit.  A drain or a
 * cancellation therefore settles one credit per slice that still has a
 * remainder, and every claimed chunk settles its own.
 *
 * Terminal state.  After @ref begin_close or @ref fail_stop every admission
 * call made with valid input reports the first fatal error when one was
 * latched, otherwise one stable "closed" error.  A fatal may be latched after
 * closing and then replaces the closed error for good.  A request outside the
 * accepted input is refused as such whether or not the gate is terminal.
 *
 * Allocation.  The queue storage is allocated in the constructor.  After that
 * the gate's own bookkeeping does not allocate in intake, claim and admission
 * when they succeed, nor in the transitions.  Completion callbacks, the arena
 * marker and the exception of a refusal are outside that statement.
 *
 * Lifetime.  The gate must outlive every chunk, permit, scope and batch it
 * handed out and every call into it; it joins nobody.
 */
class admission_gate {
  /// Fixed-capacity FIFO of queued requests; moves in O(1).
  class request_ring {
   public:
    request_ring() noexcept = default;
    explicit request_ring(std::size_t capacity);
    request_ring(request_ring&& other) noexcept;
    request_ring& operator=(request_ring&& other) noexcept;
    request_ring(request_ring const&)            = delete;
    request_ring& operator=(request_ring const&) = delete;
    ~request_ring()                              = default;

    [[nodiscard]] bool empty() const noexcept { return _count == 0; }
    [[nodiscard]] bool full() const noexcept { return _count == _capacity; }
    [[nodiscard]] std::size_t size() const noexcept { return _count; }
    [[nodiscard]] grouped_io_request& front() noexcept;
    void push(std::unique_ptr<grouped_io_request>&& request) noexcept;
    [[nodiscard]] std::unique_ptr<grouped_io_request> pop_front() noexcept;

   private:
    std::unique_ptr<std::unique_ptr<grouped_io_request>[]> _slots;
    std::size_t _capacity{0};
    std::size_t _head{0};
    std::size_t _count{0};
  };

 public:
  class drain_batch;

  /**
   * @brief One claimed chunk and the work it stands for.
   *
   * From @ref claim until destruction the chunk counts as outstanding work, so
   * close completion waits for it — also after it settled.  It owns one credit
   * of its request until @ref complete or @ref report_error settles it; a
   * chunk destroyed unsettled fails its request with @c errc::internal_error.
   * Settlement is always published before the outstanding count is released.
   *
   * A slice's completion callback travels with the chunk that takes its last
   * piece and runs once when that chunk settles.  It reports success when that
   * chunk completed and the request had not failed by then.  It does not mean
   * the whole slice was delivered — earlier chunks of the slice may still be
   * in flight — so only the request's future decides the outcome.
   */
  class claimed_chunk {
   public:
    claimed_chunk(claimed_chunk&& other) noexcept;
    claimed_chunk& operator=(claimed_chunk&& other) noexcept;
    claimed_chunk(claimed_chunk const&)            = delete;
    claimed_chunk& operator=(claimed_chunk const&) = delete;
    ~claimed_chunk();

    [[nodiscard]] rx_chunk const& chunk() const noexcept { return _chunk; }

    /// True once @ref admission_gate::acquire_get succeeded for this chunk.
    [[nodiscard]] bool admitted() const noexcept { return _admitted; }

    /// Settle the chunk as delivered.
    void complete() noexcept;

    /// Settle the chunk as failed; the first error of a request wins.
    void report_error(std::exception_ptr error) noexcept;

   private:
    friend class admission_gate;
    claimed_chunk(admission_gate* gate,
                  rx_chunk chunk,
                  std::shared_ptr<grouped_coordinator> coordinator,
                  std::shared_ptr<prepared_io_completion> completion) noexcept;
    void settle(std::exception_ptr const& error) noexcept;
    void dispose() noexcept;

    admission_gate* _gate{nullptr};  ///< null once moved from
    rx_chunk _chunk;
    std::shared_ptr<grouped_coordinator> _coordinator;
    std::shared_ptr<prepared_io_completion> _completion;
    bool _settled{false};
    bool _admitted{false};
  };

  /// Move-only permit covering one control-plane call.  Held permits keep close
  /// completion pending.
  class admission_permit {
   public:
    admission_permit(admission_permit&& other) noexcept;
    admission_permit& operator=(admission_permit&& other) noexcept;
    admission_permit(admission_permit const&)            = delete;
    admission_permit& operator=(admission_permit const&) = delete;
    ~admission_permit();

   private:
    friend class admission_gate;
    explicit admission_permit(admission_gate* gate) noexcept : _gate(gate) {}
    admission_gate* _gate{nullptr};
  };

  /// Move-only scope that holds the gate mutex, serializing its holder against
  /// terminal transitions.  Lock order is gate first, then the holder's own.
  class creation_scope {
   public:
    creation_scope(creation_scope&&) noexcept            = default;
    creation_scope& operator=(creation_scope&&) noexcept = default;
    creation_scope(creation_scope const&)                = delete;
    creation_scope& operator=(creation_scope const&)     = delete;
    ~creation_scope()                                    = default;

   private:
    friend class admission_gate;
    explicit creation_scope(std::unique_lock<std::mutex> held) noexcept : _held(std::move(held)) {}
    std::unique_lock<std::mutex> _held;
  };

  /// Proof that a detached queue was drained; minted only by the transition
  /// that detached it.
  class drain_token {
   public:
    drain_token(drain_token&& other) noexcept : _valid(other._valid) { other._valid = false; }
    drain_token& operator=(drain_token&& other) noexcept
    {
      _valid       = other._valid;
      other._valid = false;
      return *this;
    }
    drain_token(drain_token const&)            = delete;
    drain_token& operator=(drain_token const&) = delete;

   private:
    friend class admission_gate;
    friend class drain_batch;
    drain_token() noexcept = default;
    bool _valid{false};
  };

  /**
   * @brief The requests detached by one terminal transition.
   *
   * The receiver owes, in this order: @ref error_complete_all, then
   * @ref take_token, then @ref admission_gate::complete_drain on the gate the
   * batch came from.  Close does not complete until that token is returned.
   * The obligation moves with the batch.  A transition that found nothing to
   * detach returns an empty batch without a token.
   */
  class drain_batch {
   public:
    drain_batch() noexcept = default;
    drain_batch(drain_batch&&) noexcept;
    drain_batch& operator=(drain_batch&&) noexcept;
    drain_batch(drain_batch const&)            = delete;
    drain_batch& operator=(drain_batch const&) = delete;
    /// Requests still in the batch are failed with the terminal error of the
    /// transition that detached them.  This does not return the token.
    ~drain_batch();

    [[nodiscard]] bool empty() const noexcept { return _ring.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return _ring.size(); }
    [[nodiscard]] bool has_token() const noexcept { return _has_token; }

    /// Fail every contained request with @p error: one completion callback and
    /// one settlement per slice that still has an unclaimed remainder.
    void error_complete_all(std::exception_ptr error) noexcept;

    /// Precondition: @ref has_token.
    [[nodiscard]] drain_token take_token() && noexcept;

   private:
    friend class admission_gate;
    drain_batch(request_ring&& ring, std::exception_ptr fallback) noexcept;
    request_ring _ring;
    std::exception_ptr _fallback;
    bool _has_token{false};
  };

  /**
   * @param queue_cap  Bound on queued requests.
   * @param slot_bytes Largest chunk @ref claim hands out, in bytes.
   * @throws std::invalid_argument either argument is zero.
   * @throws std::overflow_error the queue storage size overflows.
   */
  admission_gate(std::size_t queue_cap, std::size_t slot_bytes);
  admission_gate(admission_gate const&)            = delete;
  admission_gate& operator=(admission_gate const&) = delete;
  /// Requests still queued are failed with the closed error.  Precondition: no
  /// other thread is using the gate and none of its chunks, permits or scopes
  /// is alive.
  ~admission_gate();

  /// Called once, under the gate mutex, by the transition that latches the
  /// first fatal error.  Bound at most once, before the gate is shared.
  using arena_marker = void (*)(void* ctx) noexcept;
  void bind_arena_marker(arena_marker fn, void* ctx) noexcept;

  /**
   * @brief Queue @p request.  Blocks only while the queue is full.
   *
   * @return Null on success; ownership moved into the gate.  Otherwise the
   *         error, with @p request untouched and still the caller's to
   *         settle: @c errc::internal_error for a request outside the
   *         accepted input, checked first; else the terminal error once the
   *         gate is terminal (also for a submitter woken by the transition).
   */
  [[nodiscard]] std::exception_ptr submit(std::unique_ptr<grouped_io_request>& request) noexcept;

  /// Next chunk of the oldest live request; waits for work.  Returns
  /// @c std::nullopt once the gate is terminal.  A request that has already
  /// failed is cancelled here instead of being handed out.
  [[nodiscard]] std::optional<claimed_chunk> claim();

  /**
   * @brief Permission to start the GET of @p chunk; marks it admitted.
   *
   * @p chunk must be a live chunk of this gate that is neither settled nor
   * admitted yet.
   *
   * @throws The terminal error once the gate is terminal; else
   *         @c errc::internal_error for a chunk outside that precondition;
   *         else @c errc::request_canceled when the chunk's request has
   *         already failed.  A throw leaves @p chunk exactly as it was.
   */
  void acquire_get(claimed_chunk& chunk);

  /// @throws The terminal error once the gate is terminal.
  [[nodiscard]] admission_permit acquire_control();

  /// @throws The terminal error once the gate is terminal.
  [[nodiscard]] creation_scope enter_creation();

  /// Latch @p fatal as the first fatal error (if none is latched) and close.
  [[nodiscard]] drain_batch fail_stop(std::exception_ptr fatal) noexcept;

  /// Close without waiting; a holder of a permit or a chunk may call it.
  [[nodiscard]] drain_batch begin_close() noexcept;

  /// Acknowledge a drained batch.  @p token comes from a batch of this gate,
  /// taken after the batch's requests were failed.
  void complete_drain(drain_token&& token) noexcept;

  /// Blocks until the gate is terminal, no control permit and no claimed chunk
  /// is alive, and a detached queue was drained.  It does not wait for threads
  /// that are inside or about to enter another member function.
  void await_closed();

  [[nodiscard]] bool terminal() const noexcept;
  [[nodiscard]] std::exception_ptr first_fatal() const noexcept;

  [[nodiscard]] std::uint64_t fail_stop_total() const noexcept;         ///< first-fatal latches
  [[nodiscard]] std::uint64_t envelope_wait_total() const noexcept;     ///< submits that blocked
  [[nodiscard]] std::uint64_t envelope_wait_ns_total() const noexcept;  ///< summed blocked time
  [[nodiscard]] std::uint64_t envelope_depth_peak() const noexcept;     ///< peak queued requests

 private:
  [[nodiscard]] std::exception_ptr const& terminal_error_locked() const noexcept;
  [[nodiscard]] drain_batch transition(std::exception_ptr fatal) noexcept;
  void release_outstanding() noexcept;
  void release_control() noexcept;

  std::size_t const _slot_bytes;
  std::exception_ptr const _closed_error;
  std::exception_ptr const _canceled_error;
  std::exception_ptr const _domain_error;
  std::exception_ptr const _unsettled_error;

  mutable std::mutex _mutex;
  std::condition_variable _submit_cv;  ///< submitters waiting for room
  std::condition_variable _claim_cv;   ///< claimers waiting for work
  std::condition_variable _closed_cv;  ///< await_closed waiters
  request_ring _queue;
  bool _closed{false};
  bool _queue_detached{false};
  bool _drained{true};
  std::exception_ptr _first_fatal;
  std::size_t _control_permits{0};
  std::size_t _outstanding{0};  ///< live claimed chunks and cancellations in progress
  arena_marker _marker{nullptr};
  void* _marker_ctx{nullptr};

  std::atomic<std::uint64_t> _fail_stop_total{0};
  std::atomic<std::uint64_t> _envelope_wait_total{0};
  std::atomic<std::uint64_t> _envelope_wait_ns_total{0};
  std::atomic<std::uint64_t> _envelope_depth_peak{0};
};

}  // namespace cucascade::io::s3rdma
