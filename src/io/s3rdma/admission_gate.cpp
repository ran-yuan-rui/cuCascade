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

#include <cucascade/io/s3rdma/admission_gate.hpp>
#include <cucascade/io/s3rdma/errors.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cucascade::io::s3rdma {

namespace {

using request_slot = std::unique_ptr<grouped_io_request>;

std::size_t validated_slot_bytes(std::size_t queue_cap, std::size_t slot_bytes)
{
  if (queue_cap == 0) { throw std::invalid_argument("admission_gate: queue_cap must be positive"); }
  if (slot_bytes == 0) {
    throw std::invalid_argument("admission_gate: slot_bytes must be positive");
  }
  return slot_bytes;
}

std::exception_ptr make_error(errc code, char const* what)
{
  return std::make_exception_ptr(error{code, what});
}

bool accepted(grouped_io_request const& request) noexcept
{
  auto const remaining = request.remaining_slices();
  if (remaining == 0 || request.coordinator == nullptr || request.obj == nullptr) { return false; }
  auto const first = request.slices.size() - remaining;
  return std::all_of(request.slices.begin() + static_cast<std::ptrdiff_t>(first),
                     request.slices.end(),
                     [](prepared_io_slice const& slice) {
                       return slice.has_device_request() && !slice.has_host_request() &&
                              slice.size() > 0;
                     });
}

}  // namespace

// ---- request_ring ----------------------------------------------------------

admission_gate::request_ring::request_ring(std::size_t capacity) : _capacity(capacity)
{
  if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(request_slot)) {
    throw std::overflow_error("admission_gate: queue storage size overflows");
  }
  _slots = std::make_unique<request_slot[]>(capacity);
}

admission_gate::request_ring::request_ring(request_ring&& other) noexcept
  : _slots(std::move(other._slots)),
    _capacity(std::exchange(other._capacity, 0)),
    _head(std::exchange(other._head, 0)),
    _count(std::exchange(other._count, 0))
{
}

admission_gate::request_ring& admission_gate::request_ring::operator=(request_ring&& other) noexcept
{
  if (this != &other) {
    _slots    = std::move(other._slots);
    _capacity = std::exchange(other._capacity, 0);
    _head     = std::exchange(other._head, 0);
    _count    = std::exchange(other._count, 0);
  }
  return *this;
}

grouped_io_request& admission_gate::request_ring::front() noexcept
{
  assert(_count > 0);
  return *_slots[_head];
}

void admission_gate::request_ring::push(request_slot&& request) noexcept
{
  assert(_count < _capacity);
  _slots[(_head + _count) % _capacity] = std::move(request);
  ++_count;
}

std::unique_ptr<grouped_io_request> admission_gate::request_ring::pop_front() noexcept
{
  assert(_count > 0);
  auto request = std::move(_slots[_head]);
  _head        = (_head + 1) % _capacity;
  --_count;
  return request;
}

// ---- claimed_chunk ---------------------------------------------------------

admission_gate::claimed_chunk::claimed_chunk(
  admission_gate* gate,
  rx_chunk chunk,
  std::shared_ptr<grouped_coordinator> coordinator,
  std::shared_ptr<prepared_io_completion> completion) noexcept
  : _gate(gate),
    _chunk(std::move(chunk)),
    _coordinator(std::move(coordinator)),
    _completion(std::move(completion))
{
}

admission_gate::claimed_chunk::claimed_chunk(claimed_chunk&& other) noexcept
  : _gate(std::exchange(other._gate, nullptr)),
    _chunk(std::move(other._chunk)),
    _coordinator(std::move(other._coordinator)),
    _completion(std::move(other._completion)),
    _settled(other._settled),
    _admitted(other._admitted)
{
}

admission_gate::claimed_chunk& admission_gate::claimed_chunk::operator=(
  claimed_chunk&& other) noexcept
{
  if (this != &other) {
    dispose();
    _gate        = std::exchange(other._gate, nullptr);
    _chunk       = std::move(other._chunk);
    _coordinator = std::move(other._coordinator);
    _completion  = std::move(other._completion);
    _settled     = other._settled;
    _admitted    = other._admitted;
  }
  return *this;
}

admission_gate::claimed_chunk::~claimed_chunk() { dispose(); }

void admission_gate::claimed_chunk::complete() noexcept { settle(nullptr); }

void admission_gate::claimed_chunk::report_error(std::exception_ptr error) noexcept
{
  if (_gate == nullptr) { return; }
  settle(error != nullptr ? error : _gate->_unsettled_error);
}

void admission_gate::claimed_chunk::settle(std::exception_ptr const& error) noexcept
{
  if (_gate == nullptr || _settled) { return; }
  _settled          = true;
  bool const failed = error != nullptr;
  if (_completion != nullptr) {
    (*_completion)({}, !failed && _coordinator->should_continue());
    _completion.reset();
  }
  if (failed) {
    _coordinator->report_error(error);
  } else {
    _coordinator->on_complete();
  }
}

void admission_gate::claimed_chunk::dispose() noexcept
{
  if (_gate == nullptr) { return; }
  // Settlement first: whoever is woken by the released count already sees the
  // request's outcome.
  if (!_settled) { settle(_gate->_unsettled_error); }
  auto* gate = std::exchange(_gate, nullptr);
  _coordinator.reset();
  _chunk.object.reset();
  gate->release_outstanding();
}

// ---- admission_permit ------------------------------------------------------

admission_gate::admission_permit::admission_permit(admission_permit&& other) noexcept
  : _gate(std::exchange(other._gate, nullptr))
{
}

admission_gate::admission_permit& admission_gate::admission_permit::operator=(
  admission_permit&& other) noexcept
{
  if (this != &other) {
    if (_gate != nullptr) { _gate->release_control(); }
    _gate = std::exchange(other._gate, nullptr);
  }
  return *this;
}

admission_gate::admission_permit::~admission_permit()
{
  if (_gate != nullptr) { _gate->release_control(); }
}

// ---- drain_batch -----------------------------------------------------------

admission_gate::drain_batch::drain_batch(request_ring&& ring, std::exception_ptr fallback) noexcept
  : _ring(std::move(ring)), _fallback(std::move(fallback)), _has_token(true)
{
}

admission_gate::drain_batch::drain_batch(drain_batch&& other) noexcept
  : _ring(std::move(other._ring)),
    _fallback(std::move(other._fallback)),
    _has_token(std::exchange(other._has_token, false))
{
}

admission_gate::drain_batch& admission_gate::drain_batch::operator=(drain_batch&& other) noexcept
{
  if (this != &other) {
    error_complete_all(_fallback);
    _ring      = std::move(other._ring);
    _fallback  = std::move(other._fallback);
    _has_token = std::exchange(other._has_token, false);
  }
  return *this;
}

admission_gate::drain_batch::~drain_batch() { error_complete_all(_fallback); }

void admission_gate::drain_batch::error_complete_all(std::exception_ptr error) noexcept
{
  while (!_ring.empty()) {
    auto request = _ring.pop_front();
    request->cancel_remaining(error != nullptr ? error : _fallback);
  }
}

admission_gate::drain_token admission_gate::drain_batch::take_token() && noexcept
{
  assert(_has_token && "take_token on a batch without a token");
  drain_token token;
  token._valid = std::exchange(_has_token, false);
  return token;
}

// ---- admission_gate --------------------------------------------------------

admission_gate::admission_gate(std::size_t queue_cap, std::size_t slot_bytes)
  : _slot_bytes(validated_slot_bytes(queue_cap, slot_bytes)),
    _closed_error(make_error(errc::transport_closed, "s3rdma: the transport is closed")),
    _canceled_error(make_error(errc::request_canceled, "s3rdma: the request has already failed")),
    _domain_error(make_error(errc::internal_error, "s3rdma: invalid request or chunk state")),
    _unsettled_error(
      make_error(errc::internal_error, "s3rdma: a claimed chunk was dropped without a result")),
    _queue(queue_cap)
{
}

admission_gate::~admission_gate()
{
  while (!_queue.empty()) {
    auto request = _queue.pop_front();
    request->cancel_remaining(_closed_error);
  }
}

void admission_gate::bind_arena_marker(arena_marker fn, void* ctx) noexcept
{
  std::lock_guard lock{_mutex};
  assert(_marker == nullptr && "the arena marker binds once");
  if (_marker == nullptr) {
    _marker     = fn;
    _marker_ctx = ctx;
  }
}

std::exception_ptr const& admission_gate::terminal_error_locked() const noexcept
{
  return _first_fatal != nullptr ? _first_fatal : _closed_error;
}

std::exception_ptr admission_gate::submit(std::unique_ptr<grouped_io_request>& request) noexcept
{
  if (request == nullptr || !accepted(*request)) { return _domain_error; }

  std::unique_lock lock{_mutex};
  if (_closed) { return terminal_error_locked(); }
  if (_queue.full()) {
    _envelope_wait_total.fetch_add(1, std::memory_order_relaxed);
    auto const wait_start = std::chrono::steady_clock::now();
    _submit_cv.wait(lock, [&] { return _closed || !_queue.full(); });
    _envelope_wait_ns_total.fetch_add(
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - wait_start)
                                   .count()),
      std::memory_order_relaxed);
    // A terminal wake refuses the submit without taking the request.
    if (_closed) { return terminal_error_locked(); }
  }
  _queue.push(std::move(request));
  auto const depth = static_cast<std::uint64_t>(_queue.size());
  if (depth > _envelope_depth_peak.load(std::memory_order_relaxed)) {
    _envelope_depth_peak.store(depth, std::memory_order_relaxed);
  }
  _claim_cv.notify_one();
  return nullptr;
}

std::optional<admission_gate::claimed_chunk> admission_gate::claim()
{
  // Declared before the lock so a finished request is destroyed after it.
  std::unique_ptr<grouped_io_request> finished;
  std::unique_lock lock{_mutex};
  for (;;) {
    _claim_cv.wait(lock, [&] { return _closed || !_queue.empty(); });
    if (_closed) { return std::nullopt; }

    grouped_io_request& front = _queue.front();
    if (!front.coordinator->should_continue()) {
      // The request already failed: cancel its remainder instead of handing it
      // out.  The cancellation runs outside the mutex and counts as
      // outstanding work until its settlements are published.
      auto failed = _queue.pop_front();
      ++_outstanding;
      _submit_cv.notify_one();
      lock.unlock();
      failed->cancel_remaining(_canceled_error);
      failed.reset();
      release_outstanding();
      lock.lock();
      continue;
    }

    prepared_io_slice& slice = front.front();
    rx_chunk chunk;
    chunk.object        = front.obj;
    chunk.object_offset = slice.offset();
    chunk.size          = std::min(_slot_bytes, slice.size());
    chunk.device_dst    = slice.d_buffer.data;
    chunk.stream        = slice.d_buffer.stream;
    chunk.device_id     = slice.d_buffer.device_id;

    auto coordinator = front.coordinator;
    std::shared_ptr<prepared_io_completion> completion;
    if (chunk.size < slice.size()) {
      // The slice keeps its own credit for the remainder; this chunk gets a new
      // one before it becomes visible.
      coordinator->add_tasks(1);
      slice.rng.offset += chunk.size;
      slice.rng.size -= chunk.size;
      slice.d_buffer.data += chunk.size;
    } else {
      // Last piece: the chunk inherits the slice's credit and its callback.
      auto last  = front.take_front();
      completion = std::move(last.on_complete);
      if (front.empty()) {
        finished = _queue.pop_front();
        _submit_cv.notify_one();
      }
    }
    // One wake per submit is not enough for a request of several chunks.
    if (!_queue.empty()) { _claim_cv.notify_one(); }
    ++_outstanding;
    return claimed_chunk{this, std::move(chunk), std::move(coordinator), std::move(completion)};
  }
}

void admission_gate::acquire_get(claimed_chunk& chunk)
{
  std::lock_guard lock{_mutex};
  if (_closed) { std::rethrow_exception(terminal_error_locked()); }
  if (chunk._gate != this || chunk._settled || chunk._admitted) {
    std::rethrow_exception(_domain_error);
  }
  if (!chunk._coordinator->should_continue()) { std::rethrow_exception(_canceled_error); }
  chunk._admitted = true;
}

admission_gate::admission_permit admission_gate::acquire_control()
{
  std::lock_guard lock{_mutex};
  if (_closed) { std::rethrow_exception(terminal_error_locked()); }
  ++_control_permits;
  return admission_permit{this};
}

admission_gate::creation_scope admission_gate::enter_creation()
{
  std::unique_lock lock{_mutex};
  if (_closed) { std::rethrow_exception(terminal_error_locked()); }
  return creation_scope{std::move(lock)};
}

admission_gate::drain_batch admission_gate::transition(std::exception_ptr fatal) noexcept
{
  drain_batch batch;
  std::lock_guard lock{_mutex};
  bool latched = false;
  if (fatal != nullptr && _first_fatal == nullptr) {
    _first_fatal = std::move(fatal);
    _fail_stop_total.fetch_add(1, std::memory_order_relaxed);
    latched = true;
  }
  _closed = true;
  if (!_queue_detached) {
    _queue_detached = true;
    if (!_queue.empty()) {
      // The whole queue leaves in O(1); admission is closed, so it is never
      // needed again.
      batch    = drain_batch{std::move(_queue), terminal_error_locked()};
      _drained = false;
    }
  }
  if (latched && _marker != nullptr) { _marker(_marker_ctx); }
  _submit_cv.notify_all();
  _claim_cv.notify_all();
  _closed_cv.notify_all();
  return batch;
}

admission_gate::drain_batch admission_gate::fail_stop(std::exception_ptr fatal) noexcept
{
  return transition(std::move(fatal));
}

admission_gate::drain_batch admission_gate::begin_close() noexcept { return transition(nullptr); }

void admission_gate::complete_drain(drain_token&& token) noexcept
{
  if (!std::exchange(token._valid, false)) { return; }
  std::lock_guard lock{_mutex};
  _drained = true;
  _closed_cv.notify_all();
}

void admission_gate::await_closed()
{
  std::unique_lock lock{_mutex};
  _closed_cv.wait(
    lock, [&] { return _closed && _control_permits == 0 && _outstanding == 0 && _drained; });
}

bool admission_gate::terminal() const noexcept
{
  std::lock_guard lock{_mutex};
  return _closed;
}

std::exception_ptr admission_gate::first_fatal() const noexcept
{
  std::lock_guard lock{_mutex};
  return _first_fatal;
}

std::uint64_t admission_gate::fail_stop_total() const noexcept
{
  return _fail_stop_total.load(std::memory_order_relaxed);
}

std::uint64_t admission_gate::envelope_wait_total() const noexcept
{
  return _envelope_wait_total.load(std::memory_order_relaxed);
}

std::uint64_t admission_gate::envelope_wait_ns_total() const noexcept
{
  return _envelope_wait_ns_total.load(std::memory_order_relaxed);
}

std::uint64_t admission_gate::envelope_depth_peak() const noexcept
{
  return _envelope_depth_peak.load(std::memory_order_relaxed);
}

void admission_gate::release_outstanding() noexcept
{
  std::lock_guard lock{_mutex};
  assert(_outstanding > 0);
  --_outstanding;
  _closed_cv.notify_all();
}

void admission_gate::release_control() noexcept
{
  std::lock_guard lock{_mutex};
  assert(_control_permits > 0);
  --_control_permits;
  _closed_cv.notify_all();
}

}  // namespace cucascade::io::s3rdma
