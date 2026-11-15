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

#include "arena.hpp"
#include "call_admission.hpp"
#include "delivery_boundary.hpp"

#include <cucascade/io/io_utils.hpp>
#include <cucascade/io/s3rdma/errors.hpp>
#include <cucascade/io/s3rdma/s3rdma_reactor.hpp>
#include <cucascade/io/uri_parser.hpp>
#include <cucascade/log/logging.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace cucascade::io::s3rdma {

//===----------------------------------------------------------------------===//
// Runtime state
//===----------------------------------------------------------------------===//

/// What a reactor and its context share once the backend runs.
struct s3rdma_reactor::runtime {
  enum class stage : std::uint8_t { created, starting, serving, failed, closing, closed };

  runtime(config const& cfg, cuda_delivery_ops const& ops, runtime_status* published)
    : status(published), calls(ops.before_call)
  {
    arenas.resize(cfg.devices.size());
    for (std::size_t i = 0; i < arenas.size(); ++i) {
      arenas[i].device = cfg.devices[i];
    }
    unavailable = std::make_exception_ptr(
      error{errc::backend_unavailable, "s3rdma: the backend is not started"});
    unsupported = std::make_exception_ptr(error{
      errc::unsupported_request, "s3rdma: only device reads for a configured device are served"});
    captured    = std::make_exception_ptr(error{
      errc::unsupported_request, "s3rdma: device reads do not support a stream that is capturing"});
    internal =
      std::make_exception_ptr(error{errc::internal_error, "s3rdma: invalid chunk or arena state"});
    stopped   = std::make_exception_ptr(error{errc::device_failed, "s3rdma: the backend stopped"});
    transport = std::make_exception_ptr(
      error{errc::transport_failed, "s3rdma: a data-plane transfer did not complete exactly"});
    device = std::make_exception_ptr(error{errc::device_failed, "s3rdma: a CUDA call failed"});
  }

  // Sessions and registered memory that may still be in use are never handed
  // back: destroying them would call the transport or CUDA.
  ~runtime()
  {
    bool const abandon = calls.stopped() || poisoned.load(std::memory_order_acquire);
    for (auto& session : sessions) {
      if (abandon) { (void)session.release(); }
    }
    for (auto& block : arenas) {
      if (abandon || block.base != nullptr || block.registered) { (void)block.registrar.release(); }
    }
  }

  runtime(runtime const&)            = delete;
  runtime& operator=(runtime const&) = delete;

  /// Run one CUDA call, unless calls may no longer start: then there is no
  /// result.  An exception counts as @c cudaErrorUnknown.
  template <typename Call>
  [[nodiscard]] std::optional<cudaError_t> cuda(backend_call call, Call&& run) noexcept
  {
    auto const entered = calls.enter(call);
    if (!entered) { return std::nullopt; }
    return detail::guarded(std::forward<Call>(run));
  }

  [[nodiscard]] detail::arena* find_arena(int device_id) noexcept
  {
    for (auto& block : arenas) {
      if (block.device == device_id) { return &block; }
    }
    return nullptr;
  }

  [[nodiscard]] std::uint64_t arenas_kept() const noexcept
  {
    return static_cast<std::uint64_t>(
      std::count_if(arenas.begin(), arenas.end(), [](detail::arena const& block) {
        return block.base != nullptr || block.registered;
      }));
  }

  runtime_status* status;

  std::mutex mutex;  ///< guards the fields up to @c go
  std::condition_variable cv;
  stage phase{stage::created};
  std::exception_ptr start_error;
  std::size_t workers_reported{0};
  std::exception_ptr worker_error;
  bool decided{false};
  bool go{false};

  std::vector<detail::arena> arenas;                         ///< one per configured device
  std::vector<std::unique_ptr<rdma_data_session>> sessions;  ///< one per worker

  std::atomic<bool> bound{false};     ///< a reactor uses this context
  std::atomic<bool> serving{false};   ///< start succeeded
  std::atomic<bool> poisoned{false};  ///< a failure was latched: the arenas stay

  /// Admits every CUDA and transport call; stopped by a CUDA error that
  /// leaves the context unusable.
  detail::call_admission calls;

  std::atomic<std::uint64_t> bytes_total{0};
  std::atomic<std::uint64_t> requests_total{0};
  std::atomic<std::uint64_t> error_total{0};
  std::atomic<std::uint64_t> short_read_total{0};
  std::atomic<std::uint64_t> retries_total{0};  ///< a GET is never issued twice
  std::atomic<std::uint64_t> flush_total{0};
  std::atomic<std::uint64_t> inflight{0};
  std::atomic<std::uint64_t> inflight_peak{0};
  std::atomic<std::uint64_t> slots_in_use{0};
  std::atomic<std::uint64_t> slots_in_use_peak{0};
  std::atomic<std::uint64_t> arena_leak_total{0};

  // Built once so that a failure can be reported without allocating.
  std::exception_ptr unavailable;
  std::exception_ptr unsupported;
  std::exception_ptr captured;
  std::exception_ptr internal;
  std::exception_ptr stopped;
  std::exception_ptr transport;
  std::exception_ptr device;
};

namespace {

using detail::guarded;

/// Counts one more while alive and records the highest count seen.
class peak_scope {
 public:
  peak_scope(std::atomic<std::uint64_t>& level, std::atomic<std::uint64_t>& peak) noexcept
    : _level(level)
  {
    auto const now = _level.fetch_add(1, std::memory_order_relaxed) + 1;
    auto seen      = peak.load(std::memory_order_relaxed);
    while (now > seen && !peak.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {}
  }
  ~peak_scope() { _level.fetch_sub(1, std::memory_order_relaxed); }
  peak_scope(peak_scope const&)            = delete;
  peak_scope& operator=(peak_scope const&) = delete;

 private:
  std::atomic<std::uint64_t>& _level;
};

transport_clients validated(transport_clients clients)
{
  if (clients.control == nullptr || clients.data_sessions == nullptr) {
    throw std::invalid_argument("s3rdma: both transport clients are required");
  }
  if (clients.rule.accepted_statuses.empty() || clients.rule.reply_ok == nullptr) {
    throw std::invalid_argument("s3rdma: the completion rule needs statuses and a reply check");
  }
  return clients;
}

cuda_delivery_ops validated(cuda_delivery_ops ops)
{
  validate(ops);
  return ops;
}

std::exception_ptr cuda_failure(char const* what,
                                cudaError_t rc,
                                std::exception_ptr const& fallback) noexcept
{
  try {
    return std::make_exception_ptr(error{
      errc::device_failed, std::string{"s3rdma: "} + what + " failed: " + cudaGetErrorName(rc)});
  } catch (...) {
    return fallback;
  }
}

std::exception_ptr make_failure(errc code,
                                char const* what,
                                std::exception_ptr const& fallback) noexcept
{
  try {
    return std::make_exception_ptr(error{code, what});
  } catch (...) {
    return fallback;
  }
}

char const* name_of(data_commit_state commit) noexcept
{
  switch (commit) {
    case data_commit_state::not_sent: return "not sent";
    case data_commit_state::sent_unknown: return "outcome unknown";
    case data_commit_state::completed: return "completed";
  }
  return "invalid";
}

// The message carries numbers only: text supplied by the transport is not
// repeated here.
std::exception_ptr get_failure(bool threw,
                               data_get_result const& result,
                               std::size_t expected,
                               std::exception_ptr const& fallback) noexcept
{
  try {
    if (threw) {
      return std::make_exception_ptr(
        error{errc::transport_failed, "s3rdma: a data-plane GET raised an error after it began"});
    }
    return std::make_exception_ptr(
      error{errc::transport_failed,
            std::string{"s3rdma: a data-plane GET did not complete exactly ("} +
              name_of(result.commit) + ", HTTP status " + std::to_string(result.http_status) +
              ", " + std::to_string(result.callback_invocations) + " request callback(s), " +
              std::to_string(result.delivered_bytes) + " of " + std::to_string(expected) +
              " bytes reported, transfer call returned " + std::to_string(result.sdk_return) +
              (result.transport_error.empty() ? "" : ", transport error reported") +
              (result.reply_tag.empty() ? ", no reply header" : "") + ")"});
  } catch (...) {
    return fallback;
  }
}

/// True when @p result is a complete transfer of exactly @p expected bytes.
/// Every condition is necessary; none is sufficient.
bool delivered(data_get_result const& result,
               std::size_t expected,
               completion_rule const& rule) noexcept
{
  if (result.commit != data_commit_state::completed || !result.transport_error.empty()) {
    return false;
  }
  if (result.callback_invocations != 1 || result.delivered_bytes != expected) { return false; }
  if (result.sdk_return < 0 || static_cast<std::uint64_t>(result.sdk_return) != expected) {
    return false;
  }
  if (result.reply_tag.empty()) { return false; }
  if (std::find(rule.accepted_statuses.begin(), rule.accepted_statuses.end(), result.http_status) ==
      rule.accepted_statuses.end()) {
    return false;
  }
  return rule.reply_ok(result.reply_tag, result.http_status);
}

/// Record why the backend stopped.  Reporting never becomes a second failure.
void log_stop(char const* event, std::exception_ptr const& cause) noexcept
{
  try {
    try {
      std::rethrow_exception(cause);
    } catch (std::exception const& failure) {
      CUCASCADE_LOG_ERROR("s3rdma: {}: {}", event, failure.what());
    }
  } catch (...) {
    // Nothing else can be done with a failure to report.
  }
}

/// Record a CUDA failure that changes nothing else.
void log_ignored(char const* event, cudaError_t rc) noexcept
{
  try {
    CUCASCADE_LOG_WARN("s3rdma: {}: {}", event, cudaGetErrorName(rc));
  } catch (...) {
    // Nothing else can be done with a failure to report.
  }
}

void close_gate(admission_gate& gate) noexcept
{
  auto batch = gate.begin_close();
  batch.error_complete_all(gate.terminal_error());
  if (batch.has_token()) { gate.complete_drain(std::move(batch).take_token()); }
}

/// Outcome of building one arena: a failed CUDA call, a call that may no
/// longer start, another failure, or none of them.
struct startup_step {
  cudaError_t code{cudaSuccess};
  char const* what{nullptr};
  std::exception_ptr failure;
  bool refused{false};
};

/// Allocate and register @p block on its device.  What was acquired before a
/// failure stays in @p block for the caller to release.
startup_step create_arena(detail::arena& block,
                          std::size_t bytes,
                          std::size_t slot,
                          cuda_delivery_ops const& ops,
                          detail::call_admission& calls,
                          rdma_data_session_factory& sessions) noexcept
{
  startup_step step;
  auto const ok = [&](backend_call call, char const* what, auto&& run) noexcept {
    auto const entered = calls.enter(call);
    if (!entered) {
      step.refused = true;
      return false;
    }
    auto const rc = guarded(run);
    if (rc != cudaSuccess) {
      step.code = rc;
      step.what = what;
    }
    return rc == cudaSuccess;
  };
  auto const device = block.device;
  if (!ok(backend_call::set_device, "cudaSetDevice", [&] { return ops.set_device(device); })) {
    return step;
  }
  write_probe_result probe;
  if (!ok(backend_call::probe_writes, "the GPUDirect RDMA write-ordering query", [&] {
        return ops.probe_writes(device, &probe);
      })) {
    return step;
  }
  try {
    switch (probe.ordering) {
      case write_ordering::owner:
      case write_ordering::all_devices: block.flush_required = false; break;
      case write_ordering::none:
        if (!probe.host_flush_supported) {
          throw error{errc::backend_unavailable,
                      "s3rdma: device " + std::to_string(device) +
                        " neither orders GPUDirect RDMA writes nor can flush them"};
        }
        block.flush_required = true;
        break;
      case write_ordering::unknown:
      default:
        throw error{errc::backend_unavailable,
                    "s3rdma: device " + std::to_string(device) +
                      " reports a GPUDirect RDMA write ordering this backend does not know"};
    }
  } catch (...) {
    step.failure = std::current_exception();
    return step;
  }
  void* memory = nullptr;
  if (!ok(backend_call::malloc_device, "cudaMalloc", [&] {
        return ops.malloc_device(&memory, bytes);
      })) {
    return step;
  }
  block.base  = static_cast<std::uint8_t*>(memory);
  block.bytes = bytes;
  // True when the transport call may start; otherwise the step is refused.
  auto const admitted = [&](detail::call_admission::scope const& entered) noexcept {
    if (!entered) { step.refused = true; }
    return static_cast<bool>(entered);
  };
  try {
    if (block.base == nullptr) {
      throw error{errc::device_failed, "s3rdma: the landing arena allocation returned no memory"};
    }
    {
      auto const entered = calls.enter(backend_call::session_acquire);
      if (!admitted(entered)) { return step; }
      block.registrar = sessions.acquire();
    }
    if (block.registrar == nullptr) {
      throw error{errc::backend_unavailable,
                  "s3rdma: the data-session factory returned no session"};
    }
    {
      auto const entered = calls.enter(backend_call::register_memory);
      if (!admitted(entered)) { return step; }
      block.registrar->register_memory(block.base, bytes);
      block.registered = true;
    }
    // Checked while still signed: a negative value means the size is unknown.
    std::int64_t capacity = -1;
    {
      auto const entered = calls.enter(backend_call::capacity_query);
      if (!admitted(entered)) { return step; }
      capacity = block.registrar->max_request_callback_size(block.base);
    }
    if (capacity < 0 || static_cast<std::uint64_t>(capacity) < slot) {
      throw error{errc::backend_unavailable,
                  "s3rdma: the registered arena cannot receive arena_slot_size bytes in one GET"};
    }
  } catch (...) {
    step.failure = std::current_exception();
  }
  return step;
}

/// True when every remaining slice of @p request is a device read this
/// backend serves.
bool device_request(grouped_io_request const& request, config const& cfg) noexcept
{
  if (dynamic_cast<s3rdma_io_object const*>(request.obj.get()) == nullptr) { return false; }
  auto const first = request.slices.size() - request.remaining_slices();
  return std::all_of(
    request.slices.begin() + static_cast<std::ptrdiff_t>(first),
    request.slices.end(),
    [&](prepared_io_slice const& slice) {
      return slice.has_device_request() && !slice.has_host_request() && slice.size() > 0 &&
             std::find(cfg.devices.begin(), cfg.devices.end(), slice.d_buffer.device_id) !=
               cfg.devices.end();
    });
}

/// Take a decimal number off the front of @p text.  No value when there is no
/// digit or the number does not fit.
std::optional<std::size_t> take_number(std::string_view& text) noexcept
{
  std::size_t number = 0;
  std::size_t digits = 0;
  while (digits < text.size() && text[digits] >= '0' && text[digits] <= '9') {
    auto const digit = static_cast<std::size_t>(text[digits] - '0');
    if (number > (std::numeric_limits<std::size_t>::max() - digit) / 10) { return std::nullopt; }
    number = number * 10 + digit;
    ++digits;
  }
  if (digits == 0) { return std::nullopt; }
  text.remove_prefix(digits);
  return number;
}

bool take(std::string_view& text, char expected) noexcept
{
  if (text.empty() || text.front() != expected) { return false; }
  text.remove_prefix(1);
  return true;
}

/**
 * First byte position of a `Content-Range: bytes <first>-<last>/<length>`
 * value, where `<length>` is a number or `*`.
 *
 * The whole value must have that form and name a possible range: first no
 * greater than last, and last below a given length.  Anything else, text
 * after the value included, yields no position.
 */
std::optional<std::size_t> content_range_start(std::string_view value) noexcept
{
  constexpr std::string_view unit{"bytes "};
  if (value.substr(0, unit.size()) != unit) { return std::nullopt; }
  value.remove_prefix(unit.size());
  auto const first = take_number(value);
  if (!first.has_value() || !take(value, '-')) { return std::nullopt; }
  auto const last = take_number(value);
  if (!last.has_value() || *last < *first || !take(value, '/')) { return std::nullopt; }
  if (value == "*") { return first; }
  auto const length = take_number(value);
  if (!length.has_value() || !value.empty() || *length <= *last) { return std::nullopt; }
  return first;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Object and context
//===----------------------------------------------------------------------===//

s3rdma_io_object::s3rdma_io_object(std::string path, object_ref ref, std::size_t size)
  : _path(std::move(path)), _ref(std::move(ref)), _size(size)
{
}

s3rdma_reactor::reactor_context::reactor_context(config cfg,
                                                 transport_clients clients,
                                                 cuda_delivery_ops delivery,
                                                 context_health_policy policy)
  : _clients(validated(std::move(clients))),
    _cfg(sanitized(std::move(cfg))),
    _delivery(validated(std::move(delivery))),
    _policy(std::move(policy)),
    _status(std::make_shared<runtime_status>()),
    _gate(*_cfg.queue_cap, _cfg.arena_slot_size),
    _runtime(std::make_unique<runtime>(_cfg, _delivery, _status.get()))
{
  _gate.bind_arena_marker(&s3rdma_reactor::mark_poisoned, _runtime.get());
}

s3rdma_reactor::reactor_context::~reactor_context() = default;

void s3rdma_reactor::mark_poisoned(void* opaque) noexcept
{
  auto* shared = static_cast<runtime*>(opaque);
  shared->poisoned.store(true, std::memory_order_release);
  shared->status->_terminal.store(true, std::memory_order_release);
}

s3rdma_reactor::s3rdma_reactor(std::shared_ptr<reactor_context> ctx) : _ctx(std::move(ctx))
{
  if (_ctx == nullptr) { throw std::invalid_argument("s3rdma_reactor: null reactor context"); }
  if (state().bound.exchange(true, std::memory_order_acq_rel)) {
    throw std::logic_error("s3rdma_reactor: the reactor context already serves a reactor");
  }
}

s3rdma_reactor::~s3rdma_reactor() { shutdown(); }

s3rdma_reactor::runtime& s3rdma_reactor::state() const noexcept { return *_ctx->_runtime; }

//===----------------------------------------------------------------------===//
// Start and close
//===----------------------------------------------------------------------===//

void s3rdma_reactor::start()
{
  auto& shared = state();
  std::exception_ptr refusal;
  {
    std::unique_lock lock{shared.mutex};
    shared.cv.wait(lock, [&] { return shared.phase != runtime::stage::starting; });
    if (shared.phase == runtime::stage::serving) { return; }
    if (shared.phase == runtime::stage::created) {
      shared.phase = runtime::stage::starting;
    } else {
      refusal = shared.start_error != nullptr ? shared.start_error : shared.unavailable;
    }
  }
  if (refusal == nullptr) { refusal = bring_up(); }
  if (refusal != nullptr) { std::rethrow_exception(refusal); }
}

std::exception_ptr s3rdma_reactor::bring_up() noexcept
{
  auto& shared    = state();
  auto& gate      = _ctx->gate();
  auto const& cfg = _ctx->cfg();
  auto const& ops = _ctx->delivery_ops();
  std::exception_ptr failure;

  // A CUDA failure of start-up: one that leaves the context unusable ends
  // every further call, any other is an ordinary start failure.
  auto const cuda_failed = [&](int device, cudaError_t rc, char const* what) noexcept {
    failure = cuda_failure(what, rc, shared.device);
    if (is_context_fatal(rc)) { context_fatal(device, rc, what); }
  };

  int previous = -1;
  if (auto const rc =
        shared.cuda(backend_call::get_device, [&] { return ops.get_device(&previous); });
      !rc.has_value()) {
    failure = stop_error();
  } else if (*rc != cudaSuccess) {
    cuda_failed(-1, *rc, "cudaGetDevice");
  }
  bool selected = false;
  for (auto& block : shared.arenas) {
    if (failure != nullptr) { break; }
    startup_step step;
    try {
      // Held while the arena is built, so that it is either complete or absent
      // when the backend stops for another reason.
      auto const scope = gate.enter_creation();
      selected         = true;
      step             = create_arena(block,
                          cfg.max_inflight * cfg.arena_slot_size,
                          cfg.arena_slot_size,
                          ops,
                          shared.calls,
                          *_ctx->clients().data_sessions);
    } catch (...) {
      step.failure = std::current_exception();
    }
    if (step.code != cudaSuccess) {
      cuda_failed(block.device, step.code, step.what);
    } else if (step.failure != nullptr) {
      failure = step.failure;
    } else if (step.refused) {
      failure = stop_error();
    }
  }
  // Restoring the caller's device is attempted only while calls may start.
  // A non-sticky restore failure preserves an earlier start error; a sticky
  // failure takes the context-fatal path.
  if (selected) {
    auto const rc = shared.cuda(backend_call::set_device, [&] { return ops.set_device(previous); });
    if (rc.has_value() && *rc != cudaSuccess) {
      if (failure == nullptr || is_context_fatal(*rc)) {
        cuda_failed(previous, *rc, "cudaSetDevice");
      } else {
        log_ignored("the CUDA device of the starting thread could not be put back", *rc);
      }
    }
  }

  if (failure == nullptr) {
    try {
      shared.sessions.resize(cfg.max_inflight);
      _threads.reserve(cfg.max_inflight);
      for (std::size_t index = 0; index < cfg.max_inflight; ++index) {
        _threads.emplace_back([this, index] { worker_main(index); });
      }
    } catch (...) {
      failure = std::current_exception();
    }
    // Every worker holds its session, or reported why not, before any of them
    // may take work.
    std::unique_lock lock{shared.mutex};
    shared.cv.wait(lock, [&] { return shared.workers_reported == _threads.size(); });
    if (failure == nullptr) { failure = shared.worker_error; }
    if (failure == nullptr) { failure = gate.terminal_error(); }
    if (failure == nullptr) {
      shared.decided = true;
      shared.go      = true;
      shared.phase   = runtime::stage::serving;
      shared.serving.store(true, std::memory_order_release);
    }
  }
  if (failure == nullptr) {
    shared.cv.notify_all();
    return nullptr;
  }

  close_gate(gate);
  finish_teardown();
  {
    std::lock_guard lock{shared.mutex};
    shared.start_error = failure;
    shared.phase       = runtime::stage::failed;
  }
  shared.cv.notify_all();
  return failure;
}

void s3rdma_reactor::shutdown() noexcept
{
  auto& shared = state();
  {
    std::unique_lock lock{shared.mutex};
    shared.cv.wait(lock, [&] {
      return shared.phase != runtime::stage::starting && shared.phase != runtime::stage::closing;
    });
    if (shared.phase == runtime::stage::closed || shared.phase == runtime::stage::failed) {
      return;
    }
    shared.phase = runtime::stage::closing;
  }
  close_gate(_ctx->gate());
  finish_teardown();
  {
    std::lock_guard lock{shared.mutex};
    shared.phase = runtime::stage::closed;
  }
  shared.cv.notify_all();
}

// The gate is closed.  Waits until nothing uses the arenas any more, then
// releases what may be released and publishes how the backend ended.
void s3rdma_reactor::finish_teardown() noexcept
{
  auto& shared = state();
  {
    std::lock_guard lock{shared.mutex};
    if (!shared.decided) {
      shared.decided = true;
      shared.go      = false;
    }
  }
  shared.cv.notify_all();

  // Every claimed chunk is settled, every control call returned and every
  // request the I/O context was dispatching is dispatched or failed, health
  // check included; then the workers are gone.  Only after both does a
  // release touch the arenas, and nothing can report a failure behind it.
  _ctx->gate().await_closed();
  for (auto& thread : _threads) {
    if (thread.joinable()) { thread.join(); }
  }
  _threads.clear();

  // After a failure the arenas may still be written to and the sessions may
  // still own a transfer: nothing is handed back.  The sessions go only once
  // every arena is released.
  auto const clean = [&]() noexcept {
    return !shared.calls.stopped() && !shared.poisoned.load(std::memory_order_acquire);
  };
  if (clean()) { release_arenas(); }
  if (clean()) {
    if (auto const entered = shared.calls.enter(backend_call::session_release)) {
      shared.sessions.clear();
    }
  }
  assert(shared.calls.in_progress() == 0 && "a call is still running behind the barrier");

  auto const kept = shared.arenas_kept();
  shared.arena_leak_total.store(kept, std::memory_order_relaxed);
  if (kept != 0) { shared.status->_arena_retained.store(true, std::memory_order_release); }
  shared.status->_torn_down.store(true, std::memory_order_release);
}

void s3rdma_reactor::release_arenas() noexcept
{
  auto& shared    = state();
  auto const& ops = _ctx->delivery_ops();
  if (shared.arenas_kept() == 0) { return; }

  // A CUDA call the release depends on.  False when it did not succeed: it
  // may no longer start, or it failed and the backend stopped for that.
  auto const succeeded = [&](int device, std::optional<cudaError_t> rc, char const* what) noexcept {
    if (rc == cudaSuccess) { return true; }
    if (rc.has_value()) {
      if (is_context_fatal(*rc)) {
        context_fatal(device, *rc, what);
      } else {
        fail_closed(cuda_failure(what, *rc, shared.device));
      }
    }
    return false;
  };
  int previous = -1;
  if (!succeeded(-1,
                 shared.cuda(backend_call::get_device, [&] { return ops.get_device(&previous); }),
                 "cudaGetDevice")) {
    return;
  }
  int current = previous;
  for (auto& block : shared.arenas) {
    if (block.base == nullptr && !block.registered) { continue; }
    if (current != block.device) {
      if (!succeeded(
            block.device,
            shared.cuda(backend_call::set_device, [&] { return ops.set_device(block.device); }),
            "cudaSetDevice")) {
        break;
      }
      current = block.device;
    }
    cudaError_t code   = cudaSuccess;
    auto const outcome = detail::release(block, ops, shared.calls, code);
    if (outcome == detail::arena_release::released) { continue; }
    // One arena that cannot be released ends the release of all of them.
    if (outcome == detail::arena_release::context_fatal) {
      context_fatal(block.device, code, "cudaFree");
    } else if (outcome == detail::arena_release::retained) {
      auto fatal = make_failure(
        errc::transport_failed, "s3rdma: a landing arena could not be released", shared.transport);
      fail_closed(fatal);
      log_stop("a landing arena is kept allocated", fatal);
    }
    break;
  }
  // Putting the caller's device back is an attempt.  Its failure undoes no
  // release; one that leaves the context unusable still stops the backend.
  if (current != previous) {
    auto const rc = shared.cuda(backend_call::set_device, [&] { return ops.set_device(previous); });
    if (rc.has_value() && *rc != cudaSuccess) {
      if (is_context_fatal(*rc)) {
        context_fatal(previous, *rc, "cudaSetDevice");
      } else {
        log_ignored("the CUDA device of the closing thread could not be put back", *rc);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Failure
//===----------------------------------------------------------------------===//

// Stop admitting work and fail what is queued.  The first call also marks the
// arenas as kept.
void s3rdma_reactor::fail_closed(std::exception_ptr fatal) noexcept
{
  auto& gate = _ctx->gate();
  auto batch = gate.fail_stop(std::move(fatal));
  batch.error_complete_all(gate.terminal_error());
  if (batch.has_token()) { gate.complete_drain(std::move(batch).take_token()); }
}

/**
 * A CUDA call reported an error that leaves the context unusable.
 *
 * The order matters.  First no CUDA or transport call of this backend may
 * start any more: stopping admission wakes the workers, and a worker that
 * leaves must not release anything.  A call that had already entered returns
 * on its own.  Then admission stops and queued requests fail.  The policy
 * runs last, before the caller reports the request that met the error, so
 * that a policy which ends the process does so before that request's waiter
 * resumes on the failed context.  Queued requests have been failed by then,
 * and other workers report theirs independently.
 */
void s3rdma_reactor::context_fatal(int device, cudaError_t code, char const* what) noexcept
{
  auto& shared     = state();
  auto fatal       = cuda_failure(what, code, shared.device);
  bool const first = shared.calls.stop();
  shared.status->_terminal.store(true, std::memory_order_release);
  fail_closed(fatal);
  if (first) {
    log_stop("a CUDA error left the context unusable; the backend stopped and keeps its memory",
             fatal);
  }
  if (first && _ctx->_policy) {
    context_fatal_report const report{device, code, what};
    try {
      _ctx->_policy(report);
    } catch (...) {
      std::terminate();
    }
  }
}

void s3rdma_reactor::check_device_health() noexcept
{
  int device    = -1;
  auto const rc = state().cuda(backend_call::get_device,
                               [&] { return _ctx->delivery_ops().get_device(&device); });
  if (rc.has_value() && *rc != cudaSuccess && is_context_fatal(*rc)) {
    context_fatal(device, *rc, "cudaGetDevice");
  }
}

std::exception_ptr s3rdma_reactor::admission_error() const noexcept
{
  if (auto terminal = _ctx->gate().terminal_error()) { return terminal; }
  if (!state().serving.load(std::memory_order_acquire)) { return state().unavailable; }
  return nullptr;
}

std::exception_ptr s3rdma_reactor::stop_error() const noexcept
{
  auto terminal = _ctx->gate().terminal_error();
  return terminal != nullptr ? terminal : state().stopped;
}

s3rdma_reactor::dispatch_scope::dispatch_scope(admission_gate::admission_permit permit,
                                               s3rdma_reactor* owner) noexcept
  : _permit(std::move(permit)), _owner(owner)
{
}

s3rdma_reactor::dispatch_scope::dispatch_scope(dispatch_scope&& other) noexcept
  : _permit(std::move(other._permit)), _owner(std::exchange(other._owner, nullptr))
{
}

s3rdma_reactor::dispatch_scope::~dispatch_scope()
{
  if (_owner != nullptr) { _owner->state().calls.leave(); }
}

/**
 * The permit makes closing wait for the dispatch, as for a control call, and
 * is refused once the backend stopped admitting work.  The dispatch is also
 * one of the backend's CUDA calls, because the framework asks CUDA for the
 * current device: it enters like any other and is refused once calls may no
 * longer start.
 */
std::optional<s3rdma_reactor::dispatch_scope> s3rdma_reactor::admit_dispatch(
  std::exception_ptr& refusal) noexcept
{
  refusal = admission_error();
  if (refusal != nullptr) { return std::nullopt; }
  try {
    auto permit = _ctx->gate().acquire_control();
    if (state().calls.try_enter(backend_call::dispatch)) {
      return dispatch_scope{std::move(permit), this};
    }
    refusal = stop_error();
  } catch (...) {
    refusal = std::current_exception();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Device path
//===----------------------------------------------------------------------===//

void s3rdma_reactor::worker_main(std::size_t index) noexcept
{
  auto& shared               = state();
  rdma_data_session* session = nullptr;
  std::exception_ptr failure;
  try {
    auto const entered = shared.calls.enter(backend_call::session_acquire);
    if (!entered) { std::rethrow_exception(stop_error()); }
    auto acquired = _ctx->clients().data_sessions->acquire();
    if (acquired == nullptr) {
      throw error{errc::backend_unavailable,
                  "s3rdma: the data-session factory returned no session"};
    }
    session                = acquired.get();
    shared.sessions[index] = std::move(acquired);
  } catch (...) {
    failure = std::current_exception();
  }
  {
    std::unique_lock lock{shared.mutex};
    ++shared.workers_reported;
    if (failure != nullptr && shared.worker_error == nullptr) { shared.worker_error = failure; }
    shared.cv.notify_all();
    shared.cv.wait(lock, [&] { return shared.decided; });
    if (!shared.go || session == nullptr) { return; }
  }

  auto& gate         = _ctx->gate();
  int current_device = -1;
  for (;;) {
    auto claimed = gate.claim();
    if (!claimed.has_value()) { return; }
    process(index, *session, current_device, std::move(*claimed));
  }
}

/**
 * One chunk: one GET into this worker's slot, then one copy to the caller.
 *
 * Everything that can fail without a transfer is done before the GET, so a
 * chunk either issues exactly one GET or none.  The slot is this worker's
 * alone and is used again only after the copy out of it completed.
 */
void s3rdma_reactor::process(std::size_t index,
                             rdma_data_session& session,
                             int& current_device,
                             admission_gate::claimed_chunk claimed) noexcept
{
  auto& shared      = state();
  auto& gate        = _ctx->gate();
  auto const& cfg   = _ctx->cfg();
  auto const& ops   = _ctx->delivery_ops();
  auto const& chunk = claimed.chunk();
  peak_scope const working{shared.inflight, shared.inflight_peak};

  auto const fail = [&](std::exception_ptr failure) noexcept {
    shared.error_total.fetch_add(1, std::memory_order_relaxed);
    claimed.report_error(std::move(failure));
  };
  // The outcome of a CUDA call made before the copy.  False when the chunk was
  // failed for it: the call may no longer start, or it reported an error.
  auto const prepared = [&](std::optional<cudaError_t> rc, char const* what) noexcept {
    if (rc == cudaSuccess) { return true; }
    if (!rc.has_value()) {
      fail(stop_error());
    } else if (is_context_fatal(*rc)) {
      context_fatal(chunk.device_id, *rc, what);
      fail(stop_error());
    } else {
      fail(cuda_failure(what, *rc, shared.device));
    }
    return false;
  };

  auto const* object = dynamic_cast<s3rdma_io_object const*>(chunk.object.get());
  auto* const block  = shared.find_arena(chunk.device_id);
  if (object == nullptr || block == nullptr || block->base == nullptr || chunk.size == 0 ||
      chunk.size > cfg.arena_slot_size || index >= cfg.max_inflight) {
    fail(shared.internal);
    return;
  }
  // index < max_inflight and size <= arena_slot_size keep the transfer inside
  // the arena of max_inflight x arena_slot_size bytes.
  auto const landing = index * cfg.arena_slot_size;

  if (current_device != chunk.device_id) {
    if (!prepared(
          shared.cuda(backend_call::set_device, [&] { return ops.set_device(chunk.device_id); }),
          "cudaSetDevice")) {
      return;
    }
    current_device = chunk.device_id;
  }

  // A capturing stream would record the copy instead of running it.
  auto capture = cudaStreamCaptureStatusNone;
  if (!prepared(shared.cuda(backend_call::stream_capture_query,
                            [&] { return ops.stream_capture_query(chunk.stream.get(), &capture); }),
                "cudaStreamIsCapturing")) {
    return;
  }
  if (capture != cudaStreamCaptureStatusNone) {
    fail(shared.captured);
    return;
  }

  cudaEvent_t event = nullptr;
  if (!prepared(shared.cuda(backend_call::event_create,
                            [&] { return ops.event_create(&event, cudaEventDisableTiming); }),
                "cudaEventCreateWithFlags")) {
    return;
  }
  // The chunk ends before its copy started, for the reason @p cause.  Its
  // event is destroyed first, and that is one more CUDA call made before the
  // copy: an error that leaves the context unusable stops the backend, any
  // other is recorded and changes nothing.  Once the backend stopped, the
  // chunk reports what it stopped with.
  auto const abandon = [&](std::exception_ptr cause) noexcept {
    auto const rc =
      shared.cuda(backend_call::event_destroy, [&] { return ops.event_destroy(event); });
    if (rc.has_value() && *rc != cudaSuccess) {
      if (is_context_fatal(*rc)) {
        context_fatal(chunk.device_id, *rc, "cudaEventDestroy");
      } else {
        log_ignored("a CUDA event could not be destroyed", *rc);
      }
    }
    fail(shared.calls.stopped() ? stop_error() : std::move(cause));
  };

  try {
    gate.acquire_get(claimed);
  } catch (...) {
    abandon(std::current_exception());
    return;
  }

  std::optional<peak_scope> slot;
  data_get_result result;
  bool threw = false;
  {
    // The permit alone does not start the GET: like every call it has to
    // enter, and it may not once calls stopped.
    auto const entered = shared.calls.enter(backend_call::data_get);
    if (!entered) {
      fail(stop_error());
      return;
    }
    slot.emplace(shared.slots_in_use, shared.slots_in_use_peak);
    try {
      result = session.get(object->ref(), chunk.object_offset, chunk.size, block->base, landing);
    } catch (...) {
      threw = true;
    }
  }
  if (threw || !delivered(result, chunk.size, _ctx->clients().rule)) {
    // The transfer may still write to the slot, so nothing can be reused.
    if (!threw && result.commit == data_commit_state::completed &&
        result.delivered_bytes < chunk.size) {
      shared.short_read_total.fetch_add(1, std::memory_order_relaxed);
    }
    auto fatal = get_failure(threw, result, chunk.size, shared.transport);
    fail_closed(fatal);
    log_stop("the backend stopped after a data-plane failure and keeps its landing memory", fatal);
    abandon(stop_error());
    return;
  }

  // The bytes arrived, but the backend stopped meanwhile: nothing more is
  // delivered from it.
  if (auto stop = gate.terminal_error(); stop != nullptr) {
    abandon(std::move(stop));
    return;
  }

  if (block->flush_required) {
    auto const rc = shared.cuda(backend_call::flush, [&] { return ops.flush(); });
    if (!rc.has_value()) {
      fail(stop_error());
      return;
    }
    if (*rc != cudaSuccess) {
      if (is_context_fatal(*rc)) {
        context_fatal(chunk.device_id, *rc, "cudaDeviceFlushGPUDirectRDMAWrites");
        fail(stop_error());
      } else {
        // The transfer completed and no copy was started: the slot is idle.
        abandon(cuda_failure("cudaDeviceFlushGPUDirectRDMAWrites", *rc, shared.device));
      }
      return;
    }
    shared.flush_total.fetch_add(1, std::memory_order_relaxed);
  }

  {
    // The copy is the last call that has to enter.  Once it has, the chunk
    // either completes or the process ends, whatever stops the backend
    // meanwhile.
    auto const entered = shared.calls.enter(backend_call::memcpy_async);
    if (!entered) {
      fail(stop_error());
      return;
    }
    detail::copy_out(
      ops, chunk.device_dst, block->base + landing, chunk.size, chunk.stream.get(), event);
  }
  slot.reset();
  if (auto const entered = shared.calls.enter(backend_call::event_destroy)) {
    detail::destroy_event(ops, event);
  }
  shared.bytes_total.fetch_add(chunk.size, std::memory_order_relaxed);
  claimed.complete();
}

//===----------------------------------------------------------------------===//
// Entries
//===----------------------------------------------------------------------===//

void s3rdma_reactor::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request == nullptr || request->coordinator == nullptr || request->empty()) { return; }
  auto& shared = state();
  auto refusal = device_request(*request, _ctx->cfg()) ? admission_error() : shared.unsupported;
  if (refusal == nullptr) {
    refusal = _ctx->gate().submit(request);
    if (refusal == nullptr) {
      shared.requests_total.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  request->cancel_remaining(refusal);
}

std::size_t s3rdma_reactor::host_read(io_object_type const& object,
                                      std::size_t offset,
                                      std::size_t size,
                                      std::uint8_t* destination)
{
  size = offset < object.size() ? std::min(size, object.size() - offset) : 0;
  if (size == 0) { return 0; }
  if (destination == nullptr) {
    throw std::invalid_argument("s3rdma: the host read destination is null");
  }
  if (auto refusal = admission_error()) { std::rethrow_exception(refusal); }

  constexpr int max_attempts = 3;
  constexpr std::chrono::milliseconds backoff{5};
  auto& gate           = _ctx->gate();
  auto& control        = *_ctx->clients().control;
  bool const whole     = offset == 0 && size == object.size();
  char const* last_try = "";
  for (int attempt = 1;; ++attempt) {
    range_get_result result;
    {
      // One permit per attempt: a close between two attempts ends the read.
      auto const permit  = gate.acquire_control();
      auto const entered = state().calls.enter(backend_call::control_range_get);
      if (!entered) { std::rethrow_exception(stop_error()); }
      result = control.range_get(object.ref(), offset, size, destination);
    }
    auto const status = result.outcome.http_status;
    auto const reject = [&](char const* why) {
      throw error{errc::control_failed,
                  std::string{"s3rdma: host read failed: "} + why + " (HTTP status " +
                    std::to_string(status) + ")"};
    };
    if (status >= 400 && status < 500) {
      reject(status == 416 ? "the server reports the range as outside the object"
                           : "the request was refused");
    }
    if (!result.outcome.transport_ok()) {
      last_try = "transport error";
    } else if (status >= 500) {
      last_try = "server error";
    } else if (status == 206) {
      auto const first = content_range_start(result.content_range);
      if (!first.has_value() || *first != offset) {
        reject("the response does not identify the requested range");
      }
      if (result.delivered_bytes == size) { return size; }
      if (result.delivered_bytes > size) { reject("the response is longer than the range"); }
      last_try = "short response";
    } else if (status == 200) {
      if (!whole) { reject("the server returned the whole object for a range request"); }
      if (result.delivered_bytes == size) { return size; }
      if (result.delivered_bytes > size) { reject("the response is longer than the object"); }
      last_try = "short response";
    } else {
      reject("unexpected status");
    }
    if (attempt == max_attempts) {
      throw error{errc::control_failed,
                  std::string{"s3rdma: host read failed after "} + std::to_string(max_attempts) +
                    " attempts: " + last_try + " (HTTP status " + std::to_string(status) + ")"};
    }
    std::this_thread::sleep_for(backoff * attempt);
  }
}

std::shared_ptr<io_object> s3rdma_reactor::open(std::string path)
{
  auto parsed = io::parse(path);
  if (parsed.scheme != "s3") {
    throw error{errc::unsupported_request, "s3rdma: only s3:// paths are served"};
  }
  if (auto refusal = admission_error()) { std::rethrow_exception(refusal); }
  object_ref ref{std::move(parsed.host), std::move(parsed.path)};
  head_result result;
  {
    auto const permit  = _ctx->gate().acquire_control();
    auto const entered = state().calls.enter(backend_call::control_head);
    if (!entered) { std::rethrow_exception(stop_error()); }
    result = _ctx->clients().control->head(ref);
  }
  if (!result.outcome.transport_ok() || result.outcome.http_status != 200) {
    throw error{errc::control_failed,
                std::string{"s3rdma: the object could not be opened ("} +
                  (result.outcome.transport_ok() ? "" : "transport error, ") + "HTTP status " +
                  std::to_string(result.outcome.http_status) + ")"};
  }
  return std::make_shared<s3rdma_io_object>(std::move(path), std::move(ref), result.object_size);
}

perf_snapshot s3rdma_reactor::perf() const noexcept
{
  auto const& shared = state();
  auto const& gate   = _ctx->gate();
  auto const read    = [](std::atomic<std::uint64_t> const& counter) noexcept {
    return counter.load(std::memory_order_relaxed);
  };
  perf_snapshot snapshot;
  snapshot.bytes_total            = read(shared.bytes_total);
  snapshot.requests_total         = read(shared.requests_total);
  snapshot.error_total            = read(shared.error_total);
  snapshot.short_read_total       = read(shared.short_read_total);
  snapshot.retries_total          = read(shared.retries_total);
  snapshot.flush_total            = read(shared.flush_total);
  snapshot.inflight_peak          = read(shared.inflight_peak);
  snapshot.slots_in_use_peak      = read(shared.slots_in_use_peak);
  snapshot.arena_leak_total       = read(shared.arena_leak_total);
  snapshot.fail_stop_total        = gate.fail_stop_total();
  snapshot.envelope_wait_total    = gate.envelope_wait_total();
  snapshot.envelope_wait_ns_total = gate.envelope_wait_ns_total();
  snapshot.envelope_depth_peak    = gate.envelope_depth_peak();
  return snapshot;
}

std::unique_ptr<s3rdma_reactor::io_object_type> s3rdma_reactor::create_io_object(
  std::string /*path*/)
{
  throw error{errc::backend_unavailable,
              "s3rdma: an object is opened through the I/O context, not the reactor"};
}

bool s3rdma_reactor::supports(std::string_view path) noexcept
{
  try {
    return io::parse(path).scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::vector<byte_range> s3rdma_reactor::align_and_coalesce(
  std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept
{
  try {
    return io_utils::align_and_coalesce(ranges, std::max<std::size_t>(alignment.value_or(1), 1));
  } catch (...) {
    return {};
  }
}

}  // namespace cucascade::io::s3rdma
