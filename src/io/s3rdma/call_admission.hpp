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

#include <cucascade/io/s3rdma/cuda_delivery.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <utility>

namespace cucascade::io::s3rdma::detail {

/**
 * @brief Decides whether a CUDA or transport call of the backend may start.
 *
 * Every such call starts through @ref enter, and @ref stop ends admission for
 * good.  Both change the same atomic word, which orders them: a call either
 * entered before the stop, and then runs to its return, or it is refused.
 * Having passed an earlier check, such as a permit of the admission gate or a
 * previous call, does not count as having entered.
 *
 * No lock is held while a call runs, so a stop never waits for one.
 */
class call_admission {
 public:
  /// One entered call, left again on destruction; empty when the call was
  /// refused.
  class scope {
   public:
    scope() noexcept = default;
    scope(scope&& other) noexcept : _owner(std::exchange(other._owner, nullptr)) {}
    scope(scope const&)            = delete;
    scope& operator=(scope const&) = delete;
    scope& operator=(scope&&)      = delete;
    ~scope()
    {
      if (_owner != nullptr) { _owner->leave(); }
    }

    explicit operator bool() const noexcept { return _owner != nullptr; }

   private:
    friend class call_admission;
    explicit scope(call_admission* owner) noexcept : _owner(owner) {}
    call_admission* _owner{nullptr};
  };

  /// @param observer Told about each call before its admission is decided; may
  ///                 be empty.  It must outlive this object.
  explicit call_admission(std::function<void(backend_call)> const& observer) noexcept
    : _observer(&observer)
  {
  }
  call_admission(call_admission const&)            = delete;
  call_admission& operator=(call_admission const&) = delete;

  /// Tell the observer about @p call, then enter it unless admission stopped.
  [[nodiscard]] scope enter(backend_call call) noexcept
  {
    return try_enter(call) ? scope{this} : scope{};
  }

  /// As @ref enter, for a caller that keeps the call entered beyond one
  /// scope: when it returns true the caller owes one @ref leave.
  [[nodiscard]] bool try_enter(backend_call call) noexcept
  {
    announce(call);
    auto seen = _word.load(std::memory_order_acquire);
    do {
      if ((seen & stopped_bit) != 0) { return false; }
    } while (!_word.compare_exchange_weak(
      seen, seen + one_call, std::memory_order_acq_rel, std::memory_order_acquire));
    return true;
  }

  void leave() noexcept { _word.fetch_sub(one_call, std::memory_order_acq_rel); }

  /// End admission.  True for the caller that ended it, false for every later
  /// one.
  bool stop() noexcept
  {
    return (_word.fetch_or(stopped_bit, std::memory_order_acq_rel) & stopped_bit) == 0;
  }

  [[nodiscard]] bool stopped() const noexcept
  {
    return (_word.load(std::memory_order_acquire) & stopped_bit) != 0;
  }

  /// Calls that entered and have not returned.
  [[nodiscard]] std::uint64_t in_progress() const noexcept
  {
    return _word.load(std::memory_order_acquire) / one_call;
  }

 private:
  void announce(backend_call call) const noexcept
  {
    if (!*_observer) { return; }
    try {
      (*_observer)(call);
    } catch (...) {
      // An observer has no say in the decision.
    }
  }

  static constexpr std::uint64_t stopped_bit = 1;
  static constexpr std::uint64_t one_call    = 2;

  std::function<void(backend_call)> const* _observer;
  std::atomic<std::uint64_t> _word{0};
};

}  // namespace cucascade::io::s3rdma::detail
