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

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace cucascade::io::s3rdma {

/// How a device orders GPUDirect RDMA writes against CUDA work.
enum class write_ordering : std::uint8_t {
  none,         ///< no ordering: writes must be flushed before the data is read
  owner,        ///< ordered for work on the device that owns the memory
  all_devices,  ///< ordered for work on every device
  unknown,      ///< the device reported a value this backend does not know
};

/// What @c cuda_delivery_ops::probe_writes reports for one device.
struct write_probe_result {
  write_ordering ordering{write_ordering::unknown};
  /// Whether the device can flush pending GPUDirect RDMA writes from the host.
  bool host_flush_supported{false};
};

/**
 * @brief The CUDA calls the backend makes, as replaceable callables.
 *
 * @ref default_delivery_ops binds each member to the CUDA runtime.  A caller
 * may replace individual members; every member must stay callable.  The
 * callables are invoked from the thread that starts or closes the backend and
 * from its worker threads.
 */
struct cuda_delivery_ops {
  std::function<cudaError_t(int*)> get_device;
  std::function<cudaError_t(int)> set_device;
  std::function<cudaError_t(void**, std::size_t)> malloc_device;
  std::function<cudaError_t(void*)> free_device;

  /// Report the write ordering of @p device and whether it supports a host
  /// flush.  Called once per device when the backend starts.
  std::function<cudaError_t(int device, write_probe_result* out)> probe_writes;

  /// Flush pending GPUDirect RDMA writes to the current device.  Called after
  /// a completed transfer on a device whose ordering is
  /// @c write_ordering::none, before the data is copied out.
  std::function<cudaError_t()> flush;

  std::function<cudaError_t(cudaStream_t, cudaStreamCaptureStatus*)> stream_capture_query;
  std::function<cudaError_t(cudaEvent_t*, unsigned)> event_create;
  std::function<cudaError_t(void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t)>
    memcpy_async;
  std::function<cudaError_t(cudaEvent_t, cudaStream_t)> event_record;
  std::function<cudaError_t(cudaEvent_t)> event_synchronize;
  std::function<cudaError_t(cudaEvent_t)> event_destroy;

  /**
   * @brief Called when a copy out of the landing memory was started and its
   * completion can no longer be established.
   *
   * It must not return: the landing memory may still be read by the device, so
   * neither it nor the request can be released.  @ref invoke_fatal terminates
   * the process if the hook returns or throws.  The hook must not block.
   */
  std::function<void(char const* what, cudaError_t rc)> fatal_hook;
};

/// The operations bound to the CUDA runtime.  Its @c fatal_hook writes one line
/// to standard error without blocking and returns.
[[nodiscard]] cuda_delivery_ops default_delivery_ops();

/// @throws std::invalid_argument a member of @p ops is not callable.  Calls
///         none of them.
void validate(cuda_delivery_ops const& ops);

/**
 * @brief True for the CUDA errors that leave the context unusable.
 *
 * After such an error every later call on the context fails, so nothing that
 * depends on it can be completed or released safely.
 */
[[nodiscard]] bool is_context_fatal(cudaError_t rc) noexcept;

/// Run @c ops.fatal_hook and terminate the process if it returns or throws.
[[noreturn]] void invoke_fatal(cuda_delivery_ops const& ops,
                               char const* what,
                               cudaError_t rc) noexcept;

/// What a @ref context_health_policy is told.
struct context_fatal_report {
  int device_id;     ///< device the failing call was made for; -1 if unknown
  cudaError_t code;  ///< the error for which @ref is_context_fatal holds
  char const* what;  ///< the call that failed; a string literal
};

/**
 * @brief Decides what happens to the process when the backend sees a CUDA
 * error for which @ref is_context_fatal holds.
 *
 * Called at most once per backend instance, after the backend stopped
 * admitting work and stopped calling CUDA and the transport.  Its landing
 * memory stays allocated and registered.  The policy may terminate the
 * process or return; an exception leaving it terminates the process.  It must
 * not call back into the backend.  An empty policy is not called.
 */
using context_health_policy = std::function<void(context_fatal_report const&)>;

}  // namespace cucascade::io::s3rdma
