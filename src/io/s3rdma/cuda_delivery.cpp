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

#include <cucascade/io/s3rdma/cuda_delivery.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace cucascade::io::s3rdma {

namespace {

constexpr auto context_fatal_codes = std::to_array<cudaError_t>({
  cudaErrorECCUncorrectable,
  cudaErrorNvlinkUncorrectable,
#if CUDART_VERSION >= 12080
  cudaErrorContained,
#endif
  cudaErrorIllegalAddress,
  cudaErrorLaunchTimeout,
  cudaErrorAssert,
  cudaErrorHardwareStackError,
  cudaErrorIllegalInstruction,
  cudaErrorMisalignedAddress,
  cudaErrorInvalidAddressSpace,
  cudaErrorInvalidPc,
  cudaErrorLaunchFailure,
#if CUDART_VERSION >= 12080
  cudaErrorTensorMemoryLeak,
#endif
  cudaErrorMpsClientTerminated,
  cudaErrorExternalDevice,
});

cudaError_t probe_device_writes(int device, write_probe_result* out)
{
  int ordering = 0;
  if (auto const rc =
        cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device);
      rc != cudaSuccess) {
    return rc;
  }
  int flush_options = 0;
  if (auto const rc =
        cudaDeviceGetAttribute(&flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device);
      rc != cudaSuccess) {
    return rc;
  }
  switch (ordering) {
    case cudaGPUDirectRDMAWritesOrderingNone: out->ordering = write_ordering::none; break;
    case cudaGPUDirectRDMAWritesOrderingOwner: out->ordering = write_ordering::owner; break;
    case cudaGPUDirectRDMAWritesOrderingAllDevices:
      out->ordering = write_ordering::all_devices;
      break;
    default: out->ordering = write_ordering::unknown; break;
  }
  out->host_flush_supported = (static_cast<unsigned>(flush_options) &
                               static_cast<unsigned>(cudaFlushGPUDirectRDMAWritesOptionHost)) != 0U;
  return cudaSuccess;
}

// One write of a bounded line.  Standard error is switched to non-blocking for
// the write so that a reader that stopped draining it cannot delay the caller.
void report_fatal(char const* what, cudaError_t rc) noexcept
{
  std::array<char, 256> line{};
  std::size_t used  = 0;
  auto const append = [&](char const* text) {
    if (text == nullptr) { return; }
    auto const length = std::min(std::strlen(text), line.size() - used);
    std::memcpy(line.data() + used, text, length);
    used += length;
  };
  append("s3rdma: fatal CUDA failure while delivering data: ");
  append(what);
  append(": ");
  append(cudaGetErrorName(rc));
  append("\n");
  int const flags = ::fcntl(STDERR_FILENO, F_GETFL);
  if (flags != -1) { (void)::fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK); }
  [[maybe_unused]] auto const written = ::write(STDERR_FILENO, line.data(), used);
  if (flags != -1) { (void)::fcntl(STDERR_FILENO, F_SETFL, flags); }
}

}  // namespace

cuda_delivery_ops default_delivery_ops()
{
  cuda_delivery_ops ops;
  ops.get_device    = [](int* device) { return cudaGetDevice(device); };
  ops.set_device    = [](int device) { return cudaSetDevice(device); };
  ops.malloc_device = [](void** pointer, std::size_t bytes) { return cudaMalloc(pointer, bytes); };
  ops.free_device   = [](void* pointer) { return cudaFree(pointer); };
  ops.probe_writes  = &probe_device_writes;
  ops.flush         = [] {
    return cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                              cudaFlushGPUDirectRDMAWritesToOwner);
  };
  ops.stream_capture_query = [](cudaStream_t stream, cudaStreamCaptureStatus* status) {
    return cudaStreamIsCapturing(stream, status);
  };
  ops.event_create = [](cudaEvent_t* event, unsigned flags) {
    return cudaEventCreateWithFlags(event, flags);
  };
  ops.memcpy_async =
    [](void* dst, void const* src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
      return cudaMemcpyAsync(dst, src, bytes, kind, stream);
    };
  ops.event_record = [](cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord(event, stream);
  };
  ops.event_synchronize = [](cudaEvent_t event) { return cudaEventSynchronize(event); };
  ops.event_destroy     = [](cudaEvent_t event) { return cudaEventDestroy(event); };
  ops.fatal_hook        = &report_fatal;
  return ops;
}

void validate(cuda_delivery_ops const& ops)
{
  bool const complete = ops.get_device && ops.set_device && ops.malloc_device && ops.free_device &&
                        ops.probe_writes && ops.flush && ops.stream_capture_query &&
                        ops.event_create && ops.memcpy_async && ops.event_record &&
                        ops.event_synchronize && ops.event_destroy && ops.fatal_hook;
  if (!complete) {
    throw std::invalid_argument("s3rdma: every CUDA delivery operation must be callable");
  }
}

bool is_context_fatal(cudaError_t rc) noexcept
{
  for (auto const code : context_fatal_codes) {
    if (rc == code) { return true; }
  }
  return false;
}

void invoke_fatal(cuda_delivery_ops const& ops, char const* what, cudaError_t rc) noexcept
{
  try {
    if (ops.fatal_hook) { ops.fatal_hook(what, rc); }
  } catch (...) {
    // A hook that throws ends the same way as one that returns.
  }
  std::terminate();
}

}  // namespace cucascade::io::s3rdma
