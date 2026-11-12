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

#include "delivery_boundary.hpp"

#include <cucascade/log/logging.hpp>

#include <cuda_runtime.h>

namespace cucascade::io::s3rdma::detail {

void destroy_event(cuda_delivery_ops const& ops, cudaEvent_t event) noexcept
{
  auto const rc = guarded([&] { return ops.event_destroy(event); });
  if (rc == cudaSuccess) { return; }
  try {
    CUCASCADE_LOG_WARN("s3rdma: a CUDA event could not be destroyed: {}", cudaGetErrorName(rc));
  } catch (...) {
    // A failing log sink does not turn finished work into a failure.
  }
}

// Nothing between a failed step and invoke_fatal may throw or allocate: the
// failure is passed on as the call's name and its error code only.
void copy_out(cuda_delivery_ops const& ops,
              std::uint8_t* destination,
              std::uint8_t const* landing,
              std::size_t bytes,
              cudaStream_t stream,
              cudaEvent_t event) noexcept
{
  cudaError_t rc = cudaErrorUnknown;
  try {
    rc = ops.memcpy_async(destination, landing, bytes, cudaMemcpyDeviceToDevice, stream);
  } catch (...) {
    invoke_fatal(ops, "cudaMemcpyAsync", cudaErrorUnknown);
  }
  if (rc != cudaSuccess) { invoke_fatal(ops, "cudaMemcpyAsync", rc); }
  try {
    rc = ops.event_record(event, stream);
  } catch (...) {
    invoke_fatal(ops, "cudaEventRecord", cudaErrorUnknown);
  }
  if (rc != cudaSuccess) { invoke_fatal(ops, "cudaEventRecord", rc); }
  try {
    rc = ops.event_synchronize(event);
  } catch (...) {
    invoke_fatal(ops, "cudaEventSynchronize", cudaErrorUnknown);
  }
  if (rc != cudaSuccess) { invoke_fatal(ops, "cudaEventSynchronize", rc); }
}

}  // namespace cucascade::io::s3rdma::detail
