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

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cucascade::io::s3rdma::detail {

/// Run a CUDA operation that may throw; an exception counts as
/// @c cudaErrorUnknown.
template <typename Call>
[[nodiscard]] cudaError_t guarded(Call&& call) noexcept
{
  try {
    return std::forward<Call>(call)();
  } catch (...) {
    return cudaErrorUnknown;
  }
}

/// Destroy @p event.  A failure is logged and otherwise ignored: by now the
/// work the event stood for is finished.
void destroy_event(cuda_delivery_ops const& ops, cudaEvent_t event) noexcept;

/**
 * @brief Copy @p bytes bytes from the landing memory at @p landing to
 * @p destination on @p stream and wait for the copy.
 *
 * Returns only after the copy completed.  Once the copy is submitted the
 * device reads the landing memory until @p event completes.  If submitting,
 * recording or waiting fails, there is no way left to learn when that is, so
 * the landing memory could be overwritten, or the request released, under a
 * running copy.  Such a failure ends in @ref invoke_fatal and does not return.
 */
void copy_out(cuda_delivery_ops const& ops,
              std::uint8_t* destination,
              std::uint8_t const* landing,
              std::size_t bytes,
              cudaStream_t stream,
              cudaEvent_t event) noexcept;

}  // namespace cucascade::io::s3rdma::detail
