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

#include <cucascade/io/s3rdma/client.hpp>
#include <cucascade/io/s3rdma/cuda_delivery.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cucascade::io::s3rdma::detail {

/**
 * @brief One block of device memory that data-plane GETs land in.
 *
 * The block is registered through @c registrar, which stays alive for as long
 * as the block is registered or allocated.
 */
struct arena {
  int device{-1};
  std::uint8_t* base{nullptr};
  std::size_t bytes{0};
  bool registered{false};
  bool flush_required{false};  ///< the device needs a flush before the block is read
  std::unique_ptr<rdma_data_session> registrar;
};

/// How @ref release left an arena.
enum class arena_release : std::uint8_t {
  released,       ///< deregistered and freed; the registrar is gone
  retained,       ///< a step failed or its outcome is unknown; the arena keeps what it held
  context_fatal,  ///< freeing reported an error for which @ref is_context_fatal holds
};

/**
 * @brief Deregister @p block, then free it, then drop its registrar.
 *
 * Stops at the first step that fails and leaves the rest untouched.  The
 * device of the block must be current on the calling thread.
 *
 * @param[out] code The CUDA error of a failed free; unchanged otherwise.
 */
[[nodiscard]] arena_release release(arena& block,
                                    cuda_delivery_ops const& ops,
                                    cudaError_t& code) noexcept;

}  // namespace cucascade::io::s3rdma::detail
