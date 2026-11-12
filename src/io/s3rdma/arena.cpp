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

namespace cucascade::io::s3rdma::detail {

arena_release release(arena& block, cuda_delivery_ops const& ops, cudaError_t& code) noexcept
{
  if (block.registered) {
    if (block.registrar == nullptr || !block.registrar->deregister_memory(block.base)) {
      return arena_release::retained;
    }
    block.registered = false;
  }
  if (block.base != nullptr) {
    cudaError_t rc = cudaErrorUnknown;
    try {
      rc = ops.free_device(block.base);
    } catch (...) {
      rc = cudaErrorUnknown;
    }
    if (rc != cudaSuccess) {
      code = rc;
      return is_context_fatal(rc) ? arena_release::context_fatal : arena_release::retained;
    }
    block.base = nullptr;
  }
  block.registrar.reset();
  return arena_release::released;
}

}  // namespace cucascade::io::s3rdma::detail
