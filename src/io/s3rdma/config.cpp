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

#include <cucascade/io/s3rdma/config.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cucascade::io::s3rdma {

config sanitized(config cfg)
{
  if (cfg.max_inflight == 0) {
    throw std::invalid_argument("s3rdma: max_inflight must be positive");
  }
  if (cfg.arena_slot_size == 0) {
    throw std::invalid_argument("s3rdma: arena_slot_size must be positive");
  }
  if (cfg.queue_cap.has_value() && *cfg.queue_cap == 0) {
    throw std::invalid_argument("s3rdma: queue_cap must be positive when set");
  }

  constexpr auto max_size = std::numeric_limits<std::size_t>::max();
  if (cfg.max_inflight > max_size / cfg.arena_slot_size) {
    throw std::overflow_error("s3rdma: max_inflight x arena_slot_size overflows");
  }
  if (!cfg.queue_cap.has_value()) {
    if (cfg.max_inflight > max_size / 4) {
      throw std::overflow_error("s3rdma: the derived queue_cap (4 x max_inflight) overflows");
    }
    cfg.queue_cap = 4 * cfg.max_inflight;
  }
  return cfg;
}

}  // namespace cucascade::io::s3rdma
