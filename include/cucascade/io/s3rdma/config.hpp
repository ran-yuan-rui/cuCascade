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

#include <cstddef>
#include <optional>

namespace cucascade::io::s3rdma {

/**
 * @brief Geometry and queueing configuration of the S3-over-RDMA reactor.
 *
 * @c max_inflight and @c arena_slot_size have no usable default: the embedder
 * sets both, and @ref sanitized rejects a configuration that leaves either at
 * zero.
 */
struct config {
  /// Number of GETs that may be outstanding at once.  Required (> 0).
  std::size_t max_inflight{0};

  /// Largest single GET, in bytes; a longer read is split into GETs of at most
  /// this size.  Required (> 0).
  std::size_t arena_slot_size{0};

  /// Bound on the number of queued logical requests.  Unset derives
  /// 4 x @c max_inflight in @ref sanitized; an explicit zero is rejected.
  std::optional<std::size_t> queue_cap{};

  /// Reported through @c ioctx::n_max_concurrent_scans(); zero opts the
  /// backend out of readahead scheduling.
  std::size_t n_max_concurrent_scans{0};

  /// An object store addresses single bytes.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }

  /// Ranges are never bridged: every byte requested lands in caller memory.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }
};

/**
 * @brief Validate @p cfg and resolve its derived fields.
 *
 * Argument checks run before the arithmetic checks.
 *
 * @return @p cfg with @c queue_cap set.
 * @throws std::invalid_argument @c max_inflight is zero, @c arena_slot_size is
 *         zero, or @c queue_cap is set to zero.
 * @throws std::overflow_error @c max_inflight x @c arena_slot_size overflows,
 *         or @c queue_cap is unset and 4 x @c max_inflight overflows (checked
 *         in that order).
 */
[[nodiscard]] config sanitized(config cfg);

}  // namespace cucascade::io::s3rdma
