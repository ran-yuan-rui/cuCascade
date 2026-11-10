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

#include <stdexcept>
#include <string>

namespace cucascade::io::s3rdma {

/**
 * @brief Failure classes reported by the S3-over-RDMA backend.
 *
 * Callers branch on the code, never on the message text.
 */
enum class errc {
  /// The request is outside what the backend serves over RDMA: it accepts
  /// device-destination slices only.
  unsupported_request = 1,
  /// The backend cannot serve the operation: its data path is not available in
  /// this build or configuration.
  backend_unavailable,
  /// The backend was closed; no further work is admitted.
  transport_closed,
  /// The request has already failed; work still pending for it is not started.
  request_canceled,
  /// An internal invariant was broken, for example work dropped without a result.
  internal_error,
};

/**
 * @brief Exception type of the S3-over-RDMA backend.
 */
class error : public std::runtime_error {
 public:
  /**
   * @param code Failure class; what callers branch on.
   * @param what Diagnostic text.
   */
  error(errc code, std::string what);

  /// The failure class.
  [[nodiscard]] errc code() const noexcept { return _code; }

 private:
  errc _code;
};

}  // namespace cucascade::io::s3rdma
