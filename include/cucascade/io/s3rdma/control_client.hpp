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

#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/s3rdma/client.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

namespace cucascade::io::s3rdma {

/// Transport settings of the HTTP control client.
struct control_client_config {
  /// CA bundle for TLS verification; empty uses the system's.
  std::string ca_bundle_path;

  /// False disables peer and host verification, for development endpoints.
  bool tls_verify{true};

  /// Bound on one HEAD or LIST request, and the margin added to the lifetime
  /// of a query-signed URL.  Must be positive.
  std::chrono::seconds request_timeout{30};

  /// A range GET has no bound on its duration: it fails when it stays below
  /// @c stall_bytes_per_second for @c stall_time.  Zero in either disables
  /// the check.
  long stall_bytes_per_second{64 << 10};
  std::chrono::seconds stall_time{30};

  /// Largest LIST response body accepted, in bytes.
  std::size_t max_list_body_bytes{std::size_t{64} << 20};
};

/**
 * @brief The control client of the backend over HTTP.
 *
 * One request per call: nothing is retried and no redirect is followed.  A
 * HEAD reports a response without one usable @c Content-Length as a transport
 * error, never as an object of size zero.  A range GET writes at most the
 * requested bytes and stops reading a body that is longer.
 *
 * @param authorizer Authorizes requests to the control endpoint, in either
 *                   signing form.  @c Range and @c If-Match are sent outside
 *                   the signature.  A failure to authorize propagates as its
 *                   exception; nothing was sent.
 * @throws std::invalid_argument @p authorizer is null or
 *         @c request_timeout is not positive.
 */
[[nodiscard]] std::shared_ptr<s3_control_client> make_control_client(
  std::shared_ptr<rest::request_authorizer> authorizer, control_client_config cfg = {});

}  // namespace cucascade::io::s3rdma
