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
#include <cucascade/io/s3rdma/control_client.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace cucascade::io::s3rdma {

/**
 * @brief Compare two endpoints by their text.
 *
 * Scheme and host are lowercased, a missing port is filled in from the
 * scheme, and one trailing slash is dropped.  Nothing is resolved.  Only a
 * pair that is equal after that is @c same_address: two names of one service
 * stay @c split, which keeps the stricter rules of a split deployment.
 */
[[nodiscard]] endpoint_topology detect_endpoint_topology(std::string_view control_endpoint,
                                                         std::string_view data_endpoint) noexcept;

/// The data plane of a deployment.  Nothing here has a usable default except
/// where stated.
struct data_plane_config {
  /// Absolute path of the binding library (see cuobj_abi.h).  Required: no
  /// search path, working directory or environment variable is consulted.
  std::string library_path;

  /// CA bundle for the request each GET sends; empty uses the system's.
  std::string ca_bundle_path;

  /// False disables peer and host verification, for development endpoints.
  bool tls_verify{true};

  /// Bound on the request one GET sends, from connect to the end of the
  /// response.  Must be positive.
  std::chrono::milliseconds request_deadline{std::chrono::seconds{60}};

  /// Response header that states how many bytes the gateway transferred;
  /// a GET delivers only when it appears once and equals the requested count.
  /// When empty, a response the completion rule accepts stands for the
  /// requested count.
  std::string bytes_header;

  /// The RDMA client repeats a request callback on its own unless its I/O
  /// retry count is zero.  The backend refuses every repeated callback
  /// itself, so a count of zero is recommended rather than needed: by default
  /// another count, or an unknown one, is only logged.  True refuses such a
  /// binding instead.
  bool require_zero_io_retry{false};
};

/**
 * @brief Data sessions over the binding library of @p cfg.
 *
 * Loads the configured library, checks its interface version and its
 * protocol, and returns a factory whose sessions transfer through it.  An
 * interface that passes the structural checks is kept by path and not looked
 * up again; the protocol and retry probe runs for every factory.  A library
 * that was loaded is never unloaded.  Each GET sends exactly one request,
 * signed by @p authorizer with the RDMA token, the range and any condition
 * inside the signature, to the endpoint @p authorizer was built for.  That
 * request follows no redirect, uses no proxy and is never repeated.
 *
 * @param authorizer Must sign additional headers; a query-signing authorizer
 *                   is refused here.
 * @param rule       What the response of a delivered GET carries; checked
 *                   before a byte count is reported to the client.
 * @throws error with @c errc::backend_unavailable the library is missing,
 *         cannot be loaded, has another interface version or protocol, fails
 *         its probe, or @p authorizer cannot sign headers.  No session was
 *         created.
 * @throws std::invalid_argument an argument is null, incomplete or not an
 *         absolute path.
 */
[[nodiscard]] std::shared_ptr<rdma_data_session_factory> make_data_session_factory(
  std::shared_ptr<rest::request_authorizer> authorizer,
  completion_rule rule,
  data_plane_config cfg);

/// Both planes of one deployment.
struct deployment {
  /// `scheme://host[:port]` of the S3 endpoint that serves HEAD, host reads
  /// and listing, and of the RDMA gateway.  Both are required; they are used
  /// here to tell the topology, the authorizers carry them for the requests.
  std::string control_endpoint;
  std::string data_endpoint;

  /// Signs control requests, in either form.
  std::shared_ptr<rest::request_authorizer> control_authorizer;
  /// Signs the request of each GET; must sign additional headers.
  std::shared_ptr<rest::request_authorizer> data_authorizer;
  /// The data plane has credentials of its own.
  bool distinct_data_credentials{false};

  control_client_config control;
  data_plane_config data;
  completion_rule rule;
};

/**
 * @brief The clients of @p where, ready for @c s3rdma_ioctx.
 *
 * Builds the control client and the data session factory, judges the
 * topology once with @ref detect_endpoint_topology, and logs one line that
 * names the consistency rule the deployment is under: a split deployment
 * relies on its publisher to make an object visible at both endpoints before
 * it is read; one address relies on the store itself.  One address with
 * distinct data credentials is logged as a warning.
 *
 * @throws std::invalid_argument an endpoint is empty, or as
 *         @ref make_control_client and @ref make_data_session_factory.
 * @throws error as @ref make_data_session_factory.
 */
[[nodiscard]] transport_clients make_transport_clients(deployment where);

}  // namespace cucascade::io::s3rdma
