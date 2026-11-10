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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cucascade::io::s3rdma {

/// Bucket and key of one object.
using object_ref = rest::object_ref;

//===----------------------------------------------------------------------===//
// Control plane (HTTP)
//===----------------------------------------------------------------------===//

/**
 * @brief Transport-level outcome of one control-plane HTTP attempt.
 *
 * The client reports; retry, validation and failure policy belong to the
 * caller.
 */
struct control_outcome {
  long http_status{0};  ///< 0 if no response status was received

  /// Empty when no transport error was observed.  A status may be present
  /// together with an error, for example when the body was cut short.
  std::string transport_error;

  /// Transport level only; says nothing about the HTTP status.
  [[nodiscard]] bool transport_ok() const noexcept { return transport_error.empty(); }
};

/// Result of @ref s3_control_client::head.
struct head_result {
  control_outcome outcome;

  /// Meaningful only for a successful HEAD that carried a usable
  /// @c Content-Length.
  std::size_t object_size{0};
};

/// Result of @ref s3_control_client::range_get.
struct range_get_result {
  control_outcome outcome;
  std::size_t delivered_bytes{0};  ///< bytes written into the caller's buffer
  std::string content_range;       ///< raw @c Content-Range response header; empty if absent
};

/**
 * @brief Host-plane operations of the backend, carried over HTTP.
 *
 * Exactly one HTTP attempt per call.  HTTP and transport outcomes are returned
 * as values.  Local failures, including allocation failures, may propagate as
 * exceptions.  Implementations are safe for concurrent calls.
 */
class s3_control_client {
 public:
  virtual ~s3_control_client() = default;

  /// Size of @p object.
  [[nodiscard]] virtual head_result head(object_ref const& object) = 0;

  /// Read @p size bytes of @p object starting at @p offset into the host
  /// buffer @p host_dst.
  [[nodiscard]] virtual range_get_result range_get(object_ref const& object,
                                                   std::size_t offset,
                                                   std::size_t size,
                                                   std::uint8_t* host_dst) = 0;
};

//===----------------------------------------------------------------------===//
// Data plane (RDMA)
//===----------------------------------------------------------------------===//

/**
 * @brief How far one data-plane GET got.
 *
 * A failure before the request left the process is provably not issued and is
 * never conflated with an ambiguous in-flight failure.
 */
enum class data_commit_state : std::uint8_t {
  /// No request left the process; the destination is untouched.
  not_sent,
  /// A request may have left the process and its outcome is unknown; the
  /// destination content is unspecified.
  sent_unknown,
  /// A response arrived and the SDK call returned.  Says nothing about success:
  /// the other fields of the result do.
  completed,
};

/**
 * @brief Everything one data-plane GET reports.
 *
 * The caller decides from all fields together whether the GET delivered.  No
 * single field is sufficient on its own, @c sdk_return included.
 */
struct data_get_result {
  data_commit_state commit{data_commit_state::not_sent};
  std::int64_t sdk_return{-1};            ///< return value of the SDK call, verbatim
  std::uint32_t callback_invocations{0};  ///< times the SDK ran the request callback for this GET
  std::size_t delivered_bytes{0};         ///< byte count the response reported for this GET
  long http_status{0};                    ///< status of the request the callback issued
  std::string reply_tag;                  ///< RDMA reply response header, raw; empty if absent
  std::string transport_error;            ///< empty when no transport error was observed
};

/**
 * @brief One data-plane session: registration plus GETs into registered memory.
 *
 * A session is used by one thread at a time and carries at most one
 * outstanding GET.  A range registered through one session is a valid
 * destination for every session acquired from the same factory until it is
 * deregistered.
 */
class rdma_data_session {
 public:
  virtual ~rdma_data_session() = default;

  /**
   * @brief Register @p bytes bytes of device memory starting at @p base.
   *
   * @throws std::exception the registration did not take effect.  An
   *         implementation that registered and then fails locally deregisters
   *         before throwing: on this path the caller frees the memory and
   *         must never free a registered range.
   */
  virtual void register_memory(std::uint8_t* base, std::size_t bytes) = 0;

  /// Undo @ref register_memory for the range that starts at @p base.
  virtual void deregister_memory(std::uint8_t* base) noexcept = 0;

  /// Largest GET the registration at @p registered_base can receive in one
  /// request, in bytes; negative when unknown.
  [[nodiscard]] virtual std::int64_t max_request_callback_size(
    std::uint8_t* registered_base) noexcept = 0;

  /**
   * @brief One GET of @p size bytes of @p object, starting at @p object_offset,
   * into registered memory at @p registered_base + @p dest_offset.
   *
   * This interface requires the exact base of the selected active registration
   * and a separate destination offset; an interior pointer must not replace
   * that base.
   *
   * With R the registered size: @p registered_base is exactly a base passed to
   * @ref register_memory, @p dest_offset <= R and @p size <= R - @p dest_offset.
   *
   * HTTP and transport outcomes are returned in the result.  A local failure
   * may propagate as an exception.
   */
  [[nodiscard]] virtual data_get_result get(object_ref const& object,
                                            std::size_t object_offset,
                                            std::size_t size,
                                            std::uint8_t* registered_base,
                                            std::size_t dest_offset) = 0;
};

/**
 * @brief Source of data-plane sessions.
 *
 * Each user acquires its own session; sessions are never shared.  Safe for
 * concurrent calls.
 */
class rdma_data_session_factory {
 public:
  virtual ~rdma_data_session_factory() = default;

  [[nodiscard]] virtual std::unique_ptr<rdma_data_session> acquire() = 0;
};

/// The clients the backend is built from.  Both are required.
struct transport_clients {
  std::shared_ptr<s3_control_client> control;
  std::shared_ptr<rdma_data_session_factory> data_sessions;
};

}  // namespace cucascade::io::s3rdma
