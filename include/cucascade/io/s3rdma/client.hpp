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
#include <cucascade/io/s3rdma/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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

/// True when @p tag is one strong entity-tag (RFC 9110, 8.8.3): a quoted
/// string of visible characters, without the weak prefix.
[[nodiscard]] inline bool is_strong_validator(std::string_view tag) noexcept
{
  if (tag.size() < 2 || tag.front() != '"' || tag.back() != '"') { return false; }
  for (char const ch : tag.substr(1, tag.size() - 2)) {
    auto const c = static_cast<unsigned char>(ch);
    if (c == '"' || c < 0x21 || c == 0x7f) { return false; }
  }
  return true;
}

/// Result of @ref s3_control_client::head.
struct head_result {
  control_outcome outcome;

  /// Meaningful only for a successful HEAD that carried a usable
  /// @c Content-Length.  A response without one is reported through
  /// @c outcome.transport_error, never as a size of zero.
  std::size_t object_size{0};

  /// The response's @c ETag as received, quotes and a weak prefix included;
  /// empty if absent.  Only a tag for which @ref is_strong_validator holds
  /// conditions later reads of the object.
  std::string validation_tag{};
};

/// Result of @ref s3_control_client::range_get.
struct range_get_result {
  control_outcome outcome;

  /// Body bytes of a 200 or 206 response.  At most the requested number is
  /// written to the caller's buffer; one more than requested reports a body
  /// that is longer, which the client stops reading.
  std::size_t delivered_bytes{0};

  std::string content_range;     ///< raw @c Content-Range response header; empty if absent
  std::string validation_tag{};  ///< raw @c ETag response header; empty if absent
};

/// Result of @ref s3_control_client::list_page.
struct list_page_result {
  control_outcome outcome;
  std::string body;  ///< the response body; meaningful for a successful request
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

  /**
   * @brief @ref range_get on the condition that the object still carries the
   * validator @p if_match (sent as @c If-Match).
   *
   * The default serves an empty @p if_match through @ref range_get and
   * refuses any other: a client that cannot send the condition must not read
   * as if it had.
   *
   * @throws error with @c errc::backend_unavailable from the default, for a
   *         non-empty @p if_match.
   */
  [[nodiscard]] virtual range_get_result range_get_if_match(object_ref const& object,
                                                            std::size_t offset,
                                                            std::size_t size,
                                                            std::uint8_t* host_dst,
                                                            std::string_view if_match)
  {
    if (if_match.empty()) { return range_get(object, offset, size, host_dst); }
    throw error{errc::backend_unavailable,
                "s3rdma: this control client cannot make a conditional read"};
  }

  /**
   * @brief One ListObjectsV2 page of @p bucket.
   *
   * @param canonical_query The request query, percent-encoded and in SigV4
   *                        canonical order, without authorization parameters.
   *
   * @throws error with @c errc::backend_unavailable from the default: listing
   *         is optional for a client.
   */
  [[nodiscard]] virtual list_page_result list_page(std::string_view /*bucket*/,
                                                   std::string_view /*canonical_query*/)
  {
    throw error{errc::backend_unavailable, "s3rdma: this control client does not list"};
  }
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
 *
 * @c sdk_return carries the SDK's value unchanged.  The backend reads it as a
 * byte count: a GET delivered only when it equals the requested size and
 * every other field agrees.
 */
struct data_get_result {
  data_commit_state commit{data_commit_state::not_sent};
  std::int64_t sdk_return{-1};            ///< return value of the SDK call, verbatim
  std::uint32_t callback_invocations{0};  ///< times the SDK ran the request callback for this GET
  std::size_t delivered_bytes{0};         ///< byte count the response reported for this GET
  long http_status{0};                    ///< status of the request the callback issued
  std::string reply_tag;                  ///< RDMA reply response header, raw; empty if absent
  std::string transport_error;            ///< empty when no transport error was observed
  std::string validation_tag{};           ///< @c ETag response header, raw; empty if absent
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

  /**
   * @brief Undo @ref register_memory for the range that starts at @p base.
   *
   * @return True when the registration is known to be gone: every resource
   *         the session held for it was returned and the registration itself
   *         was released.  False when any step failed or its outcome is
   *         unknown; the caller then keeps the memory and this session alive.
   */
  [[nodiscard]] virtual bool deregister_memory(std::uint8_t* base) noexcept = 0;

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

  /**
   * @brief @ref get on the condition that the object still carries the
   * validator @p if_match, sent as @c If-Match inside the request's
   * authorization.
   *
   * The default serves an empty @p if_match through @ref get.  For any other
   * it issues nothing and reports @c data_commit_state::not_sent with a
   * transport error: a session that cannot send the condition must not
   * transfer as if it had.
   */
  [[nodiscard]] virtual data_get_result get_if_match(object_ref const& object,
                                                     std::size_t object_offset,
                                                     std::size_t size,
                                                     std::uint8_t* registered_base,
                                                     std::size_t dest_offset,
                                                     std::string_view if_match)
  {
    if (if_match.empty()) { return get(object, object_offset, size, registered_base, dest_offset); }
    data_get_result refused;
    refused.transport_error = "the data session cannot make a conditional GET";
    return refused;
  }
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

/**
 * @brief The response a data-plane GET must carry to count as delivered.
 *
 * Both members are required.  The rule only adds conditions: a GET that fails
 * any other check of the backend is not delivered whatever the rule says.
 */
struct completion_rule {
  /// HTTP statuses of a delivered GET.
  std::vector<long> accepted_statuses;

  /// True when @p reply_tag is the reply of a delivered GET with status
  /// @p http_status.
  bool (*reply_ok)(std::string_view reply_tag, long http_status) noexcept {nullptr};

  /// Whether a GET made on the condition of a validator delivers only when
  /// its response carries that same validator.  When false a response
  /// without one is accepted; one that carries a different validator never
  /// is.
  bool require_validator_echo{true};
};

/// How the control endpoint and the data endpoint of a deployment relate.
enum class endpoint_topology : std::uint8_t {
  split,         ///< two addresses, or not provably one
  same_address,  ///< provably the same scheme, host, port and path
};

/// What the backend is built from.  Both clients and a complete rule are
/// required.
struct transport_clients {
  std::shared_ptr<s3_control_client> control;
  std::shared_ptr<rdma_data_session_factory> data_sessions;
  completion_rule rule;

  /// Reported by the backend, which does not act on it: transfers behave the
  /// same in both.
  endpoint_topology topology{endpoint_topology::split};
};

}  // namespace cucascade::io::s3rdma
