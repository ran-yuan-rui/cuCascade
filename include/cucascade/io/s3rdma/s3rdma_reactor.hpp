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

#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/s3rdma/admission_gate.hpp>
#include <cucascade/io/s3rdma/client.hpp>
#include <cucascade/io/s3rdma/config.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::io::s3rdma {

/**
 * @brief An S3 object as the S3-over-RDMA backend addresses it.
 */
class s3rdma_io_object final : public io_object {
 public:
  /// @param size Size of the object in bytes, as established by the caller.
  s3rdma_io_object(std::string path, object_ref ref, std::size_t size);

  [[nodiscard]] object_ref const& ref() const noexcept { return _ref; }
  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  object_ref _ref;
  std::size_t _size;
};

/**
 * @brief Reactor of the S3-over-RDMA backend.
 *
 * The data path is not implemented yet.  Object open, host reads and queued
 * requests fail with @c errc::backend_unavailable; no transport client is
 * called and no device is touched.
 */
class s3rdma_reactor {
 public:
  using io_object_type      = s3rdma_io_object;
  using reactor_config_type = config;

  /**
   * @brief State shared by the reactors of one I/O context: the validated
   * configuration, the transport clients and the admission gate.
   */
  class reactor_context {
   public:
    /**
     * @throws std::invalid_argument a client is null, or @p cfg is rejected by
     *         @ref sanitized.
     * @throws std::overflow_error from @ref sanitized, or the storage of the
     *         admission queue overflows.
     */
    reactor_context(config cfg, transport_clients clients);
    reactor_context(reactor_context const&)            = delete;
    reactor_context& operator=(reactor_context const&) = delete;

    /// The configuration after @ref sanitized: @c queue_cap is populated.
    [[nodiscard]] config const& cfg() const noexcept { return _cfg; }
    [[nodiscard]] transport_clients const& clients() const noexcept { return _clients; }
    [[nodiscard]] admission_gate& gate() noexcept { return _gate; }

   private:
    transport_clients _clients;
    config _cfg;
    admission_gate _gate;
  };

  /// @throws std::invalid_argument @p ctx is null.
  explicit s3rdma_reactor(std::shared_ptr<reactor_context> ctx);

  [[nodiscard]] config const& get_config() const noexcept { return _ctx->cfg(); }

  /// Fails every remaining slice of @p request with
  /// @c errc::backend_unavailable.
  void enqueue(std::unique_ptr<grouped_io_request> request) noexcept;

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  /// @throws error with @c errc::backend_unavailable.
  std::size_t host_read(io_object_type const& object,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* destination);

  void start() noexcept {}
  void shutdown() noexcept {}
  void interrupt() noexcept {}

  /// @throws error with @c errc::backend_unavailable: object open is not
  ///         implemented yet.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// True for `s3://` paths.
  [[nodiscard]] static bool supports(std::string_view path) noexcept;

  /// Sorts and coalesces @p ranges.  An @p alignment of at least 1 is honored
  /// as a lower bound.  Returns no ranges if it cannot allocate.
  [[nodiscard]] static std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept;

 private:
  std::shared_ptr<reactor_context> _ctx;
  std::exception_ptr _unavailable;
};

}  // namespace cucascade::io::s3rdma

namespace cucascade::io {

/// Capability profile of the S3-over-RDMA reactor: device reads only.
template <>
struct reactor_traits<s3rdma::s3rdma_reactor> {
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = false;
  static constexpr bool supports_vector_host_read    = false;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = true;
};

static_assert(io_reactor_c<s3rdma::s3rdma_reactor>);

}  // namespace cucascade::io
