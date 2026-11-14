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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/object_store_lister.hpp>
#include <cucascade/io/s3rdma/client.hpp>
#include <cucascade/io/s3rdma/config.hpp>
#include <cucascade/io/s3rdma/cuda_delivery.hpp>
#include <cucascade/io/s3rdma/s3rdma_reactor.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::io::s3rdma {

/**
 * @brief I/O context of the S3-over-RDMA backend: one reactor, device reads
 * over RDMA, host reads and object open over HTTP.
 *
 * Nothing is served before @c start() succeeded: reads and opens fail with
 * @c errc::backend_unavailable.  Once the backend stopped after a failure,
 * every entry reports the error it stopped with.  Some requests are answered
 * without the backend, and so in the same way in every state: a read of no
 * slices, a host read of no bytes or from at or beyond the end of the object,
 * and a request that is invalid or that the backend does not serve.
 *
 * The backend has no cache and serves no host-to-device read.  A device read
 * lands in caller device memory only; host reads run on the calling thread
 * and return an already completed future.
 */
class s3rdma_ioctx final : public templated_ioctx<s3rdma_reactor> {
 public:
  /**
   * @throws std::invalid_argument a client is null, the completion rule is
   *         incomplete, a member of @p delivery is not callable, or @p cfg is
   *         rejected by @ref sanitized.
   * @throws std::overflow_error from @ref sanitized, or the storage of the
   *         admission queue overflows.
   */
  s3rdma_ioctx(config cfg,
               transport_clients clients,
               cuda_delivery_ops delivery   = default_delivery_ops(),
               context_health_policy policy = {});
  ~s3rdma_ioctx() override;

  [[nodiscard]] io_context_type type() const noexcept override;

  /// How this backend instance ended; stays readable after the context is gone.
  [[nodiscard]] std::shared_ptr<runtime_status const> status() const noexcept;

  /// How the deployment's two endpoints relate; fixed at construction.
  [[nodiscard]] endpoint_topology topology() const noexcept { return reactor().topology(); }

  /**
   * @brief Backend entry for prepared asynchronous reads.
   *
   * Empty input yields a ready 0.  A request with a slice whose destination
   * is not device memory alone is refused with @c errc::unsupported_request;
   * a stopped backend refuses with the error it stopped with; a backend that
   * is not started refuses with @c errc::backend_unavailable.  A refusal runs
   * the completion callback of each slice once with @c false and makes no
   * CUDA call.  Everything else is dispatched to the reactor.
   */
  exec::semi_future<std::size_t> mixed_readv_async_io(
    io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept override;

  /// Reads on the calling thread; see @ref s3rdma_reactor::host_read.  The
  /// returned future is already completed.
  exec::semi_future<std::size_t> host_read_async_io(
    io_object const& object,
    std::size_t offset,
    std::size_t size,
    std::uint8_t* dst,
    io_priority priority = io_priority::automatic) noexcept override;

  /// Reads the ranges one after another on the calling thread and stops at the
  /// first that fails.  The returned future is already completed.
  exec::semi_future<std::size_t> host_readv_async_io(
    io_object const& object,
    std::span<slice const> slices,
    io_priority priority = io_priority::automatic) noexcept override;

  /**
   * @brief Stream the ListObjectsV2 pages of @p bucket under @p prefix to
   * @p sink, one call per page; @p sink returns false to stop.
   *
   * Listing runs on the control plane: one signed request per page, never
   * repeated, each under the rules of every other control call — refused
   * before a successful start and once the backend stopped.  Page size,
   * continuation and the caps are those of @c rest::object_store_lister, as
   * for the REST backend.
   *
   * @throws std::exception a page failed, the listing is malformed or a cap
   *         was exceeded; a listing is never truncated.
   */
  void list_objects_paged(std::string_view bucket,
                          std::string_view prefix,
                          std::size_t page_size,
                          std::function<bool(rest::s3::list_objects_v2_page const&)> const& sink,
                          std::optional<std::size_t> max_scanned = std::nullopt);

  /// Every object under @p prefix, in listing order; see
  /// @ref list_objects_paged.
  [[nodiscard]] std::vector<rest::s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt);

  /// @c config::list_max_matches.
  [[nodiscard]] std::size_t list_max_matches() const noexcept;

  /// The backend has one reactor, and it serves every request.
  std::vector<s3rdma_reactor*> next_reactor(io_object_type const& object,
                                            std::size_t n_slices,
                                            io_op_type type,
                                            int device_id = -1) override;

 protected:
  /// One HEAD request; the size of the object comes from its response.
  std::shared_ptr<io_object> create_io_object(std::string path) override;
  /// As the one-argument form: the hint is not used.
  std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint) override;
  /// As the one-argument form: a size supplied by the caller is not trusted.
  std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size) override;

  /// Asks CUDA whether the context is still usable and stops the backend when
  /// it is not; see @ref context_health_policy.
  void on_device_dispatch_failure() noexcept override;

 private:
  [[nodiscard]] s3rdma_reactor& reactor() const noexcept { return *_reactors.front(); }

  std::exception_ptr _unsupported;
  rest::object_store_lister _lister;
};

/// True for `s3://` paths: the scheme check to register the backend with.
[[nodiscard]] bool supports(std::string_view path) noexcept;

/**
 * @brief Factory to register the backend with, for example through
 * @c io_context_registry::replace_ioctx.
 *
 * The returned callable builds an @ref s3rdma_ioctx from its arguments and
 * does not start it.  It does not throw: when construction fails — a null
 * client, an incomplete rule or operation set, a rejected configuration, an
 * allocation failure — it returns null.  It never substitutes another
 * backend.
 */
[[nodiscard]] io_context_registry::factory_type make_ioctx_factory(
  config cfg,
  transport_clients clients,
  cuda_delivery_ops delivery   = default_delivery_ops(),
  context_health_policy policy = {});

}  // namespace cucascade::io::s3rdma
