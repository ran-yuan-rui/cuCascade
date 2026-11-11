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
#include <cucascade/io/s3rdma/client.hpp>
#include <cucascade/io/s3rdma/config.hpp>
#include <cucascade/io/s3rdma/s3rdma_reactor.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>
#include <exception>
#include <string_view>
#include <vector>

namespace cucascade::io::s3rdma {

/**
 * @brief I/O context of the S3-over-RDMA backend: one reactor, device reads only.
 *
 * The data path is not implemented yet.  The mixed read entry returns ready
 * zero for empty input and otherwise fails after invoking each supplied
 * completion callback with false.  That entry does not inspect the object,
 * call a transport client or touch a device.  Inherited read wrappers retain
 * their zero-size and argument checks; synchronous host reads also clip to
 * the caller-established object size.  All object-open variants are
 * unavailable.
 */
class s3rdma_ioctx final : public templated_ioctx<s3rdma_reactor> {
 public:
  /**
   * @throws std::invalid_argument a client is null, or @p cfg is rejected by
   *         @ref sanitized.
   * @throws std::overflow_error from @ref sanitized, or the storage of the
   *         admission queue overflows.
   */
  s3rdma_ioctx(config cfg, transport_clients clients);
  ~s3rdma_ioctx() override;

  [[nodiscard]] io_context_type type() const noexcept override;

  /**
   * @brief Backend entry for prepared asynchronous reads.
   *
   * Empty input yields a ready 0.  Otherwise the completion callback of each
   * slice runs once with @c false and the returned future carries
   * @c errc::unsupported_request when any slice has a destination other than
   * device memory alone, else @c errc::backend_unavailable.
   */
  exec::semi_future<std::size_t> mixed_readv_async_io(
    io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept override;

 private:
  std::exception_ptr _unsupported;
  std::exception_ptr _unavailable;
};

/// True for `s3://` paths: the scheme check to register the backend with.
[[nodiscard]] bool supports(std::string_view path) noexcept;

/**
 * @brief Factory to register the backend with, for example through
 * @c io_context_registry::replace_ioctx.
 *
 * The returned callable builds an @ref s3rdma_ioctx from @p cfg and
 * @p clients.  It does not throw: when construction fails — a null client, a
 * rejected configuration, an allocation failure — it returns null.  It never
 * substitutes another backend.
 */
[[nodiscard]] io_context_registry::factory_type make_ioctx_factory(config cfg,
                                                                   transport_clients clients);

}  // namespace cucascade::io::s3rdma
