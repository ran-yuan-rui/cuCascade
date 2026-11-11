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

#include <cucascade/io/s3rdma/errors.hpp>
#include <cucascade/io/s3rdma/s3rdma_ioctx.hpp>
#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace cucascade::io::s3rdma {

namespace {

std::vector<std::unique_ptr<s3rdma_reactor>> make_reactors(config cfg, transport_clients clients)
{
  std::vector<std::unique_ptr<s3rdma_reactor>> reactors;
  reactors.push_back(std::make_unique<s3rdma_reactor>(
    std::make_shared<s3rdma_reactor::reactor_context>(cfg, std::move(clients))));
  return reactors;
}

}  // namespace

s3rdma_ioctx::s3rdma_ioctx(config cfg, transport_clients clients)
  : templated_ioctx<s3rdma_reactor>(make_reactors(cfg, std::move(clients))),
    _unsupported(std::make_exception_ptr(
      error{errc::unsupported_request, "s3rdma: only device-memory destinations are supported"})),
    _unavailable(std::make_exception_ptr(
      error{errc::backend_unavailable, "s3rdma: the data path is not available"}))
{
}

s3rdma_ioctx::~s3rdma_ioctx() { this->pre_destroy(); }

io_context_type s3rdma_ioctx::type() const noexcept { return io_context_type::s3rdma; }

exec::semi_future<std::size_t> s3rdma_ioctx::mixed_readv_async_io(
  io_object const& /*object*/, std::vector<prepared_io_slice>&& slices) noexcept
{
  if (slices.empty()) { return exec::make_semi_future<std::size_t>(0); }

  bool const device_only =
    std::all_of(slices.begin(), slices.end(), [](prepared_io_slice const& slice) {
      return slice.has_device_request() && !slice.has_host_request();
    });
  for (auto& slice : slices) {
    if (slice.on_complete != nullptr) {
      (*slice.on_complete)(slice.h_buffer.fragments(), false);
      slice.on_complete.reset();
    }
  }
  return exec::make_semi_future<std::size_t>(device_only ? _unavailable : _unsupported);
}

bool supports(std::string_view path) noexcept { return s3rdma_reactor::supports(path); }

io_context_registry::factory_type make_ioctx_factory(config cfg, transport_clients clients)
{
  return [cfg, clients = std::move(clients)](
           io_context_registry::config_type const&) noexcept -> std::shared_ptr<ioctx> {
    try {
      try {
        return std::make_shared<s3rdma_ioctx>(cfg, clients);
      } catch (std::exception const& failure) {
        CUCASCADE_LOG_ERROR("s3rdma: I/O context construction failed: {}", failure.what());
      } catch (...) {
        CUCASCADE_LOG_ERROR("s3rdma: I/O context construction failed: unknown error");
      }
    } catch (...) {
      // Reporting the failure must not let anything escape either.
    }
    return nullptr;
  };
}

}  // namespace cucascade::io::s3rdma
