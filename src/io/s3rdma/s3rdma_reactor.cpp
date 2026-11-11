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

#include <cucascade/io/io_utils.hpp>
#include <cucascade/io/s3rdma/errors.hpp>
#include <cucascade/io/s3rdma/s3rdma_reactor.hpp>
#include <cucascade/io/uri_parser.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cucascade::io::s3rdma {

namespace {

constexpr char const* unavailable_message = "s3rdma: the data path is not available";

transport_clients validated(transport_clients clients)
{
  if (clients.control == nullptr || clients.data_sessions == nullptr) {
    throw std::invalid_argument("s3rdma: both transport clients are required");
  }
  return clients;
}

}  // namespace

s3rdma_io_object::s3rdma_io_object(std::string path, object_ref ref, std::size_t size)
  : _path(std::move(path)), _ref(std::move(ref)), _size(size)
{
}

s3rdma_reactor::reactor_context::reactor_context(config cfg, transport_clients clients)
  : _clients(validated(std::move(clients))),
    _cfg(sanitized(cfg)),
    _gate(*_cfg.queue_cap, _cfg.arena_slot_size)
{
}

s3rdma_reactor::s3rdma_reactor(std::shared_ptr<reactor_context> ctx)
  : _ctx(std::move(ctx)),
    _unavailable(std::make_exception_ptr(error{errc::backend_unavailable, unavailable_message}))
{
  if (_ctx == nullptr) { throw std::invalid_argument("s3rdma_reactor: null reactor context"); }
}

void s3rdma_reactor::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request != nullptr) { request->cancel_remaining(_unavailable); }
}

std::size_t s3rdma_reactor::host_read(io_object_type const& /*object*/,
                                      std::size_t /*offset*/,
                                      std::size_t /*size*/,
                                      std::uint8_t* /*destination*/)
{
  throw error{errc::backend_unavailable, unavailable_message};
}

std::unique_ptr<s3rdma_reactor::io_object_type> s3rdma_reactor::create_io_object(
  std::string /*path*/)
{
  throw error{errc::backend_unavailable, unavailable_message};
}

bool s3rdma_reactor::supports(std::string_view path) noexcept
{
  try {
    return io::parse(path).scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::vector<byte_range> s3rdma_reactor::align_and_coalesce(
  std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept
{
  try {
    return io_utils::align_and_coalesce(ranges, std::max<std::size_t>(alignment.value_or(1), 1));
  } catch (...) {
    return {};
  }
}

}  // namespace cucascade::io::s3rdma
