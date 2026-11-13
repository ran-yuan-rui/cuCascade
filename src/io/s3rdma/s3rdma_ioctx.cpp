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
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cucascade::io::s3rdma {

namespace {

std::vector<std::unique_ptr<s3rdma_reactor>> make_reactors(config cfg,
                                                           transport_clients clients,
                                                           cuda_delivery_ops delivery,
                                                           context_health_policy policy)
{
  std::vector<std::unique_ptr<s3rdma_reactor>> reactors;
  reactors.push_back(
    std::make_unique<s3rdma_reactor>(std::make_shared<s3rdma_reactor::reactor_context>(
      std::move(cfg), std::move(clients), std::move(delivery), std::move(policy))));
  return reactors;
}

s3rdma_io_object const& typed(io_object const& object)
{
  auto const* result = dynamic_cast<s3rdma_io_object const*>(&object);
  if (result == nullptr) { throw std::invalid_argument("I/O object belongs to another backend"); }
  return *result;
}

/// Fail the slices that read nothing from @p object: an empty range, or one
/// that starts at or beyond its end.  True when a slice is left to read.
bool fail_unread(s3rdma_io_object const& object, std::vector<prepared_io_slice>& slices) noexcept
{
  bool pending = false;
  for (auto& slice : slices) {
    if (slice.rng.size != 0 && slice.rng.offset < object.size()) {
      pending = true;
    } else if (slice.on_complete != nullptr) {
      (*slice.on_complete)(slice.h_buffer.fragments(), false);
      slice.on_complete.reset();
    }
  }
  return pending;
}

}  // namespace

s3rdma_ioctx::s3rdma_ioctx(config cfg,
                           transport_clients clients,
                           cuda_delivery_ops delivery,
                           context_health_policy policy)
  : templated_ioctx<s3rdma_reactor>(
      make_reactors(std::move(cfg), std::move(clients), std::move(delivery), std::move(policy))),
    _unsupported(std::make_exception_ptr(
      error{errc::unsupported_request, "s3rdma: only device-memory destinations are supported"}))
{
}

s3rdma_ioctx::~s3rdma_ioctx() { this->pre_destroy(); }

io_context_type s3rdma_ioctx::type() const noexcept { return io_context_type::s3rdma; }

std::shared_ptr<runtime_status const> s3rdma_ioctx::status() const noexcept
{
  return reactor().status();
}

exec::semi_future<std::size_t> s3rdma_ioctx::mixed_readv_async_io(
  io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept
{
  if (slices.empty()) { return exec::make_semi_future<std::size_t>(0); }

  bool const device_only =
    std::all_of(slices.begin(), slices.end(), [](prepared_io_slice const& slice) {
      return slice.has_device_request() && !slice.has_host_request();
    });
  // Decided here, before the framework dispatches: it asks CUDA for the
  // current device first, which a stopped backend must not do.
  std::exception_ptr refusal = device_only ? reactor().admission_error() : _unsupported;
  if (refusal == nullptr) {
    // The framework fails a slice that reads nothing through its callback and
    // then asks CUDA for the current device.  Those callbacks run here, before
    // the request is admitted: nothing may come between the admission and
    // that CUDA call.
    auto const* target = dynamic_cast<s3rdma_io_object const*>(&object);
    if (target != nullptr && !fail_unread(*target, slices)) {
      return exec::make_semi_future<std::size_t>(0);
    }
    // The admission lasts until the framework has queued the request or
    // reported why it could not, so that closing the backend waits for both.
    if (auto const admitted = reactor().admit_dispatch(refusal)) {
      return templated_ioctx<s3rdma_reactor>::mixed_readv_async_io(object, std::move(slices));
    }
  }
  for (auto& slice : slices) {
    if (slice.on_complete != nullptr) {
      (*slice.on_complete)(slice.h_buffer.fragments(), false);
      slice.on_complete.reset();
    }
  }
  return exec::make_semi_future<std::size_t>(refusal);
}

exec::semi_future<std::size_t> s3rdma_ioctx::host_read_async_io(io_object const& object,
                                                                std::size_t offset,
                                                                std::size_t size,
                                                                std::uint8_t* dst,
                                                                io_priority /*priority*/) noexcept
{
  try {
    return exec::make_semi_future<std::size_t>(
      reactor().host_read(typed(object), offset, size, dst));
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
}

exec::semi_future<std::size_t> s3rdma_ioctx::host_readv_async_io(io_object const& object,
                                                                 std::span<slice const> slices,
                                                                 io_priority /*priority*/) noexcept
{
  try {
    auto const& target = typed(object);
    std::size_t total  = 0;
    for (auto const& range : slices) {
      auto const read = reactor().host_read(target, range.offset(), range.size(), range.dst);
      if (read > std::numeric_limits<std::size_t>::max() - total) {
        throw std::overflow_error("s3rdma: host read byte count overflow");
      }
      total += read;
    }
    return exec::make_semi_future<std::size_t>(std::size_t{total});
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
}

std::vector<s3rdma_reactor*> s3rdma_ioctx::next_reactor(io_object_type const& /*object*/,
                                                        std::size_t /*n_slices*/,
                                                        io_op_type /*type*/,
                                                        int /*device_id*/)
{
  return {_reactors.front().get()};
}

std::shared_ptr<io_object> s3rdma_ioctx::create_io_object(std::string path)
{
  return reactor().open(std::move(path));
}

std::shared_ptr<io_object> s3rdma_ioctx::create_io_object(std::string path, open_hint /*hint*/)
{
  return reactor().open(std::move(path));
}

std::shared_ptr<io_object> s3rdma_ioctx::create_io_object(std::string path,
                                                          std::uint64_t /*known_size*/)
{
  return reactor().open(std::move(path));
}

void s3rdma_ioctx::on_device_dispatch_failure() noexcept { reactor().check_device_health(); }

bool supports(std::string_view path) noexcept { return s3rdma_reactor::supports(path); }

io_context_registry::factory_type make_ioctx_factory(config cfg,
                                                     transport_clients clients,
                                                     cuda_delivery_ops delivery,
                                                     context_health_policy policy)
{
  return [cfg      = std::move(cfg),
          clients  = std::move(clients),
          delivery = std::move(delivery),
          policy   = std::move(policy)](
           io_context_registry::config_type const&) noexcept -> std::shared_ptr<ioctx> {
    try {
      try {
        return std::make_shared<s3rdma_ioctx>(cfg, clients, delivery, policy);
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
