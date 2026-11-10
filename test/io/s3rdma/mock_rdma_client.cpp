/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mock_rdma_client.hpp"

#include <cucascade/io/s3rdma/errors.hpp>

#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace cucascade::test::s3rdma {
namespace rdma = io::s3rdma;

struct mock_rdma_client::state {
  struct step {
    get_behavior behavior;
    std::optional<std::int64_t> sdk_return;
  };
  mutable std::mutex mutex;
  std::map<std::pair<std::string, std::string>, std::vector<std::uint8_t>> objects;
  std::map<std::uint8_t*, std::size_t> registrations;
  std::deque<step> steps;
  std::optional<std::int64_t> callback_limit;
  mock_counts counts;
};

class mock_rdma_client::session final : public rdma::rdma_data_session {
 public:
  explicit session(std::shared_ptr<state> shared) : _state(std::move(shared)) {}

  void register_memory(std::uint8_t* base, std::size_t bytes) override
  {
    if (base == nullptr || bytes == 0 ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument("invalid mock registration");
    }
    std::lock_guard lock(_state->mutex);
    if (!_state->registrations.emplace(base, bytes).second) {
      throw std::invalid_argument("duplicate mock registration");
    }
    ++_state->counts.registrations;
  }

  void deregister_memory(std::uint8_t* base) noexcept override
  {
    std::lock_guard lock(_state->mutex);
    _state->counts.deregistrations += _state->registrations.erase(base);
  }

  std::int64_t max_request_callback_size(std::uint8_t* base) noexcept override
  {
    std::lock_guard lock(_state->mutex);
    auto const found = _state->registrations.find(base);
    if (found == _state->registrations.end()) { return -1; }
    return _state->callback_limit.value_or(static_cast<std::int64_t>(found->second));
  }

  rdma::data_get_result get(rdma::object_ref const& object,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t* base,
                            std::size_t landing) override
  {
    std::lock_guard lock(_state->mutex);
    ++_state->counts.gets;
    auto const registration = _state->registrations.find(base);
    if (registration == _state->registrations.end() || landing > registration->second ||
        size > registration->second - landing) {
      throw rdma::error{rdma::errc::unsupported_request, "invalid registered landing"};
    }
    state::step step{get_behavior::completed, {}};
    if (!_state->steps.empty()) {
      step = _state->steps.front();
      _state->steps.pop_front();
    }
    rdma::data_get_result result;
    result.sdk_return = step.sdk_return.value_or(0);
    if (step.behavior == get_behavior::not_sent || step.behavior == get_behavior::no_callback) {
      if (step.behavior == get_behavior::not_sent) {
        result.sdk_return = step.sdk_return.value_or(-1);
      }
      return result;
    }
    ++_state->counts.issued_requests;
    result.commit               = rdma::data_commit_state::completed;
    result.callback_invocations = step.behavior == get_behavior::reentered_callback ? 2U : 1U;
    auto const found            = _state->objects.find({object.bucket, object.key});
    if (found == _state->objects.end()) {
      result.http_status = 404;
      return result;
    }
    auto const& bytes = found->second;
    if (offset > bytes.size() || size > bytes.size() - offset || size == 0) {
      result.http_status = 416;
      return result;
    }
    auto const written = step.behavior == get_behavior::completed_short ? size - 1 : size;
    std::memcpy(base + landing, bytes.data() + offset, written);
    result.delivered_bytes = written;
    result.http_status     = 206;
    result.reply_tag       = "mock-rdma";
    if (step.behavior == get_behavior::sent_unknown) {
      result.commit          = rdma::data_commit_state::sent_unknown;
      result.sdk_return      = step.sdk_return.value_or(-1);
      result.transport_error = "completion unknown";
    }
    if (step.behavior == get_behavior::throw_error) {
      throw rdma::error{rdma::errc::internal_error, "local failure after issue"};
    }
    return result;
  }

 private:
  std::shared_ptr<state> _state;
};

mock_rdma_client::mock_rdma_client() : _state(std::make_shared<state>()) {}
mock_rdma_client::~mock_rdma_client() = default;

void mock_rdma_client::put(rdma::object_ref const& object, std::vector<std::uint8_t> bytes)
{
  std::lock_guard lock(_state->mutex);
  _state->objects.insert_or_assign({object.bucket, object.key}, std::move(bytes));
}

void mock_rdma_client::script(get_behavior behavior, std::optional<std::int64_t> sdk_return)
{
  std::lock_guard lock(_state->mutex);
  _state->steps.push_back({behavior, sdk_return});
}

void mock_rdma_client::callback_limit(std::optional<std::int64_t> bytes)
{
  std::lock_guard lock(_state->mutex);
  _state->callback_limit = bytes;
}

mock_counts mock_rdma_client::counts() const
{
  std::lock_guard lock(_state->mutex);
  return _state->counts;
}

rdma::head_result mock_rdma_client::head(rdma::object_ref const& object)
{
  std::lock_guard lock(_state->mutex);
  ++_state->counts.heads;
  auto const found = _state->objects.find({object.bucket, object.key});
  if (found == _state->objects.end()) { return {{404, {}}, 0}; }
  return {{200, {}}, found->second.size()};
}

rdma::range_get_result mock_rdma_client::range_get(rdma::object_ref const& object,
                                                   std::size_t offset,
                                                   std::size_t size,
                                                   std::uint8_t* host_dst)
{
  std::lock_guard lock(_state->mutex);
  ++_state->counts.ranges;
  auto const found = _state->objects.find({object.bucket, object.key});
  if (found == _state->objects.end()) { return {{404, {}}, 0, {}}; }
  auto const& bytes = found->second;
  if (offset > bytes.size() || size > bytes.size() - offset || size == 0) {
    return {{416, {}}, 0, {}};
  }
  if (host_dst == nullptr) { throw std::invalid_argument("null host destination"); }
  auto content_range = "bytes " + std::to_string(offset) + "-" + std::to_string(offset + size - 1) +
                       "/" + std::to_string(bytes.size());
  std::memcpy(host_dst, bytes.data() + offset, size);
  return {{206, {}}, size, std::move(content_range)};
}

std::unique_ptr<rdma::rdma_data_session> mock_rdma_client::acquire()
{
  auto result = std::make_unique<session>(_state);
  std::lock_guard lock(_state->mutex);
  ++_state->counts.sessions;
  return result;
}

}  // namespace cucascade::test::s3rdma
