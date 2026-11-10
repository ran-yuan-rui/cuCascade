/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cucascade/io/s3rdma/client.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cucascade::test::s3rdma {

enum class get_behavior {
  not_sent,
  sent_unknown,
  completed,
  completed_short,
  no_callback,
  reentered_callback,
  throw_error
};

struct mock_counts {
  std::size_t sessions{0};
  std::size_t registrations{0};
  std::size_t deregistrations{0};
  std::size_t heads{0};
  std::size_t ranges{0};
  std::size_t gets{0};
  std::size_t issued_requests{0};
};

// Host memory models a registration; these input rules do not qualify a native SDK.
class mock_rdma_client final : public io::s3rdma::s3_control_client,
                               public io::s3rdma::rdma_data_session_factory {
 public:
  mock_rdma_client();
  ~mock_rdma_client() override;
  void put(io::s3rdma::object_ref const& object, std::vector<std::uint8_t> bytes);
  void script(get_behavior behavior, std::optional<std::int64_t> sdk_return = {});
  void callback_limit(std::optional<std::int64_t> bytes);
  [[nodiscard]] mock_counts counts() const;

  io::s3rdma::head_result head(io::s3rdma::object_ref const& object) override;
  io::s3rdma::range_get_result range_get(io::s3rdma::object_ref const& object,
                                         std::size_t offset,
                                         std::size_t size,
                                         std::uint8_t* host_dst) override;
  std::unique_ptr<io::s3rdma::rdma_data_session> acquire() override;

 private:
  struct state;
  class session;
  std::shared_ptr<state> _state;
};

inline io::s3rdma::transport_clients clients(std::shared_ptr<mock_rdma_client> const& mock)
{
  return {mock, mock};
}

}  // namespace cucascade::test::s3rdma
