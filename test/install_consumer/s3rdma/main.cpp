/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cucascade/io/s3rdma/client.hpp>
#include <cucascade/io/s3rdma/config.hpp>
#include <cucascade/io/s3rdma/errors.hpp>

#include <cstdio>
#include <stdexcept>
#include <type_traits>

#ifndef CUCASCADE_HAS_S3_RDMA
#error "The component target must propagate CUCASCADE_HAS_S3_RDMA"
#endif

int main()
{
  namespace rdma = cucascade::io::s3rdma;
  static_assert(std::is_abstract_v<rdma::s3_control_client>);
  static_assert(std::is_abstract_v<rdma::rdma_data_session>);
  static_assert(std::is_abstract_v<rdma::rdma_data_session_factory>);
  try {
    rdma::config cfg;
    cfg.max_inflight    = 2;
    cfg.arena_slot_size = 1;
    auto normalized     = rdma::sanitized(cfg);
    if (normalized.queue_cap != 8 || normalized.max_inflight != 2 ||
        normalized.arena_slot_size != 1 || normalized.min_alignment_requirement() != 1 ||
        normalized.merge_gap_size() != 0) {
      return 1;
    }
    cfg.queue_cap = 3;
    if (rdma::sanitized(cfg).queue_cap != 3) { return 2; }
    try {
      (void)rdma::sanitized(rdma::config{});
      return 3;
    } catch (std::invalid_argument const&) {
    }
    try {
      throw rdma::error{rdma::errc::backend_unavailable, "test failure"};
    } catch (rdma::error const& e) {
      if (e.code() != rdma::errc::backend_unavailable) { return 4; }
    }
    rdma::transport_clients clients;
    rdma::data_get_result result;
    if (clients.control || clients.data_sessions ||
        result.commit != rdma::data_commit_state::not_sent || result.sdk_return != -1 ||
        result.callback_invocations != 0 || result.delivered_bytes != 0 ||
        result.http_status != 0 || !result.reply_tag.empty() || !result.transport_error.empty()) {
      return 5;
    }
    std::puts("s3rdma installed consumer passed");
    return 0;
  } catch (...) {
    return 6;
  }
}
