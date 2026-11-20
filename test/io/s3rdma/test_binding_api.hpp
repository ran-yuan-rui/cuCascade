/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cucascade/io/s3rdma/cuobj_abi.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace cucascade::test::s3rdma {

enum class callback_mode : int {
  normal,
  twice,
  absent,
  threaded,
  wrong_offset,
  wrong_size,
  null_descriptor,
  empty_descriptor,
  non_ascii_descriptor,
  null_request,
  trailing_nuls,
  negative_descriptor,
  minimum_descriptor
};

enum class binding_operation : int { create, register_memory, capacity, get, deregister, destroy };

struct binding_options {
  int create_result{0};
  bool create_null{false};
  int register_result{0};
  int deregister_result{0};
  std::int64_t capacity{4096};
  callback_mode mode{callback_mode::normal};
  bool override_result{false};
  std::int64_t sdk_result{0};
  int late_before{-1};
  bool fill_output{true};
  int (*write_output)(void* context, void* destination, std::size_t bytes){nullptr};
  void* write_context{nullptr};
  int throw_on{0};
  std::array<char, 128> token{"CAPABILITY-NOT-FOR-LOGS"};
  std::size_t token_length{23};
  bool poison_output{false};
  bool (*before_get)(void* context){nullptr};
  void* before_get_context{nullptr};
};

struct binding_event {
  binding_operation operation{};
  std::uintptr_t session{0};
  std::uintptr_t base{0};
  std::size_t bytes{0};
  std::size_t destination{0};
};

struct binding_snapshot {
  unsigned entries{0};
  unsigned probes{0};
  unsigned creates{0};
  unsigned destroys{0};
  unsigned registrations{0};
  unsigned deregistrations{0};
  unsigned capacities{0};
  unsigned gets{0};
  std::size_t event_count{0};
  std::array<binding_event, 256> events{};
  std::size_t result_count{0};
  std::array<std::int64_t, 256> results{};
  bool overflow{false};
  int protocol{0};
  bool has_get{false};
  bool has_put{false};
  bool ops_alive_at_destroy{false};
  std::uintptr_t last_context{0};
  std::int64_t last_offset{0};
  unsigned callback_starts{0};
  unsigned returns{0};
  std::array<std::uintptr_t, 256> contexts{};
};

using reset_binding_fn     = void (*)();
using configure_binding_fn = void (*)(binding_options const*);
using snapshot_binding_fn  = void (*)(binding_snapshot*);
using fire_binding_fn      = std::int64_t (*)(std::size_t);

}  // namespace cucascade::test::s3rdma
