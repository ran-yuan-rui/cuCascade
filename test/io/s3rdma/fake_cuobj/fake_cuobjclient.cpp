/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_binding_api.hpp"
#include "cuobjclient.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::test::s3rdma;

namespace {

struct fake_handle {
  void* context;
};

struct saved_call {
  CUObjOps_t* ops;
  void* context;
  void* base;
  std::size_t size;
  loff_t offset;
  std::string token;
};

std::mutex mutex;
binding_options options;
binding_snapshot observed;
std::vector<saved_call> saved;

void record(binding_operation operation,
            void* session,
            void* base              = nullptr,
            std::size_t bytes       = 0,
            std::size_t destination = 0)
{
  if (observed.event_count < observed.events.size()) {
    observed.events[observed.event_count++] = {operation,
                                               reinterpret_cast<std::uintptr_t>(session),
                                               reinterpret_cast<std::uintptr_t>(base),
                                               bytes,
                                               destination};
  } else {
    observed.overflow = true;
  }
}

std::int64_t invoke(saved_call const& call, callback_mode mode)
{
  {
    std::lock_guard lock{mutex};
    ++observed.callback_starts;
  }
  fake_handle handle{call.context};
  cufileRDMAInfo_t descriptor{1, static_cast<int>(call.token.size()), call.token.data()};
  if (mode == callback_mode::null_descriptor) { descriptor.desc_str = nullptr; }
  if (mode == callback_mode::negative_descriptor) { descriptor.desc_len = -1; }
  if (mode == callback_mode::minimum_descriptor) {
    descriptor.desc_len = std::numeric_limits<int>::min();
  }
  auto const result = call.ops->get(&handle,
                                    static_cast<char*>(call.base),
                                    mode == callback_mode::wrong_size ? call.size + 1 : call.size,
                                    mode == callback_mode::wrong_offset ? 17 : call.offset,
                                    mode == callback_mode::null_request ? nullptr : &descriptor);
  std::lock_guard lock{mutex};
  if (observed.result_count < observed.results.size()) {
    observed.results[observed.result_count++] = result;
  } else {
    observed.overflow = true;
  }
  return result;
}

std::int64_t fire_saved(std::size_t index)
{
  saved_call call{};
  {
    std::lock_guard lock{mutex};
    if (index >= saved.size()) { return CUCASCADE_S3RDMA_NO_RESULT; }
    call = saved[index];
  }
  return invoke(call, callback_mode::normal);
}

}  // namespace

cuObjClient::cuObjClient(CUObjOps_t& ops, cuObjProto_t proto) : _ops(&ops)
{
  std::lock_guard lock{mutex};
  ++observed.creates;
  observed.protocol = static_cast<int>(proto);
  observed.has_get  = ops.get != nullptr;
  observed.has_put  = ops.put != nullptr;
  record(binding_operation::create, this);
  if (options.throw_on == 1) { throw std::runtime_error("stand-in constructor failed"); }
}

cuObjClient::~cuObjClient()
{
  std::lock_guard lock{mutex};
  ++observed.destroys;
  observed.ops_alive_at_destroy = _ops->get != nullptr && _ops->put == nullptr;
  record(binding_operation::destroy, this);
}

cuObjErr_t cuObjClient::cuMemObjGetDescriptor(void* base, size_t bytes)
{
  std::lock_guard lock{mutex};
  ++observed.registrations;
  record(binding_operation::register_memory, this, base, bytes);
  if (options.throw_on == 2) { throw std::runtime_error("stand-in registration failed"); }
  return static_cast<cuObjErr_t>(options.register_result);
}

ssize_t cuObjClient::cuMemObjGetMaxRequestCallbackSize(void* base)
{
  std::lock_guard lock{mutex};
  ++observed.capacities;
  record(binding_operation::capacity, this, base);
  if (options.throw_on == 4) { throw std::runtime_error("stand-in capacity failed"); }
  return static_cast<ssize_t>(options.capacity);
}

cuObjErr_t cuObjClient::cuMemObjPutDescriptor(void* base)
{
  std::lock_guard lock{mutex};
  ++observed.deregistrations;
  record(binding_operation::deregister, this, base);
  if (options.throw_on == 3) { throw std::runtime_error("stand-in deregistration failed"); }
  return static_cast<cuObjErr_t>(options.deregister_result);
}

ssize_t cuObjClient::cuObjGet(
  void* context, void* ptr, size_t size, loff_t offset, loff_t buf_offset)
{
  binding_options selected;
  saved_call call{};
  {
    std::lock_guard lock{mutex};
    ++observed.gets;
    record(binding_operation::get, this, ptr, size, static_cast<std::size_t>(buf_offset));
    observed.last_context = reinterpret_cast<std::uintptr_t>(context);
    if (observed.gets <= observed.contexts.size()) {
      observed.contexts[observed.gets - 1] = observed.last_context;
    } else {
      observed.overflow = true;
    }
    observed.last_offset = offset;
    if (options.throw_on == 5) { throw std::runtime_error("stand-in GET failed"); }
    selected = options;
    call     = {
      _ops,
      context,
      ptr,
      size,
      offset,
      std::string{selected.token.data(), std::min(selected.token_length, selected.token.size())}};
    saved.push_back(call);
  }
  if (selected.before_get != nullptr && !selected.before_get(selected.before_get_context)) {
    return CUCASCADE_S3RDMA_NO_RESULT;
  }
  if (selected.late_before >= 0) {
    (void)fire_saved(static_cast<std::size_t>(selected.late_before));
  }
  std::int64_t result = CUCASCADE_S3RDMA_NO_RESULT;
  if (selected.mode == callback_mode::threaded) {
    std::thread worker([&] { result = invoke(call, callback_mode::normal); });
    worker.join();
  } else if (selected.mode != callback_mode::absent) {
    result = invoke(call, selected.mode);
    if (selected.mode == callback_mode::twice) { result = invoke(call, selected.mode); }
  }
  {
    std::lock_guard lock{mutex};
    ++observed.returns;
  }
  return static_cast<ssize_t>(selected.override_result ? selected.sdk_result : result);
}

void* cuObjClient::getCtx(const void* handle)
{
  return handle == nullptr ? nullptr : static_cast<fake_handle const*>(handle)->context;
}

bool cuObjClient::isConnected(void) { return true; }

extern "C" void test_binding_reset()
{
  std::lock_guard lock{mutex};
  observed = {};
  options  = {};
  saved.clear();
}

extern "C" void test_binding_configure(binding_options const* value)
{
  std::lock_guard lock{mutex};
  options = *value;
}

extern "C" void test_binding_snapshot(binding_snapshot* value)
{
  std::lock_guard lock{mutex};
  *value = observed;
}

extern "C" int64_t test_binding_fire(std::size_t index) { return fire_saved(index); }
