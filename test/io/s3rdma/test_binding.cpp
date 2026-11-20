/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_binding_api.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::test::s3rdma;

struct cucascade_s3rdma_session {
  std::uintptr_t id;
};

namespace {

struct saved_call {
  cucascade_s3rdma_request_fn function;
  void* context;
  std::size_t size;
  std::string token;
};

std::mutex mutex;
binding_options options;
binding_snapshot observed;
std::vector<saved_call> saved;
std::uintptr_t next_id{1};

void record(binding_operation operation,
            cucascade_s3rdma_session* session,
            void* base              = nullptr,
            std::size_t bytes       = 0,
            std::size_t destination = 0)
{
  if (observed.event_count < observed.events.size()) {
    observed.events[observed.event_count++] = {operation,
                                               session == nullptr ? 0 : session->id,
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
  auto token = call.token;
  cucascade_s3rdma_request request{token.data(), token.size(), call.size, 0};
  switch (mode) {
    case callback_mode::wrong_offset: request.offset = 1; break;
    case callback_mode::wrong_size: ++request.size; break;
    case callback_mode::null_descriptor: request.descriptor = nullptr; break;
    case callback_mode::empty_descriptor: request.descriptor_length = 0; break;
    case callback_mode::non_ascii_descriptor:
      token.push_back('\x80');
      request.descriptor        = token.data();
      request.descriptor_length = token.size();
      break;
    case callback_mode::trailing_nuls:
      token.append(3, '\0');
      request.descriptor        = token.data();
      request.descriptor_length = token.size();
      break;
    default: break;
  }
  auto const result =
    call.function(call.context, mode == callback_mode::null_request ? nullptr : &request);
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

int32_t probe(cucascade_s3rdma_probe* out)
{
  std::lock_guard lock{mutex};
  ++observed.probes;
  out->protocol       = CUCASCADE_S3RDMA_PROTOCOL_DC_V1;
  out->io_retry_count = 0;
#if defined(S3RDMA_TEST_BINDING_VARIANT_wrong_protocol)
  out->protocol = 9001;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_retry_unknown)
  out->io_retry_count = -1;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_retry_nonzero)
  out->io_retry_count = 3;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_probe_fails)
  return 1;
#else
  return 0;
#endif
}

int32_t create(cucascade_s3rdma_session** out)
{
  std::lock_guard lock{mutex};
  ++observed.creates;
  *out = nullptr;
  if (options.create_result != 0) { return options.create_result; }
  if (options.create_null) { return 0; }
  try {
    *out = new cucascade_s3rdma_session{next_id++};
  } catch (std::bad_alloc const&) {
    return 1;
  }
  record(binding_operation::create, *out);
  return 0;
}

void destroy(cucascade_s3rdma_session* session)
{
  std::lock_guard lock{mutex};
  ++observed.destroys;
  record(binding_operation::destroy, session);
  delete session;
}

int32_t register_memory(cucascade_s3rdma_session* session, void* base, std::size_t bytes)
{
  std::lock_guard lock{mutex};
  ++observed.registrations;
  record(binding_operation::register_memory, session, base, bytes);
  return options.register_result;
}

int32_t deregister(cucascade_s3rdma_session* session, void* base)
{
  std::lock_guard lock{mutex};
  ++observed.deregistrations;
  record(binding_operation::deregister, session, base);
  return options.deregister_result;
}

int64_t capacity(cucascade_s3rdma_session* session, void* base)
{
  std::lock_guard lock{mutex};
  ++observed.capacities;
  record(binding_operation::capacity, session, base);
  return options.capacity;
}

int64_t get(cucascade_s3rdma_session* session,
            void* base,
            std::size_t length,
            std::size_t destination,
            cucascade_s3rdma_request_fn request,
            void* context)
{
  binding_options selected;
  saved_call call{};
  {
    std::lock_guard lock{mutex};
    ++observed.gets;
    record(binding_operation::get, session, base, length, destination);
    selected = options;
    call     = {
      request,
      context,
      length,
      std::string{selected.token.data(), std::min(selected.token_length, selected.token.size())}};
    saved.push_back(call);
    observed.last_context = reinterpret_cast<std::uintptr_t>(context);
    if (observed.gets <= observed.contexts.size()) {
      observed.contexts[observed.gets - 1] = observed.last_context;
    } else {
      observed.overflow = true;
    }
  }
  if (selected.poison_output) {
    std::memset(static_cast<unsigned char*>(base) + destination, 0xe1, length);
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
  if (selected.fill_output && result == static_cast<std::int64_t>(length)) {
    auto* output = static_cast<unsigned char*>(base) + destination;
    if (selected.write_output != nullptr) {
      if (selected.write_output(selected.write_context, output, length) != 0) { return -EIO; }
    } else {
      std::memset(output, 0x5a, length);
    }
  }
  {
    std::lock_guard lock{mutex};
    ++observed.returns;
  }
  return selected.override_result ? selected.sdk_result : result;
}

[[maybe_unused]] cucascade_s3rdma_api const table = [] {
  cucascade_s3rdma_api value{CUCASCADE_S3RDMA_ABI_VERSION,
                             static_cast<uint32_t>(sizeof(cucascade_s3rdma_api)),
                             &probe,
                             &create,
                             &destroy,
                             &register_memory,
                             &deregister,
                             &capacity,
                             &get};
#if defined(S3RDMA_TEST_BINDING_VARIANT_wrong_version)
  ++value.abi_version;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_wrong_size)
  ++value.api_size;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_probe)
  value.probe = nullptr;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_create)
  value.session_create = nullptr;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_destroy)
  value.session_destroy = nullptr;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_register)
  value.register_memory = nullptr;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_deregister)
  value.deregister_memory = nullptr;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_capacity)
  value.max_request_size = nullptr;
#endif
#if defined(S3RDMA_TEST_BINDING_VARIANT_missing_get)
  value.get = nullptr;
#endif
  return value;
}();

}  // namespace

extern "C" void test_binding_reset()
{
  std::lock_guard lock{mutex};
  auto const entries = observed.entries;
  observed           = {};
  observed.entries   = entries;
  options            = {};
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

#if !defined(S3RDMA_TEST_BINDING_VARIANT_missing_symbol)
extern "C" cucascade_s3rdma_api const* cucascade_s3rdma_binding()
{
  std::lock_guard lock{mutex};
  ++observed.entries;
#if defined(S3RDMA_TEST_BINDING_VARIANT_null_table)
  return nullptr;
#else
  return &table;
#endif
}
#endif
