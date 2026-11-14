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

// Binds the S3-over-RDMA component to the cuObject client.  This is the only
// source of the project that includes a cuObject header; it is built as a
// library of its own, which the component loads at run time.

#include <cucascade/io/s3rdma/cuobj_abi.h>

#include <cuobjclient.h>
#include <sys/types.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <new>

struct cucascade_s3rdma_session {
  cuObjClient* client{nullptr};
  // The request callback of the GET in progress.  Kept in the session rather
  // than in the frame of get: the client may run its callback late.
  std::atomic<cucascade_s3rdma_request_fn> request{nullptr};
  std::atomic<void*> context{nullptr};
};

namespace {

// Nothing may leave a callback through the client's frames.
ssize_t on_get(const void* handle,
               char* /*ptr*/,
               size_t size,
               loff_t offset,
               const cufileRDMAInfo_t* rdma) noexcept
{
  auto* const session = static_cast<cucascade_s3rdma_session*>(cuObjClient::getCtx(handle));
  if (session == nullptr) { return -ENOTSUP; }
  auto const request = session->request.load(std::memory_order_acquire);
  if (request == nullptr) { return -ENOTSUP; }
  cucascade_s3rdma_request what{};
  what.descriptor        = rdma != nullptr ? rdma->desc_str : nullptr;
  what.descriptor_length = rdma != nullptr ? static_cast<size_t>(rdma->desc_len) : 0;
  what.size              = size;
  what.offset            = static_cast<int64_t>(offset);
  return static_cast<ssize_t>(request(session->context.load(std::memory_order_acquire), &what));
}

int32_t probe(cucascade_s3rdma_probe* out) noexcept
{
  if (out == nullptr) { return 1; }
  out->protocol = CUCASCADE_S3RDMA_PROTOCOL_DC_V1;
  // The client reads rdma_io_retry_count from its JSON configuration and
  // offers no call that reports the value in effect.
  out->io_retry_count = -1;
  return 0;
}

int32_t session_create(cucascade_s3rdma_session** out) noexcept
{
  if (out == nullptr) { return 1; }
  *out = nullptr;
  try {
    auto* const session = new cucascade_s3rdma_session;
    CUObjOps_t ops{};
    ops.get = &on_get;
    ops.put = nullptr;
    try {
      session->client = new cuObjClient(ops, CUOBJ_PROTO_RDMA_DC_V1);
    } catch (...) {
      delete session;
      return 1;
    }
    *out = session;
    return 0;
  } catch (...) {
    return 1;
  }
}

void session_destroy(cucascade_s3rdma_session* session) noexcept
{
  if (session == nullptr) { return; }
  try {
    delete session->client;
  } catch (...) {
    // The client is gone as far as this library can tell.
  }
  delete session;
}

int32_t register_memory(cucascade_s3rdma_session* session, void* base, size_t bytes) noexcept
{
  if (session == nullptr || base == nullptr || bytes == 0) { return 1; }
  try {
    return session->client->cuMemObjGetDescriptor(base, bytes) == CU_OBJ_SUCCESS ? 0 : 1;
  } catch (...) {
    return 1;
  }
}

// A GET on this path hands the descriptor to the request callback; no token
// is taken with cuMemObjGetRDMAToken, so there is none to return before the
// descriptor is released.
int32_t deregister_memory(cucascade_s3rdma_session* session, void* base) noexcept
{
  if (session == nullptr || base == nullptr) { return 1; }
  try {
    return session->client->cuMemObjPutDescriptor(base) == CU_OBJ_SUCCESS ? 0 : 1;
  } catch (...) {
    return 1;
  }
}

int64_t max_request_size(cucascade_s3rdma_session* session, void* base) noexcept
{
  if (session == nullptr || base == nullptr) { return -1; }
  try {
    return static_cast<int64_t>(session->client->cuMemObjGetMaxRequestCallbackSize(base));
  } catch (...) {
    return -1;
  }
}

int64_t get(cucascade_s3rdma_session* session,
            void* base,
            size_t length,
            size_t destination_offset,
            cucascade_s3rdma_request_fn request,
            void* context) noexcept
{
  if (session == nullptr || base == nullptr || request == nullptr) {
    return CUCASCADE_S3RDMA_NO_RESULT;
  }
  session->context.store(context, std::memory_order_release);
  session->request.store(request, std::memory_order_release);
  try {
    // The registered base with the destination as buf_offset: an interior
    // pointer would not name the registration.
    return static_cast<int64_t>(
      session->client->cuObjGet(session, base, length, 0, static_cast<loff_t>(destination_offset)));
  } catch (...) {
    return CUCASCADE_S3RDMA_NO_RESULT;
  }
}

const cucascade_s3rdma_api table{
  CUCASCADE_S3RDMA_ABI_VERSION,
  static_cast<uint32_t>(sizeof(cucascade_s3rdma_api)),
  &probe,
  &session_create,
  &session_destroy,
  &register_memory,
  &deregister_memory,
  &max_request_size,
  &get,
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const cucascade_s3rdma_api*
cucascade_s3rdma_binding(void)
{
  return &table;
}
