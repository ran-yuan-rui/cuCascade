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

#include <cerrno>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>

struct cucascade_s3rdma_session {
  // Handed to the client by reference, so it lives as long as the client.
  CUObjOps_t ops{};
  cuObjClient* client{nullptr};
};

namespace {

// The number of a GET is carried where the client keeps a pointer.
static_assert(sizeof(void*) == sizeof(std::uint64_t));

// One GET in progress: its number, its request callback and the context that
// callback runs with.  It lives in the frame of get.
struct get_call {
  std::uint64_t number{0};
  cucascade_s3rdma_request_fn request{nullptr};
  void* context{nullptr};
  get_call* next{nullptr};
};

// The GETs in progress, each under a number that is never used twice.  The
// client is handed that number, not an address, so a callback it runs after
// its GET returned finds nothing here and can never be taken for a later GET.
// It allocates nothing and needs neither construction nor destruction: the
// client may run a callback while the process starts or exits.
class call_registry {
 public:
  constexpr call_registry() = default;

  // False when every number has been used.
  [[nodiscard]] bool enter(get_call& call)
  {
    std::lock_guard lock{_mutex};
    if (_issued == std::numeric_limits<std::uint64_t>::max()) { return false; }
    call.number = ++_issued;
    call.next   = _first;
    _first      = &call;
    return true;
  }

  // An entry that cannot be removed would outlive its frame, so a failure to
  // take the lock ends the process.
  void leave(get_call& call) noexcept
  {
    std::lock_guard lock{_mutex};
    for (auto** link = &_first; *link != nullptr; link = &(*link)->next) {
      if (*link == &call) {
        *link = call.next;
        return;
      }
    }
  }

  [[nodiscard]] bool find(std::uint64_t number,
                          cucascade_s3rdma_request_fn& request,
                          void*& context)
  {
    std::lock_guard lock{_mutex};
    for (auto const* call = _first; call != nullptr; call = call->next) {
      if (call->number == number) {
        request = call->request;
        context = call->context;
        return true;
      }
    }
    return false;
  }

 private:
  std::mutex _mutex;
  std::uint64_t _issued{0};
  get_call* _first{nullptr};
};

static_assert(std::is_trivially_destructible_v<call_registry>);

constinit call_registry calls;

// Nothing may leave a callback through the client's frames.
ssize_t on_get(const void* handle,
               char* /*ptr*/,
               size_t size,
               loff_t offset,
               const cufileRDMAInfo_t* rdma) noexcept
{
  cucascade_s3rdma_request_fn request = nullptr;
  void* context                       = nullptr;
  try {
    auto const number =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(cuObjClient::getCtx(handle)));
    if (!calls.find(number, request, context)) { return -ENOTSUP; }
  } catch (...) {
    return -ENOTSUP;
  }
  // The client states the length as a signed value; a negative one describes
  // no descriptor.
  if (rdma != nullptr && rdma->desc_len < 0) { return -ENOTSUP; }
  cucascade_s3rdma_request what{};
  what.descriptor        = rdma != nullptr ? rdma->desc_str : nullptr;
  what.descriptor_length = rdma != nullptr ? static_cast<size_t>(rdma->desc_len) : 0;
  what.size              = size;
  what.offset            = static_cast<int64_t>(offset);
  return static_cast<ssize_t>(request(context, &what));
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
    auto session     = std::make_unique<cucascade_s3rdma_session>();
    session->ops.get = &on_get;
    session->ops.put = nullptr;
    session->client  = new cuObjClient(session->ops, CUOBJ_PROTO_RDMA_DC_V1);
    *out             = session.release();
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
  if (session == nullptr || base == nullptr || request == nullptr || context == nullptr) {
    return CUCASCADE_S3RDMA_NO_RESULT;
  }
  get_call call;
  call.request = request;
  call.context = context;
  try {
    if (!calls.enter(call)) { return CUCASCADE_S3RDMA_NO_RESULT; }
  } catch (...) {
    return CUCASCADE_S3RDMA_NO_RESULT;
  }
  struct leave_at_return {
    get_call& call;
    ~leave_at_return() { calls.leave(call); }
  } const leaving{call};
  try {
    // The registered base with the destination as buf_offset: an interior
    // pointer would not name the registration.
    return static_cast<int64_t>(
      session->client->cuObjGet(reinterpret_cast<void*>(static_cast<std::uintptr_t>(call.number)),
                                base,
                                length,
                                0,
                                static_cast<loff_t>(destination_offset)));
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
