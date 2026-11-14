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

/*
 * Binary interface between the S3-over-RDMA component and a library that
 * binds it to an RDMA object client.  The component loads such a library at
 * run time and never links one.
 *
 * Only plain data, opaque handles and create/destroy pairs cross this
 * interface: no C++ object, no exception, no memory allocated on one side and
 * released on the other.  A function of the binding must not let an exception
 * escape.  This header is valid C and C++.
 */

#ifndef CUCASCADE_IO_S3RDMA_CUOBJ_ABI_H
#define CUCASCADE_IO_S3RDMA_CUOBJ_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Changes whenever a structure or the meaning of a function changes. */
#define CUCASCADE_S3RDMA_ABI_VERSION 1u

/* Name of the one function a binding library exports. */
#define CUCASCADE_S3RDMA_BINDING_SYMBOL "cucascade_s3rdma_binding"

/* Transfer protocols a binding can report. */
#define CUCASCADE_S3RDMA_PROTOCOL_DC_V1 1u

/* Result of get when the client produced none, for example because it threw. */
#define CUCASCADE_S3RDMA_NO_RESULT INT64_MIN

/* One data session; owned by the binding. */
typedef struct cucascade_s3rdma_session cucascade_s3rdma_session;

/* What the client hands the request callback of a GET. */
typedef struct cucascade_s3rdma_request {
  const char* descriptor;   /* RDMA descriptor of the destination; not terminated */
  size_t descriptor_length; /* its length in bytes */
  size_t size;              /* bytes this callback is asked to transfer */
  int64_t offset;           /* offset the client passes this callback */
} cucascade_s3rdma_request;

/*
 * Request callback of a GET.  Its result is handed to the client unchanged: a
 * byte count, or a negative error value.
 */
typedef int64_t (*cucascade_s3rdma_request_fn)(void* context,
                                               const cucascade_s3rdma_request* request);

/* What a binding reports about itself and the client it drives. */
typedef struct cucascade_s3rdma_probe {
  uint32_t protocol;      /* CUCASCADE_S3RDMA_PROTOCOL_* its sessions use */
  int32_t io_retry_count; /* the client's effective I/O retry count; -1 if unknown */
} cucascade_s3rdma_probe;

/*
 * The functions of a binding.  Every int32_t result is 0 on success and
 * nonzero otherwise.
 */
typedef struct cucascade_s3rdma_api {
  uint32_t abi_version; /* CUCASCADE_S3RDMA_ABI_VERSION the binding was built for */
  uint32_t api_size;    /* sizeof(cucascade_s3rdma_api) as the binding sees it */

  /* Reports the binding's properties; creates no session. */
  int32_t (*probe)(cucascade_s3rdma_probe* out);

  int32_t (*session_create)(cucascade_s3rdma_session** out);
  void (*session_destroy)(cucascade_s3rdma_session* session);

  /* Registers [base, base + bytes).  On failure nothing stays registered. */
  int32_t (*register_memory)(cucascade_s3rdma_session* session, void* base, size_t bytes);

  /*
   * Releases the registration that starts at base.  0 only when every
   * resource held for it was returned and the registration itself is gone.
   */
  int32_t (*deregister_memory)(cucascade_s3rdma_session* session, void* base);

  /* Largest transfer one request callback can carry for base; negative if unknown. */
  int64_t (*max_request_size)(cucascade_s3rdma_session* session, void* base);

  /*
   * One GET of length bytes into base + destination_offset, base being
   * exactly a registered base.  The binding has the client run request with
   * context and returns the client's result unchanged, or
   * CUCASCADE_S3RDMA_NO_RESULT when there is none.  The client may run
   * request on another thread, more than once, and after get returned.
   */
  int64_t (*get)(cucascade_s3rdma_session* session,
                 void* base,
                 size_t length,
                 size_t destination_offset,
                 cucascade_s3rdma_request_fn request,
                 void* context);
} cucascade_s3rdma_api;

/*
 * The table of a binding library, valid for the life of the process.  A
 * binding defines this function with default visibility.
 */
const cucascade_s3rdma_api* cucascade_s3rdma_binding(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUCASCADE_IO_S3RDMA_CUOBJ_ABI_H */
