/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sys/types.h>

#include <cstddef>

typedef struct cufileRDMAInfo {
  int version;
  int desc_len;
  const char* desc_str;
} cufileRDMAInfo_t;

typedef struct CUObjOps {
  ssize_t (*get)(const void*, char*, size_t, loff_t, const cufileRDMAInfo_t*);
  ssize_t (*put)(const void*, const char*, size_t, loff_t, const cufileRDMAInfo_t*);
} CUObjOps_t;

enum cuObjErr_t { CU_OBJ_SUCCESS = 0, CU_OBJ_TEST_ERROR = 1 };
enum cuObjProto_t { CUOBJ_PROTO_RDMA_DC_V1 = 1001 };

class cuObjClient {
 public:
  cuObjClient(CUObjOps_t& ops, cuObjProto_t proto = CUOBJ_PROTO_RDMA_DC_V1);
  ~cuObjClient();
  cuObjErr_t cuMemObjGetDescriptor(void* base, size_t bytes);
  ssize_t cuMemObjGetMaxRequestCallbackSize(void* base);
  cuObjErr_t cuMemObjPutDescriptor(void* base);
  ssize_t cuObjGet(void* context, void* ptr, size_t size, loff_t offset = 0, loff_t buf_offset = 0);
  static void* getCtx(const void* handle);
  bool isConnected(void);

 private:
  CUObjOps_t* _ops;
};
