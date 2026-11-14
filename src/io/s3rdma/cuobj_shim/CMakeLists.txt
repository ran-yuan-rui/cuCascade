# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# The library that binds the S3-over-RDMA component to the cuObject client
# (CUCASCADE_BUILD_CUOBJ_SHIM). It needs the cuObject SDK; the component does
# not. It builds inside cuCascade, or on its own against an installed cuCascade:
# point CUCASCADE_S3RDMA_ABI_INCLUDE_DIR at the directory that holds
# cucascade/io/s3rdma/cuobj_abi.h.
cmake_minimum_required(VERSION 3.26.4)

include(GNUInstallDirs)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cucascade_s3rdma_cuobj LANGUAGES CXX)
  find_package(CUDAToolkit REQUIRED)
  set(CUCASCADE_S3RDMA_ABI_INCLUDE_DIR
      ""
      CACHE PATH "Directory that holds cucascade/io/s3rdma/cuobj_abi.h")
else()
  set(CUCASCADE_S3RDMA_ABI_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
endif()

set(CUCASCADE_CUOBJ_ROOT
    ""
    CACHE PATH "Prefix of the cuObject SDK; empty searches the CUDA toolkit")

find_path(
  CUCASCADE_CUOBJCLIENT_INCLUDE_DIR cuobjclient.h
  HINTS ${CUCASCADE_CUOBJ_ROOT} ${CUDAToolkit_TARGET_DIR}
  PATH_SUFFIXES include REQUIRED)
find_library(
  CUCASCADE_CUOBJCLIENT_LIBRARY cuobjclient
  HINTS ${CUCASCADE_CUOBJ_ROOT} ${CUDAToolkit_TARGET_DIR}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu REQUIRED)

add_library(cucascade_s3rdma_cuobj MODULE cuobj_binding.cpp)
target_compile_features(cucascade_s3rdma_cuobj PRIVATE cxx_std_20)
set_target_properties(
  cucascade_s3rdma_cuobj PROPERTIES CXX_EXTENSIONS OFF CXX_VISIBILITY_PRESET
                                                       hidden)
target_include_directories(
  cucascade_s3rdma_cuobj PRIVATE ${CUCASCADE_S3RDMA_ABI_INCLUDE_DIR}
                                 ${CUCASCADE_CUOBJCLIENT_INCLUDE_DIR})
target_link_libraries(
  cucascade_s3rdma_cuobj PRIVATE ${CUCASCADE_CUOBJCLIENT_LIBRARY} CUDA::cuFile
                                 CUDA::cudart)

install(TARGETS cucascade_s3rdma_cuobj
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
