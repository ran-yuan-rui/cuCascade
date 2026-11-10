cmake_minimum_required(VERSION 3.26)
project(s3rdma_install_consumer LANGUAGES CXX)

find_package(cuCascade REQUIRED COMPONENTS s3rdma)
if(NOT cuCascade_s3rdma_FOUND)
  message(FATAL_ERROR "The requested s3rdma component is unavailable")
endif()

add_executable(s3rdma_install_consumer main.cpp)
target_link_libraries(s3rdma_install_consumer
                      PRIVATE cuCascade::cucascade_io_s3rdma)
target_compile_options(s3rdma_install_consumer PRIVATE -Wall -Wextra -Wpedantic
                                                       -Werror)
