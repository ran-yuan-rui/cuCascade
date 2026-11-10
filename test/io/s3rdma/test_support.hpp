/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cucascade/io/io_request.hpp>
#include <cucascade/io/s3rdma/errors.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace cucascade::test::s3rdma {

using namespace std::chrono_literals;
namespace rdma = cucascade::io::s3rdma;

// Must outlive the gate and worker threads: a broken teardown must fail the process, not hang.
class deadline {
 public:
  deadline()
    : _worker([this] {
        std::unique_lock lock(_mutex);
        if (!_cv.wait_for(lock, 10s, [this] { return _finished; })) {
          std::fputs("S3RDMA test deadline expired\n", stderr);
          std::abort();
        }
      })
  {
  }
  ~deadline()
  {
    {
      std::lock_guard lock(_mutex);
      _finished = true;
    }
    _cv.notify_one();
    _worker.join();
  }
  deadline(deadline const&)            = delete;
  deadline& operator=(deadline const&) = delete;

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _finished{false};
  std::thread _worker;
};

class workers {
 public:
  ~workers()
  {
    for (auto& thread : _threads) {
      thread.join();
    }
  }
  template <typename F>
  auto launch(F&& function)
  {
    std::packaged_task<std::invoke_result_t<F>()> task(std::forward<F>(function));
    auto future = task.get_future();
    _threads.emplace_back(std::move(task));
    return future;
  }

 private:
  std::vector<std::thread> _threads;
};

template <typename F>
bool wait_until(F&& predicate)
{
  auto const end = std::chrono::steady_clock::now() + 2s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= end) { return false; }
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

template <typename T>
T ready(std::future<T>& future)
{
  REQUIRE(future.wait_for(2s) == std::future_status::ready);
  return future.get();
}

inline bool has_code(std::exception_ptr error, rdma::errc expected)
{
  if (error == nullptr) { return false; }
  try {
    std::rethrow_exception(error);
  } catch (rdma::error const& e) {
    return e.code() == expected;
  } catch (...) {
    return false;
  }
}

template <typename F>
void require_code(F&& function, rdma::errc expected)
{
  std::exception_ptr caught;
  try {
    std::forward<F>(function)();
  } catch (...) {
    caught = std::current_exception();
  }
  REQUIRE(has_code(caught, expected));
}

inline std::exception_ptr failure(rdma::errc code)
{
  return std::make_exception_ptr(rdma::error{code, "test failure"});
}

class object final : public io::io_object {
 public:
  std::string const& raw_file_cache_id() const noexcept override { return _path; }
  std::string const& object_path() const noexcept override { return _path; }
  std::size_t size() const noexcept override { return 4096; }

 private:
  std::string _path{"s3://test/object"};
};

struct request_fixture {
  std::array<std::uint8_t, 256> destination{};
  std::shared_ptr<object> source{std::make_shared<object>()};

  io::prepared_io_slice slice(std::size_t offset, std::size_t size, std::size_t landing = 0)
  {
    return io::prepared_io_slice{
      io::range{offset, size},
      io::device_buffer{
        destination.data() + landing, ::cuda::stream_ref{cudaStream_t{nullptr}}, 7}};
  }

  std::unique_ptr<io::grouped_io_request> request(std::size_t size = 1)
  {
    return io::grouped_io_request::create(source,
                                          std::vector<io::prepared_io_slice>{slice(0, size)});
  }
};

}  // namespace cucascade::test::s3rdma
