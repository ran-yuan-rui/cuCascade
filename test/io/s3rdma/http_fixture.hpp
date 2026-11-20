/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::test::s3rdma {

using http_headers = std::vector<std::pair<std::string, std::string>>;

class callback_gate {
 public:
  bool hold()
  {
    std::unique_lock lock{_mutex};
    ++_entries;
    _changed.notify_all();
    if (!_changed.wait_for(lock, std::chrono::seconds{3}, [&] { return _released; })) {
      _timed_out = true;
      return false;
    }
    return true;
  }
  bool wait_for_entries(unsigned count = 1)
  {
    std::unique_lock lock{_mutex};
    return _changed.wait_for(lock, std::chrono::seconds{2}, [&] { return _entries >= count; });
  }
  void release()
  {
    std::lock_guard lock{_mutex};
    _released = true;
    _changed.notify_all();
  }
  bool timed_out() const
  {
    std::lock_guard lock{_mutex};
    return _timed_out;
  }

 private:
  mutable std::mutex _mutex;
  std::condition_variable _changed;
  unsigned _entries{0};
  bool _released{false};
  bool _timed_out{false};
};

struct http_request {
  std::size_t connection{0};
  std::string method;
  std::string target;
  std::string version;
  http_headers headers;
  std::size_t body_sent{0};
  bool response_finished{false};
  bool peer_closed{false};

  [[nodiscard]] std::vector<std::string> values(std::string_view name) const;
  [[nodiscard]] std::string value(std::string_view name) const;
};

struct http_response {
  int status{200};
  http_headers headers;
  std::string body;
  bool add_content_length{true};
  bool close_connection{true};
  std::chrono::milliseconds header_delay{0};
  std::size_t chunk_bytes{16384};
  std::size_t pause_after_bytes{0};
  std::chrono::milliseconds body_pause{0};
  std::shared_ptr<callback_gate> before_headers{};
};

// Records complete wire requests; declines h2c upgrades with a normal HTTP/1.1 response.
class http_fixture {
 public:
  http_fixture();
  ~http_fixture();
  http_fixture(http_fixture const&)            = delete;
  http_fixture& operator=(http_fixture const&) = delete;

  void respond(http_response response);
  [[nodiscard]] std::string endpoint() const;
  [[nodiscard]] std::vector<http_request> requests() const;
  [[nodiscard]] std::string failure() const;
  [[nodiscard]] bool wait_for_requests(std::size_t count) const;
  [[nodiscard]] bool wait_for_peer_closes(std::size_t count) const;

 private:
  void serve() noexcept;
  void serve_connection(int fd, std::size_t connection);
  bool stopped() const;
  bool pause(std::chrono::milliseconds duration);
  bool send_all(int fd, std::string_view bytes);
  void fail(std::string message);

  int _listener{-1};
  unsigned short _port{0};
  mutable std::mutex _mutex;
  mutable std::condition_variable _changed;
  bool _stop{false};
  std::deque<http_response> _responses;
  std::vector<http_request> _requests;
  std::string _failure;
  std::thread _worker;
};

}  // namespace cucascade::test::s3rdma
