/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "http_fixture.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <exception>
#include <optional>
#include <stdexcept>

namespace cucascade::test::s3rdma {
namespace {

using namespace std::chrono_literals;

bool same_name(std::string_view a, std::string_view b)
{
  auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; };
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [&](char x, char y) {
           return lower(x) == lower(y);
         });
}

struct socket_owner {
  int fd;
  ~socket_owner() { (void)::close(fd); }
};

http_request parse_request(std::string_view wire, std::size_t connection)
{
  http_request out;
  out.connection    = connection;
  auto const eol    = wire.find("\r\n");
  auto const first  = wire.find(' ');
  auto const second = wire.find(' ', first == std::string_view::npos ? 0 : first + 1);
  if (eol == std::string_view::npos || first >= eol || second >= eol) {
    throw std::runtime_error("malformed HTTP request line");
  }
  out.method  = wire.substr(0, first);
  out.target  = wire.substr(first + 1, second - first - 1);
  out.version = wire.substr(second + 1, eol - second - 1);
  auto offset = eol + 2;
  while (offset < wire.size()) {
    auto const end = wire.find("\r\n", offset);
    if (end == offset) { break; }
    auto const colon = wire.find(':', offset);
    if (end == std::string_view::npos || colon >= end) {
      throw std::runtime_error("malformed HTTP request header");
    }
    auto value = wire.substr(colon + 1, end - colon - 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
      value.remove_suffix(1);
    }
    out.headers.emplace_back(wire.substr(offset, colon - offset), value);
    offset = end + 2;
  }
  return out;
}

}  // namespace

std::vector<std::string> http_request::values(std::string_view name) const
{
  std::vector<std::string> result;
  for (auto const& [key, value] : headers) {
    if (same_name(key, name)) { result.push_back(value); }
  }
  return result;
}

std::string http_request::value(std::string_view name) const
{
  auto const found = values(name);
  return found.size() == 1 ? found.front() : std::string{};
}

http_fixture::http_fixture()
{
  _listener = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (_listener < 0) { throw std::runtime_error("HTTP fixture socket failed"); }
  try {
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(_listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(_listener, 32) != 0) {
      throw std::runtime_error("HTTP fixture bind/listen failed");
    }
    socklen_t length = sizeof(address);
    if (::getsockname(_listener, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
      throw std::runtime_error("HTTP fixture getsockname failed");
    }
    _port   = ntohs(address.sin_port);
    _worker = std::thread([this] { serve(); });
  } catch (...) {
    (void)::close(_listener);
    throw;
  }
}

http_fixture::~http_fixture()
{
  {
    std::lock_guard lock{_mutex};
    _stop = true;
  }
  _changed.notify_all();
  if (_worker.joinable()) { _worker.join(); }
  (void)::close(_listener);
}

void http_fixture::respond(http_response response)
{
  if (response.chunk_bytes == 0) { throw std::invalid_argument("zero HTTP fixture chunk"); }
  std::lock_guard lock{_mutex};
  _responses.push_back(std::move(response));
}

std::string http_fixture::endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

std::vector<http_request> http_fixture::requests() const
{
  std::lock_guard lock{_mutex};
  return _requests;
}

std::string http_fixture::failure() const
{
  std::lock_guard lock{_mutex};
  return _failure;
}

bool http_fixture::wait_for_requests(std::size_t count) const
{
  std::unique_lock lock{_mutex};
  return _changed.wait_for(lock, 2s, [&] { return _requests.size() >= count; });
}

bool http_fixture::wait_for_peer_closes(std::size_t count) const
{
  std::unique_lock lock{_mutex};
  return _changed.wait_for(lock, 2s, [&] {
    return static_cast<std::size_t>(std::count_if(
             _requests.begin(), _requests.end(), [](auto const& r) { return r.peer_closed; })) >=
           count;
  });
}

bool http_fixture::stopped() const
{
  std::lock_guard lock{_mutex};
  return _stop;
}

bool http_fixture::pause(std::chrono::milliseconds duration)
{
  std::unique_lock lock{_mutex};
  return !_changed.wait_for(lock, duration, [&] { return _stop; });
}

void http_fixture::fail(std::string message)
{
  std::lock_guard lock{_mutex};
  if (_failure.empty()) { _failure = std::move(message); }
}

bool http_fixture::send_all(int fd, std::string_view bytes)
{
  auto const end = std::chrono::steady_clock::now() + 2s;
  while (!bytes.empty() && !stopped()) {
    if (std::chrono::steady_clock::now() >= end) { return false; }
    auto const sent = ::send(fd, bytes.data(), bytes.size(), MSG_NOSIGNAL);
    if (sent > 0) {
      bytes.remove_prefix(static_cast<std::size_t>(sent));
    } else if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      pollfd event{fd, POLLOUT, 0};
      (void)::poll(&event, 1, 20);
    } else {
      return false;
    }
  }
  return bytes.empty();
}

void http_fixture::serve() noexcept
{
  try {
    std::size_t connection = 0;
    while (!stopped()) {
      pollfd event{_listener, POLLIN, 0};
      if (::poll(&event, 1, 20) <= 0) { continue; }
      int const fd = ::accept4(_listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0) {
        if (errno == EAGAIN || errno == EINTR) { continue; }
        throw std::runtime_error("HTTP fixture accept failed");
      }
      socket_owner owner{fd};
      serve_connection(fd, ++connection);
    }
  } catch (std::exception const& e) {
    fail(e.what());
  } catch (...) {
    fail("HTTP fixture failed");
  }
}

void http_fixture::serve_connection(int fd, std::size_t connection)
{
  std::string wire;
  std::optional<std::size_t> previous;
  while (!stopped()) {
    auto const end = std::chrono::steady_clock::now() + 3s;
    while (wire.find("\r\n\r\n") == std::string::npos && !stopped()) {
      if (std::chrono::steady_clock::now() >= end) { return; }
      pollfd event{fd, POLLIN, 0};
      if (::poll(&event, 1, 20) <= 0) { continue; }
      std::array<char, 4096> bytes{};
      auto const count = ::recv(fd, bytes.data(), bytes.size(), 0);
      if (count == 0 || (count < 0 && errno == ECONNRESET)) {
        if (previous) {
          {
            std::lock_guard lock{_mutex};
            _requests[*previous].peer_closed = true;
          }
          _changed.notify_all();
        }
        return;
      }
      if (count < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) { continue; }
        throw std::runtime_error("HTTP fixture receive failed");
      }
      wire.append(bytes.data(), static_cast<std::size_t>(count));
      if (wire.size() > 65536) { throw std::runtime_error("HTTP fixture request too large"); }
    }
    if (stopped()) { return; }
    auto const size = wire.find("\r\n\r\n") + 4;
    auto request    = parse_request(std::string_view{wire}.substr(0, size), connection);
    wire.erase(0, size);
    auto const head_only = request.method == "HEAD";
    http_response response;
    std::size_t index = 0;
    {
      std::lock_guard lock{_mutex};
      index = _requests.size();
      _requests.push_back(std::move(request));
      if (_responses.empty()) {
        response.status = 500;
        _failure        = "HTTP fixture has no response for a request";
      } else {
        response = std::move(_responses.front());
        _responses.pop_front();
      }
    }
    _changed.notify_all();
    if (response.before_headers && !response.before_headers->hold()) {
      fail("HTTP fixture response gate timed out");
      return;
    }
    if (!pause(response.header_delay)) { return; }
    std::string header = "HTTP/1.1 " + std::to_string(response.status) + " Fixture\r\n";
    bool has_length    = false;
    for (auto const& [name, value] : response.headers) {
      has_length = has_length || same_name(name, "Content-Length");
      header += name + ": " + value + "\r\n";
    }
    if (response.add_content_length && !has_length) {
      header += "Content-Length: " + std::to_string(response.body.size()) + "\r\n";
    }
    header +=
      response.close_connection ? "Connection: close\r\n\r\n" : "Connection: keep-alive\r\n\r\n";
    if (!send_all(fd, header)) { return; }
    std::size_t sent = 0;
    bool paused      = false;
    while (!head_only && sent < response.body.size()) {
      if (!paused && sent >= response.pause_after_bytes) {
        paused = true;
        if (!pause(response.body_pause)) { return; }
      }
      auto const count = std::min(response.chunk_bytes, response.body.size() - sent);
      if (!send_all(fd, std::string_view{response.body}.substr(sent, count))) { return; }
      sent += count;
      {
        std::lock_guard lock{_mutex};
        _requests[index].body_sent = sent;
      }
    }
    {
      std::lock_guard lock{_mutex};
      _requests[index].response_finished = true;
    }
    _changed.notify_all();
    if (response.close_connection) { return; }
    previous = index;
  }
}

}  // namespace cucascade::test::s3rdma
