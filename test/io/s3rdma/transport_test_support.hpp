/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "http_fixture.hpp"
#include "test_binding_api.hpp"
#include "test_support.hpp"

#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/s3rdma/control_client.hpp>
#include <cucascade/io/s3rdma/data_plane.hpp>
#include <cucascade/log/logging.hpp>

#include <dlfcn.h>

#include <atomic>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::test::s3rdma {

namespace rest = cucascade::io::rest;

inline bool hold_callback(void* context) { return static_cast<callback_gate*>(context)->hold(); }

// Declare after workers so exceptional exits release gates before joining.
class release_callbacks {
 public:
  release_callbacks(std::initializer_list<std::shared_ptr<callback_gate>> gates) : gates(gates) {}
  ~release_callbacks()
  {
    for (auto const& gate : gates) {
      gate->release();
    }
  }

 private:
  std::vector<std::shared_ptr<callback_gate>> gates;
};

inline std::shared_ptr<rest::s3::sigv4_header_authorizer> signer(
  std::string endpoint, std::string key = "control-fixture", std::string secret = "control-secret")
{
  rest::s3::static_credentials credentials;
  credentials.access_key_id     = std::move(key);
  credentials.secret_access_key = std::move(secret);
  return std::make_shared<rest::s3::sigv4_header_authorizer>(
    std::move(credentials), "us-east-1", std::move(endpoint));
}

class wire_authorizer final : public rest::request_authorizer {
 public:
  explicit wire_authorizer(std::string endpoint) : endpoint(std::move(endpoint)) {}

  rest::authorized_request authorize(rest::object_ref const& object,
                                     rest::request_method,
                                     std::chrono::seconds) override
  {
    ++calls;
    if (reject) { throw cucascade::io::credential_error(message); }
    return {endpoint + "/" + object.bucket + "/" + object.key, {{"Authorization", "fixture"}}};
  }

  rest::authorized_request authorize_with_headers(rest::object_ref const& object,
                                                  rest::request_method method,
                                                  http_headers const& headers,
                                                  std::chrono::seconds timeout) override
  {
    if (!signs_headers) {
      return rest::request_authorizer::authorize_with_headers(object, method, headers, timeout);
    }
    auto result = authorize(object, method, timeout);
    result.headers.insert(result.headers.end(), headers.begin(), headers.end());
    return result;
  }

  rest::authorized_request authorize_list(std::string_view bucket,
                                          std::string_view query,
                                          std::chrono::seconds) override
  {
    ++calls;
    if (reject) { throw cucascade::io::credential_error(message); }
    return {endpoint + "/" + std::string{bucket} + "?" + std::string{query},
            {{"Authorization", "fixture"}}};
  }

  std::string endpoint;
  std::atomic<unsigned> calls{0};
  bool reject{false};
  bool signs_headers{true};
  std::string message{"authorization refused"};
};

class log_capture {
 public:
  log_capture()
    : threshold(cucascade::log::get_min_level()),
      previous(cucascade::log::set_sink(
        cucascade::log::make_sink([this](cucascade::log::record const& record) {
          std::lock_guard lock{mutex};
          records.emplace_back(record.lvl, record.message);
        }),
        cucascade::log::level::trace))
  {
  }
  ~log_capture()
  {
    cucascade::log::set_min_level(threshold);
    (void)cucascade::log::set_sink(std::move(previous));
  }
  auto snapshot() const
  {
    std::lock_guard lock{mutex};
    return records;
  }

 private:
  mutable std::mutex mutex;
  std::vector<std::pair<cucascade::log::level, std::string>> records;
  cucascade::log::level threshold;
  std::shared_ptr<cucascade::log::sink> previous;
};

inline std::string binding_path(std::string_view variant)
{
  return std::string{S3RDMA_TEST_BINDING_DIR} + "/cucascade_s3rdma_test_binding_" +
         std::string{variant} + ".so";
}

inline std::string cuobj_test_path()
{
  return std::string{S3RDMA_TEST_BINDING_DIR} + "/cucascade_s3rdma_cuobj_under_test.so";
}

class module_view {
 public:
  explicit module_view(std::string const& path, bool load = false)
    : handle(::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | (load ? 0 : RTLD_NOLOAD)))
  {
    if (handle == nullptr) { throw std::runtime_error("test module is not loaded"); }
  }
  ~module_view() { (void)::dlclose(handle); }
  module_view(module_view const&)            = delete;
  module_view& operator=(module_view const&) = delete;

  template <typename Function>
  Function symbol(char const* name) const
  {
    auto* address = ::dlsym(handle, name);
    if (address == nullptr) {
      throw std::runtime_error(std::string{"missing test symbol: "} + name);
    }
    Function result{};
    static_assert(sizeof(result) == sizeof(address));
    std::memcpy(&result, &address, sizeof(result));
    return result;
  }
  void reset() const { symbol<reset_binding_fn>("test_binding_reset")(); }
  void configure(binding_options const& options) const
  {
    symbol<configure_binding_fn>("test_binding_configure")(&options);
  }
  binding_snapshot snapshot() const
  {
    binding_snapshot result;
    symbol<snapshot_binding_fn>("test_binding_snapshot")(&result);
    return result;
  }
  std::int64_t fire(std::size_t index) const
  {
    return symbol<fire_binding_fn>("test_binding_fire")(index);
  }

 private:
  void* handle;
};

inline bool is_loaded(std::string const& path)
{
  auto* handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (handle == nullptr) { return false; }
  (void)::dlclose(handle);
  return true;
}

inline rdma::completion_rule accepted_rule()
{
  return {{206}, [](std::string_view reply, long status) noexcept {
            return reply == "accepted" && status == 206;
          }};
}

inline http_response accepted_response(std::string tag = "\"version-a\"")
{
  http_response response;
  response.status  = 206;
  response.headers = {{"x-amz-rdma-reply", "accepted"}};
  if (!tag.empty()) { response.headers.emplace_back("ETag", std::move(tag)); }
  response.close_connection = false;
  return response;
}

inline rdma::data_plane_config data_config(std::string_view variant = "good")
{
  rdma::data_plane_config config;
  config.library_path     = binding_path(variant);
  config.request_deadline = 500ms;
  return config;
}

inline void no_token(log_capture const& logs, std::string_view message = {})
{
  constexpr std::string_view token{"CAPABILITY-NOT-FOR-LOGS"};
  CHECK(message.find(token) == std::string_view::npos);
  for (auto const& [level, text] : logs.snapshot()) {
    CAPTURE(static_cast<int>(level), text);
    CHECK(text.find(token) == std::string::npos);
  }
}

}  // namespace cucascade::test::s3rdma
