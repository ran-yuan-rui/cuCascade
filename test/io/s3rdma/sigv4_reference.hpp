/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <array>
#include <climits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::test::s3rdma {

inline std::string hex_bytes(std::string_view bytes)
{
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : bytes) {
    result.push_back(alphabet[byte >> 4]);
    result.push_back(alphabet[byte & 15]);
  }
  return result;
}

inline std::string reference_sha256(std::string_view value)
{
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int length = 0;
  if (EVP_Digest(value.data(), value.size(), bytes.data(), &length, EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("reference SHA256 failed");
  }
  return hex_bytes({reinterpret_cast<char const*>(bytes.data()), length});
}

inline std::string reference_hmac(std::string_view key, std::string_view value)
{
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int length = 0;
  if (key.size() > INT_MAX || HMAC(EVP_sha256(),
                                   key.data(),
                                   static_cast<int>(key.size()),
                                   reinterpret_cast<unsigned char const*>(value.data()),
                                   value.size(),
                                   bytes.data(),
                                   &length) == nullptr) {
    throw std::runtime_error("reference HMAC failed");
  }
  return {reinterpret_cast<char const*>(bytes.data()), length};
}

inline std::string signed_header_value(
  std::vector<std::pair<std::string, std::string>> const& headers, std::string_view name)
{
  for (auto const& [key, value] : headers) {
    auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; };
    if (key.size() == name.size() &&
        std::equal(key.begin(), key.end(), name.begin(), [&](char x, char y) {
          return lower(x) == lower(y);
        })) {
      auto result = std::string_view{value};
      while (!result.empty() && result.front() == ' ') {
        result.remove_prefix(1);
      }
      while (!result.empty() && result.back() == ' ') {
        result.remove_suffix(1);
      }
      return std::string{result};
    }
  }
  throw std::runtime_error("reference signature is missing a signed header");
}

// Rebuilds the canonical request and signing key from wire fields and known test credentials.
inline std::string reference_sigv4(std::vector<std::pair<std::string, std::string>> const& headers,
                                   std::string_view method,
                                   std::string_view path,
                                   std::string_view signed_names,
                                   std::string_view secret,
                                   std::string_view region = "us-east-1")
{
  std::string canonical;
  std::size_t begin = 0;
  while (begin < signed_names.size()) {
    auto end = signed_names.find(';', begin);
    if (end == std::string_view::npos) { end = signed_names.size(); }
    auto const name = signed_names.substr(begin, end - begin);
    canonical += std::string{name} + ":" + signed_header_value(headers, name) + "\n";
    begin = end + 1;
  }
  auto const date    = signed_header_value(headers, "x-amz-date");
  auto const scope   = date.substr(0, 8) + "/" + std::string{region} + "/s3/aws4_request";
  auto const request = std::string{method} + "\n" + std::string{path} + "\n\n" + canonical + "\n" +
                       std::string{signed_names} + "\n" +
                       signed_header_value(headers, "x-amz-content-sha256");
  auto key = reference_hmac("AWS4" + std::string{secret}, date.substr(0, 8));
  key      = reference_hmac(key, region);
  key      = reference_hmac(key, "s3");
  key      = reference_hmac(key, "aws4_request");
  return hex_bytes(reference_hmac(
    key, "AWS4-HMAC-SHA256\n" + date + "\n" + scope + "\n" + reference_sha256(request)));
}

}  // namespace cucascade::test::s3rdma
