/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <exception>
#include <string_view>

using namespace cucascade::test::s3rdma;

namespace {

void marker(std::string_view message) noexcept
{
  if (::write(STDOUT_FILENO, message.data(), message.size()) !=
      static_cast<ssize_t>(message.size())) {
    std::_Exit(75);
  }
}

struct unwind_marker {
  ~unwind_marker() { marker("CLEANUP\n"); }
};

}  // namespace

int main(int argc, char** argv)
{
  rlimit no_core{0, 0};
  (void)setrlimit(RLIMIT_CORE, &no_core);
  std::set_terminate([] {
    marker("TERMINATE\n");
    std::_Exit(86);
  });
  if (argc != 2) { return 64; }
  std::string_view mode{argv[1]};
  if (mode == "default-full-pipe") {
    auto ops = rdma::default_delivery_ops();
    marker("BOUNDARY_REACHED\n");
    rdma::invoke_fatal(ops, "delivery failure", cudaErrorInvalidValue);
  }
  if (mode == "hook-return" || mode == "hook-throw") {
    auto controls  = std::make_shared<runtime_controls>();
    auto ops       = controls->delivery();
    ops.fatal_hook = [mode](char const*, cudaError_t) {
      marker("HOOK\n");
      if (mode == "hook-throw") { throw std::runtime_error("hook threw"); }
    };
    unwind_marker guard;
    marker("BOUNDARY_REACHED\n");
    rdma::invoke_fatal(ops, "delivery failure", cudaErrorInvalidValue);
  }
  try {
    unwind_marker guard;
    std::atomic<bool> boundary{false};
    runtime_fixture f;
    f.controls->log.observer = [&](observation const& event) {
      if (boundary.load() && !event.returned) { marker("CLEANUP\n"); }
    };
    auto ops       = f.controls->delivery();
    ops.fatal_hook = [](char const*, cudaError_t) {
      marker("HOOK\n");
      std::terminate();
    };
    auto policy = f.controls->policy();
    if (mode == "policy-throw") {
      policy = [](rdma::context_fatal_report const&) {
        marker("POLICY_REACHED\n");
        throw std::runtime_error("policy threw");
      };
    }
    f.context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(
      f.cfg, f.controls->clients(), std::move(ops), std::move(policy));
    f.reactor = std::make_unique<rdma::s3rdma_reactor>(f.context);
    f.start();
    action cleanup;
    cleanup.before = [](auto const&) { marker("CLEANUP\n"); };
    f.controls->script(operation::deregister_memory, cleanup);
    f.controls->script(operation::free, cleanup);
    action fault;
    fault.before = [&](auto const&) {
      marker("BOUNDARY_REACHED\n");
      boundary.store(true);
    };
    fault.throws = mode.ends_with("-throw");
    fault.result = cudaErrorInvalidValue;
    operation point;
    if (mode.starts_with("copy-")) {
      point = operation::copy;
    } else if (mode.starts_with("record-")) {
      point = operation::record;
    } else if (mode.starts_with("wait-")) {
      point = operation::wait;
    } else if (mode == "policy-throw") {
      point        = operation::capture;
      fault.before = {};
      fault.throws = false;
      fault.result = cudaErrorIllegalAddress;
    } else {
      return 64;
    }
    f.controls->script(point, fault);
    auto completion   = std::make_shared<completion_record>();
    completion->after = [] { marker("FUTURE_SETTLED\n"); };
    auto read         = f.submit({f.slice(0, 4, f.destination.data())}, completion);
    try {
      (void)std::move(read.future).get(2s);
    } catch (...) {
    }
    marker("FUTURE_SETTLED\n");
    return 70;
  } catch (...) {
    marker("SETUP_OR_PROPAGATION_FAILURE\n");
    return 71;
  }
}
