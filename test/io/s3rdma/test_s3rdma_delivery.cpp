/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"

#include <cucascade/log/logging.hpp>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <string>

extern char** environ;

using namespace cucascade::test::s3rdma;

namespace {

struct pipe_pair {
  pipe_pair()
  {
    if (pipe2(descriptors.data(), O_CLOEXEC) != 0) {
      throw std::runtime_error("pipe creation failed");
    }
  }
  ~pipe_pair()
  {
    for (auto fd : descriptors) {
      if (fd >= 0) { (void)close(fd); }
    }
  }
  std::array<int, 2> descriptors{-1, -1};
};

struct child_guard {
  ~child_guard()
  {
    if (pid > 0) {
      (void)kill(pid, SIGKILL);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    }
  }
  pid_t pid{-1};
};

struct spawn_actions {
  spawn_actions()
  {
    if (posix_spawn_file_actions_init(&value) != 0) {
      throw std::runtime_error("spawn action initialization failed");
    }
  }
  ~spawn_actions() { (void)posix_spawn_file_actions_destroy(&value); }
  posix_spawn_file_actions_t value;
};

void dies(std::string mode, bool full_stderr = false, bool policy = false)
{
  deadline bound;
  auto const* helper = std::getenv("S3RDMA_DEATH_HELPER");
  REQUIRE(helper != nullptr);
  REQUIRE(access(helper, X_OK) == 0);
  pipe_pair output;
  pipe_pair diagnostics;
  REQUIRE(fcntl(output.descriptors[0], F_SETFL, O_NONBLOCK) == 0);
  if (full_stderr) {
    REQUIRE(fcntl(diagnostics.descriptors[1], F_SETFL, O_NONBLOCK) == 0);
    std::array<char, 4096> bytes{};
    while (write(diagnostics.descriptors[1], bytes.data(), bytes.size()) > 0) {}
    REQUIRE(errno == EAGAIN);
    REQUIRE(fcntl(diagnostics.descriptors[1], F_SETFL, 0) == 0);
  }
  spawn_actions actions;
  REQUIRE(posix_spawn_file_actions_adddup2(&actions.value, output.descriptors[1], STDOUT_FILENO) ==
          0);
  REQUIRE(posix_spawn_file_actions_addclose(&actions.value, output.descriptors[0]) == 0);
  REQUIRE(posix_spawn_file_actions_addclose(&actions.value, output.descriptors[1]) == 0);
  if (full_stderr) {
    REQUIRE(posix_spawn_file_actions_adddup2(
              &actions.value, diagnostics.descriptors[1], STDERR_FILENO) == 0);
  }
  std::string executable{helper};
  std::array<char*, 3> arguments{executable.data(), mode.data(), nullptr};
  child_guard child;
  REQUIRE(posix_spawn(&child.pid, helper, &actions.value, nullptr, arguments.data(), environ) == 0);
  int status  = 0;
  bool exited = wait_until([&] {
    auto rc = waitpid(child.pid, &status, WNOHANG);
    if (rc == child.pid) {
      child.pid = -1;
      return true;
    }
    return false;
  });
  if (!exited) {
    (void)kill(child.pid, SIGKILL);
    while (waitpid(child.pid, &status, 0) < 0 && errno == EINTR) {}
    child.pid = -1;
  }
  std::string markers;
  std::array<char, 256> bytes{};
  for (;;) {
    auto count = read(output.descriptors[0], bytes.data(), bytes.size());
    if (count > 0) {
      markers.append(bytes.data(), static_cast<std::size_t>(count));
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
  CAPTURE(mode, markers, status);
  REQUIRE(exited);
  CHECK(markers.find(policy ? "POLICY_REACHED\n" : "BOUNDARY_REACHED\n") != std::string::npos);
  CHECK(markers.find("FUTURE_SETTLED") == std::string::npos);
  CHECK(markers.find("CLEANUP") == std::string::npos);
  CHECK(markers.find("SETUP_OR_PROPAGATION_FAILURE") == std::string::npos);
  if (full_stderr) {
    CHECK(((WIFEXITED(status) && WEXITSTATUS(status) == 86) ||
           (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT)));
  } else {
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 86);
    CHECK(markers.find("TERMINATE\n") != std::string::npos);
    if (!policy) { CHECK(markers.find("HOOK\n") != std::string::npos); }
  }
}

struct sink_scope {
  explicit sink_scope(std::shared_ptr<cucascade::log::sink> next)
    : threshold(cucascade::log::get_min_level()),
      previous(cucascade::log::set_sink(std::move(next), cucascade::log::level::trace))
  {
  }
  ~sink_scope()
  {
    cucascade::log::set_min_level(threshold);
    (void)cucascade::log::set_sink(std::move(previous));
  }
  cucascade::log::level threshold;
  std::shared_ptr<cucascade::log::sink> previous;
};

}  // namespace

TEST_CASE("S3RDMA memcpy failure cannot return across the boundary", "[s3rdma][death]")
{
  dies("copy-error");
}
TEST_CASE("S3RDMA memcpy throw cannot unwind across the boundary", "[s3rdma][death]")
{
  dies("copy-throw");
}
TEST_CASE("S3RDMA event record failure cannot return across the boundary", "[s3rdma][death]")
{
  dies("record-error");
}
TEST_CASE("S3RDMA event record throw cannot unwind across the boundary", "[s3rdma][death]")
{
  dies("record-throw");
}
TEST_CASE("S3RDMA event wait failure cannot return across the boundary", "[s3rdma][death]")
{
  dies("wait-error");
}
TEST_CASE("S3RDMA event wait throw cannot unwind across the boundary", "[s3rdma][death]")
{
  dies("wait-throw");
}
TEST_CASE("S3RDMA a returning fatal hook is contained by termination", "[s3rdma][death]")
{
  dies("hook-return");
}
TEST_CASE("S3RDMA a throwing fatal hook is contained by termination", "[s3rdma][death]")
{
  dies("hook-throw");
}
TEST_CASE("S3RDMA default fatal diagnostics cannot block on a full pipe", "[s3rdma][death]")
{
  dies("default-full-pipe", true);
}
TEST_CASE("S3RDMA a throwing health policy cannot unwind cleanup", "[s3rdma][death]")
{
  dies("policy-throw", false, true);
}

TEST_CASE("S3RDMA context-fatal classification matches the frozen error set", "[s3rdma][delivery]")
{
  for (auto error : {cudaErrorECCUncorrectable,
                     cudaErrorNvlinkUncorrectable,
                     cudaErrorIllegalAddress,
                     cudaErrorLaunchTimeout,
                     cudaErrorAssert,
                     cudaErrorHardwareStackError,
                     cudaErrorIllegalInstruction,
                     cudaErrorMisalignedAddress,
                     cudaErrorInvalidAddressSpace,
                     cudaErrorInvalidPc,
                     cudaErrorLaunchFailure,
                     cudaErrorMpsClientTerminated,
                     cudaErrorExternalDevice}) {
    CAPTURE(static_cast<int>(error));
    CHECK(rdma::is_context_fatal(error));
  }
#if CUDART_VERSION >= 12080
  CHECK(rdma::is_context_fatal(cudaErrorContained));
  CHECK(rdma::is_context_fatal(cudaErrorTensorMemoryLeak));
#endif
  for (auto error : {cudaSuccess,
                     cudaErrorUnknown,
                     cudaErrorInvalidResourceHandle,
                     cudaErrorStreamCaptureUnsupported,
                     cudaErrorMemoryAllocation,
                     cudaErrorInvalidValue,
                     cudaErrorInvalidDevicePointer}) {
    CAPTURE(static_cast<int>(error));
    CHECK_FALSE(rdma::is_context_fatal(error));
  }
}

TEST_CASE("S3RDMA event destruction and logging cannot undo an already delivered chunk",
          "[s3rdma][delivery]")
{
  auto const leg = GENERATE(0, 1, 2);
  std::atomic<unsigned> logged{0};
  sink_scope sink{cucascade::log::make_sink([&](cucascade::log::record const&) {
    logged.fetch_add(1);
    if (leg == 2) { throw std::runtime_error("sink threw"); }
  })};
  runtime_fixture f;
  action destroy;
  destroy.result = cudaErrorInvalidValue;
  destroy.throws = leg == 1;
  f.controls->script(operation::destroy_event, destroy);
  f.start();
  auto read = f.submit(0, 4);
  REQUIRE(wait_until([&] { return read.future.is_ready(); }));
  CHECK(std::move(read.future).get(2s) == 4);
  f.shutdown();
  CHECK(f.controls->log.count(operation::wait, true) == 1);
  CHECK(f.controls->log.count(operation::destroy_event) == 1);
  CHECK(logged.load() > 0);
  CHECK(read.completion->success.load());
  CHECK(read.completion->calls.load() == 1);
  CHECK(f.reactor->perf().fail_stop_total == 0);
  CHECK(f.controls->policies.load() == 0);
}
