/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cucascade::test::s3rdma {

void journal::append(observation value) noexcept
{
  try {
    {
      std::lock_guard lock(_mutex);
      _entries.push_back(value);
    }
    if (observer) { observer(value); }
  } catch (...) {
    _lost.store(true);
  }
}

std::vector<observation> journal::snapshot() const
{
  std::lock_guard lock(_mutex);
  return _entries;
}

std::size_t journal::count(operation op, bool returned) const
{
  auto entries = snapshot();
  return static_cast<std::size_t>(std::count_if(entries.begin(), entries.end(), [&](auto const& e) {
    return e.op == op && e.returned == returned;
  }));
}

void rendezvous::arrive_and_wait()
{
  std::unique_lock lock(_mutex);
  _entered = true;
  _cv.notify_all();
  if (!_cv.wait_for(lock, 4s, [&] { return _released; })) { _timed_out.store(true); }
}

bool rendezvous::entered()
{
  std::unique_lock lock(_mutex);
  return _cv.wait_for(lock, 2s, [&] { return _entered; });
}

void rendezvous::release() noexcept
{
  {
    std::lock_guard lock(_mutex);
    _released = true;
  }
  _cv.notify_all();
}

runtime_controls::runtime_controls(bool use_cuda) : native_cuda(use_cuda)
{
  if (native_cuda) {
    mock->writer([](auto* dst, auto const* src, auto size) {
      if (cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cannot populate device landing");
      }
    });
    write_probe = {rdma::write_ordering::owner, false};
  }
}

runtime_controls::~runtime_controls()
{
  // Test cleanup happens after runtime destruction and does not stand for an SDK release.
  for (auto* pointer : _cuda_blocks) {
    (void)cudaFree(pointer);
  }
}

void runtime_controls::script(operation op, action step)
{
  std::lock_guard lock(_mutex);
  _actions[op].push_back(std::move(step));
}

action runtime_controls::enter(observation value)
{
  action step;
  {
    std::lock_guard lock(_mutex);
    if (value.device == -1) {
      auto device  = _devices.find(value.thread);
      value.device = device == _devices.end() ? 42 : device->second;
    }
    auto& steps = _actions[value.op];
    auto match  = std::find_if(steps.begin(), steps.end(), [&](auto const& candidate) {
      return !candidate.get_offset ||
             (value.op == operation::get && value.object_offset == *candidate.get_offset);
    });
    if (match != steps.end()) {
      step = std::move(*match);
      steps.erase(match);
    }
  }
  log.append(value);
  if (step.before) { step.before(value); }
  if (step.throws) { throw std::runtime_error("scripted operation failure"); }
  return step;
}

cudaError_t runtime_controls::leave(observation value, cudaError_t result)
{
  value.result   = result;
  value.returned = true;
  log.append(value);
  return result;
}

void runtime_controls::observe_thread_exit()
{
  if (std::this_thread::get_id() == caller) { return; }
  struct exit_observer {
    std::vector<std::weak_ptr<runtime_controls>> owners;
    ~exit_observer()
    {
      for (auto const& weak : owners) {
        if (auto owner = weak.lock()) { owner->log.append({operation::worker_exit}); }
      }
    }
  };
  thread_local exit_observer observer;
  auto self = shared_from_this();
  for (auto const& old : observer.owners) {
    if (old.lock() == self) { return; }
  }
  observer.owners.push_back(self);
}

class scripted_client final : public rdma::s3_control_client,
                              public rdma::rdma_data_session_factory {
 public:
  explicit scripted_client(std::shared_ptr<runtime_controls> controls)
    : _controls(std::move(controls))
  {
  }

  class session final : public rdma::rdma_data_session {
   public:
    explicit session(std::shared_ptr<runtime_controls> controls)
      : _controls(std::move(controls)), _inner(_controls->mock->acquire())
    {
      std::lock_guard lock(_controls->_mutex);
      _controls->_sessions.insert(this);
    }
    ~session() override
    {
      bool reclaiming;
      {
        std::lock_guard lock(_controls->_mutex);
        _controls->_sessions.erase(this);
        reclaiming = _controls->_reclaiming;
      }
      if (!reclaiming) { _controls->log.append({operation::release_session}); }
    }
    void register_memory(std::uint8_t* base, std::size_t bytes) override
    {
      observation e{operation::register_memory, base, bytes};
      (void)_controls->enter(e);
      _inner->register_memory(base, bytes);
      (void)_controls->leave(e, cudaSuccess);
    }
    bool deregister_memory(std::uint8_t* base) noexcept override
    {
      observation e{operation::deregister_memory, base};
      try {
        (void)_controls->enter(e);
        auto released = _inner->deregister_memory(base);
        (void)_controls->leave(e, released ? cudaSuccess : cudaErrorUnknown);
        return released;
      } catch (...) {
        return false;
      }
    }
    std::int64_t max_request_callback_size(std::uint8_t* base) noexcept override
    {
      _controls->log.append({operation::capacity, base});
      return _inner->max_request_callback_size(base);
    }
    rdma::data_get_result get(rdma::object_ref const& object,
                              std::size_t offset,
                              std::size_t size,
                              std::uint8_t* base,
                              std::size_t landing) override
    {
      observation e{operation::get, base, size, landing};
      e.object_offset = offset;
      auto step       = _controls->enter(e);
      rdma::data_get_result result;
      {
        // Script plus consumption is indivisible, even when unequal chunks reach two workers.
        std::lock_guard lock(_controls->_get_mutex);
        _controls->mock->script(step.behavior, static_cast<std::int64_t>(size));
        result = _inner->get(object, offset, size, base, landing);
      }
      if (step.change_receipt) { step.change_receipt(result); }
      (void)_controls->leave(e, cudaSuccess);
      return result;
    }

   private:
    std::shared_ptr<runtime_controls> _controls;
    std::unique_ptr<rdma::rdma_data_session> _inner;
  };

  std::unique_ptr<rdma::rdma_data_session> acquire() override
  {
    _controls->observe_thread_exit();
    (void)_controls->enter({operation::acquire_session});
    return std::make_unique<session>(_controls);
  }
  rdma::head_result head(rdma::object_ref const& object) override
  {
    auto step   = _controls->enter({operation::head});
    auto result = _controls->mock->head(object);
    return step.head.value_or(result);
  }
  rdma::range_get_result range_get(rdma::object_ref const& object,
                                   std::size_t offset,
                                   std::size_t size,
                                   std::uint8_t* dst) override
  {
    auto step   = _controls->enter({operation::range, dst, size, offset});
    auto result = _controls->mock->range_get(object, offset, size, dst);
    return step.range.value_or(result);
  }

 private:
  std::shared_ptr<runtime_controls> _controls;
};

void runtime_controls::reclaim_sessions()
{
  for (;;) {
    rdma::rdma_data_session* session;
    {
      std::lock_guard lock(_mutex);
      _reclaiming = true;
      if (_sessions.empty()) { return; }
      session = *_sessions.begin();
    }
    delete session;
  }
}

rdma::transport_clients runtime_controls::clients()
{
  auto client = std::make_shared<scripted_client>(shared_from_this());
  return {client, client, {{206}, +[](std::string_view tag, long status) noexcept {
                             return status == 206 && tag == "mock-rdma";
                           }}};
}

rdma::context_health_policy runtime_controls::policy()
{
  return [self = shared_from_this()](rdma::context_fatal_report const& report) {
    self->policy_code.store(report.code);
    self->policies.fetch_add(1);
    self->log.append({operation::policy, nullptr, 0, 0, report.device_id, nullptr, report.code});
    if (self->policy_observer) { self->policy_observer(report); }
  };
}

namespace {

template <typename F>
cudaError_t perform(std::shared_ptr<runtime_controls> const& controls, observation e, F&& work)
{
  auto step = controls->enter(e);
  return controls->leave(e, step.result == cudaSuccess ? work() : step.result);
}

std::vector<std::uint8_t> source_bytes()
{
  std::vector<std::uint8_t> result(64);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<std::uint8_t>(3 + i * 7);
  }
  return result;
}

}  // namespace

rdma::cuda_delivery_ops runtime_controls::delivery()
{
  auto self = shared_from_this();
  rdma::cuda_delivery_ops ops;
  ops.get_device = [self](int* device) {
    return perform(self, {operation::get_device}, [&] {
      if (self->native_cuda) {
        auto rc = cudaGetDevice(device);
        if (rc == cudaSuccess) {
          std::lock_guard lock(self->_mutex);
          self->_devices[std::this_thread::get_id()] = *device;
        }
        return rc;
      }
      std::lock_guard lock(self->_mutex);
      auto [it, inserted] = self->_devices.try_emplace(std::this_thread::get_id(), 42);
      (void)inserted;
      *device = it->second;
      return cudaSuccess;
    });
  };
  ops.set_device = [self](int device) {
    return perform(self, {operation::set_device, nullptr, 0, 0, device}, [&] {
      auto rc = self->native_cuda ? cudaSetDevice(device) : cudaSuccess;
      if (rc == cudaSuccess) {
        std::lock_guard lock(self->_mutex);
        self->_devices[std::this_thread::get_id()] = device;
      }
      return rc;
    });
  };
  ops.malloc_device = [self](void** pointer, std::size_t bytes) {
    observation e{operation::allocate, nullptr, bytes};
    auto step      = self->enter(e);
    cudaError_t rc = step.result;
    if (rc == cudaSuccess) {
      if (self->native_cuda) {
        rc = cudaMalloc(pointer, bytes);
        if (rc == cudaSuccess) {
          std::lock_guard lock(self->_mutex);
          self->_cuda_blocks.insert(*pointer);
        }
      } else {
        auto block = std::make_unique<std::uint8_t[]>(bytes);
        std::fill_n(block.get(), bytes, std::uint8_t{0xa5});
        *pointer = block.get();
        std::lock_guard lock(self->_mutex);
        self->_blocks.emplace(*pointer, std::move(block));
      }
    }
    e.address = rc == cudaSuccess ? *pointer : nullptr;
    return self->leave(e, rc);
  };
  ops.free_device = [self](void* pointer) {
    return perform(self, {operation::free, pointer}, [&] {
      if (self->native_cuda) {
        auto rc = cudaFree(pointer);
        if (rc == cudaSuccess) {
          std::lock_guard lock(self->_mutex);
          self->_cuda_blocks.erase(pointer);
        }
        return rc;
      }
      std::lock_guard lock(self->_mutex);
      return self->_blocks.erase(pointer) == 1 ? cudaSuccess : cudaErrorInvalidDevicePointer;
    });
  };
  ops.probe_writes = [self](int device, rdma::write_probe_result* result) {
    return perform(self, {operation::probe, nullptr, 0, 0, device}, [&] {
      *result = self->write_probe;
      return cudaSuccess;
    });
  };
  ops.flush = [self] { return perform(self, {operation::flush}, [] { return cudaSuccess; }); };
  ops.stream_capture_query = [self](cudaStream_t stream, cudaStreamCaptureStatus* result) {
    return perform(self, {operation::capture}, [&] {
      if (self->native_cuda) { return cudaStreamIsCapturing(stream, result); }
      *result = self->capture_status;
      return cudaSuccess;
    });
  };
  ops.event_create = [self](cudaEvent_t* event, unsigned flags) {
    observation e{operation::create_event, nullptr, flags};
    auto step      = self->enter(e);
    cudaError_t rc = step.result;
    if (rc == cudaSuccess) {
      if (self->native_cuda) {
        rc = cudaEventCreateWithFlags(event, flags);
      } else {
        std::lock_guard lock(self->_mutex);
        *event = reinterpret_cast<cudaEvent_t>(self->_next_event++);
      }
    }
    e.event = rc == cudaSuccess ? *event : nullptr;
    return self->leave(e, rc);
  };
  ops.memcpy_async =
    [self](
      void* dst, void const* src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
      observation e{operation::copy, dst, bytes};
      e.source = src;
      e.kind   = kind;
      return perform(self, e, [&] {
        if (self->native_cuda) { return cudaMemcpyAsync(dst, src, bytes, kind, stream); }
        if (kind != cudaMemcpyDeviceToDevice) { return cudaErrorInvalidMemcpyDirection; }
        std::memcpy(dst, src, bytes);
        return cudaSuccess;
      });
    };
  ops.event_record = [self](cudaEvent_t event, cudaStream_t stream) {
    return perform(self, {operation::record, nullptr, 0, 0, -1, event}, [&] {
      return self->native_cuda ? cudaEventRecord(event, stream) : cudaSuccess;
    });
  };
  ops.event_synchronize = [self](cudaEvent_t event) {
    return perform(self, {operation::wait, nullptr, 0, 0, -1, event}, [&] {
      return self->native_cuda ? cudaEventSynchronize(event) : cudaSuccess;
    });
  };
  ops.event_destroy = [self](cudaEvent_t event) {
    return perform(self, {operation::destroy_event, nullptr, 0, 0, -1, event}, [&] {
      return self->native_cuda ? cudaEventDestroy(event) : cudaSuccess;
    });
  };
  ops.fatal_hook = [](char const*, cudaError_t) { std::abort(); };
  return ops;
}

rdma::config runtime_config(std::size_t workers, std::size_t slot, std::vector<int> devices)
{
  rdma::config cfg;
  cfg.max_inflight    = workers;
  cfg.arena_slot_size = slot;
  cfg.queue_cap       = 2;
  cfg.devices         = std::move(devices);
  return cfg;
}

std::shared_ptr<io::prepared_io_completion> completion_record::callback()
{
  return std::make_shared<io::prepared_io_completion>(
    [self = shared_from_this()](auto chunks, bool ok) noexcept {
      self->fragments.store(chunks.size());
      self->success.store(ok);
      self->calls.fetch_add(1);
      if (self->after) { self->after(); }
    });
}

runtime_fixture::runtime_fixture(rdma::config configuration, bool native)
  : cfg(std::move(configuration)),
    controls(std::make_shared<runtime_controls>(native)),
    source(source_bytes()),
    object(std::make_shared<rdma::s3rdma_io_object>(
      "s3://test/object", rdma::object_ref{"test", "object"}, source.size()))
{
  destination.fill(0xcc);
  controls->mock->put(object->ref(), source);
}

void runtime_fixture::construct(rdma::context_health_policy policy)
{
  context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(
    cfg, controls->clients(), controls->delivery(), std::move(policy));
  reactor = std::make_unique<rdma::s3rdma_reactor>(context);
}

void runtime_fixture::start()
{
  if (!reactor) { construct(controls->policy()); }
  reactor->start();
}

runtime_fixture::~runtime_fixture()
{
  shutdown();
  reactor.reset();
  context.reset();
  controls->reclaim_sessions();
}

io::prepared_io_slice runtime_fixture::slice(
  std::size_t offset, std::size_t size, std::uint8_t* dst, int device, cudaStream_t stream)
{
  return io::prepared_io_slice{io::range{offset, size},
                               io::device_buffer{dst, ::cuda::stream_ref{stream}, device}};
}

pending_read runtime_fixture::submit(std::vector<io::prepared_io_slice> slices,
                                     std::shared_ptr<completion_record> completion)
{
  if (!completion) { completion = std::make_shared<completion_record>(); }
  for (auto& value : slices) {
    value.on_complete = completion->callback();
  }
  auto request     = io::grouped_io_request::create(object, std::move(slices));
  auto coordinator = request->coordinator;
  auto future      = coordinator->get_future();
  reactor->enqueue(std::move(request));
  return {std::move(coordinator), std::move(future), std::move(completion)};
}

pending_read runtime_fixture::submit(std::size_t offset,
                                     std::size_t size,
                                     std::size_t destination_offset)
{
  return submit(
    {slice(offset, size, destination.data() + destination_offset, cfg.devices.front())});
}

void runtime_fixture::shutdown()
{
  if (reactor) { reactor->shutdown(); }
}

host_fixture::host_fixture() : source(source_bytes())
{
  destination.fill(0xcc);
  controls->mock->put({"test", "object"}, source);
  ctx = std::make_unique<rdma::s3rdma_ioctx>(
    runtime_config(), controls->clients(), controls->delivery(), controls->policy());
  object = std::make_shared<rdma::s3rdma_io_object>(
    "s3://test/object", rdma::object_ref{"test", "object"}, source.size());
}

void host_fixture::start() { ctx->start(); }

host_fixture::~host_fixture()
{
  ctx->shutdown();
  ctx.reset();
  controls->reclaim_sessions();
}

}  // namespace cucascade::test::s3rdma
