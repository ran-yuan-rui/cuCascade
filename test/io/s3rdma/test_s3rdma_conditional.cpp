/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_test_support.hpp"
#include "transport_test_support.hpp"

#include <set>

using namespace cucascade::test::s3rdma;
namespace io = cucascade::io;

namespace {

struct control_context {
  explicit control_context(std::shared_ptr<rdma::s3_control_client> control)
  {
    auto clients    = controls->clients();
    clients.control = std::move(control);
    context         = std::make_unique<rdma::s3rdma_ioctx>(
      runtime_config(), clients, controls->delivery(), controls->policy());
    context->start();
  }
  ~control_context()
  {
    context->shutdown();
    context.reset();
    controls->reclaim_sessions();
  }
  std::shared_ptr<io::io_object> open() { return context->open_io_object("s3://test/object"); }
  deadline bound;
  std::shared_ptr<runtime_controls> controls{std::make_shared<runtime_controls>()};
  std::unique_ptr<rdma::s3rdma_ioctx> context;
};

class echo_factory final : public rdma::rdma_data_session_factory {
  class session final : public rdma::rdma_data_session {
   public:
    session(echo_factory* owner, std::unique_ptr<rdma::rdma_data_session> inner)
      : owner(owner), inner(std::move(inner))
    {
    }
    ~session() override
    {
      std::lock_guard lock{owner->mutex};
      owner->live.erase(this);
    }
    void register_memory(std::uint8_t* base, std::size_t bytes) override
    {
      inner->register_memory(base, bytes);
    }
    bool deregister_memory(std::uint8_t* base) noexcept override
    {
      return inner->deregister_memory(base);
    }
    std::int64_t max_request_callback_size(std::uint8_t* base) noexcept override
    {
      return inner->max_request_callback_size(base);
    }
    rdma::data_get_result get(rdma::object_ref const& object,
                              std::size_t offset,
                              std::size_t bytes,
                              std::uint8_t* base,
                              std::size_t destination) override
    {
      return inner->get(object, offset, bytes, base, destination);
    }
    rdma::data_get_result get_if_match(rdma::object_ref const& object,
                                       std::size_t offset,
                                       std::size_t bytes,
                                       std::uint8_t* base,
                                       std::size_t destination,
                                       std::string_view condition) override
    {
      {
        std::lock_guard lock{owner->mutex};
        owner->conditions.emplace_back(condition);
      }
      auto result           = inner->get(object, offset, bytes, base, destination);
      result.validation_tag = owner->echo;
      return result;
    }
    echo_factory* owner;
    std::unique_ptr<rdma::rdma_data_session> inner;
  };

 public:
  explicit echo_factory(std::shared_ptr<rdma::rdma_data_session_factory> inner)
    : inner(std::move(inner))
  {
  }
  std::unique_ptr<rdma::rdma_data_session> acquire() override
  {
    auto result = std::make_unique<session>(this, inner->acquire());
    std::lock_guard lock{mutex};
    live.insert(result.get());
    return result;
  }
  void reclaim()
  {
    for (;;) {
      rdma::rdma_data_session* held = nullptr;
      {
        std::lock_guard lock{mutex};
        if (live.empty()) { break; }
        held = *live.begin();
        live.erase(live.begin());
      }
      delete held;
    }
  }
  auto observed()
  {
    std::lock_guard lock{mutex};
    return conditions;
  }
  std::string echo{"\"version-a\""};

 private:
  std::shared_ptr<rdma::rdma_data_session_factory> inner;
  std::mutex mutex;
  std::set<rdma::rdma_data_session*> live;
  std::vector<std::string> conditions;
};

struct conditional_runtime {
  conditional_runtime(std::string tag, bool require_echo)
  {
    auto clients                        = fixture.controls->clients();
    factory                             = std::make_shared<echo_factory>(clients.data_sessions);
    factory->echo                       = std::move(tag);
    clients.data_sessions               = factory;
    clients.rule.require_validator_echo = require_echo;
    fixture.object  = std::make_shared<rdma::s3rdma_io_object>("s3://test/object",
                                                              rdma::object_ref{"test", "object"},
                                                              fixture.source.size(),
                                                              "\"version-a\"");
    fixture.context = std::make_shared<rdma::s3rdma_reactor::reactor_context>(
      fixture.cfg, clients, fixture.controls->delivery(), fixture.controls->policy());
    fixture.reactor = std::make_unique<rdma::s3rdma_reactor>(fixture.context);
    fixture.start();
  }
  ~conditional_runtime()
  {
    fixture.shutdown();
    fixture.reactor.reset();
    fixture.context.reset();
    factory->reclaim();
  }
  runtime_fixture fixture;
  std::shared_ptr<echo_factory> factory;
};

}  // namespace

TEST_CASE("S3RDMA validators and object identities distinguish reusable versions",
          "[s3rdma][conditional]")
{
  for (auto const& [tag, strong] :
       std::vector<std::pair<std::string, bool>>{{"\"version-a\"", true},
                                                 {"\"\"", true},
                                                 {"W/\"version-a\"", false},
                                                 {"version-a", false},
                                                 {"", false},
                                                 {"\"a\"b\"", false},
                                                 {"\"a\nb\"", false},
                                                 {std::string{"\"a"} + '\x7f' + "\"", false}}) {
    CAPTURE(tag);
    CHECK(rdma::is_strong_validator(tag) == strong);
    rdma::s3rdma_io_object first("s3://test/object", {"test", "object"}, 32, tag);
    rdma::s3rdma_io_object second("s3://test/object", {"test", "object"}, 32, tag);
    CHECK(first.validation_tag() == tag);
    CHECK(first.validation_tag_is_strong() == strong);
    CHECK(first.condition() == (strong ? tag : ""));
    CHECK((first.identity_cache_key() == second.identity_cache_key()) == strong);
  }
  rdma::s3rdma_io_object small("s3://test/object", {"test", "object"}, 31, "\"version-a\"");
  rdma::s3rdma_io_object large("s3://test/object", {"test", "object"}, 32, "\"version-a\"");
  rdma::s3rdma_io_object changed("s3://test/object", {"test", "object"}, 32, "\"version-b\"");
  CHECK(small.identity_cache_key() != large.identity_cache_key());
  CHECK(large.identity_cache_key() != changed.identity_cache_key());
}

TEST_CASE("S3RDMA open preserves the HEAD validator and size", "[s3rdma][conditional]")
{
  http_fixture server;
  server.respond({.headers = {{"Content-Length", "32"}, {"ETag", "\"version-a\""}}});
  control_context fixture(rdma::make_control_client(signer(server.endpoint())));
  auto object = fixture.open();
  CHECK(object->size() == 32);
  CHECK(object->validation_tag() == "\"version-a\"");
  CHECK(object->validation_tag_is_strong());
  auto const* typed = dynamic_cast<rdma::s3rdma_io_object const*>(object.get());
  REQUIRE(typed != nullptr);
  CHECK(typed->condition() == "\"version-a\"");
  CHECK(server.requests().size() == 1);
}

TEST_CASE("S3RDMA host reads reject changed or unconfirmed opened versions without retry",
          "[s3rdma][conditional]")
{
  for (auto fault : {"412", "other", "absent", "none"}) {
    CAPTURE(fault);
    http_fixture server;
    server.respond({.headers = {{"Content-Length", "8"}, {"ETag", "\"version-a\""}}});
    http_response response;
    response.status  = std::string_view{fault} == "412" ? 412 : 206;
    response.headers = {{"Content-Range", "bytes 0-3/8"}};
    if (std::string_view{fault} != "absent") {
      response.headers.emplace_back(
        "ETag", std::string_view{fault} == "other" ? "\"version-b\"" : "\"version-a\"");
    }
    response.body = "data";
    server.respond(response);
    control_context fixture(rdma::make_control_client(signer(server.endpoint())));
    auto object = fixture.open();
    std::array<std::uint8_t, 4> destination{};
    if (std::string_view{fault} == "none") {
      CHECK(fixture.context->host_read(*object, 0, 4, destination.data()) == 4);
      CHECK(std::string(destination.begin(), destination.end()) == "data");
    } else {
      CHECK_THROWS_AS(fixture.context->host_read(*object, 0, 4, destination.data()),
                      io::object_changed_error);
    }
    auto const requests = server.requests();
    REQUIRE(requests.size() == 2);
    CHECK(requests[1].value("if-match") == "\"version-a\"");
    CHECK_FALSE(fixture.context->status()->terminal());
  }
}

TEST_CASE("S3RDMA weak or absent validators read unconditionally", "[s3rdma][conditional]")
{
  for (std::string const tag : {"", "W/\"weak\""}) {
    http_fixture server;
    http_response head;
    head.headers = {{"Content-Length", "4"}};
    if (!tag.empty()) { head.headers.emplace_back("ETag", tag); }
    server.respond(head);
    server.respond({.status = 206, .headers = {{"Content-Range", "bytes 0-3/4"}}, .body = "data"});
    control_context fixture(rdma::make_control_client(signer(server.endpoint())));
    auto object = fixture.open();
    CHECK_FALSE(object->validation_tag_is_strong());
    std::array<std::uint8_t, 4> destination{};
    CHECK(fixture.context->host_read(*object, 0, 4, destination.data()) == 4);
    REQUIRE(server.requests().size() == 2);
    CHECK(server.requests()[1].values("if-match").empty());
  }
}

TEST_CASE("S3RDMA a control client without conditions refuses instead of reading",
          "[s3rdma][conditional]")
{
  deadline bound;
  auto mock = std::make_shared<mock_rdma_client>();
  mock->put({"test", "object"}, {1, 2, 3, 4});
  std::array<std::uint8_t, 4> destination{};
  require_code(
    [&] {
      (void)mock->range_get_if_match({"test", "object"}, 0, 4, destination.data(), "\"version-a\"");
    },
    rdma::errc::backend_unavailable);
  CHECK(mock->counts().ranges == 0);
  CHECK(
    mock->range_get_if_match({"test", "object"}, 0, 4, destination.data(), {}).delivered_bytes ==
    4);
  CHECK(mock->counts().ranges == 1);
}

TEST_CASE("S3RDMA device completion enforces the opened validator and echo policy",
          "[s3rdma][conditional]")
{
  for (bool require_echo : {true, false}) {
    for (std::string const tag : {"\"version-a\"", "\"version-b\"", ""}) {
      CAPTURE(require_echo, tag);
      conditional_runtime test(tag, require_echo);
      auto& fixture        = test.fixture;
      auto pending         = fixture.submit(0, 4);
      bool const delivered = tag == "\"version-a\"" || (tag.empty() && !require_echo);
      if (delivered) {
        REQUIRE(wait_until([&] { return pending.future.is_ready(); }));
        CHECK(std::move(pending.future).get(2s) == 4);
        CHECK(std::equal(
          fixture.source.begin(), fixture.source.begin() + 4, fixture.destination.begin()));
        CHECK_FALSE(fixture.context->status()->terminal());
      } else {
        CHECK(has_code(error_from(pending.future), rdma::errc::transport_failed));
        CHECK(fixture.context->status()->terminal());
        CHECK(std::all_of(fixture.destination.begin(), fixture.destination.end(), [](auto b) {
          return b == 0xcc;
        }));
      }
      CHECK(test.factory->observed() == std::vector<std::string>{"\"version-a\""});
      CHECK(fixture.controls->mock->counts().gets == 1);
      CHECK(pending.completion->calls.load() == 1);
      CHECK(pending.completion->success.load() == delivered);
    }
  }
}

TEST_CASE("S3RDMA a session without conditions issues nothing and stops the backend",
          "[s3rdma][conditional]")
{
  runtime_fixture fixture;
  fixture.object = std::make_shared<rdma::s3rdma_io_object>(
    "s3://test/object", rdma::object_ref{"test", "object"}, fixture.source.size(), "\"version-a\"");
  fixture.start();
  auto pending = fixture.submit(0, 4);
  CHECK(has_code(error_from(pending.future), rdma::errc::transport_failed));
  CHECK(fixture.controls->mock->counts().gets == 0);
  CHECK(fixture.context->status()->terminal());
  CHECK(pending.completion->calls.load() == 1);
  CHECK_FALSE(pending.completion->success.load());
}
