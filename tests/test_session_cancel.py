import asyncio
import logging

from app.gateway.session_cancel import StreamCancellationRegistry
from app.models.request import RequestContext


async def _registered_sleep(registry, ctx, started):
    registry.register(ctx)
    started.set()
    try:
        await asyncio.sleep(60)
    finally:
        registry.unregister(ctx)


def test_registry_cancels_child_and_grandchild_stream_tasks():
    registry = StreamCancellationRegistry()
    parent = RequestContext(
        request_id="req_parent",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
        agent_id="agent_parent",
    )
    child = RequestContext(
        request_id="req_child",
        path="/v1/messages",
        stream=True,
        session_id="session_child",
        agent_id="agent_child",
        parent_agent_id="agent_parent",
    )
    grandchild = RequestContext(
        request_id="req_grandchild",
        path="/v1/messages",
        stream=True,
        session_id="session_grandchild",
        agent_id="agent_grandchild",
        parent_agent_id="agent_child",
    )

    async def run():
        child_started = asyncio.Event()
        grandchild_started = asyncio.Event()
        child_task = asyncio.create_task(_registered_sleep(registry, child, child_started))
        grandchild_task = asyncio.create_task(_registered_sleep(registry, grandchild, grandchild_started))
        await child_started.wait()
        await grandchild_started.wait()

        assert registry.cancel_descendants(parent) == 2

        for task in (child_task, grandchild_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("expected child stream task to be cancelled")

    asyncio.run(run())


def test_registry_cancels_child_by_parent_session_id():
    registry = StreamCancellationRegistry()
    parent = RequestContext(
        request_id="req_parent",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
    )
    child = RequestContext(
        request_id="req_child",
        path="/v1/messages",
        stream=True,
        session_id="session_child",
        parent_session_id="session_parent",
    )

    async def run():
        child_started = asyncio.Event()
        child_task = asyncio.create_task(_registered_sleep(registry, child, child_started))
        await child_started.wait()

        assert registry.cancel_descendants(parent) == 1

        try:
            await child_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected child stream task to be cancelled")

    asyncio.run(run())


def test_registry_cancels_claude_code_agent_by_shared_session():
    registry = StreamCancellationRegistry()
    parent = RequestContext(
        request_id="req_parent",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
    )
    agent = RequestContext(
        request_id="req_agent",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
        agent_id="agent_child",
    )

    async def run():
        agent_started = asyncio.Event()
        agent_task = asyncio.create_task(_registered_sleep(registry, agent, agent_started))
        await agent_started.wait()

        assert agent.stream_role() == "agent"
        assert registry.cancel_descendants(parent) == 1

        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected Claude Code agent stream task to be cancelled")

    asyncio.run(run())


def test_registry_does_not_cancel_sibling_agent_by_shared_session():
    registry = StreamCancellationRegistry()
    agent = RequestContext(
        request_id="req_agent",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
        agent_id="agent_a",
    )
    sibling = RequestContext(
        request_id="req_sibling",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
        agent_id="agent_b",
    )

    async def run():
        sibling_started = asyncio.Event()
        sibling_task = asyncio.create_task(_registered_sleep(registry, sibling, sibling_started))
        await sibling_started.wait()

        assert agent.stream_role() == "agent"
        assert registry.cancel_descendants(agent) == 0
        assert not sibling_task.cancelled()

        sibling_task.cancel()
        try:
            await sibling_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected cleanup cancellation")

    asyncio.run(run())


def test_registry_logs_cancelled_child_details(caplog):
    registry = StreamCancellationRegistry()
    parent = RequestContext(
        request_id="req_parent",
        path="/v1/messages",
        stream=True,
        session_id="session_parent",
    )
    child = RequestContext(
        request_id="req_child",
        path="/v1/messages",
        stream=True,
        session_id="session_child",
        parent_session_id="session_parent",
    )

    async def run():
        child_started = asyncio.Event()
        child_task = asyncio.create_task(_registered_sleep(registry, child, child_started))
        await child_started.wait()

        with caplog.at_level(logging.INFO):
            assert registry.cancel_descendants(parent) == 1

        try:
            await child_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected child stream task to be cancelled")

    asyncio.run(run())

    assert "stream.cancel.children count=1 role=root session=session_parent" in caplog.text
    assert "stream.cancel.child child_request=req_child role=subagent session=session_child parent_session=session_parent" in caplog.text
