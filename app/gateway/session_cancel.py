"""Track streaming request relationships for client-side cancellation."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.models.request import RequestContext


@dataclass(slots=True)
class StreamRegistration:
    ctx: RequestContext
    task: asyncio.Task


class StreamCancellationRegistry:
    """Cancel active child streams when a parent stream is interrupted."""

    def __init__(self, *, logger: logging.Logger | None = None):
        self.log = logger or logging.getLogger(__name__)
        self._streams: dict[str, StreamRegistration] = {}

    def register(self, ctx: RequestContext) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        self._streams[ctx.request_id] = StreamRegistration(ctx=ctx, task=task)
        self.log.debug(
            "[%s] stream.registered %s active_streams=%s",
            ctx.request_id,
            ctx.stream_label(),
            len(self._streams),
        )

    def unregister(self, ctx: RequestContext) -> None:
        self._streams.pop(ctx.request_id, None)
        self.log.debug("[%s] stream.unregistered %s active_streams=%s", ctx.request_id, ctx.stream_label(), len(self._streams))

    def cancel_descendants(self, ctx: RequestContext) -> int:
        descendants = self._descendants_of(ctx)
        if descendants:
            self.log.info(
                "[%s] stream.cancel.children count=%s %s active_streams=%s",
                ctx.request_id,
                len(descendants),
                ctx.stream_label(),
                len(self._streams),
            )
        for item in descendants:
            self.log.info(
                "[%s] stream.cancel.child child_request=%s %s",
                ctx.request_id,
                item.ctx.request_id,
                item.ctx.stream_label(),
            )
            item.task.cancel()
        return len(descendants)

    def _descendants_of(self, ctx: RequestContext) -> list[StreamRegistration]:
        parent_agent_ids = {ctx.agent_id} if ctx.agent_id else set()
        parent_session_ids = {ctx.session_id} if ctx.session_id else set()
        allow_same_session_agents = ctx.stream_role() == "root"
        result: list[StreamRegistration] = []
        seen_request_ids = {ctx.request_id}

        changed = True
        while changed:
            changed = False
            for item in list(self._streams.values()):
                child = item.ctx
                if child.request_id in seen_request_ids:
                    continue
                if _is_child(child, parent_agent_ids, parent_session_ids, allow_same_session_agents):
                    result.append(item)
                    seen_request_ids.add(child.request_id)
                    if child.agent_id:
                        parent_agent_ids.add(child.agent_id)
                    if child.session_id:
                        parent_session_ids.add(child.session_id)
                    changed = True

        return result


def _is_child(ctx: RequestContext, parent_agent_ids: set[str], parent_session_ids: set[str], allow_same_session_agents: bool) -> bool:
    if ctx.parent_agent_id and ctx.parent_agent_id in parent_agent_ids:
        return True
    if ctx.parent_session_id and ctx.parent_session_id in parent_session_ids:
        return True
    if allow_same_session_agents and ctx.agent_id and ctx.session_id and ctx.session_id in parent_session_ids:
        return True
    return False
