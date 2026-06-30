"""Request context model."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class RequestContext:
    request_id: str
    path: str
    model: str | None = None
    stream: bool = False
    client: str | None = None
    client_headers: dict[str, str] = field(default_factory=dict)
    session_id: str | None = None
    parent_session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    started_at: float = field(default_factory=perf_counter)

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000

    def stream_role(self) -> str:
        if self.parent_session_id or self.parent_agent_id:
            return "subagent"
        if self.agent_id:
            return "agent"
        return "root"

    def stream_label(self) -> str:
        return (
            f"role={self.stream_role()} "
            f"session={self.session_id or '-'} "
            f"parent_session={self.parent_session_id or '-'} "
            f"agent={self.agent_id or '-'} "
            f"parent_agent={self.parent_agent_id or '-'}"
        )
