"""Runtime-mode and delegate contracts for the public Agent entry point."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class RuntimeMode(str, Enum):
    """Execution profiles exposed by the unified APERTURE entry point."""

    OFFLINE = "offline"
    FULL = "full"


class AgentDelegate(Protocol):
    """The stable reset/respond surface shared by both execution modes."""

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None: ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]: ...
