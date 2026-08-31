"""Runtime-mode and delegate contracts for the public Agent entry point."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class RuntimeMode(str, Enum):
    """Explicit execution modes supported by the repository entry point."""

    OFFICIAL_SIMULATOR = "official_simulator"
    REAL_WORLD = "real_world"


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
