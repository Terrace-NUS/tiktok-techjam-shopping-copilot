from __future__ import annotations

import os
from pathlib import Path

from .catalog import CatalogIndex
from .policy import ConversationPolicy
from .ranker import ProductRanker
from .state import SessionState


class OfflineApertureAgent:
    """Model-free APERTURE execution profile for the competition boundary."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        question_mode: str | None = None,
    ) -> None:
        self.catalog = CatalogIndex(catalog_path)
        self.ranker = ProductRanker(self.catalog)
        mode = question_mode or os.environ.get("TECHJAM_QUESTION_MODE", "typed_first")
        self.policy = ConversationPolicy(self.catalog, mode=mode)
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        self.sessions[session_id] = SessionState(session_id=session_id, profile=dict(user_profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        state.observe(user_message, turn, self.catalog)
        result = self.ranker.rank(state)
        question = self.policy.choose_question(state, result, turn)
        recommended_pids = self.policy.recommendations(state, result, question, top_k, turn)
        state.record_action(question, recommended_pids)

        return {
            "message": self.policy.message_for(question),
            "ask_attribute": question,
            "recommendations": [
                {"parent_asin": self.catalog.asin(pid)} for pid in recommended_pids[:top_k]
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
