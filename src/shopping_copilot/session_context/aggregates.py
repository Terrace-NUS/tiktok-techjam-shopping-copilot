"""Immutable interaction and session aggregates."""

from __future__ import annotations

from dataclasses import dataclass

from .models import FeedbackSignal, IntentState, ProfilePrior, SearchBelief
from .operations import StateUpdateBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFeedback:
    """Feedback about products shown before the current turn."""

    product_ids: tuple[str, ...]
    signal: FeedbackSignal
    compared_to_ids: tuple[str, ...]
    evidence_text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnRecord:
    """One append-only record of a processed user turn."""

    turn: int
    user_message: str
    intent_version_before: int
    accepted_update: StateUpdateBatch | None
    intent_version_after: int
    assistant_message: str
    question: str | None
    question_key: str | None
    ask_attribute: str | None
    shown_product_ids: tuple[str, ...]
    feedback: tuple[ProductFeedback, ...]
    search_belief_probe_id: str | None

    @property
    def state_changed(self) -> bool:
        """Whether canonical intent changed during this turn."""

        return self.intent_version_before != self.intent_version_after


@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionContext:
    """Append-only turn history."""

    turns: tuple[TurnRecord, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionState:
    """Mutable-by-replacement state within a session context."""

    intent: IntentState
    interaction: InteractionContext
    search_belief: SearchBelief | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionContext:
    """Complete immutable session snapshot."""

    session_id: str
    profile: ProfilePrior | None
    state: SessionState
