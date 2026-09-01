from __future__ import annotations

from .catalog import CatalogIndex
from .ranker import RankedResult
from .state import SessionState

QUESTION_TEXT = {
    "material": "Do you have a material preference?",
    "color": "Do you have a color preference?",
    "size": "Are there any sizing or fit requirements?",
    "style": "What style or fit do you prefer?",
    "feature": "Which product features matter most to you?",
    "use_case": "What will you mainly use it for?",
    "budget": "What budget range should I use?",
    "other": "What other details matter most to you?",
}


class ConversationPolicy:
    """Deterministic clarification and first-hit-aware recommendation policy."""

    VALID_MODES = {"typed_first", "other_first"}

    def __init__(self, catalog: CatalogIndex, mode: str = "typed_first") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown question mode: {mode}")
        self.catalog = catalog
        self.mode = mode

    def choose_question(self, state: SessionState, result: RankedResult, turn: int) -> str | None:
        if turn >= 10:
            return None
        if result.candidate_count == 1 and not state.override_pending:
            return None

        if self.mode == "other_first":
            if "other" in state.exhausted_attributes:
                return None
            return "other" if state.answer_counts.get("other", 0) < 2 else None

        # A refusal of a typed question is a natural point for an open-ended
        # clarification rather than repeating the same slot.
        if (
            state.last_event == "boundary_no_preference"
            and "other" not in state.exhausted_attributes
        ):
            return "other"

        # An explicit rejection of an open-ended request means that additional
        # typed interrogation is unlikely to justify another turn.
        if "other" in state.exhausted_attributes:
            return None

        # A broad feature question is natural for vague product discovery and
        # tends to expose the most discriminative catalog language. It remains a
        # real typed question rather than an inferred hidden-card position.
        if (
            state.asked_counts.get("feature", 0) == 0
            and "feature" not in state.dont_care_attributes
            and "feature" not in state.exhausted_attributes
        ):
            return "feature"

        # Count successful replies rather than attempts: an Intent Override can
        # replace the customer reply and effectively swallow a question.
        if state.answer_counts.get("other", 0) < 2:
            return "other"

        # After two useful open-ended answers, broaden recommendation coverage
        # instead of spending the remaining turns on low-yield interrogation.
        return None

    def recommendations(
        self,
        state: SessionState,
        result: RankedResult,
        question: str | None,
        top_k: int,
        turn: int,
    ) -> list[int]:
        if not result.pids:
            return []

        # A Top-1 guess is free while asking: a correct guess converts at Rank 1;
        # a wrong guess still receives the clarification response.
        if question is not None and turn < 10:
            return [result.pids[0]]

        unseen = [pid for pid in result.pids if pid not in state.emitted_pids]
        if not unseen:
            unseen = result.pids
        return unseen[:top_k]

    @staticmethod
    def message_for(question: str | None) -> str:
        if question is None:
            return "Here are the best matches based on what you told me."
        return QUESTION_TEXT[question]
