from __future__ import annotations

from scripts.query_understanding.suites import PromptConversation, PromptSuite, PromptTurn
from scripts.retrieval.evaluate_qu_to_probe import _error_record
from shopping_copilot.query_understanding import (
    QueryUnderstandingError,
    QueryUnderstandingErrorCode,
)


def test_error_record_preserves_structured_query_understanding_failure() -> None:
    turn = PromptTurn(
        turn=2,
        user_message="release the old constraint",
        last_assistant_message=None,
        last_question=None,
    )
    conversation = PromptConversation(
        identifier="reporting_case",
        tier="full",
        turns=(turn,),
    )
    suite = PromptSuite(
        schema="test",
        suite_id="reporting-suite",
        cohort="natural",
        description="unit fixture",
        conversations=(conversation,),
    )
    error = QueryUnderstandingError(
        code=QueryUnderstandingErrorCode.REPAIR_EXHAUSTED,
        path=("dont_care_facets", 1),
        details=(
            ("attempt_count", 2),
            ("last_error", "invalid_final_state"),
            ("last_detail_facet", "length"),
        ),
    )

    record = _error_record(
        suite,
        conversation,
        turn,
        stage="query_understanding",
        elapsed_ms=12.5,
        error=error,
    )

    assert record["error"] == {
        "type": "QueryUnderstandingError",
        "code": "repair_exhausted",
        "path": ["dont_care_facets", 1],
        "details": {
            "attempt_count": 2,
            "last_error": "invalid_final_state",
            "last_detail_facet": "length",
        },
    }
