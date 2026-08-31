from __future__ import annotations

from scripts.simulator.evaluate_full_pipeline_other import _advance_context, _agent_response
from shopping_copilot.session_context import (
    IntentState,
    InteractionContext,
    ProfilePrior,
    SessionContext,
    SessionState,
)


def test_rendered_follow_up_is_committed_as_the_actual_turn_question() -> None:
    context = SessionContext(
        session_id="session",
        profile=ProfilePrior(
            purchase_frequency="",
            average_prior_rating=None,
            rating_style="",
            preference_tags=(),
            summary="",
        ),
        state=SessionState(
            intent=IntentState(
                goal=None,
                preferences=(),
                dont_care_facets=frozenset(),
                version=0,
            ),
            interaction=InteractionContext(turns=()),
            search_belief=None,
        ),
    )
    question = "Which direction should I narrow first?"
    message = f"I kept several product directions open.\n\n{question}"
    response = _agent_response(
        ["A", "B"],
        resolved=None,
        message=message,
    )

    updated = _advance_context(
        context,
        turn=1,
        user_message="I need something for winter travel",
        resolved=None,
        response=response,
        question=question,
    )

    record = updated.state.interaction.turns[0]
    assert response["ask_attribute"] == "other"
    assert response["message"] == message
    assert record.assistant_message == message
    assert record.question == question
    assert record.ask_attribute == "other"
