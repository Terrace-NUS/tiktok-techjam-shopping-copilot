"""Tests for pure interaction-history views."""

from __future__ import annotations

from shopping_copilot.session_context.aggregates import InteractionContext, TurnRecord
from shopping_copilot.session_context.operations import ClearFacet, StateUpdateBatch, SwitchGoal
from shopping_copilot.session_context.views import (
    all_shown_product_ids,
    last_non_empty_shown_products,
    most_recent_assistant_message,
    most_recent_question,
    question_keys_since_goal_switch,
)


def _batch(turn: int, *, switch_goal: bool = False) -> StateUpdateBatch:
    operation = SwitchGoal(new_goal="new goal") if switch_goal else ClearFacet(facet="color")
    return StateUpdateBatch(turn=turn, base_intent_version=0, operations=(operation,))


def _record(
    turn: int,
    *,
    shown_product_ids: tuple[str, ...] = (),
    question: str | None = None,
    question_key: str | None = None,
    assistant_message: str = "Here are some options.",
    accepted_update: StateUpdateBatch | None = None,
    intent_version_before: int = 0,
    intent_version_after: int = 0,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        user_message=f"user turn {turn}",
        intent_version_before=intent_version_before,
        accepted_update=accepted_update,
        intent_version_after=intent_version_after,
        assistant_message=assistant_message,
        question=question,
        question_key=question_key,
        ask_attribute="style" if question is not None else None,
        shown_product_ids=shown_product_ids,
        feedback=(),
        search_belief_probe_id=None,
    )


def _interaction(*records: TurnRecord) -> InteractionContext:
    return InteractionContext(turns=records)


def test_empty_history_has_empty_or_absent_views() -> None:
    interaction = _interaction()

    assert last_non_empty_shown_products(interaction) == ()
    assert all_shown_product_ids(interaction) == ()
    assert question_keys_since_goal_switch(interaction) == ()
    assert most_recent_question(interaction) is None
    assert most_recent_assistant_message(interaction) is None


def test_last_non_empty_shown_products_skips_more_recent_empty_batches() -> None:
    expected = ("sku-2", "sku-3")
    interaction = _interaction(
        _record(1, shown_product_ids=("sku-1",)),
        _record(2, shown_product_ids=expected),
        _record(3),
        _record(4),
    )

    assert last_non_empty_shown_products(interaction) is expected


def test_all_shown_product_ids_deduplicates_in_first_seen_order() -> None:
    interaction = _interaction(
        _record(1, shown_product_ids=("sku-2", "sku-1")),
        _record(2),
        _record(3, shown_product_ids=("sku-1", "sku-3", "sku-2", "sku-4")),
    )

    assert all_shown_product_ids(interaction) == ("sku-2", "sku-1", "sku-3", "sku-4")


def test_question_keys_without_a_goal_switch_are_first_seen_across_all_turns() -> None:
    interaction = _interaction(
        _record(1, question="Which color?", question_key="color"),
        _record(2),
        _record(3, question="Which material?", question_key="material"),
        _record(4, question="Still color?", question_key="color"),
    )

    assert question_keys_since_goal_switch(interaction) == ("color", "material")


def test_question_window_starts_at_the_most_recent_accepted_switch_turn() -> None:
    interaction = _interaction(
        _record(1, question="Which color?", question_key="color"),
        _record(
            2,
            question="Which style for the new goal?",
            question_key="style",
            accepted_update=_batch(2, switch_goal=True),
            intent_version_after=1,
        ),
        _record(3, question="Which material?", question_key="material"),
        _record(
            4,
            question="Which size for the latest goal?",
            question_key="size",
            accepted_update=_batch(4, switch_goal=True),
            intent_version_before=1,
            intent_version_after=2,
        ),
        _record(5, question="Still size?", question_key="size"),
        _record(6, question="Which brand?", question_key="brand"),
    )

    assert question_keys_since_goal_switch(interaction) == ("size", "brand")


def test_accepted_logical_no_op_switch_still_resets_the_question_window() -> None:
    interaction = _interaction(
        _record(1, question="Which color?", question_key="color"),
        _record(
            2,
            question="Which style?",
            question_key="style",
            accepted_update=_batch(2, switch_goal=True),
            intent_version_before=0,
            intent_version_after=0,
        ),
        _record(3, question="Which material?", question_key="material"),
    )

    assert question_keys_since_goal_switch(interaction) == ("style", "material")


def test_rejected_or_non_switch_updates_do_not_reset_the_question_window() -> None:
    interaction = _interaction(
        _record(1, question="Which color?", question_key="color"),
        _record(2, question="Which style?", question_key="style", accepted_update=None),
        _record(
            3,
            question="Which material?",
            question_key="material",
            accepted_update=_batch(3),
        ),
    )

    assert question_keys_since_goal_switch(interaction) == ("color", "style", "material")


def test_most_recent_question_skips_non_question_fallback_turns() -> None:
    interaction = _interaction(
        _record(1, question="Which color?", question_key="color"),
        _record(2, assistant_message="I could not refresh the results safely."),
        _record(3, assistant_message="Please try again."),
    )

    assert most_recent_question(interaction) == "Which color?"


def test_most_recent_assistant_message_uses_the_last_turn_including_fallbacks() -> None:
    interaction = _interaction(
        _record(1, assistant_message="Initial options."),
        _record(2, assistant_message="Safe fallback response."),
    )

    assert most_recent_assistant_message(interaction) == "Safe fallback response."
