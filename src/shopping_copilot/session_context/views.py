"""Pure derived views over immutable interaction history."""

from __future__ import annotations

from .aggregates import InteractionContext
from .operations import SwitchGoal


def last_non_empty_shown_products(interaction: InteractionContext) -> tuple[str, ...]:
    """Return the most recent non-empty externally shown product batch."""

    for record in reversed(interaction.turns):
        if record.shown_product_ids:
            return record.shown_product_ids
    return ()


def all_shown_product_ids(interaction: InteractionContext) -> tuple[str, ...]:
    """Return every shown product once, in first-seen order."""

    seen: set[str] = set()
    product_ids: list[str] = []
    for record in interaction.turns:
        for product_id in record.shown_product_ids:
            if product_id in seen:
                continue
            seen.add(product_id)
            product_ids.append(product_id)
    return tuple(product_ids)


def question_keys_since_goal_switch(interaction: InteractionContext) -> tuple[str, ...]:
    """Return first-seen question keys in the current accepted-goal window."""

    window_start = 0
    for index, record in enumerate(interaction.turns):
        batch = record.accepted_update
        if batch is not None and any(
            type(operation) is SwitchGoal for operation in batch.operations
        ):
            window_start = index

    seen: set[str] = set()
    question_keys: list[str] = []
    for record in interaction.turns[window_start:]:
        question_key = record.question_key
        if question_key is None or question_key in seen:
            continue
        seen.add(question_key)
        question_keys.append(question_key)
    return tuple(question_keys)


def most_recent_question(interaction: InteractionContext) -> str | None:
    """Return the most recent question, skipping turns that did not ask one."""

    for record in reversed(interaction.turns):
        if record.question is not None:
            return record.question
    return None


def most_recent_assistant_message(interaction: InteractionContext) -> str | None:
    """Return the assistant message from the most recent turn."""

    if not interaction.turns:
        return None
    return interaction.turns[-1].assistant_message
