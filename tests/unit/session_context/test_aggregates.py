"""Tests for immutable interaction and session aggregate values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from shopping_copilot.session_context.aggregates import (
    InteractionContext,
    ProductFeedback,
    SessionContext,
    SessionState,
    TurnRecord,
)
from shopping_copilot.session_context.models import (
    CandidateMode,
    CertaintyEvidence,
    FeedbackSignal,
    IntentState,
    ProbeQuality,
    ProfilePrior,
    SearchBelief,
)
from shopping_copilot.session_context.operations import ClearFacet, StateUpdateBatch


def _intent(*, version: int = 0) -> IntentState:
    return IntentState(
        goal=None,
        preferences=(),
        dont_care_facets=frozenset(),
        version=version,
    )


def _profile() -> ProfilePrior:
    return ProfilePrior(
        purchase_frequency="monthly",
        average_prior_rating=None,
        rating_style="balanced",
        preference_tags=("durable",),
        summary="Prefers durable products.",
    )


def _belief() -> SearchBelief:
    return SearchBelief(
        based_on_intent_version=0,
        certainty=0.8,
        certainty_method="bods_v1",
        certainty_evidence=CertaintyEvidence(
            probe_id="probe-1",
            probe_size=2,
            raw_concentration=0.8,
            quality_status=ProbeQuality.VALID,
            quality_reasons=(),
        ),
        candidate_modes=(
            CandidateMode(
                id="mode-1",
                label="primary",
                mass=1.0,
                representative_ids=("sku-1",),
            ),
        ),
        facet_stats=(),
    )


def _feedback() -> ProductFeedback:
    return ProductFeedback(
        product_ids=("sku-1",),
        signal=FeedbackSignal.POSITIVE,
        compared_to_ids=(),
        evidence_text="I like this one.",
    )


def _turn_record(
    *,
    intent_version_before: int = 0,
    intent_version_after: int = 0,
) -> TurnRecord:
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(ClearFacet(facet="color"),),
    )
    return TurnRecord(
        turn=1,
        user_message="Show me another option.",
        intent_version_before=intent_version_before,
        accepted_update=batch,
        intent_version_after=intent_version_after,
        assistant_message="Here is another option.",
        question=None,
        question_key=None,
        ask_attribute=None,
        shown_product_ids=("sku-2",),
        feedback=(_feedback(),),
        search_belief_probe_id="probe-1",
    )


def _aggregate_instances() -> tuple[object, ...]:
    feedback = _feedback()
    record = _turn_record()
    interaction = InteractionContext(turns=(record,))
    state = SessionState(
        intent=_intent(),
        interaction=interaction,
        search_belief=_belief(),
    )
    context = SessionContext(session_id="session-1", profile=_profile(), state=state)
    return feedback, record, interaction, state, context


@pytest.mark.parametrize(
    "instance",
    _aggregate_instances(),
    ids=lambda instance: type(instance).__name__,
)
def test_aggregate_values_are_frozen_and_slotted(instance: object) -> None:
    first_field = fields(instance)[0]

    assert type(instance).__dataclass_params__.frozen is True
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, first_field.name, object())


def test_aggregate_collection_fields_use_immutable_domain_types() -> None:
    feedback, record, interaction, state, context = _aggregate_instances()

    assert isinstance(feedback, ProductFeedback)
    assert isinstance(state, SessionState)
    assert isinstance(context, SessionContext)
    assert type(feedback.product_ids) is tuple
    assert type(feedback.compared_to_ids) is tuple
    assert type(record.shown_product_ids) is tuple
    assert type(record.feedback) is tuple
    assert type(interaction.turns) is tuple


def test_turn_record_derives_state_changed_from_intent_versions() -> None:
    assert _turn_record(intent_version_before=3, intent_version_after=3).state_changed is False
    assert _turn_record(intent_version_before=3, intent_version_after=4).state_changed is True


def test_turn_record_retains_the_exact_committed_batch_and_normalized_values() -> None:
    record = _turn_record()

    assert record.accepted_update is not None
    assert record.accepted_update.operations == (ClearFacet(facet="color"),)
    assert record.shown_product_ids == ("sku-2",)
    assert record.feedback == (_feedback(),)


def test_session_context_allows_absent_profile_and_search_belief() -> None:
    interaction = InteractionContext(turns=())
    state = SessionState(
        intent=_intent(),
        interaction=interaction,
        search_belief=None,
    )
    context = SessionContext(session_id="generic-session", profile=None, state=state)

    assert context.profile is None
    assert context.state.search_belief is None
    assert context.state.interaction.turns == ()


def test_session_context_preserves_one_complete_nested_snapshot() -> None:
    interaction = InteractionContext(turns=(_turn_record(),))
    belief = _belief()
    state = SessionState(
        intent=_intent(),
        interaction=interaction,
        search_belief=belief,
    )
    profile = _profile()
    context = SessionContext(session_id="session-1", profile=profile, state=state)

    assert context.profile is profile
    assert context.state is state
    assert context.state.interaction is interaction
    assert context.state.search_belief is belief
