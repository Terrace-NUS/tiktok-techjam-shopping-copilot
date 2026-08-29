from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace

import pytest
from test_runtime_projection import _build_runtime_bundle

from shopping_copilot.catalog.semantic import canonical_json_bytes, content_id_for_value
from shopping_copilot.catalog.semantic.gateway import (
    CATALOG_BOUND_SESSION_SCHEMA,
    CatalogBoundSessionStore,
    CatalogGatewayError,
    CatalogGatewayErrorCode,
    CatalogProbeToken,
    CatalogSemanticGateway,
)
from shopping_copilot.catalog.semantic.release import (
    VerifiedCatalogSemanticRelease,
    write_catalog_semantic_release,
)
from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
from shopping_copilot.query_understanding import (
    IntentMaterializer,
    build_reconcile_request,
    category_options_from_registry,
    decode_reconciled_intent,
)
from shopping_copilot.session_context import (
    AddPreference,
    CertaintyEvidence,
    ClearFacet,
    Commitment,
    FacetAuthority,
    FacetStats,
    IntentState,
    InteractionContext,
    Operator,
    Preference,
    PreferenceSource,
    ProbeQuality,
    RemovePreference,
    ReplaceFacet,
    SearchBelief,
    SemanticPolarity,
    SessionContext,
    SessionState,
    SetDontCare,
    StateUpdateBatch,
    SwitchGoal,
    TurnRecord,
    ValueMass,
    encode_snapshot,
    reduce_intent,
)


@pytest.fixture(scope="module")
def gateway_release(tmp_path_factory: pytest.TempPathFactory) -> VerifiedCatalogSemanticRelease:
    root = tmp_path_factory.mktemp("gateway-release")
    _, approved, runtime = _build_runtime_bundle(root)
    return write_catalog_semantic_release(
        approved[2],
        approved[3],
        approved[4],
        approved[5],
        approved[6],
        approved[7],
        approved[8],
        approved[9],
        runtime,
        root / "release",
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )


def _preference(
    *,
    preference_id: str,
    turn: int,
    facet: str,
    operator: Operator,
    value: str | int,
) -> Preference:
    return Preference(
        id=preference_id,
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=turn,
        evidence_text="explicit user requirement",
        interpretation_confidence=1.0,
    )


def _category_preference(
    scope_id: str,
    *,
    preference_id: str = "p_1_0_0",
    turn: int = 1,
) -> Preference:
    return _preference(
        preference_id=preference_id,
        turn=turn,
        facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        operator=Operator.EQ,
        value=scope_id,
    )


def _price_preference(
    *,
    preference_id: str = "p_1_1_0",
    turn: int = 1,
    value: int = 2500,
) -> Preference:
    return _preference(
        preference_id=preference_id,
        turn=turn,
        facet="price",
        operator=Operator.LE,
        value=value,
    )


def _category_and_price_batch(
    release: VerifiedCatalogSemanticRelease,
) -> StateUpdateBatch:
    return StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(_category_preference(release.category_registry.root_scope_id),),
            ),
            AddPreference(preference=_price_preference()),
        ),
    )


def _next_context(
    previous: SessionContext,
    *,
    intent: IntentState,
    batch: StateUpdateBatch | None,
    belief: SearchBelief | None = None,
) -> SessionContext:
    turn = len(previous.state.interaction.turns) + 1
    record = TurnRecord(
        turn=turn,
        user_message="test user turn",
        intent_version_before=previous.state.intent.version,
        accepted_update=batch,
        intent_version_after=intent.version,
        assistant_message="test response",
        question=None,
        question_key=None,
        ask_attribute=None,
        shown_product_ids=(),
        feedback=(),
        search_belief_probe_id=(belief.certainty_evidence.probe_id if belief is not None else None),
    )
    return replace(
        previous,
        state=SessionState(
            intent=intent,
            interaction=InteractionContext(turns=previous.state.interaction.turns + (record,)),
            search_belief=belief,
        ),
    )


def _belief(
    intent_version: int,
    *,
    facet: str = "price",
    value: str | int = 2500,
) -> SearchBelief:
    return SearchBelief(
        based_on_intent_version=intent_version,
        certainty=None,
        certainty_method="candidate_concentration_v1",
        certainty_evidence=CertaintyEvidence(
            probe_id="probe_1",
            probe_size=5,
            raw_concentration=None,
            quality_status=ProbeQuality.INSUFFICIENT,
            quality_reasons=("insufficient_evidence",),
        ),
        candidate_modes=(),
        facet_stats=(
            FacetStats(
                facet=facet,
                entropy=0.0,
                coverage=1.0,
                top_values=(ValueMass(value=value, mass=1.0),),
            ),
        ),
    )


def _release_without_price_capability(
    release: VerifiedCatalogSemanticRelease,
    scope_id: str,
) -> VerifiedCatalogSemanticRelease:
    if scope_id not in {item.id for item in release.category_registry.scopes}:
        root = next(
            item
            for item in release.category_registry.scopes
            if item.id == release.category_registry.root_scope_id
        )
        category_registry = replace(
            release.category_registry,
            scopes=tuple(
                sorted(
                    release.category_registry.scopes
                    + (replace(root, id=scope_id, label="Synthetic test scope"),),
                    key=lambda item: item.id,
                )
            ),
        )
    else:
        category_registry = release.category_registry
    capabilities = replace(
        release.effective_capabilities,
        entries=tuple(
            item
            for item in release.effective_capabilities.entries
            if (item.facet_id, item.category_scope_id) != ("price", scope_id)
        ),
    )
    runtime_registry = replace(
        release.runtime_registry,
        category_registry_id=content_id_for_value(category_registry),
        effective_capabilities_id=content_id_for_value(capabilities),
    )
    runtime_lexicon = replace(
        release.runtime_value_lexicon,
        runtime_registry_id=content_id_for_value(runtime_registry),
        category_registry_id=content_id_for_value(category_registry),
    )
    return replace(
        release,
        category_registry=category_registry,
        effective_capabilities=capabilities,
        runtime_registry=runtime_registry,
        runtime_value_lexicon=runtime_lexicon,
    )


def test_gateway_accepts_grounded_category_and_price_atomically(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    current = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    result = gateway.preview(
        current,
        _category_and_price_batch(gateway_release),
        catalog_semantic_release_id=gateway_release.release_id,
    )

    assert result.version == 1
    assert tuple(item.facet for item in result.preferences) == (
        SYSTEM_PRODUCT_CATEGORY_FACET_ID,
        "price",
    )


def test_gateway_accepts_normalized_hard_retrieval_derived_preference(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    material = gateway.registry.require("material")
    normalized = gateway.registry.normalize_value("material", Operator.EQ, "  SILK  ")
    assert type(normalized) is str
    preference = _preference(
        preference_id="p_1_0_0",
        turn=1,
        facet="material",
        operator=Operator.EQ,
        value=normalized,
    )

    result = gateway.preview(
        IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0),
        StateUpdateBatch(
            turn=1,
            base_intent_version=0,
            operations=(AddPreference(preference=preference),),
        ),
        catalog_semantic_release_id=gateway_release.release_id,
    )

    assert material.authority is FacetAuthority.RETRIEVAL_DERIVED
    assert result.preferences == (preference,)
    assert result.preferences[0].commitment is Commitment.HARD
    assert result.preferences[0].value == "silk"


def test_gateway_set_dont_care_clears_retrieval_derived_facet(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    initial = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    current = gateway.preview(
        initial,
        StateUpdateBatch(
            turn=1,
            base_intent_version=0,
            operations=(
                AddPreference(
                    preference=_preference(
                        preference_id="p_1_0_0",
                        turn=1,
                        facet="color",
                        operator=Operator.EQ,
                        value="black",
                    )
                ),
            ),
        ),
        catalog_semantic_release_id=gateway_release.release_id,
    )

    result = gateway.preview(
        current,
        StateUpdateBatch(
            turn=2,
            base_intent_version=current.version,
            operations=(SetDontCare(facet="color"),),
        ),
        catalog_semantic_release_id=gateway_release.release_id,
    )

    assert result.preferences == ()
    assert result.dont_care_facets == frozenset({"color"})


def test_gateway_accepts_retrieval_derived_probe_facet_stats(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    intent = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)

    gateway.validate_search_belief(
        _belief(0, facet="color", value="black"),
        intent=intent,
        catalog_semantic_release_id=gateway_release.release_id,
    )


@pytest.mark.parametrize(
    "operation",
    [
        AddPreference(preference=_category_preference("cs_unknown")),
        ClearFacet(facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID),
        SetDontCare(facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID),
        RemovePreference(preference_ids=("p_1_0_0",)),
    ],
    ids=("add", "clear", "dont_care", "remove_by_id"),
)
def test_reserved_category_rejects_ordinary_facet_operations(
    gateway_release: VerifiedCatalogSemanticRelease,
    operation: object,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    current = IntentState(
        goal=None,
        preferences=(_category_preference(gateway_release.category_registry.root_scope_id),),
        dont_care_facets=frozenset(),
        version=0,
    )
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(operation,),  # type: ignore[arg-type]
    )

    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            batch,
            catalog_semantic_release_id=gateway_release.release_id,
        )
    assert caught.value.code is CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION


def test_unknown_category_scope_has_catalog_error_before_raw_normalization(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    current = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(_category_preference("cs_unknown"),),
            ),
        ),
    )
    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            batch,
            catalog_semantic_release_id=gateway_release.release_id,
        )
    assert caught.value.code is CatalogGatewayErrorCode.UNKNOWN_CATEGORY_SCOPE


@pytest.mark.parametrize(
    "preferences",
    [
        (
            _preference(
                preference_id="p_1_0_0",
                turn=1,
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                operator=Operator.NEQ,
                value="cs_" + "1" * 64,
            ),
        ),
        (
            _category_preference("cs_" + "1" * 64),
            _category_preference(
                "cs_" + "2" * 64,
                preference_id="p_1_0_1",
            ),
        ),
    ],
    ids=("non_eq", "multiple_preferences"),
)
def test_reserved_category_replacement_requires_exact_shape(
    gateway_release: VerifiedCatalogSemanticRelease,
    preferences: tuple[Preference, ...],
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    current = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=preferences,
            ),
        ),
    )
    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            batch,
            catalog_semantic_release_id=gateway_release.release_id,
        )
    assert caught.value.code is CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION


def test_category_remove_cannot_target_a_replacement_created_in_same_batch(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    current = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    category = _category_preference(gateway_release.category_registry.root_scope_id)
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(category,),
            ),
            RemovePreference(preference_ids=(category.id,)),
        ),
    )
    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            batch,
            catalog_semantic_release_id=gateway_release.release_id,
        )
    assert caught.value.code is CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION


def test_category_replacement_must_immediately_follow_optional_goal_switch(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    current = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    batch = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            SwitchGoal(new_goal="new shopping goal"),
            AddPreference(preference=_price_preference(preference_id="p_1_1_0")),
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(
                    _category_preference(
                        gateway_release.category_registry.root_scope_id,
                        preference_id="p_1_2_0",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            batch,
            catalog_semantic_release_id=gateway_release.release_id,
        )
    assert caught.value.code is CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION
    assert caught.value.operation_index == 2


def test_switch_goal_can_only_carry_or_drop_the_existing_category(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    initial = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    current = gateway.preview(
        initial,
        _category_and_price_batch(gateway_release),
        catalog_semantic_release_id=gateway_release.release_id,
    )
    category_id, price_id = (item.id for item in current.preferences)
    carried = gateway.preview(
        current,
        StateUpdateBatch(
            turn=2,
            base_intent_version=current.version,
            operations=(
                SwitchGoal(
                    new_goal="carried goal",
                    carry_preference_ids=(category_id, price_id),
                ),
            ),
        ),
        catalog_semantic_release_id=gateway_release.release_id,
    )
    assert carried.preferences == current.preferences

    dropped = gateway.preview(
        current,
        StateUpdateBatch(
            turn=2,
            base_intent_version=current.version,
            operations=(
                SwitchGoal(
                    new_goal="broader goal",
                    carry_preference_ids=(price_id,),
                ),
            ),
        ),
        catalog_semantic_release_id=gateway_release.release_id,
    )
    assert tuple(item.facet for item in dropped.preferences) == ("price",)


def test_switch_goal_may_be_immediately_followed_by_valid_category_replacement(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    initial = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    current = gateway.preview(
        initial,
        _category_and_price_batch(gateway_release),
        catalog_semantic_release_id=gateway_release.release_id,
    )
    replacement = replace(
        _category_preference(
            gateway_release.category_registry.root_scope_id,
            preference_id="p_2_1_0",
            turn=2,
        ),
        semantic_text="all products",
        semantic_polarity=SemanticPolarity.POSITIVE,
    )
    result = gateway.preview(
        current,
        StateUpdateBatch(
            turn=2,
            base_intent_version=current.version,
            operations=(
                SwitchGoal(new_goal="replacement goal"),
                ReplaceFacet(
                    facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                    preferences=(replacement,),
                ),
            ),
        ),
        catalog_semantic_release_id=gateway_release.release_id,
    )
    assert result.preferences == (replacement,)


def test_category_change_rejects_incompatible_retained_preference_without_repair(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    root_scope = gateway_release.category_registry.root_scope_id
    target_scope = "cs_" + "1" * 64
    assert target_scope != root_scope
    release = _release_without_price_capability(gateway_release, target_scope)
    gateway = CatalogSemanticGateway(release)
    initial = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    current = gateway.preview(
        initial,
        _category_and_price_batch(release),
        catalog_semantic_release_id=release.release_id,
    )
    change = StateUpdateBatch(
        turn=2,
        base_intent_version=current.version,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(
                    _category_preference(
                        target_scope,
                        preference_id="p_2_0_0",
                        turn=2,
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            change,
            catalog_semantic_release_id=release.release_id,
        )
    assert (
        caught.value.code is CatalogGatewayErrorCode.INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE
    )
    assert current.preferences[-1].facet == "price"


def test_category_change_rejects_incompatible_retained_dont_care(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    root_scope = gateway_release.category_registry.root_scope_id
    target_scope = "cs_" + "1" * 64
    release = _release_without_price_capability(gateway_release, target_scope)
    gateway = CatalogSemanticGateway(release)
    initial = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    setup = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(_category_preference(root_scope),),
            ),
            SetDontCare(facet="price"),
        ),
    )
    current = gateway.preview(
        initial,
        setup,
        catalog_semantic_release_id=release.release_id,
    )
    change = StateUpdateBatch(
        turn=2,
        base_intent_version=current.version,
        operations=(
            ReplaceFacet(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                preferences=(
                    _category_preference(
                        target_scope,
                        preference_id="p_2_0_0",
                        turn=2,
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(CatalogGatewayError) as caught:
        gateway.preview(
            current,
            change,
            catalog_semantic_release_id=release.release_id,
        )
    assert (
        caught.value.code is CatalogGatewayErrorCode.INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE
    )
    assert current.dont_care_facets == frozenset({"price"})


def test_bound_store_commits_only_gateway_equal_intent(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="atomic-store")
    gateway = CatalogSemanticGateway(gateway_release)
    batch = _category_and_price_batch(gateway_release)
    intent = gateway.preview(
        previous.state.intent,
        batch,
        catalog_semantic_release_id=gateway_release.release_id,
    )
    candidate = _next_context(previous, intent=intent, batch=batch)

    with store.turn(session_id="atomic-store", turn=1) as transaction:
        committed = transaction.commit(candidate)
    assert store.get("atomic-store") == committed


def test_bound_transaction_exposes_read_only_update_preview(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="transaction-preview")
    batch = _category_and_price_batch(gateway_release)

    with store.turn(session_id=previous.session_id, turn=1) as transaction:
        previewed = transaction.preview_update(batch)
        with pytest.raises(CatalogGatewayError) as caught:
            transaction.preview_update(replace(batch, turn=2))

    assert previewed.version == 1
    assert caught.value.code is CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH
    assert store.get(previous.session_id) == previous


def test_query_understanding_materializes_and_commits_wide_facet_through_real_gateway(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="qu-real-gateway")
    gateway = CatalogSemanticGateway(gateway_release)
    request = build_reconcile_request(
        turn=1,
        latest_utterance="I want a silk one, but not black.",
        current_intent=previous.state.intent,
        category_options=category_options_from_registry(gateway_release.category_registry),
    )
    frame = decode_reconciled_intent(
        json.dumps(
            {
                "base_intent_version": 0,
                "disposition": "ready",
                "goal": {"action": "keep", "value": None},
                "keep_active_refs": [],
                "new_preferences": {
                    "structured": [
                        {
                            "facet": "material",
                            "relation": "eq",
                            "values": ["silk"],
                            "strength": "hard",
                            "basis": "explicit",
                            "meaning": "must be silk",
                            "evidence": "silk one",
                            "confidence": 1.0,
                        },
                        {
                            "facet": "color",
                            "relation": "not_in",
                            "values": ["black"],
                            "strength": "hard",
                            "basis": "explicit",
                            "meaning": "must not be black",
                            "evidence": "not black",
                            "confidence": 1.0,
                        },
                    ],
                    "price": [],
                    "semantic": [],
                },
                "dont_care_facets": [],
                "feedback": [],
                "directives": {
                    "diversity": "auto",
                    "comparison_requested": False,
                    "explanation_requested": False,
                },
                "clarification": {"needed": False, "reason": None, "alternatives": []},
                "summary": "Silk and not black.",
            }
        )
    )
    resolved = IntentMaterializer(
        gateway=gateway,
        grounder=gateway_release.grounder,
    ).materialize(current=previous.state.intent, request=request, frame=frame)
    assert resolved.update is not None
    candidate = _next_context(
        previous,
        intent=resolved.final_intent,
        batch=resolved.update,
    )

    with store.turn(session_id=previous.session_id, turn=1) as transaction:
        assert transaction.preview_update(resolved.update) == resolved.final_intent
        committed = transaction.commit(candidate)

    assert tuple(item.facet for item in committed.state.intent.preferences) == (
        "color",
        "material",
    )
    assert committed.state.intent.preferences[0].operator is Operator.NOT_IN


def test_bound_store_rejects_next_context_that_does_not_equal_gateway_result(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="gateway-equality")
    batch = _category_and_price_batch(gateway_release)
    mismatched = _next_context(previous, intent=previous.state.intent, batch=batch)

    with store.turn(session_id=previous.session_id, turn=1) as transaction:
        with pytest.raises(CatalogGatewayError) as caught:
            transaction.commit(mismatched)
    assert caught.value.code is CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH
    assert store.get(previous.session_id) == previous


def test_caller_constructed_search_belief_cannot_enter_live_state(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="untrusted-belief")
    belief = _belief(previous.state.intent.version)
    candidate = _next_context(
        previous,
        intent=previous.state.intent,
        batch=None,
        belief=belief,
    )

    with store.turn(session_id="untrusted-belief", turn=1) as transaction:
        with pytest.raises(CatalogGatewayError) as caught:
            transaction.commit(candidate)
    assert caught.value.code is CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF
    assert store.get("untrusted-belief") == previous


def test_private_probe_token_is_exact_and_one_use(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="trusted-belief")
    belief = _belief(previous.state.intent.version)
    candidate = _next_context(
        previous,
        intent=previous.state.intent,
        batch=None,
        belief=belief,
    )

    with store.turn(session_id="trusted-belief", turn=1) as transaction:
        token = store._probe_producer.issue_token(
            transaction,
            expected_final_intent=previous.state.intent,
            belief=belief,
        )
        committed = transaction.commit(candidate, probe_token=token)
    assert committed.state.search_belief == belief
    assert token._used is True


def test_forged_probe_token_is_rejected(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="forged-belief")
    belief = _belief(previous.state.intent.version)
    candidate = _next_context(
        previous,
        intent=previous.state.intent,
        batch=None,
        belief=belief,
    )
    forged = CatalogProbeToken(
        _authority=object(),
        _release_id=gateway_release.release_id,
        _session_id=previous.session_id,
        _transaction_token=object(),
        _captured_context=previous,
        _expected_final_intent=previous.state.intent,
        _belief=belief,
    )

    with store.turn(session_id="forged-belief", turn=1) as transaction:
        with pytest.raises(CatalogGatewayError) as caught:
            transaction.commit(candidate, probe_token=forged)
    assert caught.value.code is CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF
    assert forged._used is True


def test_probe_token_from_an_abandoned_transaction_is_rejected(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="transaction-bound-belief")
    belief = _belief(previous.state.intent.version)
    candidate = _next_context(
        previous,
        intent=previous.state.intent,
        batch=None,
        belief=belief,
    )
    with store.turn(session_id=previous.session_id, turn=1) as abandoned:
        token = store._probe_producer.issue_token(
            abandoned,
            expected_final_intent=previous.state.intent,
            belief=belief,
        )

    with store.turn(session_id=previous.session_id, turn=1) as replacement:
        with pytest.raises(CatalogGatewayError) as caught:
            replacement.commit(candidate, probe_token=token)
    assert caught.value.code is CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF
    assert token._used is True


def test_reserved_category_is_never_probe_eligible(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    gateway = CatalogSemanticGateway(gateway_release)
    intent = IntentState(goal=None, preferences=(), dont_care_facets=frozenset(), version=0)
    with pytest.raises(CatalogGatewayError) as caught:
        gateway.validate_search_belief(
            _belief(0, facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID),
            intent=intent,
            catalog_semantic_release_id=gateway_release.release_id,
        )
    assert caught.value.code is CatalogGatewayErrorCode.PROBE_FACET_NOT_ELIGIBLE


def test_catalog_bound_envelope_round_trips_and_binds_release(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="envelope-session")
    gateway = CatalogSemanticGateway(gateway_release)
    batch = _category_and_price_batch(gateway_release)
    intent = gateway.preview(
        previous.state.intent,
        batch,
        catalog_semantic_release_id=gateway_release.release_id,
    )
    candidate = _next_context(previous, intent=intent, batch=batch)
    with store.turn(session_id=previous.session_id, turn=1) as transaction:
        transaction.commit(candidate)

    encoded = store.encode(store.get(previous.session_id))
    document = json.loads(encoded)
    assert document["schema"] == CATALOG_BOUND_SESSION_SCHEMA
    assert document["catalog_semantic_release_id"] == gateway_release.release_id
    assert store.decode(encoded) == candidate

    document["catalog_semantic_release_id"] = "sha256:" + "0" * 64
    with pytest.raises(CatalogGatewayError) as caught:
        store.decode(canonical_json_bytes(document))
    assert caught.value.code is CatalogGatewayErrorCode.RELEASE_MISMATCH


def test_envelope_hash_and_canonical_bytes_fail_closed(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    context = store.reset(session_id="envelope-integrity")
    encoded = store.encode(context)
    document = json.loads(encoded)
    document["session_snapshot_sha256"] = "0" * 64

    with pytest.raises(CatalogGatewayError) as caught:
        store.decode(canonical_json_bytes(document))
    assert caught.value.code is CatalogGatewayErrorCode.SESSION_SNAPSHOT_HASH_MISMATCH

    with pytest.raises(CatalogGatewayError) as caught:
        store.decode(encoded + b"\n")
    assert caught.value.code is CatalogGatewayErrorCode.INVALID_SESSION_ENVELOPE


def test_envelope_replay_rejects_raw_valid_gateway_bypass(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    previous = store.reset(session_id="gateway-replay")
    bypass = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(
            AddPreference(
                preference=_category_preference(
                    gateway_release.category_registry.root_scope_id,
                )
            ),
        ),
    )
    raw_intent = reduce_intent(previous.state.intent, bypass, store._gateway.registry)
    raw_valid_context = _next_context(previous, intent=raw_intent, batch=bypass)
    inner = encode_snapshot(raw_valid_context, store._gateway.registry)
    envelope = canonical_json_bytes(
        {
            "schema": CATALOG_BOUND_SESSION_SCHEMA,
            "session_id": previous.session_id,
            "catalog_semantic_release_id": gateway_release.release_id,
            "session_snapshot_sha256": hashlib.sha256(inner).hexdigest(),
            "session_snapshot_base64url": base64.urlsafe_b64encode(inner)
            .rstrip(b"=")
            .decode("ascii"),
        }
    )

    with pytest.raises(CatalogGatewayError) as caught:
        store.decode(envelope)
    assert caught.value.code is CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION


def test_envelope_outer_and_inner_session_ids_must_match(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    context = store.reset(session_id="session-id-binding")
    document = json.loads(store.encode(context))
    document["session_id"] = "different-session"

    with pytest.raises(CatalogGatewayError) as caught:
        store.decode(canonical_json_bytes(document))
    assert caught.value.code is CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH


def test_every_store_method_rejects_an_explicit_different_release(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    wrong = "sha256:" + "0" * 64
    with pytest.raises(CatalogGatewayError) as caught:
        store.reset(session_id="wrong-release", expected_release_id=wrong)
    assert caught.value.code is CatalogGatewayErrorCode.RELEASE_MISMATCH


def test_store_exposes_no_public_raw_commit_or_registry_handle(
    gateway_release: VerifiedCatalogSemanticRelease,
) -> None:
    store = CatalogBoundSessionStore(gateway_release)
    public_names = {name for name in dir(store) if not name.startswith("_")}
    assert "commit" not in public_names
    assert "registry" not in public_names
    assert "raw_store" not in public_names
