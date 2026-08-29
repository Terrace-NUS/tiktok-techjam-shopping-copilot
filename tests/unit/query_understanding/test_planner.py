from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from shopping_copilot.catalog.semantic.runtime import (
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    GroundedPredicate,
    GroundingDisposition,
    RuntimeValueGroundingResult,
)
from shopping_copilot.query_understanding import (
    BehavioralDirectives,
    CategoryOption,
    ClarificationNeed,
    DiversityMode,
    FeedbackFrame,
    GoalAction,
    GoalFrame,
    IntentMaterializer,
    NewPreferenceFrame,
    PreferenceBasis,
    PreferenceRelation,
    PreferenceStrength,
    PricePreferenceFrame,
    ProviderResult,
    ProviderTrace,
    QueryUnderstandingError,
    QueryUnderstandingErrorCode,
    QueryUnderstandingService,
    ReconciledIntentFrame,
    SemanticPreferenceFrame,
    ShownProductView,
    StructuredPreferenceFrame,
    UnderstandingDisposition,
    build_reconcile_request,
)
from shopping_copilot.session_context import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    ClearFacet,
    Commitment,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    FeedbackSignal,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    ReplaceFacet,
    SemanticPolarity,
    SetDontCare,
    StateUpdateBatch,
    SwitchGoal,
    canonical_number,
    canonical_text,
    reduce_intent,
    with_retrieval_derived_facets,
)


def test_typed_preference_frames_reject_plain_string_relations() -> None:
    with pytest.raises(TypeError, match="structured preference relation"):
        StructuredPreferenceFrame(
            facet="color",
            relation=cast(PreferenceRelation, "eq"),
            values=("black",),
            strength=PreferenceStrength.HARD,
            basis=PreferenceBasis.EXPLICIT,
            meaning="user requirement",
            evidence="user requirement",
            confidence=1.0,
        )

    with pytest.raises(TypeError, match="price preference relation"):
        PricePreferenceFrame(
            relation=cast(PreferenceRelation, "le"),
            value_usd="120",
            strength=PreferenceStrength.HARD,
            basis=PreferenceBasis.EXPLICIT,
            meaning="user requirement",
            evidence="user requirement",
            confidence=1.0,
        )


class _Previewer:
    release_id = "test-release"

    def __init__(self) -> None:
        self._registry = with_retrieval_derived_facets(
            FacetRegistry(
                specs=(
                    FacetSpec(
                        id="price",
                        kind=FacetKind.NUMERIC,
                        operators=NUMERIC_OPERATORS,
                        normalizer=canonical_number,
                    ),
                    FacetSpec(
                        id=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                        kind=FacetKind.CATEGORICAL,
                        operators=CATEGORICAL_OPERATORS,
                        normalizer=canonical_text,
                    ),
                )
            )
        )

    @property
    def registry(self) -> FacetRegistry:
        return self._registry

    def preview(
        self,
        current: IntentState,
        batch: StateUpdateBatch,
        *,
        catalog_semantic_release_id: str,
    ) -> IntentState:
        assert catalog_semantic_release_id == self.release_id
        return reduce_intent(current, batch, self.registry)


class _Grounder:
    def ground(self, candidate, *, final_category_scope_id):
        assert candidate.facet_id == "price"
        assert isinstance(candidate.operator, Operator)
        assert candidate.value is not None
        return RuntimeValueGroundingResult(
            facet_id="price",
            disposition=GroundingDisposition.GROUNDED,
            predicates=(
                GroundedPredicate(
                    facet_id="price",
                    operator=candidate.operator,
                    value=candidate.value,
                ),
            ),
            reason_code=None,
            candidate_values=(),
            semantic_text=None,
            semantic_polarity=None,
        )


@pytest.fixture
def materializer() -> IntentMaterializer:
    return IntentMaterializer(gateway=_Previewer(), grounder=_Grounder())


def _current(
    *preferences: Preference,
    version: int = 0,
    goal: str | None = None,
) -> IntentState:
    return IntentState(
        goal=goal,
        preferences=preferences,
        dont_care_facets=frozenset(),
        version=version,
    )


def _old_preference(
    *,
    facet: str = "color",
    operator: Operator = Operator.EQ,
    value: str | int = "black",
) -> Preference:
    return Preference(
        id="p_1_0_0",
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text="old explicit preference",
        interpretation_confidence=1.0,
    )


def _request(
    current: IntentState,
    *,
    turn: int,
    utterance: str = "test utterance",
    shown: tuple[ShownProductView, ...] = (),
):
    return build_reconcile_request(
        turn=turn,
        latest_utterance=utterance,
        current_intent=current,
        category_options=(
            CategoryOption(ref="category_0", scope_id="scope_all", label="All", is_root=True),
        ),
        shown_products=shown,
    )


def _new_preference(
    *,
    facet: str | None,
    relation: PreferenceRelation,
    values: tuple[str, ...] = (),
    numeric_value_usd: str | None = None,
    strength: PreferenceStrength = PreferenceStrength.HARD,
    basis: PreferenceBasis = PreferenceBasis.EXPLICIT,
    meaning: str = "explicit condition",
    semantic_polarity: SemanticPolarity = SemanticPolarity.POSITIVE,
) -> NewPreferenceFrame:
    common = {
        "strength": strength,
        "basis": basis,
        "meaning": meaning,
        "evidence": "user said it",
        "confidence": 1.0,
    }
    if facet == "price":
        assert numeric_value_usd is not None
        return PricePreferenceFrame(
            relation=relation,
            value_usd=numeric_value_usd,
            **common,
        )
    if facet is None:
        return SemanticPreferenceFrame(polarity=semantic_polarity, **common)
    return StructuredPreferenceFrame(
        facet=facet,
        relation=relation,
        values=values,
        **common,
    )


def _frame(
    current: IntentState,
    *,
    keep: tuple[str, ...] = (),
    new: tuple[NewPreferenceFrame, ...] = (),
    dont_care: tuple[str, ...] = (),
    feedback: tuple[FeedbackFrame, ...] = (),
    disposition: UnderstandingDisposition = UnderstandingDisposition.READY,
    goal: GoalFrame | None = None,
) -> ReconciledIntentFrame:
    return ReconciledIntentFrame(
        base_intent_version=current.version,
        disposition=disposition,
        goal=goal or GoalFrame(action=GoalAction.KEEP, value=None),
        keep_active_refs=keep,
        structured_preferences=tuple(
            item for item in new if isinstance(item, StructuredPreferenceFrame)
        ),
        price_preferences=tuple(item for item in new if isinstance(item, PricePreferenceFrame)),
        semantic_preferences=tuple(
            item for item in new if isinstance(item, SemanticPreferenceFrame)
        ),
        dont_care_facets=dont_care,
        feedback=feedback,
        directives=BehavioralDirectives(
            diversity=DiversityMode.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        clarification=ClarificationNeed(needed=False, reason=None, alternatives=()),
        summary="understood",
    )


def test_materializer_commits_wide_structured_facet_and_canonicalizes_value(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    request = _request(current, turn=1, utterance="I want silk")
    result = materializer.materialize(
        current=current,
        request=request,
        frame=_frame(
            current,
            new=(
                _new_preference(
                    facet="material",
                    relation=PreferenceRelation.EQ,
                    values=("  SILK  ",),
                ),
            ),
        ),
    )

    assert result.update is not None
    assert type(result.update.operations[0]) is ReplaceFacet
    assert result.final_intent.preferences[0].id == "p_1_0_0"
    assert result.final_intent.preferences[0].value == "silk"
    assert result.final_intent.preferences[0].commitment is Commitment.HARD


def test_complete_frame_distinguishes_removing_black_from_excluding_black(
    materializer: IntentMaterializer,
) -> None:
    current = _current(_old_preference(), version=1)
    request = _request(current, turn=2)

    removed = materializer.materialize(
        current=current,
        request=request,
        frame=_frame(current),
    )
    excluded = materializer.materialize(
        current=current,
        request=request,
        frame=_frame(
            current,
            new=(
                _new_preference(
                    facet="color",
                    relation=PreferenceRelation.NOT_IN,
                    values=("black",),
                    meaning="must not be black",
                ),
            ),
        ),
    )

    assert removed.update is not None
    assert type(removed.update.operations[0]) is ClearFacet
    assert removed.final_intent.preferences == ()
    assert excluded.final_intent.preferences[0].operator is Operator.NOT_IN
    assert excluded.final_intent.preferences[0].value == ("black",)


def test_keep_ref_is_no_change_and_reuses_exact_preference(
    materializer: IntentMaterializer,
) -> None:
    old = _old_preference()
    current = _current(old, version=1)
    request = _request(current, turn=2)
    result = materializer.materialize(
        current=current,
        request=request,
        frame=_frame(
            current,
            keep=("active_0",),
            disposition=UnderstandingDisposition.NO_CHANGE,
        ),
    )

    assert result.update is None
    assert result.final_intent is current
    assert result.final_intent.preferences[0] is old


def test_dont_care_clears_active_facet(materializer: IntentMaterializer) -> None:
    current = _current(_old_preference(), version=1)
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=2),
        frame=_frame(current, dont_care=("color",)),
    )

    assert result.update is not None
    assert type(result.update.operations[0]) is SetDontCare
    assert result.final_intent.preferences == ()
    assert result.final_intent.dont_care_facets == frozenset({"color"})


def test_unknown_subfacet_dont_care_is_ignored_after_ref_omission(
    materializer: IntentMaterializer,
) -> None:
    waterproof = _old_preference(facet="feature", value="waterproof")
    low_heel = replace(
        _old_preference(facet="feature", value="low heel"),
        id="p_1_1_0",
        facet=None,
        operator=None,
        value=None,
        semantic_text="must have a low heel",
        semantic_polarity=SemanticPolarity.POSITIVE,
    )
    current = _current(waterproof, low_heel, version=1)
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=2),
        frame=_frame(
            current,
            keep=("active_0",),
            dont_care=("heel_height",),
        ),
    )

    assert result.final_intent.preferences == (waterproof,)
    assert result.final_intent.dont_care_facets == frozenset()
    assert result.ignored_dont_care_facets == ("heel_height",)


def test_dont_care_alias_sets_registered_marker(
    materializer: IntentMaterializer,
) -> None:
    material = _old_preference(facet="material", value="steel")
    current = _current(material, version=1)
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=2),
        frame=_frame(current, dont_care=("metal",)),
    )

    assert result.final_intent.preferences == ()
    assert result.final_intent.dont_care_facets == frozenset({"material"})
    assert result.ignored_dont_care_facets == ()


def test_kept_ref_wins_over_conflicting_dont_care_marker(
    materializer: IntentMaterializer,
) -> None:
    waterproof = _old_preference(facet="feature", value="waterproof")
    current = _current(waterproof, version=1)
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=2),
        frame=_frame(current, keep=("active_0",), dont_care=("feature",)),
    )

    assert result.final_intent is current
    assert result.ignored_dont_care_facets == ("feature",)


def test_unknown_facet_falls_back_to_semantic_only(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=1),
        frame=_frame(
            current,
            new=(
                _new_preference(
                    facet="vibe",
                    relation=PreferenceRelation.EQ,
                    values=("quiet luxury",),
                    meaning="quiet luxury vibe",
                ),
            ),
        ),
    )

    preference = result.final_intent.preferences[0]
    assert preference.facet is None
    assert preference.semantic_text == "quiet luxury vibe"
    assert result.semantic_fallback_facets == ("vibe",)


def test_non_price_numeric_range_falls_back_to_semantic_only(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=1, utterance="40 mm or smaller"),
        frame=_frame(
            current,
            new=(
                _new_preference(
                    facet="case_size",
                    relation=PreferenceRelation.LE,
                    values=("40 mm",),
                    meaning="watch case size must be 40 mm or smaller",
                ),
            ),
        ),
    )

    preference = result.final_intent.preferences[0]
    assert preference.facet is None
    assert preference.semantic_text == "watch case size must be 40 mm or smaller"
    assert result.semantic_fallback_facets == ("case_size",)


def test_revise_goal_removes_stale_wording_without_switching_product_task(
    materializer: IntentMaterializer,
) -> None:
    red = _old_preference(facet="color", value="red")
    size = replace(_old_preference(facet="size", value="7"), id="p_1_1_0")
    current = _current(red, size, version=1, goal="red leather heels")
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=2, utterance="Any color or material is fine."),
        frame=_frame(
            current,
            keep=("active_1",),
            goal=GoalFrame(action=GoalAction.REVISE, value="heels"),
        ),
    )

    assert result.update is not None
    assert type(result.update.operations[0]) is SwitchGoal
    assert result.final_intent.goal == "heels"
    assert result.final_intent.preferences == (size,)


def test_second_positive_categorical_requirement_falls_back_to_semantic(
    materializer: IntentMaterializer,
) -> None:
    old = _old_preference(facet="feature", value="lightweight")
    current = _current(old, version=1)
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=2),
        frame=_frame(
            current,
            keep=("active_0",),
            new=(
                _new_preference(
                    facet="feature",
                    relation=PreferenceRelation.EQ,
                    values=("leather sole",),
                    meaning="must have a leather sole",
                ),
            ),
        ),
    )

    assert result.final_intent.preferences[0] is old
    fallback = result.final_intent.preferences[1]
    assert fallback.facet is None
    assert fallback.semantic_text == "must have a leather sole"
    assert fallback.semantic_polarity is SemanticPolarity.POSITIVE
    assert result.semantic_fallback_facets == ("feature",)


def test_price_uses_usd_string_and_commits_integer_cents(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=1),
        frame=_frame(
            current,
            new=(
                _new_preference(
                    facet="price",
                    relation=PreferenceRelation.LE,
                    numeric_value_usd="99.95",
                    meaning="at most $99.95",
                ),
            ),
        ),
    )

    preference = result.final_intent.preferences[0]
    assert preference.facet == "price"
    assert preference.operator is Operator.LE
    assert preference.value == 9995
    assert type(preference.value) is int


def test_mixed_typed_preferences_materialize_in_canonical_group_order(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=1),
        frame=_frame(
            current,
            new=(
                _new_preference(
                    facet="material",
                    relation=PreferenceRelation.EQ,
                    values=("silk",),
                ),
                _new_preference(
                    facet="price",
                    relation=PreferenceRelation.LE,
                    numeric_value_usd="120",
                ),
                _new_preference(
                    facet=None,
                    relation=PreferenceRelation.EQ,
                    meaning="must not look touristy",
                    semantic_polarity=SemanticPolarity.NEGATIVE,
                ),
            ),
        ),
    )

    assert tuple(item.facet for item in result.final_intent.preferences) == (
        "material",
        "price",
        None,
    )
    assert result.final_intent.preferences[1].value == 12000
    assert result.final_intent.preferences[2].semantic_polarity is SemanticPolarity.NEGATIVE
    assert tuple(item.id for item in result.final_intent.preferences) == (
        "p_1_0_0",
        "p_1_1_0",
        "p_1_2_0",
    )


@pytest.mark.parametrize(
    ("preference", "expected_group"),
    [
        (
            _new_preference(
                facet="style",
                relation=PreferenceRelation.EQ,
                values=("minimal",),
                basis=PreferenceBasis.INFERRED,
                strength=PreferenceStrength.HARD,
            ),
            "structured",
        ),
        (
            _new_preference(
                facet="price",
                relation=PreferenceRelation.LE,
                numeric_value_usd="100",
                basis=PreferenceBasis.INFERRED,
                strength=PreferenceStrength.HARD,
            ),
            "price",
        ),
        (
            _new_preference(
                facet=None,
                relation=PreferenceRelation.EQ,
                basis=PreferenceBasis.INFERRED,
                strength=PreferenceStrength.HARD,
            ),
            "semantic",
        ),
    ],
)
def test_inferred_hard_condition_is_rejected_for_repair(
    materializer: IntentMaterializer,
    preference: NewPreferenceFrame,
    expected_group: str,
) -> None:
    current = _current()
    with pytest.raises(QueryUnderstandingError) as caught:
        materializer.materialize(
            current=current,
            request=_request(current, turn=1),
            frame=_frame(
                current,
                new=(preference,),
            ),
        )

    assert caught.value.code is QueryUnderstandingErrorCode.INVALID_PREFERENCE
    assert caught.value.path == ("new_preferences", expected_group)
    assert dict(caught.value.details) == {"reason": "inferred_hard"}


def test_feedback_resolves_only_model_safe_product_refs(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    shown = (ShownProductView(ref="product_0", product_ids=("asin-1",), label="Blue bag"),)
    result = materializer.materialize(
        current=current,
        request=_request(current, turn=1, shown=shown),
        frame=_frame(
            current,
            feedback=(
                FeedbackFrame(
                    target_refs=("product_0",),
                    signal=FeedbackSignal.REJECTED,
                    compared_to_refs=(),
                    evidence="not this one",
                ),
            ),
            disposition=UnderstandingDisposition.NO_CHANGE,
        ),
    )

    assert result.feedback[0].product_ids == ("asin-1",)
    assert result.feedback[0].signal is FeedbackSignal.REJECTED


class _ScriptedProvider:
    def __init__(self, results: tuple[ProviderResult, ...]) -> None:
        self.results = list(results)
        self.repairs: list[str | None] = []

    def reconcile(self, request, *, repair_instruction=None):
        self.repairs.append(repair_instruction)
        return self.results.pop(0)


def _provider_result(frame: ReconciledIntentFrame, response_id: str) -> ProviderResult:
    return ProviderResult(
        frame=frame,
        trace=ProviderTrace(
            response_id=response_id,
            model="deepseek-v4-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )


def test_service_repairs_once_without_applying_invalid_first_frame(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    invalid = _frame(
        current,
        new=(
            _new_preference(
                facet="style",
                relation=PreferenceRelation.EQ,
                values=("minimal",),
                basis=PreferenceBasis.INFERRED,
                strength=PreferenceStrength.HARD,
            ),
        ),
    )
    repaired = replace(
        invalid,
        structured_preferences=(
            replace(
                invalid.structured_preferences[0],
                strength=PreferenceStrength.SOFT,
            ),
        ),
    )
    provider = _ScriptedProvider(
        (_provider_result(invalid, "attempt-1"), _provider_result(repaired, "attempt-2"))
    )
    result = QueryUnderstandingService(
        provider=provider,
        materializer=materializer,
    ).resolve(current=current, request=_request(current, turn=1))

    assert provider.repairs[0] is None
    assert provider.repairs[1] is not None
    assert "reason=inferred_hard" in provider.repairs[1]
    assert "path=new_preferences.structured" in provider.repairs[1]
    assert "allowed_dont_care_facets=[" in provider.repairs[1]
    assert "use goal.revise" in provider.repairs[1]
    assert len(result.trace.attempts) == 2
    assert result.final_intent.preferences[0].commitment is Commitment.SOFT
    assert current.preferences == ()


def test_service_reports_safe_reason_when_repair_is_exhausted(
    materializer: IntentMaterializer,
) -> None:
    current = _current()
    invalid = _frame(
        current,
        new=(
            _new_preference(
                facet="style",
                relation=PreferenceRelation.EQ,
                values=("minimal",),
                basis=PreferenceBasis.INFERRED,
                strength=PreferenceStrength.HARD,
            ),
        ),
    )
    provider = _ScriptedProvider(
        (_provider_result(invalid, "attempt-1"), _provider_result(invalid, "attempt-2"))
    )

    with pytest.raises(QueryUnderstandingError) as caught:
        QueryUnderstandingService(
            provider=provider,
            materializer=materializer,
        ).resolve(current=current, request=_request(current, turn=1))

    assert caught.value.code is QueryUnderstandingErrorCode.REPAIR_EXHAUSTED
    assert dict(caught.value.details) == {
        "attempt_count": 2,
        "last_detail_reason": "inferred_hard",
        "last_error": "invalid_preference",
        "last_path": "new_preferences.structured",
    }
    assert "reason=inferred_hard" in provider.repairs[1]
