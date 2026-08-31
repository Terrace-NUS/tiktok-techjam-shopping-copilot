"""Trusted translation from a complete model frame to reducer operations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import NoReturn, Protocol, TypeAlias

from shopping_copilot.catalog.semantic.gateway import CatalogGatewayError
from shopping_copilot.catalog.semantic.runtime import (
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    ExtractedRuntimeValueCandidate,
    GroundingDisposition,
    RuntimeValueGroundingResult,
)
from shopping_copilot.facet_language import material_keywords
from shopping_copilot.session_context import (
    AddPreference,
    ClearFacet,
    Commitment,
    FacetAuthority,
    FacetKind,
    FacetRegistry,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    ProductFeedback,
    RemovePreference,
    ReplaceFacet,
    SemanticPolarity,
    SessionContextError,
    SetDontCare,
    StateUpdateBatch,
    SwitchGoal,
)
from shopping_copilot.session_context.models import PreferenceValue
from shopping_copilot.session_context.operations import StateOperation
from shopping_copilot.session_context.wide_facets import RETRIEVAL_DERIVED_FACET_IDS

from .errors import QueryUnderstandingError, QueryUnderstandingErrorCode
from .models import (
    CategoryOption,
    MaterializationResult,
    PreferenceBasis,
    PreferenceFrame,
    PreferenceRelation,
    PreferenceStrength,
    PricePreferenceFrame,
    ReconciledIntentFrame,
    ReconcileRequest,
    SemanticPreferenceFrame,
    ShownProductView,
    StructuredPreferenceFrame,
    UnderstandingDisposition,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreferenceCandidate:
    facet: str | None
    operator: Operator | None
    value: PreferenceValue | None
    semantic_text: str | None
    semantic_polarity: SemanticPolarity | None
    commitment: Commitment
    source: PreferenceSource
    source_turn: int
    evidence_text: str
    interpretation_confidence: float


_TargetPreference: TypeAlias = Preference | _PreferenceCandidate


class IntentPreviewer(Protocol):
    """Narrow Gateway seam needed by the pure materializer."""

    @property
    def registry(self) -> FacetRegistry: ...

    @property
    def release_id(self) -> str: ...

    def preview(
        self,
        current: IntentState,
        batch: StateUpdateBatch,
        *,
        catalog_semantic_release_id: str,
    ) -> IntentState: ...


class ValueGrounder(Protocol):
    """Narrow release-bound grounding seam used for catalog-verified facets."""

    def ground(
        self,
        candidate: ExtractedRuntimeValueCandidate,
        *,
        final_category_scope_id: str | None,
    ) -> RuntimeValueGroundingResult: ...


class IntentMaterializer:
    """Normalize, diff, allocate IDs, and Gateway-preview one model frame."""

    __slots__ = ("_gateway", "_grounder")

    def __init__(
        self,
        *,
        gateway: IntentPreviewer,
        grounder: ValueGrounder,
    ) -> None:
        self._gateway = gateway
        self._grounder = grounder

    def materialize(
        self,
        *,
        current: IntentState,
        request: ReconcileRequest,
        frame: ReconciledIntentFrame,
    ) -> MaterializationResult:
        """Return an immutable preview; never write a store or SessionContext."""

        self._validate_versions(current=current, request=request, frame=frame)
        active_by_ref = self._active_by_ref(current=current, request=request)
        kept = self._resolve_kept(frame.keep_active_refs, active_by_ref=active_by_ref)
        category_frames = tuple(
            item for item in frame.structured_preferences if _facet_alias(item.facet) == "category"
        )
        if len(category_frames) > 1:
            _fail(
                QueryUnderstandingErrorCode.INVALID_FINAL_STATE,
                path=("new_preferences", "structured"),
            )
        target: list[_TargetPreference] = list(kept)
        category_candidates = tuple(
            self._category_candidate(
                item,
                request=request,
            )
            for item in category_frames
        )
        target.extend(category_candidates)
        category_target = tuple(
            item for item in target if item.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID
        )
        if len(category_target) > 1:
            _fail(
                QueryUnderstandingErrorCode.INVALID_FINAL_STATE,
                path=("new_preferences", "structured"),
            )
        final_scope_id = self._final_scope_id(
            category_target,
            options=request.category_options,
        )

        fallback_facets: list[str] = []
        for structured_item in frame.structured_preferences:
            if _facet_alias(structured_item.facet) == "category":
                continue
            candidates, fallback_facet = self._structured_preference_candidates(
                structured_item,
                turn=request.turn,
                final_scope_id=final_scope_id,
            )
            candidates, conjunction_fallback = self._fallback_categorical_conjunction(
                structured_item,
                candidates=candidates,
                target=target,
                turn=request.turn,
            )
            target.extend(candidates)
            if fallback_facet is not None or conjunction_fallback:
                fallback_facets.append(fallback_facet or structured_item.facet)

        for price_item in frame.price_preferences:
            candidates, fallback_facet = self._price_preference_candidates(
                price_item,
                turn=request.turn,
                final_scope_id=final_scope_id,
            )
            target.extend(candidates)
            if fallback_facet is not None:
                fallback_facets.append(fallback_facet)

        target.extend(
            self._semantic_candidate(
                semantic_item,
                turn=request.turn,
                polarity=semantic_item.polarity,
            )
            for semantic_item in frame.semantic_preferences
        )

        target = self._reuse_existing(target, current=current)
        _require_unique_semantics(target)
        desired_dont_care, ignored_dont_care = self._normalize_dont_care(
            frame.dont_care_facets,
            allowed=request.allowed_dont_care_facets,
        )
        target_facets = {item.facet for item in target if item.facet is not None}
        conflicting_dont_care = desired_dont_care.intersection(target_facets)
        if conflicting_dont_care:
            desired_dont_care = desired_dont_care.difference(conflicting_dont_care)
            ignored_dont_care = tuple(sorted((*ignored_dont_care, *conflicting_dont_care)))

        goal = current.goal if frame.goal.action.value == "keep" else frame.goal.value
        assert goal is None or type(goal) is str
        goal_changed = goal != current.goal
        goal_switched = goal_changed and frame.goal.action.value == "switch"
        operations: list[StateOperation] = []
        old_target = tuple(item for item in target if type(item) is Preference)
        if goal_changed:
            assert goal is not None
            operations.append(
                SwitchGoal(
                    new_goal=goal,
                    carry_preference_ids=tuple(sorted(item.id for item in old_target)),
                )
            )
            baseline_preferences = old_target
            baseline_dont_care = frozenset[str]()
        else:
            baseline_preferences = current.preferences
            baseline_dont_care = current.dont_care_facets

        target = self._represent_category_broadening(
            target,
            baseline_preferences=baseline_preferences,
            request=request,
            goal_switched=goal_switched,
        )
        self._append_structured_operations(
            operations,
            baseline_preferences=baseline_preferences,
            baseline_dont_care=baseline_dont_care,
            target=tuple(target),
            desired_dont_care=desired_dont_care,
            turn=request.turn,
        )
        self._append_semantic_operations(
            operations,
            baseline_preferences=baseline_preferences,
            target=tuple(target),
            turn=request.turn,
        )

        feedback = self._materialize_feedback(frame=frame, request=request)
        if not operations:
            update = None
            final_intent = current
        else:
            update = StateUpdateBatch(
                turn=request.turn,
                base_intent_version=current.version,
                operations=tuple(operations),
            )
            try:
                final_intent = self._gateway.preview(
                    current,
                    update,
                    catalog_semantic_release_id=self._gateway.release_id,
                )
            except (CatalogGatewayError, SessionContextError, TypeError, ValueError) as error:
                raise QueryUnderstandingError(
                    code=QueryUnderstandingErrorCode.PREVIEW_REJECTED,
                    details=(("reason", _safe_reason(error)),),
                ) from error
        if frame.disposition is UnderstandingDisposition.NO_CHANGE and update is not None:
            _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE, path=("disposition",))
        return MaterializationResult(
            update=update,
            final_intent=final_intent,
            feedback=feedback,
            directives=frame.directives,
            clarification=frame.clarification,
            semantic_fallback_facets=tuple(sorted(set(fallback_facets))),
            ignored_dont_care_facets=ignored_dont_care,
        )

    @staticmethod
    def _validate_versions(
        *,
        current: IntentState,
        request: ReconcileRequest,
        frame: ReconciledIntentFrame,
    ) -> None:
        if request.base_intent_version != current.version:
            _fail(
                QueryUnderstandingErrorCode.STALE_INTENT_VERSION,
                details=(
                    ("actual", request.base_intent_version),
                    ("expected", current.version),
                ),
            )
        if frame.base_intent_version != current.version:
            _fail(
                QueryUnderstandingErrorCode.STALE_INTENT_VERSION,
                path=("base_intent_version",),
                details=(
                    ("actual", frame.base_intent_version),
                    ("expected", current.version),
                ),
            )

    @staticmethod
    def _active_by_ref(
        *,
        current: IntentState,
        request: ReconcileRequest,
    ) -> dict[str, Preference]:
        expected_refs = tuple(f"active_{index}" for index in range(len(current.preferences)))
        actual_refs = tuple(item.ref for item in request.active_preferences)
        if actual_refs != expected_refs or len(actual_refs) != len(current.preferences):
            _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE)
        return {
            view.ref: preference
            for view, preference in zip(
                request.active_preferences,
                current.preferences,
                strict=True,
            )
        }

    @staticmethod
    def _resolve_kept(
        refs: tuple[str, ...],
        *,
        active_by_ref: dict[str, Preference],
    ) -> tuple[Preference, ...]:
        kept: list[Preference] = []
        for index, ref in enumerate(refs):
            preference = active_by_ref.get(ref)
            if preference is None:
                _fail(
                    QueryUnderstandingErrorCode.UNKNOWN_ACTIVE_REF,
                    path=("keep_active_refs", index),
                )
            kept.append(preference)
        return tuple(kept)

    def _category_candidate(
        self,
        frame: StructuredPreferenceFrame,
        *,
        request: ReconcileRequest,
    ) -> _PreferenceCandidate:
        if frame.relation is not PreferenceRelation.EQ or len(frame.values) != 1:
            _fail(
                QueryUnderstandingErrorCode.INVALID_PREFERENCE,
                path=("new_preferences", "structured"),
                details=(("reason", "category_requires_single_eq"),),
            )
        by_ref = {option.ref: option for option in request.category_options}
        option = by_ref.get(frame.values[0])
        if option is None:
            _fail(
                QueryUnderstandingErrorCode.UNKNOWN_CATEGORY_REF,
                path=("new_preferences", "structured"),
            )
        return self._structured_candidate(
            frame,
            turn=request.turn,
            facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            operator=Operator.EQ,
            value=option.scope_id,
        )

    def _structured_preference_candidates(
        self,
        frame: StructuredPreferenceFrame,
        *,
        turn: int,
        final_scope_id: str,
    ) -> tuple[tuple[_PreferenceCandidate, ...], str | None]:
        self._validate_source_strength(frame)
        facet = frame.facet
        if frame.relation in (
            PreferenceRelation.LT,
            PreferenceRelation.LE,
            PreferenceRelation.GT,
            PreferenceRelation.GE,
        ):
            return (
                self._semantic_candidate(
                    frame,
                    turn=turn,
                    polarity=SemanticPolarity.POSITIVE,
                ),
            ), facet
        spec = self._gateway.registry.get(facet)
        if spec is None:
            return (
                self._semantic_candidate(
                    frame,
                    turn=turn,
                    polarity=_polarity(frame.relation),
                ),
            ), facet
        operator = Operator(frame.relation.value)
        if spec.authority is FacetAuthority.RETRIEVAL_DERIVED:
            if facet == "material":
                operator, raw_value = _material_keyword_condition(operator, frame.values)
            else:
                raw_value = (
                    frame.values[0]
                    if operator in (Operator.EQ, Operator.NEQ)
                    else tuple(frame.values)
                )
            try:
                normalized = self._gateway.registry.normalize_value(
                    facet,
                    operator,
                    raw_value,
                )
            except SessionContextError as error:
                raise QueryUnderstandingError(
                    code=QueryUnderstandingErrorCode.INVALID_PREFERENCE,
                ) from error
            return (
                self._structured_candidate(
                    frame,
                    turn=turn,
                    facet=facet,
                    operator=operator,
                    value=normalized,
                ),
            ), None

        raw_value = (
            frame.values[0] if operator in (Operator.EQ, Operator.NEQ) else tuple(frame.values)
        )
        try:
            grounded = self._grounder.ground(
                ExtractedRuntimeValueCandidate(
                    facet_id=facet,
                    operator=operator,
                    value=raw_value,
                    alternative_values=(),
                    semantic_text=frame.meaning,
                    semantic_polarity=_polarity(frame.relation),
                ),
                final_category_scope_id=final_scope_id,
            )
        except (TypeError, ValueError) as error:
            raise QueryUnderstandingError(
                code=QueryUnderstandingErrorCode.INVALID_PREFERENCE,
            ) from error
        if grounded.disposition is GroundingDisposition.GROUNDED:
            return tuple(
                self._structured_candidate(
                    frame,
                    turn=turn,
                    facet=predicate.facet_id,
                    operator=predicate.operator,
                    value=predicate.value,
                )
                for predicate in grounded.predicates
            ), None
        if grounded.disposition is GroundingDisposition.SEMANTIC_ONLY:
            return (
                self._semantic_candidate(
                    frame,
                    turn=turn,
                    polarity=_polarity(frame.relation),
                ),
            ), facet
        _fail(QueryUnderstandingErrorCode.INVALID_PREFERENCE)

    def _price_preference_candidates(
        self,
        frame: PricePreferenceFrame,
        *,
        turn: int,
        final_scope_id: str,
    ) -> tuple[tuple[_PreferenceCandidate, ...], str | None]:
        self._validate_source_strength(frame)
        operator = Operator(frame.relation.value)
        raw_value = _usd_to_cents(
            frame.value_usd,
            path=("new_preferences", "price"),
        )
        try:
            grounded = self._grounder.ground(
                ExtractedRuntimeValueCandidate(
                    facet_id="price",
                    operator=operator,
                    value=raw_value,
                    alternative_values=(),
                    semantic_text=frame.meaning,
                    semantic_polarity=SemanticPolarity.POSITIVE,
                ),
                final_category_scope_id=final_scope_id,
            )
        except (TypeError, ValueError) as error:
            raise QueryUnderstandingError(
                code=QueryUnderstandingErrorCode.INVALID_PREFERENCE,
                path=("new_preferences", "price"),
            ) from error
        if grounded.disposition is GroundingDisposition.GROUNDED:
            return tuple(
                self._structured_candidate(
                    frame,
                    turn=turn,
                    facet=predicate.facet_id,
                    operator=predicate.operator,
                    value=predicate.value,
                )
                for predicate in grounded.predicates
            ), None
        if grounded.disposition is GroundingDisposition.SEMANTIC_ONLY:
            return (
                self._semantic_candidate(
                    frame,
                    turn=turn,
                    polarity=SemanticPolarity.POSITIVE,
                ),
            ), "price"
        _fail(
            QueryUnderstandingErrorCode.INVALID_PREFERENCE,
            path=("new_preferences", "price"),
        )

    def _fallback_categorical_conjunction(
        self,
        frame: StructuredPreferenceFrame,
        *,
        candidates: tuple[_PreferenceCandidate, ...],
        target: list[_TargetPreference],
        turn: int,
    ) -> tuple[tuple[_PreferenceCandidate, ...], bool]:
        accepted: list[_PreferenceCandidate] = []
        needs_semantic_fallback = False
        for candidate in candidates:
            existing = (*target, *accepted)
            if any(_signature(item) == _signature(candidate) for item in existing):
                continue
            if self._categorical_conjunction_conflicts(candidate, existing=existing):
                needs_semantic_fallback = True
                continue
            accepted.append(candidate)
        if needs_semantic_fallback:
            accepted.append(
                self._semantic_candidate(
                    frame,
                    turn=turn,
                    polarity=_polarity(frame.relation),
                )
            )
        return tuple(accepted), needs_semantic_fallback

    def _categorical_conjunction_conflicts(
        self,
        candidate: _PreferenceCandidate,
        *,
        existing: tuple[_TargetPreference, ...],
    ) -> bool:
        facet = candidate.facet
        if facet is None or candidate.operator not in (Operator.EQ, Operator.IN):
            return False
        spec = self._gateway.registry.get(facet)
        if spec is None or spec.kind is not FacetKind.CATEGORICAL:
            return False
        positives = tuple(
            item
            for item in existing
            if item.facet == facet and item.operator in (Operator.EQ, Operator.IN)
        )
        if not positives:
            return False
        if any(item.commitment is candidate.commitment for item in positives):
            return True
        common = set(_categorical_selector_values(candidate))
        for item in positives:
            common.intersection_update(_categorical_selector_values(item))
        return not common

    @staticmethod
    def _validate_source_strength(frame: PreferenceFrame) -> None:
        if frame.basis is PreferenceBasis.INFERRED and frame.strength is PreferenceStrength.HARD:
            _fail(
                QueryUnderstandingErrorCode.INVALID_PREFERENCE,
                path=_preference_group_path(frame),
                details=(("reason", "inferred_hard"),),
            )

    def _structured_candidate(
        self,
        frame: PreferenceFrame,
        *,
        turn: int,
        facet: str,
        operator: Operator,
        value: PreferenceValue,
    ) -> _PreferenceCandidate:
        self._validate_source_strength(frame)
        hybrid = facet in RETRIEVAL_DERIVED_FACET_IDS
        return _PreferenceCandidate(
            facet=facet,
            operator=operator,
            value=value,
            semantic_text=frame.meaning if hybrid else None,
            semantic_polarity=_operator_polarity(operator) if hybrid else None,
            commitment=_commitment(frame.strength),
            source=_source(frame.basis),
            source_turn=turn,
            evidence_text=frame.evidence,
            interpretation_confidence=float(frame.confidence),
        )

    @staticmethod
    def _semantic_candidate(
        frame: PreferenceFrame,
        *,
        turn: int,
        polarity: SemanticPolarity,
    ) -> _PreferenceCandidate:
        IntentMaterializer._validate_source_strength(frame)
        return _PreferenceCandidate(
            facet=None,
            operator=None,
            value=None,
            semantic_text=frame.meaning,
            semantic_polarity=polarity,
            commitment=_commitment(frame.strength),
            source=_source(frame.basis),
            source_turn=turn,
            evidence_text=frame.evidence,
            interpretation_confidence=float(frame.confidence),
        )

    @staticmethod
    def _reuse_existing(
        target: list[_TargetPreference],
        *,
        current: IntentState,
    ) -> list[_TargetPreference]:
        by_signature = {_signature(item): item for item in current.preferences}
        return [
            by_signature.get(_signature(item), item) if type(item) is _PreferenceCandidate else item
            for item in target
        ]

    def _normalize_dont_care(
        self,
        facets: tuple[str, ...],
        *,
        allowed: tuple[str, ...],
    ) -> tuple[frozenset[str], tuple[str, ...]]:
        normalized: set[str] = set()
        ignored: set[str] = set()
        allowed_set = set(allowed)
        for facet in facets:
            candidate = "_".join(facet.strip().casefold().replace("-", " ").split())
            canonical = _DONT_CARE_ALIASES.get(candidate, candidate)
            if (
                canonical == "category"
                or canonical not in allowed_set
                or self._gateway.registry.get(canonical) is None
            ):
                ignored.add(canonical)
                continue
            normalized.add(canonical)
        return frozenset(normalized), tuple(sorted(ignored))

    @staticmethod
    def _final_scope_id(
        category_target: tuple[_TargetPreference, ...],
        *,
        options: tuple[CategoryOption, ...],
    ) -> str:
        if category_target:
            value = category_target[0].value
            if type(value) is not str:
                _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE)
            return value
        roots = tuple(option for option in options if option.is_root)
        if len(roots) != 1:
            _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE)
        return roots[0].scope_id

    def _represent_category_broadening(
        self,
        target: list[_TargetPreference],
        *,
        baseline_preferences: tuple[Preference, ...],
        request: ReconcileRequest,
        goal_switched: bool,
    ) -> list[_TargetPreference]:
        if goal_switched or any(item.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID for item in target):
            return target
        baseline_categories = tuple(
            item for item in baseline_preferences if item.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID
        )
        if not baseline_categories:
            return target
        roots = tuple(option for option in request.category_options if option.is_root)
        if len(roots) != 1:
            _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE)
        if baseline_categories[0].value == roots[0].scope_id:
            target.append(baseline_categories[0])
            return target
        target.append(
            _PreferenceCandidate(
                facet=SYSTEM_PRODUCT_CATEGORY_FACET_ID,
                operator=Operator.EQ,
                value=roots[0].scope_id,
                semantic_text=None,
                semantic_polarity=None,
                commitment=Commitment.HARD,
                source=PreferenceSource.USER_EXPLICIT,
                source_turn=request.turn,
                evidence_text=request.latest_utterance,
                interpretation_confidence=1.0,
            )
        )
        return target

    def _append_structured_operations(
        self,
        operations: list[StateOperation],
        *,
        baseline_preferences: tuple[Preference, ...],
        baseline_dont_care: frozenset[str],
        target: tuple[_TargetPreference, ...],
        desired_dont_care: frozenset[str],
        turn: int,
    ) -> None:
        baseline_by_facet = _structured_by_facet(baseline_preferences)
        target_by_facet = _structured_by_facet(target)
        facets = sorted(
            set(baseline_by_facet)
            | set(target_by_facet)
            | set(baseline_dont_care)
            | set(desired_dont_care)
        )
        if SYSTEM_PRODUCT_CATEGORY_FACET_ID in facets:
            facets.remove(SYSTEM_PRODUCT_CATEGORY_FACET_ID)
            facets.insert(0, SYSTEM_PRODUCT_CATEGORY_FACET_ID)
        for facet in facets:
            baseline = baseline_by_facet.get(facet, ())
            desired = target_by_facet.get(facet, ())
            if facet in desired_dont_care:
                if facet not in baseline_dont_care or baseline:
                    operations.append(SetDontCare(facet=facet))
                continue
            if desired:
                if {_signature(item) for item in desired} != {
                    _signature(item) for item in baseline
                } or facet in baseline_dont_care:
                    operation_index = len(operations)
                    operations.append(
                        ReplaceFacet(
                            facet=facet,
                            preferences=_materialize_replacement(
                                desired,
                                turn=turn,
                                operation_index=operation_index,
                            ),
                        )
                    )
                continue
            if baseline or facet in baseline_dont_care:
                if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                    _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE)
                operations.append(ClearFacet(facet=facet))

    @staticmethod
    def _append_semantic_operations(
        operations: list[StateOperation],
        *,
        baseline_preferences: tuple[Preference, ...],
        target: tuple[_TargetPreference, ...],
        turn: int,
    ) -> None:
        baseline = tuple(item for item in baseline_preferences if item.facet is None)
        desired = tuple(item for item in target if item.facet is None)
        desired_old_ids = {item.id for item in desired if type(item) is Preference}
        removed = tuple(sorted(item.id for item in baseline if item.id not in desired_old_ids))
        if removed:
            operations.append(RemovePreference(preference_ids=removed))
        for item in desired:
            if type(item) is Preference:
                continue
            assert type(item) is _PreferenceCandidate
            operation_index = len(operations)
            operations.append(
                AddPreference(
                    preference=_to_preference(
                        item,
                        preference_id=f"p_{turn}_{operation_index}_0",
                    )
                )
            )

    @staticmethod
    def _materialize_feedback(
        *,
        frame: ReconciledIntentFrame,
        request: ReconcileRequest,
    ) -> tuple[ProductFeedback, ...]:
        by_ref = {item.ref: item for item in request.shown_products}
        results: list[ProductFeedback] = []
        for frame_index, item in enumerate(frame.feedback):
            target_ids = _product_ids_for_refs(
                item.target_refs,
                by_ref=by_ref,
                path=("feedback", frame_index, "target_refs"),
            )
            compared_ids = _product_ids_for_refs(
                item.compared_to_refs,
                by_ref=by_ref,
                path=("feedback", frame_index, "compared_to_refs"),
            )
            results.append(
                ProductFeedback(
                    product_ids=target_ids,
                    signal=item.signal,
                    compared_to_ids=compared_ids,
                    evidence_text=item.evidence,
                )
            )
        return tuple(results)


def _structured_by_facet(
    preferences: tuple[_TargetPreference, ...],
) -> dict[str, tuple[_TargetPreference, ...]]:
    grouped: dict[str, list[_TargetPreference]] = {}
    for item in preferences:
        if item.facet is not None:
            grouped.setdefault(item.facet, []).append(item)
    return {facet: tuple(items) for facet, items in grouped.items()}


def _materialize_replacement(
    items: tuple[_TargetPreference, ...],
    *,
    turn: int,
    operation_index: int,
) -> tuple[Preference, ...]:
    old = tuple(sorted((item for item in items if type(item) is Preference), key=lambda x: x.id))
    new = tuple(item for item in items if type(item) is _PreferenceCandidate)
    materialized: list[Preference] = list(old)
    for preference_index, item in enumerate(new, start=len(old)):
        materialized.append(
            _to_preference(
                item,
                preference_id=f"p_{turn}_{operation_index}_{preference_index}",
            )
        )
    return tuple(materialized)


def _to_preference(
    candidate: _PreferenceCandidate,
    *,
    preference_id: str,
) -> Preference:
    return Preference(
        id=preference_id,
        facet=candidate.facet,
        operator=candidate.operator,
        value=candidate.value,
        semantic_text=candidate.semantic_text,
        semantic_polarity=candidate.semantic_polarity,
        commitment=candidate.commitment,
        source=candidate.source,
        source_turn=candidate.source_turn,
        evidence_text=candidate.evidence_text,
        interpretation_confidence=candidate.interpretation_confidence,
    )


def _signature(item: _TargetPreference) -> object:
    value = item.value
    typed_value = (
        tuple((type(member), member) for member in value)
        if type(value) is tuple
        else (type(value), value)
    )
    return (
        item.facet,
        item.operator,
        typed_value,
        item.semantic_text,
        item.semantic_polarity,
        item.commitment,
        item.source,
        (type(item.interpretation_confidence), item.interpretation_confidence),
    )


def _categorical_selector_values(item: _TargetPreference) -> tuple[object, ...]:
    value = item.value
    return value if type(value) is tuple else (value,)


def _require_unique_semantics(target: list[_TargetPreference]) -> None:
    signatures = [_signature(item) for item in target]
    if len(set(signatures)) != len(signatures):
        _fail(QueryUnderstandingErrorCode.INVALID_FINAL_STATE)


def _product_ids_for_refs(
    refs: tuple[str, ...],
    *,
    by_ref: dict[str, ShownProductView],
    path: tuple[str | int, ...],
) -> tuple[str, ...]:
    result: list[str] = []
    for index, ref in enumerate(refs):
        view = by_ref.get(ref)
        if view is None:
            _fail(QueryUnderstandingErrorCode.UNKNOWN_PRODUCT_REF, path=path + (index,))
        result.extend(view.product_ids)
    return tuple(dict.fromkeys(result))


def _usd_to_cents(
    value: str,
    *,
    path: tuple[str | int, ...],
) -> int:
    try:
        dollars = Decimal(value)
    except InvalidOperation as error:
        raise QueryUnderstandingError(
            code=QueryUnderstandingErrorCode.INVALID_PREFERENCE,
            path=path,
            details=(("reason", "invalid_usd"),),
        ) from error
    cents = dollars * 100
    if not dollars.is_finite() or dollars < 0 or cents != cents.to_integral_value():
        _fail(
            QueryUnderstandingErrorCode.INVALID_PREFERENCE,
            path=path,
            details=(("reason", "invalid_usd"),),
        )
    return int(cents)


def _preference_group_path(frame: PreferenceFrame) -> tuple[str, str]:
    if isinstance(frame, StructuredPreferenceFrame):
        return ("new_preferences", "structured")
    if isinstance(frame, PricePreferenceFrame):
        return ("new_preferences", "price")
    if isinstance(frame, SemanticPreferenceFrame):
        return ("new_preferences", "semantic")
    raise AssertionError(f"unhandled preference frame: {type(frame).__name__}")


def _facet_alias(facet: str | None) -> str | None:
    return "category" if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID else facet


def _material_keyword_condition(
    operator: Operator,
    values: tuple[str, ...],
) -> tuple[Operator, PreferenceValue]:
    anchors = tuple(
        dict.fromkeys(anchor for value in values for anchor in material_keywords(value))
    )
    if len(anchors) == 1 and operator in (Operator.EQ, Operator.NEQ):
        return operator, anchors[0]
    if operator in (Operator.EQ, Operator.IN):
        return Operator.IN, anchors
    return Operator.NOT_IN, anchors


_DONT_CARE_ALIASES = {
    "budget": "price",
    "colour": "color",
    "metal": "material",
}


def _source(basis: PreferenceBasis) -> PreferenceSource:
    return (
        PreferenceSource.USER_EXPLICIT
        if basis is PreferenceBasis.EXPLICIT
        else PreferenceSource.SYSTEM_INFERRED
    )


def _commitment(strength: PreferenceStrength) -> Commitment:
    return Commitment.HARD if strength is PreferenceStrength.HARD else Commitment.SOFT


def _polarity(relation: PreferenceRelation) -> SemanticPolarity:
    return (
        SemanticPolarity.NEGATIVE
        if relation
        in (
            PreferenceRelation.NEQ,
            PreferenceRelation.NOT_IN,
        )
        else SemanticPolarity.POSITIVE
    )


def _operator_polarity(operator: Operator) -> SemanticPolarity:
    return (
        SemanticPolarity.NEGATIVE
        if operator in (Operator.NEQ, Operator.NOT_IN)
        else SemanticPolarity.POSITIVE
    )


def _safe_reason(error: BaseException) -> str:
    code = getattr(error, "code", None)
    value = getattr(code, "value", None)
    return value if type(value) is str else type(error).__name__


def _fail(
    code: QueryUnderstandingErrorCode,
    *,
    path: tuple[str | int, ...] = (),
    details: tuple[tuple[str, str | int | float | bool], ...] = (),
) -> NoReturn:
    raise QueryUnderstandingError(code=code, path=path, details=details)
