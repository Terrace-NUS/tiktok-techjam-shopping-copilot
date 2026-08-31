"""Deterministic hard-constraint resolution before retrieval truncation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from shopping_copilot.catalog.semantic.category import CategoryMatchResult, match_category
from shopping_copilot.catalog.semantic.facet import (
    USD_CENT_UNIT,
    NumericValue,
    match_numeric_interval,
    safe_filter_keeps,
)
from shopping_copilot.catalog.semantic.release import VerifiedCatalogSemanticRelease
from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
from shopping_copilot.query_compiler import (
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
)
from shopping_copilot.session_context import Operator
from shopping_copilot.session_context.models import ScalarValue

from .dense import DenseEligibilityMask, DenseIndex
from .errors import CompiledQueryBindingError
from .evidence import (
    RETRIEVAL_EVIDENCE_POLICY_ID,
    RETRIEVAL_EVIDENCE_PRODUCT_FACT_POLICY_ID,
    RETRIEVAL_EVIDENCE_PRODUCT_FACT_REPLACEMENT_POLICY_ID,
    RetrievalEvidenceIndex,
)

_SUPPORTED_EVIDENCE_POLICIES = frozenset(
    {
        RETRIEVAL_EVIDENCE_POLICY_ID,
        RETRIEVAL_EVIDENCE_PRODUCT_FACT_POLICY_ID,
        RETRIEVAL_EVIDENCE_PRODUCT_FACT_REPLACEMENT_POLICY_ID,
    }
)

_EXCLUSION_OPERATORS = frozenset({Operator.NEQ, Operator.NOT_IN})
_INCLUSION_OPERATORS = frozenset(
    {Operator.EQ, Operator.IN, Operator.LT, Operator.LE, Operator.GT, Operator.GE}
)
_PRICE_OPERATORS = frozenset({Operator.LT, Operator.LE, Operator.GT, Operator.GE})


class ConstraintDisposition(str, Enum):
    """What the resolver did with one compiled hard constraint."""

    APPLIED = "applied"
    RELAXED_TO_RANKING = "relaxed_to_ranking"
    SKIPPED_EMPTY_UPSTREAM = "skipped_empty_upstream"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstraintResolutionTrace:
    """One count-conserving explanation in actual constraint execution order."""

    preference_id: str
    facet: str
    operator: Operator
    before_count: int
    matched_count: int
    after_count: int
    disposition: ConstraintDisposition
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedHardMask:
    """One dense-index-bound mask and its asymmetric relaxation audit."""

    eligible_mask: DenseEligibilityMask
    eligible_parent_asins: tuple[str, ...]
    hard_filter_relaxed: bool
    relaxed_constraints: tuple[CompiledHardConstraint, ...]
    trace: tuple[ConstraintResolutionTrace, ...]


class HardMaskResolutionError(ValueError):
    """A compiled constraint cannot be safely resolved against bound evidence."""


class HardMaskResolver:
    """Resolve compiled exclusions first, then fail-soft includes."""

    __slots__ = (
        "_assignments",
        "_dense_index",
        "_evidence_index",
        "_parent_asins",
        "_price_by_parent",
        "_release",
        "_scopes",
    )

    def __init__(
        self,
        *,
        release: VerifiedCatalogSemanticRelease,
        evidence_index: RetrievalEvidenceIndex,
        dense_index: DenseIndex,
    ) -> None:
        if type(release) is not VerifiedCatalogSemanticRelease:
            raise TypeError("release must be a verified Catalog Semantic release")
        if type(evidence_index) is not RetrievalEvidenceIndex:
            raise TypeError("evidence_index must be a RetrievalEvidenceIndex")
        if type(dense_index) is not DenseIndex:
            raise TypeError("dense_index must be an exact DenseIndex")
        self._validate_bindings(release, evidence_index, dense_index)

        self._release = release
        self._evidence_index = evidence_index
        self._dense_index = dense_index
        self._parent_asins = dense_index.parent_asins
        self._assignments = {
            item.parent_asin: item for item in release.product_category_assignments.assignments
        }
        self._scopes = {item.id: item for item in release.category_registry.scopes}
        self._price_by_parent = {
            item.parent_asin: item
            for item in release.product_facet_index.entries
            if item.facet_id == "price"
        }

    def resolve(self, query: CompiledQuery) -> ResolvedHardMask:
        """Return a bound mask without ever relaxing a user exclusion."""

        self._validate_query(query)
        exclusions = tuple(
            item for item in query.hard_constraints if item.operator in _EXCLUSION_OPERATORS
        )
        inclusions = tuple(
            item for item in query.hard_constraints if item.operator not in _EXCLUSION_OPERATORS
        )

        eligible = set(self._parent_asins)
        trace: list[ConstraintResolutionTrace] = []
        relaxed: list[CompiledHardConstraint] = []

        for constraint in exclusions:
            self._validate_constraint(constraint, exclusion=True)
            before = len(eligible)
            matched = eligible.intersection(self._matching_products(constraint))
            eligible.difference_update(matched)
            trace.append(
                ConstraintResolutionTrace(
                    preference_id=constraint.preference_id,
                    facet=constraint.facet,
                    operator=constraint.operator,
                    before_count=before,
                    matched_count=len(matched),
                    after_count=len(eligible),
                    disposition=ConstraintDisposition.APPLIED,
                    reason="non_relaxable_exclusion",
                )
            )

        exclusions_emptied = not eligible
        for constraint in inclusions:
            self._validate_constraint(constraint, exclusion=False)
            before = len(eligible)
            if exclusions_emptied:
                trace.append(
                    ConstraintResolutionTrace(
                        preference_id=constraint.preference_id,
                        facet=constraint.facet,
                        operator=constraint.operator,
                        before_count=0,
                        matched_count=0,
                        after_count=0,
                        disposition=ConstraintDisposition.SKIPPED_EMPTY_UPSTREAM,
                        reason="non_relaxable_exclusions_emptied_catalog",
                    )
                )
                continue

            matched = eligible.intersection(self._matching_products(constraint))
            if not matched:
                relaxed.append(constraint)
                trace.append(
                    ConstraintResolutionTrace(
                        preference_id=constraint.preference_id,
                        facet=constraint.facet,
                        operator=constraint.operator,
                        before_count=before,
                        matched_count=0,
                        after_count=before,
                        disposition=ConstraintDisposition.RELAXED_TO_RANKING,
                        reason="include_would_empty_eligible_pool",
                    )
                )
                continue

            eligible = matched
            trace.append(
                ConstraintResolutionTrace(
                    preference_id=constraint.preference_id,
                    facet=constraint.facet,
                    operator=constraint.operator,
                    before_count=before,
                    matched_count=len(matched),
                    after_count=len(eligible),
                    disposition=ConstraintDisposition.APPLIED,
                    reason="include_intersection_nonempty",
                )
            )

        eligible_parent_asins = tuple(sorted(eligible))
        return ResolvedHardMask(
            eligible_mask=self._dense_index.make_eligibility_mask(eligible_parent_asins),
            eligible_parent_asins=eligible_parent_asins,
            hard_filter_relaxed=bool(relaxed),
            relaxed_constraints=tuple(relaxed),
            trace=tuple(trace),
        )

    @staticmethod
    def _validate_bindings(
        release: VerifiedCatalogSemanticRelease,
        evidence: RetrievalEvidenceIndex,
        dense: DenseIndex,
    ) -> None:
        catalog_id = release.manifest.catalog_id
        release_id = release.release_id
        if (
            release.category_registry.catalog_id != catalog_id
            or release.product_category_assignments.catalog_id != catalog_id
            or release.product_facet_index.catalog_id != catalog_id
            or evidence.catalog_id != catalog_id
            or dense.manifest.catalog_id != catalog_id
        ):
            raise CompiledQueryBindingError("hard-mask artifacts use different catalogs")
        if (
            evidence.catalog_semantic_release_id != release_id
            or dense.manifest.catalog_semantic_release_id != release_id
        ):
            raise CompiledQueryBindingError("hard-mask artifacts use different semantic releases")
        if evidence.policy_id not in _SUPPORTED_EVIDENCE_POLICIES:
            raise CompiledQueryBindingError("hard-mask evidence uses an unsupported policy")

        release_products = tuple(
            item.parent_asin for item in release.product_category_assignments.assignments
        )
        if release_products != dense.parent_asins or evidence.parent_asins != dense.parent_asins:
            raise CompiledQueryBindingError("hard-mask artifacts contain different product sets")
        product_set = set(dense.parent_asins)
        if any(item.parent_asin not in product_set for item in release.product_facet_index.entries):
            raise CompiledQueryBindingError("price evidence references a different product set")

    def _validate_query(self, query: CompiledQuery) -> None:
        if type(query) is not CompiledQuery:
            raise TypeError("query must be an exact CompiledQuery")
        if (
            query.catalog_id != self._release.manifest.catalog_id
            or query.catalog_semantic_release_id != self._release.release_id
            or query.category_graph_id != self._release.category_registry.category_graph_id
        ):
            raise CompiledQueryBindingError("compiled query differs from hard-mask bindings")

    @staticmethod
    def _validate_constraint(
        constraint: CompiledHardConstraint,
        *,
        exclusion: bool,
    ) -> None:
        if type(constraint) is not CompiledHardConstraint:
            raise HardMaskResolutionError("hard constraint has an invalid representation")
        expected_operators = _EXCLUSION_OPERATORS if exclusion else _INCLUSION_OPERATORS
        if constraint.operator not in expected_operators:
            raise HardMaskResolutionError("hard constraint operator is unsupported")
        if constraint.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
            if constraint.policy is not ConstraintPolicy.VERIFIED_CATEGORY:
                raise HardMaskResolutionError("category constraint has the wrong evidence policy")
            if constraint.operator not in {
                Operator.EQ,
                Operator.IN,
                Operator.NEQ,
                Operator.NOT_IN,
            }:
                raise HardMaskResolutionError("category constraint operator is unsupported")
        elif constraint.facet == "price":
            if constraint.policy is not ConstraintPolicy.CONSERVATIVE_PRICE:
                raise HardMaskResolutionError("price constraint has the wrong evidence policy")
            if constraint.operator not in _PRICE_OPERATORS:
                raise HardMaskResolutionError("price constraint operator is unsupported")
        elif constraint.policy is not ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE:
            raise HardMaskResolutionError("text constraint has the wrong evidence policy")
        elif constraint.operator not in {
            Operator.EQ,
            Operator.IN,
            Operator.NEQ,
            Operator.NOT_IN,
        }:
            raise HardMaskResolutionError("text constraint operator is unsupported")

    def _matching_products(self, constraint: CompiledHardConstraint) -> frozenset[str]:
        values = _constraint_values(constraint)
        if constraint.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
            return self._matching_categories(values)
        if constraint.facet == "price":
            return self._matching_prices(constraint.operator, values)
        matches: set[str] = set()
        for value in values:
            if type(value) is not str:
                raise HardMaskResolutionError("text constraint requires string values")
            try:
                observed = self._evidence_index.match(constraint.facet, value)
            except (TypeError, ValueError) as error:
                raise HardMaskResolutionError(
                    f"retrieval evidence cannot resolve {constraint.facet!r}: {error}"
                ) from error
            matches.update(observed)
        if any(type(item) is not str or item not in self._assignments for item in matches):
            raise HardMaskResolutionError("retrieval evidence match crossed the product binding")
        return frozenset(matches)

    def _matching_categories(self, values: tuple[ScalarValue, ...]) -> frozenset[str]:
        scopes = []
        for value in values:
            if type(value) is not str:
                raise HardMaskResolutionError("category constraint requires scope IDs")
            try:
                scopes.append(self._scopes[value])
            except KeyError as error:
                raise HardMaskResolutionError(
                    "category constraint names an unknown scope"
                ) from error
        return frozenset(
            parent_asin
            for parent_asin, assignment in self._assignments.items()
            if any(
                match_category(assignment, scope) is CategoryMatchResult.SATISFIED
                for scope in scopes
            )
        )

    def _matching_prices(
        self,
        operator: Operator,
        values: tuple[ScalarValue, ...],
    ) -> frozenset[str]:
        allowed = tuple(_price_interval(operator, value) for value in values)
        matches = set(self._parent_asins)
        for parent_asin, product in self._price_by_parent.items():
            if not any(
                safe_filter_keeps(match_numeric_interval(product, item)) for item in allowed
            ):
                matches.discard(parent_asin)
        return frozenset(matches)


def _constraint_values(constraint: CompiledHardConstraint) -> tuple[ScalarValue, ...]:
    value = constraint.value
    if constraint.operator in {Operator.IN, Operator.NOT_IN}:
        if type(value) is not tuple:
            raise HardMaskResolutionError("IN and NOT_IN constraints require a value tuple")
        values = value
    else:
        if type(value) is tuple:
            raise HardMaskResolutionError("non-set constraints require one scalar value")
        values = (cast(ScalarValue, value),)
    if not values:
        raise HardMaskResolutionError("hard constraint values cannot be empty")
    if any(type(item) not in (str, int, float, bool) for item in values):
        raise HardMaskResolutionError("hard constraint contains an invalid scalar value")
    return values


def _price_interval(operator: Operator, value: ScalarValue) -> NumericValue:
    if type(value) is not int:
        raise HardMaskResolutionError("price constraint requires an integer cent value")
    numeric = value
    if operator is Operator.LT:
        return NumericValue(
            kind="numeric",
            lower=None,
            lower_inclusive=False,
            upper=numeric,
            upper_inclusive=False,
            unit=USD_CENT_UNIT,
        )
    if operator is Operator.LE:
        return NumericValue(
            kind="numeric",
            lower=None,
            lower_inclusive=False,
            upper=numeric,
            upper_inclusive=True,
            unit=USD_CENT_UNIT,
        )
    if operator is Operator.GT:
        return NumericValue(
            kind="numeric",
            lower=numeric,
            lower_inclusive=False,
            upper=None,
            upper_inclusive=False,
            unit=USD_CENT_UNIT,
        )
    if operator is Operator.GE:
        return NumericValue(
            kind="numeric",
            lower=numeric,
            lower_inclusive=True,
            upper=None,
            upper_inclusive=False,
            unit=USD_CENT_UNIT,
        )
    raise HardMaskResolutionError("price constraint operator is unsupported")
