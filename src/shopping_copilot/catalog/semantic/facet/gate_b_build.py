"""Deterministic Gate-B price review analysis without runtime promotion."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..canonical import content_id_for_value
from ..category import (
    CategoryRegistry,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
)
from ..errors import GateBReviewBuildError
from .gate_a_models import GateACandidateBuild, NumericValue, ProductFacetStatus
from .gate_b_models import (
    GATE_B_PRICE_REVIEW_SCHEMA,
    GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA,
    GATE_B_REVIEW_BUILDER_VERSION,
    PRICE_INTENT_NORMALIZER_ID,
    BudgetSafetyAuditRow,
    EffectiveFacetCapability,
    GateBPriceReviewProposal,
    GateBReviewState,
    PublicScenarioPriceSummary,
    PublicTargetPriceAudit,
    PublicTargetPriceRecord,
    RuntimePromotionDecision,
    ScopePriceReview,
)
from .resolution_matching import FacetMatchResult, match_numeric_interval, safe_filter_keeps
from .resolution_models import (
    RESOLUTION_POLICY_ID,
    ResolutionCandidateBuild,
    ResolvedProductFacetValue,
)

DEFAULT_BUDGET_CENTS = (1_000, 2_500, 5_000, 10_000)
DEFAULT_PUBLIC_TARGET_COUNT = 200
_UNKNOWN_AUDIT_BUDGET_CENTS = 5_000


@dataclass(frozen=True, slots=True, kw_only=True)
class GateBPriceReviewBuild:
    """Review proposal plus its separately content-addressed public-target audit."""

    proposal: GateBPriceReviewProposal
    public_target_audit: PublicTargetPriceAudit

    def __post_init__(self) -> None:
        if type(self.proposal) is not GateBPriceReviewProposal:
            raise TypeError("GateBPriceReviewBuild.proposal is invalid")
        if type(self.public_target_audit) is not PublicTargetPriceAudit:
            raise TypeError("GateBPriceReviewBuild.public_target_audit is invalid")
        if self.proposal.public_target_audit_id != content_id_for_value(self.public_target_audit):
            raise ValueError("Gate-B proposal does not pin its public target audit")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PublicSample:
    sample_id: str
    scenario_type: str
    parent_asin: str


class _DuplicateJsonKeyError(ValueError):
    pass


def build_gate_b_price_review(
    public_set_path: str | Path,
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    resolution: ResolutionCandidateBuild,
    expected_public_target_count: int = DEFAULT_PUBLIC_TARGET_COUNT,
) -> GateBPriceReviewBuild:
    """Build an owner-reviewable price capability proposal from verified CS3 truth."""

    _validate_upstream_pins(
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        resolution=resolution,
    )
    public_set_id, samples = _load_public_samples(
        Path(public_set_path),
        expected_count=expected_public_target_count,
        catalog_ids={item.parent_asin for item in assignments.assignments},
    )
    public_audit = _build_public_target_audit(
        public_set_id=public_set_id,
        samples=samples,
        resolution=resolution,
    )
    capabilities = tuple(
        EffectiveFacetCapability(
            facet_id="price",
            category_scope_id=scope.id,
            decision=RuntimePromotionDecision.RUNTIME_ACCEPT,
            resolution_policy_id=RESOLUTION_POLICY_ID,
            intent_committable=True,
            retrieval_eligible=True,
            probe_eligible=True,
            clarification_eligible=False,
        )
        for scope in sorted(registry.scopes, key=lambda item: item.id)
    )
    scope_reviews = _build_scope_reviews(
        registry=registry,
        assignments=assignments,
        resolution=resolution,
        samples=samples,
        capabilities=capabilities,
    )
    budget_rows = tuple(
        _build_budget_safety_row(
            budget_cents=budget_cents,
            assignments=assignments,
            resolution=resolution,
        )
        for budget_cents in DEFAULT_BUDGET_CENTS
    )
    proposal = GateBPriceReviewProposal(
        schema=GATE_B_PRICE_REVIEW_SCHEMA,
        builder_version=GATE_B_REVIEW_BUILDER_VERSION,
        review_state=GateBReviewState.AWAITING_OWNER_APPROVAL,
        catalog_id=resolution.evidence_store.catalog_id,
        category_registry_id=resolution.category_registry_id,
        facet_schema_id=resolution.facet_schema_id,
        facet_applicability_id=resolution.evidence_store.facet_applicability_id,
        product_facet_index_id=content_id_for_value(resolution.product_facet_index),
        catalog_facet_stats_id=content_id_for_value(resolution.stats),
        public_target_audit_id=content_id_for_value(public_audit),
        resolution_policy_id=RESOLUTION_POLICY_ID,
        proposed_intent_normalizer_id=PRICE_INTENT_NORMALIZER_ID,
        reviewed_value_aliases=(),
        proposed_capabilities=capabilities,
        scope_reviews=scope_reviews,
        budget_safety_rows=budget_rows,
        recommendation_summary=(
            "Propose exact-scope RUNTIME_ACCEPT for price with intent, retrieval, and Probe "
            "enabled; keep proactive clarification disabled until end-to-end utility evidence "
            "exists. Retrieval must retain UNKNOWN and drop only proven VIOLATED products."
        ),
    )
    return GateBPriceReviewBuild(proposal=proposal, public_target_audit=public_audit)


def _build_scope_reviews(
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    resolution: ResolutionCandidateBuild,
    samples: tuple[_PublicSample, ...],
    capabilities: tuple[EffectiveFacetCapability, ...],
) -> tuple[ScopePriceReview, ...]:
    entry_by_asin = {
        item.parent_asin: item
        for item in resolution.product_facet_index.entries
        if item.facet_id == "price"
    }
    assignment_by_asin = {item.parent_asin: item for item in assignments.assignments}
    stats_by_scope = {
        item.category_scope_id: item for item in resolution.stats.rows if item.facet_id == "price"
    }
    capability_by_scope = {item.category_scope_id: item for item in capabilities}
    reviews: list[ScopePriceReview] = []
    for scope in registry.scopes:
        stats = stats_by_scope.get(scope.id)
        if stats is None:
            raise GateBReviewBuildError("CS3 stats do not cover every category scope")
        known_entries = tuple(
            entry_by_asin[assignment.parent_asin]
            for assignment in assignments.assignments
            if _assignment_intersects_scope(assignment, scope.member_node_ids)
            and assignment.parent_asin in entry_by_asin
        )
        values = tuple(_require_price_value(item) for item in known_entries)
        lower_values = tuple(sorted(cast(int, item.lower) for item in values))
        public_scope_samples = tuple(
            sample
            for sample in samples
            if _assignment_intersects_scope(
                assignment_by_asin[sample.parent_asin],
                scope.member_node_ids,
            )
        )
        reviews.append(
            ScopePriceReview(
                facet_id="price",
                category_scope_id=scope.id,
                scope_label=scope.label,
                scope_product_count=stats.scope_product_count,
                known_count=stats.known_count,
                unknown_count=stats.unknown_count,
                conflict_count=stats.conflict_count,
                not_applicable_count=stats.not_applicable_count,
                exact_interval_count=sum(item.upper is not None for item in values),
                lower_bound_interval_count=sum(item.upper is None for item in values),
                public_target_count=len(public_scope_samples),
                public_target_known_count=sum(
                    sample.parent_asin in entry_by_asin for sample in public_scope_samples
                ),
                minimum_lower_cents=lower_values[0] if lower_values else None,
                median_lower_cents=_nearest_rank(lower_values, numerator=1, denominator=2),
                p90_lower_cents=_nearest_rank(lower_values, numerator=9, denominator=10),
                maximum_lower_cents=lower_values[-1] if lower_values else None,
                proposed_capability=capability_by_scope[scope.id],
            )
        )
    return tuple(sorted(reviews, key=lambda item: (item.facet_id, item.category_scope_id)))


def _build_budget_safety_row(
    *,
    budget_cents: int,
    assignments: ProductCategoryAssignmentSet,
    resolution: ResolutionCandidateBuild,
) -> BudgetSafetyAuditRow:
    allowed = _upper_budget(budget_cents)
    entry_by_asin = {
        item.parent_asin: item
        for item in resolution.product_facet_index.entries
        if item.facet_id == "price"
    }
    counts: Counter[FacetMatchResult] = Counter()
    for assignment in assignments.assignments:
        product = entry_by_asin.get(
            assignment.parent_asin,
            _implicit_unknown(assignment.parent_asin),
        )
        counts[match_numeric_interval(product, allowed)] += 1
    product_count = len(assignments.assignments)
    return BudgetSafetyAuditRow(
        budget_cents=budget_cents,
        product_count=product_count,
        satisfied_count=counts[FacetMatchResult.SATISFIED],
        violated_count=counts[FacetMatchResult.VIOLATED],
        unknown_count=counts[FacetMatchResult.UNKNOWN],
        not_applicable_count=counts[FacetMatchResult.NOT_APPLICABLE],
        safe_retained_count=product_count - counts[FacetMatchResult.VIOLATED],
        unsafe_satisfied_only_retained_count=counts[FacetMatchResult.SATISFIED],
    )


def _build_public_target_audit(
    *,
    public_set_id: str,
    samples: tuple[_PublicSample, ...],
    resolution: ResolutionCandidateBuild,
) -> PublicTargetPriceAudit:
    entry_by_asin = {
        item.parent_asin: item
        for item in resolution.product_facet_index.entries
        if item.facet_id == "price"
    }
    evidence_by_asin = {
        item.parent_asin: item
        for item in resolution.evidence_store.evidence
        if item.facet_id == "price"
    }
    records: list[PublicTargetPriceRecord] = []
    scenario_counts: dict[str, Counter[ProductFacetStatus]] = defaultdict(Counter)
    for sample in samples:
        product = entry_by_asin.get(sample.parent_asin, _implicit_unknown(sample.parent_asin))
        evidence = evidence_by_asin.get(sample.parent_asin)
        value = _optional_price_value(product)
        compatible_budget = None
        if value is not None:
            compatible_budget = cast(int, value.upper if value.upper is not None else value.lower)
        allowed = _upper_budget(
            compatible_budget if compatible_budget is not None else _UNKNOWN_AUDIT_BUDGET_CENTS
        )
        result = match_numeric_interval(product, allowed)
        records.append(
            PublicTargetPriceRecord(
                sample_id=sample.sample_id,
                scenario_type=sample.scenario_type,
                parent_asin=sample.parent_asin,
                price_status=product.status,
                evidence_status=evidence.status if evidence is not None else None,
                price_value=value,
                compatible_budget_cents=compatible_budget,
                safe_match_result=result,
                safe_retained=safe_filter_keeps(result),
                unsafe_satisfied_only_retained=result is FacetMatchResult.SATISFIED,
            )
        )
        scenario_counts[sample.scenario_type][product.status] += 1
    ordered_records = tuple(sorted(records, key=lambda item: item.sample_id))
    status_counts = Counter(item.price_status for item in ordered_records)
    scenario_summaries = tuple(
        PublicScenarioPriceSummary(
            scenario_type=scenario,
            target_count=sum(counts.values()),
            known_count=counts[ProductFacetStatus.KNOWN],
            unknown_count=counts[ProductFacetStatus.UNKNOWN],
            conflict_count=counts[ProductFacetStatus.CONFLICT],
            not_applicable_count=counts[ProductFacetStatus.NOT_APPLICABLE],
        )
        for scenario, counts in sorted(scenario_counts.items())
    )
    return PublicTargetPriceAudit(
        schema=GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA,
        public_set_id=public_set_id,
        catalog_id=resolution.evidence_store.catalog_id,
        product_facet_index_id=content_id_for_value(resolution.product_facet_index),
        target_count=len(ordered_records),
        known_count=status_counts[ProductFacetStatus.KNOWN],
        unknown_count=status_counts[ProductFacetStatus.UNKNOWN],
        conflict_count=status_counts[ProductFacetStatus.CONFLICT],
        not_applicable_count=status_counts[ProductFacetStatus.NOT_APPLICABLE],
        compatible_budget_safe_retained_count=sum(
            int(item.safe_retained) for item in ordered_records
        ),
        unsafe_satisfied_only_retained_count=sum(
            int(item.unsafe_satisfied_only_retained) for item in ordered_records
        ),
        scenario_summaries=scenario_summaries,
        records=ordered_records,
    )


def _validate_upstream_pins(
    *,
    registry: CategoryRegistry,
    assignments: ProductCategoryAssignmentSet,
    gate_a: GateACandidateBuild,
    resolution: ResolutionCandidateBuild,
) -> None:
    if registry.catalog_id != assignments.catalog_id:
        raise GateBReviewBuildError("Gate-B category catalog pins differ")
    if resolution.evidence_store.catalog_id != registry.catalog_id:
        raise GateBReviewBuildError("Gate-B resolution catalog pin is stale")
    if resolution.category_registry_id != gate_a.category_registry_id:
        raise GateBReviewBuildError("Gate-B CategoryRegistry pin is stale")
    if resolution.facet_schema_id != content_id_for_value(gate_a.facet_schema):
        raise GateBReviewBuildError("Gate-B facet schema pin is stale")
    if resolution.evidence_store.facet_applicability_id != content_id_for_value(
        gate_a.applicability
    ):
        raise GateBReviewBuildError("Gate-B applicability pin is stale")
    if tuple(item.id for item in gate_a.facet_schema.facets) != ("price",):
        raise GateBReviewBuildError("Gate-B review v0 supports exactly price")
    if any(
        item.status is not ProductCategoryAssignmentStatus.KNOWN for item in assignments.assignments
    ):
        raise GateBReviewBuildError("Gate-B review requires KNOWN category assignments")


def _load_public_samples(
    path: Path,
    *,
    expected_count: int,
    catalog_ids: set[str],
) -> tuple[str, tuple[_PublicSample, ...]]:
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected public target count must be positive")
    digest = hashlib.sha256()
    samples: list[_PublicSample] = []
    seen_sample_ids: set[str] = set()
    try:
        stream = path.open("rb")
    except OSError as error:
        raise GateBReviewBuildError("public target set is unavailable") from error
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise GateBReviewBuildError(
                    f"blank public target line at physical line {line_number}"
                )
            try:
                parsed: object = json.loads(
                    raw_line.decode("utf-8"),
                    parse_constant=_reject_nonfinite_token,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except (
                UnicodeDecodeError,
                _DuplicateJsonKeyError,
                json.JSONDecodeError,
                RecursionError,
                ValueError,
            ) as error:
                raise GateBReviewBuildError(
                    f"invalid public target JSON at physical line {line_number}"
                ) from error
            if type(parsed) is not dict:
                raise GateBReviewBuildError("public target row must be an object")
            row = cast(dict[str, object], parsed)
            sample_id = row.get("sample_id")
            scenario_type = row.get("scenario_type")
            ground_truth = row.get("ground_truth")
            if (
                type(sample_id) is not str
                or not sample_id
                or type(scenario_type) is not str
                or not scenario_type
                or type(ground_truth) is not dict
            ):
                raise GateBReviewBuildError("public target row has invalid required fields")
            parent_asin = cast(dict[str, object], ground_truth).get("parent_asin")
            if type(parent_asin) is not str or parent_asin not in catalog_ids:
                raise GateBReviewBuildError("public target references an unknown catalog product")
            if sample_id in seen_sample_ids:
                raise GateBReviewBuildError("public sample IDs must be unique")
            seen_sample_ids.add(sample_id)
            samples.append(
                _PublicSample(
                    sample_id=sample_id,
                    scenario_type=scenario_type,
                    parent_asin=parent_asin,
                )
            )
    if len(samples) != expected_count:
        raise GateBReviewBuildError(
            f"public target set must contain exactly {expected_count} records"
        )
    ordered = tuple(sorted(samples, key=lambda item: item.sample_id))
    return f"sha256:{digest.hexdigest()}", ordered


def _assignment_intersects_scope(
    assignment: ProductCategoryAssignment,
    member_node_ids: tuple[str, ...],
) -> bool:
    return assignment.status is ProductCategoryAssignmentStatus.KNOWN and bool(
        set(assignment.leaf_node_ids).intersection(member_node_ids)
    )


def _nearest_rank(
    values: tuple[int, ...],
    *,
    numerator: int,
    denominator: int,
) -> int | None:
    if not values:
        return None
    rank = (len(values) * numerator + denominator - 1) // denominator
    return values[max(0, rank - 1)]


def _require_price_value(item: ResolvedProductFacetValue) -> NumericValue:
    value = _optional_price_value(item)
    if value is None:
        raise GateBReviewBuildError("KNOWN price row is missing NumericValue")
    return value


def _optional_price_value(item: ResolvedProductFacetValue) -> NumericValue | None:
    if item.status is not ProductFacetStatus.KNOWN:
        return None
    if type(item.value) is not NumericValue:
        raise GateBReviewBuildError("price index contains a non-numeric KNOWN value")
    if item.value.unit != "USD_CENT" or type(item.value.lower) is not int:
        raise GateBReviewBuildError("price index value is not integer USD cents")
    if item.value.upper is not None and type(item.value.upper) is not int:
        raise GateBReviewBuildError("price upper endpoint is not integer cents")
    return item.value


def _upper_budget(value: int) -> NumericValue:
    return NumericValue(
        kind="numeric",
        lower=None,
        lower_inclusive=False,
        upper=value,
        upper_inclusive=True,
        unit="USD_CENT",
    )


def _implicit_unknown(parent_asin: str) -> ResolvedProductFacetValue:
    return ResolvedProductFacetValue(
        parent_asin=parent_asin,
        facet_id="price",
        status=ProductFacetStatus.UNKNOWN,
        value=None,
        evidence_ids=(),
        resolution_policy_id=RESOLUTION_POLICY_ID,
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite JSON token: {raw}")
