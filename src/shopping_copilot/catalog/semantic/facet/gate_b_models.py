"""Typed Gate-B capability proposals and deterministic price review evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..canonical import IJSON_SAFE_INTEGER_MAX, validate_semantic_string
from .gate_a_models import EvidenceStatus, NumericValue, ProductFacetStatus
from .resolution_matching import FacetMatchResult
from .resolution_models import RESOLUTION_POLICY_ID

GATE_B_PRICE_REVIEW_SCHEMA: Literal["shopping-copilot/gate-b-price-review/v0"] = (
    "shopping-copilot/gate-b-price-review/v0"
)
GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA: Literal["shopping-copilot/gate-b-public-target-audit/v0"] = (
    "shopping-copilot/gate-b-public-target-audit/v0"
)
GATE_B_REVIEW_BUILDER_VERSION = "catalog_semantic_gate_b_review_v0"
PRICE_INTENT_NORMALIZER_ID = "usd_cent_int_v1"
GATE_B_SELECTION_SCHEMA: Literal["shopping-copilot/gate-b-selection/v0"] = (
    "shopping-copilot/gate-b-selection/v0"
)
EFFECTIVE_FACET_CAPABILITIES_SCHEMA: Literal["shopping-copilot/effective-facet-capabilities/v0"] = (
    "shopping-copilot/effective-facet-capabilities/v0"
)
GATE_B_CANDIDATE_SCHEMA: Literal["shopping-copilot/gate-b-candidate/v0"] = (
    "shopping-copilot/gate-b-candidate/v0"
)
GATE_B_BUILDER_VERSION = "catalog_semantic_gate_b_v0"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCOPE_ID_PATTERN = re.compile(r"^cs_[0-9a-f]{64}$")
_SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class RuntimePromotionDecision(str, Enum):
    """Human Gate-B disposition for one exact facet/category scope."""

    RUNTIME_ACCEPT = "runtime_accept"
    SEARCH_ONLY = "search_only"
    SEMANTIC_ONLY = "semantic_only"
    REJECT = "reject"


class GateBReviewState(str, Enum):
    """Publication state of a review proposal before owner approval."""

    AWAITING_OWNER_APPROVAL = "awaiting_owner_approval"


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveFacetCapability:
    """Proposed exact-scope capability row from the normative Gate-B contract."""

    facet_id: str
    category_scope_id: str
    decision: RuntimePromotionDecision
    resolution_policy_id: Literal["structured_resolution_v1"]
    intent_committable: bool
    retrieval_eligible: bool
    probe_eligible: bool
    clarification_eligible: bool

    def __post_init__(self) -> None:
        _require_identifier(
            self.facet_id,
            pattern=_SEMANTIC_ID_PATTERN,
            name="EffectiveFacetCapability.facet_id",
        )
        _require_identifier(
            self.category_scope_id,
            pattern=_SCOPE_ID_PATTERN,
            name="EffectiveFacetCapability.category_scope_id",
        )
        if type(self.decision) is not RuntimePromotionDecision:
            raise TypeError("EffectiveFacetCapability.decision is invalid")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("EffectiveFacetCapability resolution policy is invalid")
        flags = (
            self.intent_committable,
            self.retrieval_eligible,
            self.probe_eligible,
            self.clarification_eligible,
        )
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("EffectiveFacetCapability flags must be boolean")
        if self.clarification_eligible and not self.probe_eligible:
            raise ValueError("clarification eligibility requires Probe eligibility")
        if self.probe_eligible and not self.retrieval_eligible:
            raise ValueError("Probe eligibility requires retrieval eligibility")
        if self.intent_committable and not self.retrieval_eligible:
            raise ValueError("intent commitment requires retrieval eligibility")
        if self.intent_committable and self.decision is not RuntimePromotionDecision.RUNTIME_ACCEPT:
            raise ValueError("only RUNTIME_ACCEPT may be intent-committable")
        if self.decision in (
            RuntimePromotionDecision.SEMANTIC_ONLY,
            RuntimePromotionDecision.REJECT,
        ) and any(flags):
            raise ValueError("SEMANTIC_ONLY and REJECT must disable all capabilities")
        if self.decision is RuntimePromotionDecision.SEARCH_ONLY and (
            self.intent_committable or self.clarification_eligible
        ):
            raise ValueError("SEARCH_ONLY cannot commit intent or clarify")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopePriceReview:
    """One exact category scope's resolved price evidence and proposal."""

    facet_id: str
    category_scope_id: str
    scope_label: str
    scope_product_count: int
    known_count: int
    unknown_count: int
    conflict_count: int
    not_applicable_count: int
    exact_interval_count: int
    lower_bound_interval_count: int
    public_target_count: int
    public_target_known_count: int
    minimum_lower_cents: int | None
    median_lower_cents: int | None
    p90_lower_cents: int | None
    maximum_lower_cents: int | None
    proposed_capability: EffectiveFacetCapability

    def __post_init__(self) -> None:
        _require_identifier(
            self.facet_id,
            pattern=_SEMANTIC_ID_PATTERN,
            name="ScopePriceReview.facet_id",
        )
        _require_identifier(
            self.category_scope_id,
            pattern=_SCOPE_ID_PATTERN,
            name="ScopePriceReview.category_scope_id",
        )
        validate_semantic_string(self.scope_label, name="ScopePriceReview.scope_label")
        for name in (
            "scope_product_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
            "exact_interval_count",
            "lower_bound_interval_count",
            "public_target_count",
            "public_target_known_count",
        ):
            _require_nonnegative_int(getattr(self, name), name=f"ScopePriceReview.{name}")
        if (
            self.known_count + self.unknown_count + self.conflict_count + self.not_applicable_count
            != self.scope_product_count
        ):
            raise ValueError("ScopePriceReview status counts do not conserve products")
        if self.exact_interval_count + self.lower_bound_interval_count != self.known_count:
            raise ValueError("ScopePriceReview interval lanes do not sum to KNOWN")
        if self.public_target_known_count > self.public_target_count:
            raise ValueError("ScopePriceReview public known count exceeds targets")
        quantiles = (
            self.minimum_lower_cents,
            self.median_lower_cents,
            self.p90_lower_cents,
            self.maximum_lower_cents,
        )
        if self.known_count:
            if any(value is None for value in quantiles):
                raise ValueError("known scope requires price quantiles")
            observed = tuple(value for value in quantiles if value is not None)
            if observed != tuple(sorted(observed)):
                raise ValueError("ScopePriceReview quantiles must be nondecreasing")
        elif any(value is not None for value in quantiles):
            raise ValueError("scope without known values cannot have quantiles")
        for value in quantiles:
            if value is not None:
                _require_nonnegative_int(value, name="ScopePriceReview quantile")
        if type(self.proposed_capability) is not EffectiveFacetCapability:
            raise TypeError("ScopePriceReview.proposed_capability is invalid")
        if (
            self.proposed_capability.facet_id != self.facet_id
            or self.proposed_capability.category_scope_id != self.category_scope_id
        ):
            raise ValueError("ScopePriceReview capability key differs from its scope")


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetSafetyAuditRow:
    """Catalog-wide result of one synthetic upper-budget constraint."""

    budget_cents: int
    product_count: int
    satisfied_count: int
    violated_count: int
    unknown_count: int
    not_applicable_count: int
    safe_retained_count: int
    unsafe_satisfied_only_retained_count: int

    def __post_init__(self) -> None:
        for name in (
            "budget_cents",
            "product_count",
            "satisfied_count",
            "violated_count",
            "unknown_count",
            "not_applicable_count",
            "safe_retained_count",
            "unsafe_satisfied_only_retained_count",
        ):
            _require_nonnegative_int(getattr(self, name), name=f"BudgetSafetyAuditRow.{name}")
        if self.product_count <= 0:
            raise ValueError("BudgetSafetyAuditRow.product_count must be positive")
        if (
            self.satisfied_count
            + self.violated_count
            + self.unknown_count
            + self.not_applicable_count
            != self.product_count
        ):
            raise ValueError("BudgetSafetyAuditRow results do not conserve products")
        if self.safe_retained_count != self.product_count - self.violated_count:
            raise ValueError("safe retention must drop only VIOLATED")
        if self.unsafe_satisfied_only_retained_count != self.satisfied_count:
            raise ValueError("satisfied-only retention count is inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicTargetPriceRecord:
    """One public target's resolved price and conservative retention outcome."""

    sample_id: str
    scenario_type: str
    parent_asin: str
    price_status: ProductFacetStatus
    evidence_status: EvidenceStatus | None
    price_value: NumericValue | None
    compatible_budget_cents: int | None
    safe_match_result: FacetMatchResult
    safe_retained: bool
    unsafe_satisfied_only_retained: bool

    def __post_init__(self) -> None:
        validate_semantic_string(self.sample_id, name="PublicTargetPriceRecord.sample_id")
        validate_semantic_string(
            self.scenario_type,
            name="PublicTargetPriceRecord.scenario_type",
        )
        validate_semantic_string(self.parent_asin, name="PublicTargetPriceRecord.parent_asin")
        if type(self.price_status) is not ProductFacetStatus:
            raise TypeError("PublicTargetPriceRecord.price_status is invalid")
        if self.evidence_status is not None and type(self.evidence_status) is not EvidenceStatus:
            raise TypeError("PublicTargetPriceRecord.evidence_status is invalid")
        if self.price_value is not None and type(self.price_value) is not NumericValue:
            raise TypeError("PublicTargetPriceRecord.price_value is invalid")
        if self.price_status is ProductFacetStatus.KNOWN:
            if self.price_value is None or self.evidence_status is not EvidenceStatus.VALID:
                raise ValueError("KNOWN public target requires valid price evidence")
            if self.compatible_budget_cents is None:
                raise ValueError("KNOWN public target requires a compatible budget")
        elif self.price_value is not None or self.compatible_budget_cents is not None:
            raise ValueError("non-KNOWN public target cannot carry a price or compatible budget")
        if self.compatible_budget_cents is not None:
            _require_nonnegative_int(
                self.compatible_budget_cents,
                name="PublicTargetPriceRecord.compatible_budget_cents",
            )
        if type(self.safe_match_result) is not FacetMatchResult:
            raise TypeError("PublicTargetPriceRecord.safe_match_result is invalid")
        if (
            type(self.safe_retained) is not bool
            or type(self.unsafe_satisfied_only_retained) is not bool
        ):
            raise TypeError("PublicTargetPriceRecord retention flags must be boolean")
        if self.safe_retained != (self.safe_match_result is not FacetMatchResult.VIOLATED):
            raise ValueError("PublicTargetPriceRecord safe retention is inconsistent")
        if self.unsafe_satisfied_only_retained != (
            self.safe_match_result is FacetMatchResult.SATISFIED
        ):
            raise ValueError("PublicTargetPriceRecord satisfied-only retention is inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicScenarioPriceSummary:
    """Public target price state counts for one simulator scenario label."""

    scenario_type: str
    target_count: int
    known_count: int
    unknown_count: int
    conflict_count: int
    not_applicable_count: int

    def __post_init__(self) -> None:
        validate_semantic_string(
            self.scenario_type,
            name="PublicScenarioPriceSummary.scenario_type",
        )
        for name in (
            "target_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
        ):
            _require_nonnegative_int(
                getattr(self, name),
                name=f"PublicScenarioPriceSummary.{name}",
            )
        if (
            self.known_count + self.unknown_count + self.conflict_count + self.not_applicable_count
            != self.target_count
        ):
            raise ValueError("PublicScenarioPriceSummary counts do not conserve targets")


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicTargetPriceAudit:
    """Complete 200-target price retention audit bound to the public JSONL bytes."""

    schema: Literal["shopping-copilot/gate-b-public-target-audit/v0"]
    public_set_id: str
    catalog_id: str
    product_facet_index_id: str
    target_count: int
    known_count: int
    unknown_count: int
    conflict_count: int
    not_applicable_count: int
    compatible_budget_safe_retained_count: int
    unsafe_satisfied_only_retained_count: int
    scenario_summaries: tuple[PublicScenarioPriceSummary, ...]
    records: tuple[PublicTargetPriceRecord, ...]

    def __post_init__(self) -> None:
        if self.schema != GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA:
            raise ValueError("PublicTargetPriceAudit.schema is invalid")
        for name in ("public_set_id", "catalog_id", "product_facet_index_id"):
            _require_content_id(getattr(self, name), name=f"PublicTargetPriceAudit.{name}")
        for name in (
            "target_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
            "compatible_budget_safe_retained_count",
            "unsafe_satisfied_only_retained_count",
        ):
            _require_nonnegative_int(getattr(self, name), name=f"PublicTargetPriceAudit.{name}")
        if (
            self.known_count + self.unknown_count + self.conflict_count + self.not_applicable_count
            != self.target_count
        ):
            raise ValueError("PublicTargetPriceAudit status counts do not conserve targets")
        _require_exact_tuple(
            self.scenario_summaries,
            PublicScenarioPriceSummary,
            name="PublicTargetPriceAudit.scenario_summaries",
        )
        scenario_keys = tuple(item.scenario_type for item in self.scenario_summaries)
        if scenario_keys != tuple(sorted(set(scenario_keys))):
            raise ValueError("PublicTargetPriceAudit scenario summaries must be sorted and unique")
        _require_exact_tuple(
            self.records,
            PublicTargetPriceRecord,
            name="PublicTargetPriceAudit.records",
        )
        record_keys = tuple(item.sample_id for item in self.records)
        if record_keys != tuple(sorted(set(record_keys))):
            raise ValueError("PublicTargetPriceAudit records must be sorted and unique")
        if len(self.records) != self.target_count:
            raise ValueError("PublicTargetPriceAudit record count differs from target_count")
        if sum(item.target_count for item in self.scenario_summaries) != self.target_count:
            raise ValueError("PublicTargetPriceAudit scenario counts do not sum to targets")
        if (
            sum(int(item.safe_retained) for item in self.records)
            != self.compatible_budget_safe_retained_count
        ):
            raise ValueError("PublicTargetPriceAudit safe retained count is inconsistent")
        if (
            sum(int(item.unsafe_satisfied_only_retained) for item in self.records)
            != self.unsafe_satisfied_only_retained_count
        ):
            raise ValueError("PublicTargetPriceAudit unsafe retained count is inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class GateBPriceReviewProposal:
    """Deterministic Gate-B price recommendation that carries no approval authority."""

    schema: Literal["shopping-copilot/gate-b-price-review/v0"]
    builder_version: str
    review_state: GateBReviewState
    catalog_id: str
    category_registry_id: str
    facet_schema_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    catalog_facet_stats_id: str
    public_target_audit_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    proposed_intent_normalizer_id: str
    reviewed_value_aliases: tuple[str, ...]
    proposed_capabilities: tuple[EffectiveFacetCapability, ...]
    scope_reviews: tuple[ScopePriceReview, ...]
    budget_safety_rows: tuple[BudgetSafetyAuditRow, ...]
    recommendation_summary: str

    def __post_init__(self) -> None:
        if self.schema != GATE_B_PRICE_REVIEW_SCHEMA:
            raise ValueError("GateBPriceReviewProposal.schema is invalid")
        if self.builder_version != GATE_B_REVIEW_BUILDER_VERSION:
            raise ValueError("GateBPriceReviewProposal.builder_version is unsupported")
        if self.review_state is not GateBReviewState.AWAITING_OWNER_APPROVAL:
            raise ValueError("GateBPriceReviewProposal cannot claim approval")
        for name in (
            "catalog_id",
            "category_registry_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
            "catalog_facet_stats_id",
            "public_target_audit_id",
        ):
            _require_content_id(getattr(self, name), name=f"GateBPriceReviewProposal.{name}")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("GateBPriceReviewProposal resolution policy is invalid")
        if self.proposed_intent_normalizer_id != PRICE_INTENT_NORMALIZER_ID:
            raise ValueError("GateBPriceReviewProposal intent normalizer is unsupported")
        if type(self.reviewed_value_aliases) is not tuple or self.reviewed_value_aliases:
            raise ValueError("numeric price v0 must have an empty value-alias list")
        _require_exact_tuple(
            self.proposed_capabilities,
            EffectiveFacetCapability,
            name="GateBPriceReviewProposal.proposed_capabilities",
        )
        capability_keys = tuple(
            (item.facet_id, item.category_scope_id) for item in self.proposed_capabilities
        )
        if capability_keys != tuple(sorted(set(capability_keys))):
            raise ValueError("GateB proposed capabilities must be sorted and unique")
        _require_exact_tuple(
            self.scope_reviews,
            ScopePriceReview,
            name="GateBPriceReviewProposal.scope_reviews",
        )
        scope_keys = tuple((item.facet_id, item.category_scope_id) for item in self.scope_reviews)
        if scope_keys != capability_keys:
            raise ValueError("GateB scope reviews do not cover the exact capability proposal")
        if tuple(item.proposed_capability for item in self.scope_reviews) != (
            self.proposed_capabilities
        ):
            raise ValueError("GateB scope review capabilities differ from proposal")
        _require_exact_tuple(
            self.budget_safety_rows,
            BudgetSafetyAuditRow,
            name="GateBPriceReviewProposal.budget_safety_rows",
        )
        budgets = tuple(item.budget_cents for item in self.budget_safety_rows)
        if budgets != tuple(sorted(set(budgets))):
            raise ValueError("GateB budget safety rows must be sorted and unique")
        validate_semantic_string(
            self.recommendation_summary,
            name="GateBPriceReviewProposal.recommendation_summary",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GateBSelection:
    """Source-controlled repository-owner approval of one exact review proposal."""

    schema: Literal["shopping-copilot/gate-b-selection/v0"]
    builder_version: str
    catalog_id: str
    category_registry_id: str
    facet_schema_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    catalog_facet_stats_id: str
    gate_b_review_proposal_id: str
    public_target_audit_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    intent_value_normalizer_id: str
    reviewed_value_aliases: tuple[str, ...]
    approvals: tuple[EffectiveFacetCapability, ...]
    rationale: str

    def __post_init__(self) -> None:
        if self.schema != GATE_B_SELECTION_SCHEMA:
            raise ValueError("GateBSelection.schema is invalid")
        if self.builder_version != GATE_B_BUILDER_VERSION:
            raise ValueError("GateBSelection.builder_version is unsupported")
        for name in (
            "catalog_id",
            "category_registry_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
            "catalog_facet_stats_id",
            "gate_b_review_proposal_id",
            "public_target_audit_id",
        ):
            _require_content_id(getattr(self, name), name=f"GateBSelection.{name}")
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("GateBSelection resolution policy is invalid")
        if self.intent_value_normalizer_id != PRICE_INTENT_NORMALIZER_ID:
            raise ValueError("GateBSelection intent normalizer is unsupported")
        if type(self.reviewed_value_aliases) is not tuple or self.reviewed_value_aliases:
            raise ValueError("numeric price v0 must have an empty value-alias list")
        _require_exact_tuple(
            self.approvals,
            EffectiveFacetCapability,
            name="GateBSelection.approvals",
        )
        approval_keys = tuple((item.facet_id, item.category_scope_id) for item in self.approvals)
        if not approval_keys or approval_keys != tuple(sorted(set(approval_keys))):
            raise ValueError("GateBSelection approvals must be non-empty, sorted, and unique")
        validate_semantic_string(self.rationale, name="GateBSelection.rationale")


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveFacetCapabilitySet:
    """Normative exact-scope runtime permissions published by approved Gate B."""

    schema: Literal["shopping-copilot/effective-facet-capabilities/v0"]
    category_registry_id: str
    facet_schema_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    entries: tuple[EffectiveFacetCapability, ...]

    def __post_init__(self) -> None:
        if self.schema != EFFECTIVE_FACET_CAPABILITIES_SCHEMA:
            raise ValueError("EffectiveFacetCapabilitySet.schema is invalid")
        for name in (
            "category_registry_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
        ):
            _require_content_id(
                getattr(self, name),
                name=f"EffectiveFacetCapabilitySet.{name}",
            )
        if self.resolution_policy_id != RESOLUTION_POLICY_ID:
            raise ValueError("EffectiveFacetCapabilitySet resolution policy is invalid")
        _require_exact_tuple(
            self.entries,
            EffectiveFacetCapability,
            name="EffectiveFacetCapabilitySet.entries",
        )
        keys = tuple((item.facet_id, item.category_scope_id) for item in self.entries)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("effective capabilities must be non-empty, sorted, and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class GateBCandidateBuild:
    """Owner-approved Gate-B selection and its deterministic contract projection."""

    schema: Literal["shopping-copilot/gate-b-candidate/v0"]
    builder_version: str
    catalog_id: str
    catalog_facet_stats_id: str
    gate_b_review_proposal_id: str
    public_target_audit_id: str
    selection: GateBSelection
    capabilities: EffectiveFacetCapabilitySet

    def __post_init__(self) -> None:
        if self.schema != GATE_B_CANDIDATE_SCHEMA:
            raise ValueError("GateBCandidateBuild.schema is invalid")
        if self.builder_version != GATE_B_BUILDER_VERSION:
            raise ValueError("GateBCandidateBuild.builder_version is unsupported")
        for name in (
            "catalog_id",
            "catalog_facet_stats_id",
            "gate_b_review_proposal_id",
            "public_target_audit_id",
        ):
            _require_content_id(getattr(self, name), name=f"GateBCandidateBuild.{name}")
        if type(self.selection) is not GateBSelection:
            raise TypeError("GateBCandidateBuild.selection is invalid")
        if type(self.capabilities) is not EffectiveFacetCapabilitySet:
            raise TypeError("GateBCandidateBuild.capabilities is invalid")
        if (
            self.selection.catalog_id != self.catalog_id
            or self.selection.catalog_facet_stats_id != self.catalog_facet_stats_id
            or self.selection.gate_b_review_proposal_id != self.gate_b_review_proposal_id
            or self.selection.public_target_audit_id != self.public_target_audit_id
            or self.selection.category_registry_id != self.capabilities.category_registry_id
            or self.selection.facet_schema_id != self.capabilities.facet_schema_id
            or self.selection.facet_applicability_id != self.capabilities.facet_applicability_id
            or self.selection.product_facet_index_id != self.capabilities.product_facet_index_id
            or self.selection.resolution_policy_id != self.capabilities.resolution_policy_id
            or self.selection.approvals != self.capabilities.entries
        ):
            raise ValueError("Gate-B candidate pins or capability projection differ")


def _require_content_id(value: object, *, name: str) -> None:
    _require_identifier(value, pattern=_CONTENT_ID_PATTERN, name=name)


def _require_identifier(value: object, *, pattern: re.Pattern[str], name: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must be a non-negative I-JSON integer")


def _require_exact_tuple(values: object, expected_type: type[object], *, name: str) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(item) is not expected_type for item in values):
        raise TypeError(f"{name} contains an invalid item")
