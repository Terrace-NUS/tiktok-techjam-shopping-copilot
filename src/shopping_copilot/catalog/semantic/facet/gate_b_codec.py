"""Strict canonical codecs for generated Gate-B price review evidence."""

from __future__ import annotations

import json
from typing import Literal, cast

from ..canonical import IJSON_SAFE_INTEGER_MAX, canonical_json_bytes, content_id_for_value
from ..errors import GateBReviewCodecError
from .gate_a_models import EvidenceStatus, NumericValue, ProductFacetStatus
from .gate_b_models import (
    GATE_B_PRICE_REVIEW_SCHEMA,
    GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA,
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
from .resolution_matching import FacetMatchResult
from .resolution_models import RESOLUTION_POLICY_ID


class _DuplicateJsonKeyError(ValueError):
    pass


def encode_gate_b_price_review(proposal: GateBPriceReviewProposal) -> bytes:
    """Encode one non-authoritative Gate-B proposal as canonical bytes."""

    if type(proposal) is not GateBPriceReviewProposal:
        raise TypeError("Gate-B encoder requires GateBPriceReviewProposal")
    return canonical_json_bytes(proposal)


def decode_gate_b_price_review(data: bytes) -> GateBPriceReviewProposal:
    """Strictly decode canonical Gate-B price proposal bytes."""

    root = _object(
        _load_canonical_json(data, name="GateBPriceReviewProposal"),
        fields={
            "schema",
            "builder_version",
            "review_state",
            "catalog_id",
            "category_registry_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
            "catalog_facet_stats_id",
            "public_target_audit_id",
            "resolution_policy_id",
            "proposed_intent_normalizer_id",
            "reviewed_value_aliases",
            "proposed_capabilities",
            "scope_reviews",
            "budget_safety_rows",
            "recommendation_summary",
        },
        name="GateBPriceReviewProposal",
    )
    try:
        _require_schema(root["schema"], GATE_B_PRICE_REVIEW_SCHEMA, name="proposal")
        return GateBPriceReviewProposal(
            schema=GATE_B_PRICE_REVIEW_SCHEMA,
            builder_version=_string(root["builder_version"], name="builder_version"),
            review_state=GateBReviewState(_string(root["review_state"], name="review_state")),
            catalog_id=_string(root["catalog_id"], name="catalog_id"),
            category_registry_id=_string(root["category_registry_id"], name="category_registry_id"),
            facet_schema_id=_string(root["facet_schema_id"], name="facet_schema_id"),
            facet_applicability_id=_string(
                root["facet_applicability_id"], name="facet_applicability_id"
            ),
            product_facet_index_id=_string(
                root["product_facet_index_id"], name="product_facet_index_id"
            ),
            catalog_facet_stats_id=_string(
                root["catalog_facet_stats_id"], name="catalog_facet_stats_id"
            ),
            public_target_audit_id=_string(
                root["public_target_audit_id"], name="public_target_audit_id"
            ),
            resolution_policy_id=_resolution_policy(root["resolution_policy_id"]),
            proposed_intent_normalizer_id=_string(
                root["proposed_intent_normalizer_id"],
                name="proposed_intent_normalizer_id",
            ),
            reviewed_value_aliases=tuple(
                _string(item, name=f"reviewed_value_aliases[{index}]")
                for index, item in enumerate(
                    _array(root["reviewed_value_aliases"], name="reviewed_value_aliases")
                )
            ),
            proposed_capabilities=tuple(
                _decode_capability(item, name=f"proposed_capabilities[{index}]")
                for index, item in enumerate(_array(root["proposed_capabilities"], name="caps"))
            ),
            scope_reviews=tuple(
                _decode_scope_review(item, name=f"scope_reviews[{index}]")
                for index, item in enumerate(_array(root["scope_reviews"], name="scopes"))
            ),
            budget_safety_rows=tuple(
                _decode_budget_row(item, name=f"budget_safety_rows[{index}]")
                for index, item in enumerate(_array(root["budget_safety_rows"], name="budgets"))
            ),
            recommendation_summary=_string(
                root["recommendation_summary"], name="recommendation_summary"
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateBReviewCodecError):
            raise
        raise GateBReviewCodecError(f"invalid Gate-B price proposal: {error}") from error


def encode_public_target_price_audit(audit: PublicTargetPriceAudit) -> bytes:
    """Encode the public-target safety audit as canonical bytes."""

    if type(audit) is not PublicTargetPriceAudit:
        raise TypeError("public-target encoder requires PublicTargetPriceAudit")
    return canonical_json_bytes(audit)


def decode_public_target_price_audit(data: bytes) -> PublicTargetPriceAudit:
    """Strictly decode canonical public-target safety audit bytes."""

    root = _object(
        _load_canonical_json(data, name="PublicTargetPriceAudit"),
        fields={
            "schema",
            "public_set_id",
            "catalog_id",
            "product_facet_index_id",
            "target_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
            "compatible_budget_safe_retained_count",
            "unsafe_satisfied_only_retained_count",
            "scenario_summaries",
            "records",
        },
        name="PublicTargetPriceAudit",
    )
    try:
        _require_schema(root["schema"], GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA, name="public audit")
        return PublicTargetPriceAudit(
            schema=GATE_B_PUBLIC_TARGET_AUDIT_SCHEMA,
            public_set_id=_string(root["public_set_id"], name="public_set_id"),
            catalog_id=_string(root["catalog_id"], name="catalog_id"),
            product_facet_index_id=_string(
                root["product_facet_index_id"], name="product_facet_index_id"
            ),
            target_count=_nonnegative_int(root["target_count"], name="target_count"),
            known_count=_nonnegative_int(root["known_count"], name="known_count"),
            unknown_count=_nonnegative_int(root["unknown_count"], name="unknown_count"),
            conflict_count=_nonnegative_int(root["conflict_count"], name="conflict_count"),
            not_applicable_count=_nonnegative_int(
                root["not_applicable_count"], name="not_applicable_count"
            ),
            compatible_budget_safe_retained_count=_nonnegative_int(
                root["compatible_budget_safe_retained_count"],
                name="compatible_budget_safe_retained_count",
            ),
            unsafe_satisfied_only_retained_count=_nonnegative_int(
                root["unsafe_satisfied_only_retained_count"],
                name="unsafe_satisfied_only_retained_count",
            ),
            scenario_summaries=tuple(
                _decode_scenario_summary(item, name=f"scenario_summaries[{index}]")
                for index, item in enumerate(
                    _array(root["scenario_summaries"], name="scenario_summaries")
                )
            ),
            records=tuple(
                _decode_public_record(item, name=f"records[{index}]")
                for index, item in enumerate(_array(root["records"], name="records"))
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateBReviewCodecError):
            raise
        raise GateBReviewCodecError(f"invalid public-target price audit: {error}") from error


def gate_b_review_candidate_document(
    proposal: GateBPriceReviewProposal,
    audit: PublicTargetPriceAudit,
) -> dict[str, object]:
    """Return compact metadata that explicitly cannot confer runtime authority."""

    if type(proposal) is not GateBPriceReviewProposal or type(audit) is not PublicTargetPriceAudit:
        raise TypeError("Gate-B candidate document requires exact proposal and audit types")
    return {
        "schema": "shopping-copilot/gate-b-review-candidate/v0",
        "builder_version": proposal.builder_version,
        "review_state": proposal.review_state.value,
        "catalog_id": proposal.catalog_id,
        "category_registry_id": proposal.category_registry_id,
        "facet_schema_id": proposal.facet_schema_id,
        "facet_applicability_id": proposal.facet_applicability_id,
        "product_facet_index_id": proposal.product_facet_index_id,
        "catalog_facet_stats_id": proposal.catalog_facet_stats_id,
        "public_set_id": audit.public_set_id,
        "public_target_audit_id": content_id_for_value(audit),
        "price_review_proposal_id": content_id_for_value(proposal),
        "proposed_intent_normalizer_id": PRICE_INTENT_NORMALIZER_ID,
        "reviewed_value_alias_count": len(proposal.reviewed_value_aliases),
        "proposed_exact_scope_count": len(proposal.proposed_capabilities),
        "runtime_capability_published": False,
        "source_controlled_approval_present": False,
    }


def _decode_capability(value: object, *, name: str) -> EffectiveFacetCapability:
    item = _object(
        value,
        fields={
            "facet_id",
            "category_scope_id",
            "decision",
            "resolution_policy_id",
            "intent_committable",
            "retrieval_eligible",
            "probe_eligible",
            "clarification_eligible",
        },
        name=name,
    )
    return EffectiveFacetCapability(
        facet_id=_string(item["facet_id"], name=f"{name}.facet_id"),
        category_scope_id=_string(item["category_scope_id"], name=f"{name}.category_scope_id"),
        decision=RuntimePromotionDecision(_string(item["decision"], name=f"{name}.decision")),
        resolution_policy_id=_resolution_policy(item["resolution_policy_id"]),
        intent_committable=_boolean(item["intent_committable"], name=f"{name}.intent_committable"),
        retrieval_eligible=_boolean(item["retrieval_eligible"], name=f"{name}.retrieval_eligible"),
        probe_eligible=_boolean(item["probe_eligible"], name=f"{name}.probe_eligible"),
        clarification_eligible=_boolean(
            item["clarification_eligible"], name=f"{name}.clarification_eligible"
        ),
    )


def _decode_scope_review(value: object, *, name: str) -> ScopePriceReview:
    item = _object(
        value,
        fields={
            "facet_id",
            "category_scope_id",
            "scope_label",
            "scope_product_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
            "exact_interval_count",
            "lower_bound_interval_count",
            "public_target_count",
            "public_target_known_count",
            "minimum_lower_cents",
            "median_lower_cents",
            "p90_lower_cents",
            "maximum_lower_cents",
            "proposed_capability",
        },
        name=name,
    )
    return ScopePriceReview(
        facet_id=_string(item["facet_id"], name=f"{name}.facet_id"),
        category_scope_id=_string(item["category_scope_id"], name=f"{name}.category_scope_id"),
        scope_label=_string(item["scope_label"], name=f"{name}.scope_label"),
        scope_product_count=_nonnegative_int(
            item["scope_product_count"], name=f"{name}.scope_product_count"
        ),
        known_count=_nonnegative_int(item["known_count"], name=f"{name}.known_count"),
        unknown_count=_nonnegative_int(item["unknown_count"], name=f"{name}.unknown_count"),
        conflict_count=_nonnegative_int(item["conflict_count"], name=f"{name}.conflict_count"),
        not_applicable_count=_nonnegative_int(
            item["not_applicable_count"], name=f"{name}.not_applicable_count"
        ),
        exact_interval_count=_nonnegative_int(
            item["exact_interval_count"], name=f"{name}.exact_interval_count"
        ),
        lower_bound_interval_count=_nonnegative_int(
            item["lower_bound_interval_count"], name=f"{name}.lower_bound_interval_count"
        ),
        public_target_count=_nonnegative_int(
            item["public_target_count"], name=f"{name}.public_target_count"
        ),
        public_target_known_count=_nonnegative_int(
            item["public_target_known_count"], name=f"{name}.public_target_known_count"
        ),
        minimum_lower_cents=_optional_nonnegative_int(
            item["minimum_lower_cents"], name=f"{name}.minimum_lower_cents"
        ),
        median_lower_cents=_optional_nonnegative_int(
            item["median_lower_cents"], name=f"{name}.median_lower_cents"
        ),
        p90_lower_cents=_optional_nonnegative_int(
            item["p90_lower_cents"], name=f"{name}.p90_lower_cents"
        ),
        maximum_lower_cents=_optional_nonnegative_int(
            item["maximum_lower_cents"], name=f"{name}.maximum_lower_cents"
        ),
        proposed_capability=_decode_capability(
            item["proposed_capability"], name=f"{name}.proposed_capability"
        ),
    )


def _decode_budget_row(value: object, *, name: str) -> BudgetSafetyAuditRow:
    item = _object(
        value,
        fields={
            "budget_cents",
            "product_count",
            "satisfied_count",
            "violated_count",
            "unknown_count",
            "not_applicable_count",
            "safe_retained_count",
            "unsafe_satisfied_only_retained_count",
        },
        name=name,
    )
    return BudgetSafetyAuditRow(
        **{field: _nonnegative_int(item[field], name=f"{name}.{field}") for field in item}
    )


def _decode_scenario_summary(value: object, *, name: str) -> PublicScenarioPriceSummary:
    item = _object(
        value,
        fields={
            "scenario_type",
            "target_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
        },
        name=name,
    )
    return PublicScenarioPriceSummary(
        scenario_type=_string(item["scenario_type"], name=f"{name}.scenario_type"),
        target_count=_nonnegative_int(item["target_count"], name=f"{name}.target_count"),
        known_count=_nonnegative_int(item["known_count"], name=f"{name}.known_count"),
        unknown_count=_nonnegative_int(item["unknown_count"], name=f"{name}.unknown_count"),
        conflict_count=_nonnegative_int(item["conflict_count"], name=f"{name}.conflict_count"),
        not_applicable_count=_nonnegative_int(
            item["not_applicable_count"], name=f"{name}.not_applicable_count"
        ),
    )


def _decode_public_record(value: object, *, name: str) -> PublicTargetPriceRecord:
    item = _object(
        value,
        fields={
            "sample_id",
            "scenario_type",
            "parent_asin",
            "price_status",
            "evidence_status",
            "price_value",
            "compatible_budget_cents",
            "safe_match_result",
            "safe_retained",
            "unsafe_satisfied_only_retained",
        },
        name=name,
    )
    evidence_raw = item["evidence_status"]
    return PublicTargetPriceRecord(
        sample_id=_string(item["sample_id"], name=f"{name}.sample_id"),
        scenario_type=_string(item["scenario_type"], name=f"{name}.scenario_type"),
        parent_asin=_string(item["parent_asin"], name=f"{name}.parent_asin"),
        price_status=ProductFacetStatus(_string(item["price_status"], name=f"{name}.price_status")),
        evidence_status=(
            None
            if evidence_raw is None
            else EvidenceStatus(_string(evidence_raw, name=f"{name}.evidence_status"))
        ),
        price_value=_decode_optional_price(item["price_value"], name=f"{name}.price_value"),
        compatible_budget_cents=_optional_nonnegative_int(
            item["compatible_budget_cents"], name=f"{name}.compatible_budget_cents"
        ),
        safe_match_result=FacetMatchResult(
            _string(item["safe_match_result"], name=f"{name}.safe_match_result")
        ),
        safe_retained=_boolean(item["safe_retained"], name=f"{name}.safe_retained"),
        unsafe_satisfied_only_retained=_boolean(
            item["unsafe_satisfied_only_retained"],
            name=f"{name}.unsafe_satisfied_only_retained",
        ),
    )


def _decode_optional_price(value: object, *, name: str) -> NumericValue | None:
    if value is None:
        return None
    item = _object(
        value,
        fields={"kind", "lower", "lower_inclusive", "upper", "upper_inclusive", "unit"},
        name=name,
    )
    if item["kind"] != "numeric":
        raise GateBReviewCodecError(f"{name}.kind must be numeric")
    return NumericValue(
        kind="numeric",
        lower=_optional_nonnegative_int(item["lower"], name=f"{name}.lower"),
        lower_inclusive=_boolean(item["lower_inclusive"], name=f"{name}.lower_inclusive"),
        upper=_optional_nonnegative_int(item["upper"], name=f"{name}.upper"),
        upper_inclusive=_boolean(item["upper_inclusive"], name=f"{name}.upper_inclusive"),
        unit=_string(item["unit"], name=f"{name}.unit"),
    )


def _load_canonical_json(data: bytes, *, name: str) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise GateBReviewCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        canonical = canonical_json_bytes(parsed)
    except _DuplicateJsonKeyError as error:
        raise GateBReviewCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        if isinstance(error, GateBReviewCodecError):
            raise
        raise GateBReviewCodecError(f"{name} is not valid contract JSON") from error
    if data != canonical:
        raise GateBReviewCodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GateBReviewCodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        raise GateBReviewCodecError(f"{name} has invalid fields")
    return result


def _array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise GateBReviewCodecError(f"{name} must be an array")
    return cast(list[object], value)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise GateBReviewCodecError(f"{name} must be a string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise GateBReviewCodecError(f"{name} must be boolean")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise GateBReviewCodecError(f"{name} must be a non-negative I-JSON integer")
    return value


def _optional_nonnegative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name=name)


def _require_schema(value: object, expected: str, *, name: str) -> None:
    result = _string(value, name=f"{name}.schema")
    if result != expected:
        raise GateBReviewCodecError(f"{name}.schema is unsupported")


def _resolution_policy(value: object) -> Literal["structured_resolution_v1"]:
    result = _string(value, name="resolution_policy_id")
    if result != RESOLUTION_POLICY_ID:
        raise GateBReviewCodecError("resolution_policy_id is unsupported")
    return result


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite JSON token: {raw}")
