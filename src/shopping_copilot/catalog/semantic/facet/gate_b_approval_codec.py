"""Codecs for owner-approved Gate-B input and effective capability artifacts."""

from __future__ import annotations

import json
from typing import Literal, cast

from ..canonical import canonical_json_bytes, content_id_for_value
from ..errors import GateBCodecError
from .gate_b_models import (
    EFFECTIVE_FACET_CAPABILITIES_SCHEMA,
    GATE_B_BUILDER_VERSION,
    GATE_B_SELECTION_SCHEMA,
    EffectiveFacetCapability,
    EffectiveFacetCapabilitySet,
    GateBCandidateBuild,
    GateBSelection,
    RuntimePromotionDecision,
)
from .resolution_models import RESOLUTION_POLICY_ID


class _DuplicateJsonKeyError(ValueError):
    pass


def decode_gate_b_selection(data: bytes) -> GateBSelection:
    """Decode one human-authored approval document with no unknown fields."""

    root = _object(
        _load_json(data, name="Gate-B selection", require_canonical=False),
        fields={
            "schema",
            "builder_version",
            "catalog_id",
            "category_registry_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
            "catalog_facet_stats_id",
            "gate_b_review_proposal_id",
            "public_target_audit_id",
            "resolution_policy_id",
            "intent_value_normalizer_id",
            "reviewed_value_aliases",
            "approvals",
            "rationale",
        },
        name="Gate-B selection",
    )
    try:
        if root["schema"] != GATE_B_SELECTION_SCHEMA:
            raise GateBCodecError("Gate-B selection schema is invalid")
        if root["builder_version"] != GATE_B_BUILDER_VERSION:
            raise GateBCodecError("Gate-B selection builder version is unsupported")
        return GateBSelection(
            schema=GATE_B_SELECTION_SCHEMA,
            builder_version=GATE_B_BUILDER_VERSION,
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
            gate_b_review_proposal_id=_string(
                root["gate_b_review_proposal_id"], name="gate_b_review_proposal_id"
            ),
            public_target_audit_id=_string(
                root["public_target_audit_id"], name="public_target_audit_id"
            ),
            resolution_policy_id=_policy(root["resolution_policy_id"]),
            intent_value_normalizer_id=_string(
                root["intent_value_normalizer_id"], name="intent_value_normalizer_id"
            ),
            reviewed_value_aliases=tuple(
                _string(item, name=f"reviewed_value_aliases[{index}]")
                for index, item in enumerate(
                    _array(root["reviewed_value_aliases"], name="reviewed_value_aliases")
                )
            ),
            approvals=tuple(
                _decode_capability(item, name=f"approvals[{index}]")
                for index, item in enumerate(_array(root["approvals"], name="approvals"))
            ),
            rationale=_string(root["rationale"], name="rationale"),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateBCodecError):
            raise
        raise GateBCodecError(f"invalid Gate-B selection: {error}") from error


def encode_effective_facet_capabilities(capabilities: EffectiveFacetCapabilitySet) -> bytes:
    """Encode the normative exact-scope capability set as canonical bytes."""

    if type(capabilities) is not EffectiveFacetCapabilitySet:
        raise TypeError("capability encoder requires EffectiveFacetCapabilitySet")
    return canonical_json_bytes(capabilities)


def decode_effective_facet_capabilities(data: bytes) -> EffectiveFacetCapabilitySet:
    """Strictly decode canonical effective capability artifact bytes."""

    root = _object(
        _load_json(data, name="EffectiveFacetCapabilitySet", require_canonical=True),
        fields={
            "schema",
            "category_registry_id",
            "facet_schema_id",
            "facet_applicability_id",
            "product_facet_index_id",
            "resolution_policy_id",
            "entries",
        },
        name="EffectiveFacetCapabilitySet",
    )
    try:
        if root["schema"] != EFFECTIVE_FACET_CAPABILITIES_SCHEMA:
            raise GateBCodecError("EffectiveFacetCapabilitySet.schema is invalid")
        return EffectiveFacetCapabilitySet(
            schema=EFFECTIVE_FACET_CAPABILITIES_SCHEMA,
            category_registry_id=_string(root["category_registry_id"], name="category_registry_id"),
            facet_schema_id=_string(root["facet_schema_id"], name="facet_schema_id"),
            facet_applicability_id=_string(
                root["facet_applicability_id"], name="facet_applicability_id"
            ),
            product_facet_index_id=_string(
                root["product_facet_index_id"], name="product_facet_index_id"
            ),
            resolution_policy_id=_policy(root["resolution_policy_id"]),
            entries=tuple(
                _decode_capability(item, name=f"entries[{index}]")
                for index, item in enumerate(_array(root["entries"], name="entries"))
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateBCodecError):
            raise
        raise GateBCodecError(f"invalid EffectiveFacetCapabilitySet: {error}") from error


def gate_b_candidate_document(build: GateBCandidateBuild) -> dict[str, object]:
    """Return compact approved Gate-B candidate metadata."""

    if type(build) is not GateBCandidateBuild:
        raise TypeError("candidate document requires GateBCandidateBuild")
    return {
        "schema": build.schema,
        "builder_version": build.builder_version,
        "catalog_id": build.catalog_id,
        "catalog_facet_stats_id": build.catalog_facet_stats_id,
        "gate_b_review_proposal_id": build.gate_b_review_proposal_id,
        "public_target_audit_id": build.public_target_audit_id,
        "gate_b_selection_id": content_id_for_value(build.selection),
        "effective_facet_capabilities_id": content_id_for_value(build.capabilities),
        "approved_exact_scope_count": len(build.capabilities.entries),
        "owner_approval_recorded": True,
        "runtime_integration_complete": False,
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
        resolution_policy_id=_policy(item["resolution_policy_id"]),
        intent_committable=_boolean(item["intent_committable"], name=f"{name}.intent_committable"),
        retrieval_eligible=_boolean(item["retrieval_eligible"], name=f"{name}.retrieval_eligible"),
        probe_eligible=_boolean(item["probe_eligible"], name=f"{name}.probe_eligible"),
        clarification_eligible=_boolean(
            item["clarification_eligible"], name=f"{name}.clarification_eligible"
        ),
    )


def _load_json(data: bytes, *, name: str, require_canonical: bool) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise GateBCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError as error:
        raise GateBCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GateBCodecError(f"{name} is not valid JSON") from error
    if require_canonical and data != canonical_json_bytes(parsed):
        raise GateBCodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GateBCodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        raise GateBCodecError(f"{name} has invalid fields")
    return result


def _array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise GateBCodecError(f"{name} must be an array")
    return cast(list[object], value)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise GateBCodecError(f"{name} must be a string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise GateBCodecError(f"{name} must be boolean")
    return value


def _policy(value: object) -> Literal["structured_resolution_v1"]:
    result = _string(value, name="resolution_policy_id")
    if result != RESOLUTION_POLICY_ID:
        raise GateBCodecError("resolution_policy_id is unsupported")
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
