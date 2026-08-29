"""Strict canonical codecs for CS3 evidence, index, statistics, and audit artifacts."""

from __future__ import annotations

import json
from collections import Counter
from typing import Literal, cast

from ..canonical import IJSON_SAFE_INTEGER_MAX, canonical_json_bytes, content_id_for_value
from ..errors import ResolutionCodecError
from .gate_a_models import (
    BooleanValue,
    CategoricalValue,
    EvidenceStatus,
    NumericValue,
    ProductFacetStatus,
    ResolvedFacetValue,
    TextValue,
    ValueCompleteness,
)
from .resolution_models import (
    CATALOG_FACET_STATS_SCHEMA,
    CATALOG_READ_ONLY_AUDIT_SCHEMA,
    FACET_EVIDENCE_STORE_SCHEMA,
    PRODUCT_FACET_INDEX_SCHEMA,
    RESOLUTION_POLICY_ID,
    CatalogFacetStatsArtifact,
    CatalogReadOnlyAudit,
    FacetEvidenceStore,
    FacetScopeCatalogStats,
    FacetValueEvidence,
    ProductFacetIndex,
    ResolutionCandidateBuild,
    ResolvedProductFacetValue,
    ResolvedValueCount,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def encode_facet_evidence_store(store: FacetEvidenceStore) -> bytes:
    """Encode one immutable evidence store as canonical artifact bytes."""

    if type(store) is not FacetEvidenceStore:
        raise TypeError("FacetEvidenceStore encoder requires the exact contract type")
    return canonical_json_bytes(store)


def decode_facet_evidence_store(data: bytes) -> FacetEvidenceStore:
    """Strictly decode canonical FacetEvidenceStore bytes."""

    document = _load_canonical_json(data, name="FacetEvidenceStore")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "product_category_assignment_id",
                "facet_applicability_id",
                "facet_source_bindings_id",
                "resolution_policy_id",
                "evidence",
            },
            name="FacetEvidenceStore",
        )
        if root["schema"] != FACET_EVIDENCE_STORE_SCHEMA:
            raise ResolutionCodecError("FacetEvidenceStore.schema is invalid")
        return FacetEvidenceStore(
            schema=FACET_EVIDENCE_STORE_SCHEMA,
            catalog_id=_expect_string(root["catalog_id"], name="FacetEvidenceStore.catalog_id"),
            product_category_assignment_id=_expect_string(
                root["product_category_assignment_id"],
                name="FacetEvidenceStore.product_category_assignment_id",
            ),
            facet_applicability_id=_expect_string(
                root["facet_applicability_id"],
                name="FacetEvidenceStore.facet_applicability_id",
            ),
            facet_source_bindings_id=_expect_string(
                root["facet_source_bindings_id"],
                name="FacetEvidenceStore.facet_source_bindings_id",
            ),
            resolution_policy_id=_decode_policy(root["resolution_policy_id"]),
            evidence=tuple(
                _decode_evidence(item, index=index)
                for index, item in enumerate(
                    _expect_array(root["evidence"], name="FacetEvidenceStore.evidence")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ResolutionCodecError):
            raise
        raise ResolutionCodecError(f"invalid FacetEvidenceStore: {error}") from error


def encode_product_facet_index(index: ProductFacetIndex) -> bytes:
    """Encode one sparse ProductFacetIndex as canonical artifact bytes."""

    if type(index) is not ProductFacetIndex:
        raise TypeError("ProductFacetIndex encoder requires the exact contract type")
    return canonical_json_bytes(index)


def decode_product_facet_index(data: bytes) -> ProductFacetIndex:
    """Strictly decode canonical ProductFacetIndex bytes."""

    document = _load_canonical_json(data, name="ProductFacetIndex")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "product_category_assignment_id",
                "facet_applicability_id",
                "facet_source_bindings_id",
                "facet_evidence_store_id",
                "resolution_policy_id",
                "entries",
            },
            name="ProductFacetIndex",
        )
        if root["schema"] != PRODUCT_FACET_INDEX_SCHEMA:
            raise ResolutionCodecError("ProductFacetIndex.schema is invalid")
        return ProductFacetIndex(
            schema=PRODUCT_FACET_INDEX_SCHEMA,
            catalog_id=_expect_string(root["catalog_id"], name="ProductFacetIndex.catalog_id"),
            product_category_assignment_id=_expect_string(
                root["product_category_assignment_id"],
                name="ProductFacetIndex.product_category_assignment_id",
            ),
            facet_applicability_id=_expect_string(
                root["facet_applicability_id"],
                name="ProductFacetIndex.facet_applicability_id",
            ),
            facet_source_bindings_id=_expect_string(
                root["facet_source_bindings_id"],
                name="ProductFacetIndex.facet_source_bindings_id",
            ),
            facet_evidence_store_id=_expect_string(
                root["facet_evidence_store_id"],
                name="ProductFacetIndex.facet_evidence_store_id",
            ),
            resolution_policy_id=_decode_policy(root["resolution_policy_id"]),
            entries=tuple(
                _decode_index_entry(item, index=index)
                for index, item in enumerate(
                    _expect_array(root["entries"], name="ProductFacetIndex.entries")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ResolutionCodecError):
            raise
        raise ResolutionCodecError(f"invalid ProductFacetIndex: {error}") from error


def encode_catalog_facet_stats(stats: CatalogFacetStatsArtifact) -> bytes:
    """Encode one complete resolved-statistics artifact as canonical bytes."""

    if type(stats) is not CatalogFacetStatsArtifact:
        raise TypeError("CatalogFacetStatsArtifact encoder requires the exact contract type")
    return canonical_json_bytes(stats)


def decode_catalog_facet_stats(data: bytes) -> CatalogFacetStatsArtifact:
    """Strictly decode canonical CatalogFacetStatsArtifact bytes."""

    document = _load_canonical_json(data, name="CatalogFacetStatsArtifact")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "category_registry_id",
                "product_category_assignment_id",
                "facet_schema_id",
                "facet_applicability_id",
                "product_facet_index_id",
                "resolution_policy_id",
                "rows",
            },
            name="CatalogFacetStatsArtifact",
        )
        if root["schema"] != CATALOG_FACET_STATS_SCHEMA:
            raise ResolutionCodecError("CatalogFacetStatsArtifact.schema is invalid")
        return CatalogFacetStatsArtifact(
            schema=CATALOG_FACET_STATS_SCHEMA,
            catalog_id=_expect_string(
                root["catalog_id"], name="CatalogFacetStatsArtifact.catalog_id"
            ),
            category_registry_id=_expect_string(
                root["category_registry_id"],
                name="CatalogFacetStatsArtifact.category_registry_id",
            ),
            product_category_assignment_id=_expect_string(
                root["product_category_assignment_id"],
                name="CatalogFacetStatsArtifact.product_category_assignment_id",
            ),
            facet_schema_id=_expect_string(
                root["facet_schema_id"], name="CatalogFacetStatsArtifact.facet_schema_id"
            ),
            facet_applicability_id=_expect_string(
                root["facet_applicability_id"],
                name="CatalogFacetStatsArtifact.facet_applicability_id",
            ),
            product_facet_index_id=_expect_string(
                root["product_facet_index_id"],
                name="CatalogFacetStatsArtifact.product_facet_index_id",
            ),
            resolution_policy_id=_decode_policy(root["resolution_policy_id"]),
            rows=tuple(
                _decode_stats_row(item, index=index)
                for index, item in enumerate(
                    _expect_array(root["rows"], name="CatalogFacetStatsArtifact.rows")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ResolutionCodecError):
            raise
        raise ResolutionCodecError(f"invalid CatalogFacetStatsArtifact: {error}") from error


def decode_catalog_read_only_audit(data: bytes) -> CatalogReadOnlyAudit:
    """Strictly decode the deterministic catalog read-only audit artifact."""

    document = _load_canonical_json(data, name="CatalogReadOnlyAudit")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id_before",
                "catalog_id_after_staging",
                "byte_size_before",
                "byte_size_after_staging",
                "unchanged",
                "output_is_separate",
            },
            name="CatalogReadOnlyAudit",
        )
        if root["schema"] != CATALOG_READ_ONLY_AUDIT_SCHEMA:
            raise ResolutionCodecError("CatalogReadOnlyAudit.schema is invalid")
        return CatalogReadOnlyAudit(
            schema=CATALOG_READ_ONLY_AUDIT_SCHEMA,
            catalog_id_before=_expect_string(
                root["catalog_id_before"], name="CatalogReadOnlyAudit.catalog_id_before"
            ),
            catalog_id_after_staging=_expect_string(
                root["catalog_id_after_staging"],
                name="CatalogReadOnlyAudit.catalog_id_after_staging",
            ),
            byte_size_before=_expect_nonnegative_int(
                root["byte_size_before"], name="CatalogReadOnlyAudit.byte_size_before"
            ),
            byte_size_after_staging=_expect_nonnegative_int(
                root["byte_size_after_staging"],
                name="CatalogReadOnlyAudit.byte_size_after_staging",
            ),
            unchanged=_expect_bool(root["unchanged"], name="CatalogReadOnlyAudit.unchanged"),
            output_is_separate=_expect_bool(
                root["output_is_separate"],
                name="CatalogReadOnlyAudit.output_is_separate",
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ResolutionCodecError):
            raise
        raise ResolutionCodecError(f"invalid CatalogReadOnlyAudit: {error}") from error


def resolution_candidate_document(build: ResolutionCandidateBuild) -> dict[str, object]:
    """Return compact CS3 metadata without duplicating large artifact payloads."""

    if type(build) is not ResolutionCandidateBuild:
        raise TypeError("resolution_candidate_document requires ResolutionCandidateBuild")
    evidence_statuses = Counter(item.status.value for item in build.evidence_store.evidence)
    index_statuses = Counter(item.status.value for item in build.product_facet_index.entries)
    return {
        "schema": build.schema,
        "builder_version": build.builder_version,
        "catalog_id": build.evidence_store.catalog_id,
        "category_registry_id": build.category_registry_id,
        "product_category_assignment_id": build.evidence_store.product_category_assignment_id,
        "facet_schema_id": build.facet_schema_id,
        "facet_applicability_id": build.evidence_store.facet_applicability_id,
        "facet_source_bindings_id": build.evidence_store.facet_source_bindings_id,
        "gate_a_selection_id": build.gate_a_selection_id,
        "facet_evidence_store_id": content_id_for_value(build.evidence_store),
        "product_facet_index_id": content_id_for_value(build.product_facet_index),
        "catalog_facet_stats_id": content_id_for_value(build.stats),
        "resolution_policy_id": RESOLUTION_POLICY_ID,
        "evidence_count": len(build.evidence_store.evidence),
        "evidence_status_counts": {
            status.value: evidence_statuses[status.value] for status in EvidenceStatus
        },
        "stored_index_entry_count": len(build.product_facet_index.entries),
        "stored_index_status_counts": {
            status.value: index_statuses[status.value]
            for status in (ProductFacetStatus.KNOWN, ProductFacetStatus.CONFLICT)
        },
        "stats_row_count": len(build.stats.rows),
        "gate_b_runtime_approved": False,
    }


def _decode_evidence(value: object, *, index: int) -> FacetValueEvidence:
    name = f"FacetEvidenceStore.evidence[{index}]"
    item = _expect_object(
        value,
        fields={
            "id",
            "parent_asin",
            "facet_id",
            "binding_id",
            "status",
            "raw_value_json",
            "canonical_value",
        },
        name=name,
    )
    return FacetValueEvidence(
        id=_expect_string(item["id"], name=f"{name}.id"),
        parent_asin=_expect_string(item["parent_asin"], name=f"{name}.parent_asin"),
        facet_id=_expect_string(item["facet_id"], name=f"{name}.facet_id"),
        binding_id=_expect_string(item["binding_id"], name=f"{name}.binding_id"),
        status=EvidenceStatus(_expect_string(item["status"], name=f"{name}.status")),
        raw_value_json=_expect_string(item["raw_value_json"], name=f"{name}.raw_value_json"),
        canonical_value=_decode_optional_value(
            item["canonical_value"], name=f"{name}.canonical_value"
        ),
    )


def _decode_index_entry(value: object, *, index: int) -> ResolvedProductFacetValue:
    name = f"ProductFacetIndex.entries[{index}]"
    item = _expect_object(
        value,
        fields={
            "parent_asin",
            "facet_id",
            "status",
            "value",
            "evidence_ids",
            "resolution_policy_id",
        },
        name=name,
    )
    return ResolvedProductFacetValue(
        parent_asin=_expect_string(item["parent_asin"], name=f"{name}.parent_asin"),
        facet_id=_expect_string(item["facet_id"], name=f"{name}.facet_id"),
        status=ProductFacetStatus(_expect_string(item["status"], name=f"{name}.status")),
        value=_decode_optional_value(item["value"], name=f"{name}.value"),
        evidence_ids=_string_tuple(item["evidence_ids"], name=f"{name}.evidence_ids"),
        resolution_policy_id=_decode_policy(item["resolution_policy_id"]),
    )


def _decode_stats_row(value: object, *, index: int) -> FacetScopeCatalogStats:
    name = f"CatalogFacetStatsArtifact.rows[{index}]"
    item = _expect_object(
        value,
        fields={
            "facet_id",
            "category_scope_id",
            "scope_product_count",
            "known_count",
            "unknown_count",
            "conflict_count",
            "not_applicable_count",
            "known_value_counts",
        },
        name=name,
    )
    return FacetScopeCatalogStats(
        facet_id=_expect_string(item["facet_id"], name=f"{name}.facet_id"),
        category_scope_id=_expect_string(
            item["category_scope_id"], name=f"{name}.category_scope_id"
        ),
        scope_product_count=_expect_nonnegative_int(
            item["scope_product_count"], name=f"{name}.scope_product_count"
        ),
        known_count=_expect_nonnegative_int(item["known_count"], name=f"{name}.known_count"),
        unknown_count=_expect_nonnegative_int(item["unknown_count"], name=f"{name}.unknown_count"),
        conflict_count=_expect_nonnegative_int(
            item["conflict_count"], name=f"{name}.conflict_count"
        ),
        not_applicable_count=_expect_nonnegative_int(
            item["not_applicable_count"], name=f"{name}.not_applicable_count"
        ),
        known_value_counts=tuple(
            _decode_value_count(raw, name=f"{name}.known_value_counts[{value_index}]")
            for value_index, raw in enumerate(
                _expect_array(item["known_value_counts"], name=f"{name}.known_value_counts")
            )
        ),
    )


def _decode_value_count(value: object, *, name: str) -> ResolvedValueCount:
    item = _expect_object(
        value,
        fields={"canonical_value_json", "product_count"},
        name=name,
    )
    return ResolvedValueCount(
        canonical_value_json=_expect_string(
            item["canonical_value_json"], name=f"{name}.canonical_value_json"
        ),
        product_count=_expect_nonnegative_int(item["product_count"], name=f"{name}.product_count"),
    )


def _decode_optional_value(value: object, *, name: str) -> ResolvedFacetValue | None:
    if value is None:
        return None
    item = _expect_object_any(value, name=name)
    kind = _expect_string(item.get("kind"), name=f"{name}.kind")
    if kind == "categorical":
        _require_fields(item, {"kind", "values", "completeness"}, name=name)
        return CategoricalValue(
            kind="categorical",
            values=_string_tuple(item["values"], name=f"{name}.values"),
            completeness=ValueCompleteness(
                _expect_string(item["completeness"], name=f"{name}.completeness")
            ),
        )
    if kind == "boolean":
        _require_fields(item, {"kind", "value"}, name=name)
        return BooleanValue(
            kind="boolean",
            value=_expect_bool(item["value"], name=f"{name}.value"),
        )
    if kind == "text":
        _require_fields(item, {"kind", "value"}, name=name)
        return TextValue(
            kind="text",
            value=_expect_string(item["value"], name=f"{name}.value"),
        )
    if kind == "numeric":
        _require_fields(
            item,
            {"kind", "lower", "lower_inclusive", "upper", "upper_inclusive", "unit"},
            name=name,
        )
        return NumericValue(
            kind="numeric",
            lower=_expect_optional_number(item["lower"], name=f"{name}.lower"),
            lower_inclusive=_expect_bool(item["lower_inclusive"], name=f"{name}.lower_inclusive"),
            upper=_expect_optional_number(item["upper"], name=f"{name}.upper"),
            upper_inclusive=_expect_bool(item["upper_inclusive"], name=f"{name}.upper_inclusive"),
            unit=_expect_string(item["unit"], name=f"{name}.unit"),
        )
    raise ResolutionCodecError(f"{name}.kind is unsupported")


def _load_canonical_json(data: bytes, *, name: str) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ResolutionCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        canonical = canonical_json_bytes(parsed)
    except _DuplicateJsonKeyError as error:
        raise ResolutionCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        if isinstance(error, ResolutionCodecError):
            raise
        raise ResolutionCodecError(f"{name} is not valid contract JSON") from error
    if data != canonical:
        raise ResolutionCodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _decode_policy(value: object) -> Literal["structured_resolution_v1"]:
    policy = _expect_string(value, name="resolution_policy_id")
    if policy != RESOLUTION_POLICY_ID:
        raise ResolutionCodecError("resolution_policy_id is unsupported")
    return RESOLUTION_POLICY_ID


def _expect_object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    result = _expect_object_any(value, name=name)
    _require_fields(result, fields, name=name)
    return result


def _expect_object_any(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ResolutionCodecError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _require_fields(value: dict[str, object], fields: set[str], *, name: str) -> None:
    if set(value) != fields:
        raise ResolutionCodecError(f"{name} has invalid fields")


def _expect_array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ResolutionCodecError(f"{name} must be an array")
    return cast(list[object], value)


def _expect_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ResolutionCodecError(f"{name} must be a string")
    return value


def _expect_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ResolutionCodecError(f"{name} must be boolean")
    return value


def _expect_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise ResolutionCodecError(f"{name} must be a non-negative I-JSON integer")
    return value


def _expect_optional_number(value: object, *, name: str) -> int | float | None:
    if value is None or type(value) in (int, float):
        return cast(int | float | None, value)
    raise ResolutionCodecError(f"{name} must be a number or null")


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(
        _expect_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(_expect_array(value, name=name))
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
