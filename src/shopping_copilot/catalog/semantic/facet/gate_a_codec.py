"""Strict codec for source-controlled Gate-A decisions."""

from __future__ import annotations

import json
from typing import cast

from ..canonical import canonical_json_bytes, content_id_for_value
from ..errors import GateACodecError
from .gate_a_models import (
    CATALOG_FACET_SCHEMA,
    FACET_APPLICABILITY_SCHEMA,
    FACET_SOURCE_BINDINGS_SCHEMA,
    GATE_A_BUILDER_VERSION,
    GATE_A_SELECTION_SCHEMA,
    CatalogFacetDefinition,
    CatalogFacetSchema,
    FacetApplicability,
    FacetApplicabilitySet,
    FacetDataType,
    FacetSourceBinding,
    FacetSourceBindingSet,
    GateACandidateBuild,
    GateADecision,
    GateAFacetApproval,
    GateASelection,
    ItemCardinality,
    PriceExtractionAudit,
    PriceExtractionExpectation,
    ValueCompleteness,
)
from .models import SourceKind, SourceLocator


class _DuplicateJsonKeyError(ValueError):
    pass


def decode_gate_a_selection(data: bytes) -> GateASelection:
    """Decode one human-authored Gate-A selection with no unknown fields."""

    document = _load_json(data, name="Gate-A selection")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "category_registry_id",
                "product_category_assignment_id",
                "source_profile_manifest_sha256",
                "builder_version",
                "approvals",
            },
            name="Gate-A selection",
        )
        if root["schema"] != GATE_A_SELECTION_SCHEMA:
            raise GateACodecError("Gate-A selection schema is invalid")
        if root["builder_version"] != GATE_A_BUILDER_VERSION:
            raise GateACodecError("Gate-A selection builder version is unsupported")
        return GateASelection(
            schema=GATE_A_SELECTION_SCHEMA,
            catalog_id=_expect_string(root["catalog_id"], name="selection.catalog_id"),
            category_registry_id=_expect_string(
                root["category_registry_id"],
                name="selection.category_registry_id",
            ),
            product_category_assignment_id=_expect_string(
                root["product_category_assignment_id"],
                name="selection.product_category_assignment_id",
            ),
            source_profile_manifest_sha256=_expect_string(
                root["source_profile_manifest_sha256"],
                name="selection.source_profile_manifest_sha256",
            ),
            builder_version=GATE_A_BUILDER_VERSION,
            approvals=tuple(
                _decode_approval(item, index=index)
                for index, item in enumerate(
                    _expect_array(root["approvals"], name="selection.approvals")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateACodecError):
            raise
        raise GateACodecError(f"invalid Gate-A selection: {error}") from error


def decode_catalog_facet_schema(data: bytes) -> CatalogFacetSchema:
    """Strictly decode one canonical CatalogFacetSchema artifact."""

    document = _load_json(data, name="CatalogFacetSchema", require_canonical=True)
    try:
        root = _expect_object(
            document,
            fields={"schema", "facets"},
            name="CatalogFacetSchema",
        )
        if root["schema"] != CATALOG_FACET_SCHEMA:
            raise GateACodecError("CatalogFacetSchema.schema is invalid")
        return CatalogFacetSchema(
            schema=CATALOG_FACET_SCHEMA,
            facets=tuple(
                _decode_definition(item, name=f"CatalogFacetSchema.facets[{index}]")
                for index, item in enumerate(
                    _expect_array(root["facets"], name="CatalogFacetSchema.facets")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateACodecError):
            raise
        raise GateACodecError(f"invalid CatalogFacetSchema: {error}") from error


def decode_facet_applicability_set(data: bytes) -> FacetApplicabilitySet:
    """Strictly decode one canonical FacetApplicabilitySet artifact."""

    document = _load_json(data, name="FacetApplicabilitySet", require_canonical=True)
    try:
        root = _expect_object(
            document,
            fields={"schema", "category_registry_id", "facet_schema_id", "entries"},
            name="FacetApplicabilitySet",
        )
        if root["schema"] != FACET_APPLICABILITY_SCHEMA:
            raise GateACodecError("FacetApplicabilitySet.schema is invalid")
        return FacetApplicabilitySet(
            schema=FACET_APPLICABILITY_SCHEMA,
            category_registry_id=_expect_string(
                root["category_registry_id"],
                name="FacetApplicabilitySet.category_registry_id",
            ),
            facet_schema_id=_expect_string(
                root["facet_schema_id"],
                name="FacetApplicabilitySet.facet_schema_id",
            ),
            entries=tuple(
                _decode_applicability(item, name=f"FacetApplicabilitySet.entries[{index}]")
                for index, item in enumerate(
                    _expect_array(root["entries"], name="FacetApplicabilitySet.entries")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateACodecError):
            raise
        raise GateACodecError(f"invalid FacetApplicabilitySet: {error}") from error


def decode_facet_source_binding_set(data: bytes) -> FacetSourceBindingSet:
    """Strictly decode one canonical FacetSourceBindingSet artifact."""

    document = _load_json(data, name="FacetSourceBindingSet", require_canonical=True)
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "category_registry_id",
                "facet_schema_id",
                "facet_applicability_id",
                "bindings",
            },
            name="FacetSourceBindingSet",
        )
        if root["schema"] != FACET_SOURCE_BINDINGS_SCHEMA:
            raise GateACodecError("FacetSourceBindingSet.schema is invalid")
        return FacetSourceBindingSet(
            schema=FACET_SOURCE_BINDINGS_SCHEMA,
            category_registry_id=_expect_string(
                root["category_registry_id"],
                name="FacetSourceBindingSet.category_registry_id",
            ),
            facet_schema_id=_expect_string(
                root["facet_schema_id"],
                name="FacetSourceBindingSet.facet_schema_id",
            ),
            facet_applicability_id=_expect_string(
                root["facet_applicability_id"],
                name="FacetSourceBindingSet.facet_applicability_id",
            ),
            bindings=tuple(
                _decode_binding(item, name=f"FacetSourceBindingSet.bindings[{index}]")
                for index, item in enumerate(
                    _expect_array(root["bindings"], name="FacetSourceBindingSet.bindings")
                )
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateACodecError):
            raise
        raise GateACodecError(f"invalid FacetSourceBindingSet: {error}") from error


def decode_price_extraction_audits(data: bytes) -> tuple[PriceExtractionAudit, ...]:
    """Strictly decode canonical Gate-A extraction audit rows."""

    document = _load_json(data, name="Gate-A extraction audit", require_canonical=True)
    try:
        return tuple(
            _decode_price_audit(item, index=index)
            for index, item in enumerate(_expect_array(document, name="Gate-A extraction audit"))
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, GateACodecError):
            raise
        raise GateACodecError(f"invalid Gate-A extraction audit: {error}") from error


def gate_a_candidate_document(build: GateACandidateBuild) -> dict[str, object]:
    """Return compact candidate metadata without duplicating artifact payloads."""

    if type(build) is not GateACandidateBuild:
        raise TypeError("gate_a_candidate_document requires GateACandidateBuild")
    return {
        "schema": build.schema,
        "catalog_id": build.catalog_id,
        "category_registry_id": build.category_registry_id,
        "product_category_assignment_id": build.product_category_assignment_id,
        "source_profile_manifest_sha256": build.source_profile_manifest_sha256,
        "builder_version": build.builder_version,
        "gate_a_selection_id": content_id_for_value(build.selection),
        "approved_facet_ids": [item.id for item in build.facet_schema.facets],
        "facet_schema_id": build.applicability.facet_schema_id,
        "facet_applicability_id": build.bindings.facet_applicability_id,
        "binding_count": len(build.bindings.bindings),
    }


def _decode_approval(value: object, *, index: int) -> GateAFacetApproval:
    name = f"selection.approvals[{index}]"
    item = _expect_object(
        value,
        fields={
            "decision",
            "definition",
            "applicability",
            "bindings",
            "extraction_expectation",
            "rationale",
        },
        name=name,
    )
    definition = _decode_definition(item["definition"], name=f"{name}.definition")
    applicability = _decode_applicability(
        item["applicability"],
        name=f"{name}.applicability",
    )
    return GateAFacetApproval(
        decision=GateADecision(_expect_string(item["decision"], name=f"{name}.decision")),
        definition=definition,
        applicability=applicability,
        bindings=tuple(
            _decode_binding(binding, name=f"{name}.bindings[{binding_index}]")
            for binding_index, binding in enumerate(
                _expect_array(item["bindings"], name=f"{name}.bindings")
            )
        ),
        extraction_expectation=_decode_expectation(
            item["extraction_expectation"],
            name=f"{name}.extraction_expectation",
        ),
        rationale=_expect_string(item["rationale"], name=f"{name}.rationale"),
    )


def _decode_definition(value: object, *, name: str) -> CatalogFacetDefinition:
    item = _expect_object(
        value,
        fields={"id", "name", "data_type", "item_cardinality"},
        name=name,
    )
    return CatalogFacetDefinition(
        id=_expect_string(item["id"], name=f"{name}.id"),
        name=_expect_string(item["name"], name=f"{name}.name"),
        data_type=FacetDataType(_expect_string(item["data_type"], name=f"{name}.data_type")),
        item_cardinality=ItemCardinality(
            _expect_string(item["item_cardinality"], name=f"{name}.item_cardinality")
        ),
    )


def _decode_applicability(value: object, *, name: str) -> FacetApplicability:
    item = _expect_object(
        value,
        fields={"facet_id", "category_scope_ids"},
        name=name,
    )
    return FacetApplicability(
        facet_id=_expect_string(item["facet_id"], name=f"{name}.facet_id"),
        category_scope_ids=_string_tuple(
            item["category_scope_ids"],
            name=f"{name}.category_scope_ids",
        ),
    )


def _decode_binding(value: object, *, name: str) -> FacetSourceBinding:
    item = _expect_object(
        value,
        fields={
            "id",
            "facet_id",
            "source",
            "applicable_category_scope_ids",
            "extractor_id",
            "catalog_value_normalizer_id",
            "priority",
            "completeness",
            "resolver_id",
        },
        name=name,
    )
    source = _expect_object(
        item["source"],
        fields={"kind", "key"},
        name=f"{name}.source",
    )
    return FacetSourceBinding(
        id=_expect_string(item["id"], name=f"{name}.id"),
        facet_id=_expect_string(item["facet_id"], name=f"{name}.facet_id"),
        source=SourceLocator(
            kind=SourceKind(_expect_string(source["kind"], name=f"{name}.source.kind")),
            key=_expect_string(source["key"], name=f"{name}.source.key"),
        ),
        applicable_category_scope_ids=_string_tuple(
            item["applicable_category_scope_ids"],
            name=f"{name}.applicable_category_scope_ids",
        ),
        extractor_id=_expect_string(item["extractor_id"], name=f"{name}.extractor_id"),
        catalog_value_normalizer_id=_expect_string(
            item["catalog_value_normalizer_id"],
            name=f"{name}.catalog_value_normalizer_id",
        ),
        priority=_expect_int(item["priority"], name=f"{name}.priority"),
        completeness=ValueCompleteness(
            _expect_string(item["completeness"], name=f"{name}.completeness")
        ),
        resolver_id=_expect_string(item["resolver_id"], name=f"{name}.resolver_id"),
    )


def _decode_expectation(value: object, *, name: str) -> PriceExtractionExpectation:
    fields = {
        "product_count",
        "source_present_count",
        "source_missing_count",
        "valid_count",
        "empty_count",
        "invalid_count",
        "exact_interval_count",
        "lower_bound_interval_count",
        "zero_exact_count",
    }
    item = _expect_object(value, fields=fields, name=name)
    return PriceExtractionExpectation(
        product_count=_expect_int(item["product_count"], name=f"{name}.product_count"),
        source_present_count=_expect_int(
            item["source_present_count"],
            name=f"{name}.source_present_count",
        ),
        source_missing_count=_expect_int(
            item["source_missing_count"],
            name=f"{name}.source_missing_count",
        ),
        valid_count=_expect_int(item["valid_count"], name=f"{name}.valid_count"),
        empty_count=_expect_int(item["empty_count"], name=f"{name}.empty_count"),
        invalid_count=_expect_int(item["invalid_count"], name=f"{name}.invalid_count"),
        exact_interval_count=_expect_int(
            item["exact_interval_count"],
            name=f"{name}.exact_interval_count",
        ),
        lower_bound_interval_count=_expect_int(
            item["lower_bound_interval_count"],
            name=f"{name}.lower_bound_interval_count",
        ),
        zero_exact_count=_expect_int(
            item["zero_exact_count"],
            name=f"{name}.zero_exact_count",
        ),
    )


def _decode_price_audit(value: object, *, index: int) -> PriceExtractionAudit:
    name = f"Gate-A extraction audit[{index}]"
    fields = {
        "facet_id",
        "binding_id",
        "product_count",
        "source_present_count",
        "source_missing_count",
        "valid_count",
        "empty_count",
        "invalid_count",
        "exact_interval_count",
        "lower_bound_interval_count",
        "zero_exact_count",
    }
    item = _expect_object(value, fields=fields, name=name)
    return PriceExtractionAudit(
        facet_id=_expect_string(item["facet_id"], name=f"{name}.facet_id"),
        binding_id=_expect_string(item["binding_id"], name=f"{name}.binding_id"),
        product_count=_expect_int(item["product_count"], name=f"{name}.product_count"),
        source_present_count=_expect_int(
            item["source_present_count"], name=f"{name}.source_present_count"
        ),
        source_missing_count=_expect_int(
            item["source_missing_count"], name=f"{name}.source_missing_count"
        ),
        valid_count=_expect_int(item["valid_count"], name=f"{name}.valid_count"),
        empty_count=_expect_int(item["empty_count"], name=f"{name}.empty_count"),
        invalid_count=_expect_int(item["invalid_count"], name=f"{name}.invalid_count"),
        exact_interval_count=_expect_int(
            item["exact_interval_count"], name=f"{name}.exact_interval_count"
        ),
        lower_bound_interval_count=_expect_int(
            item["lower_bound_interval_count"],
            name=f"{name}.lower_bound_interval_count",
        ),
        zero_exact_count=_expect_int(item["zero_exact_count"], name=f"{name}.zero_exact_count"),
    )


def _load_json(data: bytes, *, name: str, require_canonical: bool = False) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise GateACodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError as error:
        raise GateACodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GateACodecError(f"{name} is not valid strict JSON") from error
    if require_canonical and canonical_json_bytes(parsed) != data:
        raise GateACodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _expect_object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GateACodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        raise GateACodecError(f"{name} has invalid fields")
    return result


def _expect_array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise GateACodecError(f"{name} must be an array")
    return cast(list[object], value)


def _expect_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise GateACodecError(f"{name} must be a string")
    return value


def _expect_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise GateACodecError(f"{name} must be an integer")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(
        _expect_string(item, name=f"{name} item") for item in _expect_array(value, name=name)
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
