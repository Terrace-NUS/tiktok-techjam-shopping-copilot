"""Strict canonical codecs for CS5A runtime projection artifacts."""

from __future__ import annotations

import json
from typing import Literal, cast

from ..canonical import canonical_json_bytes, content_id_for_value
from ..errors import RuntimeProjectionCodecError
from ..facet.resolution_models import RESOLUTION_POLICY_ID
from .models import (
    RUNTIME_FACET_REGISTRY_SCHEMA,
    RUNTIME_VALUE_LEXICON_SCHEMA,
    NumericRuntimeDomain,
    RuntimeFacetRegistryArtifact,
    RuntimeFacetSpecRecord,
    RuntimeProjectionCandidateBuild,
    RuntimeValueLexicon,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def encode_runtime_facet_registry(value: RuntimeFacetRegistryArtifact) -> bytes:
    if type(value) is not RuntimeFacetRegistryArtifact:
        raise TypeError("runtime registry encoder requires exact artifact type")
    return canonical_json_bytes(value)


def decode_runtime_facet_registry(data: bytes) -> RuntimeFacetRegistryArtifact:
    root = _object(
        _load_canonical_json(data, name="RuntimeFacetRegistryArtifact"),
        fields={
            "schema",
            "category_registry_id",
            "facet_schema_id",
            "effective_capabilities_id",
            "resolution_policy_id",
            "entries",
        },
        name="RuntimeFacetRegistryArtifact",
    )
    try:
        if root["schema"] != RUNTIME_FACET_REGISTRY_SCHEMA:
            raise RuntimeProjectionCodecError("runtime registry schema is invalid")
        return RuntimeFacetRegistryArtifact(
            schema=RUNTIME_FACET_REGISTRY_SCHEMA,
            category_registry_id=_string(root["category_registry_id"], name="category_registry_id"),
            facet_schema_id=_string(root["facet_schema_id"], name="facet_schema_id"),
            effective_capabilities_id=_string(
                root["effective_capabilities_id"], name="effective_capabilities_id"
            ),
            resolution_policy_id=_policy(root["resolution_policy_id"]),
            entries=tuple(
                _decode_spec_record(item, name=f"entries[{index}]")
                for index, item in enumerate(_array(root["entries"], name="entries"))
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, RuntimeProjectionCodecError):
            raise
        raise RuntimeProjectionCodecError(f"invalid runtime facet registry: {error}") from error


def encode_runtime_value_lexicon(value: RuntimeValueLexicon) -> bytes:
    if type(value) is not RuntimeValueLexicon:
        raise TypeError("runtime lexicon encoder requires exact artifact type")
    return canonical_json_bytes(value)


def decode_runtime_value_lexicon(data: bytes) -> RuntimeValueLexicon:
    root = _object(
        _load_canonical_json(data, name="RuntimeValueLexicon"),
        fields={
            "schema",
            "runtime_registry_id",
            "category_registry_id",
            "facet_applicability_id",
            "product_facet_index_id",
            "resolution_policy_id",
            "domains",
        },
        name="RuntimeValueLexicon",
    )
    try:
        if root["schema"] != RUNTIME_VALUE_LEXICON_SCHEMA:
            raise RuntimeProjectionCodecError("runtime lexicon schema is invalid")
        return RuntimeValueLexicon(
            schema=RUNTIME_VALUE_LEXICON_SCHEMA,
            runtime_registry_id=_string(root["runtime_registry_id"], name="runtime_registry_id"),
            category_registry_id=_string(root["category_registry_id"], name="category_registry_id"),
            facet_applicability_id=_string(
                root["facet_applicability_id"], name="facet_applicability_id"
            ),
            product_facet_index_id=_string(
                root["product_facet_index_id"], name="product_facet_index_id"
            ),
            resolution_policy_id=_policy(root["resolution_policy_id"]),
            domains=tuple(
                _decode_numeric_domain(item, name=f"domains[{index}]")
                for index, item in enumerate(_array(root["domains"], name="domains"))
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, RuntimeProjectionCodecError):
            raise
        raise RuntimeProjectionCodecError(f"invalid runtime value lexicon: {error}") from error


def runtime_projection_candidate_document(
    build: RuntimeProjectionCandidateBuild,
) -> dict[str, object]:
    """Return compact CS5 metadata without claiming later consumer integration."""

    if type(build) is not RuntimeProjectionCandidateBuild:
        raise TypeError("runtime candidate document requires exact build type")
    return {
        "schema": build.schema,
        "builder_version": build.builder_version,
        "catalog_id": build.catalog_id,
        "gate_b_selection_id": build.gate_b_selection_id,
        "effective_capabilities_id": build.effective_capabilities_id,
        "runtime_facet_registry_id": content_id_for_value(build.runtime_registry),
        "runtime_value_lexicon_id": content_id_for_value(build.runtime_lexicon),
        "ordinary_runtime_facet_count": len(build.runtime_lexicon.domains),
        "reserved_runtime_facet_count": 1,
        "grounding_implemented": True,
        "retrieval_integrated": False,
        "session_gateway_integrated": False,
    }


def _decode_spec_record(value: object, *, name: str) -> RuntimeFacetSpecRecord:
    item = _object(
        value,
        fields={"facet_id", "kind", "operator_values", "intent_value_normalizer_id"},
        name=name,
    )
    kind = _string(item["kind"], name=f"{name}.kind")
    if kind not in ("categorical", "numeric"):
        raise RuntimeProjectionCodecError(f"{name}.kind is invalid")
    return RuntimeFacetSpecRecord(
        facet_id=_string(item["facet_id"], name=f"{name}.facet_id"),
        kind=cast(Literal["categorical", "numeric"], kind),
        operator_values=tuple(
            _string(raw, name=f"{name}.operator_values[{index}]")
            for index, raw in enumerate(
                _array(item["operator_values"], name=f"{name}.operator_values")
            )
        ),
        intent_value_normalizer_id=_string(
            item["intent_value_normalizer_id"],
            name=f"{name}.intent_value_normalizer_id",
        ),
    )


def _decode_numeric_domain(value: object, *, name: str) -> NumericRuntimeDomain:
    item = _object(
        value,
        fields={
            "kind",
            "facet_id",
            "intent_value_normalizer_id",
            "canonical_unit",
            "integer_only",
        },
        name=name,
    )
    if item["kind"] != "numeric" or item["facet_id"] != "price":
        raise RuntimeProjectionCodecError(f"{name} supports only numeric price")
    if item["canonical_unit"] != "USD_CENT" or item["integer_only"] is not True:
        raise RuntimeProjectionCodecError(f"{name} must use integer USD_CENT")
    return NumericRuntimeDomain(
        kind="numeric",
        facet_id="price",
        intent_value_normalizer_id=_string(
            item["intent_value_normalizer_id"],
            name=f"{name}.intent_value_normalizer_id",
        ),
        canonical_unit="USD_CENT",
        integer_only=True,
    )


def _load_canonical_json(data: bytes, *, name: str) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise RuntimeProjectionCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError as error:
        raise RuntimeProjectionCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RuntimeProjectionCodecError(f"{name} is not valid JSON") from error
    if data != canonical_json_bytes(parsed):
        raise RuntimeProjectionCodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeProjectionCodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        raise RuntimeProjectionCodecError(f"{name} has invalid fields")
    return result


def _array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise RuntimeProjectionCodecError(f"{name} must be an array")
    return cast(list[object], value)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise RuntimeProjectionCodecError(f"{name} must be a string")
    return value


def _policy(value: object) -> Literal["structured_resolution_v1"]:
    result = _string(value, name="resolution_policy_id")
    if result != RESOLUTION_POLICY_ID:
        raise RuntimeProjectionCodecError("resolution_policy_id is unsupported")
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
