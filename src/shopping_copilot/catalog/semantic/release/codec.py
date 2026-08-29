"""Strict canonical codecs for the CS6 release manifest and reviewed config."""

from __future__ import annotations

import json
from typing import cast

from ..canonical import canonical_json_bytes, content_id_for_bytes
from ..errors import ReleaseCodecError
from ..facet.resolution_models import RESOLUTION_POLICY_ID
from .models import (
    ARTIFACT_SPEC,
    CATALOG_SEMANTIC_RELEASE_SCHEMA,
    ArtifactKind,
    ArtifactRef,
    CatalogSemanticReleaseManifest,
    ReviewedSemanticConfig,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def encode_release_manifest(value: CatalogSemanticReleaseManifest) -> bytes:
    if type(value) is not CatalogSemanticReleaseManifest:
        raise TypeError("release manifest encoder requires exact manifest type")
    return canonical_json_bytes(value)


def decode_release_manifest(data: bytes) -> CatalogSemanticReleaseManifest:
    """Decode exact canonical bytes and reject every unknown or missing field."""

    root = _object(
        _load_canonical_json(data, name="CatalogSemanticReleaseManifest"),
        fields={
            "schema",
            "catalog_id",
            "category_registry_id",
            "product_category_assignment_id",
            "facet_schema_id",
            "facet_applicability_id",
            "facet_source_bindings_id",
            "facet_evidence_store_id",
            "product_facet_index_id",
            "facet_stats_id",
            "effective_capabilities_id",
            "runtime_value_lexicon_id",
            "runtime_registry_id",
            "reviewed_config_id",
            "resolution_policy_id",
            "builder_version",
            "artifacts",
        },
        name="CatalogSemanticReleaseManifest",
    )
    try:
        if root["schema"] != CATALOG_SEMANTIC_RELEASE_SCHEMA:
            raise ReleaseCodecError("release manifest schema is invalid")
        policy = _string(root["resolution_policy_id"], name="resolution_policy_id")
        if policy != RESOLUTION_POLICY_ID:
            raise ReleaseCodecError("release resolution policy is unsupported")
        return CatalogSemanticReleaseManifest(
            schema=CATALOG_SEMANTIC_RELEASE_SCHEMA,
            catalog_id=_string(root["catalog_id"], name="catalog_id"),
            category_registry_id=_string(
                root["category_registry_id"],
                name="category_registry_id",
            ),
            product_category_assignment_id=_string(
                root["product_category_assignment_id"],
                name="product_category_assignment_id",
            ),
            facet_schema_id=_string(root["facet_schema_id"], name="facet_schema_id"),
            facet_applicability_id=_string(
                root["facet_applicability_id"],
                name="facet_applicability_id",
            ),
            facet_source_bindings_id=_string(
                root["facet_source_bindings_id"],
                name="facet_source_bindings_id",
            ),
            facet_evidence_store_id=_string(
                root["facet_evidence_store_id"],
                name="facet_evidence_store_id",
            ),
            product_facet_index_id=_string(
                root["product_facet_index_id"],
                name="product_facet_index_id",
            ),
            facet_stats_id=_string(root["facet_stats_id"], name="facet_stats_id"),
            effective_capabilities_id=_string(
                root["effective_capabilities_id"],
                name="effective_capabilities_id",
            ),
            runtime_value_lexicon_id=_string(
                root["runtime_value_lexicon_id"],
                name="runtime_value_lexicon_id",
            ),
            runtime_registry_id=_string(
                root["runtime_registry_id"],
                name="runtime_registry_id",
            ),
            reviewed_config_id=_string(
                root["reviewed_config_id"],
                name="reviewed_config_id",
            ),
            resolution_policy_id=RESOLUTION_POLICY_ID,
            builder_version=_string(root["builder_version"], name="builder_version"),
            artifacts=tuple(
                _decode_artifact_ref(item, name=f"artifacts[{index}]")
                for index, item in enumerate(_array(root["artifacts"], name="artifacts"))
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ReleaseCodecError):
            raise
        raise ReleaseCodecError(f"invalid release manifest: {error}") from error


def encode_reviewed_semantic_config(value: ReviewedSemanticConfig) -> bytes:
    if type(value) is not ReviewedSemanticConfig:
        raise TypeError("reviewed config encoder requires exact ReviewedSemanticConfig")
    return canonical_json_bytes(value)


def release_id_for_manifest(value: CatalogSemanticReleaseManifest) -> str:
    """Return the contract's external release ID over exact canonical manifest bytes."""

    return content_id_for_bytes(encode_release_manifest(value))


def _decode_artifact_ref(value: object, *, name: str) -> ArtifactRef:
    item = _object(
        value,
        fields={"kind", "schema", "content_id", "byte_size"},
        name=name,
    )
    kind_raw = _string(item["kind"], name=f"{name}.kind")
    if kind_raw not in ARTIFACT_SPEC:
        raise ReleaseCodecError(f"{name}.kind is invalid")
    byte_size = item["byte_size"]
    if type(byte_size) is not int:
        raise ReleaseCodecError(f"{name}.byte_size must be an integer")
    return ArtifactRef(
        kind=cast(ArtifactKind, kind_raw),
        schema=_string(item["schema"], name=f"{name}.schema"),
        content_id=_string(item["content_id"], name=f"{name}.content_id"),
        byte_size=byte_size,
    )


def _load_canonical_json(data: bytes, *, name: str) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ReleaseCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError as error:
        raise ReleaseCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ReleaseCodecError(f"{name} is not valid JSON") from error
    if data != canonical_json_bytes(parsed):
        raise ReleaseCodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ReleaseCodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        raise ReleaseCodecError(f"{name} has invalid fields")
    return result


def _array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ReleaseCodecError(f"{name} must be an array")
    return cast(list[object], value)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ReleaseCodecError(f"{name} must be a string")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite JSON token: {raw}")
