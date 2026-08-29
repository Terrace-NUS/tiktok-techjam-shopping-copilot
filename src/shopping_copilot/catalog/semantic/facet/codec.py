"""Strict codecs for CS2 Gate-A profiling inputs and proposal artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import cast

from ..canonical import canonical_json_bytes
from ..errors import FacetProfileCodecError
from .models import (
    FACET_PROFILE_BUILDER_VERSION,
    GATE_A_PROFILE_SELECTION_SCHEMA,
    GateAProfileSelection,
    GateASourceProfileBuild,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def decode_profile_selection(data: bytes) -> GateAProfileSelection:
    """Decode a human-authored profiling selection with exact allowed fields."""

    document = _load_json(data, name="Gate-A profile selection")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "category_registry_id",
                "product_category_assignment_id",
                "builder_version",
                "top_level_keys",
                "include_all_details",
                "sample_seed",
                "sample_limit",
                "top_value_limit",
            },
            name="Gate-A profile selection",
        )
        if root["schema"] != GATE_A_PROFILE_SELECTION_SCHEMA:
            raise FacetProfileCodecError("Gate-A profile selection schema is invalid")
        if root["builder_version"] != FACET_PROFILE_BUILDER_VERSION:
            raise FacetProfileCodecError("Gate-A profile selection builder version is unsupported")
        return GateAProfileSelection(
            schema=GATE_A_PROFILE_SELECTION_SCHEMA,
            catalog_id=_expect_string(root["catalog_id"], name="selection.catalog_id"),
            category_registry_id=_expect_string(
                root["category_registry_id"],
                name="selection.category_registry_id",
            ),
            product_category_assignment_id=_expect_string(
                root["product_category_assignment_id"],
                name="selection.product_category_assignment_id",
            ),
            builder_version=FACET_PROFILE_BUILDER_VERSION,
            top_level_keys=_string_tuple(
                root["top_level_keys"],
                name="selection.top_level_keys",
            ),
            include_all_details=_expect_bool(
                root["include_all_details"],
                name="selection.include_all_details",
            ),
            sample_seed=_expect_string(root["sample_seed"], name="selection.sample_seed"),
            sample_limit=_expect_int(root["sample_limit"], name="selection.sample_limit"),
            top_value_limit=_expect_int(
                root["top_value_limit"],
                name="selection.top_value_limit",
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, FacetProfileCodecError):
            raise
        raise FacetProfileCodecError(f"invalid Gate-A profile selection: {error}") from error


def profile_document(build: GateASourceProfileBuild) -> dict[str, object]:
    """Return proposal metadata and inventories; dense rows live in JSONL files."""

    if type(build) is not GateASourceProfileBuild:
        raise TypeError("profile_document requires GateASourceProfileBuild")
    return {
        "schema": build.schema,
        "catalog_id": build.catalog_id,
        "category_registry_id": build.category_registry_id,
        "product_category_assignment_id": build.product_category_assignment_id,
        "builder_version": build.builder_version,
        "selection": build.selection,
        "top_level_fields": list(build.top_level_fields),
        "scopes": list(build.scopes),
        "sources": list(build.sources),
        "scope_source_profile_count": len(build.scope_source_profiles),
        "sample_count": len(build.samples),
    }


def canonical_json_lines(values: Iterable[object]) -> bytes:
    """Encode an iterable as canonical UTF-8 JSONL with one final newline per row."""

    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def _load_json(data: bytes, *, name: str) -> object:
    if type(data) is not bytes:
        raise TypeError(f"{name} input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise FacetProfileCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except _DuplicateJsonKeyError as error:
        raise FacetProfileCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise FacetProfileCodecError(f"{name} is not valid strict JSON") from error


def _expect_object(value: object, *, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise FacetProfileCodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        raise FacetProfileCodecError(f"{name} has invalid fields")
    return result


def _expect_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise FacetProfileCodecError(f"{name} must be a string")
    return value


def _expect_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise FacetProfileCodecError(f"{name} must be a boolean")
    return value


def _expect_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise FacetProfileCodecError(f"{name} must be an integer")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise FacetProfileCodecError(f"{name} must be an array")
    result: list[str] = []
    for item in cast(list[object], value):
        result.append(_expect_string(item, name=f"{name} item"))
    return tuple(result)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite JSON token: {raw}")
