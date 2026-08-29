"""Strict JSON codecs for reviewed inputs and category candidate artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import cast

from ..canonical import (
    IJSON_SAFE_INTEGER_MAX,
    canonical_json_bytes,
    validate_semantic_string,
)
from ..errors import CategoryCodecError
from .models import (
    CATEGORY_GRAPH_PROPOSAL_SCHEMA,
    CATEGORY_REGISTRY_SCHEMA,
    CATEGORY_SCOPE_SELECTION_SCHEMA,
    PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA,
    CategoryGraphProposal,
    CategoryNode,
    CategoryRegistry,
    CategoryScope,
    CategoryScopeSelection,
    CategoryScopeSelectionDocument,
    ProductCategoryAssignment,
    ProductCategoryAssignmentSet,
    ProductCategoryAssignmentStatus,
    require_builder_version,
    require_content_id,
)
from .normalization import CATEGORY_BUILDER_VERSION, CATEGORY_UNICODE_DATA_VERSION
from .validation import (
    category_graph_id,
    validate_category_nodes,
    validate_category_registry,
    validate_product_category_assignment_set,
)


class _DuplicateJsonKeyError(ValueError):
    pass


def encode_category_registry(registry: CategoryRegistry) -> bytes:
    """Encode one fully validated CategoryRegistry as exact artifact bytes."""

    if type(registry) is not CategoryRegistry:
        raise TypeError("CategoryRegistry encoder requires the exact contract type")
    validate_category_registry(registry)
    return canonical_json_bytes(registry)


def decode_category_registry(data: bytes) -> CategoryRegistry:
    """Strictly decode canonical CategoryRegistry bytes and all invariants."""

    document = _load_json(data, require_canonical=True, name="CategoryRegistry")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "category_graph_id",
                "root_scope_id",
                "nodes",
                "scopes",
            },
            name="CategoryRegistry",
        )
        nodes = tuple(
            _decode_category_node(item, index=index)
            for index, item in enumerate(
                _expect_array(root["nodes"], name="CategoryRegistry.nodes")
            )
        )
        scopes = tuple(
            _decode_category_scope(item, index=index)
            for index, item in enumerate(
                _expect_array(root["scopes"], name="CategoryRegistry.scopes")
            )
        )
        if root["schema"] != CATEGORY_REGISTRY_SCHEMA:
            raise CategoryCodecError("CategoryRegistry.schema is invalid")
        registry = CategoryRegistry(
            schema=CATEGORY_REGISTRY_SCHEMA,
            catalog_id=_expect_string(root["catalog_id"], name="CategoryRegistry.catalog_id"),
            category_graph_id=_expect_string(
                root["category_graph_id"],
                name="CategoryRegistry.category_graph_id",
            ),
            root_scope_id=_expect_string(
                root["root_scope_id"], name="CategoryRegistry.root_scope_id"
            ),
            nodes=nodes,
            scopes=scopes,
        )
        validate_category_registry(registry)
        return registry
    except (TypeError, ValueError) as error:
        raise CategoryCodecError(f"invalid CategoryRegistry: {error}") from error


def encode_product_category_assignment_set(
    assignment_set: ProductCategoryAssignmentSet,
    *,
    registry: CategoryRegistry,
) -> bytes:
    """Encode one validated assignment artifact as exact canonical bytes."""

    if type(assignment_set) is not ProductCategoryAssignmentSet:
        raise TypeError("ProductCategoryAssignmentSet encoder requires the exact contract type")
    validate_product_category_assignment_set(assignment_set, registry=registry)
    return canonical_json_bytes(assignment_set)


def decode_product_category_assignment_set(
    data: bytes,
    *,
    registry: CategoryRegistry,
) -> ProductCategoryAssignmentSet:
    """Strictly decode canonical assignment bytes and graph references."""

    document = _load_json(
        data,
        require_canonical=True,
        name="ProductCategoryAssignmentSet",
    )
    try:
        root = _expect_object(
            document,
            fields={"schema", "catalog_id", "category_graph_id", "assignments"},
            name="ProductCategoryAssignmentSet",
        )
        assignments = tuple(
            _decode_assignment(item, index=index)
            for index, item in enumerate(
                _expect_array(
                    root["assignments"],
                    name="ProductCategoryAssignmentSet.assignments",
                )
            )
        )
        if root["schema"] != PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA:
            raise CategoryCodecError("ProductCategoryAssignmentSet.schema is invalid")
        assignment_set = ProductCategoryAssignmentSet(
            schema=PRODUCT_CATEGORY_ASSIGNMENT_SCHEMA,
            catalog_id=_expect_string(
                root["catalog_id"],
                name="ProductCategoryAssignmentSet.catalog_id",
            ),
            category_graph_id=_expect_string(
                root["category_graph_id"],
                name="ProductCategoryAssignmentSet.category_graph_id",
            ),
            assignments=assignments,
        )
        validate_product_category_assignment_set(assignment_set, registry=registry)
        return assignment_set
    except (TypeError, ValueError) as error:
        raise CategoryCodecError(f"invalid ProductCategoryAssignmentSet: {error}") from error


def decode_scope_selection(data: bytes) -> CategoryScopeSelectionDocument:
    """Decode human-authored selection JSON with strict fields and references."""

    document = _load_json(data, require_canonical=False, name="scope selection")
    try:
        root = _expect_object(
            document,
            fields={"schema", "catalog_id", "category_graph_id", "builder_version", "scopes"},
            name="scope selection",
        )
        selections = tuple(
            _decode_scope_selection(item, index=index)
            for index, item in enumerate(
                _expect_array(root["scopes"], name="scope selection.scopes")
            )
        )
        if root["schema"] != CATEGORY_SCOPE_SELECTION_SCHEMA:
            raise CategoryCodecError("scope selection.schema is invalid")
        return CategoryScopeSelectionDocument(
            schema=CATEGORY_SCOPE_SELECTION_SCHEMA,
            catalog_id=_expect_string(root["catalog_id"], name="scope selection.catalog_id"),
            category_graph_id=_expect_string(
                root["category_graph_id"],
                name="scope selection.category_graph_id",
            ),
            builder_version=_expect_string(
                root["builder_version"],
                name="scope selection.builder_version",
            ),
            scopes=selections,
        )
    except (TypeError, ValueError) as error:
        raise CategoryCodecError(f"invalid scope selection: {error}") from error


def category_graph_proposal_document(
    proposal: CategoryGraphProposal,
) -> dict[str, object]:
    """Return Pass-A metadata and nodes; provenance is written separately."""

    validate_category_nodes(proposal.nodes)
    if proposal.category_graph_id != category_graph_id(proposal.catalog_id, proposal.nodes):
        raise CategoryCodecError("category graph proposal graph ID is invalid")
    if proposal.builder_version != CATEGORY_BUILDER_VERSION:
        raise CategoryCodecError("category graph proposal builder version is unsupported")
    if proposal.unicode_data_version != CATEGORY_UNICODE_DATA_VERSION:
        raise CategoryCodecError("category graph proposal Unicode data version is unsupported")
    if proposal.raw_prefix_count != len(proposal.raw_path_mappings):
        raise CategoryCodecError("category graph proposal raw prefix count is invalid")
    return {
        "schema": CATEGORY_GRAPH_PROPOSAL_SCHEMA,
        "catalog_id": proposal.catalog_id,
        "category_graph_id": proposal.category_graph_id,
        "builder_version": proposal.builder_version,
        "unicode_data_version": proposal.unicode_data_version,
        "catalog_byte_size": proposal.catalog_byte_size,
        "product_count": proposal.product_count,
        "raw_prefix_count": proposal.raw_prefix_count,
        "canonical_node_count": len(proposal.nodes),
        "collision_count": len(proposal.collisions),
        "nodes": list(proposal.nodes),
    }


def decode_graph_proposal_document(data: bytes) -> dict[str, object]:
    """Strictly decode and validate canonical Pass-A graph metadata."""

    document = _load_json(data, require_canonical=True, name="category graph proposal")
    try:
        root = _expect_object(
            document,
            fields={
                "schema",
                "catalog_id",
                "category_graph_id",
                "builder_version",
                "unicode_data_version",
                "catalog_byte_size",
                "product_count",
                "raw_prefix_count",
                "canonical_node_count",
                "collision_count",
                "nodes",
            },
            name="category graph proposal",
        )
        if root["schema"] != CATEGORY_GRAPH_PROPOSAL_SCHEMA:
            raise CategoryCodecError("category graph proposal schema is invalid")
        nodes = tuple(
            _decode_category_node(item, index=index)
            for index, item in enumerate(
                _expect_array(root["nodes"], name="category graph proposal.nodes")
            )
        )
        validate_category_nodes(nodes)
        catalog_id = require_content_id(
            _expect_string(root["catalog_id"], name="category graph proposal.catalog_id"),
            name="category graph proposal.catalog_id",
        )
        graph_id = require_content_id(
            _expect_string(
                root["category_graph_id"],
                name="category graph proposal.category_graph_id",
            ),
            name="category graph proposal.category_graph_id",
        )
        builder_version = require_builder_version(
            _expect_string(
                root["builder_version"],
                name="category graph proposal.builder_version",
            )
        )
        unicode_data_version = validate_semantic_string(
            _expect_string(
                root["unicode_data_version"],
                name="category graph proposal.unicode_data_version",
            ),
            name="category graph proposal.unicode_data_version",
        )
        catalog_byte_size = _expect_positive_safe_integer(
            root["catalog_byte_size"],
            name="category graph proposal.catalog_byte_size",
        )
        product_count = _expect_positive_safe_integer(
            root["product_count"],
            name="category graph proposal.product_count",
        )
        raw_prefix_count = _expect_positive_safe_integer(
            root["raw_prefix_count"],
            name="category graph proposal.raw_prefix_count",
        )
        canonical_node_count = _expect_positive_safe_integer(
            root["canonical_node_count"],
            name="category graph proposal.canonical_node_count",
        )
        collision_count = _expect_nonnegative_safe_integer(
            root["collision_count"],
            name="category graph proposal.collision_count",
        )
        if builder_version != CATEGORY_BUILDER_VERSION:
            raise CategoryCodecError("category graph proposal builder version is unsupported")
        if unicode_data_version != CATEGORY_UNICODE_DATA_VERSION:
            raise CategoryCodecError("category graph proposal Unicode data version is unsupported")
        if graph_id != category_graph_id(catalog_id, nodes):
            raise CategoryCodecError("category graph proposal graph ID is invalid")
        if canonical_node_count != len(nodes):
            raise CategoryCodecError("category graph proposal node count is invalid")
        if canonical_node_count > raw_prefix_count:
            raise CategoryCodecError(
                "category graph proposal has more canonical nodes than raw prefixes"
            )
    except (TypeError, ValueError) as error:
        if isinstance(error, CategoryCodecError):
            raise
        raise CategoryCodecError(f"invalid category graph proposal: {error}") from error
    return {
        "schema": CATEGORY_GRAPH_PROPOSAL_SCHEMA,
        "catalog_id": catalog_id,
        "category_graph_id": graph_id,
        "builder_version": builder_version,
        "unicode_data_version": unicode_data_version,
        "catalog_byte_size": catalog_byte_size,
        "product_count": product_count,
        "raw_prefix_count": raw_prefix_count,
        "canonical_node_count": canonical_node_count,
        "collision_count": collision_count,
        "nodes": nodes,
    }


def _decode_category_node(value: object, *, index: int) -> CategoryNode:
    name = f"CategoryRegistry.nodes[{index}]"
    item = _expect_object(
        value,
        fields={"id", "parent_id", "canonical_path"},
        name=name,
    )
    parent = item["parent_id"]
    if parent is not None and type(parent) is not str:
        raise CategoryCodecError(f"{name}.parent_id must be string or null")
    return CategoryNode(
        id=_expect_string(item["id"], name=f"{name}.id"),
        parent_id=parent,
        canonical_path=_string_tuple(item["canonical_path"], name=f"{name}.canonical_path"),
    )


def _decode_category_scope(value: object, *, index: int) -> CategoryScope:
    name = f"CategoryRegistry.scopes[{index}]"
    item = _expect_object(
        value,
        fields={"id", "label", "root_node_ids", "member_node_ids"},
        name=name,
    )
    return CategoryScope(
        id=_expect_string(item["id"], name=f"{name}.id"),
        label=_expect_string(item["label"], name=f"{name}.label"),
        root_node_ids=_string_tuple(item["root_node_ids"], name=f"{name}.root_node_ids"),
        member_node_ids=_string_tuple(item["member_node_ids"], name=f"{name}.member_node_ids"),
    )


def _decode_assignment(value: object, *, index: int) -> ProductCategoryAssignment:
    name = f"ProductCategoryAssignmentSet.assignments[{index}]"
    item = _expect_object(
        value,
        fields={"parent_asin", "status", "leaf_node_ids"},
        name=name,
    )
    try:
        status = ProductCategoryAssignmentStatus(
            _expect_string(item["status"], name=f"{name}.status")
        )
    except ValueError as error:
        raise CategoryCodecError(f"{name}.status is invalid") from error
    return ProductCategoryAssignment(
        parent_asin=_expect_string(item["parent_asin"], name=f"{name}.parent_asin"),
        status=status,
        leaf_node_ids=_string_tuple(item["leaf_node_ids"], name=f"{name}.leaf_node_ids"),
    )


def _decode_scope_selection(value: object, *, index: int) -> CategoryScopeSelection:
    name = f"scope selection.scopes[{index}]"
    item = _expect_object(
        value,
        fields={"label", "root_node_ids"},
        name=name,
    )
    return CategoryScopeSelection(
        label=_expect_string(item["label"], name=f"{name}.label"),
        root_node_ids=_string_tuple(item["root_node_ids"], name=f"{name}.root_node_ids"),
    )


def _load_json(data: bytes, *, require_canonical: bool, name: str) -> object:
    if type(data) is not bytes:
        raise TypeError("artifact input must be bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise CategoryCodecError(f"{name} must not contain a UTF-8 BOM")
    try:
        text = data.decode("utf-8")
        parsed: object = json.loads(
            text,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        canonical = canonical_json_bytes(parsed)
    except _DuplicateJsonKeyError as error:
        raise CategoryCodecError(f"{name} contains duplicate object members") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        if isinstance(error, CategoryCodecError):
            raise
        raise CategoryCodecError(f"{name} is not valid contract JSON") from error
    if require_canonical and data != canonical:
        raise CategoryCodecError(f"{name} bytes are not canonical JSON")
    return parsed


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite number token: {raw}")


def _expect_object(
    value: object,
    *,
    fields: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise CategoryCodecError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != fields:
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields)
        raise CategoryCodecError(f"{name} fields are invalid; missing={missing}, unknown={unknown}")
    return result


def _expect_array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise CategoryCodecError(f"{name} must be an array")
    return cast(list[object], value)


def _expect_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise CategoryCodecError(f"{name} must be a string")
    return value


def _expect_nonnegative_safe_integer(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= IJSON_SAFE_INTEGER_MAX:
        raise CategoryCodecError(f"{name} must be a non-negative I-JSON safe integer")
    return value


def _expect_positive_safe_integer(value: object, *, name: str) -> int:
    result = _expect_nonnegative_safe_integer(value, name=name)
    if result == 0:
        raise CategoryCodecError(f"{name} must be positive")
    return result


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    items = _expect_array(value, name=name)
    return tuple(_expect_string(item, name=f"{name}[{index}]") for index, item in enumerate(items))


def canonical_json_lines(items: Iterable[object]) -> bytes:
    """Encode audit-only JSONL rows with one canonical record per line."""

    return b"".join(canonical_json_bytes(item) + b"\n" for item in items)
