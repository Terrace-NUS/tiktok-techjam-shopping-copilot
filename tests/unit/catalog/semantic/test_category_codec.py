from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic import CategoryCodecError, canonical_json_bytes
from shopping_copilot.catalog.semantic.category import (
    CATEGORY_SCOPE_SELECTION_SCHEMA,
    CategoryCandidateBuild,
    CategoryRegistry,
    CategoryScopeSelection,
    CategoryScopeSelectionDocument,
    ProductCategoryAssignmentSet,
    build_category_candidate,
    build_category_graph_proposal,
    decode_category_registry,
    decode_product_category_assignment_set,
    decode_scope_selection,
    encode_category_registry,
    encode_product_category_assignment_set,
)
from shopping_copilot.catalog.semantic.raw_catalog import scan_raw_catalog


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExtendedRegistry(CategoryRegistry):
    unexpected: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExtendedAssignmentSet(ProductCategoryAssignmentSet):
    unexpected: str


def _write_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog.jsonl"
    rows = (
        {"parent_asin": "p-shoes", "categories": ["Clothing", "Shoes"], "details": {}},
        {"parent_asin": "p-clothing", "categories": ["Clothing"], "details": {}},
        {"parent_asin": "p-belt", "categories": ["Accessories", "Belts"], "details": {}},
    )
    catalog.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    return catalog


def _candidate(tmp_path: Path) -> CategoryCandidateBuild:
    scan = scan_raw_catalog(_write_catalog(tmp_path), expected_product_count=3)
    proposal = build_category_graph_proposal(scan)
    root_node_ids = tuple(sorted(node.id for node in proposal.nodes if node.parent_id is None))
    selection = CategoryScopeSelectionDocument(
        schema=CATEGORY_SCOPE_SELECTION_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_graph_id=proposal.category_graph_id,
        builder_version=proposal.builder_version,
        scopes=(CategoryScopeSelection(label="All products", root_node_ids=root_node_ids),),
    )
    return build_category_candidate(scan, selection, enforce_official_gate=False)


def _with_duplicate_catalog_id(payload: bytes, catalog_id: str) -> bytes:
    duplicate_member = canonical_json_bytes(catalog_id)
    return b'{"catalog_id":' + duplicate_member + b"," + payload[1:]


def test_category_artifacts_round_trip_as_exact_canonical_bytes(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)

    registry_bytes = encode_category_registry(candidate.registry)
    decoded_registry = decode_category_registry(registry_bytes)
    assignment_bytes = encode_product_category_assignment_set(
        candidate.assignments,
        registry=candidate.registry,
    )
    decoded_assignments = decode_product_category_assignment_set(
        assignment_bytes,
        registry=decoded_registry,
    )

    assert decoded_registry == candidate.registry
    assert decoded_assignments == candidate.assignments
    assert encode_category_registry(decoded_registry) == registry_bytes
    assert (
        encode_product_category_assignment_set(
            decoded_assignments,
            registry=decoded_registry,
        )
        == assignment_bytes
    )


def test_formal_encoders_reject_dataclass_subclasses_with_extra_fields(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    registry = candidate.registry
    extended_registry = _ExtendedRegistry(
        schema=registry.schema,
        catalog_id=registry.catalog_id,
        category_graph_id=registry.category_graph_id,
        root_scope_id=registry.root_scope_id,
        nodes=registry.nodes,
        scopes=registry.scopes,
        unexpected="not a contract field",
    )
    assignments = candidate.assignments
    extended_assignments = _ExtendedAssignmentSet(
        schema=assignments.schema,
        catalog_id=assignments.catalog_id,
        category_graph_id=assignments.category_graph_id,
        assignments=assignments.assignments,
        unexpected="not a contract field",
    )

    with pytest.raises(TypeError, match="exact contract type"):
        encode_category_registry(extended_registry)
    with pytest.raises(TypeError, match="exact contract type"):
        encode_product_category_assignment_set(
            extended_assignments,
            registry=registry,
        )


@pytest.mark.parametrize("artifact", ["registry", "assignments"])
def test_category_artifact_decoders_reject_unknown_fields(
    tmp_path: Path,
    artifact: str,
) -> None:
    candidate = _candidate(tmp_path)
    if artifact == "registry":
        document = json.loads(encode_category_registry(candidate.registry))
        decoder = decode_category_registry
        decoder_kwargs: dict[str, object] = {}
    else:
        document = json.loads(
            encode_product_category_assignment_set(
                candidate.assignments,
                registry=candidate.registry,
            )
        )
        decoder = decode_product_category_assignment_set
        decoder_kwargs = {"registry": candidate.registry}
    document["unexpected"] = True

    with pytest.raises(CategoryCodecError, match=r"unknown=\['unexpected'\]"):
        decoder(canonical_json_bytes(document), **decoder_kwargs)


@pytest.mark.parametrize("artifact", ["registry", "assignments"])
def test_category_artifact_decoders_reject_duplicate_members(
    tmp_path: Path,
    artifact: str,
) -> None:
    candidate = _candidate(tmp_path)
    if artifact == "registry":
        payload = encode_category_registry(candidate.registry)
        decoder = decode_category_registry
        decoder_kwargs: dict[str, object] = {}
    else:
        payload = encode_product_category_assignment_set(
            candidate.assignments,
            registry=candidate.registry,
        )
        decoder = decode_product_category_assignment_set
        decoder_kwargs = {"registry": candidate.registry}
    duplicated = _with_duplicate_catalog_id(payload, candidate.registry.catalog_id)

    with pytest.raises(CategoryCodecError, match="duplicate object members"):
        decoder(duplicated, **decoder_kwargs)


@pytest.mark.parametrize("artifact", ["registry", "assignments"])
def test_category_artifact_decoders_reject_noncanonical_bytes(
    tmp_path: Path,
    artifact: str,
) -> None:
    candidate = _candidate(tmp_path)
    if artifact == "registry":
        payload = encode_category_registry(candidate.registry)
        decoder = decode_category_registry
        decoder_kwargs: dict[str, object] = {}
    else:
        payload = encode_product_category_assignment_set(
            candidate.assignments,
            registry=candidate.registry,
        )
        decoder = decode_product_category_assignment_set
        decoder_kwargs = {"registry": candidate.registry}

    with pytest.raises(CategoryCodecError, match="not canonical JSON"):
        decoder(payload + b"\n", **decoder_kwargs)


def test_category_registry_decoder_rejects_tampered_graph_identity(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    document = json.loads(encode_category_registry(candidate.registry))
    document["category_graph_id"] = "sha256:" + "0" * 64

    with pytest.raises(CategoryCodecError, match="category_graph_id"):
        decode_category_registry(canonical_json_bytes(document))


def test_assignment_decoder_rejects_tampered_leaf_reference(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    document = json.loads(
        encode_product_category_assignment_set(
            candidate.assignments,
            registry=candidate.registry,
        )
    )
    document["assignments"][0]["leaf_node_ids"] = ["cn_" + "f" * 64]

    with pytest.raises(CategoryCodecError, match="unknown CategoryNode"):
        decode_product_category_assignment_set(
            canonical_json_bytes(document),
            registry=candidate.registry,
        )


def test_scope_selection_allows_human_formatting_but_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    root_scope = next(
        scope for scope in candidate.registry.scopes if scope.id == candidate.registry.root_scope_id
    )
    document = {
        "schema": CATEGORY_SCOPE_SELECTION_SCHEMA,
        "catalog_id": candidate.registry.catalog_id,
        "category_graph_id": candidate.registry.category_graph_id,
        "builder_version": candidate.builder_version,
        "scopes": [
            {
                "label": "All products",
                "root_node_ids": list(root_scope.root_node_ids),
            }
        ],
    }
    pretty = json.dumps(document, indent=2).encode("utf-8")

    assert decode_scope_selection(pretty).catalog_id == candidate.registry.catalog_id

    document["unexpected"] = True
    with pytest.raises(CategoryCodecError, match=r"unknown=\['unexpected'\]"):
        decode_scope_selection(canonical_json_bytes(document))
