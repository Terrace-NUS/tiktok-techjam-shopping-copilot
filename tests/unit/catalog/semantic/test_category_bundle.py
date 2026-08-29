from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic import (
    CategoryBundleBusyError,
    CategoryBundleIntegrityError,
    CategorySelectionError,
    canonical_json_bytes,
)
from shopping_copilot.catalog.semantic.category import (
    CATEGORY_BUILDER_VERSION,
    CATEGORY_SCOPE_SELECTION_SCHEMA,
    CategoryCandidateBuild,
    CategoryNode,
    CategoryRegistry,
    CategoryScopeSelection,
    CategoryScopeSelectionDocument,
    build_category_graph_proposal,
    category_graph_id,
    category_node_id,
    decode_category_registry,
    decode_product_category_assignment_set,
    encode_category_registry,
    encode_product_category_assignment_set,
    materialize_category_scope,
    reviewed_category_scopes_candidate_document,
    validate_category_bundle,
    write_category_candidate_bundle,
    write_category_graph_proposal_bundle,
)
from shopping_copilot.catalog.semantic.category.bundle import (
    CANDIDATE_ARTIFACT_FILENAMES,
    CATEGORY_BUNDLE_MANIFEST_FILENAME,
    PROPOSAL_ARTIFACT_FILENAMES,
)
from shopping_copilot.catalog.semantic.category.reporting import category_candidate_markdown
from shopping_copilot.catalog.semantic.raw_catalog import scan_raw_catalog


def _write_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog.jsonl"
    rows = (
        {"parent_asin": "p-shoes", "categories": ["Clothing", "Shoes"], "details": {}},
        {"parent_asin": "p-clothing", "categories": ["Clothing"], "details": {}},
        {"parent_asin": "p-belt", "categories": ["Accessories", "Belts"], "details": {}},
    )
    catalog.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    return catalog


def _selection_for_catalog(catalog: Path) -> CategoryScopeSelectionDocument:
    scan = scan_raw_catalog(catalog, expected_product_count=3)
    proposal = build_category_graph_proposal(scan)
    root_node_ids = tuple(sorted(node.id for node in proposal.nodes if node.parent_id is None))
    return CategoryScopeSelectionDocument(
        schema=CATEGORY_SCOPE_SELECTION_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_graph_id=proposal.category_graph_id,
        builder_version=proposal.builder_version,
        scopes=(CategoryScopeSelection(label="All products", root_node_ids=root_node_ids),),
    )


def _write_selection(path: Path, selection: CategoryScopeSelectionDocument) -> None:
    path.write_bytes(canonical_json_bytes(selection))


def _bundle_bytes(output: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_pass_a_bundle_is_complete_valid_and_byte_deterministic(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    output = tmp_path / "proposal"

    first = write_category_graph_proposal_bundle(
        catalog,
        output,
        expected_product_count=3,
    )
    first_bytes = _bundle_bytes(output)
    validate_category_bundle(output)
    second = write_category_graph_proposal_bundle(
        catalog,
        output,
        expected_product_count=3,
    )
    second_bytes = _bundle_bytes(output)

    assert first == second
    assert first_bytes == second_bytes
    assert set(first_bytes) == {
        *PROPOSAL_ARTIFACT_FILENAMES,
        CATEGORY_BUNDLE_MANIFEST_FILENAME,
    }
    manifest = json.loads(first_bytes[CATEGORY_BUNDLE_MANIFEST_FILENAME])
    assert manifest["kind"] == "proposal"
    assert manifest["expected_product_count"] == 3
    assert manifest["official_p0_assignment_gate"] is False
    assert first_bytes[CATEGORY_BUNDLE_MANIFEST_FILENAME] == canonical_json_bytes(manifest)


def test_bundle_writer_rejects_input_inside_output_and_concurrent_writer(
    tmp_path: Path,
) -> None:
    nested_output = tmp_path / "nested-output"
    nested_output.mkdir()
    nested_catalog = _write_catalog(nested_output)
    with pytest.raises(ValueError, match="must not live inside"):
        write_category_graph_proposal_bundle(
            nested_catalog,
            nested_output,
            expected_product_count=3,
        )

    catalog = _write_catalog(tmp_path)
    output = tmp_path / "proposal"
    lock_path = tmp_path / ".proposal.write.lock"
    lock_path.write_bytes(b"")
    with pytest.raises(CategoryBundleBusyError, match="already being written"):
        write_category_graph_proposal_bundle(
            catalog,
            output,
            expected_product_count=3,
        )


def test_official_candidate_gate_cannot_be_weakened_to_fixture_count(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    selection_path = tmp_path / "scope-selection.json"
    _write_selection(selection_path, _selection_for_catalog(catalog))

    with pytest.raises(ValueError, match="requires expected_product_count=50000"):
        write_category_candidate_bundle(
            catalog,
            selection_path,
            tmp_path / "candidate",
            expected_product_count=3,
            enforce_official_gate=True,
        )


def test_pass_b_small_fixture_explicitly_disables_official_gate_and_is_repeatable(
    tmp_path: Path,
) -> None:
    catalog = _write_catalog(tmp_path)
    selection_path = tmp_path / "scope-selection.json"
    _write_selection(selection_path, _selection_for_catalog(catalog))
    output = tmp_path / "candidate"

    first = write_category_candidate_bundle(
        catalog,
        selection_path,
        output,
        expected_product_count=3,
        enforce_official_gate=False,
    )
    first_bytes = _bundle_bytes(output)
    with pytest.raises(CategoryBundleIntegrityError, match="requires the exact raw catalog"):
        validate_category_bundle(output)
    validate_category_bundle(output, catalog_path=catalog)
    second = write_category_candidate_bundle(
        catalog,
        selection_path,
        output,
        expected_product_count=3,
        enforce_official_gate=False,
    )
    second_bytes = _bundle_bytes(output)

    assert first == second
    assert first_bytes == second_bytes
    assert len(first.assignments.assignments) == 3
    assert set(first_bytes) == {
        *CANDIDATE_ARTIFACT_FILENAMES,
        CATEGORY_BUNDLE_MANIFEST_FILENAME,
    }
    manifest = json.loads(first_bytes[CATEGORY_BUNDLE_MANIFEST_FILENAME])
    assert manifest["kind"] == "candidate"
    assert manifest["expected_product_count"] == 3
    assert manifest["official_p0_assignment_gate"] is False


def test_pass_a_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    output = tmp_path / "proposal"
    write_category_graph_proposal_bundle(catalog, output, expected_product_count=3)
    manifest_path = output / CATEGORY_BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["expected_product_count"] = 4
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CategoryBundleIntegrityError, match="product count"):
        validate_category_bundle(output)


def test_pass_a_artifact_tamper_is_rejected_even_if_manifest_is_rehashed(
    tmp_path: Path,
) -> None:
    catalog = _write_catalog(tmp_path)
    output = tmp_path / "proposal"
    write_category_graph_proposal_bundle(catalog, output, expected_product_count=3)
    report_path = output / "report.md"
    tampered_report = report_path.read_bytes() + b"\nTampered review result.\n"
    report_path.write_bytes(tampered_report)

    manifest_path = output / CATEGORY_BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    report_entry = next(
        entry for entry in manifest["artifacts"] if entry["filename"] == "report.md"
    )
    report_entry["byte_size"] = len(tampered_report)
    report_entry["sha256"] = hashlib.sha256(tampered_report).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CategoryBundleIntegrityError, match="report"):
        validate_category_bundle(output)


def test_pass_b_rebinds_assignments_to_exact_raw_product_terminals(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    selection_path = tmp_path / "scope-selection.json"
    _write_selection(selection_path, _selection_for_catalog(catalog))
    output = tmp_path / "candidate"
    write_category_candidate_bundle(
        catalog,
        selection_path,
        output,
        expected_product_count=3,
        enforce_official_gate=False,
    )

    assignment_path = output / "product-category-assignment.json"
    document = json.loads(assignment_path.read_bytes())
    document["assignments"][0]["parent_asin"] = "p-alien"
    tampered = canonical_json_bytes(document)
    assignment_path.write_bytes(tampered)
    manifest_path = output / CATEGORY_BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    assignment_entry = next(
        entry
        for entry in manifest["artifacts"]
        if entry["filename"] == "product-category-assignment.json"
    )
    assignment_entry["byte_size"] = len(tampered)
    assignment_entry["sha256"] = hashlib.sha256(tampered).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CategoryBundleIntegrityError, match="differ from exact raw catalog"):
        validate_category_bundle(output, catalog_path=catalog)


def test_pass_b_rebinds_registry_graph_to_exact_raw_catalog(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    selection_path = tmp_path / "scope-selection.json"
    _write_selection(selection_path, _selection_for_catalog(catalog))
    output = tmp_path / "candidate"
    write_category_candidate_bundle(
        catalog,
        selection_path,
        output,
        expected_product_count=3,
        enforce_official_gate=False,
    )

    original_registry = decode_category_registry((output / "category-registry.json").read_bytes())
    original_assignments = decode_product_category_assignment_set(
        (output / "product-category-assignment.json").read_bytes(),
        registry=original_registry,
    )
    extra_node = CategoryNode(
        id=category_node_id(("unobserved",)),
        parent_id=None,
        canonical_path=("unobserved",),
    )
    nodes = tuple(sorted((*original_registry.nodes, extra_node), key=lambda node: node.id))
    graph_id = category_graph_id(original_registry.catalog_id, nodes)
    original_root = next(
        scope for scope in original_registry.scopes if scope.id == original_registry.root_scope_id
    )
    root_scope = materialize_category_scope(
        graph_id=graph_id,
        nodes=nodes,
        label=original_root.label,
        root_node_ids=tuple(sorted(node.id for node in nodes if node.parent_id is None)),
    )
    registry = CategoryRegistry(
        schema=original_registry.schema,
        catalog_id=original_registry.catalog_id,
        category_graph_id=graph_id,
        root_scope_id=root_scope.id,
        nodes=nodes,
        scopes=(root_scope,),
    )
    assignments = replace(original_assignments, category_graph_id=graph_id)
    candidate = CategoryCandidateBuild(
        builder_version=CATEGORY_BUILDER_VERSION,
        registry=registry,
        assignments=assignments,
    )
    payloads = {
        "category-registry.json": encode_category_registry(registry),
        "product-category-assignment.json": encode_product_category_assignment_set(
            assignments,
            registry=registry,
        ),
        "reviewed-category-scopes.candidate.json": canonical_json_bytes(
            reviewed_category_scopes_candidate_document(candidate)
        ),
        "report.md": category_candidate_markdown(candidate).encode("utf-8"),
    }
    for filename, payload in payloads.items():
        (output / filename).write_bytes(payload)
    manifest_path = output / CATEGORY_BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["category_graph_id"] = graph_id
    for artifact in manifest["artifacts"]:
        payload = payloads[artifact["filename"]]
        artifact["byte_size"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CategoryBundleIntegrityError, match="registry graph differs"):
        validate_category_bundle(output, catalog_path=catalog)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("catalog_id", "sha256:" + "0" * 64, "catalog_id is stale"),
        ("category_graph_id", "sha256:" + "1" * 64, "category_graph_id is stale"),
        ("builder_version", "different_builder", "builder_version is unsupported"),
    ],
)
def test_pass_b_rejects_stale_selection_pins(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    catalog = _write_catalog(tmp_path)
    selection = _selection_for_catalog(catalog)
    document = json.loads(canonical_json_bytes(selection))
    document[field] = replacement
    selection_path = tmp_path / "stale-selection.json"
    selection_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(CategorySelectionError, match=message):
        write_category_candidate_bundle(
            catalog,
            selection_path,
            tmp_path / "candidate",
            expected_product_count=3,
            enforce_official_gate=False,
        )
