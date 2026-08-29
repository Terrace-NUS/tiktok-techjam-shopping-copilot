from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic import (
    FacetProfileBundleIntegrityError,
    FacetProfileCodecError,
    FacetProfileSelectionError,
    canonical_json_bytes,
    content_id_for_bytes,
)
from shopping_copilot.catalog.semantic.category import (
    CATEGORY_SCOPE_SELECTION_SCHEMA,
    CategoryScopeSelection,
    CategoryScopeSelectionDocument,
    build_category_graph_proposal,
    category_node_id,
    normalize_category_path,
    write_category_candidate_bundle,
)
from shopping_copilot.catalog.semantic.facet import (
    FACET_PROFILE_BUILDER_VERSION,
    GATE_A_PROFILE_SELECTION_SCHEMA,
    GateAProfileSelection,
    SourceKind,
    SourceLocator,
    decode_profile_selection,
    validate_gate_a_source_profile_bundle,
    write_gate_a_source_profile_bundle,
)
from shopping_copilot.catalog.semantic.facet.cli import main
from shopping_copilot.catalog.semantic.raw_catalog import scan_raw_catalog


def _write_catalog(path: Path) -> None:
    rows = (
        {
            "parent_asin": "p-shirt",
            "categories": ["Root", "Apparel"],
            "details": {"": "unusable-key", "Color": "Red"},
            "price": 10.25,
            "store": "Store A",
            "title": "shirt",
        },
        {
            "parent_asin": "p-shoe",
            "categories": ["Root", "Shoes"],
            "details": {"Color": "  ", "Material": "Leather"},
            "price": "from 12.00",
            "store": None,
            "title": "shoe",
        },
        {
            "parent_asin": "p-bag",
            "categories": ["Root", "Bags"],
            "details": {"Capacity": [], "Color": None},
            "price": None,
            "store": "Store B",
            "title": "bag",
        },
    )
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    scan = scan_raw_catalog(catalog, expected_product_count=3)
    proposal = build_category_graph_proposal(scan)
    root_ids = tuple(sorted(node.id for node in proposal.nodes if node.parent_id is None))
    apparel_id = category_node_id(normalize_category_path(("Root", "Apparel")))
    category_selection = CategoryScopeSelectionDocument(
        schema=CATEGORY_SCOPE_SELECTION_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_graph_id=proposal.category_graph_id,
        builder_version=proposal.builder_version,
        scopes=(
            CategoryScopeSelection(label="All products", root_node_ids=root_ids),
            CategoryScopeSelection(label="Apparel", root_node_ids=(apparel_id,)),
        ),
    )
    category_selection_path = tmp_path / "category-selection.json"
    category_selection_path.write_bytes(canonical_json_bytes(category_selection))
    category_candidate = tmp_path / "category-candidate"
    write_category_candidate_bundle(
        catalog,
        category_selection_path,
        category_candidate,
        expected_product_count=3,
        enforce_official_gate=False,
    )
    registry_bytes = (category_candidate / "category-registry.json").read_bytes()
    assignment_bytes = (category_candidate / "product-category-assignment.json").read_bytes()
    profile_selection = GateAProfileSelection(
        schema=GATE_A_PROFILE_SELECTION_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_registry_id=content_id_for_bytes(registry_bytes),
        product_category_assignment_id=content_id_for_bytes(assignment_bytes),
        builder_version=FACET_PROFILE_BUILDER_VERSION,
        top_level_keys=("price", "store"),
        include_all_details=True,
        sample_seed="test-profile-v0",
        sample_limit=2,
        top_value_limit=3,
    )
    profile_selection_path = tmp_path / "profile-selection.json"
    profile_selection_path.write_bytes(canonical_json_bytes(profile_selection))
    return catalog, category_candidate, profile_selection_path


def _build(tmp_path: Path):
    catalog, category_candidate, profile_selection = _inputs(tmp_path)
    output = tmp_path / "profile"
    build = write_gate_a_source_profile_bundle(
        catalog,
        category_candidate,
        profile_selection,
        output,
        expected_product_count=3,
        enforce_official_gate=False,
    )
    return build, catalog, category_candidate, profile_selection, output


def test_profile_is_exhaustive_category_conditioned_and_keeps_empty_raw_key(
    tmp_path: Path,
) -> None:
    build, *_ = _build(tmp_path)

    assert len(build.scopes) == 2
    assert len(build.sources) == 6
    assert len(build.scope_source_profiles) == 12
    assert SourceLocator(kind=SourceKind.DETAILS, key="") in build.sources
    root = next(scope for scope in build.scopes if scope.is_root)
    apparel = next(scope for scope in build.scopes if scope.label == "Apparel")
    color = SourceLocator(kind=SourceKind.DETAILS, key="Color")
    root_color = next(
        item
        for item in build.scope_source_profiles
        if item.source == color and item.category_scope_id == root.category_scope_id
    )
    apparel_color = next(
        item
        for item in build.scope_source_profiles
        if item.source == color and item.category_scope_id == apparel.category_scope_id
    )
    assert (
        root_color.product_count,
        root_color.present_count,
        root_color.nonempty_count,
        root_color.null_count,
        root_color.empty_count,
    ) == (3, 3, 1, 1, 1)
    assert root_color.distinct_value_count == 3
    assert root_color.distinct_nonempty_value_count == 1
    assert apparel_color.product_count == apparel_color.nonempty_count == 1


def test_price_audit_never_interprets_string_lane(tmp_path: Path) -> None:
    build, *_ = _build(tmp_path)

    assert build.price_audit.present_count == 3
    assert build.price_audit.null_count == 1
    assert build.price_audit.numeric_count == 1
    assert build.price_audit.numeric_exact_cent_count == 1
    assert build.price_audit.numeric_non_cent_count == 0
    assert build.price_audit.minimum_exact_cents == 1025
    assert build.price_audit.maximum_exact_cents == 1025
    assert build.price_audit.string_count == 1
    assert build.price_audit.string_values[0].canonical_value_json == '"from 12.00"'


def test_profile_bundle_is_reproducible_and_revalidates_upstream_truth(tmp_path: Path) -> None:
    build, catalog, category_candidate, selection, output = _build(tmp_path)
    first = {path.name: path.read_bytes() for path in output.iterdir()}

    second_build = write_gate_a_source_profile_bundle(
        catalog,
        category_candidate,
        selection,
        output,
        expected_product_count=3,
        enforce_official_gate=False,
    )
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    validate_gate_a_source_profile_bundle(
        output,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        selection_path=selection,
        expected_product_count=3,
        enforce_official_gate=False,
    )

    assert second_build == build
    assert second == first


def test_profile_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    _, catalog, category_candidate, selection, output = _build(tmp_path)
    report = output / "report.md"
    report.write_bytes(report.read_bytes() + b"tampered")

    with pytest.raises(FacetProfileBundleIntegrityError):
        validate_gate_a_source_profile_bundle(
            output,
            catalog_path=catalog,
            category_candidate_dir=category_candidate,
            selection_path=selection,
            expected_product_count=3,
            enforce_official_gate=False,
        )


def test_stale_category_registry_pin_fails_closed(tmp_path: Path) -> None:
    catalog, category_candidate, selection_path = _inputs(tmp_path)
    selection = decode_profile_selection(selection_path.read_bytes())
    stale = replace(selection, category_registry_id="sha256:" + "0" * 64)
    selection_path.write_bytes(canonical_json_bytes(stale))

    with pytest.raises(FacetProfileSelectionError, match="CategoryRegistry pin"):
        write_gate_a_source_profile_bundle(
            catalog,
            category_candidate,
            selection_path,
            tmp_path / "profile",
            expected_product_count=3,
            enforce_official_gate=False,
        )


def test_unobserved_selected_top_level_key_fails_closed(tmp_path: Path) -> None:
    catalog, category_candidate, selection_path = _inputs(tmp_path)
    selection = decode_profile_selection(selection_path.read_bytes())
    invalid = replace(selection, top_level_keys=("missing", "price", "store"))
    selection_path.write_bytes(canonical_json_bytes(invalid))

    with pytest.raises(FacetProfileSelectionError, match="not observed"):
        write_gate_a_source_profile_bundle(
            catalog,
            category_candidate,
            selection_path,
            tmp_path / "profile",
            expected_product_count=3,
            enforce_official_gate=False,
        )


def test_official_gate_cannot_skip_details_or_change_product_count(tmp_path: Path) -> None:
    catalog, category_candidate, selection_path = _inputs(tmp_path)
    selection = decode_profile_selection(selection_path.read_bytes())
    selection_path.write_bytes(canonical_json_bytes(replace(selection, include_all_details=False)))

    with pytest.raises(ValueError, match="50,000"):
        write_gate_a_source_profile_bundle(
            catalog,
            category_candidate,
            selection_path,
            tmp_path / "profile",
            expected_product_count=3,
            enforce_official_gate=True,
        )


def test_selection_codec_rejects_duplicate_and_unknown_fields(tmp_path: Path) -> None:
    _, _, selection_path = _inputs(tmp_path)
    raw = selection_path.read_text(encoding="utf-8")
    duplicated = raw.replace(
        '"catalog_id":', '"catalog_id":"sha256:' + "0" * 64 + '","catalog_id":'
    )
    with pytest.raises(FacetProfileCodecError, match="duplicate"):
        decode_profile_selection(duplicated.encode())

    document = json.loads(raw)
    document["unexpected"] = True
    with pytest.raises(FacetProfileCodecError, match="invalid fields"):
        decode_profile_selection(json.dumps(document).encode())


def test_cli_enforces_official_product_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog, category_candidate, selection = _inputs(tmp_path)
    output = tmp_path / "profile"

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "build",
                str(catalog),
                str(category_candidate),
                str(selection),
                str(output),
            ]
        )
    assert raised.value.code == 1
    assert "catalog-facet-profile: error:" in capsys.readouterr().err


def test_report_preserves_approval_boundary(tmp_path: Path) -> None:
    _, _, _, _, output = _build(tmp_path)
    report = (output / "report.md").read_text(encoding="utf-8")

    assert "all remain `NEEDS_REVIEW`" in report
    assert "No facet, applicability, binding, extractor" in report
    assert "does not strip currency text" in report
