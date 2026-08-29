from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic import (
    GateABundleIntegrityError,
    GateACodecError,
    GateASelectionError,
    canonical_json_bytes,
    content_id_for_bytes,
)
from shopping_copilot.catalog.semantic.category import (
    CATEGORY_SCOPE_SELECTION_SCHEMA,
    CategoryScopeSelection,
    CategoryScopeSelectionDocument,
    build_category_graph_proposal,
    write_category_candidate_bundle,
)
from shopping_copilot.catalog.semantic.facet import (
    CATALOG_VALUE_NORMALIZER_REGISTRY,
    EXTRACTOR_REGISTRY,
    FACET_PROFILE_BUILDER_VERSION,
    GATE_A_BUILDER_VERSION,
    GATE_A_PROFILE_SELECTION_SCHEMA,
    GATE_A_SELECTION_SCHEMA,
    RESOLVER_REGISTRY,
    CatalogFacetDefinition,
    CategoricalValue,
    EvidenceStatus,
    FacetApplicability,
    FacetDataType,
    FacetSourceBinding,
    GateADecision,
    GateAFacetApproval,
    GateAProfileSelection,
    GateASelection,
    ItemCardinality,
    NumericValue,
    PriceExtractionExpectation,
    PriceNormalizationLane,
    ProductFacetStatus,
    SourceKind,
    SourceLocator,
    ValueCompleteness,
    decode_gate_a_selection,
    normalize_usd_cent_interval_v1,
    resolve_priority_exact_v1,
    validate_gate_a_candidate_bundle,
    write_gate_a_candidate_bundle,
    write_gate_a_source_profile_bundle,
)
from shopping_copilot.catalog.semantic.raw_catalog import scan_raw_catalog


def _write_catalog(path: Path) -> None:
    rows = (
        {
            "parent_asin": "p-exact",
            "categories": ["Root", "Apparel"],
            "details": {"Color": "Red"},
            "price": 10.25,
            "store": "Store A",
        },
        {
            "parent_asin": "p-from",
            "categories": ["Root", "Shoes"],
            "details": {"Material": "Leather"},
            "price": "from 12.00",
            "store": "Store B",
        },
        {
            "parent_asin": "p-null",
            "categories": ["Root", "Bags"],
            "details": {"Capacity": "Small"},
            "price": None,
            "store": None,
        },
        {
            "parent_asin": "p-invalid",
            "categories": ["Root", "Apparel"],
            "details": {"Color": "Blue"},
            "price": "—",
            "store": "Store C",
        },
        {
            "parent_asin": "p-zero",
            "categories": ["Root", "Shoes"],
            "details": {"Color": "Black"},
            "price": 0,
            "store": "Store D",
        },
    )
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    scan = scan_raw_catalog(catalog, expected_product_count=5)
    proposal = build_category_graph_proposal(scan)
    root_node_ids = tuple(sorted(node.id for node in proposal.nodes if node.parent_id is None))
    category_selection = CategoryScopeSelectionDocument(
        schema=CATEGORY_SCOPE_SELECTION_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_graph_id=proposal.category_graph_id,
        builder_version=proposal.builder_version,
        scopes=(
            CategoryScopeSelection(
                label="All products",
                root_node_ids=root_node_ids,
            ),
        ),
    )
    category_selection_path = tmp_path / "category-selection.json"
    category_selection_path.write_bytes(canonical_json_bytes(category_selection))
    category_candidate = tmp_path / "category-candidate"
    category_build = write_category_candidate_bundle(
        catalog,
        category_selection_path,
        category_candidate,
        expected_product_count=5,
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
        sample_seed="gate-a-test-v0",
        sample_limit=2,
        top_value_limit=5,
    )
    profile_selection_path = tmp_path / "profile-selection.json"
    profile_selection_path.write_bytes(canonical_json_bytes(profile_selection))
    source_profile = tmp_path / "source-profile"
    write_gate_a_source_profile_bundle(
        catalog,
        category_candidate,
        profile_selection_path,
        source_profile,
        expected_product_count=5,
        enforce_official_gate=False,
    )
    source_profile_hash = hashlib.sha256(
        (source_profile / "bundle-manifest.json").read_bytes()
    ).hexdigest()
    root_scope_id = category_build.registry.root_scope_id
    binding = FacetSourceBinding(
        id="price_top_level_v1",
        facet_id="price",
        source=SourceLocator(kind=SourceKind.TOP_LEVEL, key="price"),
        applicable_category_scope_ids=(root_scope_id,),
        extractor_id="top_level_price_usd_v1",
        catalog_value_normalizer_id="usd_cent_interval_v1",
        priority=0,
        completeness=ValueCompleteness.COMPLETE,
        resolver_id="priority_exact_v1",
    )
    gate_a_selection = GateASelection(
        schema=GATE_A_SELECTION_SCHEMA,
        catalog_id=proposal.catalog_id,
        category_registry_id=content_id_for_bytes(registry_bytes),
        product_category_assignment_id=content_id_for_bytes(assignment_bytes),
        source_profile_manifest_sha256=source_profile_hash,
        builder_version=GATE_A_BUILDER_VERSION,
        approvals=(
            GateAFacetApproval(
                decision=GateADecision.EXTRACTION_APPROVED,
                definition=CatalogFacetDefinition(
                    id="price",
                    name="Price",
                    data_type=FacetDataType.NUMERIC,
                    item_cardinality=ItemCardinality.SINGLE,
                ),
                applicability=FacetApplicability(
                    facet_id="price",
                    category_scope_ids=(root_scope_id,),
                ),
                bindings=(binding,),
                extraction_expectation=PriceExtractionExpectation(
                    product_count=5,
                    source_present_count=5,
                    source_missing_count=0,
                    valid_count=3,
                    empty_count=1,
                    invalid_count=1,
                    exact_interval_count=2,
                    lower_bound_interval_count=1,
                    zero_exact_count=1,
                ),
                rationale="Exact deterministic test price rule; runtime remains unapproved.",
            ),
        ),
    )
    gate_a_selection_path = tmp_path / "gate-a-selection.json"
    gate_a_selection_path.write_bytes(canonical_json_bytes(gate_a_selection))
    return (
        catalog,
        category_candidate,
        profile_selection_path,
        source_profile,
        gate_a_selection_path,
    )


def _build(tmp_path: Path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "gate-a-candidate"
    candidate = write_gate_a_candidate_bundle(
        *inputs,
        output,
        expected_product_count=5,
        enforce_official_gate=False,
    )
    return candidate, inputs, output


@pytest.mark.parametrize(
    ("raw", "status", "lane", "lower", "upper"),
    [
        (Decimal("12.99"), EvidenceStatus.VALID, PriceNormalizationLane.EXACT, 1299, 1299),
        (0, EvidenceStatus.VALID, PriceNormalizationLane.EXACT, 0, 0),
        (
            "from 12.99",
            EvidenceStatus.VALID,
            PriceNormalizationLane.LOWER_BOUND,
            1299,
            None,
        ),
        (None, EvidenceStatus.EMPTY, PriceNormalizationLane.EMPTY, None, None),
        ("—", EvidenceStatus.INVALID, PriceNormalizationLane.INVALID, None, None),
        (12.99, EvidenceStatus.INVALID, PriceNormalizationLane.INVALID, None, None),
        (Decimal("1.001"), EvidenceStatus.INVALID, PriceNormalizationLane.INVALID, None, None),
        (Decimal("-1"), EvidenceStatus.INVALID, PriceNormalizationLane.INVALID, None, None),
        (Decimal("-0"), EvidenceStatus.INVALID, PriceNormalizationLane.INVALID, None, None),
        ("from $12.99", EvidenceStatus.INVALID, PriceNormalizationLane.INVALID, None, None),
    ],
)
def test_price_normalizer_is_exact_and_never_rounds(
    raw: object,
    status: EvidenceStatus,
    lane: PriceNormalizationLane,
    lower: int | None,
    upper: int | None,
) -> None:
    result = normalize_usd_cent_interval_v1(raw)

    assert result.status is status
    assert result.lane is lane
    if result.value is None:
        assert lower is None and upper is None
    else:
        assert result.value.lower == lower
        assert result.value.upper == upper
        assert result.value.unit == "USD_CENT"


def test_closed_implementation_registries_have_no_fallback() -> None:
    assert tuple(EXTRACTOR_REGISTRY) == ("top_level_price_usd_v1",)
    assert tuple(CATALOG_VALUE_NORMALIZER_REGISTRY) == ("usd_cent_interval_v1",)
    assert tuple(RESOLVER_REGISTRY) == ("priority_exact_v1",)
    with pytest.raises(TypeError):
        EXTRACTOR_REGISTRY["other"] = lambda row: row  # type: ignore[index, assignment]


def test_priority_exact_resolver_agrees_conflicts_and_preserves_completeness() -> None:
    exact = NumericValue(
        kind="numeric",
        lower=1299,
        lower_inclusive=True,
        upper=1299,
        upper_inclusive=True,
        unit="USD_CENT",
    )
    other = replace(exact, lower=1300, upper=1300)
    assert resolve_priority_exact_v1((exact, exact)).status is ProductFacetStatus.KNOWN
    assert resolve_priority_exact_v1((exact, other)).status is ProductFacetStatus.CONFLICT

    partial = CategoricalValue(
        kind="categorical",
        values=("red",),
        completeness=ValueCompleteness.PARTIAL,
    )
    complete = replace(partial, completeness=ValueCompleteness.COMPLETE)
    resolved = resolve_priority_exact_v1((partial, complete))
    assert resolved.status is ProductFacetStatus.KNOWN
    assert isinstance(resolved.value, CategoricalValue)
    assert resolved.value.completeness is ValueCompleteness.COMPLETE


def test_gate_a_candidate_is_pinned_reproducible_and_runtime_unapproved(
    tmp_path: Path,
) -> None:
    candidate, inputs, output = _build(tmp_path)
    first = {path.name: path.read_bytes() for path in output.iterdir()}

    second = write_gate_a_candidate_bundle(
        *inputs,
        output,
        expected_product_count=5,
        enforce_official_gate=False,
    )
    validate_gate_a_candidate_bundle(
        output,
        catalog_path=inputs[0],
        category_candidate_dir=inputs[1],
        profile_selection_path=inputs[2],
        source_profile_dir=inputs[3],
        gate_a_selection_path=inputs[4],
        expected_product_count=5,
        enforce_official_gate=False,
    )

    assert second == candidate
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    assert candidate.price_audits[0].valid_count == 3
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "EXTRACTION_APPROVED" in report
    assert "Runtime / Gate B decision: **NOT APPROVED**" in report
    assert "budget filtering" in report


def test_gate_a_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    _, inputs, output = _build(tmp_path)
    report = output / "report.md"
    report.write_bytes(report.read_bytes() + b"tampered")

    with pytest.raises(GateABundleIntegrityError):
        validate_gate_a_candidate_bundle(
            output,
            catalog_path=inputs[0],
            category_candidate_dir=inputs[1],
            profile_selection_path=inputs[2],
            source_profile_dir=inputs[3],
            gate_a_selection_path=inputs[4],
            expected_product_count=5,
            enforce_official_gate=False,
        )


def test_stale_source_profile_pin_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    selection = decode_gate_a_selection(inputs[4].read_bytes())
    inputs[4].write_bytes(
        canonical_json_bytes(replace(selection, source_profile_manifest_sha256="0" * 64))
    )

    with pytest.raises(GateASelectionError, match="manifest pin"):
        write_gate_a_candidate_bundle(
            *inputs,
            tmp_path / "gate-a-candidate",
            expected_product_count=5,
            enforce_official_gate=False,
        )


def test_gate_a_codec_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    *_, selection_path = _inputs(tmp_path)
    raw = selection_path.read_text(encoding="utf-8")
    duplicated = raw.replace(
        '"catalog_id":',
        '"catalog_id":"sha256:' + "0" * 64 + '","catalog_id":',
    )
    with pytest.raises(GateACodecError, match="duplicate"):
        decode_gate_a_selection(duplicated.encode())

    document = json.loads(raw)
    document["unexpected"] = True
    with pytest.raises(GateACodecError, match="invalid fields"):
        decode_gate_a_selection(json.dumps(document).encode())
