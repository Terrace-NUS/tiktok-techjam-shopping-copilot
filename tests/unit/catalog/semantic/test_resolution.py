from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest
from test_gate_a import _build as _build_gate_a

from shopping_copilot.catalog.semantic import (
    ResolutionBundleIntegrityError,
    canonical_json_bytes,
)
from shopping_copilot.catalog.semantic.facet import (
    CATALOG_READ_ONLY_AUDIT_SCHEMA,
    RESOLUTION_ARTIFACT_FILENAMES,
    RESOLUTION_POLICY_ID,
    CatalogReadOnlyAudit,
    EvidenceStatus,
    FacetMatchResult,
    NumericValue,
    ProductFacetStatus,
    ResolvedProductFacetValue,
    canonical_raw_value_json,
    decode_catalog_facet_stats,
    decode_catalog_read_only_audit,
    decode_facet_evidence_store,
    decode_product_facet_index,
    evidence_id_for,
    lookup_product_facet,
    match_numeric_interval,
    safe_filter_keeps,
    validate_resolution_candidate_bundle,
    write_resolution_candidate_bundle,
)
from shopping_copilot.catalog.semantic.facet.gate_a_bundle import (
    load_gate_a_candidate_bundle,
)


def _build_resolution(tmp_path: Path):
    _, gate_a_inputs, gate_a_output = _build_gate_a(tmp_path)
    catalog = gate_a_inputs[0]
    category_candidate = gate_a_inputs[1]
    output = tmp_path / "resolution-candidate"
    before = catalog.read_bytes()
    candidate = write_resolution_candidate_bundle(
        catalog,
        category_candidate,
        gate_a_output,
        output,
        expected_product_count=5,
        enforce_official_gate=False,
    )
    assert catalog.read_bytes() == before
    return candidate, catalog, category_candidate, gate_a_output, output


def test_cs3_build_is_read_only_sparse_and_reproducible(tmp_path: Path) -> None:
    candidate, catalog, category_candidate, gate_a_output, output = _build_resolution(tmp_path)
    first = {path.name: path.read_bytes() for path in output.iterdir()}

    assert {item.status for item in candidate.evidence_store.evidence} == {
        EvidenceStatus.VALID,
        EvidenceStatus.EMPTY,
        EvidenceStatus.INVALID,
    }
    assert len(candidate.evidence_store.evidence) == 5
    assert len(candidate.product_facet_index.entries) == 3
    root = next(row for row in candidate.stats.rows if row.scope_product_count == 5)
    assert (root.known_count, root.unknown_count, root.conflict_count) == (3, 2, 0)
    assert root.not_applicable_count == 0

    validate_resolution_candidate_bundle(
        output,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        gate_a_candidate_dir=gate_a_output,
        expected_product_count=5,
        enforce_official_gate=False,
    )
    second = write_resolution_candidate_bundle(
        catalog,
        category_candidate,
        gate_a_output,
        output,
        expected_product_count=5,
        enforce_official_gate=False,
    )
    assert second == candidate
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    assert set(first) == {*RESOLUTION_ARTIFACT_FILENAMES, "bundle-manifest.json"}


def test_evidence_preserves_raw_json_and_full_payload_identity(tmp_path: Path) -> None:
    candidate, *_ = _build_resolution(tmp_path)
    evidence = {item.parent_asin: item for item in candidate.evidence_store.evidence}

    exact = evidence["p-exact"]
    assert exact.raw_value_json == "10.25"
    assert isinstance(exact.canonical_value, NumericValue)
    assert (exact.canonical_value.lower, exact.canonical_value.upper) == (1025, 1025)
    assert evidence["p-null"].raw_value_json == "null"
    assert evidence["p-invalid"].canonical_value is None
    changed_id = evidence_id_for(
        parent_asin=exact.parent_asin,
        facet_id=exact.facet_id,
        binding_id=exact.binding_id,
        status=EvidenceStatus.INVALID,
        raw_value_json=exact.raw_value_json,
        canonical_value=None,
    )
    assert changed_id != exact.id
    assert canonical_raw_value_json(-0.0) == "0"


def test_cs3_artifacts_round_trip_and_declare_no_gate_b_permission(tmp_path: Path) -> None:
    candidate, *_, output = _build_resolution(tmp_path)

    assert (
        decode_facet_evidence_store((output / "facet-evidence-store.json").read_bytes())
        == candidate.evidence_store
    )
    assert (
        decode_product_facet_index((output / "product-facet-index.json").read_bytes())
        == candidate.product_facet_index
    )
    assert (
        decode_catalog_facet_stats((output / "catalog-facet-stats.json").read_bytes())
        == candidate.stats
    )
    audit = decode_catalog_read_only_audit((output / "catalog-read-only-audit.json").read_bytes())
    assert audit == CatalogReadOnlyAudit(
        schema=CATALOG_READ_ONLY_AUDIT_SCHEMA,
        catalog_id_before=candidate.evidence_store.catalog_id,
        catalog_id_after_staging=candidate.evidence_store.catalog_id,
        byte_size_before=audit.byte_size_before,
        byte_size_after_staging=audit.byte_size_before,
        unchanged=True,
        output_is_separate=True,
    )
    document = json.loads((output / "candidate.json").read_bytes())
    assert document["gate_b_runtime_approved"] is False
    assert document["resolution_policy_id"] == RESOLUTION_POLICY_ID
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Raw catalog handling: **READ ONLY**" in report
    assert "Gate B runtime use: **NOT APPROVED**" in report


def test_safe_numeric_filter_never_drops_unknown_or_partial_overlap() -> None:
    budget = NumericValue(
        kind="numeric",
        lower=None,
        lower_inclusive=False,
        upper=1000,
        upper_inclusive=True,
        unit="USD_CENT",
    )
    unknown = ResolvedProductFacetValue(
        parent_asin="p-unknown",
        facet_id="price",
        status=ProductFacetStatus.UNKNOWN,
        value=None,
        evidence_ids=(),
        resolution_policy_id=RESOLUTION_POLICY_ID,
    )
    lower_bound = replace(
        unknown,
        status=ProductFacetStatus.KNOWN,
        value=NumericValue(
            kind="numeric",
            lower=500,
            lower_inclusive=True,
            upper=None,
            upper_inclusive=False,
            unit="USD_CENT",
        ),
        evidence_ids=("ev_" + "1" * 64,),
    )
    over_budget = replace(
        lower_bound,
        value=NumericValue(
            kind="numeric",
            lower=1200,
            lower_inclusive=True,
            upper=1200,
            upper_inclusive=True,
            unit="USD_CENT",
        ),
    )

    assert match_numeric_interval(unknown, budget) is FacetMatchResult.UNKNOWN
    assert safe_filter_keeps(match_numeric_interval(unknown, budget))
    assert match_numeric_interval(lower_bound, budget) is FacetMatchResult.UNKNOWN
    assert safe_filter_keeps(match_numeric_interval(lower_bound, budget))
    assert match_numeric_interval(over_budget, budget) is FacetMatchResult.VIOLATED
    assert not safe_filter_keeps(match_numeric_interval(over_budget, budget))


def test_sparse_lookup_derives_unknown_instead_of_violation(tmp_path: Path) -> None:
    candidate, _, category_candidate, gate_a_output, _ = _build_resolution(tmp_path)
    registry_bytes = (category_candidate / "category-registry.json").read_bytes()
    assignment_bytes = (category_candidate / "product-category-assignment.json").read_bytes()
    from shopping_copilot.catalog.semantic.category import (
        decode_category_registry,
        decode_product_category_assignment_set,
    )

    registry = decode_category_registry(registry_bytes)
    assignments = decode_product_category_assignment_set(assignment_bytes, registry=registry)
    gate_a = load_gate_a_candidate_bundle(gate_a_output)
    result = lookup_product_facet(
        "p-null",
        "price",
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        index=candidate.product_facet_index,
    )
    assert result.status is ProductFacetStatus.UNKNOWN
    assert result.value is None


def test_output_guard_refuses_to_contain_catalog_and_read_only_file_builds(
    tmp_path: Path,
) -> None:
    _, gate_a_inputs, gate_a_output = _build_gate_a(tmp_path)
    catalog = gate_a_inputs[0]
    category_candidate = gate_a_inputs[1]
    before = catalog.read_bytes()
    with pytest.raises(ValueError, match="must not contain"):
        write_resolution_candidate_bundle(
            catalog,
            category_candidate,
            gate_a_output,
            tmp_path,
            expected_product_count=5,
            enforce_official_gate=False,
        )
    assert catalog.read_bytes() == before

    catalog.chmod(stat.S_IREAD)
    try:
        write_resolution_candidate_bundle(
            catalog,
            category_candidate,
            gate_a_output,
            tmp_path / "resolution-read-only",
            expected_product_count=5,
            enforce_official_gate=False,
        )
    finally:
        catalog.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert catalog.read_bytes() == before


def test_cs3_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    _, catalog, category_candidate, gate_a_output, output = _build_resolution(tmp_path)
    stats = output / "catalog-facet-stats.json"
    stats.write_bytes(stats.read_bytes() + b"tampered")

    with pytest.raises(ResolutionBundleIntegrityError):
        validate_resolution_candidate_bundle(
            output,
            catalog_path=catalog,
            category_candidate_dir=category_candidate,
            gate_a_candidate_dir=gate_a_output,
            expected_product_count=5,
            enforce_official_gate=False,
        )


def test_read_only_audit_rejects_any_catalog_change_claim() -> None:
    with pytest.raises(ValueError, match="unchanged"):
        CatalogReadOnlyAudit(
            schema=CATALOG_READ_ONLY_AUDIT_SCHEMA,
            catalog_id_before="sha256:" + "0" * 64,
            catalog_id_after_staging="sha256:" + "1" * 64,
            byte_size_before=10,
            byte_size_after_staging=10,
            unchanged=True,
            output_is_separate=True,
        )


def test_candidate_json_is_canonical(tmp_path: Path) -> None:
    _, *_, output = _build_resolution(tmp_path)
    document = json.loads((output / "candidate.json").read_bytes())
    assert (output / "candidate.json").read_bytes() == canonical_json_bytes(document)
