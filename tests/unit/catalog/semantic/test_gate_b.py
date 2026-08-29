from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_resolution import _build_resolution

from shopping_copilot.catalog.semantic import (
    GateBReviewBundleIntegrityError,
    GateBReviewCodecError,
    canonical_json_bytes,
)
from shopping_copilot.catalog.semantic.facet import (
    GATE_B_REVIEW_ARTIFACT_FILENAMES,
    PRICE_INTENT_NORMALIZER_ID,
    EffectiveFacetCapability,
    FacetMatchResult,
    GateBReviewState,
    ProductFacetStatus,
    RuntimePromotionDecision,
    decode_gate_b_price_review,
    decode_public_target_price_audit,
    validate_gate_b_review_bundle,
    write_gate_b_review_bundle,
)


def _write_public_set(path: Path) -> None:
    products = ("p-exact", "p-from", "p-null", "p-invalid", "p-zero")
    rows = (
        {
            "sample_id": f"public_{index:04d}",
            "scenario_type": "buying" if index % 2 else "browsing",
            "ground_truth": {"parent_asin": parent_asin},
        }
        for index, parent_asin in enumerate(products, start=1)
    )
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _build_gate_b(tmp_path: Path):
    resolution, catalog, category_candidate, gate_a_candidate, resolution_dir = _build_resolution(
        tmp_path
    )
    public_set = tmp_path / "public-set.jsonl"
    _write_public_set(public_set)
    output = tmp_path / "gate-b-review"
    build = write_gate_b_review_bundle(
        catalog,
        category_candidate,
        gate_a_candidate,
        resolution_dir,
        public_set,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    return (
        build,
        resolution,
        catalog,
        category_candidate,
        gate_a_candidate,
        resolution_dir,
        public_set,
        output,
    )


def test_gate_b_packet_proposes_permissions_without_publishing_them(tmp_path: Path) -> None:
    build, _, catalog, category, gate_a, resolution_dir, public_set, output = _build_gate_b(
        tmp_path
    )
    proposal = build.proposal
    assert proposal.review_state is GateBReviewState.AWAITING_OWNER_APPROVAL
    assert proposal.proposed_intent_normalizer_id == PRICE_INTENT_NORMALIZER_ID
    assert proposal.reviewed_value_aliases == ()
    assert len(proposal.proposed_capabilities) == 1
    capability = proposal.proposed_capabilities[0]
    assert capability.decision is RuntimePromotionDecision.RUNTIME_ACCEPT
    assert (
        capability.intent_committable,
        capability.retrieval_eligible,
        capability.probe_eligible,
        capability.clarification_eligible,
    ) == (True, True, True, False)

    candidate = json.loads((output / "candidate.json").read_bytes())
    manifest = json.loads((output / "bundle-manifest.json").read_bytes())
    assert candidate["runtime_capability_published"] is False
    assert candidate["source_controlled_approval_present"] is False
    assert manifest["runtime_capability_published"] is False
    assert manifest["source_controlled_approval_present"] is False
    assert set(path.name for path in output.iterdir()) == {
        *GATE_B_REVIEW_ARTIFACT_FILENAMES,
        "bundle-manifest.json",
    }
    validate_gate_b_review_bundle(
        output,
        catalog_path=catalog,
        category_candidate_dir=category,
        gate_a_candidate_dir=gate_a,
        resolution_candidate_dir=resolution_dir,
        public_set_path=public_set,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )


def test_public_target_audit_keeps_unknown_and_partial_price_intervals(tmp_path: Path) -> None:
    build, *_ = _build_gate_b(tmp_path)
    audit = build.public_target_audit
    assert (audit.known_count, audit.unknown_count) == (3, 2)
    assert (audit.conflict_count, audit.not_applicable_count) == (0, 0)
    assert audit.compatible_budget_safe_retained_count == 5
    assert audit.unsafe_satisfied_only_retained_count == 2
    records = {item.parent_asin: item for item in audit.records}
    assert records["p-null"].price_status is ProductFacetStatus.UNKNOWN
    assert records["p-null"].safe_match_result is FacetMatchResult.UNKNOWN
    assert records["p-null"].safe_retained
    assert records["p-from"].safe_match_result is FacetMatchResult.UNKNOWN
    assert records["p-from"].safe_retained
    assert not records["p-from"].unsafe_satisfied_only_retained

    ten_dollars = build.proposal.budget_safety_rows[0]
    assert ten_dollars.budget_cents == 1000
    assert (
        ten_dollars.satisfied_count,
        ten_dollars.violated_count,
        ten_dollars.unknown_count,
        ten_dollars.safe_retained_count,
    ) == (1, 2, 2, 3)


def test_gate_b_codecs_round_trip_and_reject_noncanonical_bytes(tmp_path: Path) -> None:
    build, *_, output = _build_gate_b(tmp_path)
    proposal_bytes = (output / "price-review-proposal.json").read_bytes()
    audit_bytes = (output / "public-target-audit.json").read_bytes()
    assert decode_gate_b_price_review(proposal_bytes) == build.proposal
    assert decode_public_target_price_audit(audit_bytes) == build.public_target_audit
    with pytest.raises(GateBReviewCodecError, match="canonical"):
        decode_gate_b_price_review(proposal_bytes + b"\n")


def test_gate_b_bundle_is_byte_reproducible_and_tampering_fails(tmp_path: Path) -> None:
    build, _, catalog, category, gate_a, resolution_dir, public_set, output = _build_gate_b(
        tmp_path
    )
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    second = write_gate_b_review_bundle(
        catalog,
        category,
        gate_a,
        resolution_dir,
        public_set,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    assert second == build
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    proposal_path = output / "price-review-proposal.json"
    proposal_path.write_bytes(proposal_path.read_bytes() + b"tampered")
    with pytest.raises(GateBReviewBundleIntegrityError):
        validate_gate_b_review_bundle(
            output,
            catalog_path=catalog,
            category_candidate_dir=category,
            gate_a_candidate_dir=gate_a,
            resolution_candidate_dir=resolution_dir,
            public_set_path=public_set,
            expected_product_count=5,
            expected_public_target_count=5,
            enforce_official_gate=False,
        )


def test_capability_implications_fail_closed(tmp_path: Path) -> None:
    build, *_ = _build_gate_b(tmp_path)
    accepted = build.proposal.proposed_capabilities[0]
    with pytest.raises(ValueError, match="Probe eligibility"):
        replace(accepted, retrieval_eligible=False)
    with pytest.raises(ValueError, match="clarification eligibility"):
        replace(accepted, probe_eligible=False, clarification_eligible=True)
    with pytest.raises(ValueError, match="only RUNTIME_ACCEPT"):
        replace(accepted, decision=RuntimePromotionDecision.SEARCH_ONLY)
    with pytest.raises(ValueError, match="all capabilities"):
        EffectiveFacetCapability(
            facet_id=accepted.facet_id,
            category_scope_id=accepted.category_scope_id,
            decision=RuntimePromotionDecision.SEMANTIC_ONLY,
            resolution_policy_id=accepted.resolution_policy_id,
            intent_committable=False,
            retrieval_eligible=True,
            probe_eligible=False,
            clarification_eligible=False,
        )


def test_gate_b_output_cannot_contain_any_input(tmp_path: Path) -> None:
    _, catalog, category, gate_a, resolution_dir = _build_resolution(tmp_path)
    public_set = tmp_path / "public-set.jsonl"
    _write_public_set(public_set)
    with pytest.raises(ValueError, match="must not contain"):
        write_gate_b_review_bundle(
            catalog,
            category,
            gate_a,
            resolution_dir,
            public_set,
            tmp_path,
            expected_product_count=5,
            expected_public_target_count=5,
            enforce_official_gate=False,
        )
