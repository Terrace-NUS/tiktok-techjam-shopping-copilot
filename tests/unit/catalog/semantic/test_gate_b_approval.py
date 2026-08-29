from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_gate_b import _build_gate_b

from shopping_copilot.catalog.semantic import (
    GateBBundleIntegrityError,
    GateBSelectionError,
    canonical_json_bytes,
    content_id_for_value,
)
from shopping_copilot.catalog.semantic.facet import RuntimePromotionDecision
from shopping_copilot.catalog.semantic.facet.gate_b_approval_bundle import (
    GATE_B_CANDIDATE_ARTIFACT_FILENAMES,
    load_gate_b_candidate_bundle,
    validate_gate_b_candidate_bundle,
    write_gate_b_candidate_bundle,
)
from shopping_copilot.catalog.semantic.facet.gate_b_approval_codec import (
    decode_effective_facet_capabilities,
    decode_gate_b_selection,
)
from shopping_copilot.catalog.semantic.facet.gate_b_models import (
    GATE_B_BUILDER_VERSION,
    GATE_B_SELECTION_SCHEMA,
    GateBSelection,
)


def _write_selection(path: Path, review_build) -> GateBSelection:
    proposal = review_build.proposal
    audit = review_build.public_target_audit
    selection = GateBSelection(
        schema=GATE_B_SELECTION_SCHEMA,
        builder_version=GATE_B_BUILDER_VERSION,
        catalog_id=proposal.catalog_id,
        category_registry_id=proposal.category_registry_id,
        facet_schema_id=proposal.facet_schema_id,
        facet_applicability_id=proposal.facet_applicability_id,
        product_facet_index_id=proposal.product_facet_index_id,
        catalog_facet_stats_id=proposal.catalog_facet_stats_id,
        gate_b_review_proposal_id=content_id_for_value(proposal),
        public_target_audit_id=content_id_for_value(audit),
        resolution_policy_id=proposal.resolution_policy_id,
        intent_value_normalizer_id=proposal.proposed_intent_normalizer_id,
        reviewed_value_aliases=proposal.reviewed_value_aliases,
        approvals=proposal.proposed_capabilities,
        rationale="Fixture owner approved the exact reviewed price proposal.",
    )
    path.write_bytes(canonical_json_bytes(selection))
    return selection


def _build_approved(tmp_path: Path):
    (
        review_build,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
    ) = _build_gate_b(tmp_path)
    selection_path = tmp_path / "gate-b-selection.json"
    selection = _write_selection(selection_path, review_build)
    output = tmp_path / "gate-b-candidate"
    candidate = write_gate_b_candidate_bundle(
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
        selection_path,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    return (
        candidate,
        selection,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
        selection_path,
        output,
    )


def test_owner_approval_materializes_normative_exact_scope_capabilities(tmp_path: Path) -> None:
    candidate, selection, *_, output = _build_approved(tmp_path)
    assert candidate.selection == selection
    assert candidate.capabilities.entries == selection.approvals
    assert len(candidate.capabilities.entries) == 1
    capability = candidate.capabilities.entries[0]
    assert capability.decision is RuntimePromotionDecision.RUNTIME_ACCEPT
    assert (
        capability.intent_committable,
        capability.retrieval_eligible,
        capability.probe_eligible,
        capability.clarification_eligible,
    ) == (True, True, True, False)
    document = json.loads((output / "candidate.json").read_bytes())
    assert document["owner_approval_recorded"] is True
    assert document["runtime_integration_complete"] is False
    assert set(path.name for path in output.iterdir()) == {
        *GATE_B_CANDIDATE_ARTIFACT_FILENAMES,
        "bundle-manifest.json",
    }
    assert (
        decode_gate_b_selection((output / "reviewed-gate-b-selection.json").read_bytes())
        == selection
    )
    assert (
        decode_effective_facet_capabilities(
            (output / "effective-facet-capabilities.json").read_bytes()
        )
        == candidate.capabilities
    )
    assert load_gate_b_candidate_bundle(output) == candidate


def test_approval_bundle_is_reproducible_and_exactly_revalidates(tmp_path: Path) -> None:
    (
        candidate,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
        selection_path,
        output,
    ) = _build_approved(tmp_path)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    validate_gate_b_candidate_bundle(
        output,
        catalog_path=catalog,
        category_candidate_dir=category,
        gate_a_candidate_dir=gate_a,
        resolution_candidate_dir=resolution,
        public_set_path=public_set,
        gate_b_review_dir=review_dir,
        gate_b_selection_path=selection_path,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    second = write_gate_b_candidate_bundle(
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
        selection_path,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    assert second == candidate
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first


def test_selection_cannot_silently_differ_from_reviewed_proposal(tmp_path: Path) -> None:
    (
        review_build,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
    ) = _build_gate_b(tmp_path)
    selection_path = tmp_path / "gate-b-selection.json"
    selection = _write_selection(selection_path, review_build)
    changed_capability = replace(
        selection.approvals[0],
        decision=RuntimePromotionDecision.SEARCH_ONLY,
        intent_committable=False,
        probe_eligible=False,
    )
    changed = replace(selection, approvals=(changed_capability,))
    selection_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(GateBSelectionError, match="differ from the owner-reviewed proposal"):
        write_gate_b_candidate_bundle(
            catalog,
            category,
            gate_a,
            resolution,
            public_set,
            review_dir,
            selection_path,
            tmp_path / "candidate",
            expected_product_count=5,
            expected_public_target_count=5,
            enforce_official_gate=False,
        )


def test_approved_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    (
        _,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
        selection_path,
        output,
    ) = _build_approved(tmp_path)
    capabilities = output / "effective-facet-capabilities.json"
    capabilities.write_bytes(capabilities.read_bytes() + b"tampered")
    with pytest.raises(GateBBundleIntegrityError):
        validate_gate_b_candidate_bundle(
            output,
            catalog_path=catalog,
            category_candidate_dir=category,
            gate_a_candidate_dir=gate_a,
            resolution_candidate_dir=resolution,
            public_set_path=public_set,
            gate_b_review_dir=review_dir,
            gate_b_selection_path=selection_path,
            expected_product_count=5,
            expected_public_target_count=5,
            enforce_official_gate=False,
        )


def test_approved_output_cannot_contain_an_input(tmp_path: Path) -> None:
    (
        review_build,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        review_dir,
    ) = _build_gate_b(tmp_path)
    selection_path = tmp_path / "gate-b-selection.json"
    _write_selection(selection_path, review_build)
    with pytest.raises(ValueError, match="must not contain"):
        write_gate_b_candidate_bundle(
            catalog,
            category,
            gate_a,
            resolution,
            public_set,
            review_dir,
            selection_path,
            tmp_path,
            expected_product_count=5,
            expected_public_target_count=5,
            enforce_official_gate=False,
        )
