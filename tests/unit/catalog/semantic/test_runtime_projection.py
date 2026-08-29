from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_gate_b_approval import _build_approved

from shopping_copilot.catalog.semantic import (
    IJSON_SAFE_INTEGER_MAX,
    IJSON_SAFE_INTEGER_MIN,
    RuntimeProjectionBuildError,
    RuntimeProjectionBundleIntegrityError,
    RuntimeProjectionCodecError,
    content_id_for_value,
)
from shopping_copilot.catalog.semantic.category import decode_category_registry
from shopping_copilot.catalog.semantic.facet import load_gate_a_candidate_bundle
from shopping_copilot.catalog.semantic.runtime import (
    CATEGORY_SCOPE_ID_NORMALIZER_ID,
    RUNTIME_PROJECTION_ARTIFACT_FILENAMES,
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    ExactCapabilityIndex,
    build_runtime_projection_candidate,
    decode_runtime_facet_registry,
    decode_runtime_value_lexicon,
    encode_runtime_facet_registry,
    encode_runtime_value_lexicon,
    load_projected_session_facet_registry,
    load_runtime_projection_bundle,
    normalize_usd_cent_int_v1,
    project_session_facet_registry,
    validate_runtime_projection_bundle,
    write_runtime_projection_bundle,
)
from shopping_copilot.session_context import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    FacetAuthority,
    FacetKind,
    Operator,
    SessionContextError,
)


def _build_runtime(tmp_path: Path):
    approved = _build_approved(tmp_path)
    gate_b, _, _, category_dir, gate_a_dir, *_ = approved
    registry = decode_category_registry((category_dir / "category-registry.json").read_bytes())
    gate_a = load_gate_a_candidate_bundle(gate_a_dir)
    runtime = build_runtime_projection_candidate(
        registry=registry,
        gate_a=gate_a,
        gate_b=gate_b,
    )
    session_registry = project_session_facet_registry(
        runtime_registry=runtime.runtime_registry,
        runtime_lexicon=runtime.runtime_lexicon,
        category_registry=registry,
        capabilities=gate_b.capabilities,
    )
    return runtime, session_registry, registry, gate_a, gate_b


def test_runtime_projection_contains_price_and_reserved_category(tmp_path: Path) -> None:
    runtime, session_registry, registry, _, gate_b = _build_runtime(tmp_path)
    records = {item.facet_id: item for item in runtime.runtime_registry.entries}
    assert tuple(records) == ("price", SYSTEM_PRODUCT_CATEGORY_FACET_ID)
    assert records["price"].kind == "numeric"
    assert records["price"].operator_values == tuple(
        sorted(item.value for item in NUMERIC_OPERATORS)
    )
    assert records[SYSTEM_PRODUCT_CATEGORY_FACET_ID].kind == "categorical"
    assert records[SYSTEM_PRODUCT_CATEGORY_FACET_ID].operator_values == tuple(
        sorted(item.value for item in CATEGORICAL_OPERATORS)
    )
    assert (
        records[SYSTEM_PRODUCT_CATEGORY_FACET_ID].intent_value_normalizer_id
        == CATEGORY_SCOPE_ID_NORMALIZER_ID
    )
    assert len(runtime.runtime_lexicon.domains) == 1
    domain = runtime.runtime_lexicon.domains[0]
    assert (domain.facet_id, domain.canonical_unit, domain.integer_only) == (
        "price",
        "USD_CENT",
        True,
    )
    assert runtime.runtime_registry.effective_capabilities_id == content_id_for_value(
        gate_b.capabilities
    )
    assert session_registry.require("price").kind is FacetKind.NUMERIC
    assert session_registry.require(SYSTEM_PRODUCT_CATEGORY_FACET_ID).kind is FacetKind.CATEGORICAL
    assert session_registry.require("price").authority is FacetAuthority.CATALOG_VERIFIED
    assert (
        session_registry.require(SYSTEM_PRODUCT_CATEGORY_FACET_ID).authority
        is FacetAuthority.CATALOG_VERIFIED
    )
    assert session_registry.get("color") is None
    assert len(registry.scopes) == 1


@pytest.mark.parametrize(
    "value",
    [
        IJSON_SAFE_INTEGER_MIN,
        -1,
        0,
        1,
        IJSON_SAFE_INTEGER_MAX,
    ],
)
def test_usd_cent_normalizer_accepts_exact_safe_integer_fixed_points(value: int) -> None:
    assert normalize_usd_cent_int_v1(value) == value


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, -1.0, "2500", IJSON_SAFE_INTEGER_MIN - 1, IJSON_SAFE_INTEGER_MAX + 1],
)
def test_usd_cent_normalizer_rejects_every_other_input(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_usd_cent_int_v1(value)  # type: ignore[arg-type]


def test_session_registry_applies_release_bound_value_validation(tmp_path: Path) -> None:
    _, session_registry, registry, _, _ = _build_runtime(tmp_path)
    assert session_registry.normalize_value("price", Operator.LE, 2500) == 2500
    with pytest.raises(SessionContextError):
        session_registry.normalize_value("price", Operator.LE, 2500.0)
    scope_id = registry.scopes[0].id
    assert (
        session_registry.normalize_value(SYSTEM_PRODUCT_CATEGORY_FACET_ID, Operator.EQ, scope_id)
        == scope_id
    )
    with pytest.raises(SessionContextError):
        session_registry.normalize_value(
            SYSTEM_PRODUCT_CATEGORY_FACET_ID,
            Operator.EQ,
            "cs_" + "0" * 64,
        )


def test_exact_capability_lookup_has_no_inheritance_or_fallback(tmp_path: Path) -> None:
    _, _, registry, _, gate_b = _build_runtime(tmp_path)
    index = ExactCapabilityIndex(gate_b.capabilities)
    matched = index.lookup("price", registry.scopes[0].id)
    assert matched.matched
    assert (
        matched.intent_committable,
        matched.retrieval_eligible,
        matched.probe_eligible,
        matched.clarification_eligible,
    ) == (True, True, True, False)
    missing = index.lookup("price", "cs_" + "0" * 64)
    assert not missing.matched
    assert missing.decision is None
    assert not any(
        (
            missing.intent_committable,
            missing.retrieval_eligible,
            missing.probe_eligible,
            missing.clarification_eligible,
        )
    )
    unknown_facet = index.lookup("brand", registry.scopes[0].id)
    assert not unknown_facet.matched


def test_runtime_artifact_codecs_are_canonical_and_round_trip(tmp_path: Path) -> None:
    runtime, *_ = _build_runtime(tmp_path)
    registry_bytes = encode_runtime_facet_registry(runtime.runtime_registry)
    lexicon_bytes = encode_runtime_value_lexicon(runtime.runtime_lexicon)
    assert decode_runtime_facet_registry(registry_bytes) == runtime.runtime_registry
    assert decode_runtime_value_lexicon(lexicon_bytes) == runtime.runtime_lexicon
    with pytest.raises(RuntimeProjectionCodecError, match="canonical"):
        decode_runtime_facet_registry(registry_bytes + b"\n")


def test_runtime_projection_rejects_stale_capability_pin(tmp_path: Path) -> None:
    runtime, _, registry, _, gate_b = _build_runtime(tmp_path)
    stale_registry = replace(
        runtime.runtime_registry,
        effective_capabilities_id="sha256:" + "0" * 64,
    )
    with pytest.raises(RuntimeProjectionBuildError, match="capability pin"):
        project_session_facet_registry(
            runtime_registry=stale_registry,
            runtime_lexicon=runtime.runtime_lexicon,
            category_registry=registry,
            capabilities=gate_b.capabilities,
        )


def _build_runtime_bundle(tmp_path: Path):
    approved = _build_approved(tmp_path)
    (
        _,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        gate_b_review,
        gate_b_selection,
        gate_b_candidate,
    ) = approved
    output = tmp_path / "runtime-projection"
    build = write_runtime_projection_bundle(
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        gate_b_review,
        gate_b_selection,
        gate_b_candidate,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    return build, approved, output


def test_runtime_bundle_is_loadable_and_records_cs5b_grounding(
    tmp_path: Path,
) -> None:
    build, approved, output = _build_runtime_bundle(tmp_path)
    assert load_runtime_projection_bundle(output) == build
    document = json.loads((output / "candidate.json").read_bytes())
    assert document["grounding_implemented"] is True
    assert document["retrieval_integrated"] is False
    assert document["session_gateway_integrated"] is False
    manifest = json.loads((output / "bundle-manifest.json").read_bytes())
    assert manifest["grounding_implemented"] is True
    assert set(path.name for path in output.iterdir()) == {
        *RUNTIME_PROJECTION_ARTIFACT_FILENAMES,
        "bundle-manifest.json",
    }
    category = approved[3]
    gate_b_candidate = approved[9]
    session_registry = load_projected_session_facet_registry(
        output,
        category_candidate_dir=category,
        gate_b_candidate_dir=gate_b_candidate,
    )
    assert session_registry.normalize_value("price", Operator.LE, 2500) == 2500


def test_runtime_bundle_rebuild_is_byte_reproducible(tmp_path: Path) -> None:
    build, approved, output = _build_runtime_bundle(tmp_path)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    (
        _,
        _,
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        gate_b_review,
        gate_b_selection,
        gate_b_candidate,
    ) = approved
    validate_runtime_projection_bundle(
        output,
        catalog_path=catalog,
        category_candidate_dir=category,
        gate_a_candidate_dir=gate_a,
        resolution_candidate_dir=resolution,
        public_set_path=public_set,
        gate_b_review_dir=gate_b_review,
        gate_b_selection_path=gate_b_selection,
        gate_b_candidate_dir=gate_b_candidate,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    second = write_runtime_projection_bundle(
        catalog,
        category,
        gate_a,
        resolution,
        public_set,
        gate_b_review,
        gate_b_selection,
        gate_b_candidate,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    assert second == build
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first


def test_runtime_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    _, approved, output = _build_runtime_bundle(tmp_path)
    registry_path = output / "runtime-facet-registry.json"
    registry_path.write_bytes(registry_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeProjectionBundleIntegrityError):
        validate_runtime_projection_bundle(
            output,
            catalog_path=approved[2],
            category_candidate_dir=approved[3],
            gate_a_candidate_dir=approved[4],
            resolution_candidate_dir=approved[5],
            public_set_path=approved[6],
            gate_b_review_dir=approved[7],
            gate_b_selection_path=approved[8],
            gate_b_candidate_dir=approved[9],
            expected_product_count=5,
            expected_public_target_count=5,
            enforce_official_gate=False,
        )
