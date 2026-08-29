from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from test_runtime_projection import _build_runtime_bundle

from shopping_copilot.catalog.semantic import (
    ReleaseBundleIntegrityError,
    ReleaseCodecError,
    content_id_for_value,
)
from shopping_copilot.catalog.semantic.release import (
    ARTIFACT_KINDS,
    ARTIFACT_SPEC,
    RELEASE_MANIFEST_FILENAME,
    decode_release_manifest,
    encode_release_manifest,
    load_catalog_semantic_release,
    release_id_for_manifest,
    validate_catalog_semantic_release,
    write_catalog_semantic_release,
)
from shopping_copilot.catalog.semantic.runtime import (
    ExtractedRuntimeValueCandidate,
    GroundingDisposition,
)
from shopping_copilot.session_context import Operator, SemanticPolarity


@pytest.fixture(scope="module")
def release_case(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("release-case")
    _, approved, runtime = _build_runtime_bundle(root)
    output = root / "release"
    release = write_catalog_semantic_release(
        approved[2],
        approved[3],
        approved[4],
        approved[5],
        approved[6],
        approved[7],
        approved[8],
        approved[9],
        runtime,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    return release, approved, runtime, output


def _copy_release(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def test_release_contains_exactly_thirteen_refs_and_self_verifies(release_case) -> None:
    release, approved, _, output = release_case
    assert tuple(item.kind for item in release.manifest.artifacts) == ARTIFACT_KINDS
    assert release.release_id == release_id_for_manifest(release.manifest)
    assert release.manifest.reviewed_config_id == content_id_for_value(release.reviewed_config)
    assert {path.name for path in output.iterdir()} == {
        RELEASE_MANIFEST_FILENAME,
        *(spec[2] for spec in ARTIFACT_SPEC.values()),
    }
    assert (output / "catalog.jsonl").read_bytes() == approved[2].read_bytes()
    assert (
        validate_catalog_semantic_release(
            output,
            expected_release_id=release.release_id,
            expected_product_count=5,
        )
        == release.release_id
    )


def test_verified_release_exposes_the_cs5b_grounder(release_case) -> None:
    release, _, _, _ = release_case
    scope_id = release.category_registry.scopes[0].id
    result = release.grounder.ground(
        ExtractedRuntimeValueCandidate(
            facet_id="price",
            operator=Operator.LE,
            value=2500,
            alternative_values=(),
            semantic_text="不超过 25 美元",
            semantic_polarity=SemanticPolarity.POSITIVE,
        ),
        final_category_scope_id=scope_id,
    )
    assert result.disposition is GroundingDisposition.GROUNDED


def test_republishing_same_generation_is_idempotent(release_case) -> None:
    first, approved, runtime, output = release_case
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    second = write_catalog_semantic_release(
        approved[2],
        approved[3],
        approved[4],
        approved[5],
        approved[6],
        approved[7],
        approved[8],
        approved[9],
        runtime,
        output,
        expected_product_count=5,
        expected_public_target_count=5,
        enforce_official_gate=False,
    )
    assert second.release_id == first.release_id
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_release_manifest_codec_is_strict_and_canonical(release_case) -> None:
    release, _, _, _ = release_case
    encoded = encode_release_manifest(release.manifest)
    assert decode_release_manifest(encoded) == release.manifest
    with pytest.raises(ReleaseCodecError, match="canonical"):
        decode_release_manifest(encoded + b"\n")


def test_wrong_expected_release_id_fails_closed(release_case) -> None:
    _, _, _, output = release_case
    with pytest.raises(ReleaseBundleIntegrityError, match="expected ID"):
        load_catalog_semantic_release(
            output,
            expected_release_id="sha256:" + "0" * 64,
            expected_product_count=5,
        )


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("runtime-value-lexicon.json", "tamper"),
        ("reviewed-semantic-config.json", "tamper"),
        (RELEASE_MANIFEST_FILENAME, "tamper"),
    ],
)
def test_any_material_byte_tampering_is_rejected(
    release_case,
    tmp_path: Path,
    filename: str,
    mutation: str,
) -> None:
    _, _, _, output = release_case
    copied = _copy_release(output, tmp_path / mutation)
    path = copied / filename
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ReleaseBundleIntegrityError):
        load_catalog_semantic_release(copied, expected_product_count=5)


def test_missing_or_extra_member_is_rejected(release_case, tmp_path: Path) -> None:
    _, _, _, output = release_case
    missing = _copy_release(output, tmp_path / "missing")
    (missing / "facet-stats-never-exists.json").write_text("unused", encoding="utf-8")
    (missing / "catalog-facet-stats.json").unlink()
    with pytest.raises(ReleaseBundleIntegrityError, match="members"):
        load_catalog_semantic_release(missing, expected_product_count=5)

    extra = _copy_release(output, tmp_path / "extra")
    (extra / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ReleaseBundleIntegrityError, match="members"):
        load_catalog_semantic_release(extra, expected_product_count=5)
