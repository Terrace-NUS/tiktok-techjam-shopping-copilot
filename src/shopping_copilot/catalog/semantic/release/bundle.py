"""Atomic CS6 release publication and self-contained strict loading."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from ..category import (
    decode_category_registry,
    decode_product_category_assignment_set,
    validate_category_bundle,
)
from ..errors import (
    CatalogSemanticError,
    ReleaseBundleBusyError,
    ReleaseBundleIntegrityError,
)
from ..facet import (
    decode_catalog_facet_schema,
    decode_catalog_facet_stats,
    decode_effective_facet_capabilities,
    decode_facet_applicability_set,
    decode_facet_evidence_store,
    decode_facet_source_binding_set,
    decode_product_facet_index,
)
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT, scan_raw_catalog
from ..runtime import (
    decode_runtime_facet_registry,
    decode_runtime_value_lexicon,
    validate_runtime_projection_bundle,
)
from .build import (
    DecodedReleaseArtifacts,
    build_release_manifest,
    build_reviewed_semantic_config,
    validate_decoded_release,
)
from .codec import (
    decode_release_manifest,
    encode_release_manifest,
    encode_reviewed_semantic_config,
    release_id_for_manifest,
)
from .models import (
    ARTIFACT_KINDS,
    ARTIFACT_SPEC,
    RELEASE_MANIFEST_FILENAME,
    ArtifactKind,
    ArtifactRef,
    VerifiedCatalogSemanticRelease,
)

_LOCK_FILENAME = ".catalog-semantic-release.write.lock"


def write_catalog_semantic_release(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    gate_b_review_dir: str | Path,
    gate_b_selection_path: str | Path,
    gate_b_candidate_dir: str | Path,
    runtime_projection_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = 200,
    enforce_official_gate: bool = True,
) -> VerifiedCatalogSemanticRelease:
    """Validate all upstream candidates and atomically publish one immutable release."""

    catalog = Path(catalog_path)
    category = Path(category_candidate_dir)
    gate_a = Path(gate_a_candidate_dir)
    resolution = Path(resolution_candidate_dir)
    gate_b = Path(gate_b_candidate_dir)
    runtime = Path(runtime_projection_dir)
    target = Path(output_dir)
    validate_category_bundle(category, catalog_path=catalog)
    validate_runtime_projection_bundle(
        runtime,
        catalog_path=catalog,
        category_candidate_dir=category,
        gate_a_candidate_dir=gate_a,
        resolution_candidate_dir=resolution,
        public_set_path=public_set_path,
        gate_b_review_dir=gate_b_review_dir,
        gate_b_selection_path=gate_b_selection_path,
        gate_b_candidate_dir=gate_b,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    source_paths = _source_artifact_paths(
        catalog=catalog,
        category=category,
        gate_a=gate_a,
        resolution=resolution,
        gate_b=gate_b,
        runtime=runtime,
    )
    artifacts = _decode_artifacts(source_paths)
    reviewed_config = build_reviewed_semantic_config(artifacts)
    reviewed_bytes = encode_reviewed_semantic_config(reviewed_config)
    refs = _artifact_refs(source_paths, reviewed_config_bytes=reviewed_bytes)
    manifest = build_release_manifest(refs)
    release_id = release_id_for_manifest(manifest)
    scan = scan_raw_catalog(catalog, expected_product_count=expected_product_count)
    validate_decoded_release(
        scan=scan,
        refs={item.kind: item for item in refs},
        artifacts=artifacts,
        reviewed_config=reviewed_config,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_release_writer(target):
        if target.exists():
            existing = load_catalog_semantic_release(
                target,
                expected_release_id=release_id,
                expected_product_count=expected_product_count,
            )
            return existing
        with TemporaryDirectory(prefix=".catalog-semantic-release-", dir=target.parent) as temp:
            generation = Path(temp) / "generation"
            generation.mkdir()
            for kind, source in source_paths.items():
                destination = generation / ARTIFACT_SPEC[kind][2]
                shutil.copyfile(source, destination)
            (generation / ARTIFACT_SPEC["reviewed_config"][2]).write_bytes(reviewed_bytes)
            (generation / RELEASE_MANIFEST_FILENAME).write_bytes(encode_release_manifest(manifest))
            verified = load_catalog_semantic_release(
                generation,
                expected_release_id=release_id,
                expected_product_count=expected_product_count,
            )
            if target.exists():
                raise ReleaseBundleIntegrityError("release target appeared during publication")
            os.replace(generation, target)
            return verified


def load_catalog_semantic_release(
    release_dir: str | Path,
    *,
    expected_release_id: str | None = None,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
) -> VerifiedCatalogSemanticRelease:
    """Load and deeply verify a self-contained release before exposing runtime objects."""

    try:
        return _load_catalog_semantic_release(
            Path(release_dir),
            expected_release_id=expected_release_id,
            expected_product_count=expected_product_count,
        )
    except ReleaseBundleIntegrityError:
        raise
    except (CatalogSemanticError, OSError, RecursionError, TypeError, ValueError) as error:
        raise ReleaseBundleIntegrityError(
            f"catalog semantic release is invalid: {error}"
        ) from error


def validate_catalog_semantic_release(
    release_dir: str | Path,
    *,
    expected_release_id: str | None = None,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
) -> str:
    """Validate a release and return its computed external release ID."""

    return load_catalog_semantic_release(
        release_dir,
        expected_release_id=expected_release_id,
        expected_product_count=expected_product_count,
    ).release_id


def _load_catalog_semantic_release(
    target: Path,
    *,
    expected_release_id: str | None,
    expected_product_count: int,
) -> VerifiedCatalogSemanticRelease:
    expected_filenames = {
        RELEASE_MANIFEST_FILENAME,
        *(spec[2] for spec in ARTIFACT_SPEC.values()),
    }
    try:
        observed_filenames = {path.name for path in target.iterdir()}
    except OSError as error:
        raise ReleaseBundleIntegrityError("release directory is unavailable") from error
    if observed_filenames != expected_filenames:
        raise ReleaseBundleIntegrityError("release directory members are incomplete or unexpected")
    manifest = decode_release_manifest((target / RELEASE_MANIFEST_FILENAME).read_bytes())
    release_id = release_id_for_manifest(manifest)
    if expected_release_id is not None and release_id != expected_release_id:
        raise ReleaseBundleIntegrityError("computed release ID differs from the expected ID")
    refs = {item.kind: item for item in manifest.artifacts}
    source_paths: dict[ArtifactKind, Path] = {
        kind: target / ARTIFACT_SPEC[kind][2]
        for kind in ARTIFACT_KINDS
        if kind != "reviewed_config"
    }
    for kind in ARTIFACT_KINDS:
        path = target / ARTIFACT_SPEC[kind][2]
        digest, byte_size = _hash_file(path)
        ref = refs[kind]
        if digest != ref.content_id or byte_size != ref.byte_size:
            raise ReleaseBundleIntegrityError(f"release artifact bytes differ: {kind}")
    artifacts = _decode_artifacts(source_paths)
    reviewed_config = build_reviewed_semantic_config(artifacts)
    reviewed_bytes = (target / ARTIFACT_SPEC["reviewed_config"][2]).read_bytes()
    if reviewed_bytes != encode_reviewed_semantic_config(reviewed_config):
        raise ReleaseBundleIntegrityError("reviewed config differs from exact artifact projection")
    if manifest.builder_version != reviewed_config.builder_version:
        raise ReleaseBundleIntegrityError("manifest and reviewed config builder versions differ")
    scan = scan_raw_catalog(
        target / ARTIFACT_SPEC["catalog"][2],
        expected_product_count=expected_product_count,
    )
    grounder = validate_decoded_release(
        scan=scan,
        refs=refs,
        artifacts=artifacts,
        reviewed_config=reviewed_config,
    )
    return VerifiedCatalogSemanticRelease(
        release_id=release_id,
        manifest=manifest,
        category_registry=artifacts.category_registry,
        product_category_assignments=artifacts.product_category_assignments,
        facet_schema=artifacts.facet_schema,
        facet_applicability=artifacts.facet_applicability,
        facet_source_bindings=artifacts.facet_source_bindings,
        facet_evidence_store=artifacts.facet_evidence_store,
        product_facet_index=artifacts.product_facet_index,
        facet_stats=artifacts.facet_stats,
        effective_capabilities=artifacts.effective_capabilities,
        runtime_value_lexicon=artifacts.runtime_value_lexicon,
        runtime_registry=artifacts.runtime_registry,
        reviewed_config=reviewed_config,
        grounder=grounder,
    )


def _source_artifact_paths(
    *,
    catalog: Path,
    category: Path,
    gate_a: Path,
    resolution: Path,
    gate_b: Path,
    runtime: Path,
) -> dict[ArtifactKind, Path]:
    return {
        "catalog": catalog,
        "category_registry": category / "category-registry.json",
        "product_category_assignment": category / "product-category-assignment.json",
        "facet_schema": gate_a / "catalog-facet-schema.json",
        "facet_applicability": gate_a / "facet-applicability.json",
        "facet_source_bindings": gate_a / "facet-source-bindings.json",
        "facet_evidence_store": resolution / "facet-evidence-store.json",
        "product_facet_index": resolution / "product-facet-index.json",
        "facet_stats": resolution / "catalog-facet-stats.json",
        "effective_capabilities": gate_b / "effective-facet-capabilities.json",
        "runtime_value_lexicon": runtime / "runtime-value-lexicon.json",
        "runtime_registry": runtime / "runtime-facet-registry.json",
    }


def _decode_artifacts(paths: Mapping[ArtifactKind, Path]) -> DecodedReleaseArtifacts:
    registry = decode_category_registry(paths["category_registry"].read_bytes())
    assignments = decode_product_category_assignment_set(
        paths["product_category_assignment"].read_bytes(),
        registry=registry,
    )
    return DecodedReleaseArtifacts(
        category_registry=registry,
        product_category_assignments=assignments,
        facet_schema=decode_catalog_facet_schema(paths["facet_schema"].read_bytes()),
        facet_applicability=decode_facet_applicability_set(
            paths["facet_applicability"].read_bytes()
        ),
        facet_source_bindings=decode_facet_source_binding_set(
            paths["facet_source_bindings"].read_bytes()
        ),
        facet_evidence_store=decode_facet_evidence_store(
            paths["facet_evidence_store"].read_bytes()
        ),
        product_facet_index=decode_product_facet_index(paths["product_facet_index"].read_bytes()),
        facet_stats=decode_catalog_facet_stats(paths["facet_stats"].read_bytes()),
        effective_capabilities=decode_effective_facet_capabilities(
            paths["effective_capabilities"].read_bytes()
        ),
        runtime_value_lexicon=decode_runtime_value_lexicon(
            paths["runtime_value_lexicon"].read_bytes()
        ),
        runtime_registry=decode_runtime_facet_registry(paths["runtime_registry"].read_bytes()),
    )


def _artifact_refs(
    paths: dict[ArtifactKind, Path],
    *,
    reviewed_config_bytes: bytes,
) -> tuple[ArtifactRef, ...]:
    refs: list[ArtifactRef] = []
    for kind in ARTIFACT_KINDS:
        if kind == "reviewed_config":
            digest = f"sha256:{hashlib.sha256(reviewed_config_bytes).hexdigest()}"
            byte_size = len(reviewed_config_bytes)
        else:
            digest, byte_size = _hash_file(paths[kind])
        refs.append(
            ArtifactRef(
                kind=kind,
                schema=ARTIFACT_SPEC[kind][1],
                content_id=digest,
                byte_size=byte_size,
            )
        )
    return tuple(refs)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return f"sha256:{digest.hexdigest()}", byte_size


@contextmanager
def _exclusive_release_writer(target: Path) -> Iterator[None]:
    lock = target.parent / _LOCK_FILENAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ReleaseBundleBusyError(f"release publication is already running: {target}") from error
    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
