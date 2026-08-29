"""Atomic CS3 publication with catalog read-only proof and exact rebuild validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ..canonical import canonical_json_bytes, content_id_for_bytes, content_id_for_value
from ..category import (
    CategoryRegistry,
    decode_category_registry,
    decode_product_category_assignment_set,
    validate_category_bundle,
)
from ..errors import (
    ResolutionBuildError,
    ResolutionBundleBusyError,
    ResolutionBundleIntegrityError,
)
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT
from .gate_a_bundle import load_gate_a_candidate_bundle
from .resolution_build import (
    build_resolution_candidate,
    validate_evidence_store,
    validate_product_facet_index,
    validate_stats_artifact,
)
from .resolution_codec import (
    decode_catalog_facet_stats,
    decode_catalog_read_only_audit,
    decode_facet_evidence_store,
    decode_product_facet_index,
    encode_catalog_facet_stats,
    encode_facet_evidence_store,
    encode_product_facet_index,
    resolution_candidate_document,
)
from .resolution_models import (
    CATALOG_READ_ONLY_AUDIT_SCHEMA,
    RESOLUTION_POLICY_ID,
    CatalogReadOnlyAudit,
    ResolutionCandidateBuild,
)
from .resolution_reporting import resolution_candidate_markdown

RESOLUTION_BUNDLE_SCHEMA = "shopping-copilot/resolution-candidate-bundle/v0"
RESOLUTION_MANIFEST_FILENAME = "bundle-manifest.json"
RESOLUTION_ARTIFACT_FILENAMES = (
    "candidate.json",
    "catalog-facet-stats.json",
    "catalog-read-only-audit.json",
    "facet-evidence-store.json",
    "product-facet-index.json",
    "report.md",
)

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def write_resolution_candidate_bundle(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    enforce_official_gate: bool = True,
) -> ResolutionCandidateBuild:
    """Build and atomically publish CS3 outputs beside, never inside, the catalog."""

    catalog = Path(catalog_path)
    category_candidate = Path(category_candidate_dir)
    gate_a_candidate = Path(gate_a_candidate_dir)
    target = Path(output_dir)
    _validate_paths((catalog, category_candidate, gate_a_candidate), target)
    before_id, before_size = _hash_file(catalog)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer(target):
        build, registry = _rebuild_candidate(
            catalog,
            category_candidate,
            gate_a_candidate,
            expected_product_count=expected_product_count,
            enforce_official_gate=enforce_official_gate,
        )
        if before_id != build.evidence_store.catalog_id:
            raise ResolutionBuildError("catalog changed before CS3 build started")
        payloads = _candidate_payloads(build, registry=registry)
        _publish_bundle(
            target,
            catalog_path=catalog,
            catalog_id_before=before_id,
            catalog_size_before=before_size,
            build=build,
            expected_product_count=expected_product_count,
            enforce_official_gate=enforce_official_gate,
            payloads=payloads,
        )
    final_id, final_size = _hash_file(catalog)
    if (final_id, final_size) != (before_id, before_size):
        raise ResolutionBuildError("catalog changed during CS3 bundle publication")
    validate_resolution_candidate_bundle(
        target,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        gate_a_candidate_dir=gate_a_candidate,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    return build


def validate_resolution_candidate_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    enforce_official_gate: bool = True,
) -> None:
    """Validate CS3 bytes, read-only proof, cross-references, and reproducibility."""

    target = Path(output_dir)
    manifest = _load_and_validate_manifest(target)
    payloads = _load_manifest_payloads(target, manifest)
    evidence_store = decode_facet_evidence_store(payloads["facet-evidence-store.json"])
    product_facet_index = decode_product_facet_index(payloads["product-facet-index.json"])
    stats = decode_catalog_facet_stats(payloads["catalog-facet-stats.json"])
    audit = decode_catalog_read_only_audit(payloads["catalog-read-only-audit.json"])

    expected, registry = _rebuild_candidate(
        Path(catalog_path),
        Path(category_candidate_dir),
        Path(gate_a_candidate_dir),
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    decoded = ResolutionCandidateBuild(
        schema=expected.schema,
        builder_version=expected.builder_version,
        category_registry_id=expected.category_registry_id,
        facet_schema_id=expected.facet_schema_id,
        gate_a_selection_id=expected.gate_a_selection_id,
        evidence_store=evidence_store,
        product_facet_index=product_facet_index,
        stats=stats,
    )
    gate_a = load_gate_a_candidate_bundle(gate_a_candidate_dir)
    try:
        registry_bytes = (Path(category_candidate_dir) / "category-registry.json").read_bytes()
        assignment_bytes = (
            Path(category_candidate_dir) / "product-category-assignment.json"
        ).read_bytes()
    except OSError as error:
        raise ResolutionBundleIntegrityError("CS3 category input is unavailable") from error
    assignments = decode_product_category_assignment_set(
        assignment_bytes,
        registry=registry,
    )
    validate_evidence_store(
        decoded.evidence_store,
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        product_category_assignment_id=content_id_for_bytes(assignment_bytes),
    )
    validate_product_facet_index(
        decoded.product_facet_index,
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        evidence_store=decoded.evidence_store,
    )
    validate_stats_artifact(
        decoded.stats,
        registry=registry,
        assignments=assignments,
        gate_a=gate_a,
        category_registry_id=content_id_for_bytes(registry_bytes),
        facet_schema_id=decoded.facet_schema_id,
        product_facet_index=decoded.product_facet_index,
    )
    if decoded != expected:
        raise ResolutionBundleIntegrityError("CS3 artifacts differ from exact upstream truth")

    expected_manifest_fields = _manifest_identity_fields(
        expected,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    for name, value in expected_manifest_fields.items():
        if manifest[name] != value:
            raise ResolutionBundleIntegrityError(f"CS3 manifest field is stale: {name}")

    current_id, current_size = _hash_file(Path(catalog_path))
    if (
        audit.catalog_id_before != current_id
        or audit.catalog_id_after_staging != current_id
        or audit.byte_size_before != current_size
        or audit.byte_size_after_staging != current_size
    ):
        raise ResolutionBundleIntegrityError("catalog read-only audit differs from current bytes")
    expected_payloads = _candidate_payloads(expected, registry=registry)
    expected_payloads["catalog-read-only-audit.json"] = canonical_json_bytes(audit)
    for filename in RESOLUTION_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise ResolutionBundleIntegrityError(
                f"CS3 artifact differs from exact upstream truth: {filename}"
            )


def _rebuild_candidate(
    catalog_path: Path,
    category_candidate_dir: Path,
    gate_a_candidate_dir: Path,
    *,
    expected_product_count: int,
    enforce_official_gate: bool,
) -> tuple[ResolutionCandidateBuild, CategoryRegistry]:
    validate_category_bundle(category_candidate_dir, catalog_path=catalog_path)
    gate_a = load_gate_a_candidate_bundle(gate_a_candidate_dir)
    try:
        registry_bytes = (category_candidate_dir / "category-registry.json").read_bytes()
        assignment_bytes = (
            category_candidate_dir / "product-category-assignment.json"
        ).read_bytes()
    except OSError as error:
        raise ResolutionBuildError("CS3 category input is unavailable") from error
    registry = decode_category_registry(registry_bytes)
    assignments = decode_product_category_assignment_set(
        assignment_bytes,
        registry=registry,
    )
    build = build_resolution_candidate(
        catalog_path,
        registry=registry,
        assignments=assignments,
        category_registry_id=content_id_for_bytes(registry_bytes),
        product_category_assignment_id=content_id_for_bytes(assignment_bytes),
        gate_a=gate_a,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    return build, registry


def _candidate_payloads(
    build: ResolutionCandidateBuild,
    *,
    registry: CategoryRegistry,
) -> dict[str, bytes]:
    return {
        "candidate.json": canonical_json_bytes(resolution_candidate_document(build)),
        "catalog-facet-stats.json": encode_catalog_facet_stats(build.stats),
        "facet-evidence-store.json": encode_facet_evidence_store(build.evidence_store),
        "product-facet-index.json": encode_product_facet_index(build.product_facet_index),
        "report.md": resolution_candidate_markdown(build, registry=registry).encode("utf-8"),
    }


def _publish_bundle(
    target: Path,
    *,
    catalog_path: Path,
    catalog_id_before: str,
    catalog_size_before: int,
    build: ResolutionCandidateBuild,
    expected_product_count: int,
    enforce_official_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".catalog-resolution-", dir=target.parent) as temporary:
        staging = Path(temporary)
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
        catalog_id_after, catalog_size_after = _hash_file(catalog_path)
        audit = CatalogReadOnlyAudit(
            schema=CATALOG_READ_ONLY_AUDIT_SCHEMA,
            catalog_id_before=catalog_id_before,
            catalog_id_after_staging=catalog_id_after,
            byte_size_before=catalog_size_before,
            byte_size_after_staging=catalog_size_after,
            unchanged=True,
            output_is_separate=True,
        )
        payloads["catalog-read-only-audit.json"] = canonical_json_bytes(audit)
        (staging / "catalog-read-only-audit.json").write_bytes(
            payloads["catalog-read-only-audit.json"]
        )
        artifacts = [
            {
                "byte_size": len(payloads[filename]),
                "filename": filename,
                "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
            }
            for filename in sorted(payloads)
        ]
        manifest = {
            "schema": RESOLUTION_BUNDLE_SCHEMA,
            **_manifest_identity_fields(
                build,
                expected_product_count=expected_product_count,
                enforce_official_gate=enforce_official_gate,
            ),
            "catalog_read_only": True,
            "artifacts": artifacts,
        }
        (staging / RESOLUTION_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        target.mkdir(parents=True, exist_ok=True)
        for filename in RESOLUTION_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / RESOLUTION_MANIFEST_FILENAME,
            target / RESOLUTION_MANIFEST_FILENAME,
        )


def _manifest_identity_fields(
    build: ResolutionCandidateBuild,
    *,
    expected_product_count: int,
    enforce_official_gate: bool,
) -> dict[str, object]:
    return {
        "catalog_id": build.evidence_store.catalog_id,
        "category_registry_id": build.category_registry_id,
        "product_category_assignment_id": build.evidence_store.product_category_assignment_id,
        "facet_schema_id": build.facet_schema_id,
        "facet_applicability_id": build.evidence_store.facet_applicability_id,
        "facet_source_bindings_id": build.evidence_store.facet_source_bindings_id,
        "gate_a_selection_id": build.gate_a_selection_id,
        "facet_evidence_store_id": content_id_for_value(build.evidence_store),
        "product_facet_index_id": content_id_for_value(build.product_facet_index),
        "catalog_facet_stats_id": content_id_for_value(build.stats),
        "resolution_policy_id": RESOLUTION_POLICY_ID,
        "builder_version": build.builder_version,
        "expected_product_count": expected_product_count,
        "official_gate": enforce_official_gate,
    }


def _load_and_validate_manifest(target: Path) -> dict[str, object]:
    try:
        data = (target / RESOLUTION_MANIFEST_FILENAME).read_bytes()
        parsed: object = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ResolutionBundleIntegrityError("CS3 manifest is unavailable or invalid") from error
    if type(parsed) is not dict or canonical_json_bytes(parsed) != data:
        raise ResolutionBundleIntegrityError("CS3 manifest is not canonical")
    manifest = cast(dict[str, object], parsed)
    expected_fields = {
        "schema",
        "catalog_id",
        "category_registry_id",
        "product_category_assignment_id",
        "facet_schema_id",
        "facet_applicability_id",
        "facet_source_bindings_id",
        "gate_a_selection_id",
        "facet_evidence_store_id",
        "product_facet_index_id",
        "catalog_facet_stats_id",
        "resolution_policy_id",
        "builder_version",
        "expected_product_count",
        "official_gate",
        "catalog_read_only",
        "artifacts",
    }
    if set(manifest) != expected_fields:
        raise ResolutionBundleIntegrityError("CS3 manifest has invalid fields")
    if manifest["schema"] != RESOLUTION_BUNDLE_SCHEMA:
        raise ResolutionBundleIntegrityError("CS3 manifest schema is unsupported")
    for name in (
        "catalog_id",
        "category_registry_id",
        "product_category_assignment_id",
        "facet_schema_id",
        "facet_applicability_id",
        "facet_source_bindings_id",
        "gate_a_selection_id",
        "facet_evidence_store_id",
        "product_facet_index_id",
        "catalog_facet_stats_id",
    ):
        value = manifest[name]
        if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise ResolutionBundleIntegrityError(f"CS3 manifest {name} is invalid")
    if manifest["resolution_policy_id"] != RESOLUTION_POLICY_ID:
        raise ResolutionBundleIntegrityError("CS3 manifest resolution policy is unsupported")
    if type(manifest["builder_version"]) is not str:
        raise ResolutionBundleIntegrityError("CS3 manifest builder version is invalid")
    if type(manifest["expected_product_count"]) is not int:
        raise ResolutionBundleIntegrityError("CS3 manifest product count is invalid")
    if type(manifest["official_gate"]) is not bool:
        raise ResolutionBundleIntegrityError("CS3 manifest gate flag is invalid")
    if manifest["catalog_read_only"] is not True:
        raise ResolutionBundleIntegrityError("CS3 manifest must declare catalog read-only")
    if type(manifest["artifacts"]) is not list:
        raise ResolutionBundleIntegrityError("CS3 manifest artifacts are invalid")
    return manifest


def _load_manifest_payloads(
    target: Path,
    manifest: dict[str, object],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    previous_name: str | None = None
    for raw_artifact in cast(list[object], manifest["artifacts"]):
        if type(raw_artifact) is not dict:
            raise ResolutionBundleIntegrityError("CS3 artifact entry is invalid")
        artifact = cast(dict[str, object], raw_artifact)
        if set(artifact) != {"byte_size", "filename", "sha256"}:
            raise ResolutionBundleIntegrityError("CS3 artifact entry fields are invalid")
        filename = artifact["filename"]
        byte_size = artifact["byte_size"]
        digest = artifact["sha256"]
        if type(filename) is not str or filename not in RESOLUTION_ARTIFACT_FILENAMES:
            raise ResolutionBundleIntegrityError("CS3 artifact filename is invalid")
        if filename in payloads or (previous_name is not None and filename <= previous_name):
            raise ResolutionBundleIntegrityError("CS3 artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise ResolutionBundleIntegrityError("CS3 artifact byte size is invalid")
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ResolutionBundleIntegrityError("CS3 artifact hash is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise ResolutionBundleIntegrityError(
                f"CS3 artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != digest:
            raise ResolutionBundleIntegrityError(f"CS3 artifact failed integrity: {filename}")
        payloads[filename] = payload
        previous_name = filename
    if set(payloads) != set(RESOLUTION_ARTIFACT_FILENAMES):
        raise ResolutionBundleIntegrityError("CS3 artifact set is incomplete")
    return payloads


def _validate_paths(inputs: tuple[Path, ...], target: Path) -> None:
    resolved_target = target.resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"CS3 output is not a directory: {target}")
    for input_path in inputs:
        resolved_input = input_path.resolve()
        if resolved_input == resolved_target or resolved_target in resolved_input.parents:
            raise ValueError("CS3 output must not contain or replace any input path")
    if target.exists():
        allowed = {*RESOLUTION_ARTIFACT_FILENAMES, RESOLUTION_MANIFEST_FILENAME}
        unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
        if unexpected:
            raise ResolutionBundleIntegrityError(
                f"CS3 output contains an unexpected entry: {unexpected[0]}"
            )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        stream = path.open("rb")
    except OSError as error:
        raise ResolutionBuildError("catalog is unavailable for read-only verification") from error
    with stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


@contextmanager
def _exclusive_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock_path = resolved.parent / f".{resolved.name or 'catalog-resolution'}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ResolutionBundleBusyError(
            f"CS3 candidate is already being written: {resolved}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
