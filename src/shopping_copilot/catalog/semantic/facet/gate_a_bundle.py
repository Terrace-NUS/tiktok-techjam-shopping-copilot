"""Atomic publication and exact rebuild validation for Gate-A candidates."""

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
    decode_category_registry,
    decode_product_category_assignment_set,
    validate_category_bundle,
)
from ..errors import (
    GateABuildError,
    GateABundleBusyError,
    GateABundleIntegrityError,
)
from .bundle import validate_gate_a_source_profile_bundle
from .gate_a_build import build_gate_a_candidate
from .gate_a_codec import (
    decode_catalog_facet_schema,
    decode_facet_applicability_set,
    decode_facet_source_binding_set,
    decode_gate_a_selection,
    decode_price_extraction_audits,
    gate_a_candidate_document,
)
from .gate_a_models import GATE_A_BUILDER_VERSION, GateACandidateBuild
from .gate_a_reporting import gate_a_candidate_markdown

GATE_A_BUNDLE_SCHEMA = "shopping-copilot/gate-a-candidate-bundle/v0"
GATE_A_MANIFEST_FILENAME = "bundle-manifest.json"
GATE_A_ARTIFACT_FILENAMES = (
    "candidate.json",
    "catalog-facet-schema.json",
    "extraction-audit.json",
    "facet-applicability.json",
    "facet-source-bindings.json",
    "report.md",
    "reviewed-gate-a-selection.json",
)

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def write_gate_a_candidate_bundle(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    profile_selection_path: str | Path,
    source_profile_dir: str | Path,
    gate_a_selection_path: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = 50_000,
    enforce_official_gate: bool = True,
) -> GateACandidateBuild:
    """Rebuild and atomically publish reviewed Gate-A candidate artifacts."""

    catalog = Path(catalog_path)
    category_candidate = Path(category_candidate_dir)
    profile_selection = Path(profile_selection_path)
    source_profile = Path(source_profile_dir)
    gate_a_selection = Path(gate_a_selection_path)
    target = Path(output_dir)
    _validate_paths(
        catalog,
        category_candidate,
        profile_selection,
        source_profile,
        gate_a_selection,
        target,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer(target):
        build = _rebuild_candidate(
            catalog,
            category_candidate,
            profile_selection,
            source_profile,
            gate_a_selection,
            expected_product_count=expected_product_count,
            enforce_official_gate=enforce_official_gate,
        )
        payloads = _candidate_payloads(build)
        _publish_bundle(
            target,
            build=build,
            expected_product_count=expected_product_count,
            enforce_official_gate=enforce_official_gate,
            payloads=payloads,
        )
    validate_gate_a_candidate_bundle(
        target,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        profile_selection_path=profile_selection,
        source_profile_dir=source_profile,
        gate_a_selection_path=gate_a_selection,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    return build


def validate_gate_a_candidate_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    profile_selection_path: str | Path,
    source_profile_dir: str | Path,
    gate_a_selection_path: str | Path,
    expected_product_count: int = 50_000,
    enforce_official_gate: bool = True,
) -> None:
    """Validate hashes, pins, and exact reproducibility from all upstream truth."""

    target = Path(output_dir)
    manifest = _load_and_validate_manifest(target)
    payloads = _load_manifest_payloads(target, manifest)
    expected = _rebuild_candidate(
        Path(catalog_path),
        Path(category_candidate_dir),
        Path(profile_selection_path),
        Path(source_profile_dir),
        Path(gate_a_selection_path),
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    expected_manifest_fields = {
        "catalog_id": expected.catalog_id,
        "category_registry_id": expected.category_registry_id,
        "product_category_assignment_id": expected.product_category_assignment_id,
        "source_profile_manifest_sha256": expected.source_profile_manifest_sha256,
        "gate_a_selection_id": content_id_for_value(expected.selection),
        "builder_version": expected.builder_version,
        "expected_product_count": expected_product_count,
        "official_gate": enforce_official_gate,
    }
    for name, value in expected_manifest_fields.items():
        if manifest[name] != value:
            raise GateABundleIntegrityError(f"Gate-A manifest field is stale: {name}")
    expected_payloads = _candidate_payloads(expected)
    for filename in GATE_A_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise GateABundleIntegrityError(
                f"Gate-A artifact differs from exact upstream truth: {filename}"
            )


def load_gate_a_candidate_bundle(output_dir: str | Path) -> GateACandidateBuild:
    """Load a self-consistent Gate-A bundle after exact artifact verification.

    This downstream loader verifies the candidate's manifest, canonical artifacts,
    copied reviewed selection, cross-references, and derived report. Rebuilding from
    the raw catalog remains the stronger stage-boundary check performed by
    :func:`validate_gate_a_candidate_bundle`.
    """

    target = Path(output_dir)
    manifest = _load_and_validate_manifest(target)
    payloads = _load_manifest_payloads(target, manifest)
    selection = decode_gate_a_selection(payloads["reviewed-gate-a-selection.json"])
    facet_schema = decode_catalog_facet_schema(payloads["catalog-facet-schema.json"])
    applicability = decode_facet_applicability_set(payloads["facet-applicability.json"])
    bindings = decode_facet_source_binding_set(payloads["facet-source-bindings.json"])
    audits = decode_price_extraction_audits(payloads["extraction-audit.json"])
    try:
        build = GateACandidateBuild(
            schema="shopping-copilot/gate-a-candidate/v0",
            catalog_id=selection.catalog_id,
            category_registry_id=selection.category_registry_id,
            product_category_assignment_id=selection.product_category_assignment_id,
            source_profile_manifest_sha256=selection.source_profile_manifest_sha256,
            builder_version=selection.builder_version,
            selection=selection,
            facet_schema=facet_schema,
            applicability=applicability,
            bindings=bindings,
            price_audits=audits,
        )
    except (TypeError, ValueError) as error:
        raise GateABundleIntegrityError("Gate-A artifacts are not mutually consistent") from error

    expected_manifest_fields = {
        "catalog_id": build.catalog_id,
        "category_registry_id": build.category_registry_id,
        "product_category_assignment_id": build.product_category_assignment_id,
        "source_profile_manifest_sha256": build.source_profile_manifest_sha256,
        "gate_a_selection_id": content_id_for_value(build.selection),
        "builder_version": build.builder_version,
    }
    for name, value in expected_manifest_fields.items():
        if manifest[name] != value:
            raise GateABundleIntegrityError(f"Gate-A manifest field is stale: {name}")
    expected_count = manifest["expected_product_count"]
    if type(expected_count) is not int or expected_count <= 0:
        raise GateABundleIntegrityError("Gate-A manifest product count is invalid")
    if any(audit.product_count != expected_count for audit in build.price_audits):
        raise GateABundleIntegrityError("Gate-A audit product count differs from manifest")

    expected_payloads = _candidate_payloads(build)
    for filename in GATE_A_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise GateABundleIntegrityError(
                f"Gate-A artifact is not the exact reviewed projection: {filename}"
            )
    return build


def _rebuild_candidate(
    catalog_path: Path,
    category_candidate_dir: Path,
    profile_selection_path: Path,
    source_profile_dir: Path,
    gate_a_selection_path: Path,
    *,
    expected_product_count: int,
    enforce_official_gate: bool,
) -> GateACandidateBuild:
    validate_category_bundle(category_candidate_dir, catalog_path=catalog_path)
    rebuilt_source_profile = validate_gate_a_source_profile_bundle(
        source_profile_dir,
        catalog_path=catalog_path,
        category_candidate_dir=category_candidate_dir,
        selection_path=profile_selection_path,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    try:
        registry_bytes = (category_candidate_dir / "category-registry.json").read_bytes()
        assignment_bytes = (
            category_candidate_dir / "product-category-assignment.json"
        ).read_bytes()
        source_profile_manifest_bytes = (source_profile_dir / "bundle-manifest.json").read_bytes()
        gate_a_selection_bytes = gate_a_selection_path.read_bytes()
    except OSError as error:
        raise GateABuildError("Gate-A input is unavailable") from error

    registry = decode_category_registry(registry_bytes)
    assignments = decode_product_category_assignment_set(
        assignment_bytes,
        registry=registry,
    )
    gate_a_selection = decode_gate_a_selection(gate_a_selection_bytes)
    return build_gate_a_candidate(
        catalog_path,
        registry=registry,
        assignments=assignments,
        category_registry_id=content_id_for_bytes(registry_bytes),
        product_category_assignment_id=content_id_for_bytes(assignment_bytes),
        source_profile=rebuilt_source_profile,
        source_profile_manifest_sha256=hashlib.sha256(source_profile_manifest_bytes).hexdigest(),
        selection=gate_a_selection,
    )


def _candidate_payloads(build: GateACandidateBuild) -> dict[str, bytes]:
    return {
        "candidate.json": canonical_json_bytes(gate_a_candidate_document(build)),
        "catalog-facet-schema.json": canonical_json_bytes(build.facet_schema),
        "extraction-audit.json": canonical_json_bytes(build.price_audits),
        "facet-applicability.json": canonical_json_bytes(build.applicability),
        "facet-source-bindings.json": canonical_json_bytes(build.bindings),
        "report.md": gate_a_candidate_markdown(build).encode("utf-8"),
        "reviewed-gate-a-selection.json": canonical_json_bytes(build.selection),
    }


def _publish_bundle(
    target: Path,
    *,
    build: GateACandidateBuild,
    expected_product_count: int,
    enforce_official_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".catalog-gate-a-", dir=target.parent) as temporary:
        staging = Path(temporary)
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
        artifacts = [
            {
                "byte_size": len(payloads[filename]),
                "filename": filename,
                "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
            }
            for filename in sorted(payloads)
        ]
        manifest = {
            "schema": GATE_A_BUNDLE_SCHEMA,
            "catalog_id": build.catalog_id,
            "category_registry_id": build.category_registry_id,
            "product_category_assignment_id": build.product_category_assignment_id,
            "source_profile_manifest_sha256": build.source_profile_manifest_sha256,
            "gate_a_selection_id": content_id_for_value(build.selection),
            "builder_version": build.builder_version,
            "expected_product_count": expected_product_count,
            "official_gate": enforce_official_gate,
            "artifacts": artifacts,
        }
        (staging / GATE_A_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        target.mkdir(parents=True, exist_ok=True)
        for filename in GATE_A_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / GATE_A_MANIFEST_FILENAME,
            target / GATE_A_MANIFEST_FILENAME,
        )


def _load_and_validate_manifest(target: Path) -> dict[str, object]:
    try:
        manifest_bytes = (target / GATE_A_MANIFEST_FILENAME).read_bytes()
        parsed: object = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GateABundleIntegrityError("Gate-A manifest is unavailable or invalid") from error
    if type(parsed) is not dict or canonical_json_bytes(parsed) != manifest_bytes:
        raise GateABundleIntegrityError("Gate-A manifest is not canonical")
    manifest = cast(dict[str, object], parsed)
    expected_fields = {
        "schema",
        "catalog_id",
        "category_registry_id",
        "product_category_assignment_id",
        "source_profile_manifest_sha256",
        "gate_a_selection_id",
        "builder_version",
        "expected_product_count",
        "official_gate",
        "artifacts",
    }
    if set(manifest) != expected_fields:
        raise GateABundleIntegrityError("Gate-A manifest has invalid fields")
    if manifest["schema"] != GATE_A_BUNDLE_SCHEMA:
        raise GateABundleIntegrityError("Gate-A manifest schema is unsupported")
    for name in ("catalog_id", "category_registry_id", "product_category_assignment_id"):
        value = manifest[name]
        if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise GateABundleIntegrityError(f"Gate-A manifest {name} is invalid")
    profile_hash = manifest["source_profile_manifest_sha256"]
    if type(profile_hash) is not str or _SHA256_PATTERN.fullmatch(profile_hash) is None:
        raise GateABundleIntegrityError("Gate-A source-profile hash is invalid")
    selection_id = manifest["gate_a_selection_id"]
    if type(selection_id) is not str or _CONTENT_ID_PATTERN.fullmatch(selection_id) is None:
        raise GateABundleIntegrityError("Gate-A reviewed selection ID is invalid")
    if manifest["builder_version"] != GATE_A_BUILDER_VERSION:
        raise GateABundleIntegrityError("Gate-A manifest builder version is unsupported")
    if type(manifest["expected_product_count"]) is not int:
        raise GateABundleIntegrityError("Gate-A manifest product count is invalid")
    if type(manifest["official_gate"]) is not bool:
        raise GateABundleIntegrityError("Gate-A manifest gate flag is invalid")
    if type(manifest["artifacts"]) is not list:
        raise GateABundleIntegrityError("Gate-A manifest artifacts are invalid")
    return manifest


def _load_manifest_payloads(
    target: Path,
    manifest: dict[str, object],
) -> dict[str, bytes]:
    raw_artifacts = cast(list[object], manifest["artifacts"])
    payloads: dict[str, bytes] = {}
    previous_name: str | None = None
    for raw_artifact in raw_artifacts:
        if type(raw_artifact) is not dict:
            raise GateABundleIntegrityError("Gate-A manifest artifact entry is invalid")
        artifact = cast(dict[str, object], raw_artifact)
        if set(artifact) != {"byte_size", "filename", "sha256"}:
            raise GateABundleIntegrityError("Gate-A manifest artifact fields are invalid")
        filename = artifact["filename"]
        byte_size = artifact["byte_size"]
        sha256 = artifact["sha256"]
        if type(filename) is not str or filename not in GATE_A_ARTIFACT_FILENAMES:
            raise GateABundleIntegrityError("Gate-A artifact filename is invalid")
        if filename in payloads or (previous_name is not None and filename <= previous_name):
            raise GateABundleIntegrityError("Gate-A artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise GateABundleIntegrityError("Gate-A artifact byte size is invalid")
        if type(sha256) is not str or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise GateABundleIntegrityError("Gate-A artifact hash is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise GateABundleIntegrityError(
                f"Gate-A artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != sha256:
            raise GateABundleIntegrityError(f"Gate-A artifact failed integrity check: {filename}")
        payloads[filename] = payload
        previous_name = filename
    if set(payloads) != set(GATE_A_ARTIFACT_FILENAMES):
        raise GateABundleIntegrityError("Gate-A artifact set is incomplete")
    return payloads


def _validate_paths(
    catalog_path: Path,
    category_candidate_dir: Path,
    profile_selection_path: Path,
    source_profile_dir: Path,
    gate_a_selection_path: Path,
    target: Path,
) -> None:
    resolved_target = target.resolve()
    inputs = {
        catalog_path.resolve(),
        category_candidate_dir.resolve(),
        profile_selection_path.resolve(),
        source_profile_dir.resolve(),
        gate_a_selection_path.resolve(),
    }
    if resolved_target in inputs:
        raise ValueError("Gate-A output path collides with an input")
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Gate-A output is not a directory: {target}")
    if target.exists():
        allowed = {*GATE_A_ARTIFACT_FILENAMES, GATE_A_MANIFEST_FILENAME}
        unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
        if unexpected:
            raise GateABundleIntegrityError(
                f"Gate-A output contains an unexpected entry: {unexpected[0]}"
            )


@contextmanager
def _exclusive_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock_path = resolved.parent / f".{resolved.name or 'catalog-gate-a'}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise GateABundleBusyError(
            f"Gate-A candidate is already being written: {resolved}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
