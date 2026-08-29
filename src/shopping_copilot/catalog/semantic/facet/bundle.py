"""Atomic publication and independent validation for CS2 profile proposals."""

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

from ..canonical import canonical_json_bytes, content_id_for_bytes
from ..category import (
    decode_category_registry,
    decode_product_category_assignment_set,
    validate_category_bundle,
)
from ..errors import (
    FacetProfileBuildError,
    FacetProfileBundleBusyError,
    FacetProfileBundleIntegrityError,
)
from .codec import canonical_json_lines, decode_profile_selection, profile_document
from .models import FACET_PROFILE_BUILDER_VERSION, GateASourceProfileBuild
from .profiling import build_gate_a_source_profile
from .reporting import gate_a_source_profile_markdown

FACET_PROFILE_BUNDLE_SCHEMA = "shopping-copilot/gate-a-source-profile-bundle/v0"
FACET_PROFILE_MANIFEST_FILENAME = "bundle-manifest.json"
FACET_PROFILE_ARTIFACT_FILENAMES = (
    "price-audit.json",
    "profile.json",
    "report.md",
    "scope-source-profiles.jsonl",
    "source-samples.jsonl",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def write_gate_a_source_profile_bundle(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = 50_000,
    enforce_official_gate: bool = True,
) -> GateASourceProfileBuild:
    """Build and atomically publish the pre-approval CS2 source profile."""

    source = Path(catalog_path)
    category_candidate = Path(category_candidate_dir)
    selection_source = Path(selection_path)
    target = Path(output_dir)
    _validate_paths(source, category_candidate, selection_source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer(target):
        build = _rebuild_profile(
            source,
            category_candidate,
            selection_source,
            expected_product_count=expected_product_count,
            enforce_official_gate=enforce_official_gate,
        )
        payloads = _profile_payloads(build)
        _publish_bundle(
            target,
            build=build,
            expected_product_count=expected_product_count,
            enforce_official_gate=enforce_official_gate,
            payloads=payloads,
        )
    validate_gate_a_source_profile_bundle(
        target,
        catalog_path=source,
        category_candidate_dir=category_candidate,
        selection_path=selection_source,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    return build


def validate_gate_a_source_profile_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    selection_path: str | Path,
    expected_product_count: int = 50_000,
    enforce_official_gate: bool = True,
) -> GateASourceProfileBuild:
    """Validate hashes, pins, and exact reproducibility against upstream truth."""

    target = Path(output_dir)
    manifest = _load_and_validate_manifest(target)
    payloads = _load_manifest_payloads(target, manifest)
    expected = _rebuild_profile(
        Path(catalog_path),
        Path(category_candidate_dir),
        Path(selection_path),
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    if manifest["catalog_id"] != expected.catalog_id:
        raise FacetProfileBundleIntegrityError("profile manifest catalog pin is stale")
    if manifest["category_registry_id"] != expected.category_registry_id:
        raise FacetProfileBundleIntegrityError("profile manifest CategoryRegistry pin is stale")
    if manifest["product_category_assignment_id"] != expected.product_category_assignment_id:
        raise FacetProfileBundleIntegrityError("profile manifest assignment pin is stale")
    if manifest["builder_version"] != expected.builder_version:
        raise FacetProfileBundleIntegrityError("profile manifest builder version is stale")
    if manifest["expected_product_count"] != expected_product_count:
        raise FacetProfileBundleIntegrityError("profile manifest product gate is stale")
    if manifest["exhaustive_details_gate"] is not enforce_official_gate:
        raise FacetProfileBundleIntegrityError("profile manifest official gate is stale")
    expected_payloads = _profile_payloads(expected)
    for filename in FACET_PROFILE_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise FacetProfileBundleIntegrityError(
                f"profile artifact differs from exact upstream truth: {filename}"
            )
    return expected


def _rebuild_profile(
    catalog_path: Path,
    category_candidate_dir: Path,
    selection_path: Path,
    *,
    expected_product_count: int,
    enforce_official_gate: bool,
) -> GateASourceProfileBuild:
    validate_category_bundle(category_candidate_dir, catalog_path=catalog_path)
    try:
        registry_bytes = (category_candidate_dir / "category-registry.json").read_bytes()
        assignment_bytes = (
            category_candidate_dir / "product-category-assignment.json"
        ).read_bytes()
        selection_bytes = selection_path.read_bytes()
    except OSError as error:
        raise FacetProfileBuildError("Gate-A profile input is unavailable") from error
    registry = decode_category_registry(registry_bytes)
    assignments = decode_product_category_assignment_set(
        assignment_bytes,
        registry=registry,
    )
    selection = decode_profile_selection(selection_bytes)
    return build_gate_a_source_profile(
        catalog_path,
        registry=registry,
        assignments=assignments,
        category_registry_id=content_id_for_bytes(registry_bytes),
        product_category_assignment_id=content_id_for_bytes(assignment_bytes),
        selection=selection,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )


def _profile_payloads(build: GateASourceProfileBuild) -> dict[str, bytes]:
    return {
        "price-audit.json": canonical_json_bytes(build.price_audit),
        "profile.json": canonical_json_bytes(profile_document(build)),
        "report.md": gate_a_source_profile_markdown(build).encode("utf-8"),
        "scope-source-profiles.jsonl": canonical_json_lines(build.scope_source_profiles),
        "source-samples.jsonl": canonical_json_lines(build.samples),
    }


def _publish_bundle(
    target: Path,
    *,
    build: GateASourceProfileBuild,
    expected_product_count: int,
    enforce_official_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".catalog-facet-profile-", dir=target.parent) as temporary:
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
            "schema": FACET_PROFILE_BUNDLE_SCHEMA,
            "catalog_id": build.catalog_id,
            "category_registry_id": build.category_registry_id,
            "product_category_assignment_id": build.product_category_assignment_id,
            "builder_version": build.builder_version,
            "expected_product_count": expected_product_count,
            "exhaustive_details_gate": enforce_official_gate,
            "artifacts": artifacts,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        (staging / FACET_PROFILE_MANIFEST_FILENAME).write_bytes(manifest_bytes)
        target.mkdir(parents=True, exist_ok=True)
        for filename in FACET_PROFILE_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / FACET_PROFILE_MANIFEST_FILENAME,
            target / FACET_PROFILE_MANIFEST_FILENAME,
        )


def _load_and_validate_manifest(target: Path) -> dict[str, object]:
    try:
        manifest_bytes = (target / FACET_PROFILE_MANIFEST_FILENAME).read_bytes()
        parsed: object = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise FacetProfileBundleIntegrityError(
            "profile manifest is unavailable or invalid"
        ) from error
    if type(parsed) is not dict or canonical_json_bytes(parsed) != manifest_bytes:
        raise FacetProfileBundleIntegrityError("profile manifest is not canonical")
    manifest = cast(dict[str, object], parsed)
    expected_fields = {
        "schema",
        "catalog_id",
        "category_registry_id",
        "product_category_assignment_id",
        "builder_version",
        "expected_product_count",
        "exhaustive_details_gate",
        "artifacts",
    }
    if set(manifest) != expected_fields:
        raise FacetProfileBundleIntegrityError("profile manifest has invalid fields")
    if manifest["schema"] != FACET_PROFILE_BUNDLE_SCHEMA:
        raise FacetProfileBundleIntegrityError("profile manifest schema is unsupported")
    for name in ("catalog_id", "category_registry_id", "product_category_assignment_id"):
        value = manifest[name]
        if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise FacetProfileBundleIntegrityError(f"profile manifest {name} is invalid")
    if manifest["builder_version"] != FACET_PROFILE_BUILDER_VERSION:
        raise FacetProfileBundleIntegrityError("profile manifest builder version is unsupported")
    if type(manifest["expected_product_count"]) is not int:
        raise FacetProfileBundleIntegrityError("profile manifest product count is invalid")
    if type(manifest["exhaustive_details_gate"]) is not bool:
        raise FacetProfileBundleIntegrityError("profile manifest gate flag is invalid")
    if type(manifest["artifacts"]) is not list:
        raise FacetProfileBundleIntegrityError("profile manifest artifacts are invalid")
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
            raise FacetProfileBundleIntegrityError("profile manifest artifact entry is invalid")
        artifact = cast(dict[str, object], raw_artifact)
        if set(artifact) != {"byte_size", "filename", "sha256"}:
            raise FacetProfileBundleIntegrityError("profile manifest artifact fields are invalid")
        filename = artifact["filename"]
        byte_size = artifact["byte_size"]
        sha256 = artifact["sha256"]
        if type(filename) is not str or filename not in FACET_PROFILE_ARTIFACT_FILENAMES:
            raise FacetProfileBundleIntegrityError("profile artifact filename is invalid")
        if filename in payloads or (previous_name is not None and filename <= previous_name):
            raise FacetProfileBundleIntegrityError("profile artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise FacetProfileBundleIntegrityError("profile artifact byte size is invalid")
        if type(sha256) is not str or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise FacetProfileBundleIntegrityError("profile artifact hash is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise FacetProfileBundleIntegrityError(
                f"profile artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != sha256:
            raise FacetProfileBundleIntegrityError(
                f"profile artifact failed integrity check: {filename}"
            )
        payloads[filename] = payload
        previous_name = filename
    if set(payloads) != set(FACET_PROFILE_ARTIFACT_FILENAMES):
        raise FacetProfileBundleIntegrityError("profile artifact set is incomplete")
    return payloads


def _validate_paths(
    catalog_path: Path,
    category_candidate_dir: Path,
    selection_path: Path,
    target: Path,
) -> None:
    resolved_target = target.resolve()
    if resolved_target in {
        catalog_path.resolve(),
        category_candidate_dir.resolve(),
        selection_path.resolve(),
    }:
        raise ValueError("profile output path collides with an input")
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"profile output is not a directory: {target}")
    if target.exists():
        allowed = {*FACET_PROFILE_ARTIFACT_FILENAMES, FACET_PROFILE_MANIFEST_FILENAME}
        unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
        if unexpected:
            raise FacetProfileBundleIntegrityError(
                f"profile output contains an unexpected entry: {unexpected[0]}"
            )


@contextmanager
def _exclusive_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock_path = resolved.parent / f".{resolved.name or 'catalog-facet-profile'}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FacetProfileBundleBusyError(
            f"Gate-A source profile is already being written: {resolved}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
