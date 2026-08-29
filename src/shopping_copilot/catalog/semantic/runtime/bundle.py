"""Atomic publication, exact validation, and safe loading for CS5A."""

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

from shopping_copilot.session_context import FacetRegistry

from ..canonical import canonical_json_bytes, content_id_for_value
from ..category import decode_category_registry
from ..errors import (
    RuntimeProjectionBuildError,
    RuntimeProjectionBundleBusyError,
    RuntimeProjectionBundleIntegrityError,
)
from ..facet.gate_a_bundle import load_gate_a_candidate_bundle
from ..facet.gate_b_approval_bundle import (
    load_gate_b_candidate_bundle,
    validate_gate_b_candidate_bundle,
)
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT
from .build import build_runtime_projection_candidate, project_session_facet_registry
from .codec import (
    decode_runtime_facet_registry,
    decode_runtime_value_lexicon,
    encode_runtime_facet_registry,
    encode_runtime_value_lexicon,
    runtime_projection_candidate_document,
)
from .grounding import RuntimeValueGrounder
from .models import (
    RUNTIME_PROJECTION_BUILDER_VERSION,
    RUNTIME_PROJECTION_CANDIDATE_SCHEMA,
    RuntimeProjectionCandidateBuild,
)
from .reporting import runtime_projection_candidate_markdown

RUNTIME_PROJECTION_BUNDLE_SCHEMA = "shopping-copilot/runtime-projection-bundle/v0"
RUNTIME_PROJECTION_MANIFEST_FILENAME = "bundle-manifest.json"
RUNTIME_PROJECTION_ARTIFACT_FILENAMES = (
    "candidate.json",
    "report.md",
    "runtime-facet-registry.json",
    "runtime-value-lexicon.json",
)

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def write_runtime_projection_bundle(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    gate_b_review_dir: str | Path,
    gate_b_selection_path: str | Path,
    gate_b_candidate_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = 200,
    enforce_official_gate: bool = True,
) -> RuntimeProjectionCandidateBuild:
    """Build and atomically publish the approved CS5A runtime projection."""

    inputs = tuple(
        Path(item)
        for item in (
            catalog_path,
            category_candidate_dir,
            gate_a_candidate_dir,
            resolution_candidate_dir,
            public_set_path,
            gate_b_review_dir,
            gate_b_selection_path,
            gate_b_candidate_dir,
        )
    )
    target = Path(output_dir)
    _validate_paths(inputs, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer(target):
        build = _rebuild_candidate(
            *inputs,
            expected_product_count=expected_product_count,
            expected_public_target_count=expected_public_target_count,
            enforce_official_gate=enforce_official_gate,
        )
        payloads = _candidate_payloads(build)
        _publish_bundle(
            target,
            build=build,
            expected_product_count=expected_product_count,
            expected_public_target_count=expected_public_target_count,
            enforce_official_gate=enforce_official_gate,
            payloads=payloads,
        )
    validate_runtime_projection_bundle(
        target,
        catalog_path=inputs[0],
        category_candidate_dir=inputs[1],
        gate_a_candidate_dir=inputs[2],
        resolution_candidate_dir=inputs[3],
        public_set_path=inputs[4],
        gate_b_review_dir=inputs[5],
        gate_b_selection_path=inputs[6],
        gate_b_candidate_dir=inputs[7],
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    return build


def validate_runtime_projection_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    gate_b_review_dir: str | Path,
    gate_b_selection_path: str | Path,
    gate_b_candidate_dir: str | Path,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = 200,
    enforce_official_gate: bool = True,
) -> None:
    """Rebuild from the complete approval chain and compare every CS5A byte."""

    target = Path(output_dir)
    manifest = _load_manifest(target)
    payloads = _load_payloads(target, manifest)
    decoded_registry = decode_runtime_facet_registry(payloads["runtime-facet-registry.json"])
    decoded_lexicon = decode_runtime_value_lexicon(payloads["runtime-value-lexicon.json"])
    inputs = tuple(
        Path(item)
        for item in (
            catalog_path,
            category_candidate_dir,
            gate_a_candidate_dir,
            resolution_candidate_dir,
            public_set_path,
            gate_b_review_dir,
            gate_b_selection_path,
            gate_b_candidate_dir,
        )
    )
    expected = _rebuild_candidate(
        *inputs,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    if decoded_registry != expected.runtime_registry or decoded_lexicon != expected.runtime_lexicon:
        raise RuntimeProjectionBundleIntegrityError("CS5A artifacts differ from upstream truth")
    expected_fields = _manifest_identity_fields(
        expected,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    for name, value in expected_fields.items():
        if manifest[name] != value:
            raise RuntimeProjectionBundleIntegrityError(f"CS5A manifest field is stale: {name}")
    expected_payloads = _candidate_payloads(expected)
    for filename in RUNTIME_PROJECTION_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise RuntimeProjectionBundleIntegrityError(
                f"CS5A artifact differs from exact upstream truth: {filename}"
            )


def load_runtime_projection_bundle(
    output_dir: str | Path,
) -> RuntimeProjectionCandidateBuild:
    """Load a hash-verified CS5A bundle without rebuilding the full approval chain."""

    target = Path(output_dir)
    manifest = _load_manifest(target)
    payloads = _load_payloads(target, manifest)
    runtime_registry = decode_runtime_facet_registry(payloads["runtime-facet-registry.json"])
    runtime_lexicon = decode_runtime_value_lexicon(payloads["runtime-value-lexicon.json"])
    try:
        build = RuntimeProjectionCandidateBuild(
            schema=RUNTIME_PROJECTION_CANDIDATE_SCHEMA,
            builder_version=RUNTIME_PROJECTION_BUILDER_VERSION,
            catalog_id=cast(str, manifest["catalog_id"]),
            gate_b_selection_id=cast(str, manifest["gate_b_selection_id"]),
            effective_capabilities_id=cast(str, manifest["effective_capabilities_id"]),
            runtime_registry=runtime_registry,
            runtime_lexicon=runtime_lexicon,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeProjectionBundleIntegrityError("CS5A artifacts disagree") from error
    expected_fields = _manifest_identity_fields(
        build,
        expected_product_count=cast(int, manifest["expected_product_count"]),
        expected_public_target_count=cast(int, manifest["expected_public_target_count"]),
        enforce_official_gate=cast(bool, manifest["official_gate"]),
    )
    for name, value in expected_fields.items():
        if manifest[name] != value:
            raise RuntimeProjectionBundleIntegrityError(f"CS5A manifest field is stale: {name}")
    if payloads["candidate.json"] != canonical_json_bytes(
        runtime_projection_candidate_document(build)
    ):
        raise RuntimeProjectionBundleIntegrityError("CS5A candidate metadata is stale")
    return build


def load_projected_session_facet_registry(
    runtime_projection_dir: str | Path,
    *,
    category_candidate_dir: str | Path,
    gate_b_candidate_dir: str | Path,
) -> FacetRegistry:
    """Load verified DTOs and resolve the two release-bound FacetSpec callables."""

    runtime = load_runtime_projection_bundle(runtime_projection_dir)
    gate_b = load_gate_b_candidate_bundle(gate_b_candidate_dir)
    try:
        registry = decode_category_registry(
            (Path(category_candidate_dir) / "category-registry.json").read_bytes()
        )
    except OSError as error:
        raise RuntimeProjectionBuildError("CategoryRegistry is unavailable") from error
    return project_session_facet_registry(
        runtime_registry=runtime.runtime_registry,
        runtime_lexicon=runtime.runtime_lexicon,
        category_registry=registry,
        capabilities=gate_b.capabilities,
    )


def load_runtime_value_grounder(
    runtime_projection_dir: str | Path,
    *,
    category_candidate_dir: str | Path,
    gate_b_candidate_dir: str | Path,
) -> RuntimeValueGrounder:
    """Load one CS5B grounder from mutually pinned, validated candidate artifacts."""

    runtime = load_runtime_projection_bundle(runtime_projection_dir)
    gate_b = load_gate_b_candidate_bundle(gate_b_candidate_dir)
    try:
        registry = decode_category_registry(
            (Path(category_candidate_dir) / "category-registry.json").read_bytes()
        )
    except OSError as error:
        raise RuntimeProjectionBuildError("CategoryRegistry is unavailable") from error
    return RuntimeValueGrounder(
        runtime_registry=runtime.runtime_registry,
        runtime_lexicon=runtime.runtime_lexicon,
        category_registry=registry,
        capabilities=gate_b.capabilities,
    )


def _rebuild_candidate(
    catalog: Path,
    category_candidate: Path,
    gate_a_candidate: Path,
    resolution_candidate: Path,
    public_set: Path,
    gate_b_review: Path,
    gate_b_selection: Path,
    gate_b_candidate: Path,
    *,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
) -> RuntimeProjectionCandidateBuild:
    validate_gate_b_candidate_bundle(
        gate_b_candidate,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        gate_a_candidate_dir=gate_a_candidate,
        resolution_candidate_dir=resolution_candidate,
        public_set_path=public_set,
        gate_b_review_dir=gate_b_review,
        gate_b_selection_path=gate_b_selection,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    try:
        registry = decode_category_registry(
            (category_candidate / "category-registry.json").read_bytes()
        )
    except OSError as error:
        raise RuntimeProjectionBuildError("CS5A CategoryRegistry input is unavailable") from error
    gate_a = load_gate_a_candidate_bundle(gate_a_candidate)
    gate_b = load_gate_b_candidate_bundle(gate_b_candidate)
    return build_runtime_projection_candidate(
        registry=registry,
        gate_a=gate_a,
        gate_b=gate_b,
    )


def _candidate_payloads(build: RuntimeProjectionCandidateBuild) -> dict[str, bytes]:
    return {
        "candidate.json": canonical_json_bytes(runtime_projection_candidate_document(build)),
        "report.md": runtime_projection_candidate_markdown(build).encode("utf-8"),
        "runtime-facet-registry.json": encode_runtime_facet_registry(build.runtime_registry),
        "runtime-value-lexicon.json": encode_runtime_value_lexicon(build.runtime_lexicon),
    }


def _publish_bundle(
    target: Path,
    *,
    build: RuntimeProjectionCandidateBuild,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".runtime-projection-", dir=target.parent) as temporary:
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
            "schema": RUNTIME_PROJECTION_BUNDLE_SCHEMA,
            **_manifest_identity_fields(
                build,
                expected_product_count=expected_product_count,
                expected_public_target_count=expected_public_target_count,
                enforce_official_gate=enforce_official_gate,
            ),
            "grounding_implemented": True,
            "retrieval_integrated": False,
            "session_gateway_integrated": False,
            "artifacts": artifacts,
        }
        (staging / RUNTIME_PROJECTION_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        target.mkdir(parents=True, exist_ok=True)
        for filename in RUNTIME_PROJECTION_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / RUNTIME_PROJECTION_MANIFEST_FILENAME,
            target / RUNTIME_PROJECTION_MANIFEST_FILENAME,
        )


def _manifest_identity_fields(
    build: RuntimeProjectionCandidateBuild,
    *,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
) -> dict[str, object]:
    return {
        "builder_version": build.builder_version,
        "catalog_id": build.catalog_id,
        "category_registry_id": build.runtime_registry.category_registry_id,
        "facet_schema_id": build.runtime_registry.facet_schema_id,
        "facet_applicability_id": build.runtime_lexicon.facet_applicability_id,
        "product_facet_index_id": build.runtime_lexicon.product_facet_index_id,
        "gate_b_selection_id": build.gate_b_selection_id,
        "effective_capabilities_id": build.effective_capabilities_id,
        "runtime_facet_registry_id": content_id_for_value(build.runtime_registry),
        "runtime_value_lexicon_id": content_id_for_value(build.runtime_lexicon),
        "expected_product_count": expected_product_count,
        "expected_public_target_count": expected_public_target_count,
        "official_gate": enforce_official_gate,
    }


def _load_manifest(target: Path) -> dict[str, object]:
    try:
        data = (target / RUNTIME_PROJECTION_MANIFEST_FILENAME).read_bytes()
        parsed: object = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RuntimeProjectionBundleIntegrityError("CS5A manifest is unavailable") from error
    if type(parsed) is not dict or canonical_json_bytes(parsed) != data:
        raise RuntimeProjectionBundleIntegrityError("CS5A manifest is not canonical")
    manifest = cast(dict[str, object], parsed)
    fields = {
        "schema",
        "builder_version",
        "catalog_id",
        "category_registry_id",
        "facet_schema_id",
        "facet_applicability_id",
        "product_facet_index_id",
        "gate_b_selection_id",
        "effective_capabilities_id",
        "runtime_facet_registry_id",
        "runtime_value_lexicon_id",
        "expected_product_count",
        "expected_public_target_count",
        "official_gate",
        "grounding_implemented",
        "retrieval_integrated",
        "session_gateway_integrated",
        "artifacts",
    }
    if set(manifest) != fields or manifest["schema"] != RUNTIME_PROJECTION_BUNDLE_SCHEMA:
        raise RuntimeProjectionBundleIntegrityError("CS5A manifest fields are invalid")
    for name in (
        "catalog_id",
        "category_registry_id",
        "facet_schema_id",
        "facet_applicability_id",
        "product_facet_index_id",
        "gate_b_selection_id",
        "effective_capabilities_id",
        "runtime_facet_registry_id",
        "runtime_value_lexicon_id",
    ):
        value = manifest[name]
        if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise RuntimeProjectionBundleIntegrityError(f"CS5A manifest {name} is invalid")
    if (
        manifest["builder_version"] != RUNTIME_PROJECTION_BUILDER_VERSION
        or type(manifest["expected_product_count"]) is not int
        or type(manifest["expected_public_target_count"]) is not int
        or type(manifest["official_gate"]) is not bool
        or manifest["grounding_implemented"] is not True
        or manifest["retrieval_integrated"] is not False
        or manifest["session_gateway_integrated"] is not False
        or type(manifest["artifacts"]) is not list
    ):
        raise RuntimeProjectionBundleIntegrityError("CS5A manifest controls are invalid")
    return manifest


def _load_payloads(target: Path, manifest: dict[str, object]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    previous: str | None = None
    for raw in cast(list[object], manifest["artifacts"]):
        if type(raw) is not dict:
            raise RuntimeProjectionBundleIntegrityError("CS5A artifact entry is invalid")
        artifact = cast(dict[str, object], raw)
        if set(artifact) != {"byte_size", "filename", "sha256"}:
            raise RuntimeProjectionBundleIntegrityError("CS5A artifact fields are invalid")
        filename = artifact["filename"]
        byte_size = artifact["byte_size"]
        digest = artifact["sha256"]
        if type(filename) is not str or filename not in RUNTIME_PROJECTION_ARTIFACT_FILENAMES:
            raise RuntimeProjectionBundleIntegrityError("CS5A artifact filename is invalid")
        if filename in payloads or (previous is not None and filename <= previous):
            raise RuntimeProjectionBundleIntegrityError("CS5A artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise RuntimeProjectionBundleIntegrityError("CS5A artifact byte size is invalid")
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeProjectionBundleIntegrityError("CS5A artifact hash is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise RuntimeProjectionBundleIntegrityError(
                f"CS5A artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeProjectionBundleIntegrityError(
                f"CS5A artifact failed integrity: {filename}"
            )
        payloads[filename] = payload
        previous = filename
    if set(payloads) != set(RUNTIME_PROJECTION_ARTIFACT_FILENAMES):
        raise RuntimeProjectionBundleIntegrityError("CS5A artifact set is incomplete")
    return payloads


def _validate_paths(inputs: tuple[Path, ...], target: Path) -> None:
    resolved_target = target.resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"CS5A output is not a directory: {target}")
    for input_path in inputs:
        resolved_input = input_path.resolve()
        if resolved_input == resolved_target or resolved_target in resolved_input.parents:
            raise ValueError("CS5A output must not contain or replace any input")
    if target.exists():
        allowed = {*RUNTIME_PROJECTION_ARTIFACT_FILENAMES, RUNTIME_PROJECTION_MANIFEST_FILENAME}
        unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
        if unexpected:
            raise RuntimeProjectionBundleIntegrityError(
                f"CS5A output contains an unexpected entry: {unexpected[0]}"
            )


@contextmanager
def _exclusive_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock_path = resolved.parent / f".{resolved.name or 'runtime-projection'}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeProjectionBundleBusyError(
            f"CS5A candidate is already being written: {resolved}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
