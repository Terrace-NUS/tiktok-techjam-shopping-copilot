"""Atomic publication and exact validation of owner-approved Gate-B capabilities."""

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
from ..category import CategoryRegistry, decode_category_registry
from ..errors import (
    GateBBuildError,
    GateBBundleBusyError,
    GateBBundleIntegrityError,
)
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT
from .gate_a_bundle import load_gate_a_candidate_bundle
from .gate_b_approval_build import build_gate_b_candidate
from .gate_b_approval_codec import (
    decode_effective_facet_capabilities,
    decode_gate_b_selection,
    encode_effective_facet_capabilities,
    gate_b_candidate_document,
)
from .gate_b_approval_reporting import gate_b_candidate_markdown
from .gate_b_build import DEFAULT_PUBLIC_TARGET_COUNT, GateBPriceReviewBuild
from .gate_b_bundle import validate_gate_b_review_bundle
from .gate_b_codec import decode_gate_b_price_review, decode_public_target_price_audit
from .gate_b_models import (
    GATE_B_BUILDER_VERSION,
    GATE_B_CANDIDATE_SCHEMA,
    GateBCandidateBuild,
)
from .resolution_codec import (
    decode_catalog_facet_stats,
    decode_facet_evidence_store,
    decode_product_facet_index,
)
from .resolution_models import RESOLUTION_CANDIDATE_SCHEMA, ResolutionCandidateBuild

GATE_B_CANDIDATE_BUNDLE_SCHEMA = "shopping-copilot/gate-b-candidate-bundle/v0"
GATE_B_CANDIDATE_MANIFEST_FILENAME = "bundle-manifest.json"
GATE_B_CANDIDATE_ARTIFACT_FILENAMES = (
    "candidate.json",
    "effective-facet-capabilities.json",
    "report.md",
    "reviewed-gate-b-selection.json",
)

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def write_gate_b_candidate_bundle(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    gate_b_review_dir: str | Path,
    gate_b_selection_path: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = DEFAULT_PUBLIC_TARGET_COUNT,
    enforce_official_gate: bool = True,
) -> GateBCandidateBuild:
    """Publish the exact approved capability set beside all immutable inputs."""

    catalog = Path(catalog_path)
    category_candidate = Path(category_candidate_dir)
    gate_a_candidate = Path(gate_a_candidate_dir)
    resolution_candidate = Path(resolution_candidate_dir)
    public_set = Path(public_set_path)
    gate_b_review = Path(gate_b_review_dir)
    gate_b_selection = Path(gate_b_selection_path)
    target = Path(output_dir)
    inputs = (
        catalog,
        category_candidate,
        gate_a_candidate,
        resolution_candidate,
        public_set,
        gate_b_review,
        gate_b_selection,
    )
    _validate_paths(inputs, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer(target):
        build, registry = _rebuild_candidate(
            catalog,
            category_candidate,
            gate_a_candidate,
            resolution_candidate,
            public_set,
            gate_b_review,
            gate_b_selection,
            expected_product_count=expected_product_count,
            expected_public_target_count=expected_public_target_count,
            enforce_official_gate=enforce_official_gate,
        )
        payloads = _candidate_payloads(build, registry=registry)
        _publish_bundle(
            target,
            build=build,
            expected_product_count=expected_product_count,
            expected_public_target_count=expected_public_target_count,
            enforce_official_gate=enforce_official_gate,
            payloads=payloads,
        )
    validate_gate_b_candidate_bundle(
        target,
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
    return build


def validate_gate_b_candidate_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    gate_b_review_dir: str | Path,
    gate_b_selection_path: str | Path,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = DEFAULT_PUBLIC_TARGET_COUNT,
    enforce_official_gate: bool = True,
) -> None:
    """Rebuild and compare the approved selection, capabilities, report, and manifest."""

    target = Path(output_dir)
    manifest = _load_manifest(target)
    payloads = _load_payloads(target, manifest)
    decoded_selection = decode_gate_b_selection(payloads["reviewed-gate-b-selection.json"])
    decoded_capabilities = decode_effective_facet_capabilities(
        payloads["effective-facet-capabilities.json"]
    )
    expected, registry = _rebuild_candidate(
        Path(catalog_path),
        Path(category_candidate_dir),
        Path(gate_a_candidate_dir),
        Path(resolution_candidate_dir),
        Path(public_set_path),
        Path(gate_b_review_dir),
        Path(gate_b_selection_path),
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    if decoded_selection != expected.selection or decoded_capabilities != expected.capabilities:
        raise GateBBundleIntegrityError("approved Gate-B artifacts differ from upstream truth")
    expected_fields = _manifest_identity_fields(
        expected,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    for name, value in expected_fields.items():
        if manifest[name] != value:
            raise GateBBundleIntegrityError(f"approved Gate-B manifest field is stale: {name}")
    expected_payloads = _candidate_payloads(expected, registry=registry)
    for filename in GATE_B_CANDIDATE_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise GateBBundleIntegrityError(
                f"approved Gate-B artifact differs from exact truth: {filename}"
            )


def load_gate_b_candidate_bundle(output_dir: str | Path) -> GateBCandidateBuild:
    """Load a hash-verified approved capability bundle for downstream consumers.

    This verifies the bundle's own canonical bytes and cross-references. Stage
    boundaries should still call :func:`validate_gate_b_candidate_bundle` to
    rebuild from the frozen catalog and every reviewed upstream input.
    """

    target = Path(output_dir)
    manifest = _load_manifest(target)
    payloads = _load_payloads(target, manifest)
    selection = decode_gate_b_selection(payloads["reviewed-gate-b-selection.json"])
    capabilities = decode_effective_facet_capabilities(
        payloads["effective-facet-capabilities.json"]
    )
    try:
        build = GateBCandidateBuild(
            schema=GATE_B_CANDIDATE_SCHEMA,
            builder_version=GATE_B_BUILDER_VERSION,
            catalog_id=cast(str, manifest["catalog_id"]),
            catalog_facet_stats_id=cast(str, manifest["catalog_facet_stats_id"]),
            gate_b_review_proposal_id=cast(str, manifest["gate_b_review_proposal_id"]),
            public_target_audit_id=cast(str, manifest["public_target_audit_id"]),
            selection=selection,
            capabilities=capabilities,
        )
    except (TypeError, ValueError) as error:
        raise GateBBundleIntegrityError("approved Gate-B artifacts disagree") from error
    expected_fields = _manifest_identity_fields(
        build,
        expected_product_count=cast(int, manifest["expected_product_count"]),
        expected_public_target_count=cast(int, manifest["expected_public_target_count"]),
        enforce_official_gate=cast(bool, manifest["official_gate"]),
    )
    for name, value in expected_fields.items():
        if manifest[name] != value:
            raise GateBBundleIntegrityError(f"approved Gate-B manifest field is stale: {name}")
    if payloads["candidate.json"] != canonical_json_bytes(gate_b_candidate_document(build)):
        raise GateBBundleIntegrityError("approved Gate-B candidate metadata is stale")
    if payloads["reviewed-gate-b-selection.json"] != canonical_json_bytes(selection):
        raise GateBBundleIntegrityError("approved Gate-B selection copy is not canonical")
    if payloads["effective-facet-capabilities.json"] != encode_effective_facet_capabilities(
        capabilities
    ):
        raise GateBBundleIntegrityError("approved Gate-B capability bytes are stale")
    return build


def _rebuild_candidate(
    catalog: Path,
    category_candidate: Path,
    gate_a_candidate: Path,
    resolution_candidate: Path,
    public_set: Path,
    gate_b_review: Path,
    gate_b_selection: Path,
    *,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
) -> tuple[GateBCandidateBuild, CategoryRegistry]:
    validate_gate_b_review_bundle(
        gate_b_review,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        gate_a_candidate_dir=gate_a_candidate,
        resolution_candidate_dir=resolution_candidate,
        public_set_path=public_set,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    try:
        selection_bytes = gate_b_selection.read_bytes()
        registry_bytes = (category_candidate / "category-registry.json").read_bytes()
        evidence_bytes = (resolution_candidate / "facet-evidence-store.json").read_bytes()
        index_bytes = (resolution_candidate / "product-facet-index.json").read_bytes()
        stats_bytes = (resolution_candidate / "catalog-facet-stats.json").read_bytes()
        review_proposal_bytes = (gate_b_review / "price-review-proposal.json").read_bytes()
        public_audit_bytes = (gate_b_review / "public-target-audit.json").read_bytes()
    except OSError as error:
        raise GateBBuildError("approved Gate-B input is unavailable") from error
    selection = decode_gate_b_selection(selection_bytes)
    registry = decode_category_registry(registry_bytes)
    gate_a = load_gate_a_candidate_bundle(gate_a_candidate)
    try:
        resolution = ResolutionCandidateBuild(
            schema=RESOLUTION_CANDIDATE_SCHEMA,
            builder_version=gate_a.builder_version,
            category_registry_id=content_id_for_bytes(registry_bytes),
            facet_schema_id=content_id_for_value(gate_a.facet_schema),
            gate_a_selection_id=content_id_for_value(gate_a.selection),
            evidence_store=decode_facet_evidence_store(evidence_bytes),
            product_facet_index=decode_product_facet_index(index_bytes),
            stats=decode_catalog_facet_stats(stats_bytes),
        )
        review = GateBPriceReviewBuild(
            proposal=decode_gate_b_price_review(review_proposal_bytes),
            public_target_audit=decode_public_target_price_audit(public_audit_bytes),
        )
    except (TypeError, ValueError) as error:
        raise GateBBundleIntegrityError("approved Gate-B inputs disagree") from error
    return (
        build_gate_b_candidate(
            selection,
            registry=registry,
            gate_a=gate_a,
            resolution=resolution,
            review=review,
        ),
        registry,
    )


def _candidate_payloads(
    build: GateBCandidateBuild,
    *,
    registry: CategoryRegistry,
) -> dict[str, bytes]:
    return {
        "candidate.json": canonical_json_bytes(gate_b_candidate_document(build)),
        "effective-facet-capabilities.json": encode_effective_facet_capabilities(
            build.capabilities
        ),
        "report.md": gate_b_candidate_markdown(build, registry=registry).encode("utf-8"),
        "reviewed-gate-b-selection.json": canonical_json_bytes(build.selection),
    }


def _publish_bundle(
    target: Path,
    *,
    build: GateBCandidateBuild,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".gate-b-candidate-", dir=target.parent) as temporary:
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
            "schema": GATE_B_CANDIDATE_BUNDLE_SCHEMA,
            **_manifest_identity_fields(
                build,
                expected_product_count=expected_product_count,
                expected_public_target_count=expected_public_target_count,
                enforce_official_gate=enforce_official_gate,
            ),
            "owner_approval_recorded": True,
            "runtime_integration_complete": False,
            "artifacts": artifacts,
        }
        (staging / GATE_B_CANDIDATE_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        target.mkdir(parents=True, exist_ok=True)
        for filename in GATE_B_CANDIDATE_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / GATE_B_CANDIDATE_MANIFEST_FILENAME,
            target / GATE_B_CANDIDATE_MANIFEST_FILENAME,
        )


def _manifest_identity_fields(
    build: GateBCandidateBuild,
    *,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
) -> dict[str, object]:
    return {
        "builder_version": GATE_B_BUILDER_VERSION,
        "catalog_id": build.catalog_id,
        "category_registry_id": build.capabilities.category_registry_id,
        "facet_schema_id": build.capabilities.facet_schema_id,
        "facet_applicability_id": build.capabilities.facet_applicability_id,
        "product_facet_index_id": build.capabilities.product_facet_index_id,
        "catalog_facet_stats_id": build.catalog_facet_stats_id,
        "gate_b_review_proposal_id": build.gate_b_review_proposal_id,
        "public_target_audit_id": build.public_target_audit_id,
        "gate_b_selection_id": content_id_for_value(build.selection),
        "effective_facet_capabilities_id": content_id_for_value(build.capabilities),
        "expected_product_count": expected_product_count,
        "expected_public_target_count": expected_public_target_count,
        "official_gate": enforce_official_gate,
    }


def _load_manifest(target: Path) -> dict[str, object]:
    try:
        data = (target / GATE_B_CANDIDATE_MANIFEST_FILENAME).read_bytes()
        parsed: object = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GateBBundleIntegrityError("approved Gate-B manifest is unavailable") from error
    if type(parsed) is not dict or canonical_json_bytes(parsed) != data:
        raise GateBBundleIntegrityError("approved Gate-B manifest is not canonical")
    manifest = cast(dict[str, object], parsed)
    fields = {
        "schema",
        "builder_version",
        "catalog_id",
        "category_registry_id",
        "facet_schema_id",
        "facet_applicability_id",
        "product_facet_index_id",
        "catalog_facet_stats_id",
        "gate_b_review_proposal_id",
        "public_target_audit_id",
        "gate_b_selection_id",
        "effective_facet_capabilities_id",
        "expected_product_count",
        "expected_public_target_count",
        "official_gate",
        "owner_approval_recorded",
        "runtime_integration_complete",
        "artifacts",
    }
    if set(manifest) != fields or manifest["schema"] != GATE_B_CANDIDATE_BUNDLE_SCHEMA:
        raise GateBBundleIntegrityError("approved Gate-B manifest fields are invalid")
    for name in (
        "catalog_id",
        "category_registry_id",
        "facet_schema_id",
        "facet_applicability_id",
        "product_facet_index_id",
        "catalog_facet_stats_id",
        "gate_b_review_proposal_id",
        "public_target_audit_id",
        "gate_b_selection_id",
        "effective_facet_capabilities_id",
    ):
        value = manifest[name]
        if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise GateBBundleIntegrityError(f"approved Gate-B manifest {name} is invalid")
    if (
        manifest["builder_version"] != GATE_B_BUILDER_VERSION
        or type(manifest["expected_product_count"]) is not int
        or type(manifest["expected_public_target_count"]) is not int
        or type(manifest["official_gate"]) is not bool
        or manifest["owner_approval_recorded"] is not True
        or manifest["runtime_integration_complete"] is not False
        or type(manifest["artifacts"]) is not list
    ):
        raise GateBBundleIntegrityError("approved Gate-B manifest controls are invalid")
    return manifest


def _load_payloads(target: Path, manifest: dict[str, object]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    previous: str | None = None
    for raw in cast(list[object], manifest["artifacts"]):
        if type(raw) is not dict:
            raise GateBBundleIntegrityError("approved Gate-B artifact entry is invalid")
        artifact = cast(dict[str, object], raw)
        if set(artifact) != {"byte_size", "filename", "sha256"}:
            raise GateBBundleIntegrityError("approved Gate-B artifact fields are invalid")
        filename = artifact["filename"]
        byte_size = artifact["byte_size"]
        digest = artifact["sha256"]
        if type(filename) is not str or filename not in GATE_B_CANDIDATE_ARTIFACT_FILENAMES:
            raise GateBBundleIntegrityError("approved Gate-B artifact filename is invalid")
        if filename in payloads or (previous is not None and filename <= previous):
            raise GateBBundleIntegrityError("approved Gate-B artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise GateBBundleIntegrityError("approved Gate-B artifact byte size is invalid")
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise GateBBundleIntegrityError("approved Gate-B artifact hash is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise GateBBundleIntegrityError(
                f"approved Gate-B artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != digest:
            raise GateBBundleIntegrityError(
                f"approved Gate-B artifact failed integrity: {filename}"
            )
        payloads[filename] = payload
        previous = filename
    if set(payloads) != set(GATE_B_CANDIDATE_ARTIFACT_FILENAMES):
        raise GateBBundleIntegrityError("approved Gate-B artifact set is incomplete")
    return payloads


def _validate_paths(inputs: tuple[Path, ...], target: Path) -> None:
    resolved_target = target.resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"approved Gate-B output is not a directory: {target}")
    for input_path in inputs:
        resolved_input = input_path.resolve()
        if resolved_input == resolved_target or resolved_target in resolved_input.parents:
            raise ValueError("approved Gate-B output must not contain or replace any input")
    if target.exists():
        allowed = {
            *GATE_B_CANDIDATE_ARTIFACT_FILENAMES,
            GATE_B_CANDIDATE_MANIFEST_FILENAME,
        }
        unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
        if unexpected:
            raise GateBBundleIntegrityError(
                f"approved Gate-B output contains an unexpected entry: {unexpected[0]}"
            )


@contextmanager
def _exclusive_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock_path = resolved.parent / f".{resolved.name or 'gate-b-candidate'}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise GateBBundleBusyError(
            f"approved Gate-B candidate is already being written: {resolved}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
