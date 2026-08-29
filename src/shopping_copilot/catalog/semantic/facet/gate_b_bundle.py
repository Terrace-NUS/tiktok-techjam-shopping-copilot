"""Atomic publication and exact rebuild validation for Gate-B review packets."""

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
    ProductCategoryAssignmentSet,
    decode_category_registry,
    decode_product_category_assignment_set,
)
from ..errors import (
    GateBReviewBuildError,
    GateBReviewBundleBusyError,
    GateBReviewBundleIntegrityError,
)
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT
from .gate_a_bundle import load_gate_a_candidate_bundle
from .gate_a_models import GateACandidateBuild
from .gate_b_build import (
    DEFAULT_PUBLIC_TARGET_COUNT,
    GateBPriceReviewBuild,
    build_gate_b_price_review,
)
from .gate_b_codec import (
    decode_gate_b_price_review,
    decode_public_target_price_audit,
    encode_gate_b_price_review,
    encode_public_target_price_audit,
    gate_b_review_candidate_document,
)
from .gate_b_models import GATE_B_REVIEW_BUILDER_VERSION
from .gate_b_reporting import gate_b_price_review_markdown
from .resolution_bundle import validate_resolution_candidate_bundle
from .resolution_codec import (
    decode_catalog_facet_stats,
    decode_facet_evidence_store,
    decode_product_facet_index,
)
from .resolution_models import RESOLUTION_CANDIDATE_SCHEMA, ResolutionCandidateBuild

GATE_B_REVIEW_BUNDLE_SCHEMA = "shopping-copilot/gate-b-review-bundle/v0"
GATE_B_REVIEW_MANIFEST_FILENAME = "bundle-manifest.json"
GATE_B_REVIEW_ARTIFACT_FILENAMES = (
    "candidate.json",
    "price-review-proposal.json",
    "public-target-audit.json",
    "report.md",
)

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def write_gate_b_review_bundle(
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = DEFAULT_PUBLIC_TARGET_COUNT,
    enforce_official_gate: bool = True,
) -> GateBPriceReviewBuild:
    """Build and atomically publish review evidence without publishing capabilities."""

    catalog = Path(catalog_path)
    category_candidate = Path(category_candidate_dir)
    gate_a_candidate = Path(gate_a_candidate_dir)
    resolution_candidate = Path(resolution_candidate_dir)
    public_set = Path(public_set_path)
    target = Path(output_dir)
    inputs = (
        catalog,
        category_candidate,
        gate_a_candidate,
        resolution_candidate,
        public_set,
    )
    _validate_paths(inputs, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer(target):
        build, resolution = _rebuild_review(
            catalog,
            category_candidate,
            gate_a_candidate,
            resolution_candidate,
            public_set,
            expected_product_count=expected_product_count,
            expected_public_target_count=expected_public_target_count,
            enforce_official_gate=enforce_official_gate,
        )
        payloads = _review_payloads(build, resolution=resolution)
        _publish_bundle(
            target,
            build=build,
            expected_product_count=expected_product_count,
            expected_public_target_count=expected_public_target_count,
            enforce_official_gate=enforce_official_gate,
            payloads=payloads,
        )
    validate_gate_b_review_bundle(
        target,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        gate_a_candidate_dir=gate_a_candidate,
        resolution_candidate_dir=resolution_candidate,
        public_set_path=public_set,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    return build


def validate_gate_b_review_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path,
    category_candidate_dir: str | Path,
    gate_a_candidate_dir: str | Path,
    resolution_candidate_dir: str | Path,
    public_set_path: str | Path,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    expected_public_target_count: int = DEFAULT_PUBLIC_TARGET_COUNT,
    enforce_official_gate: bool = True,
) -> None:
    """Validate packet hashes, canonical DTOs, upstream pins, and exact reproducibility."""

    target = Path(output_dir)
    manifest = _load_manifest(target)
    payloads = _load_payloads(target, manifest)
    decoded_proposal = decode_gate_b_price_review(payloads["price-review-proposal.json"])
    decoded_audit = decode_public_target_price_audit(payloads["public-target-audit.json"])
    try:
        decoded = GateBPriceReviewBuild(
            proposal=decoded_proposal,
            public_target_audit=decoded_audit,
        )
    except (TypeError, ValueError) as error:
        raise GateBReviewBundleIntegrityError("Gate-B review artifacts disagree") from error

    expected, resolution = _rebuild_review(
        Path(catalog_path),
        Path(category_candidate_dir),
        Path(gate_a_candidate_dir),
        Path(resolution_candidate_dir),
        Path(public_set_path),
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    if decoded != expected:
        raise GateBReviewBundleIntegrityError("Gate-B packet differs from exact upstream truth")
    expected_fields = _manifest_identity_fields(
        expected,
        expected_product_count=expected_product_count,
        expected_public_target_count=expected_public_target_count,
        enforce_official_gate=enforce_official_gate,
    )
    for name, value in expected_fields.items():
        if manifest[name] != value:
            raise GateBReviewBundleIntegrityError(f"Gate-B manifest field is stale: {name}")
    expected_payloads = _review_payloads(expected, resolution=resolution)
    for filename in GATE_B_REVIEW_ARTIFACT_FILENAMES:
        if payloads[filename] != expected_payloads[filename]:
            raise GateBReviewBundleIntegrityError(
                f"Gate-B review artifact differs from exact upstream truth: {filename}"
            )


def _rebuild_review(
    catalog: Path,
    category_candidate: Path,
    gate_a_candidate: Path,
    resolution_candidate: Path,
    public_set: Path,
    *,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
) -> tuple[GateBPriceReviewBuild, ResolutionCandidateBuild]:
    validate_resolution_candidate_bundle(
        resolution_candidate,
        catalog_path=catalog,
        category_candidate_dir=category_candidate,
        gate_a_candidate_dir=gate_a_candidate,
        expected_product_count=expected_product_count,
        enforce_official_gate=enforce_official_gate,
    )
    registry, assignments, gate_a, resolution = _load_verified_inputs(
        category_candidate,
        gate_a_candidate,
        resolution_candidate,
    )
    return (
        build_gate_b_price_review(
            public_set,
            registry=registry,
            assignments=assignments,
            gate_a=gate_a,
            resolution=resolution,
            expected_public_target_count=expected_public_target_count,
        ),
        resolution,
    )


def _load_verified_inputs(
    category_candidate: Path,
    gate_a_candidate: Path,
    resolution_candidate: Path,
) -> tuple[
    CategoryRegistry,
    ProductCategoryAssignmentSet,
    GateACandidateBuild,
    ResolutionCandidateBuild,
]:
    try:
        registry_bytes = (category_candidate / "category-registry.json").read_bytes()
        assignment_bytes = (category_candidate / "product-category-assignment.json").read_bytes()
        evidence_bytes = (resolution_candidate / "facet-evidence-store.json").read_bytes()
        index_bytes = (resolution_candidate / "product-facet-index.json").read_bytes()
        stats_bytes = (resolution_candidate / "catalog-facet-stats.json").read_bytes()
    except OSError as error:
        raise GateBReviewBuildError("Gate-B verified input is unavailable") from error
    registry = decode_category_registry(registry_bytes)
    assignments = decode_product_category_assignment_set(
        assignment_bytes,
        registry=registry,
    )
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
    except (TypeError, ValueError) as error:
        raise GateBReviewBundleIntegrityError("verified CS3 artifacts disagree") from error
    return registry, assignments, gate_a, resolution


def _review_payloads(
    build: GateBPriceReviewBuild,
    *,
    resolution: ResolutionCandidateBuild,
) -> dict[str, bytes]:
    return {
        "candidate.json": canonical_json_bytes(
            gate_b_review_candidate_document(build.proposal, build.public_target_audit)
        ),
        "price-review-proposal.json": encode_gate_b_price_review(build.proposal),
        "public-target-audit.json": encode_public_target_price_audit(build.public_target_audit),
        "report.md": gate_b_price_review_markdown(build, resolution=resolution).encode("utf-8"),
    }


def _publish_bundle(
    target: Path,
    *,
    build: GateBPriceReviewBuild,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".gate-b-review-", dir=target.parent) as temporary:
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
            "schema": GATE_B_REVIEW_BUNDLE_SCHEMA,
            **_manifest_identity_fields(
                build,
                expected_product_count=expected_product_count,
                expected_public_target_count=expected_public_target_count,
                enforce_official_gate=enforce_official_gate,
            ),
            "runtime_capability_published": False,
            "source_controlled_approval_present": False,
            "artifacts": artifacts,
        }
        (staging / GATE_B_REVIEW_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        target.mkdir(parents=True, exist_ok=True)
        for filename in GATE_B_REVIEW_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / GATE_B_REVIEW_MANIFEST_FILENAME,
            target / GATE_B_REVIEW_MANIFEST_FILENAME,
        )


def _manifest_identity_fields(
    build: GateBPriceReviewBuild,
    *,
    expected_product_count: int,
    expected_public_target_count: int,
    enforce_official_gate: bool,
) -> dict[str, object]:
    proposal = build.proposal
    audit = build.public_target_audit
    return {
        "builder_version": GATE_B_REVIEW_BUILDER_VERSION,
        "catalog_id": proposal.catalog_id,
        "category_registry_id": proposal.category_registry_id,
        "facet_schema_id": proposal.facet_schema_id,
        "facet_applicability_id": proposal.facet_applicability_id,
        "product_facet_index_id": proposal.product_facet_index_id,
        "catalog_facet_stats_id": proposal.catalog_facet_stats_id,
        "public_set_id": audit.public_set_id,
        "public_target_audit_id": content_id_for_value(audit),
        "price_review_proposal_id": content_id_for_value(proposal),
        "expected_product_count": expected_product_count,
        "expected_public_target_count": expected_public_target_count,
        "official_gate": enforce_official_gate,
    }


def _load_manifest(target: Path) -> dict[str, object]:
    try:
        data = (target / GATE_B_REVIEW_MANIFEST_FILENAME).read_bytes()
        parsed: object = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GateBReviewBundleIntegrityError(
            "Gate-B manifest is unavailable or invalid"
        ) from error
    if type(parsed) is not dict or canonical_json_bytes(parsed) != data:
        raise GateBReviewBundleIntegrityError("Gate-B manifest is not canonical")
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
        "public_set_id",
        "public_target_audit_id",
        "price_review_proposal_id",
        "expected_product_count",
        "expected_public_target_count",
        "official_gate",
        "runtime_capability_published",
        "source_controlled_approval_present",
        "artifacts",
    }
    if set(manifest) != fields or manifest["schema"] != GATE_B_REVIEW_BUNDLE_SCHEMA:
        raise GateBReviewBundleIntegrityError("Gate-B manifest has invalid fields or schema")
    for name in (
        "catalog_id",
        "category_registry_id",
        "facet_schema_id",
        "facet_applicability_id",
        "product_facet_index_id",
        "catalog_facet_stats_id",
        "public_set_id",
        "public_target_audit_id",
        "price_review_proposal_id",
    ):
        value = manifest[name]
        if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise GateBReviewBundleIntegrityError(f"Gate-B manifest {name} is invalid")
    if manifest["builder_version"] != GATE_B_REVIEW_BUILDER_VERSION:
        raise GateBReviewBundleIntegrityError("Gate-B manifest builder version is unsupported")
    if (
        type(manifest["expected_product_count"]) is not int
        or type(manifest["expected_public_target_count"]) is not int
        or type(manifest["official_gate"]) is not bool
        or manifest["runtime_capability_published"] is not False
        or manifest["source_controlled_approval_present"] is not False
        or type(manifest["artifacts"]) is not list
    ):
        raise GateBReviewBundleIntegrityError("Gate-B manifest control fields are invalid")
    return manifest


def _load_payloads(target: Path, manifest: dict[str, object]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    previous: str | None = None
    for raw in cast(list[object], manifest["artifacts"]):
        if type(raw) is not dict:
            raise GateBReviewBundleIntegrityError("Gate-B artifact entry is invalid")
        artifact = cast(dict[str, object], raw)
        if set(artifact) != {"byte_size", "filename", "sha256"}:
            raise GateBReviewBundleIntegrityError("Gate-B artifact fields are invalid")
        filename = artifact["filename"]
        byte_size = artifact["byte_size"]
        digest = artifact["sha256"]
        if type(filename) is not str or filename not in GATE_B_REVIEW_ARTIFACT_FILENAMES:
            raise GateBReviewBundleIntegrityError("Gate-B artifact filename is invalid")
        if filename in payloads or (previous is not None and filename <= previous):
            raise GateBReviewBundleIntegrityError("Gate-B artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise GateBReviewBundleIntegrityError("Gate-B artifact byte size is invalid")
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise GateBReviewBundleIntegrityError("Gate-B artifact hash is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise GateBReviewBundleIntegrityError(
                f"Gate-B artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != digest:
            raise GateBReviewBundleIntegrityError(f"Gate-B artifact failed integrity: {filename}")
        payloads[filename] = payload
        previous = filename
    if set(payloads) != set(GATE_B_REVIEW_ARTIFACT_FILENAMES):
        raise GateBReviewBundleIntegrityError("Gate-B artifact set is incomplete")
    return payloads


def _validate_paths(inputs: tuple[Path, ...], target: Path) -> None:
    resolved_target = target.resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Gate-B output is not a directory: {target}")
    for input_path in inputs:
        resolved_input = input_path.resolve()
        if resolved_input == resolved_target or resolved_target in resolved_input.parents:
            raise ValueError("Gate-B output must not contain or replace any input path")
    if target.exists():
        allowed = {*GATE_B_REVIEW_ARTIFACT_FILENAMES, GATE_B_REVIEW_MANIFEST_FILENAME}
        unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
        if unexpected:
            raise GateBReviewBundleIntegrityError(
                f"Gate-B output contains an unexpected entry: {unexpected[0]}"
            )


@contextmanager
def _exclusive_writer(target: Path) -> Iterator[None]:
    resolved = target.resolve()
    lock_path = resolved.parent / f".{resolved.name or 'gate-b-review'}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise GateBReviewBundleBusyError(
            f"Gate-B review packet is already being written: {resolved}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
