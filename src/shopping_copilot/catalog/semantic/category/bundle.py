"""Atomic candidate-bundle publication and independent integrity validation."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ..canonical import canonical_json_bytes, sha256_hex
from ..errors import CategoryBuildError, CategoryBundleBusyError, CategoryBundleIntegrityError
from ..raw_catalog import OFFICIAL_PRODUCT_COUNT, RawCatalogScan, scan_raw_catalog
from .build import (
    build_category_candidate,
    build_category_graph_proposal,
    category_scope_selection_template,
    reviewed_category_scopes_candidate_document,
)
from .codec import (
    canonical_json_lines,
    category_graph_proposal_document,
    decode_category_registry,
    decode_graph_proposal_document,
    decode_product_category_assignment_set,
    decode_scope_selection,
    encode_category_registry,
    encode_product_category_assignment_set,
)
from .models import (
    CATEGORY_GRAPH_PROPOSAL_SCHEMA,
    CategoryCandidateBuild,
    CategoryGraphProposal,
    CategoryNode,
    CategoryNormalizationCollision,
    RawPathMapping,
)
from .normalization import (
    CATEGORY_BUILDER_VERSION,
    CATEGORY_UNICODE_DATA_VERSION,
    category_node_id,
    normalize_category_path,
)
from .reporting import category_candidate_markdown, category_graph_proposal_markdown
from .validation import (
    category_graph_id,
    validate_official_p0_assignments,
    validate_product_category_assignment_set,
)

CATEGORY_BUNDLE_MANIFEST_FILENAME = "bundle-manifest.json"
CATEGORY_BUNDLE_SCHEMA = "shopping-copilot/category-build-bundle/v0"
COLLISION_REPORT_SCHEMA = "shopping-copilot/category-normalization-collision-report/v0"

PROPOSAL_ARTIFACT_FILENAMES = (
    "category-graph-proposal.json",
    "collision-report.json",
    "raw-path-mapping.jsonl",
    "report.md",
    "category-scope-selection.template.json",
)
CANDIDATE_ARTIFACT_FILENAMES = (
    "category-registry.json",
    "product-category-assignment.json",
    "report.md",
    "reviewed-category-scopes.candidate.json",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def write_category_graph_proposal_bundle(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
) -> CategoryGraphProposal:
    """Strictly scan raw catalog and atomically publish Pass-A review outputs."""

    source = Path(catalog_path)
    target = Path(output_dir)
    _validate_output_target(source, target, PROPOSAL_ARTIFACT_FILENAMES)
    _validate_no_stale_bundle_artifacts(target, PROPOSAL_ARTIFACT_FILENAMES)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_bundle_writer(target):
        scan = scan_raw_catalog(
            source,
            expected_product_count=expected_product_count,
        )
        proposal = build_category_graph_proposal(scan)
        payloads = _proposal_payloads(proposal)
        _publish_bundle(
            target,
            kind="proposal",
            catalog_id=proposal.catalog_id,
            category_graph_id=proposal.category_graph_id,
            builder_version=proposal.builder_version,
            expected_product_count=expected_product_count,
            official_p0_assignment_gate=False,
            payloads=payloads,
        )
    validate_category_bundle(target)
    return proposal


def write_category_candidate_bundle(
    catalog_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    expected_product_count: int = OFFICIAL_PRODUCT_COUNT,
    enforce_official_gate: bool = True,
) -> CategoryCandidateBuild:
    """Rebuild exact graph and atomically publish Pass-B category candidates."""

    source = Path(catalog_path)
    selection_source = Path(selection_path)
    target = Path(output_dir)
    _validate_output_target(source, target, CANDIDATE_ARTIFACT_FILENAMES)
    _validate_output_target(selection_source, target, CANDIDATE_ARTIFACT_FILENAMES)
    _validate_no_stale_bundle_artifacts(target, CANDIDATE_ARTIFACT_FILENAMES)
    if enforce_official_gate and expected_product_count != OFFICIAL_PRODUCT_COUNT:
        raise ValueError(
            f"official P0 assignment gate requires expected_product_count={OFFICIAL_PRODUCT_COUNT}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_bundle_writer(target):
        selection = decode_scope_selection(selection_source.read_bytes())
        scan = scan_raw_catalog(
            source,
            expected_product_count=expected_product_count,
        )
        candidate = build_category_candidate(
            scan,
            selection,
            enforce_official_gate=enforce_official_gate,
        )
        payloads = _candidate_payloads(candidate)
        _publish_bundle(
            target,
            kind="candidate",
            catalog_id=candidate.registry.catalog_id,
            category_graph_id=candidate.registry.category_graph_id,
            builder_version=candidate.builder_version,
            expected_product_count=expected_product_count,
            official_p0_assignment_gate=enforce_official_gate,
            payloads=payloads,
        )
    validate_category_bundle(target, catalog_path=source)
    return candidate


def validate_category_bundle(
    output_dir: str | Path,
    *,
    catalog_path: str | Path | None = None,
) -> None:
    """Reload a bundle; candidate validation rebinds to ``catalog_path`` truth."""

    target = Path(output_dir)
    manifest_path = target / CATEGORY_BUNDLE_MANIFEST_FILENAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise CategoryBundleIntegrityError("category bundle manifest is unavailable") from error
    manifest = _decode_manifest(manifest_bytes)
    kind = cast(str, manifest["kind"])
    filenames = PROPOSAL_ARTIFACT_FILENAMES if kind == "proposal" else CANDIDATE_ARTIFACT_FILENAMES
    payloads = _validate_manifest_artifacts(target, manifest, filenames)
    if kind == "proposal":
        _validate_proposal_payloads(manifest, payloads)
    else:
        if catalog_path is None:
            raise CategoryBundleIntegrityError(
                "candidate bundle validation requires the exact raw catalog"
            )
        scan = scan_raw_catalog(
            catalog_path,
            expected_product_count=cast(int, manifest["expected_product_count"]),
        )
        if scan.catalog_id != manifest["catalog_id"]:
            raise CategoryBundleIntegrityError("candidate raw catalog ID differs from manifest")
        _validate_candidate_payloads(manifest, payloads, scan=scan)


def _proposal_payloads(proposal: CategoryGraphProposal) -> dict[str, bytes]:
    collision_document = {
        "schema": COLLISION_REPORT_SCHEMA,
        "catalog_id": proposal.catalog_id,
        "category_graph_id": proposal.category_graph_id,
        "collisions": list(proposal.collisions),
    }
    return {
        "category-graph-proposal.json": canonical_json_bytes(
            category_graph_proposal_document(proposal)
        ),
        "collision-report.json": canonical_json_bytes(collision_document),
        "raw-path-mapping.jsonl": canonical_json_lines(proposal.raw_path_mappings),
        "report.md": category_graph_proposal_markdown(proposal).encode("utf-8"),
        "category-scope-selection.template.json": canonical_json_bytes(
            category_scope_selection_template(proposal)
        ),
    }


def _candidate_payloads(candidate: CategoryCandidateBuild) -> dict[str, bytes]:
    return {
        "category-registry.json": encode_category_registry(candidate.registry),
        "product-category-assignment.json": encode_product_category_assignment_set(
            candidate.assignments,
            registry=candidate.registry,
        ),
        "report.md": category_candidate_markdown(candidate).encode("utf-8"),
        "reviewed-category-scopes.candidate.json": canonical_json_bytes(
            reviewed_category_scopes_candidate_document(candidate)
        ),
    }


def _publish_bundle(
    target: Path,
    *,
    kind: str,
    catalog_id: str,
    category_graph_id: str,
    builder_version: str,
    expected_product_count: int,
    official_p0_assignment_gate: bool,
    payloads: dict[str, bytes],
) -> None:
    with TemporaryDirectory(prefix=".catalog-category-", dir=target.parent) as temporary:
        staging = Path(temporary)
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
        artifacts = [
            {
                "byte_size": len(payloads[filename]),
                "filename": filename,
                "sha256": sha256_hex(payloads[filename]),
            }
            for filename in sorted(payloads)
        ]
        manifest = {
            "schema": CATEGORY_BUNDLE_SCHEMA,
            "kind": kind,
            "catalog_id": catalog_id,
            "category_graph_id": category_graph_id,
            "builder_version": builder_version,
            "expected_product_count": expected_product_count,
            "official_p0_assignment_gate": official_p0_assignment_gate,
            "artifacts": artifacts,
        }
        (staging / CATEGORY_BUNDLE_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))

        target.mkdir(parents=True, exist_ok=True)
        for filename in sorted(payloads):
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / CATEGORY_BUNDLE_MANIFEST_FILENAME,
            target / CATEGORY_BUNDLE_MANIFEST_FILENAME,
        )


@contextmanager
def _exclusive_bundle_writer(target: Path) -> Iterator[None]:
    resolved_target = target.resolve()
    target_name = resolved_target.name or "catalog-category"
    lock_path = resolved_target.parent / f".{target_name}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise CategoryBundleBusyError(
            f"category bundle is already being written: {resolved_target}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _validate_output_target(
    source: Path,
    target: Path,
    artifact_filenames: tuple[str, ...],
) -> None:
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"category output is not a directory: {target}")
    resolved_source = source.resolve()
    resolved_target = target.resolve()
    if resolved_source == resolved_target or resolved_target in resolved_source.parents:
        raise ValueError("category inputs must not live inside the generated bundle directory")
    generated = {
        (resolved_target / filename).resolve()
        for filename in (*artifact_filenames, CATEGORY_BUNDLE_MANIFEST_FILENAME)
    }
    if resolved_source in generated:
        raise ValueError("input path collides with a generated category file")


def _validate_no_stale_bundle_artifacts(
    target: Path,
    artifact_filenames: tuple[str, ...],
) -> None:
    if not target.exists():
        return
    allowed = {*artifact_filenames, CATEGORY_BUNDLE_MANIFEST_FILENAME}
    known = {
        *PROPOSAL_ARTIFACT_FILENAMES,
        *CANDIDATE_ARTIFACT_FILENAMES,
        CATEGORY_BUNDLE_MANIFEST_FILENAME,
    }
    stale = sorted(path.name for path in target.iterdir() if path.name in known - allowed)
    if stale:
        raise ValueError(f"output contains stale category bundle artifacts: {stale}")


def _decode_manifest(data: bytes) -> dict[str, object]:
    document = _load_canonical_json(data, name="category bundle manifest")
    if type(document) is not dict:
        raise CategoryBundleIntegrityError("category bundle manifest must be an object")
    manifest = cast(dict[str, object], document)
    expected_fields = {
        "schema",
        "kind",
        "catalog_id",
        "category_graph_id",
        "builder_version",
        "expected_product_count",
        "official_p0_assignment_gate",
        "artifacts",
    }
    if set(manifest) != expected_fields:
        raise CategoryBundleIntegrityError("category bundle manifest fields are invalid")
    if manifest["schema"] != CATEGORY_BUNDLE_SCHEMA:
        raise CategoryBundleIntegrityError("category bundle manifest schema is invalid")
    if manifest["kind"] not in ("proposal", "candidate"):
        raise CategoryBundleIntegrityError("category bundle kind is invalid")
    if (
        type(manifest["catalog_id"]) is not str
        or _CONTENT_ID_PATTERN.fullmatch(manifest["catalog_id"]) is None
    ):
        raise CategoryBundleIntegrityError("category bundle catalog_id is invalid")
    if (
        type(manifest["category_graph_id"]) is not str
        or _CONTENT_ID_PATTERN.fullmatch(manifest["category_graph_id"]) is None
    ):
        raise CategoryBundleIntegrityError("category bundle graph ID is invalid")
    if manifest["builder_version"] != CATEGORY_BUILDER_VERSION:
        raise CategoryBundleIntegrityError("category bundle builder version is unsupported")
    count = manifest["expected_product_count"]
    if type(count) is not int or count <= 0:
        raise CategoryBundleIntegrityError("category bundle product count is invalid")
    official_gate = manifest["official_p0_assignment_gate"]
    if type(official_gate) is not bool:
        raise CategoryBundleIntegrityError("category bundle official gate flag is invalid")
    if official_gate and (manifest["kind"] != "candidate" or count != OFFICIAL_PRODUCT_COUNT):
        raise CategoryBundleIntegrityError("category bundle official gate declaration is invalid")
    if type(manifest["artifacts"]) is not list:
        raise CategoryBundleIntegrityError("category bundle artifacts must be an array")
    return manifest


def _validate_manifest_artifacts(
    target: Path,
    manifest: dict[str, object],
    expected_filenames: tuple[str, ...],
) -> dict[str, bytes]:
    expected = set(expected_filenames)
    previous_filename: str | None = None
    observed: dict[str, bytes] = {}
    for raw_entry in cast(list[object], manifest["artifacts"]):
        if type(raw_entry) is not dict:
            raise CategoryBundleIntegrityError("category artifact entry must be an object")
        entry = cast(dict[str, object], raw_entry)
        if set(entry) != {"byte_size", "filename", "sha256"}:
            raise CategoryBundleIntegrityError("category artifact entry fields are invalid")
        filename = entry["filename"]
        byte_size = entry["byte_size"]
        digest = entry["sha256"]
        if type(filename) is not str or filename not in expected:
            raise CategoryBundleIntegrityError("category artifact filename is invalid")
        if previous_filename is not None and filename <= previous_filename:
            raise CategoryBundleIntegrityError("category artifact entries are not sorted")
        if type(byte_size) is not int or byte_size < 0:
            raise CategoryBundleIntegrityError("category artifact byte size is invalid")
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise CategoryBundleIntegrityError("category artifact digest is invalid")
        try:
            payload = (target / filename).read_bytes()
        except OSError as error:
            raise CategoryBundleIntegrityError(
                f"category artifact is unavailable: {filename}"
            ) from error
        if len(payload) != byte_size or sha256_hex(payload) != digest:
            raise CategoryBundleIntegrityError(
                f"category artifact failed integrity check: {filename}"
            )
        observed[filename] = payload
        previous_filename = filename
    if set(observed) != expected:
        raise CategoryBundleIntegrityError("category bundle artifact set is incomplete")
    return observed


def _validate_proposal_payloads(
    manifest: dict[str, object],
    payloads: dict[str, bytes],
) -> None:
    graph_document = decode_graph_proposal_document(payloads["category-graph-proposal.json"])
    nodes = graph_document["nodes"]
    if type(nodes) is not tuple:
        raise CategoryBundleIntegrityError("decoded graph nodes are invalid")
    if graph_document["catalog_id"] != manifest["catalog_id"]:
        raise CategoryBundleIntegrityError("proposal catalog ID differs from manifest")
    if graph_document["category_graph_id"] != manifest["category_graph_id"]:
        raise CategoryBundleIntegrityError("proposal graph ID differs from manifest")
    if graph_document["builder_version"] != manifest["builder_version"]:
        raise CategoryBundleIntegrityError("proposal builder version differs from manifest")
    if graph_document["unicode_data_version"] != CATEGORY_UNICODE_DATA_VERSION:
        raise CategoryBundleIntegrityError("proposal Unicode data version is unsupported")
    if category_graph_id(cast(str, manifest["catalog_id"]), nodes) != manifest["category_graph_id"]:
        raise CategoryBundleIntegrityError("proposal graph ID does not match its nodes")

    mappings = _decode_mapping_jsonl(payloads["raw-path-mapping.jsonl"])
    _validate_mapping_graph_integrity(mappings, nodes)
    collisions = _derive_collisions(mappings)
    proposal = CategoryGraphProposal(
        schema=CATEGORY_GRAPH_PROPOSAL_SCHEMA,
        catalog_id=cast(str, graph_document["catalog_id"]),
        category_graph_id=cast(str, graph_document["category_graph_id"]),
        builder_version=cast(str, graph_document["builder_version"]),
        unicode_data_version=graph_document["unicode_data_version"],
        catalog_byte_size=cast(int, graph_document["catalog_byte_size"]),
        product_count=cast(int, graph_document["product_count"]),
        raw_prefix_count=cast(int, graph_document["raw_prefix_count"]),
        nodes=nodes,
        raw_path_mappings=mappings,
        collisions=collisions,
    )
    if proposal.product_count != manifest["expected_product_count"]:
        raise CategoryBundleIntegrityError("proposal product count differs from manifest")
    if proposal.raw_prefix_count != len(mappings):
        raise CategoryBundleIntegrityError("proposal raw prefix count is invalid")
    if graph_document["canonical_node_count"] != len(nodes):
        raise CategoryBundleIntegrityError("proposal canonical node count is invalid")
    if graph_document["collision_count"] != len(collisions):
        raise CategoryBundleIntegrityError("proposal collision count is invalid")
    if sum(mapping.direct_product_count for mapping in mappings) != proposal.product_count:
        raise CategoryBundleIntegrityError("proposal direct support does not sum to products")
    root_node_ids = {node.id for node in nodes if node.parent_id is None}
    if (
        sum(
            mapping.subtree_product_count
            for mapping in mappings
            if mapping.node_id in root_node_ids
        )
        != proposal.product_count
    ):
        raise CategoryBundleIntegrityError("proposal root support does not sum to products")

    expected_collision_bytes = canonical_json_bytes(
        {
            "schema": COLLISION_REPORT_SCHEMA,
            "catalog_id": proposal.catalog_id,
            "category_graph_id": proposal.category_graph_id,
            "collisions": list(collisions),
        }
    )
    if payloads["collision-report.json"] != expected_collision_bytes:
        raise CategoryBundleIntegrityError("collision report does not match raw mappings")
    if payloads["category-scope-selection.template.json"] != canonical_json_bytes(
        category_scope_selection_template(proposal)
    ):
        raise CategoryBundleIntegrityError("scope selection template is inconsistent")
    if payloads["report.md"] != category_graph_proposal_markdown(proposal).encode("utf-8"):
        raise CategoryBundleIntegrityError("proposal report is inconsistent")


def _validate_candidate_payloads(
    manifest: dict[str, object],
    payloads: dict[str, bytes],
    *,
    scan: RawCatalogScan,
) -> None:
    registry = decode_category_registry(payloads["category-registry.json"])
    proposal = build_category_graph_proposal(scan)
    assignments = decode_product_category_assignment_set(
        payloads["product-category-assignment.json"],
        registry=registry,
    )
    candidate = CategoryCandidateBuild(
        builder_version=cast(str, manifest["builder_version"]),
        registry=registry,
        assignments=assignments,
    )
    if registry.catalog_id != manifest["catalog_id"]:
        raise CategoryBundleIntegrityError("candidate catalog ID differs from manifest")
    if registry.category_graph_id != manifest["category_graph_id"]:
        raise CategoryBundleIntegrityError("candidate graph ID differs from manifest")
    if (
        candidate.builder_version != proposal.builder_version
        or registry.catalog_id != proposal.catalog_id
        or registry.category_graph_id != proposal.category_graph_id
        or registry.nodes != proposal.nodes
    ):
        raise CategoryBundleIntegrityError(
            "candidate registry graph differs from exact raw catalog truth"
        )
    terminal_node_ids_by_product = {
        record.parent_asin: {category_node_id(normalize_category_path(record.raw_path))}
        for record in scan.records
    }
    try:
        validate_product_category_assignment_set(
            assignments,
            registry=registry,
            expected_product_ids=set(terminal_node_ids_by_product),
            terminal_node_ids_by_product=terminal_node_ids_by_product,
        )
    except CategoryBuildError as error:
        raise CategoryBundleIntegrityError(
            "candidate assignments differ from exact raw catalog truth"
        ) from error
    expected_product_count = cast(int, manifest["expected_product_count"])
    if len(assignments.assignments) != expected_product_count:
        raise CategoryBundleIntegrityError("candidate assignment count differs from manifest")
    if manifest["official_p0_assignment_gate"] is True:
        validate_official_p0_assignments(assignments)
    if payloads["reviewed-category-scopes.candidate.json"] != canonical_json_bytes(
        reviewed_category_scopes_candidate_document(candidate)
    ):
        raise CategoryBundleIntegrityError("reviewed scope fragment is inconsistent")
    if payloads["report.md"] != category_candidate_markdown(candidate).encode("utf-8"):
        raise CategoryBundleIntegrityError("candidate report is inconsistent")


def _decode_mapping_jsonl(data: bytes) -> tuple[RawPathMapping, ...]:
    mappings: list[RawPathMapping] = []
    if not data:
        raise CategoryBundleIntegrityError("raw path mapping JSONL is empty")
    for raw_line in data.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            raise CategoryBundleIntegrityError("raw path mapping line lacks LF terminator")
        payload = raw_line[:-1]
        document = _load_canonical_json(payload, name="raw path mapping row")
        if type(document) is not dict:
            raise CategoryBundleIntegrityError("raw path mapping row must be an object")
        row = cast(dict[str, object], document)
        if set(row) != {
            "raw_path",
            "canonical_path",
            "node_id",
            "direct_product_count",
            "subtree_product_count",
        }:
            raise CategoryBundleIntegrityError("raw path mapping fields are invalid")
        raw_path = _string_tuple(row["raw_path"], name="raw_path")
        canonical_path = _string_tuple(row["canonical_path"], name="canonical_path")
        node_id = row["node_id"]
        direct = row["direct_product_count"]
        subtree = row["subtree_product_count"]
        if type(node_id) is not str:
            raise CategoryBundleIntegrityError("raw path mapping node_id is invalid")
        if type(direct) is not int or direct < 0:
            raise CategoryBundleIntegrityError("raw path direct support is invalid")
        if type(subtree) is not int or subtree <= 0 or direct > subtree:
            raise CategoryBundleIntegrityError("raw path subtree support is invalid")
        try:
            mapping = RawPathMapping(
                raw_path=raw_path,
                canonical_path=canonical_path,
                node_id=node_id,
                direct_product_count=direct,
                subtree_product_count=subtree,
            )
        except (TypeError, ValueError) as error:
            raise CategoryBundleIntegrityError(
                f"raw path mapping row is invalid: {error}"
            ) from error
        mappings.append(mapping)
    result = tuple(mappings)
    if result != tuple(sorted(result, key=lambda item: canonical_json_bytes(item.raw_path))):
        raise CategoryBundleIntegrityError("raw path mappings are not canonically ordered")
    if len({item.raw_path for item in result}) != len(result):
        raise CategoryBundleIntegrityError("raw path mappings contain duplicate raw paths")
    return result


def _derive_collisions(
    mappings: tuple[RawPathMapping, ...],
) -> tuple[CategoryNormalizationCollision, ...]:
    raw_paths_by_canonical: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for mapping in mappings:
        raw_paths_by_canonical[mapping.canonical_path].append(mapping.raw_path)
    return tuple(
        CategoryNormalizationCollision(
            canonical_path=canonical_path,
            raw_paths=tuple(sorted(raw_paths, key=canonical_json_bytes)),
        )
        for canonical_path, raw_paths in sorted(
            raw_paths_by_canonical.items(),
            key=lambda item: canonical_json_bytes(item[0]),
        )
        if len(raw_paths) > 1
    )


def _validate_mapping_graph_integrity(
    mappings: tuple[RawPathMapping, ...],
    nodes: tuple[CategoryNode, ...],
) -> None:
    nodes_by_id = {node.id: node for node in nodes}
    for mapping in mappings:
        canonical_path = normalize_category_path(mapping.raw_path)
        if mapping.canonical_path != canonical_path:
            raise CategoryBundleIntegrityError(
                "raw path mapping does not match the closed normalizer"
            )
        if mapping.node_id != category_node_id(canonical_path):
            raise CategoryBundleIntegrityError("raw path mapping node ID is invalid")
        node = nodes_by_id.get(mapping.node_id)
        if node is None or node.canonical_path != canonical_path:
            raise CategoryBundleIntegrityError("raw path mapping does not reference its graph node")
    if {mapping.node_id for mapping in mappings} != set(nodes_by_id):
        raise CategoryBundleIntegrityError("raw path mappings do not cover every graph node")


class _DuplicateJsonKeyError(ValueError):
    pass


def _load_canonical_json(data: bytes, *, name: str) -> object:
    try:
        text = data.decode("utf-8")
        parsed: object = json.loads(
            text,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        expected = canonical_json_bytes(parsed)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise CategoryBundleIntegrityError(f"{name} is invalid JSON") from error
    if data != expected:
        raise CategoryBundleIntegrityError(f"{name} is not canonical JSON")
    return parsed


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_token(raw: str) -> object:
    raise ValueError(f"non-finite number token: {raw}")


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise CategoryBundleIntegrityError(f"{name} must be an array")
    items = cast(list[object], value)
    if any(type(item) is not str for item in items):
        raise CategoryBundleIntegrityError(f"{name} must contain only strings")
    return tuple(cast(str, item) for item in items)
