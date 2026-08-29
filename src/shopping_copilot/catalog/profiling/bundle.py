"""Filesystem bundle writer for one raw catalog profiling run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import CatalogProfile, ProfileConfig
from .profiler import canonical_json_dumps, profile_catalog
from .reporting import (
    CanonicalAssignmentJsonlSink,
    catalog_profile_to_markdown,
    write_catalog_profile_json,
    write_category_detail_coverage_jsonl,
    write_category_nodes_jsonl,
    write_detail_keys_jsonl,
)

PROFILE_ARTIFACT_FILENAMES = (
    "profile.json",
    "report.md",
    "category-nodes.jsonl",
    "detail-keys.jsonl",
    "category-detail-coverage.jsonl",
    "product-category-assignments.jsonl",
)
PROFILE_BUNDLE_MANIFEST_FILENAME = "bundle-manifest.json"
PROFILE_BUNDLE_FILENAMES = (*PROFILE_ARTIFACT_FILENAMES, PROFILE_BUNDLE_MANIFEST_FILENAME)

_BUNDLE_SCHEMA = "shopping-copilot/raw-catalog-profile-bundle/v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProfileBundleIntegrityError(ValueError):
    """Raised when a generated bundle is incomplete or internally inconsistent."""


class ProfileBundleBusyError(RuntimeError):
    """Raised when another writer owns the bundle publication lock."""


def write_profile_bundle(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    config: ProfileConfig | None = None,
) -> CatalogProfile:
    """Profile a catalog and replace the generated report files on success.

    All files are first written into a sibling temporary directory. A failed
    profiling or rendering pass therefore leaves an existing output bundle
    untouched. Artifacts are replaced before an integrity manifest is
    published last; readers must call :func:`validate_profile_bundle` and fail
    closed if a process or machine failure interrupted publication. The source
    catalog is only ever opened for binary reading.
    """

    source = Path(catalog_path)
    target = Path(output_dir)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"profile output is not a directory: {target}")
    resolved_source = source.resolve()
    generated_targets = tuple(
        (target / filename).resolve() for filename in PROFILE_BUNDLE_FILENAMES
    )
    if resolved_source in generated_targets:
        raise ValueError("catalog path collides with a generated profile file")

    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_bundle_writer(target):
        return _write_profile_bundle_locked(source, target, config=config)


def _write_profile_bundle_locked(
    source: Path,
    target: Path,
    *,
    config: ProfileConfig | None,
) -> CatalogProfile:
    with TemporaryDirectory(prefix=".catalog-profile-", dir=target.parent) as temporary:
        staging = Path(temporary)
        with (staging / "product-category-assignments.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as assignment_stream:
            profile = profile_catalog(
                source,
                config=config,
                assignment_sink=CanonicalAssignmentJsonlSink(assignment_stream),
            )

        with (staging / "profile.json").open("w", encoding="utf-8", newline="\n") as stream:
            write_catalog_profile_json(profile, stream)
        with (staging / "report.md").open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(catalog_profile_to_markdown(profile))
        with (staging / "category-nodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            write_category_nodes_jsonl(profile, stream)
        with (staging / "detail-keys.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            write_detail_keys_jsonl(profile, stream)
        with (staging / "category-detail-coverage.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            write_category_detail_coverage_jsonl(profile, stream)

        manifest = _build_manifest(staging, profile)
        with (staging / PROFILE_BUNDLE_MANIFEST_FILENAME).open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(canonical_json_dumps(manifest))
            stream.write("\n")

        target.mkdir(parents=True, exist_ok=True)
        for filename in PROFILE_ARTIFACT_FILENAMES:
            os.replace(staging / filename, target / filename)
        os.replace(
            staging / PROFILE_BUNDLE_MANIFEST_FILENAME,
            target / PROFILE_BUNDLE_MANIFEST_FILENAME,
        )
    validate_profile_bundle(target)
    return profile


@contextmanager
def _exclusive_bundle_writer(target: Path) -> Iterator[None]:
    resolved_target = target.resolve()
    target_name = resolved_target.name or "catalog-profile"
    lock_path = resolved_target.parent / f".{target_name}.write.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ProfileBundleBusyError(
            f"profile bundle is already being written: {resolved_target}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def validate_profile_bundle(output_dir: str | Path) -> None:
    """Verify the canonical manifest and every generated artifact hash and size."""

    target = Path(output_dir)
    manifest_path = target / PROFILE_BUNDLE_MANIFEST_FILENAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ProfileBundleIntegrityError("profile bundle manifest is unavailable") from error
    try:
        manifest: object = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ProfileBundleIntegrityError("profile bundle manifest is not valid JSON") from error
    try:
        expected_manifest_bytes = (canonical_json_dumps(manifest) + "\n").encode("utf-8")
    except (RecursionError, UnicodeEncodeError, ValueError) as error:
        raise ProfileBundleIntegrityError("profile bundle manifest is not canonical") from error
    if manifest_bytes != expected_manifest_bytes or type(manifest) is not dict:
        raise ProfileBundleIntegrityError("profile bundle manifest is not canonical")

    fields = manifest
    if set(fields) != {"schema", "catalog_sha256", "profile_schema_version", "artifacts"}:
        raise ProfileBundleIntegrityError("profile bundle manifest has invalid fields")
    if fields["schema"] != _BUNDLE_SCHEMA:
        raise ProfileBundleIntegrityError("profile bundle manifest has an unknown schema")
    catalog_sha256 = fields["catalog_sha256"]
    profile_schema_version = fields["profile_schema_version"]
    artifacts = fields["artifacts"]
    if type(catalog_sha256) is not str or _SHA256_PATTERN.fullmatch(catalog_sha256) is None:
        raise ProfileBundleIntegrityError("profile bundle catalog hash is invalid")
    if type(profile_schema_version) is not str or not profile_schema_version:
        raise ProfileBundleIntegrityError("profile bundle profile schema is invalid")
    if type(artifacts) is not list:
        raise ProfileBundleIntegrityError("profile bundle artifacts must be an array")

    expected_names = set(PROFILE_ARTIFACT_FILENAMES)
    observed_names: set[str] = set()
    previous_name: str | None = None
    for raw_artifact in artifacts:
        if type(raw_artifact) is not dict or set(raw_artifact) != {
            "byte_size",
            "filename",
            "sha256",
        }:
            raise ProfileBundleIntegrityError("profile bundle artifact entry is invalid")
        filename = raw_artifact["filename"]
        byte_size = raw_artifact["byte_size"]
        sha256 = raw_artifact["sha256"]
        if type(filename) is not str or filename not in expected_names:
            raise ProfileBundleIntegrityError("profile bundle artifact filename is invalid")
        if filename in observed_names or (previous_name is not None and filename <= previous_name):
            raise ProfileBundleIntegrityError("profile bundle artifact order is invalid")
        if type(byte_size) is not int or byte_size < 0:
            raise ProfileBundleIntegrityError("profile bundle artifact size is invalid")
        if type(sha256) is not str or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ProfileBundleIntegrityError("profile bundle artifact hash is invalid")

        artifact_path = target / filename
        try:
            actual_size = artifact_path.stat().st_size
            actual_hash = _file_sha256(artifact_path)
        except OSError as error:
            raise ProfileBundleIntegrityError(
                f"profile artifact is unavailable: {filename}"
            ) from error
        if actual_size != byte_size or actual_hash != sha256:
            raise ProfileBundleIntegrityError(
                f"profile artifact failed integrity check: {filename}"
            )
        observed_names.add(filename)
        previous_name = filename
    if observed_names != expected_names:
        raise ProfileBundleIntegrityError("profile bundle artifact set is incomplete")

    try:
        profile_document = json.loads((target / "profile.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ProfileBundleIntegrityError("profile document cannot be decoded") from error
    if type(profile_document) is not dict:
        raise ProfileBundleIntegrityError("profile document must be an object")
    if profile_document.get("catalog_sha256") != catalog_sha256:
        raise ProfileBundleIntegrityError("profile and bundle catalog hashes differ")
    if profile_document.get("schema_version") != profile_schema_version:
        raise ProfileBundleIntegrityError("profile and bundle schema versions differ")


def _build_manifest(staging: Path, profile: CatalogProfile) -> dict[str, object]:
    artifacts = []
    for filename in sorted(PROFILE_ARTIFACT_FILENAMES):
        path = staging / filename
        artifacts.append(
            {
                "byte_size": path.stat().st_size,
                "filename": filename,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema": _BUNDLE_SCHEMA,
        "catalog_sha256": profile.catalog_sha256,
        "profile_schema_version": profile.schema_version,
        "artifacts": artifacts,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
