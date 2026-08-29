from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic import content_id_for_bytes
from shopping_copilot.retrieval.calibration import (
    BOUND_PROBE_POLICY_ID,
    TRANSPARENCY_CALIBRATION_SCHEMA,
    BoundTransparencyCalibration,
    TransparencyCalibrationConfigError,
    load_bound_transparency_calibration,
)
from shopping_copilot.retrieval.transparency import TransparencyCalibration

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

CATALOG_ID = "sha256:" + "1" * 64
RELEASE_ID = "sha256:" + "2" * 64
INDEX_ID = "sha256:" + "3" * 64
SUITE_ID = "sha256:" + "4" * 64


def _document() -> dict[str, object]:
    return {
        "schema": TRANSPARENCY_CALIBRATION_SCHEMA,
        "policy_id": "semantic_mode_linear_v1",
        "probe_policy_id": BOUND_PROBE_POLICY_ID,
        "probe_k": 80,
        "mode_similarity_threshold": 0.94,
        "catalog_id": CATALOG_ID,
        "catalog_semantic_release_id": RELEASE_ID,
        "dense_index_id": INDEX_ID,
        "prompt_suite_id": SUITE_ID,
        "evaluation_report": "artifacts/retrieval/transparency-v1/report.json",
        "low_anchor": 0.25,
        "high_anchor": 0.45,
        "approved": True,
    }


def _write(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load(
    tmp_path: Path, document: dict[str, object] | None = None
) -> BoundTransparencyCalibration:
    return load_bound_transparency_calibration(
        _write(tmp_path / "calibration.json", _document() if document is None else document)
    )


def test_loads_bound_calibration_and_exposes_estimator_view(tmp_path: Path) -> None:
    bound = _load(tmp_path)

    assert bound.schema == TRANSPARENCY_CALIBRATION_SCHEMA
    assert bound.catalog_id == CATALOG_ID
    assert bound.mode_similarity_threshold == pytest.approx(0.94)
    assert bound.calibration == TransparencyCalibration(
        policy_id="semantic_mode_linear_v1",
        low_anchor=0.25,
        high_anchor=0.45,
        approved=True,
    )
    with pytest.raises(FrozenInstanceError):
        bound.probe_k = 40  # type: ignore[misc]

    bound.validate_runtime(
        catalog_id=CATALOG_ID,
        release_id=RELEASE_ID,
        dense_index_id=INDEX_ID,
        probe_k=80,
        mode_threshold=0.94,
    )


def test_frozen_repository_calibration_points_to_source_controlled_evidence() -> None:
    bound = load_bound_transparency_calibration(
        REPOSITORY_ROOT / "config/retrieval/transparency-calibration-v1.json"
    )
    assert bound.approved is True
    assert bound.low_anchor == pytest.approx(0.256963026520931)
    assert bound.high_anchor == pytest.approx(0.4483984914520624)
    assert (REPOSITORY_ROOT / bound.evaluation_report).is_file()
    assert bound.prompt_suite_id == content_id_for_bytes(
        (REPOSITORY_ROOT / "config/retrieval/transparency-prompts-v1.json").read_bytes()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "shopping-copilot/transparency-calibration/v0"),
        ("probe_policy_id", "adaptive_probe_v1"),
        ("probe_k", 0),
        ("probe_k", True),
        ("mode_similarity_threshold", 1.01),
        ("mode_similarity_threshold", float("inf")),
        ("catalog_id", "sha256:not-a-digest"),
        ("catalog_semantic_release_id", "2" * 64),
        ("dense_index_id", "sha256:" + "A" * 64),
        ("prompt_suite_id", ""),
        ("evaluation_report", ""),
        ("low_anchor", 0.5),
        ("high_anchor", float("nan")),
        ("approved", 1),
    ],
)
def test_rejects_tampered_or_invalid_binding(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _document()
    document[field] = value

    with pytest.raises(TransparencyCalibrationConfigError):
        _load(tmp_path, document)


@pytest.mark.parametrize("field", ["approved", "catalog_id"])
def test_rejects_missing_and_extra_fields(tmp_path: Path, field: str) -> None:
    missing = _document()
    del missing[field]
    with pytest.raises(TransparencyCalibrationConfigError, match="invalid fields"):
        _load(tmp_path, missing)

    extra = _document()
    extra["unexpected"] = "value"
    with pytest.raises(TransparencyCalibrationConfigError, match="invalid fields"):
        _load(tmp_path, extra)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    payload = json.dumps(_document())
    duplicate = payload[:-1] + ', "probe_k": 40}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(TransparencyCalibrationConfigError, match="not valid JSON"):
        load_bound_transparency_calibration(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("catalog_id", "sha256:" + "a" * 64, "catalog_id"),
        ("release_id", "sha256:" + "b" * 64, "catalog_semantic_release_id"),
        ("dense_index_id", "sha256:" + "c" * 64, "dense_index_id"),
        ("probe_k", 79, "probe_k"),
        ("mode_threshold", 0.93, "mode_similarity_threshold"),
    ],
)
def test_runtime_binding_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    bound = _load(tmp_path)
    runtime: dict[str, object] = {
        "catalog_id": CATALOG_ID,
        "release_id": RELEASE_ID,
        "dense_index_id": INDEX_ID,
        "probe_k": 80,
        "mode_threshold": 0.94,
    }
    runtime[field] = value

    with pytest.raises(TransparencyCalibrationConfigError, match=message):
        bound.validate_runtime(**runtime)  # type: ignore[arg-type]
