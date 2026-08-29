"""Strict loading and runtime binding for transparency calibration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .transparency import TransparencyCalibration

TRANSPARENCY_CALIBRATION_SCHEMA: Literal["shopping-copilot/transparency-calibration/v1"] = (
    "shopping-copilot/transparency-calibration/v1"
)
BOUND_PROBE_POLICY_ID = "fixed_multiview_probe_v1"

_CONTENT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "approved",
        "catalog_id",
        "catalog_semantic_release_id",
        "dense_index_id",
        "evaluation_report",
        "high_anchor",
        "low_anchor",
        "mode_similarity_threshold",
        "policy_id",
        "probe_k",
        "probe_policy_id",
        "prompt_suite_id",
        "schema",
    }
)


class TransparencyCalibrationConfigError(ValueError):
    """The calibration document or its runtime binding is invalid."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BoundTransparencyCalibration:
    """Approved calibration bound to the exact Probe and catalog artifacts."""

    schema: Literal["shopping-copilot/transparency-calibration/v1"]
    policy_id: str
    probe_policy_id: str
    probe_k: int
    mode_similarity_threshold: float
    catalog_id: str
    catalog_semantic_release_id: str
    dense_index_id: str
    prompt_suite_id: str
    evaluation_report: str
    low_anchor: float
    high_anchor: float
    approved: bool

    def __post_init__(self) -> None:
        if self.schema != TRANSPARENCY_CALIBRATION_SCHEMA:
            raise TransparencyCalibrationConfigError("calibration schema is invalid")
        if self.probe_policy_id != BOUND_PROBE_POLICY_ID:
            raise TransparencyCalibrationConfigError("calibration Probe policy is invalid")
        if type(self.probe_k) is not int or self.probe_k <= 0:
            raise TransparencyCalibrationConfigError("probe_k must be a positive integer")

        threshold = _finite_number(
            self.mode_similarity_threshold,
            name="mode_similarity_threshold",
        )
        if not 0.0 <= threshold <= 1.0:
            raise TransparencyCalibrationConfigError("mode_similarity_threshold must lie in [0, 1]")
        object.__setattr__(self, "mode_similarity_threshold", threshold)

        for name in (
            "catalog_id",
            "catalog_semantic_release_id",
            "dense_index_id",
            "prompt_suite_id",
        ):
            _require_content_id(getattr(self, name), name=name)
        if (
            type(self.evaluation_report) is not str
            or not self.evaluation_report
            or self.evaluation_report != self.evaluation_report.strip()
        ):
            raise TransparencyCalibrationConfigError(
                "evaluation_report must be a non-empty trimmed string"
            )

        low = _finite_number(self.low_anchor, name="low_anchor")
        high = _finite_number(self.high_anchor, name="high_anchor")
        if type(self.approved) is not bool:
            raise TransparencyCalibrationConfigError("approved must be a boolean")
        try:
            calibration = TransparencyCalibration(
                policy_id=self.policy_id,
                low_anchor=low,
                high_anchor=high,
                approved=self.approved,
            )
        except (TypeError, ValueError) as error:
            raise TransparencyCalibrationConfigError(str(error)) from error
        object.__setattr__(self, "low_anchor", calibration.low_anchor)
        object.__setattr__(self, "high_anchor", calibration.high_anchor)

    @property
    def calibration(self) -> TransparencyCalibration:
        """Return the estimator-facing calibration without its artifact bindings."""

        return TransparencyCalibration(
            policy_id=self.policy_id,
            low_anchor=self.low_anchor,
            high_anchor=self.high_anchor,
            approved=self.approved,
        )

    def validate_runtime(
        self,
        *,
        catalog_id: str,
        release_id: str,
        dense_index_id: str,
        probe_k: int,
        mode_threshold: float,
    ) -> None:
        """Fail closed unless runtime artifacts and Probe knobs match this binding."""

        _require_content_id(catalog_id, name="runtime catalog_id")
        _require_content_id(release_id, name="runtime release_id")
        _require_content_id(dense_index_id, name="runtime dense_index_id")
        if type(probe_k) is not int or probe_k <= 0:
            raise TransparencyCalibrationConfigError("runtime probe_k must be a positive integer")
        observed_threshold = _finite_number(mode_threshold, name="runtime mode_threshold")
        if not 0.0 <= observed_threshold <= 1.0:
            raise TransparencyCalibrationConfigError("runtime mode_threshold must lie in [0, 1]")

        observed = (
            catalog_id,
            release_id,
            dense_index_id,
            probe_k,
            observed_threshold,
        )
        expected = (
            self.catalog_id,
            self.catalog_semantic_release_id,
            self.dense_index_id,
            self.probe_k,
            self.mode_similarity_threshold,
        )
        names = (
            "catalog_id",
            "catalog_semantic_release_id",
            "dense_index_id",
            "probe_k",
            "mode_similarity_threshold",
        )
        for name, actual, required in zip(names, observed, expected, strict=True):
            if actual != required:
                raise TransparencyCalibrationConfigError(
                    f"runtime {name} differs from the calibration binding"
                )


def load_bound_transparency_calibration(
    path: str | Path,
) -> BoundTransparencyCalibration:
    """Load an exact-field JSON calibration document and validate every binding."""

    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise TransparencyCalibrationConfigError("calibration document is unavailable") from error
    try:
        decoded: object = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise TransparencyCalibrationConfigError(
            "calibration document is not valid JSON"
        ) from error
    if type(decoded) is not dict:
        raise TransparencyCalibrationConfigError("calibration document must be an object")
    fields = cast(dict[str, object], decoded)
    if set(fields) != _FIELDS:
        raise TransparencyCalibrationConfigError("calibration document has invalid fields")
    try:
        return BoundTransparencyCalibration(
            schema=cast(CalibrationSchema, fields["schema"]),
            policy_id=cast(str, fields["policy_id"]),
            probe_policy_id=cast(str, fields["probe_policy_id"]),
            probe_k=cast(int, fields["probe_k"]),
            mode_similarity_threshold=cast(float, fields["mode_similarity_threshold"]),
            catalog_id=cast(str, fields["catalog_id"]),
            catalog_semantic_release_id=cast(str, fields["catalog_semantic_release_id"]),
            dense_index_id=cast(str, fields["dense_index_id"]),
            prompt_suite_id=cast(str, fields["prompt_suite_id"]),
            evaluation_report=cast(str, fields["evaluation_report"]),
            low_anchor=cast(float, fields["low_anchor"]),
            high_anchor=cast(float, fields["high_anchor"]),
            approved=cast(bool, fields["approved"]),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, TransparencyCalibrationConfigError):
            raise
        raise TransparencyCalibrationConfigError(str(error)) from error


class _DuplicateJsonKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite(raw: str) -> object:
    raise ValueError(f"non-finite number: {raw}")


def _require_content_id(value: object, *, name: str) -> str:
    if type(value) is not str or _CONTENT_ID_PATTERN.fullmatch(value) is None:
        raise TransparencyCalibrationConfigError(f"{name} must be a full sha256 content ID")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise TransparencyCalibrationConfigError(f"{name} must be a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise TransparencyCalibrationConfigError(f"{name} must be a finite number")
    return result


CalibrationSchema = Literal["shopping-copilot/transparency-calibration/v1"]
