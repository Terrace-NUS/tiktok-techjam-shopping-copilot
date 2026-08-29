"""Pure calibration and diagnostics for catalog-grounded intent transparency.

This module deliberately accepts a small, implementation-independent evidence
DTO.  Probe construction, semantic-mode clustering, and lexical retrieval stay
outside this boundary.  In particular, only equal-mode coherence may affect
the measured transparency value; all other observations are diagnostic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import cast

from shopping_copilot.session_context import (
    CertaintyEvidence,
    ProbeQuality,
    SearchBelief,
)

CONTROLLER_FALLBACK = 0.5
_CANONICAL_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_FLOAT_TOLERANCE = 1e-12


class DiagnosticStatus(str, Enum):
    """Overall trustworthiness of one Probe observation."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class TransparencyEvidence:
    """Mode and route observations consumed by the transparency estimator.

    ``mode_coherence`` is the sole scalar input to calibration.  Counts,
    lexical evidence, route overlap, and warnings exist only so ``D_t`` can
    explain retrieval health.
    """

    probe_id: str
    intent_version: int
    probe_k: int
    eligible_count: int
    dense_hits: tuple[str, ...]
    lexical_hits: tuple[str, ...]
    listing_coherence: float | None
    mode_coherence: float | None
    mode_count: int
    largest_mode_share: float
    effective_mode_count: float
    duplicate_warning: bool
    lexical_available: bool
    lexical_token_coverage: float | None
    lexical_mean_normalized_idf: float | None
    hard_filter_relaxed: bool

    def __post_init__(self) -> None:
        _require_nonempty_trimmed(self.probe_id, name="probe_id")
        _require_nonnegative_int(self.intent_version, name="intent_version")
        _require_positive_int(self.probe_k, name="probe_k")
        _require_nonnegative_int(self.eligible_count, name="eligible_count")
        _require_identifier_tuple(self.dense_hits, name="dense_hits")
        _require_identifier_tuple(self.lexical_hits, name="lexical_hits")

        dense_count = len(self.dense_hits)
        lexical_count = len(self.lexical_hits)
        if dense_count > self.probe_k or lexical_count > self.probe_k:
            raise ValueError("Probe hit counts cannot exceed probe_k")
        if dense_count > self.eligible_count or lexical_count > self.eligible_count:
            raise ValueError("Probe hit counts cannot exceed eligible_count")

        listing = _optional_bounded_float(
            self.listing_coherence,
            name="listing_coherence",
            lower=-1.0,
            upper=1.0,
        )
        modes = _optional_bounded_float(
            self.mode_coherence,
            name="mode_coherence",
            lower=-1.0,
            upper=1.0,
        )
        object.__setattr__(self, "listing_coherence", listing)
        object.__setattr__(self, "mode_coherence", modes)

        _require_nonnegative_int(self.mode_count, name="mode_count")
        if self.mode_count > dense_count:
            raise ValueError("mode_count cannot exceed the dense hit count")
        largest = _bounded_float(
            self.largest_mode_share,
            name="largest_mode_share",
            lower=0.0,
            upper=1.0,
        )
        effective = _nonnegative_float(
            self.effective_mode_count,
            name="effective_mode_count",
        )
        object.__setattr__(self, "largest_mode_share", largest)
        object.__setattr__(self, "effective_mode_count", effective)
        _validate_mode_summary(
            mode_count=self.mode_count,
            largest_mode_share=largest,
            effective_mode_count=effective,
            mode_coherence=modes,
        )

        _require_bool(self.duplicate_warning, name="duplicate_warning")
        _require_bool(self.lexical_available, name="lexical_available")
        token_coverage = _optional_bounded_float(
            self.lexical_token_coverage,
            name="lexical_token_coverage",
            lower=0.0,
            upper=1.0,
        )
        mean_idf = _optional_bounded_float(
            self.lexical_mean_normalized_idf,
            name="lexical_mean_normalized_idf",
            lower=0.0,
            upper=1.0,
        )
        object.__setattr__(self, "lexical_token_coverage", token_coverage)
        object.__setattr__(self, "lexical_mean_normalized_idf", mean_idf)
        if not self.lexical_available and (
            self.lexical_hits or token_coverage is not None or mean_idf is not None
        ):
            raise ValueError("unavailable lexical evidence cannot contain hits or metrics")
        _require_bool(self.hard_filter_relaxed, name="hard_filter_relaxed")


@dataclass(frozen=True, slots=True, kw_only=True)
class TransparencyCalibration:
    """Versioned monotonic calibration anchors for one Probe policy."""

    policy_id: str
    low_anchor: float
    high_anchor: float
    approved: bool = True

    def __post_init__(self) -> None:
        _require_canonical_identifier(self.policy_id, name="policy_id")
        low = _bounded_float(
            self.low_anchor,
            name="low_anchor",
            lower=-1.0,
            upper=1.0,
        )
        high = _bounded_float(
            self.high_anchor,
            name="high_anchor",
            lower=-1.0,
            upper=1.0,
        )
        if not low < high:
            raise ValueError("low_anchor must be strictly below high_anchor")
        _require_bool(self.approved, name="approved")
        object.__setattr__(self, "low_anchor", low)
        object.__setattr__(self, "high_anchor", high)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeDiagnostics:
    """Ephemeral retrieval-health sidecar, also called ``D_t``."""

    probe_id: str
    intent_version: int
    status: DiagnosticStatus
    reason_codes: tuple[str, ...]
    probe_k: int
    eligible_count: int
    dense_count: int
    lexical_count: int
    fill_ratio: float
    lexical_available: bool
    route_overlap_count: int | None
    route_overlap: float | None
    listing_coherence: float | None
    mode_coherence: float | None
    mode_count: int
    largest_mode_share: float
    effective_mode_count: float
    duplicate_warning: bool
    lexical_token_coverage: float | None
    lexical_mean_normalized_idf: float | None
    hard_filter_relaxed: bool

    def __post_init__(self) -> None:
        _require_nonempty_trimmed(self.probe_id, name="probe_id")
        _require_nonnegative_int(self.intent_version, name="intent_version")
        if type(self.status) is not DiagnosticStatus:
            raise TypeError("status must be a DiagnosticStatus")
        _require_reason_codes(self.reason_codes)
        if self.status is DiagnosticStatus.HEALTHY and self.reason_codes:
            raise ValueError("healthy diagnostics cannot contain reason codes")
        if self.status is not DiagnosticStatus.HEALTHY and not self.reason_codes:
            raise ValueError("non-healthy diagnostics require reason codes")

        _require_positive_int(self.probe_k, name="probe_k")
        for name in ("eligible_count", "dense_count", "lexical_count", "mode_count"):
            _require_nonnegative_int(getattr(self, name), name=name)
        if self.dense_count > self.probe_k or self.lexical_count > self.probe_k:
            raise ValueError("diagnostic hit counts cannot exceed probe_k")
        if self.dense_count > self.eligible_count or self.lexical_count > self.eligible_count:
            raise ValueError("diagnostic hit counts cannot exceed eligible_count")
        if self.mode_count > self.dense_count:
            raise ValueError("diagnostic mode_count cannot exceed dense_count")

        fill = _bounded_float(self.fill_ratio, name="fill_ratio", lower=0.0, upper=1.0)
        expected_fill = self.dense_count / self.probe_k
        if not math.isclose(fill, expected_fill, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
            raise ValueError("fill_ratio does not match dense_count / probe_k")
        object.__setattr__(self, "fill_ratio", fill)
        _require_bool(self.lexical_available, name="lexical_available")
        _validate_overlap(self)

        listing = _optional_bounded_float(
            self.listing_coherence,
            name="listing_coherence",
            lower=-1.0,
            upper=1.0,
        )
        modes = _optional_bounded_float(
            self.mode_coherence,
            name="mode_coherence",
            lower=-1.0,
            upper=1.0,
        )
        largest = _bounded_float(
            self.largest_mode_share,
            name="largest_mode_share",
            lower=0.0,
            upper=1.0,
        )
        effective = _nonnegative_float(
            self.effective_mode_count,
            name="effective_mode_count",
        )
        _validate_mode_summary(
            mode_count=self.mode_count,
            largest_mode_share=largest,
            effective_mode_count=effective,
            mode_coherence=modes,
        )
        object.__setattr__(self, "listing_coherence", listing)
        object.__setattr__(self, "mode_coherence", modes)
        object.__setattr__(self, "largest_mode_share", largest)
        object.__setattr__(self, "effective_mode_count", effective)

        _require_bool(self.duplicate_warning, name="duplicate_warning")
        token_coverage = _optional_bounded_float(
            self.lexical_token_coverage,
            name="lexical_token_coverage",
            lower=0.0,
            upper=1.0,
        )
        mean_idf = _optional_bounded_float(
            self.lexical_mean_normalized_idf,
            name="lexical_mean_normalized_idf",
            lower=0.0,
            upper=1.0,
        )
        object.__setattr__(self, "lexical_token_coverage", token_coverage)
        object.__setattr__(self, "lexical_mean_normalized_idf", mean_idf)
        if not self.lexical_available and (
            self.lexical_count or token_coverage is not None or mean_idf is not None
        ):
            raise ValueError("unavailable lexical diagnostics cannot contain route evidence")
        _require_bool(self.hard_filter_relaxed, name="hard_filter_relaxed")


@dataclass(frozen=True, slots=True, kw_only=True)
class TransparencyEstimate:
    """Measured transparency plus a neutral, explicitly separate fallback."""

    probe_id: str
    intent_version: int
    policy_id: str
    certainty: float | None
    raw_mode_coherence: float | None
    controller_fallback: float
    diagnostics: ProbeDiagnostics

    def __post_init__(self) -> None:
        _require_nonempty_trimmed(self.probe_id, name="probe_id")
        _require_nonnegative_int(self.intent_version, name="intent_version")
        _require_canonical_identifier(self.policy_id, name="policy_id")
        certainty = _optional_bounded_float(
            self.certainty,
            name="certainty",
            lower=0.0,
            upper=1.0,
        )
        raw = _optional_bounded_float(
            self.raw_mode_coherence,
            name="raw_mode_coherence",
            lower=-1.0,
            upper=1.0,
        )
        fallback = _bounded_float(
            self.controller_fallback,
            name="controller_fallback",
            lower=0.0,
            upper=1.0,
        )
        if fallback != CONTROLLER_FALLBACK:
            raise ValueError("controller_fallback must be the neutral 0.5 policy")
        if type(self.diagnostics) is not ProbeDiagnostics:
            raise TypeError("diagnostics must be ProbeDiagnostics")
        if (
            self.probe_id != self.diagnostics.probe_id
            or self.intent_version != self.diagnostics.intent_version
        ):
            raise ValueError("estimate identity differs from its diagnostics")
        if certainty is None and self.diagnostics.status is DiagnosticStatus.HEALTHY:
            raise ValueError("healthy diagnostics require measured certainty")
        if certainty is not None and self.diagnostics.status is DiagnosticStatus.UNAVAILABLE:
            raise ValueError("unavailable diagnostics cannot carry measured certainty")
        object.__setattr__(self, "certainty", certainty)
        object.__setattr__(self, "raw_mode_coherence", raw)
        object.__setattr__(self, "controller_fallback", fallback)


class TransparencyEstimator:
    """Apply one frozen calibration without allowing diagnostics into ``C_t``."""

    __slots__ = ("_calibration",)

    def __init__(self, calibration: TransparencyCalibration) -> None:
        if type(calibration) is not TransparencyCalibration:
            raise TypeError("calibration must be TransparencyCalibration")
        self._calibration = calibration

    @property
    def calibration(self) -> TransparencyCalibration:
        return self._calibration

    def estimate(self, evidence: TransparencyEvidence) -> TransparencyEstimate:
        if type(evidence) is not TransparencyEvidence:
            raise TypeError("evidence must be TransparencyEvidence")

        diagnostics, unavailable = _build_diagnostics(
            evidence,
            calibration=self._calibration,
        )
        raw = evidence.mode_coherence
        certainty = None
        if not unavailable:
            assert raw is not None
            certainty = _clip_probability(
                (raw - self._calibration.low_anchor)
                / (self._calibration.high_anchor - self._calibration.low_anchor)
            )
        return TransparencyEstimate(
            probe_id=evidence.probe_id,
            intent_version=evidence.intent_version,
            policy_id=self._calibration.policy_id,
            certainty=certainty,
            raw_mode_coherence=raw,
            controller_fallback=CONTROLLER_FALLBACK,
            diagnostics=diagnostics,
        )


def estimate_transparency(
    evidence: TransparencyEvidence,
    calibration: TransparencyCalibration,
) -> TransparencyEstimate:
    """Convenience pure function for one evidence/calibration pair."""

    return TransparencyEstimator(calibration).estimate(evidence)


def project_search_belief(estimate: TransparencyEstimate) -> SearchBelief:
    """Project measured certainty into the existing Session Context contract.

    ``ProbeDiagnostics`` describes wider retrieval health.  Session Context's
    ``ProbeQuality`` instead follows its certainty availability truth table:
    a measured certainty is ``VALID`` even when a non-blocking lexical or
    under-filled warning makes ``D_t`` degraded. A degraded observation without
    a score is ``LOW_QUALITY``; an unavailable calibration or geometric signal
    is ``INSUFFICIENT``.
    """

    if type(estimate) is not TransparencyEstimate:
        raise TypeError("estimate must be TransparencyEstimate")
    diagnostics = estimate.diagnostics
    if estimate.certainty is not None:
        quality = ProbeQuality.VALID
        quality_reasons: tuple[str, ...] = ()
    elif diagnostics.status is DiagnosticStatus.DEGRADED:
        quality = ProbeQuality.LOW_QUALITY
        quality_reasons = diagnostics.reason_codes
    else:
        quality = ProbeQuality.INSUFFICIENT
        quality_reasons = diagnostics.reason_codes

    raw_display = (
        None
        if estimate.raw_mode_coherence is None
        else _clip_probability(estimate.raw_mode_coherence)
    )
    return SearchBelief(
        based_on_intent_version=estimate.intent_version,
        certainty=estimate.certainty,
        certainty_method=estimate.policy_id,
        certainty_evidence=CertaintyEvidence(
            probe_id=estimate.probe_id,
            probe_size=diagnostics.dense_count,
            raw_concentration=raw_display,
            quality_status=quality,
            quality_reasons=quality_reasons,
        ),
        candidate_modes=(),
        facet_stats=(),
    )


def _build_diagnostics(
    evidence: TransparencyEvidence,
    *,
    calibration: TransparencyCalibration,
) -> tuple[ProbeDiagnostics, bool]:
    unavailable_reasons: set[str] = set()
    degraded_reasons: set[str] = set()
    dense_count = len(evidence.dense_hits)
    lexical_count = len(evidence.lexical_hits)

    if not calibration.approved:
        unavailable_reasons.add("calibration_unapproved")
    if evidence.eligible_count == 0:
        unavailable_reasons.add("eligible_catalog_empty")
    if dense_count == 0:
        unavailable_reasons.add("dense_probe_empty")
    underfilled = 0 < dense_count < evidence.probe_k
    if underfilled:
        degraded_reasons.add("dense_probe_underfilled")
    if evidence.mode_count < 2:
        unavailable_reasons.add("insufficient_modes")
    elif evidence.mode_coherence is None:
        unavailable_reasons.add("mode_coherence_unavailable")

    if evidence.listing_coherence is None:
        degraded_reasons.add("listing_coherence_unavailable")
    if not evidence.lexical_available:
        degraded_reasons.add("lexical_unavailable")
    else:
        if lexical_count == 0:
            degraded_reasons.add("lexical_probe_empty")
        if evidence.lexical_token_coverage is None or evidence.lexical_mean_normalized_idf is None:
            degraded_reasons.add("lexical_metrics_unavailable")
    if evidence.duplicate_warning:
        degraded_reasons.add("duplicate_concentration")
    if evidence.hard_filter_relaxed:
        degraded_reasons.add("hard_filter_relaxed")

    unavailable = bool(unavailable_reasons)
    if unavailable:
        status = DiagnosticStatus.UNAVAILABLE
    elif degraded_reasons:
        status = DiagnosticStatus.DEGRADED
    else:
        status = DiagnosticStatus.HEALTHY
    reason_codes = tuple(sorted(unavailable_reasons | degraded_reasons))

    if evidence.lexical_available:
        observed_overlap = len(set(evidence.dense_hits).intersection(evidence.lexical_hits))
        overlap_count: int | None = observed_overlap
        route_overlap: float | None = observed_overlap / evidence.probe_k
    else:
        overlap_count = None
        route_overlap = None

    diagnostics = ProbeDiagnostics(
        probe_id=evidence.probe_id,
        intent_version=evidence.intent_version,
        status=status,
        reason_codes=reason_codes,
        probe_k=evidence.probe_k,
        eligible_count=evidence.eligible_count,
        dense_count=dense_count,
        lexical_count=lexical_count,
        fill_ratio=dense_count / evidence.probe_k,
        lexical_available=evidence.lexical_available,
        route_overlap_count=overlap_count,
        route_overlap=route_overlap,
        listing_coherence=evidence.listing_coherence,
        mode_coherence=evidence.mode_coherence,
        mode_count=evidence.mode_count,
        largest_mode_share=evidence.largest_mode_share,
        effective_mode_count=evidence.effective_mode_count,
        duplicate_warning=evidence.duplicate_warning,
        lexical_token_coverage=evidence.lexical_token_coverage,
        lexical_mean_normalized_idf=evidence.lexical_mean_normalized_idf,
        hard_filter_relaxed=evidence.hard_filter_relaxed,
    )
    return diagnostics, unavailable


def _validate_overlap(diagnostics: ProbeDiagnostics) -> None:
    count = diagnostics.route_overlap_count
    ratio = diagnostics.route_overlap
    if not diagnostics.lexical_available:
        if count is not None or ratio is not None:
            raise ValueError("unavailable lexical route cannot carry overlap")
        return
    if type(count) is not int or count < 0:
        raise ValueError("route_overlap_count must be a non-negative integer")
    if count > min(diagnostics.dense_count, diagnostics.lexical_count):
        raise ValueError("route_overlap_count exceeds a route hit count")
    observed = _bounded_float(ratio, name="route_overlap", lower=0.0, upper=1.0)
    expected = count / diagnostics.probe_k
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError("route_overlap does not match overlap_count / probe_k")
    object.__setattr__(diagnostics, "route_overlap", observed)


def _validate_mode_summary(
    *,
    mode_count: int,
    largest_mode_share: float,
    effective_mode_count: float,
    mode_coherence: float | None,
) -> None:
    if mode_count == 0:
        if largest_mode_share != 0.0 or effective_mode_count != 0.0:
            raise ValueError("an empty mode summary requires zero shares and effective count")
        if mode_coherence is not None:
            raise ValueError("an empty mode summary cannot carry mode coherence")
        return
    if largest_mode_share <= 0.0:
        raise ValueError("a non-empty mode summary requires a positive largest share")
    minimum_largest = 1.0 / mode_count
    if largest_mode_share + _FLOAT_TOLERANCE < minimum_largest:
        raise ValueError("largest_mode_share is inconsistent with mode_count")
    if effective_mode_count < 1.0 or effective_mode_count > mode_count + _FLOAT_TOLERANCE:
        raise ValueError("effective_mode_count is inconsistent with mode_count")
    if mode_count < 2 and mode_coherence is not None:
        raise ValueError("fewer than two modes cannot carry mode coherence")


def _require_identifier_tuple(value: object, *, name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for index, item in enumerate(value):
        _require_nonempty_trimmed(item, name=f"{name}[{index}]")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique product IDs")


def _require_reason_codes(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("reason_codes must be a tuple")
    for reason in value:
        _require_canonical_identifier(reason, name="reason code")
    if value != tuple(sorted(set(value))):
        raise ValueError("reason_codes must be sorted and unique")


def _require_nonempty_trimmed(value: object, *, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")


def _require_canonical_identifier(value: object, *, name: str) -> None:
    if type(value) is not str or _CANONICAL_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lower-snake-case identifier")


def _require_bool(value: object, *, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: object, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _optional_bounded_float(
    value: object,
    *,
    name: str,
    lower: float,
    upper: float,
) -> float | None:
    if value is None:
        return None
    return _bounded_float(value, name=name, lower=lower, upper=upper)


def _bounded_float(value: object, *, name: str, lower: float, upper: float) -> float:
    result = _finite_float(value, name=name)
    if result < lower or result > upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(cast(int | float, value))
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))
