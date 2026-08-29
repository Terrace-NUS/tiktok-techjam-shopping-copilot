"""Runtime Fuzzy Intent Volume and its user-facing transparency projection.

This module is deliberately separate from :mod:`transparency`, which preserves
the earlier fixed Top-K semantic-mode metric for compatibility.  Intent Volume
measures the density-corrected mass left by the complete Session Context.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from shopping_copilot.query_compiler import CompiledHardConstraint, CompiledQuery
from shopping_copilot.session_context import Commitment, IntentState, Operator, SemanticPolarity
from shopping_copilot.session_context.models import Preference, PreferenceValue

from .dense import DenseIndex
from .embedding import TextEmbedder
from .hard_mask import ResolvedHardMask

INTENT_TRANSPARENCY_SCHEMA: Literal["shopping-copilot/intent-transparency/v1"] = (
    "shopping-copilot/intent-transparency/v1"
)
INTENT_VOLUME_POLICY_ID = "soft_hybrid_intent_volume_v1"
INTENT_VOLUME_MAPPING_ID = "catalog_log_volume_v1"

FloatVector = NDArray[np.float32]
BoolVector = NDArray[np.bool_]


class IntentVolumeStatus(str, Enum):
    """Whether the measurement is available and how cautiously it should be read."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class IntentVolumeDirection(str, Enum):
    """Observed state transition without imposing false monotonicity."""

    INITIAL = "initial"
    NARROWER = "narrower"
    BROADER = "broader"
    STABLE = "stable"
    MOVED = "moved"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentVolumePolicy:
    """Frozen hackathon runtime parameters selected by the expanded experiment."""

    policy_id: str = INTENT_VOLUME_POLICY_ID
    mapping_id: str = INTENT_VOLUME_MAPPING_ID
    density_temperature: float = 0.025
    membership_quantile: float = 0.85
    membership_temperature: float = 0.06
    hard_mismatch_floor: float = 0.01
    soft_preference_exponent: float = 0.5
    stable_relative_tolerance: float = 0.10
    diagnostic_top_k: int = 20
    approved: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, name="policy_id")
        _require_identifier(self.mapping_id, name="mapping_id")
        _require_positive(self.density_temperature, name="density_temperature")
        _require_open_probability(self.membership_quantile, name="membership_quantile")
        _require_positive(self.membership_temperature, name="membership_temperature")
        _require_open_probability(self.hard_mismatch_floor, name="hard_mismatch_floor")
        _require_open_probability(
            self.soft_preference_exponent,
            name="soft_preference_exponent",
            upper_inclusive=True,
        )
        tolerance = _require_finite(
            self.stable_relative_tolerance,
            name="stable_relative_tolerance",
        )
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("stable_relative_tolerance must lie in [0, 1)")
        if type(self.diagnostic_top_k) is not int or self.diagnostic_top_k <= 0:
            raise ValueError("diagnostic_top_k must be a positive integer")
        if type(self.approved) is not bool:
            raise TypeError("approved must be a boolean")


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogDensitySnapshot:
    """One density vector bound to an immutable dense index and temperature."""

    index_id: str
    catalog_semantic_release_id: str
    temperature: float
    values: FloatVector

    def __post_init__(self) -> None:
        _require_text(self.index_id, name="index_id")
        _require_text(
            self.catalog_semantic_release_id,
            name="catalog_semantic_release_id",
        )
        _require_positive(self.temperature, name="temperature")
        observed = np.asarray(self.values)
        if observed.ndim != 1 or observed.dtype != np.float32:
            raise TypeError("density values must be a float32 vector")
        if not np.isfinite(observed).all() or np.any(observed < 1.0 - 1e-4):
            raise ValueError("density values must be finite and at least one")
        owned = np.array(observed, dtype=np.float32, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "values", owned)


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentVolumeDiagnostics:
    """D_t: measurement health kept separate from the transparency scalar."""

    status: IntentVolumeStatus
    reason_codes: tuple[str, ...]
    semantic_factor_count: int
    hard_factor_count: int
    relaxed_hard_preference_ids: tuple[str, ...]
    top_all_hard_compliance: float | None
    top_mean_hard_factor_compliance: float | None
    active_facets: tuple[str, ...]
    dont_care_facets: tuple[str, ...]
    open_facets: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not IntentVolumeStatus:
            raise TypeError("status must be IntentVolumeStatus")
        for name in ("semantic_factor_count", "hard_factor_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "reason_codes",
            "relaxed_hard_preference_ids",
            "active_facets",
            "dont_care_facets",
            "open_facets",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or value != tuple(sorted(set(value))):
                raise ValueError(f"{name} must be a sorted unique tuple")
        for name in (
            "top_all_hard_compliance",
            "top_mean_hard_factor_compliance",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_probability(value, name=name)

    def as_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "semantic_factor_count": self.semantic_factor_count,
            "hard_factor_count": self.hard_factor_count,
            "relaxed_hard_preference_ids": list(self.relaxed_hard_preference_ids),
            "top_all_hard_compliance": self.top_all_hard_compliance,
            "top_mean_hard_factor_compliance": self.top_mean_hard_factor_compliance,
            "active_facets": list(self.active_facets),
            "dont_care_facets": list(self.dont_care_facets),
            "open_facets": list(self.open_facets),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentTransparencyEstimate:
    """Serializable runtime output consumed by UI and future retrieval control."""

    schema: Literal["shopping-copilot/intent-transparency/v1"]
    session_id: str
    dense_index_id: str
    catalog_semantic_release_id: str
    policy_id: str
    mapping_id: str
    intent_version: int
    goal: str | None
    transparency: float | None
    change: float | None
    direction: IntentVolumeDirection
    remaining_intent_volume: float | None
    catalog_reference_volume: float
    goal_reference_volume: float | None
    diagnostics: IntentVolumeDiagnostics

    def __post_init__(self) -> None:
        if self.schema != INTENT_TRANSPARENCY_SCHEMA:
            raise ValueError("intent transparency schema is invalid")
        _require_text(self.session_id, name="session_id")
        _require_text(self.dense_index_id, name="dense_index_id")
        _require_text(
            self.catalog_semantic_release_id,
            name="catalog_semantic_release_id",
        )
        _require_identifier(self.policy_id, name="policy_id")
        _require_identifier(self.mapping_id, name="mapping_id")
        if type(self.intent_version) is not int or self.intent_version < 0:
            raise ValueError("intent_version must be a non-negative integer")
        if self.goal is not None and (type(self.goal) is not str or not self.goal.strip()):
            raise ValueError("goal must be null or non-empty")
        if self.transparency is not None:
            _require_probability(self.transparency, name="transparency")
        if self.change is not None:
            change = _require_finite(self.change, name="change")
            if not -1.0 <= change <= 1.0:
                raise ValueError("change must lie in [-1, 1]")
        if type(self.direction) is not IntentVolumeDirection:
            raise TypeError("direction must be IntentVolumeDirection")
        if self.remaining_intent_volume is not None:
            _require_nonnegative(self.remaining_intent_volume, name="remaining_intent_volume")
        _require_positive(self.catalog_reference_volume, name="catalog_reference_volume")
        if self.goal_reference_volume is not None:
            _require_nonnegative(self.goal_reference_volume, name="goal_reference_volume")
        if type(self.diagnostics) is not IntentVolumeDiagnostics:
            raise TypeError("diagnostics must be IntentVolumeDiagnostics")
        unavailable = self.diagnostics.status is IntentVolumeStatus.UNAVAILABLE
        if unavailable != (self.transparency is None):
            raise ValueError("unavailable status and transparency availability disagree")
        if unavailable != (self.direction is IntentVolumeDirection.UNAVAILABLE):
            raise ValueError("unavailable status and direction disagree")

    def as_payload(self) -> dict[str, object]:
        """Return the stable JSON-facing v1 contract."""

        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "dense_index_id": self.dense_index_id,
            "catalog_semantic_release_id": self.catalog_semantic_release_id,
            "policy_id": self.policy_id,
            "mapping_id": self.mapping_id,
            "intent_version": self.intent_version,
            "goal": self.goal,
            "transparency": self.transparency,
            "change": self.change,
            "direction": self.direction.value,
            "remaining_intent_volume": self.remaining_intent_volume,
            "catalog_reference_volume": self.catalog_reference_volume,
            "goal_reference_volume": self.goal_reference_volume,
            "diagnostics": self.diagnostics.as_payload(),
        }


@dataclass(frozen=True, slots=True)
class _SemanticFactor:
    text: str
    polarity: int
    exponent: float
    source: str


class HardConstraintResolver(Protocol):
    """Small structural boundary implemented by :class:`HardMaskResolver`."""

    def resolve(self, query: CompiledQuery) -> ResolvedHardMask:
        """Resolve one compiled hard constraint against the bound catalog."""


class IntentVolumeEstimator:
    """Measure one complete IntentState over the full density-corrected catalog."""

    __slots__ = (
        "_catalog_reference_volume",
        "_density",
        "_density_weights",
        "_embedder",
        "_factor_score_cache",
        "_index",
        "_policy",
        "_resolver",
    )

    def __init__(
        self,
        *,
        dense_index: DenseIndex,
        embedder: TextEmbedder,
        hard_mask_resolver: HardConstraintResolver,
        density: CatalogDensitySnapshot,
        policy: IntentVolumePolicy | None = None,
    ) -> None:
        if type(dense_index) is not DenseIndex:
            raise TypeError("dense_index must be an exact DenseIndex")
        observed_policy = policy or IntentVolumePolicy()
        if type(observed_policy) is not IntentVolumePolicy:
            raise TypeError("policy must be IntentVolumePolicy")
        if embedder.spec != dense_index.manifest.embedding:
            raise ValueError("embedder specification differs from the dense index")
        if density.index_id != dense_index.index_id or (
            density.catalog_semantic_release_id != dense_index.manifest.catalog_semantic_release_id
        ):
            raise ValueError("density snapshot differs from the dense index binding")
        if density.values.shape != (dense_index.manifest.product_count,):
            raise ValueError("density snapshot has the wrong product count")
        if not math.isclose(
            density.temperature,
            observed_policy.density_temperature,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("density temperature differs from the runtime policy")
        self._index = dense_index
        self._embedder = embedder
        self._resolver = hard_mask_resolver
        self._density = density
        self._policy = observed_policy
        self._density_weights = np.asarray(1.0 / density.values, dtype=np.float64)
        self._catalog_reference_volume = float(np.sum(self._density_weights))
        self._factor_score_cache: dict[str, FloatVector] = {}

    @property
    def policy(self) -> IntentVolumePolicy:
        return self._policy

    @property
    def catalog_reference_volume(self) -> float:
        return self._catalog_reference_volume

    def estimate(
        self,
        *,
        session_id: str,
        intent: IntentState,
        compiled: CompiledQuery,
        previous: IntentTransparencyEstimate | None = None,
        goal_switched: bool = False,
        open_facets: Sequence[str] = (),
    ) -> IntentTransparencyEstimate:
        """Measure and classify one accepted Session Context state."""

        _require_text(session_id, name="session_id")
        if type(intent) is not IntentState:
            raise TypeError("intent must be an exact IntentState")
        if type(compiled) is not CompiledQuery:
            raise TypeError("compiled must be an exact CompiledQuery")
        if type(goal_switched) is not bool:
            raise TypeError("goal_switched must be a boolean")
        normalized_open_facets = _sorted_texts(open_facets, name="open_facets")
        self._validate_compiled(intent, compiled)
        self._validate_previous(session_id, intent, previous)

        if not self._policy.approved:
            return self._unavailable(
                session_id=session_id,
                intent=intent,
                reason="policy_unapproved",
                open_facets=normalized_open_facets,
            )
        if not compiled.search_ready:
            return self._unavailable(
                session_id=session_id,
                intent=intent,
                reason="intent_not_searchable",
                open_facets=normalized_open_facets,
            )

        hard_masks, relaxed_ids = self._resolve_hard_masks(compiled)
        hard_ids = {constraint.preference_id for constraint in compiled.hard_constraints}
        factors = _semantic_factors(
            intent,
            excluded_preference_ids=hard_ids - set(relaxed_ids),
            soft_exponent=self._policy.soft_preference_exponent,
        )
        if not factors:
            return self._unavailable(
                session_id=session_id,
                intent=intent,
                reason="intent_has_no_volume_factors",
                open_facets=normalized_open_facets,
            )
        log_compatibility = self._semantic_log_compatibility(factors)
        if hard_masks:
            mismatch = math.log(self._policy.hard_mismatch_floor)
            for mask in hard_masks:
                log_compatibility += np.where(mask, 0.0, mismatch)

        compatibility = np.exp(log_compatibility)
        remaining_volume = float(np.dot(self._density_weights, compatibility))
        goal_volume = None
        if intent.goal is not None:
            goal_factor = _SemanticFactor(
                text=f"Product goal: {intent.goal}",
                polarity=1,
                exponent=1.0,
                source="goal",
            )
            goal_volume = float(
                np.dot(
                    self._density_weights,
                    np.exp(self._semantic_log_compatibility((goal_factor,))),
                )
            )
        transparency = project_intent_transparency(
            remaining_volume,
            reference_volume=self._catalog_reference_volume,
        )
        top_all, top_mean = _top_hard_compliance(
            log_compatibility,
            hard_masks,
            top_k=self._policy.diagnostic_top_k,
        )
        reasons: set[str] = set()
        if relaxed_ids:
            reasons.add("hard_constraint_relaxed")
        if top_all is not None and top_all < 0.5:
            reasons.add("low_all_hard_top_compliance")
        status = IntentVolumeStatus.DEGRADED if reasons else IntentVolumeStatus.HEALTHY
        change, direction = _transition(
            remaining_volume=remaining_volume,
            transparency=transparency,
            previous=previous,
            goal_switched=goal_switched,
            stable_relative_tolerance=self._policy.stable_relative_tolerance,
        )
        diagnostics = IntentVolumeDiagnostics(
            status=status,
            reason_codes=tuple(sorted(reasons)),
            semantic_factor_count=len(factors),
            hard_factor_count=len(compiled.hard_constraints),
            relaxed_hard_preference_ids=relaxed_ids,
            top_all_hard_compliance=top_all,
            top_mean_hard_factor_compliance=top_mean,
            active_facets=tuple(
                sorted({item.facet for item in intent.preferences if item.facet is not None})
            ),
            dont_care_facets=tuple(sorted(intent.dont_care_facets)),
            open_facets=normalized_open_facets,
        )
        return IntentTransparencyEstimate(
            schema=INTENT_TRANSPARENCY_SCHEMA,
            session_id=session_id,
            dense_index_id=self._index.index_id,
            catalog_semantic_release_id=self._index.manifest.catalog_semantic_release_id,
            policy_id=self._policy.policy_id,
            mapping_id=self._policy.mapping_id,
            intent_version=intent.version,
            goal=intent.goal,
            transparency=transparency,
            change=change,
            direction=direction,
            remaining_intent_volume=remaining_volume,
            catalog_reference_volume=self._catalog_reference_volume,
            goal_reference_volume=goal_volume,
            diagnostics=diagnostics,
        )

    def _semantic_log_compatibility(
        self,
        factors: Sequence[_SemanticFactor],
    ) -> NDArray[np.float64]:
        result = np.zeros(self._index.manifest.product_count, dtype=np.float64)
        for factor in factors:
            scores = self._factor_scores(factor.text)
            threshold = float(np.quantile(scores, self._policy.membership_quantile))
            z = (np.asarray(scores, dtype=np.float64) - threshold) / (
                self._policy.membership_temperature
            )
            signed = factor.polarity * z
            result += -np.logaddexp(0.0, -signed) * factor.exponent
        return result

    def _factor_scores(self, text: str) -> FloatVector:
        key = " ".join(text.split()).casefold()
        observed = self._factor_score_cache.get(key)
        if observed is not None:
            return observed
        vector = self._embedder.encode_query(text)
        scores = self._index.score_vector(vector).values
        owned = np.array(scores, dtype=np.float32, copy=True)
        owned.setflags(write=False)
        result = owned
        self._factor_score_cache[key] = result
        return result

    def _resolve_hard_masks(
        self,
        compiled: CompiledQuery,
    ) -> tuple[tuple[BoolVector, ...], tuple[str, ...]]:
        masks: list[BoolVector] = []
        relaxed: set[str] = set()
        for constraint in compiled.hard_constraints:
            resolution = self._resolver.resolve(_single_constraint_query(compiled, constraint))
            if resolution.hard_filter_relaxed:
                relaxed.add(constraint.preference_id)
                continue
            values = np.asarray(resolution.eligible_mask.values)
            if values.shape != (self._index.manifest.product_count,) or values.dtype != np.bool_:
                raise ValueError("hard resolver returned an invalid mask")
            masks.append(values)
        return tuple(masks), tuple(sorted(relaxed))

    def _validate_compiled(self, intent: IntentState, compiled: CompiledQuery) -> None:
        if compiled.intent_version != intent.version:
            raise ValueError("compiled query and intent version differ")
        if compiled.catalog_id != self._index.manifest.catalog_id or (
            compiled.catalog_semantic_release_id != self._index.manifest.catalog_semantic_release_id
        ):
            raise ValueError("compiled query differs from the active catalog binding")

    def _validate_previous(
        self,
        session_id: str,
        intent: IntentState,
        previous: IntentTransparencyEstimate | None,
    ) -> None:
        if previous is None:
            return
        if type(previous) is not IntentTransparencyEstimate:
            raise TypeError("previous must be IntentTransparencyEstimate")
        if previous.session_id != session_id:
            raise ValueError("previous estimate belongs to another session")
        if previous.dense_index_id != self._index.index_id or (
            previous.catalog_semantic_release_id != self._index.manifest.catalog_semantic_release_id
        ):
            raise ValueError("previous estimate belongs to another catalog binding")
        if previous.policy_id != self._policy.policy_id:
            raise ValueError("previous estimate uses another policy")
        if previous.intent_version > intent.version:
            raise ValueError("previous estimate is newer than the current intent")

    def _unavailable(
        self,
        *,
        session_id: str,
        intent: IntentState,
        reason: str,
        open_facets: tuple[str, ...],
    ) -> IntentTransparencyEstimate:
        diagnostics = IntentVolumeDiagnostics(
            status=IntentVolumeStatus.UNAVAILABLE,
            reason_codes=(reason,),
            semantic_factor_count=0,
            hard_factor_count=0,
            relaxed_hard_preference_ids=(),
            top_all_hard_compliance=None,
            top_mean_hard_factor_compliance=None,
            active_facets=tuple(
                sorted({item.facet for item in intent.preferences if item.facet is not None})
            ),
            dont_care_facets=tuple(sorted(intent.dont_care_facets)),
            open_facets=open_facets,
        )
        return IntentTransparencyEstimate(
            schema=INTENT_TRANSPARENCY_SCHEMA,
            session_id=session_id,
            dense_index_id=self._index.index_id,
            catalog_semantic_release_id=self._index.manifest.catalog_semantic_release_id,
            policy_id=self._policy.policy_id,
            mapping_id=self._policy.mapping_id,
            intent_version=intent.version,
            goal=intent.goal,
            transparency=None,
            change=None,
            direction=IntentVolumeDirection.UNAVAILABLE,
            remaining_intent_volume=None,
            catalog_reference_volume=self._catalog_reference_volume,
            goal_reference_volume=None,
            diagnostics=diagnostics,
        )


def load_catalog_density(
    path: Path,
    *,
    dense_index: DenseIndex,
    temperature: float,
) -> CatalogDensitySnapshot:
    """Load one verified temperature row from the offline density cache."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if type(dense_index) is not DenseIndex:
        raise TypeError("dense_index must be an exact DenseIndex")
    requested = _require_positive(temperature, name="temperature")
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["index_id"].item()) != dense_index.index_id:
            raise ValueError("density cache belongs to another dense index")
        temperatures = tuple(float(item) for item in cached["temperatures"])
        observed = next(
            (item for item in temperatures if math.isclose(item, requested, abs_tol=1e-12)),
            None,
        )
        if observed is None:
            raise ValueError("density cache does not contain the requested temperature")
        values = np.asarray(cached[f"density_{observed:.3f}"], dtype=np.float32)
    return CatalogDensitySnapshot(
        index_id=dense_index.index_id,
        catalog_semantic_release_id=dense_index.manifest.catalog_semantic_release_id,
        temperature=observed,
        values=values,
    )


def project_intent_transparency(
    remaining_volume: float,
    *,
    reference_volume: float,
) -> float:
    """Map remaining catalog mass to the stable 0–1 display scale."""

    remaining = _require_nonnegative(remaining_volume, name="remaining_volume")
    reference = _require_positive(reference_volume, name="reference_volume")
    value = 1.0 - math.log1p(min(remaining, reference)) / math.log1p(reference)
    return min(1.0, max(0.0, value))


def _transition(
    *,
    remaining_volume: float,
    transparency: float,
    previous: IntentTransparencyEstimate | None,
    goal_switched: bool,
    stable_relative_tolerance: float,
) -> tuple[float | None, IntentVolumeDirection]:
    if (
        previous is None
        or previous.transparency is None
        or (previous.remaining_intent_volume is None)
    ):
        return None, IntentVolumeDirection.INITIAL
    change = transparency - previous.transparency
    if goal_switched:
        return change, IntentVolumeDirection.MOVED
    prior = previous.remaining_intent_volume
    scale = max(prior, remaining_volume, 1e-12)
    if abs(remaining_volume - prior) / scale <= stable_relative_tolerance:
        return change, IntentVolumeDirection.STABLE
    if remaining_volume < prior:
        return change, IntentVolumeDirection.NARROWER
    return change, IntentVolumeDirection.BROADER


def _semantic_factors(
    intent: IntentState,
    *,
    excluded_preference_ids: set[str],
    soft_exponent: float,
) -> tuple[_SemanticFactor, ...]:
    factors: list[_SemanticFactor] = []
    if intent.goal:
        factors.append(
            _SemanticFactor(
                text=f"Product goal: {intent.goal}",
                polarity=1,
                exponent=1.0,
                source="goal",
            )
        )
    for preference in intent.preferences:
        if preference.id in excluded_preference_ids:
            continue
        text = _preference_text(preference)
        if text is None:
            continue
        polarity = _preference_polarity(preference)
        exponent = soft_exponent if preference.commitment is Commitment.SOFT else 1.0
        factors.append(
            _SemanticFactor(
                text=text,
                polarity=polarity,
                exponent=exponent,
                source=f"preference:{preference.id}",
            )
        )
    unique: dict[tuple[str, int], _SemanticFactor] = {}
    for factor in factors:
        key = (factor.text.casefold(), factor.polarity)
        observed = unique.get(key)
        if observed is None or factor.exponent > observed.exponent:
            unique[key] = factor
    return tuple(unique.values())


def _preference_text(preference: Preference) -> str | None:
    if preference.semantic_text and preference.semantic_text.strip():
        return preference.semantic_text.strip()
    if preference.evidence_text.strip():
        prefix = "Preference" if preference.facet is None else preference.facet.replace("_", " ")
        return f"{prefix}: {preference.evidence_text.strip()}"
    if preference.value is None:
        return None
    facet = preference.facet or "preference"
    return f"{facet.replace('_', ' ')}: {_render_value(preference.value)}"


def _preference_polarity(preference: Preference) -> int:
    if preference.operator in (Operator.NEQ, Operator.NOT_IN):
        return -1
    if preference.semantic_polarity is SemanticPolarity.NEGATIVE:
        return -1
    return 1


def _render_value(value: PreferenceValue) -> str:
    if isinstance(value, tuple):
        return " or ".join(str(item) for item in value)
    return str(value)


def _single_constraint_query(
    compiled: CompiledQuery,
    constraint: CompiledHardConstraint,
) -> CompiledQuery:
    return replace(
        compiled,
        hard_constraints=(constraint,),
        ranking_preferences=(),
    )


def _top_hard_compliance(
    log_compatibility: NDArray[np.float64],
    hard_masks: Sequence[BoolVector],
    *,
    top_k: int,
) -> tuple[float | None, float | None]:
    if not hard_masks:
        return None, None
    count = min(top_k, log_compatibility.size)
    top_indices = np.argsort(-log_compatibility, kind="stable")[:count]
    matrix = np.stack(hard_masks)
    all_hard = float(np.mean(np.all(matrix[:, top_indices], axis=0)))
    mean_hard = float(np.mean(matrix[:, top_indices]))
    return all_hard, mean_hard


def _sorted_texts(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = []
    for value in values:
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        result.append(value.strip())
    return tuple(sorted(set(result)))


def _require_identifier(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if not text.replace("_", "").isalnum() or not text[0].isalpha() or text != text.lower():
        raise ValueError(f"{name} must be a lowercase identifier")
    return text


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise TypeError(f"{name} must be a finite number")
    return number


def _require_positive(value: object, *, name: str) -> float:
    number = _require_finite(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_nonnegative(value: object, *, name: str) -> float:
    number = _require_finite(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _require_probability(value: object, *, name: str) -> float:
    number = _require_finite(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _require_open_probability(
    value: object,
    *,
    name: str,
    upper_inclusive: bool = False,
) -> float:
    number = _require_finite(value, name=name)
    upper_ok = number <= 1.0 if upper_inclusive else number < 1.0
    if number <= 0.0 or not upper_ok:
        interval = "(0, 1]" if upper_inclusive else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return number
