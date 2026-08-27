"""Immutable leaf values for the session-context domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

ScalarValue: TypeAlias = str | int | float | bool
PreferenceValue: TypeAlias = ScalarValue | tuple[ScalarValue, ...]

MASS_TOLERANCE = 1e-9


class Operator(str, Enum):
    """Committed structured-preference operators."""

    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class SemanticPolarity(str, Enum):
    """Polarity of a semantic-only preference."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class Commitment(str, Enum):
    """Whether a preference is a hard requirement or a soft desire."""

    HARD = "hard"
    SOFT = "soft"


class PreferenceSource(str, Enum):
    """Trusted provenance class for a preference."""

    USER_EXPLICIT = "user_explicit"
    BEHAVIORAL_FEEDBACK = "behavioral_feedback"
    SYSTEM_INFERRED = "system_inferred"


class ProbeQuality(str, Enum):
    """Availability and reliability of Probe evidence."""

    VALID = "valid"
    LOW_QUALITY = "low_quality"
    INSUFFICIENT = "insufficient"


class FeedbackSignal(str, Enum):
    """User feedback recorded against previously shown products."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    SELECTED = "selected"
    REJECTED = "rejected"
    COMPARATIVE = "comparative"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfilePrior:
    """Immutable profile supplied at session reset."""

    purchase_frequency: str
    average_prior_rating: float | None
    rating_style: str
    preference_tags: tuple[str, ...]
    summary: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferenceDraft:
    """Untrusted preference interpretation before grounding and ID assignment."""

    facet: str | None
    operator: Operator | None
    value: PreferenceValue | None
    semantic_text: str | None
    semantic_polarity: SemanticPolarity | None
    commitment: Commitment
    source: PreferenceSource
    source_turn: int
    evidence_text: str
    interpretation_confidence: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Preference(PreferenceDraft):
    """Grounded preference accepted at the committed-operation boundary."""

    id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentState:
    """Current canonical user-need facts."""

    goal: str | None
    preferences: tuple[Preference, ...]
    dont_care_facets: frozenset[str]
    version: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CertaintyEvidence:
    """Evidence supporting an available or unavailable certainty value."""

    probe_id: str
    probe_size: int
    raw_concentration: float | None
    quality_status: ProbeQuality
    quality_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ValueMass:
    """Conditional mass for one canonical facet value."""

    value: ScalarValue
    mass: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FacetStats:
    """Probe statistics for one canonical facet."""

    facet: str
    entropy: float
    coverage: float
    top_values: tuple[ValueMass, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateMode:
    """One coherent mode in the Probe candidate space."""

    id: str
    label: str
    mass: float
    representative_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchBelief:
    """Immutable catalog observation associated with one intent version."""

    based_on_intent_version: int
    certainty: float | None
    certainty_method: str
    certainty_evidence: CertaintyEvidence
    candidate_modes: tuple[CandidateMode, ...]
    facet_stats: tuple[FacetStats, ...]
