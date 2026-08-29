"""Provider-independent Query Understanding data contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from shopping_copilot.session_context import (
    FeedbackSignal,
    IntentState,
    ProductFeedback,
    SemanticPolarity,
    StateUpdateBatch,
)
from shopping_copilot.session_context.models import PreferenceValue


class UnderstandingDisposition(str, Enum):
    READY = "ready"
    NO_CHANGE = "no_change"
    NEEDS_CLARIFICATION = "needs_clarification"


class GoalAction(str, Enum):
    KEEP = "keep"
    REVISE = "revise"
    SWITCH = "switch"


class PreferenceRelation(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class PreferenceStrength(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class PreferenceBasis(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class DiversityMode(str, Enum):
    AUTO = "auto"
    INCREASE = "increase"
    DECREASE = "decrease"


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryOption:
    """Local model reference bound to one trusted category scope."""

    ref: str
    scope_id: str
    label: str
    is_root: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ShownProductView:
    """Local reference for product feedback without exposing raw catalog state."""

    ref: str
    product_ids: tuple[str, ...]
    label: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivePreferenceView:
    """A model-safe view of one active preference."""

    ref: str
    facet: str | None
    relation: str
    value: PreferenceValue | None
    meaning: str
    strength: str
    source: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconcileRequest:
    """Everything the model may inspect for one interpretation turn."""

    turn: int
    base_intent_version: int
    latest_utterance: str
    current_goal: str | None
    active_preferences: tuple[ActivePreferenceView, ...]
    dont_care_facets: tuple[str, ...]
    last_assistant_message: str | None
    last_question: str | None
    category_options: tuple[CategoryOption, ...]
    shown_products: tuple[ShownProductView, ...]
    allowed_dont_care_facets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class GoalFrame:
    action: GoalAction
    value: str | None

    def __post_init__(self) -> None:
        if type(self.action) is not GoalAction:
            raise TypeError("goal action must be a GoalAction")
        if self.action is GoalAction.KEEP and self.value is not None:
            raise ValueError("keep goal cannot include a value")
        if self.action in (GoalAction.REVISE, GoalAction.SWITCH) and (
            type(self.value) is not str or not self.value.strip()
        ):
            raise ValueError("revise and switch goal actions require a non-empty value")


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferenceFrame:
    """Fields shared by the three unambiguous model-facing preference shapes."""

    strength: PreferenceStrength
    basis: PreferenceBasis
    meaning: str
    evidence: str
    confidence: float

    def __post_init__(self) -> None:
        if type(self.strength) is not PreferenceStrength:
            raise TypeError("preference strength is invalid")
        if type(self.basis) is not PreferenceBasis:
            raise TypeError("preference basis is invalid")
        if type(self.meaning) is not str or not self.meaning.strip():
            raise ValueError("preference meaning must be non-empty")
        if type(self.evidence) is not str or not self.evidence.strip():
            raise ValueError("preference evidence must be non-empty")
        if type(self.confidence) not in (int, float) or (
            type(self.confidence) is float and not math.isfinite(self.confidence)
        ):
            raise TypeError("preference confidence must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("preference confidence must be between zero and one")


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredPreferenceFrame(PreferenceFrame):
    """One named facet condition, including semantic-fallback numeric ranges."""

    facet: str
    relation: PreferenceRelation
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        PreferenceFrame.__post_init__(self)
        if type(self.facet) is not str or not self.facet.strip():
            raise ValueError("structured preference facet must be non-empty")
        if type(self.relation) is not PreferenceRelation:
            raise TypeError("structured preference relation must be a PreferenceRelation")
        if (
            type(self.values) is not tuple
            or not self.values
            or any(type(value) is not str or not value.strip() for value in self.values)
        ):
            raise TypeError("structured preference values must be non-empty strings")
        if (
            self.relation in (PreferenceRelation.EQ, PreferenceRelation.NEQ)
            and len(self.values) != 1
        ):
            raise ValueError("eq and neq require exactly one structured value")


@dataclass(frozen=True, slots=True, kw_only=True)
class PricePreferenceFrame(PreferenceFrame):
    """One USD price bound with no facet/value-shape ambiguity."""

    relation: PreferenceRelation
    value_usd: str

    def __post_init__(self) -> None:
        PreferenceFrame.__post_init__(self)
        if type(self.relation) is not PreferenceRelation:
            raise TypeError("price preference relation must be a PreferenceRelation")
        if self.relation not in (
            PreferenceRelation.LT,
            PreferenceRelation.LE,
            PreferenceRelation.GT,
            PreferenceRelation.GE,
        ):
            raise ValueError("price preference relation is invalid")
        if type(self.value_usd) is not str or not self.value_usd.strip():
            raise ValueError("price value_usd must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticPreferenceFrame(PreferenceFrame):
    """One open-ended positive or negative natural-language condition."""

    polarity: SemanticPolarity

    def __post_init__(self) -> None:
        PreferenceFrame.__post_init__(self)
        if type(self.polarity) is not SemanticPolarity:
            raise TypeError("semantic preference polarity is invalid")


NewPreferenceFrame: TypeAlias = (
    StructuredPreferenceFrame | PricePreferenceFrame | SemanticPreferenceFrame
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackFrame:
    target_refs: tuple[str, ...]
    signal: FeedbackSignal
    compared_to_refs: tuple[str, ...]
    evidence: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BehavioralDirectives:
    diversity: DiversityMode
    comparison_requested: bool
    explanation_requested: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ClarificationNeed:
    needed: bool
    reason: str | None
    alternatives: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconciledIntentFrame:
    """Untrusted complete target state emitted by exactly one tool call."""

    base_intent_version: int
    disposition: UnderstandingDisposition
    goal: GoalFrame
    keep_active_refs: tuple[str, ...]
    structured_preferences: tuple[StructuredPreferenceFrame, ...]
    price_preferences: tuple[PricePreferenceFrame, ...]
    semantic_preferences: tuple[SemanticPreferenceFrame, ...]
    dont_care_facets: tuple[str, ...]
    feedback: tuple[FeedbackFrame, ...]
    directives: BehavioralDirectives
    clarification: ClarificationNeed
    summary: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderTrace:
    """Safe provider metadata; never includes credentials or raw tool arguments."""

    response_id: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderResult:
    frame: ReconciledIntentFrame
    trace: ProviderTrace


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterializationResult:
    update: StateUpdateBatch | None
    final_intent: IntentState
    feedback: tuple[ProductFeedback, ...]
    directives: BehavioralDirectives
    clarification: ClarificationNeed
    semantic_fallback_facets: tuple[str, ...]
    ignored_dont_care_facets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class UnderstandingTrace:
    attempts: tuple[ProviderTrace, ...]
    interpretation_summary: str
    semantic_fallback_facets: tuple[str, ...]
    ignored_dont_care_facets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedTurnIntent:
    """Accepted intent ready for query compilation and Probe, but not yet committed."""

    update: StateUpdateBatch | None
    final_intent: IntentState
    feedback: tuple[ProductFeedback, ...]
    directives: BehavioralDirectives
    clarification: ClarificationNeed
    trace: UnderstandingTrace
