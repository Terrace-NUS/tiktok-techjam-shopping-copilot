"""Immutable output contracts for deterministic query compilation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from shopping_copilot.session_context import (
    Commitment,
    Operator,
    PreferenceSource,
    SemanticPolarity,
)
from shopping_copilot.session_context.models import PreferenceValue

COMPILED_QUERY_SCHEMA: Literal["shopping-copilot/compiled-query/v0"] = (
    "shopping-copilot/compiled-query/v0"
)
QUERY_COMPILER_VERSION = "query_compiler_v0"


class ConstraintPolicy(str, Enum):
    """Evidence semantics required from the future hard-mask builder."""

    VERIFIED_CATEGORY = "verified_category"
    CONSERVATIVE_PRICE = "conservative_price"
    CLOSED_WORLD_RETRIEVAL_EVIDENCE = "closed_world_retrieval_evidence"


class CompilationTarget(str, Enum):
    """One observable destination for an active preference."""

    Q_LEX = "q_lex"
    Q_SEM = "q_sem"
    HARD_CONSTRAINT = "hard_constraint"
    RANKING_PREFERENCE = "ranking_preference"
    NOOP = "noop"


class RankingReason(str, Enum):
    """Why a condition remains ranking evidence instead of becoming a mask."""

    SOFT_COMMITMENT = "soft_commitment"
    NON_EXPLICIT_SOURCE = "non_explicit_source"
    SEMANTIC_ONLY = "semantic_only"
    UNSUPPORTED_HARD_FACET = "unsupported_hard_facet"


class DiversityDirective(str, Enum):
    """Explicit override for the default T_t-controlled diversity policy."""

    AUTO = "auto"
    INCREASE = "increase"
    DECREASE = "decrease"


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledHardConstraint:
    """One explicit structured requirement awaiting catalog mask resolution."""

    preference_id: str
    facet: str
    operator: Operator
    value: PreferenceValue
    policy: ConstraintPolicy


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledRankingPreference:
    """A preference scored by retrieval/ranking rather than enforced as a mask."""

    preference_id: str
    facet: str | None
    operator: Operator | None
    value: PreferenceValue | None
    semantic_text: str | None
    semantic_polarity: SemanticPolarity | None
    commitment: Commitment
    source: PreferenceSource
    reason: RankingReason


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferenceCompilationTrace:
    """Explain exactly where one Session Context preference was compiled."""

    preference_id: str
    targets: tuple[CompilationTarget, ...]
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledDirectives:
    """Provider-independent retrieval behavior requested by the user."""

    diversity: DiversityDirective
    comparison_requested: bool
    explanation_requested: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledQuery:
    """Complete deterministic bridge from resolved intent to retrieval."""

    schema: Literal["shopping-copilot/compiled-query/v0"]
    compiler_version: str
    catalog_id: str
    catalog_semantic_release_id: str
    category_graph_id: str
    intent_version: int
    q_lex: str
    q_sem: str
    search_ready: bool
    hard_constraints: tuple[CompiledHardConstraint, ...]
    ranking_preferences: tuple[CompiledRankingPreference, ...]
    dont_care_facets: tuple[str, ...]
    directives: CompiledDirectives
    requires_clarification: bool
    clarification_reason: str | None
    trace: tuple[PreferenceCompilationTrace, ...]
