"""Immutable contracts for direction-aware shortlist and DeepSeek judgement."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, cast

from shopping_copilot.query_compiler import CompiledQuery
from shopping_copilot.session_context import IntentState

from ..vector_diversity import VectorCandidate

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJsonValue: TypeAlias = JsonScalar | tuple["FrozenJsonValue", ...] | Mapping[
    str, "FrozenJsonValue"
]

RANKING_CONTRACT_VERSION = "deepseek_candidate_judgement_v1"


class CandidateVerdict(str, Enum):
    STRONG_MATCH = "strong_match"
    POSSIBLE_MATCH = "possible_match"
    WEAK_MATCH = "weak_match"


class QualityRankingMode(str, Enum):
    DEEPSEEK = "deepseek"
    BGE_FALLBACK = "bge_fallback"


@dataclass(frozen=True, slots=True, kw_only=True)
class RankingUserProfile:
    """Versioned opaque envelope owned by the future long-term-memory module."""

    schema: str
    version: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.schema) is not str or not self.schema.strip():
            raise ValueError("user profile schema must be non-empty")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("user profile version must be positive")
        if not isinstance(self.payload, Mapping):
            raise TypeError("user profile payload must be a mapping")
        frozen = _freeze_json_mapping(self.payload, name="user profile payload")
        object.__setattr__(self, "payload", frozen)

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "payload": _thaw_json(cast(Mapping[str, FrozenJsonValue], self.payload)),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RankingCandidateCard:
    """Compact product evidence shown to BGE-selected DeepSeek ranking."""

    parent_asin: str
    shortlist_rank: int
    original_candidate_rank: int
    bge_relevance: float
    normalized_bge_score: float
    direction_id: str | None
    routes: tuple[str, ...]
    product_text: str

    def __post_init__(self) -> None:
        _require_text(self.parent_asin, name="candidate parent_asin")
        for name, value in (
            ("shortlist_rank", self.shortlist_rank),
            ("original_candidate_rank", self.original_candidate_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        _require_probability(self.bge_relevance, name="bge_relevance")
        _require_probability(self.normalized_bge_score, name="normalized_bge_score")
        if self.direction_id is not None:
            _require_text(self.direction_id, name="direction_id")
        if type(self.routes) is not tuple or any(
            type(route) is not str or not route.strip() for route in self.routes
        ):
            raise TypeError("candidate routes must be non-empty strings")
        if len(set(self.routes)) != len(self.routes):
            raise ValueError("candidate routes must be unique")
        _require_text(self.product_text, name="product_text")


@dataclass(frozen=True, slots=True, kw_only=True)
class RankingShortlist:
    """Direction-protected BGE shortlist presented to one DeepSeek call."""

    model_id: str
    requested_top_k: int
    protected_per_direction: int
    cards: tuple[RankingCandidateCard, ...]

    def __post_init__(self) -> None:
        _require_text(self.model_id, name="shortlist model_id")
        if type(self.requested_top_k) is not int or self.requested_top_k <= 0:
            raise ValueError("requested_top_k must be positive")
        if type(self.protected_per_direction) is not int or self.protected_per_direction < 0:
            raise ValueError("protected_per_direction must be non-negative")
        _validate_card_ranks(self.cards)
        if len(self.cards) > self.requested_top_k:
            raise ValueError("shortlist contains more cards than requested")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekRankingRequest:
    """One globally comparable batch of candidates for the resolved current intent."""

    request_id: str
    intent: IntentState
    compiled_query: CompiledQuery
    shortlist: RankingShortlist
    user_profile: RankingUserProfile | None = None

    def __post_init__(self) -> None:
        _require_text(self.request_id, name="request_id")
        if type(self.intent) is not IntentState:
            raise TypeError("intent must be an exact IntentState")
        if type(self.compiled_query) is not CompiledQuery:
            raise TypeError("compiled_query must be an exact CompiledQuery")
        if type(self.shortlist) is not RankingShortlist:
            raise TypeError("shortlist must be an exact RankingShortlist")
        if self.user_profile is not None and type(self.user_profile) is not RankingUserProfile:
            raise TypeError("user_profile must be a RankingUserProfile or None")
        if self.intent.version != self.compiled_query.intent_version:
            raise ValueError("intent and compiled query versions differ")
        if not self.compiled_query.search_ready:
            raise ValueError("compiled query is not search-ready")
        if not self.shortlist.cards:
            raise ValueError("DeepSeek ranking request requires candidates")


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateJudgement:
    """DeepSeek's evidence-aware assessment of one product, not a slate decision."""

    parent_asin: str
    fit_score: int
    verdict: CandidateVerdict
    matched_preference_ids: tuple[str, ...]
    unsupported_preference_ids: tuple[str, ...]
    conflict_preference_ids: tuple[str, ...]
    concerns: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.parent_asin, name="judgement parent_asin")
        if type(self.fit_score) is not int or not 0 <= self.fit_score <= 100:
            raise ValueError("fit_score must be an integer in [0, 100]")
        if type(self.verdict) is not CandidateVerdict:
            raise TypeError("verdict must be a CandidateVerdict")
        _validate_verdict_score(self.verdict, self.fit_score)
        for name, values in (
            ("matched_preference_ids", self.matched_preference_ids),
            ("unsupported_preference_ids", self.unsupported_preference_ids),
            ("conflict_preference_ids", self.conflict_preference_ids),
            ("concerns", self.concerns),
        ):
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise TypeError(f"{name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        preference_groups = (
            set(self.matched_preference_ids),
            set(self.unsupported_preference_ids),
            set(self.conflict_preference_ids),
        )
        if any(preference_groups[left] & preference_groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("preference judgement groups must be disjoint")
        _require_text(self.reason, name="judgement reason")
        if len(self.reason) > 400:
            raise ValueError("judgement reason must not exceed 400 characters")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekRankingTrace:
    response_id: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekJudgementResult:
    judgements: tuple[CandidateJudgement, ...]
    trace: DeepSeekRankingTrace


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityRankingHit:
    parent_asin: str
    rank: int
    shortlist_rank: int
    bge_relevance: float
    deepseek_fit: float | None
    quality: float
    verdict: CandidateVerdict | None
    matched_preference_ids: tuple[str, ...]
    unsupported_preference_ids: tuple[str, ...]
    conflict_preference_ids: tuple[str, ...]
    concerns: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityRankingResult:
    mode: QualityRankingMode
    deepseek_weight: float
    fallback_reason: str | None
    attempts: int
    traces: tuple[DeepSeekRankingTrace, ...]
    hits: tuple[QualityRankingHit, ...]

    @property
    def candidates(self) -> tuple[VectorCandidate, ...]:
        return tuple(
            VectorCandidate(
                parent_asin=hit.parent_asin,
                candidate_rank=hit.rank,
                relevance=hit.quality,
            )
            for hit in self.hits
        )


def _validate_card_ranks(cards: tuple[RankingCandidateCard, ...]) -> None:
    if type(cards) is not tuple:
        raise TypeError("shortlist cards must be a tuple")
    seen: set[str] = set()
    for expected_rank, card in enumerate(cards, start=1):
        if type(card) is not RankingCandidateCard:
            raise TypeError("shortlist contains an invalid card")
        if card.shortlist_rank != expected_rank:
            raise ValueError("shortlist ranks must be contiguous")
        if card.parent_asin in seen:
            raise ValueError("shortlist products must be unique")
        seen.add(card.parent_asin)


def _validate_verdict_score(verdict: CandidateVerdict, score: int) -> None:
    expected = (
        CandidateVerdict.STRONG_MATCH
        if score >= 75
        else CandidateVerdict.POSSIBLE_MATCH
        if score >= 40
        else CandidateVerdict.WEAK_MATCH
    )
    if verdict is not expected:
        raise ValueError("verdict does not match fit_score band")


def _freeze_json_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> Mapping[str, FrozenJsonValue]:
    result: dict[str, FrozenJsonValue] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = _freeze_json(item, name=f"{name}.{key}")
    return MappingProxyType(result)


def _freeze_json(value: object, *, name: str) -> FrozenJsonValue:
    if value is None or type(value) in (str, bool, int):
        return cast(JsonScalar, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(cast(Mapping[str, object], value), name=name)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name=f"{name}[]") for item in value)
    raise TypeError(f"{name} is not JSON-compatible")


def _thaw_json(value: FrozenJsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_probability(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def canonical_json(value: object) -> str:
    """Compact deterministic JSON used for model input and tests."""

    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
