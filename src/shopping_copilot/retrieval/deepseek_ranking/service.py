"""Quality ranking service with one repair attempt and a BGE fallback."""

from __future__ import annotations

from typing import Protocol

from .errors import DeepSeekRankingError, DeepSeekRankingErrorCode
from .models import (
    CandidateJudgement,
    DeepSeekJudgementResult,
    DeepSeekRankingRequest,
    DeepSeekRankingTrace,
    QualityRankingHit,
    QualityRankingMode,
    QualityRankingResult,
)

DEFAULT_DEEPSEEK_WEIGHT = 0.8
REPAIRABLE_ERRORS = frozenset(
    {
        DeepSeekRankingErrorCode.INVALID_PROVIDER_RESPONSE,
        DeepSeekRankingErrorCode.INVALID_TOOL_CALL,
        DeepSeekRankingErrorCode.INVALID_JUDGEMENTS,
    }
)
REPAIR_INSTRUCTION = (
    "Return every supplied candidate_id exactly once, use only supplied preference IDs, "
    "make each verdict agree with its numeric score band, and place each preference ID "
    "in at most one of matched, unsupported, or conflict."
)


class CandidateJudge(Protocol):
    def judge(
        self,
        request: DeepSeekRankingRequest,
        *,
        repair_instruction: str | None = None,
    ) -> DeepSeekJudgementResult: ...


class DeepSeekQualityRanker:
    """Blend evidence-aware DeepSeek fit with BGE relevance."""

    __slots__ = ("_deepseek_weight", "_provider", "_repair_once")

    def __init__(
        self,
        *,
        provider: CandidateJudge,
        deepseek_weight: float = DEFAULT_DEEPSEEK_WEIGHT,
        repair_once: bool = True,
    ) -> None:
        if not 0.0 <= deepseek_weight <= 1.0:
            raise ValueError("deepseek_weight must be in [0, 1]")
        if type(repair_once) is not bool:
            raise TypeError("repair_once must be a bool")
        self._provider = provider
        self._deepseek_weight = float(deepseek_weight)
        self._repair_once = repair_once

    def rank(self, request: DeepSeekRankingRequest) -> QualityRankingResult:
        if type(request) is not DeepSeekRankingRequest:
            raise TypeError("request must be an exact DeepSeekRankingRequest")
        attempts = 1
        traces: list[DeepSeekRankingTrace] = []
        try:
            result = self._provider.judge(request)
        except DeepSeekRankingError as first_error:
            if not self._repair_once or first_error.code not in REPAIRABLE_ERRORS:
                return self._fallback(request, attempts=attempts, error=first_error)
            attempts += 1
            try:
                result = self._provider.judge(
                    request,
                    repair_instruction=REPAIR_INSTRUCTION,
                )
            except DeepSeekRankingError as second_error:
                return self._fallback(request, attempts=attempts, error=second_error)
        traces.append(result.trace)
        return self._blend(
            request,
            result.judgements,
            attempts=attempts,
            traces=tuple(traces),
        )

    def _blend(
        self,
        request: DeepSeekRankingRequest,
        judgements: tuple[CandidateJudgement, ...],
        *,
        attempts: int,
        traces: tuple[DeepSeekRankingTrace, ...],
    ) -> QualityRankingResult:
        by_asin = {item.parent_asin: item for item in judgements}
        hits: list[QualityRankingHit] = []
        for card in request.shortlist.cards:
            judgement = by_asin[card.parent_asin]
            deepseek_fit = judgement.fit_score / 100.0
            quality = (
                self._deepseek_weight * deepseek_fit
                + (1.0 - self._deepseek_weight) * card.bge_relevance
            )
            hits.append(
                QualityRankingHit(
                    parent_asin=card.parent_asin,
                    rank=0,
                    shortlist_rank=card.shortlist_rank,
                    bge_relevance=card.bge_relevance,
                    deepseek_fit=deepseek_fit,
                    quality=quality,
                    verdict=judgement.verdict,
                    matched_preference_ids=judgement.matched_preference_ids,
                    unsupported_preference_ids=judgement.unsupported_preference_ids,
                    conflict_preference_ids=judgement.conflict_preference_ids,
                    concerns=judgement.concerns,
                    reason=judgement.reason,
                )
            )
        return QualityRankingResult(
            mode=QualityRankingMode.DEEPSEEK,
            deepseek_weight=self._deepseek_weight,
            fallback_reason=None,
            attempts=attempts,
            traces=traces,
            hits=_rank_hits(hits),
        )

    def _fallback(
        self,
        request: DeepSeekRankingRequest,
        *,
        attempts: int,
        error: DeepSeekRankingError,
    ) -> QualityRankingResult:
        hits = [
            QualityRankingHit(
                parent_asin=card.parent_asin,
                rank=0,
                shortlist_rank=card.shortlist_rank,
                bge_relevance=card.bge_relevance,
                deepseek_fit=None,
                quality=card.bge_relevance,
                verdict=None,
                matched_preference_ids=(),
                unsupported_preference_ids=(),
                conflict_preference_ids=(),
                concerns=(),
                reason=None,
            )
            for card in request.shortlist.cards
        ]
        return QualityRankingResult(
            mode=QualityRankingMode.BGE_FALLBACK,
            deepseek_weight=self._deepseek_weight,
            fallback_reason=str(error),
            attempts=attempts,
            traces=(),
            hits=_rank_hits(hits),
        )


def _rank_hits(hits: list[QualityRankingHit]) -> tuple[QualityRankingHit, ...]:
    ordered = sorted(hits, key=lambda item: (-item.quality, item.parent_asin))
    return tuple(
        QualityRankingHit(
            parent_asin=item.parent_asin,
            rank=rank,
            shortlist_rank=item.shortlist_rank,
            bge_relevance=item.bge_relevance,
            deepseek_fit=item.deepseek_fit,
            quality=item.quality,
            verdict=item.verdict,
            matched_preference_ids=item.matched_preference_ids,
            unsupported_preference_ids=item.unsupported_preference_ids,
            conflict_preference_ids=item.conflict_preference_ids,
            concerns=item.concerns,
            reason=item.reason,
        )
        for rank, item in enumerate(ordered, start=1)
    )
