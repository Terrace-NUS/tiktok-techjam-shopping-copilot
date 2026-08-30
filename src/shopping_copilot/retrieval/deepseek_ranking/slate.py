"""Final T-aware slate selection over DeepSeek/BGE quality scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

from shopping_copilot.query_compiler import DiversityDirective

from ..dense import DenseIndex
from ..ranking import GreedyDPPSelector, VectorSlateResult
from ..vector_diversity import VectorDiversityPolicy
from .models import QualityRankingResult


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalQualitySlate:
    transparency: float
    relevance_weight: float
    result: VectorSlateResult


class TransparencyAwareDPPFinalizer:
    """Let T control slate breadth after individual product quality is known."""

    def __init__(
        self,
        *,
        index: DenseIndex,
        diversity_policy: VectorDiversityPolicy | None = None,
        directive_adjustment: float = 0.10,
    ) -> None:
        if type(index) is not DenseIndex:
            raise TypeError("index must be an exact DenseIndex")
        resolved_policy = diversity_policy or VectorDiversityPolicy()
        if type(resolved_policy) is not VectorDiversityPolicy:
            raise TypeError("diversity_policy must be an exact VectorDiversityPolicy")
        if (
            type(directive_adjustment) is not float
            or not math.isfinite(directive_adjustment)
            or not 0.0 <= directive_adjustment <= 1.0
        ):
            raise ValueError("directive_adjustment must be a finite float in [0, 1]")
        self._policy = resolved_policy
        self._directive_adjustment = directive_adjustment
        self._selector = GreedyDPPSelector(index=index)

    def select(
        self,
        ranking: QualityRankingResult,
        *,
        transparency: float,
        top_k: int = 10,
        directive: DiversityDirective = DiversityDirective.AUTO,
    ) -> FinalQualitySlate:
        if type(ranking) is not QualityRankingResult:
            raise TypeError("ranking must be an exact QualityRankingResult")
        if type(directive) is not DiversityDirective:
            raise TypeError("directive must be a DiversityDirective")
        relevance_weight = self._policy.relevance_weight(transparency)
        if directive is DiversityDirective.INCREASE:
            relevance_weight -= self._directive_adjustment
        elif directive is DiversityDirective.DECREASE:
            relevance_weight += self._directive_adjustment
        relevance_weight = float(min(1.0, max(0.0, relevance_weight)))
        return FinalQualitySlate(
            transparency=transparency,
            relevance_weight=relevance_weight,
            result=self._selector.select(
                ranking.candidates,
                top_k=top_k,
                relevance_weight=relevance_weight,
            ),
        )
