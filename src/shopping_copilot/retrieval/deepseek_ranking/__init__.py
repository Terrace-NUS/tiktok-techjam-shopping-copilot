"""DeepSeek candidate judgement and direction-aware shortlist."""

from .deepseek import DeepSeekRankingConfig, DeepSeekRankingProvider
from .errors import DeepSeekRankingError, DeepSeekRankingErrorCode
from .models import (
    RANKING_CONTRACT_VERSION,
    CandidateJudgement,
    CandidateVerdict,
    DeepSeekJudgementResult,
    DeepSeekRankingRequest,
    DeepSeekRankingTrace,
    QualityRankingHit,
    QualityRankingMode,
    QualityRankingResult,
    RankingCandidateCard,
    RankingShortlist,
    RankingUserProfile,
)
from .pipeline import (
    DeepSeekQualityPipeline,
    QualityPipelineResult,
    QualityPipelineTimings,
)
from .service import DEFAULT_DEEPSEEK_WEIGHT, DeepSeekQualityRanker
from .shortlist import (
    DEFAULT_PROTECTED_PER_DIRECTION,
    DEFAULT_SHORTLIST_K,
    DirectionAwareShortlister,
    compact_product_text,
)
from .slate import FinalQualitySlate, TransparencyAwareDPPFinalizer
from .wire import TOOL_NAME, candidate_judgement_tool, decode_candidate_judgements

__all__ = (
    "RANKING_CONTRACT_VERSION",
    "CandidateJudgement",
    "CandidateVerdict",
    "DeepSeekJudgementResult",
    "DeepSeekQualityPipeline",
    "DeepSeekQualityRanker",
    "DeepSeekRankingConfig",
    "DeepSeekRankingError",
    "DeepSeekRankingErrorCode",
    "DeepSeekRankingRequest",
    "DeepSeekRankingProvider",
    "DeepSeekRankingTrace",
    "DEFAULT_PROTECTED_PER_DIRECTION",
    "DEFAULT_DEEPSEEK_WEIGHT",
    "DEFAULT_SHORTLIST_K",
    "DirectionAwareShortlister",
    "FinalQualitySlate",
    "QualityRankingHit",
    "QualityRankingMode",
    "QualityRankingResult",
    "QualityPipelineResult",
    "QualityPipelineTimings",
    "RankingCandidateCard",
    "RankingShortlist",
    "RankingUserProfile",
    "TOOL_NAME",
    "TransparencyAwareDPPFinalizer",
    "candidate_judgement_tool",
    "compact_product_text",
    "decode_candidate_judgements",
)
