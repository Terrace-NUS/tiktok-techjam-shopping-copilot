"""Composition helpers for APERTURE's offline and full execution profiles."""

from .contracts import AgentDelegate, RuntimeMode
from .full import FullApertureConfig, build_full_aperture_agent
from .offline import OfflineApertureAgent
from .quality_ranking import (
    ApertureRankingCoordinator,
    ApertureRankingResult,
    RankingFailure,
)
from .response_generation import (
    DeterministicResponseComposer,
    ProductNarrative,
    ResponseNarrative,
)

__all__ = [
    "AgentDelegate",
    "ApertureRankingCoordinator",
    "ApertureRankingResult",
    "DeterministicResponseComposer",
    "FullApertureConfig",
    "OfflineApertureAgent",
    "ProductNarrative",
    "RankingFailure",
    "ResponseNarrative",
    "RuntimeMode",
    "build_full_aperture_agent",
]
