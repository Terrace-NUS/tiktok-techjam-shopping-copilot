"""Composition helpers for official-simulator and real-world execution."""

from .contracts import AgentDelegate, RuntimeMode
from .quality_ranking import (
    RankingFailure,
    RealWorldRankingCoordinator,
    RealWorldRankingResult,
)
from .real_world import RealWorldConfig, build_real_world_agent
from .response_generation import (
    DeterministicResponseComposer,
    ProductNarrative,
    ResponseNarrative,
)
from .toy_simulator import Agent as ToySimulatorAgent

__all__ = [
    "AgentDelegate",
    "DeterministicResponseComposer",
    "ProductNarrative",
    "RankingFailure",
    "RealWorldConfig",
    "RealWorldRankingCoordinator",
    "RealWorldRankingResult",
    "ResponseNarrative",
    "RuntimeMode",
    "ToySimulatorAgent",
    "build_real_world_agent",
]
