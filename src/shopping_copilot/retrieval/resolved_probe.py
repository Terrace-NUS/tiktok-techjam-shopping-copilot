"""Thin composition of hard-mask resolution and the fixed multi-view Probe."""

from __future__ import annotations

from dataclasses import dataclass

from shopping_copilot.query_compiler import CompiledQuery

from .hard_mask import HardMaskResolver, ResolvedHardMask
from .multi_probe import CompiledProbeRun, CompiledProbeRunner


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedCompiledProbeRun:
    """Keep the applied mask decision beside the Probe evidence it produced."""

    mask_resolution: ResolvedHardMask
    probe_run: CompiledProbeRun


class ResolvedCompiledProbeRunner:
    """Resolve hard constraints once, then run every Probe view under that mask."""

    __slots__ = ("_probe_runner", "_resolver")

    def __init__(
        self,
        *,
        resolver: HardMaskResolver,
        probe_runner: CompiledProbeRunner,
    ) -> None:
        if type(resolver) is not HardMaskResolver:
            raise TypeError("resolver must be an exact HardMaskResolver")
        if type(probe_runner) is not CompiledProbeRunner:
            raise TypeError("probe_runner must be an exact CompiledProbeRunner")
        self._resolver = resolver
        self._probe_runner = probe_runner

    def run(self, query: CompiledQuery) -> ResolvedCompiledProbeRun:
        """Resolve once and pass the resulting mask and relaxation flag unchanged."""

        resolution = self._resolver.resolve(query)
        probe_run = self._probe_runner.run(
            query,
            eligible_mask=resolution.eligible_mask,
            hard_filter_relaxed=resolution.hard_filter_relaxed,
        )
        return ResolvedCompiledProbeRun(
            mask_resolution=resolution,
            probe_run=probe_run,
        )
