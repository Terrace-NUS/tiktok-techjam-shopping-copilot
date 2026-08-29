"""Minimal fixed-Probe entry point consuming a compiled semantic query."""

from __future__ import annotations

from dataclasses import dataclass

from shopping_copilot.query_compiler import CompiledQuery

from .dense import DenseEligibilityMask, DenseRetriever, DenseSearchResult
from .errors import CompiledQueryBindingError, CompiledQueryNotSearchableError
from .probe import DenseProbeObservation, FixedDenseProbe


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledDenseProbeRun:
    """One score snapshot and observation produced under the same fixed policy."""

    query: CompiledQuery
    ranking: DenseSearchResult
    observation: DenseProbeObservation


class CompiledDenseProbeRunner:
    """Run a C-independent dense Probe with a construction-time fixed depth."""

    __slots__ = ("_probe", "_probe_k", "_retriever")

    def __init__(self, *, retriever: DenseRetriever, probe_k: int = 40) -> None:
        if type(retriever) is not DenseRetriever:
            raise TypeError("retriever must be an exact DenseRetriever")
        if type(probe_k) is not int or probe_k <= 0:
            raise ValueError("probe_k must be a positive integer")
        self._retriever = retriever
        self._probe = FixedDenseProbe(retriever.index)
        self._probe_k = probe_k

    @property
    def probe_k(self) -> int:
        return self._probe_k

    def run(
        self,
        query: CompiledQuery,
        *,
        eligible_mask: DenseEligibilityMask | None = None,
    ) -> CompiledDenseProbeRun:
        """Score ``q_sem`` once, after any caller-resolved hard eligibility mask."""

        if type(query) is not CompiledQuery:
            raise TypeError("query must be an exact CompiledQuery")
        if not query.search_ready:
            raise CompiledQueryNotSearchableError("compiled query contains no searchable intent")
        manifest = self._retriever.index.manifest
        if (
            query.catalog_id != manifest.catalog_id
            or query.catalog_semantic_release_id != manifest.catalog_semantic_release_id
        ):
            raise CompiledQueryBindingError(
                "compiled query and dense index use different catalog bindings"
            )
        ranking = self._retriever.search_with_scores(
            query.q_sem,
            top_k=self._probe_k,
            eligible_mask=eligible_mask,
        )
        observation = self._probe.observe(ranking, probe_k=self._probe_k)
        return CompiledDenseProbeRun(
            query=query,
            ranking=ranking,
            observation=observation,
        )
