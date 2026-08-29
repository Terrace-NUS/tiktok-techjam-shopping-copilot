"""Compiled fixed multi-view Probe joining lexical and semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from shopping_copilot.catalog.semantic.canonical import content_id_for_value
from shopping_copilot.query_compiler import CompiledQuery
from shopping_copilot.session_context import CandidateMode, SearchBelief

from .dense import DenseEligibilityMask, DenseIndex, DenseRetriever, DenseSearchResult
from .errors import CompiledQueryBindingError, CompiledQueryNotSearchableError
from .lexical import LEXICAL_PROBE_K, LexicalProbe, LexicalProbeObservation
from .modes import (
    DEFAULT_MODE_SIMILARITY_THRESHOLD,
    SemanticModeObservation,
    SemanticModeProbe,
)
from .transparency import (
    TransparencyCalibration,
    TransparencyEstimate,
    TransparencyEstimator,
    TransparencyEvidence,
    project_search_belief,
)

MULTI_PROBE_SCHEMA: Literal["shopping-copilot/fixed-multiview-probe/v1"] = (
    "shopping-copilot/fixed-multiview-probe/v1"
)
MULTI_PROBE_POLICY_ID = "fixed_multiview_probe_v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeSnapshot:
    """Reproducible raw evidence emitted before transparency calibration."""

    schema: Literal["shopping-copilot/fixed-multiview-probe/v1"]
    probe_id: str
    probe_policy_id: str
    compiled_query_digest: str
    eligibility_digest: str
    intent_version: int
    catalog_id: str
    catalog_semantic_release_id: str
    dense_index_id: str
    probe_k: int
    eligible_count: int
    mode_threshold: float
    lexical: LexicalProbeObservation
    semantic: SemanticModeObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledProbeRun:
    """One shared dense ranking, raw snapshot, estimate, and belief projection."""

    query: CompiledQuery
    ranking: DenseSearchResult
    snapshot: ProbeSnapshot
    estimate: TransparencyEstimate
    search_belief: SearchBelief


class CompiledProbeRunner:
    """Run a C-independent lexical + semantic Probe under one bound mask."""

    __slots__ = (
        "_estimator",
        "_lexical",
        "_mode_probe",
        "_mode_threshold",
        "_probe_k",
        "_retriever",
    )

    def __init__(
        self,
        *,
        retriever: DenseRetriever,
        lexical_probe: LexicalProbe,
        calibration: TransparencyCalibration,
        probe_k: int = LEXICAL_PROBE_K,
        mode_threshold: float = DEFAULT_MODE_SIMILARITY_THRESHOLD,
    ) -> None:
        if type(retriever) is not DenseRetriever:
            raise TypeError("retriever must be an exact DenseRetriever")
        if type(lexical_probe) is not LexicalProbe:
            raise TypeError("lexical_probe must be an exact LexicalProbe")
        if type(probe_k) is not int or probe_k <= 0:
            raise ValueError("probe_k must be a positive integer")
        if lexical_probe.probe_k != probe_k:
            raise ValueError("lexical Probe depth must equal the multi-view Probe depth")
        if lexical_probe.parent_asins != frozenset(retriever.index.parent_asins):
            raise CompiledQueryBindingError(
                "lexical Probe and dense index contain different catalog products"
            )
        if type(mode_threshold) is not float or not 0.0 <= mode_threshold <= 1.0:
            raise ValueError("mode_threshold must be a float between zero and one")

        self._retriever = retriever
        self._lexical = lexical_probe
        self._mode_probe = SemanticModeProbe(retriever.index)
        self._estimator = TransparencyEstimator(calibration)
        self._probe_k = probe_k
        self._mode_threshold = mode_threshold

    @property
    def probe_k(self) -> int:
        return self._probe_k

    @property
    def mode_threshold(self) -> float:
        return self._mode_threshold

    @property
    def dense_index(self) -> DenseIndex:
        """Expose the verified index binding needed by the hard-mask resolver."""

        return self._retriever.index

    def run(
        self,
        query: CompiledQuery,
        *,
        eligible_mask: DenseEligibilityMask | None = None,
        hard_filter_relaxed: bool = False,
    ) -> CompiledProbeRun:
        """Observe one compiled query; the resulting ``C_t`` never controls this run."""

        self._validate_query(query)
        if type(hard_filter_relaxed) is not bool:
            raise TypeError("hard_filter_relaxed must be a boolean")

        ranking = self._retriever.search_with_scores(
            query.q_sem,
            top_k=self._probe_k,
            eligible_mask=eligible_mask,
        )
        eligible_parent_asins = self._eligible_parent_asins(eligible_mask)
        lexical = self._lexical.observe(
            query.q_lex,
            eligible_parent_asins=eligible_parent_asins,
        )
        semantic = self._mode_probe.observe(
            ranking,
            probe_k=self._probe_k,
            threshold=self._mode_threshold,
        )
        snapshot = self._snapshot(
            query=query,
            eligible_parent_asins=eligible_parent_asins,
            lexical=lexical,
            semantic=semantic,
        )
        estimate = self._estimator.estimate(
            _transparency_evidence(
                snapshot,
                hard_filter_relaxed=hard_filter_relaxed,
            )
        )
        belief = replace(
            project_search_belief(estimate),
            candidate_modes=_candidate_modes(semantic),
        )
        return CompiledProbeRun(
            query=query,
            ranking=ranking,
            snapshot=snapshot,
            estimate=estimate,
            search_belief=belief,
        )

    def _validate_query(self, query: CompiledQuery) -> None:
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
                "compiled query and Probe indexes use different catalog bindings"
            )

    def _eligible_parent_asins(
        self,
        eligible_mask: DenseEligibilityMask | None,
    ) -> tuple[str, ...]:
        index = self._retriever.index
        if eligible_mask is None:
            return index.parent_asins
        index._require_eligibility_mask(eligible_mask)
        return tuple(
            parent_asin
            for parent_asin, eligible in zip(
                index.parent_asins,
                eligible_mask.values,
                strict=True,
            )
            if bool(eligible)
        )

    def _snapshot(
        self,
        *,
        query: CompiledQuery,
        eligible_parent_asins: tuple[str, ...],
        lexical: LexicalProbeObservation,
        semantic: SemanticModeObservation,
    ) -> ProbeSnapshot:
        compiled_query_digest = content_id_for_value(query)
        eligibility_digest = content_id_for_value(eligible_parent_asins)
        identity = {
            "schema": MULTI_PROBE_SCHEMA,
            "probe_policy_id": MULTI_PROBE_POLICY_ID,
            "compiled_query_digest": compiled_query_digest,
            "eligibility_digest": eligibility_digest,
            "dense_index_id": self._retriever.index.index_id,
            "probe_k": self._probe_k,
            "mode_threshold": self._mode_threshold,
        }
        return ProbeSnapshot(
            schema=MULTI_PROBE_SCHEMA,
            probe_id=content_id_for_value(identity),
            probe_policy_id=MULTI_PROBE_POLICY_ID,
            compiled_query_digest=compiled_query_digest,
            eligibility_digest=eligibility_digest,
            intent_version=query.intent_version,
            catalog_id=query.catalog_id,
            catalog_semantic_release_id=query.catalog_semantic_release_id,
            dense_index_id=self._retriever.index.index_id,
            probe_k=self._probe_k,
            eligible_count=len(eligible_parent_asins),
            mode_threshold=self._mode_threshold,
            lexical=lexical,
            semantic=semantic,
        )


def _transparency_evidence(
    snapshot: ProbeSnapshot,
    *,
    hard_filter_relaxed: bool,
) -> TransparencyEvidence:
    lexical = snapshot.lexical
    semantic = snapshot.semantic
    lexical_token_coverage = (
        lexical.matched_token_count / len(lexical.tokens)
        if lexical.available and lexical.tokens
        else None
    )
    return TransparencyEvidence(
        probe_id=snapshot.probe_id,
        intent_version=snapshot.intent_version,
        probe_k=snapshot.probe_k,
        eligible_count=snapshot.eligible_count,
        dense_hits=tuple(hit.parent_asin for hit in semantic.hits),
        lexical_hits=tuple(hit.parent_asin for hit in lexical.hits),
        listing_coherence=semantic.raw_listing_coherence.debiased_pairwise_cosine,
        mode_coherence=semantic.equal_mode_coherence.debiased_pairwise_cosine,
        mode_count=len(semantic.modes),
        largest_mode_share=semantic.largest_mode_share,
        effective_mode_count=semantic.effective_mode_count,
        duplicate_warning=semantic.duplicate_concentration_warning,
        lexical_available=lexical.available,
        lexical_token_coverage=lexical_token_coverage,
        lexical_mean_normalized_idf=lexical.mean_normalized_idf,
        hard_filter_relaxed=hard_filter_relaxed,
    )


def _candidate_modes(observation: SemanticModeObservation) -> tuple[CandidateMode, ...]:
    observed_count = len(observation.hits)
    if observed_count == 0:
        return ()
    modes = (
        CandidateMode(
            id=mode.id,
            label=f"Semantic mode led by {mode.leader_id}",
            mass=mode.size / observed_count,
            representative_ids=mode.representative_ids,
        )
        for mode in observation.modes
    )
    return tuple(sorted(modes, key=lambda mode: (-mode.mass, mode.id)))
