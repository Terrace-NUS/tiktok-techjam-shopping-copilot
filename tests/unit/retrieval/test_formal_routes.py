from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledQuery,
    CompiledRankingPreference,
    DiversityDirective,
    RankingReason,
)
from shopping_copilot.retrieval import (
    FacetRoute,
    ReciprocalRankFusion,
    RelativeScoreFusion,
    RetrievalRoute,
    RouteHit,
    RouteObservation,
    build_retrieval_evidence_index,
)
from shopping_copilot.session_context import (
    Commitment,
    Operator,
    PreferenceSource,
    SemanticPolarity,
)

RELEASE_ID = "sha256:" + "b" * 64
GRAPH_ID = "cg_test"


def _write_catalog(path: Path) -> str:
    rows = [
        {
            "parent_asin": "A",
            "title": "Red linen summer shoe",
            "store": "One",
            "categories": ["Shoes"],
            "features": ["Lightweight"],
            "description": ["For summer"],
            "details": {"Color": "Red", "Material": "Linen"},
        },
        {
            "parent_asin": "B",
            "title": "Red leather formal shoe",
            "store": "Two",
            "categories": ["Shoes"],
            "features": ["Formal"],
            "description": ["For ceremony"],
            "details": {"Color": "Red", "Material": "Leather"},
        },
        {
            "parent_asin": "C",
            "title": "Blue linen hat",
            "store": "Three",
            "categories": ["Hats"],
            "features": ["Lightweight"],
            "description": ["For summer"],
            "details": {"Color": "Blue", "Material": "Linen"},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
        newline="",
    )
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _preference(
    preference_id: str,
    *,
    facet: str,
    operator: Operator,
    value: str,
) -> CompiledRankingPreference:
    return CompiledRankingPreference(
        preference_id=preference_id,
        facet=facet,
        operator=operator,
        value=value,
        semantic_text=None,
        semantic_polarity=SemanticPolarity.POSITIVE,
        commitment=Commitment.SOFT,
        source=PreferenceSource.USER_EXPLICIT,
        reason=RankingReason.SOFT_COMMITMENT,
    )


def _query(
    catalog_id: str,
    *preferences: CompiledRankingPreference,
) -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=catalog_id,
        catalog_semantic_release_id=RELEASE_ID,
        category_graph_id=GRAPH_ID,
        intent_version=1,
        q_lex="summer shoe",
        q_sem="Looking for a summer shoe.",
        search_ready=True,
        hard_constraints=(),
        ranking_preferences=tuple(preferences),
        dont_care_facets=(),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def test_facet_route_uses_positive_matches_and_soft_negative_penalty(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog_id = _write_catalog(catalog)
    evidence = build_retrieval_evidence_index(
        catalog,
        catalog_id=catalog_id,
        catalog_semantic_release_id=RELEASE_ID,
        expected_parent_asins={"A", "B", "C"},
    )
    query = _query(
        catalog_id,
        _preference("red", facet="color", operator=Operator.EQ, value="red"),
        _preference(
            "no_leather",
            facet="material",
            operator=Operator.NEQ,
            value="leather",
        ),
    )

    result = FacetRoute(evidence_index=evidence).search(
        query,
        eligible_parent_asins=("A", "B", "C"),
        top_k=3,
    )

    assert result.available is True
    assert [hit.parent_asin for hit in result.hits] == ["A", "B"]
    assert result.hits[0].raw_score == pytest.approx(1.0)
    assert result.hits[1].raw_score == pytest.approx(0.65)


def test_facet_route_applies_eligibility_before_top_k(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog_id = _write_catalog(catalog)
    evidence = build_retrieval_evidence_index(
        catalog,
        catalog_id=catalog_id,
        catalog_semantic_release_id=RELEASE_ID,
        expected_parent_asins={"A", "B", "C"},
    )
    query = _query(
        catalog_id,
        _preference("red", facet="color", operator=Operator.EQ, value="red"),
    )

    result = FacetRoute(evidence_index=evidence).search(
        query,
        eligible_parent_asins=("B",),
        top_k=1,
    )

    assert [hit.parent_asin for hit in result.hits] == ["B"]


def _observation(route: RetrievalRoute, *parent_asins: str) -> RouteObservation:
    return RouteObservation(
        route=route,
        requested_top_k=3,
        available=True,
        reason=None,
        hits=tuple(
            RouteHit(parent_asin=parent_asin, rank=rank, raw_score=float(4 - rank))
            for rank, parent_asin in enumerate(parent_asins, start=1)
        ),
    )


def test_rrf_rewards_cross_route_agreement_without_mixing_raw_scores() -> None:
    fused = ReciprocalRankFusion(rank_constant=60).fuse(
        (
            _observation(RetrievalRoute.DENSE, "A", "B", "C"),
            _observation(RetrievalRoute.LEXICAL, "B", "D", "A"),
            _observation(RetrievalRoute.FACET, "B", "E", "C"),
        ),
        top_k=5,
    )

    assert fused[0].parent_asin == "B"
    assert {item.route for item in fused[0].contributions} == {
        RetrievalRoute.DENSE,
        RetrievalRoute.LEXICAL,
        RetrievalRoute.FACET,
    }
    assert [item.parent_asin for item in fused].count("B") == 1


def test_relative_score_fusion_handles_lexical_score_orientation() -> None:
    observations = (
        RouteObservation(
            route=RetrievalRoute.DENSE,
            requested_top_k=2,
            available=True,
            reason=None,
            hits=(
                RouteHit(parent_asin="A", rank=1, raw_score=0.9),
                RouteHit(parent_asin="B", rank=2, raw_score=0.5),
            ),
        ),
        RouteObservation(
            route=RetrievalRoute.LEXICAL,
            requested_top_k=2,
            available=True,
            reason=None,
            hits=(
                RouteHit(parent_asin="B", rank=1, raw_score=-10.0),
                RouteHit(parent_asin="C", rank=2, raw_score=-2.0),
            ),
        ),
    )

    fused = RelativeScoreFusion(agreement_power=1.0).fuse(observations, top_k=3)

    assert [item.parent_asin for item in fused] == ["B", "A", "C"]
    assert len(fused[0].contributions) == 2
