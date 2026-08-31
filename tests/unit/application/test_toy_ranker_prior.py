from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from shopping_copilot.application.toy_simulator.catalog import CatalogIndex
from shopping_copilot.application.toy_simulator.ranker import AmbiguityPrior, ProductRanker
from shopping_copilot.application.toy_simulator.state import SessionState


@pytest.fixture
def catalog(tmp_path: Path) -> CatalogIndex:
    products = (
        {
            "parent_asin": "POPULAR",
            "title": "Popular item",
            "categories": ["Clothing", "Items"],
            "rating_number": 100,
        },
        {
            "parent_asin": "PUBLIC_LIKE",
            "title": "Public-like item",
            "categories": ["Clothing", "Items"],
            "rating_number": 1,
        },
        {
            "parent_asin": "OTHER",
            "title": "Other item",
            "categories": ["Clothing", "Items"],
            "rating_number": 5,
        },
    )
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    return CatalogIndex(path)


def test_strict_tie_uses_public_likeness_before_review_count(catalog: CatalogIndex) -> None:
    prior = AmbiguityPrior(scores=(0.0, 1.0, 0.2), strength=1.0, evidence_window=0.0)
    ranker = ProductRanker(catalog, ambiguity_prior=prior)

    result = ranker.rank(SessionState(session_id="session", profile={}))

    assert [catalog.asin(pid) for pid in result.pids] == ["PUBLIC_LIKE", "OTHER", "POPULAR"]


def test_prior_cannot_cross_evidence_outside_window(catalog: CatalogIndex) -> None:
    prior = AmbiguityPrior(scores=(0.0, 1.0, 0.0), strength=1.0, evidence_window=0.01)
    ranker = ProductRanker(catalog, ambiguity_prior=prior)

    ranked = ranker._apply_ambiguity_prior(
        [0, 1],
        scores=defaultdict(float, {0: 1.0, 1: 0.95}),
        constraint_hits=defaultdict(int, {0: 1, 1: 1}),
        positional_hits=defaultdict(int),
    )

    assert ranked == [0, 1]


def test_prior_can_reorder_same_signature_inside_window(catalog: CatalogIndex) -> None:
    prior = AmbiguityPrior(scores=(0.0, 1.0, 0.0), strength=0.2, evidence_window=0.1)
    ranker = ProductRanker(catalog, ambiguity_prior=prior)

    ranked = ranker._apply_ambiguity_prior(
        [0, 1],
        scores=defaultdict(float, {0: 1.0, 1: 0.95}),
        constraint_hits=defaultdict(int, {0: 1, 1: 1}),
        positional_hits=defaultdict(int),
    )

    assert ranked == [1, 0]


def test_prior_reorders_only_inside_frozen_display_depth(catalog: CatalogIndex) -> None:
    prior = AmbiguityPrior(
        scores=(0.0, 0.5, 1.0),
        strength=1.0,
        evidence_window=0.0,
        reorder_depth=2,
    )
    ranker = ProductRanker(catalog, ambiguity_prior=prior)

    ranked = ranker._apply_ambiguity_prior(
        [0, 1, 2],
        scores=defaultdict(float),
        constraint_hits=defaultdict(int),
        positional_hits=defaultdict(int),
    )

    assert ranked == [1, 0, 2]


def test_invalid_prior_is_rejected(catalog: CatalogIndex) -> None:
    with pytest.raises(ValueError, match="one score per catalog product"):
        ProductRanker(
            catalog,
            ambiguity_prior=AmbiguityPrior(
                scores=(0.5,),
                strength=0.1,
                evidence_window=0.0,
            ),
        )
