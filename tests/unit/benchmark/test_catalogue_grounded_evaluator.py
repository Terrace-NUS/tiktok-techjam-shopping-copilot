from __future__ import annotations

from pathlib import Path

from evaluator.catalogue_grounded_evaluator import _load_journeys, exploration_score


def _product(title: str, category: str, feature: str) -> dict[str, object]:
    return {
        "title": title,
        "categories": ["Clothing, Shoes & Jewelry", category],
        "features": [feature],
    }


def test_exploration_score_combines_grounded_relevance_and_directions() -> None:
    products = {
        "target": _product("Quiet commuter backpack", "Backpacks", "waterproof nylon"),
        "a": _product("Nylon work backpack", "Backpacks", "waterproof nylon"),
        "b": _product("Nylon laptop tote", "Totes", "waterproof nylon"),
        "c": _product("Nylon messenger", "Messenger Bags", "waterproof nylon"),
        "d": _product("Nylon duffel", "Duffels", "waterproof nylon"),
    }

    score, diagnostic = exploration_score(
        ["a", "b", "c", "d"],
        products=products,
        target="target",
        active_fact_tokens={"waterproof", "nylon"},
    )

    assert score == 1.0
    assert diagnostic["relevance_rate"] == 1.0
    assert diagnostic["diversity_rate"] == 1.0


def test_exploration_score_is_zero_for_an_empty_slate() -> None:
    products = {"target": _product("Backpack", "Backpacks", "nylon")}

    score, diagnostic = exploration_score(
        [],
        products=products,
        target="target",
        active_fact_tokens={"nylon"},
    )

    assert score == 0.0
    assert diagnostic["direction_count"] == 0


def test_released_catalogue_grounded_journeys_validate() -> None:
    journeys = _load_journeys(Path("benchmarks/catalogue_grounded_200/journeys.jsonl"))

    assert len(journeys) == 200
