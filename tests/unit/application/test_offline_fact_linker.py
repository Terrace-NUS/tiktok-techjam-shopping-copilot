from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopping_copilot.application.offline.catalog import CatalogIndex
from shopping_copilot.application.offline.state import SessionState

PRODUCTS = (
    {
        "parent_asin": "A",
        "title": "Black running top",
        "features": [
            "100% Polyester",
            "Machine Wash",
            "Quick-dry; breathable design",
        ],
        "details": {"Color": "Black", "Department": "Women"},
        "description": ["A lightweight running top."],
        "categories": ["Clothing", "Women", "Running Tops"],
        "store": "Example",
        "price": 29.0,
        "rating_number": 100,
    },
    {
        "parent_asin": "B",
        "title": "Brown walking shoe",
        "features": ["Leather upper", "Wipe Clean", "½ in"],
        "details": {"Color": "Brown", "Department": "Women"},
        "description": ["A leather walking shoe."],
        "categories": ["Clothing", "Women", "Walking Shoes"],
        "store": "Example",
        "price": 59.0,
        "rating_number": 20,
    },
)


@pytest.fixture
def catalog(tmp_path: Path) -> CatalogIndex:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in PRODUCTS),
        encoding="utf-8",
    )
    return CatalogIndex(path)


def test_links_original_catalog_facts_without_prompt_markers(catalog: CatalogIndex) -> None:
    matches = catalog.link_message_facts(
        "Two details are especially important: 100% Polyester and Machine Wash.",
    )

    assert [match.text for match in matches] == ["100% Polyester", "Machine Wash"]


def test_prefers_long_original_fact_over_nested_material(catalog: CatalogIndex) -> None:
    matches = catalog.link_message_facts(
        "Please keep the Quick dry, breathable design and 100% Polyester construction.",
    )

    assert [match.text for match in matches] == [
        "Quick dry, breathable design",
        "100% Polyester",
    ]


def test_does_not_link_rare_catalog_fragment_from_common_preposition(
    catalog: CatalogIndex,
) -> None:
    matches = catalog.link_message_facts(
        "Let's go in a different direction; now I need 100% Polyester.",
    )

    assert [match.text for match in matches] == ["100% Polyester"]


def test_links_category_without_fixed_initial_prompt(catalog: CatalogIndex) -> None:
    match = catalog.link_message_category(
        "For a trip, let me explore the Women Walking Shoes category.",
    )

    assert match is not None
    assert match.text == "Women Walking Shoes"


def test_paraphrased_initial_message_updates_category(catalog: CatalogIndex) -> None:
    state = SessionState(session_id="session", profile={})

    state.observe(
        "I'm shopping within Women Walking Shoes; Leather upper is important.",
        1,
        catalog,
    )

    assert state.category == "Women Walking Shoes"
    assert [constraint.text for constraint in state.active_constraints] == ["Leather upper"]


def test_paraphrased_reply_updates_constraints(catalog: CatalogIndex) -> None:
    state = SessionState(session_id="session", profile={})
    state.last_ask = "other"

    state.observe(
        "A couple of useful details are 100% Polyester and Machine Wash.",
        2,
        catalog,
    )

    assert [constraint.text for constraint in state.active_constraints] == [
        "100% Polyester",
        "Machine Wash",
    ]
    assert state.answer_counts == {"other": 1}
    assert state.last_event == "constraint"


def test_paraphrased_override_reopens_products_and_only_erases_initial_preference(
    catalog: CatalogIndex,
) -> None:
    state = SessionState(session_id="session", profile={})
    state.observe(
        "I'm considering Women Walking Shoes, and Leather upper matters at first.",
        1,
        catalog,
    )
    state.last_ask = "other"
    state.observe("Useful details include 100% Polyester and Machine Wash.", 2, catalog)
    state.emitted_pids.update({0, 1})

    state.observe(
        "Let's go in a different direction; now I need Color: Black.",
        3,
        catalog,
    )

    assert state.scenario == "intent_override"
    assert state.override_seen is True
    assert state.emitted_pids == set()
    assert state.active_norms == {"100% polyester", "machine wash", "color: black"}
    initial = next(item for item in state.constraints if item.normalized == "leather upper")
    assert initial.active is False


def test_paraphrased_no_preference_uses_last_structured_question(catalog: CatalogIndex) -> None:
    state = SessionState(session_id="session", profile={})
    state.last_ask = "material"

    state.observe("Either is fine; use your judgement.", 2, catalog)

    assert state.scenario == "boundary"
    assert state.dont_care_attributes == {"material"}
    assert state.last_event == "boundary_no_preference"
