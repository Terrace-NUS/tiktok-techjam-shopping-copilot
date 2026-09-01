from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from shopping_copilot.application import FullApertureConfig
from starter.agent import Agent

PRODUCTS = (
    {
        "parent_asin": "A",
        "title": "Green nylon running shoe",
        "features": ["Breathable mesh upper", "Lightweight cushioning"],
        "details": {"Department": "Women", "Width": "wide"},
        "description": ["A nylon outdoor running shoe in green."],
        "categories": ["Clothing", "Shoes", "Running Shoes"],
        "store": "Example",
        "price": 49.0,
        "rating_number": 25,
    },
    {
        "parent_asin": "B",
        "title": "Red leather running shoe",
        "features": ["Leather upper", "Firm sole"],
        "details": {"Department": "Women", "Width": "narrow"},
        "description": ["A red walking shoe."],
        "categories": ["Clothing", "Shoes", "Running Shoes"],
        "store": "Example",
        "price": 79.0,
        "rating_number": 20,
    },
)


class FakeFullApertureAgent:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, dict[str, object]]] = []

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        self.reset_calls.append((session_id, user_profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        return {
            "message": "full pipeline",
            "ask_attribute": "other",
            "recommendations": ["A", "A", "B"],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        }

    def last_audit(self, session_id: str) -> dict[str, object]:
        return {"session_id": session_id, "mode": "full"}


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in PRODUCTS),
        encoding="utf-8",
    )
    return path


def test_default_mode_is_model_free_offline_aperture(catalog_path: Path) -> None:
    agent = Agent(catalog_path)

    assert agent.mode == "offline"
    agent.reset("session", {})
    response = agent.respond(
        "session",
        "I'm looking for Shoes Running Shoes, but I'm still exploring.",
        1,
        10,
    )

    assert response["ask_attribute"] == "feature"
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    assert len(response["recommendations"]) == 1
    assert set(response) == {"message", "ask_attribute", "recommendations", "usage"}


def test_full_mode_requires_explicit_api_configuration(catalog_path: Path) -> None:
    with pytest.raises(ValueError, match="requires deepseek_api_key"):
        Agent(catalog_path, mode="full")


def test_default_mode_rejects_unused_api_configuration(catalog_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid with mode='full'"):
        Agent(catalog_path, deepseek_api_key="secret")


def test_full_mode_builds_delegate_and_normalizes_contract(catalog_path: Path) -> None:
    delegate = FakeFullApertureAgent()
    config = FullApertureConfig(api_key="secret")
    with patch("starter.agent.build_full_aperture_agent", return_value=delegate) as build:
        agent = Agent(
            catalog_path,
            mode="full",
            full_config=config,
        )

    build.assert_called_once_with(catalog_path, config)
    assert agent.mode == "full"
    agent.reset("session", {"summary": "profile"})
    response = agent.respond("session", "show me shoes", 1, 10)

    assert delegate.reset_calls == [("session", {"summary": "profile"})]
    assert response == {
        "message": "full pipeline",
        "ask_attribute": "other",
        "recommendations": [{"parent_asin": "A"}, {"parent_asin": "B"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    assert agent.last_audit("session") == {
        "session_id": "session",
        "mode": "full",
    }


def test_full_shorthand_builds_config_from_api_key(catalog_path: Path) -> None:
    delegate = FakeFullApertureAgent()
    with patch("starter.agent.build_full_aperture_agent", return_value=delegate) as build:
        Agent(catalog_path, mode="full", deepseek_api_key="secret")

    built_config = build.call_args.args[1]
    assert isinstance(built_config, FullApertureConfig)
    assert built_config.api_key == "secret"
    assert built_config.product_card_sidecar is None


def test_unknown_mode_is_rejected_before_catalog_initialization(catalog_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        Agent(catalog_path, mode="hidden_benchmark")
