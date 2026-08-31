"""Official Agent entry point with an explicit real-world opt-in."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.application import (  # noqa: E402
    AgentDelegate,
    RealWorldConfig,
    RuntimeMode,
    ToySimulatorAgent,
    build_real_world_agent,
)

_ASK_ATTRIBUTES = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)


class Agent:
    """Expose one official API while keeping the two strategies isolated.

    ``Agent()`` is deliberately model-free and uses the official-simulator
    specialist. The API-backed full system is constructed only when the caller
    explicitly passes ``mode="real_world"`` and DeepSeek configuration.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        question_mode: str | None = None,
        *,
        mode: RuntimeMode | str = RuntimeMode.OFFICIAL_SIMULATOR,
        deepseek_api_key: str | None = None,
        real_world_config: RealWorldConfig | None = None,
    ) -> None:
        try:
            runtime_mode = RuntimeMode(mode)
        except ValueError as error:
            allowed = ", ".join(item.value for item in RuntimeMode)
            raise ValueError(f"mode must be one of: {allowed}") from error

        if runtime_mode is RuntimeMode.OFFICIAL_SIMULATOR:
            if deepseek_api_key is not None or real_world_config is not None:
                raise ValueError("DeepSeek configuration is only valid with mode='real_world'")
            delegate: AgentDelegate = ToySimulatorAgent(
                catalog_path,
                question_mode=question_mode,
            )
        else:
            if question_mode is not None:
                raise ValueError("question_mode is only valid in official_simulator mode")
            if deepseek_api_key is not None and real_world_config is not None:
                raise ValueError("pass either deepseek_api_key or real_world_config, not both")
            config = real_world_config
            if config is None:
                if deepseek_api_key is None:
                    raise ValueError(
                        "mode='real_world' requires deepseek_api_key or real_world_config"
                    )
                config = RealWorldConfig(api_key=deepseek_api_key)
            delegate = build_real_world_agent(catalog_path, config)

        self._mode = runtime_mode
        self._delegate = delegate

    @property
    def mode(self) -> str:
        """Return the selected mode for local diagnostics and tests."""

        return self._mode.value

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        self._delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        return _official_response(
            self._delegate.respond(session_id, user_message, turn, top_k),
            top_k=top_k,
        )

    def last_audit(self, session_id: str) -> dict[str, object]:
        """Expose full-pipeline audit data when the selected delegate provides it."""

        method = getattr(self._delegate, "last_audit", None)
        if not callable(method):
            raise LookupError(f"{self.mode} mode does not provide turn audits")
        result = method(session_id)
        if type(result) is not dict:
            raise TypeError("delegate last_audit() must return a dict")
        return cast(dict[str, object], result)


def _official_response(response: dict[str, object], *, top_k: int) -> dict[str, object]:
    """Normalize both delegates to the exact organizer response contract."""

    if type(response) is not dict:
        raise TypeError("delegate respond() must return a dict")
    message = response.get("message")
    ask_attribute = response.get("ask_attribute")
    raw_recommendations = response.get("recommendations")
    if type(message) is not str:
        raise TypeError("delegate response.message must be a string")
    if ask_attribute is not None and ask_attribute not in _ASK_ATTRIBUTES:
        raise ValueError("delegate returned an unsupported ask_attribute")
    if type(raw_recommendations) is not list:
        raise TypeError("delegate response.recommendations must be a list")

    recommendations: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_item in raw_recommendations:
        score: object | None = None
        if type(raw_item) is str:
            parent_asin_value: object = raw_item
        elif type(raw_item) is dict:
            item = cast(dict[str, Any], raw_item)
            parent_asin_value = item.get("parent_asin")
            score = item.get("score")
        else:
            raise TypeError("each recommendation must be a string or object")
        if type(parent_asin_value) is not str or not parent_asin_value:
            raise ValueError("each recommendation requires a non-empty parent_asin")
        parent_asin = cast(str, parent_asin_value)
        if parent_asin in seen:
            continue
        normalized: dict[str, object] = {"parent_asin": parent_asin}
        if score is not None:
            if type(score) not in (int, float):
                raise TypeError("recommendation.score must be numeric")
            normalized["score"] = float(cast(int | float, score))
        recommendations.append(normalized)
        seen.add(parent_asin)
        if len(recommendations) >= min(top_k, 100):
            break

    raw_usage = response.get("usage")
    prompt_tokens = 0
    completion_tokens = 0
    if raw_usage is not None:
        if type(raw_usage) is not dict:
            raise TypeError("delegate response.usage must be an object")
        usage = cast(dict[str, object], raw_usage)
        prompt_tokens = _token_count(usage.get("prompt_tokens", 0), "prompt_tokens")
        completion_tokens = _token_count(
            usage.get("completion_tokens", 0),
            "completion_tokens",
        )

    return {
        "message": message,
        "ask_attribute": ask_attribute,
        "recommendations": recommendations,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _token_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"usage.{name} must be a non-negative integer")
    return value
