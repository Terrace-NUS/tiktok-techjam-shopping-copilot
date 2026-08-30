"""Synchronous DeepSeek native-tool provider for catalog product facts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from shopping_copilot.providers import DeepSeekTransport, UrllibDeepSeekTransport

from .errors import ProductFactError, ProductFactErrorCode
from .models import ProductFactRequest, ProductFactResult, ProductFactTrace
from .prompt import build_messages
from .wire import TOOL_NAME, decode_product_fact_card, product_fact_card_tool


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekProductFactConfig:
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 90.0
    max_tokens: int = 8192
    temperature: float = 0.0
    strict_tools: bool = False
    disable_thinking: bool = True

    def __post_init__(self) -> None:
        if type(self.model) is not str or not self.model.strip():
            raise ValueError("DeepSeek model must be non-empty")
        if type(self.base_url) is not str or not self.base_url.startswith("https://"):
            raise ValueError("DeepSeek base_url must be HTTPS")
        if type(self.timeout_seconds) not in (int, float) or self.timeout_seconds <= 0:
            raise ValueError("DeepSeek timeout must be positive")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("DeepSeek max_tokens must be positive")
        if type(self.temperature) not in (int, float) or not 0 <= self.temperature <= 2:
            raise ValueError("DeepSeek temperature must be between zero and two")


class DeepSeekProductFactProvider:
    __slots__ = ("_api_key", "_config", "_transport")

    def __init__(
        self,
        *,
        api_key: str | None,
        config: DeepSeekProductFactConfig | None = None,
        transport: DeepSeekTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._config = config or DeepSeekProductFactConfig()
        self._transport = transport or UrllibDeepSeekTransport()

    def extract(
        self,
        request: ProductFactRequest,
        *,
        repair_instruction: str | None = None,
    ) -> ProductFactResult:
        if not self._api_key or not self._api_key.strip():
            raise ProductFactError(ProductFactErrorCode.MISSING_API_KEY)
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": list(build_messages(request, repair_instruction=repair_instruction)),
            "stream": False,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "tools": [product_fact_card_tool(strict=self._config.strict_tools)],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }
        if self._config.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            response = self._transport.post_json(
                url=self._endpoint(),
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                body=body,
                timeout_seconds=float(self._config.timeout_seconds),
            )
        except TimeoutError as error:
            raise ProductFactError(ProductFactErrorCode.PROVIDER_TIMEOUT) from error
        except OSError as error:
            raise ProductFactError(ProductFactErrorCode.PROVIDER_UNAVAILABLE) from error
        self._raise_for_status(response.status)
        decoded = _decode_response(response.body)
        arguments, trace = _extract_tool_call(decoded)
        try:
            card = decode_product_fact_card(arguments, request)
        except (TypeError, ValueError) as error:
            raise ProductFactError(
                ProductFactErrorCode.INVALID_FACT_CARD,
                str(error),
            ) from error
        return ProductFactResult(card=card, trace=trace)

    def _endpoint(self) -> str:
        base = self._config.base_url.rstrip("/")
        if self._config.strict_tools and base == "https://api.deepseek.com":
            base = f"{base}/beta"
        return f"{base}/chat/completions"

    @staticmethod
    def _raise_for_status(status: int) -> None:
        if status == 200:
            return
        if status in (401, 403):
            code = ProductFactErrorCode.PROVIDER_AUTH
        elif status == 429:
            code = ProductFactErrorCode.PROVIDER_RATE_LIMIT
        elif status >= 500:
            code = ProductFactErrorCode.PROVIDER_UNAVAILABLE
        else:
            code = ProductFactErrorCode.INVALID_PROVIDER_RESPONSE
        raise ProductFactError(code, f"DeepSeek HTTP status {status}")


def _decode_response(body: bytes) -> dict[str, object]:
    try:
        decoded: object = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ProductFactError(ProductFactErrorCode.INVALID_PROVIDER_RESPONSE) from error
    if type(decoded) is not dict:
        raise ProductFactError(ProductFactErrorCode.INVALID_PROVIDER_RESPONSE)
    return cast(dict[str, object], decoded)


def _extract_tool_call(response: dict[str, object]) -> tuple[str, ProductFactTrace]:
    choices = response.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ProductFactError(ProductFactErrorCode.INVALID_TOOL_CALL)
    message = cast(dict[str, object], choices[0]).get("message")
    if type(message) is not dict:
        raise ProductFactError(ProductFactErrorCode.INVALID_TOOL_CALL)
    calls = cast(dict[str, object], message).get("tool_calls")
    if type(calls) is not list or len(calls) != 1 or type(calls[0]) is not dict:
        raise ProductFactError(ProductFactErrorCode.INVALID_TOOL_CALL)
    function = cast(dict[str, object], calls[0]).get("function")
    if type(function) is not dict:
        raise ProductFactError(ProductFactErrorCode.INVALID_TOOL_CALL)
    function_object = cast(dict[str, object], function)
    if (
        function_object.get("name") != TOOL_NAME
        or type(function_object.get("arguments")) is not str
    ):
        raise ProductFactError(ProductFactErrorCode.INVALID_TOOL_CALL)
    usage = response.get("usage")
    usage_object = cast(dict[str, object], usage) if type(usage) is dict else {}
    trace = ProductFactTrace(
        response_id=_optional_string(response.get("id")),
        model=_optional_string(response.get("model")),
        prompt_tokens=_optional_int(usage_object.get("prompt_tokens")),
        completion_tokens=_optional_int(usage_object.get("completion_tokens")),
        total_tokens=_optional_int(usage_object.get("total_tokens")),
    )
    return cast(str, function_object["arguments"]), trace


def _optional_string(value: object) -> str | None:
    return value if type(value) is str else None


def _optional_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
