"""Synchronous DeepSeek native-tool adapter for candidate judgement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from shopping_copilot.providers import DeepSeekTransport, UrllibDeepSeekTransport
from shopping_copilot.providers import HttpResponse as HttpResponse

from .errors import DeepSeekRankingError, DeepSeekRankingErrorCode
from .models import (
    DeepSeekJudgementResult,
    DeepSeekRankingRequest,
    DeepSeekRankingTrace,
)
from .prompt import build_messages
from .wire import TOOL_NAME, candidate_judgement_tool, decode_candidate_judgements


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekRankingConfig:
    """Explicit request settings for one complete candidate batch."""

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
            raise ValueError("DeepSeek timeout_seconds must be positive")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("DeepSeek max_tokens must be positive")
        if type(self.temperature) not in (int, float) or not 0 <= self.temperature <= 2:
            raise ValueError("DeepSeek temperature must be between zero and two")
        if type(self.strict_tools) is not bool or type(self.disable_thinking) is not bool:
            raise TypeError("DeepSeek boolean settings must be bool values")


class DeepSeekRankingProvider:
    """Ask DeepSeek for one exact judgement per shortlisted product."""

    __slots__ = ("_api_key", "_config", "_transport")

    def __init__(
        self,
        *,
        api_key: str | None,
        config: DeepSeekRankingConfig | None = None,
        transport: DeepSeekTransport | None = None,
    ) -> None:
        if api_key is not None and type(api_key) is not str:
            raise TypeError("DeepSeek API key must be a string or None")
        self._api_key = api_key
        self._config = config or DeepSeekRankingConfig()
        self._transport = transport or UrllibDeepSeekTransport()

    def judge(
        self,
        request: DeepSeekRankingRequest,
        *,
        repair_instruction: str | None = None,
    ) -> DeepSeekJudgementResult:
        """Call Chat Completions with one forced native function invocation."""

        if type(request) is not DeepSeekRankingRequest:
            raise TypeError("request must be an exact DeepSeekRankingRequest")
        if not self._api_key or not self._api_key.strip():
            raise DeepSeekRankingError(DeepSeekRankingErrorCode.MISSING_API_KEY)
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": list(
                build_messages(request, repair_instruction=repair_instruction)
            ),
            "stream": False,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "tools": [candidate_judgement_tool(strict=self._config.strict_tools)],
            "tool_choice": {
                "type": "function",
                "function": {"name": TOOL_NAME},
            },
        }
        if self._config.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
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
            raise DeepSeekRankingError(
                DeepSeekRankingErrorCode.PROVIDER_TIMEOUT
            ) from error
        except OSError as error:
            raise DeepSeekRankingError(
                DeepSeekRankingErrorCode.PROVIDER_UNAVAILABLE
            ) from error
        self._raise_for_status(response.status)
        decoded = _decode_response_json(response.body)
        arguments, trace = _extract_tool_call(decoded)
        return DeepSeekJudgementResult(
            judgements=decode_candidate_judgements(arguments, request),
            trace=trace,
        )

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
            code = DeepSeekRankingErrorCode.PROVIDER_AUTH
        elif status == 429:
            code = DeepSeekRankingErrorCode.PROVIDER_RATE_LIMIT
        elif status >= 500:
            code = DeepSeekRankingErrorCode.PROVIDER_UNAVAILABLE
        else:
            code = DeepSeekRankingErrorCode.INVALID_PROVIDER_RESPONSE
        raise DeepSeekRankingError(code, f"HTTP status {status}")


def _decode_response_json(body: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(
            body.decode(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_PROVIDER_RESPONSE
        ) from error
    if type(decoded) is not dict:
        raise DeepSeekRankingError(
            DeepSeekRankingErrorCode.INVALID_PROVIDER_RESPONSE
        )
    return cast(dict[str, object], decoded)


def _extract_tool_call(
    response: dict[str, object],
) -> tuple[str, DeepSeekRankingTrace]:
    choices = response.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise DeepSeekRankingError(DeepSeekRankingErrorCode.INVALID_TOOL_CALL)
    choice = cast(dict[str, object], choices[0])
    message = choice.get("message")
    if type(message) is not dict:
        raise DeepSeekRankingError(DeepSeekRankingErrorCode.INVALID_TOOL_CALL)
    calls = cast(dict[str, object], message).get("tool_calls")
    if type(calls) is not list or len(calls) != 1 or type(calls[0]) is not dict:
        raise DeepSeekRankingError(DeepSeekRankingErrorCode.INVALID_TOOL_CALL)
    function = cast(dict[str, object], calls[0]).get("function")
    if type(function) is not dict:
        raise DeepSeekRankingError(DeepSeekRankingErrorCode.INVALID_TOOL_CALL)
    function_object = cast(dict[str, object], function)
    if (
        function_object.get("name") != TOOL_NAME
        or type(function_object.get("arguments")) is not str
    ):
        raise DeepSeekRankingError(DeepSeekRankingErrorCode.INVALID_TOOL_CALL)
    return cast(str, function_object["arguments"]), _provider_trace(response)


def _provider_trace(response: dict[str, object]) -> DeepSeekRankingTrace:
    usage = response.get("usage")
    usage_object = cast(dict[str, object], usage) if type(usage) is dict else {}
    return DeepSeekRankingTrace(
        response_id=_optional_string(response.get("id")),
        model=_optional_string(response.get("model")),
        prompt_tokens=_optional_nonnegative_int(usage_object.get("prompt_tokens")),
        completion_tokens=_optional_nonnegative_int(
            usage_object.get("completion_tokens")
        ),
        total_tokens=_optional_nonnegative_int(usage_object.get("total_tokens")),
    )


def _optional_string(value: object) -> str | None:
    return value if type(value) is str else None


def _optional_nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")
