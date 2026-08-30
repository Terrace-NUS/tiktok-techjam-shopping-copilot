"""Typed failures for DeepSeek candidate judgement."""

from __future__ import annotations

from enum import Enum


class DeepSeekRankingErrorCode(str, Enum):
    MISSING_API_KEY = "missing_api_key"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    INVALID_TOOL_CALL = "invalid_tool_call"
    INVALID_JUDGEMENTS = "invalid_judgements"


class DeepSeekRankingError(RuntimeError):
    """One stable failure code plus a safe, optional explanation."""

    def __init__(
        self,
        code: DeepSeekRankingErrorCode,
        message: str | None = None,
    ) -> None:
        self.code = code
        super().__init__(code.value if message is None else f"{code.value}: {message}")
