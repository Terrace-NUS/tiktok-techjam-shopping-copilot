"""Stable errors for product-fact provider and decoding failures."""

from __future__ import annotations

from enum import Enum


class ProductFactErrorCode(str, Enum):
    MISSING_API_KEY = "missing_api_key"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    INVALID_TOOL_CALL = "invalid_tool_call"
    INVALID_FACT_CARD = "invalid_fact_card"


class ProductFactError(RuntimeError):
    def __init__(self, code: ProductFactErrorCode, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code.value)
