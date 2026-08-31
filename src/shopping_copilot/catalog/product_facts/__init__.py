"""LLM product fact cards aligned with the user-side facet language."""

from .deepseek import DeepSeekProductFactConfig, DeepSeekProductFactProvider
from .errors import ProductFactError, ProductFactErrorCode
from .models import (
    ProductFact,
    ProductFactCard,
    ProductFactPolarity,
    ProductFactRequest,
    ProductFactResult,
    ProductFactTrace,
    ProductSourceItem,
)
from .prompt import PRODUCT_FACT_PROMPT_VERSION
from .sidecar import (
    PRODUCT_FACT_SIDECAR_SCHEMA,
    VerifiedProductFactCard,
    load_product_fact_sidecar,
)
from .source import product_fact_request_from_raw_line
from .wire import TOOL_NAME, decode_product_fact_card, product_fact_card_tool

__all__ = (
    "PRODUCT_FACT_PROMPT_VERSION",
    "PRODUCT_FACT_SIDECAR_SCHEMA",
    "TOOL_NAME",
    "DeepSeekProductFactConfig",
    "DeepSeekProductFactProvider",
    "ProductFact",
    "ProductFactCard",
    "ProductFactPolarity",
    "ProductFactRequest",
    "ProductFactResult",
    "ProductFactTrace",
    "ProductFactError",
    "ProductFactErrorCode",
    "ProductSourceItem",
    "VerifiedProductFactCard",
    "decode_product_fact_card",
    "product_fact_card_tool",
    "product_fact_request_from_raw_line",
    "load_product_fact_sidecar",
)
