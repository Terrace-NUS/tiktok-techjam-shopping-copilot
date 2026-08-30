from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from shopping_copilot.catalog.product_facts.deepseek import (
    DeepSeekProductFactProvider,
)
from shopping_copilot.catalog.product_facts.models import ProductFactPolarity
from shopping_copilot.catalog.product_facts.prompt import SYSTEM_PROMPT
from shopping_copilot.catalog.product_facts.source import product_fact_request_from_raw_line
from shopping_copilot.catalog.product_facts.wire import (
    TOOL_NAME,
    decode_product_fact_card,
    product_fact_card_tool,
)
from shopping_copilot.facet_language import SHARED_FACT_EXTRACTION_RULES
from shopping_copilot.providers import HttpResponse
from shopping_copilot.query_understanding.prompt import SYSTEM_PROMPT as QU_SYSTEM_PROMPT


def _raw_product() -> bytes:
    return json.dumps(
        {
            "parent_asin": "P1",
            "title": "A full product title",
            "features": [
                "95% gossypium, 5% spandex",
                "Heel measures approximately 1.57 inches",
            ],
            "description": ["100% Cotton cups. Colors: White and Black."],
            "categories": ["Clothing", "Women's Shoes"],
            "details": {"Department": "Womens", "Outer Material": "Fabric"},
            "price": 29.99,
            "store": "Example",
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _arguments(*, evidence: str = "95% gossypium, 5% spandex") -> str:
    return json.dumps(
        {
            "parent_asin": "P1",
            "facts": [
                {
                    "facet": "material",
                    "value": "95% gossypium, 5% spandex",
                    "aliases": ["cotton", "spandex"],
                    "polarity": "present",
                    "component": None,
                    "meaning": "The product is a cotton and spandex blend.",
                    "evidence": evidence,
                    "source_ref": "features_0",
                    "confidence": 0.99,
                }
            ],
            "summary": "Cotton blend product.",
        }
    )


def _duplicate_arguments() -> str:
    document = json.loads(_arguments())
    duplicate = dict(document["facts"][0])
    duplicate["aliases"] = ["cotton fibre"]
    duplicate["evidence"] = "95% gossypium"
    document["facts"].append(duplicate)
    return json.dumps(document)


def _response(arguments: str) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=json.dumps(
            {
                "id": "response-1",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": TOOL_NAME,
                                        "arguments": arguments,
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        ).encode("utf-8"),
    )


@dataclass(slots=True)
class _RecordingTransport:
    response: HttpResponse
    body: bytes | None = None

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        del url, headers, timeout_seconds
        self.body = body
        return self.response


def test_source_projection_keeps_full_description_and_addressable_metadata() -> None:
    request = product_fact_request_from_raw_line(_raw_product())
    sources = {item.ref: item.text for item in request.sources}

    assert request.parent_asin == "P1"
    assert sources["description_0"] == "100% Cotton cups. Colors: White and Black."
    assert sources["features_0"] == "95% gossypium, 5% spandex"
    assert "Department: Womens" in sources.values()
    assert "Outer Material: Fabric" in sources.values()


def test_user_and_product_prompts_share_the_normative_fact_language() -> None:
    assert SHARED_FACT_EXTRACTION_RULES in QU_SYSTEM_PROMPT
    assert SHARED_FACT_EXTRACTION_RULES in SYSTEM_PROMPT
    assert "不需要节省 token" in SYSTEM_PROMPT


def test_tool_schema_and_decoder_keep_aliases_polarity_and_exact_citation() -> None:
    request = product_fact_request_from_raw_line(_raw_product())
    tool = product_fact_card_tool(strict=False)
    card = decode_product_fact_card(_arguments(), request)

    assert tool["type"] == "function"
    assert card.facts[0].aliases == ("cotton", "spandex")
    assert card.facts[0].polarity is ProductFactPolarity.PRESENT

    with pytest.raises(ValueError, match="no grounded facts"):
        decode_product_fact_card(_arguments(evidence="cotton blend"), request)


def test_decoder_merges_exact_fact_duplicates_without_losing_aliases() -> None:
    request = product_fact_request_from_raw_line(_raw_product())

    card = decode_product_fact_card(_duplicate_arguments(), request)

    assert len(card.facts) == 1
    assert card.facts[0].aliases == ("cotton", "spandex", "cotton fibre")
    assert card.facts[0].evidence == "95% gossypium, 5% spandex"


def test_decoder_deduplicates_aliases_inside_one_model_fact() -> None:
    request = product_fact_request_from_raw_line(_raw_product())
    document = json.loads(_arguments())
    document["facts"][0]["aliases"] = ["cotton", "Cotton", "spandex"]

    card = decode_product_fact_card(json.dumps(document), request)

    assert card.facts[0].aliases == ("cotton", "spandex")


def test_decoder_recovers_the_exact_source_span_after_whitespace_normalization() -> None:
    raw = _raw_product().replace(b"95% gossypium", "95%\u00a0gossypium".encode())
    request = product_fact_request_from_raw_line(raw)

    card = decode_product_fact_card(_arguments(), request)

    assert card.facts[0].evidence == "95%\u00a0gossypium, 5% spandex"


def test_decoder_recovers_source_case_and_unicode_quotes_without_paraphrasing() -> None:
    raw = json.dumps(
        {
            "parent_asin": "P1",
            "title": "Fishing size 17.7’’",
        },
        ensure_ascii=False,
    ).encode()
    request = product_fact_request_from_raw_line(raw)
    document = json.loads(_arguments())
    document["facts"][0]["source_ref"] = "title"
    document["facts"][0]["evidence"] = "fishing size 17.7''"

    card = decode_product_fact_card(json.dumps(document), request)

    assert card.facts[0].evidence == "Fishing size 17.7’’"


def test_decoder_repairs_source_ref_and_drops_only_ungrounded_sibling_fact() -> None:
    request = product_fact_request_from_raw_line(_raw_product())
    document = json.loads(_arguments())
    document["facts"][0]["source_ref"] = "title"
    invalid = dict(document["facts"][0])
    invalid["value"] = "invented"
    invalid["evidence"] = "not present in any source"
    document["facts"].append(invalid)

    card = decode_product_fact_card(json.dumps(document), request)

    assert len(card.facts) == 1
    assert card.facts[0].source_ref == "features_0"
    assert len(card.warnings) == 1
    assert "dropped" in card.warnings[0]


def test_deepseek_provider_uses_forced_native_tool_call_with_complete_sources() -> None:
    request = product_fact_request_from_raw_line(_raw_product())
    transport = _RecordingTransport(_response(_arguments()))
    provider = DeepSeekProductFactProvider(api_key="secret", transport=transport)

    result = provider.extract(request)

    assert result.card.parent_asin == "P1"
    assert result.trace.total_tokens == 150
    assert transport.body is not None
    payload = json.loads(transport.body)
    assert payload["tool_choice"]["function"]["name"] == TOOL_NAME
    user_payload = json.loads(payload["messages"][1]["content"])
    assert any(
        source["text"] == "100% Cotton cups. Colors: White and Black."
        for source in user_payload["sources"]
    )
