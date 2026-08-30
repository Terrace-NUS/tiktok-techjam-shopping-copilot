"""Native function schema and grounded decoder for product fact cards."""

from __future__ import annotations

import json
import re
from typing import cast

from .models import ProductFact, ProductFactCard, ProductFactPolarity, ProductFactRequest

TOOL_NAME = "extract_product_fact_card"
_FACET_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def product_fact_card_tool(*, strict: bool) -> dict[str, object]:
    function: dict[str, object] = {
        "name": TOOL_NAME,
        "description": (
            "Extract an exhaustive, source-grounded product fact card without truncating "
            "description facts."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "parent_asin": {"type": "string"},
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "facet": {
                                "type": "string",
                                "description": "Product attribute in lower_snake_case.",
                            },
                            "value": {
                                "type": "string",
                                "description": (
                                    "Source-faithful value retaining composition, units, and qualifiers."
                                ),
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Only explicit or unambiguous shopping-equivalent names."
                                ),
                            },
                            "polarity": {
                                "type": "string",
                                "enum": [item.value for item in ProductFactPolarity],
                            },
                            "component": {"type": ["string", "null"]},
                            "meaning": {"type": "string"},
                            "evidence": {
                                "type": "string",
                                "description": "Exact contiguous quote from source_ref.",
                            },
                            "source_ref": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "facet",
                            "value",
                            "aliases",
                            "polarity",
                            "component",
                            "meaning",
                            "evidence",
                            "source_ref",
                            "confidence",
                        ],
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["parent_asin", "facts", "summary"],
        },
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def decode_product_fact_card(arguments: str, request: ProductFactRequest) -> ProductFactCard:
    try:
        decoded: object = json.loads(arguments, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("product fact tool arguments must be unique-key JSON") from error
    if type(decoded) is not dict:
        raise ValueError("product fact tool arguments must be an object")
    root = cast(dict[str, object], decoded)
    if set(root) != {"parent_asin", "facts", "summary"}:
        raise ValueError("product fact tool arguments have unexpected fields")
    if root["parent_asin"] != request.parent_asin:
        raise ValueError("product fact card parent_asin does not match request")
    if type(root["facts"]) is not list:
        raise ValueError("product fact card facts must be an array")
    source_by_ref = {item.ref: item.text for item in request.sources}
    raw_facts = cast(list[object], root["facts"])
    decoded_facts: list[ProductFact] = []
    warnings: list[str] = []
    for index, item in enumerate(raw_facts):
        try:
            decoded_facts.append(_decode_fact(item, source_by_ref=source_by_ref))
        except (TypeError, ValueError) as error:
            warnings.append(f"facts[{index}] dropped: {error}")
    if raw_facts and not decoded_facts:
        raise ValueError("product fact card contains no grounded facts")
    facts = _deduplicate_facts(tuple(decoded_facts))
    summary = root["summary"]
    if type(summary) is not str or not summary.strip():
        raise ValueError("product fact card summary must be non-empty")
    return ProductFactCard(
        parent_asin=request.parent_asin,
        facts=facts,
        summary=summary,
        warnings=tuple(warnings),
    )


def _deduplicate_facts(facts: tuple[ProductFact, ...]) -> tuple[ProductFact, ...]:
    """Merge exact semantic duplicates while preserving the first direct citation."""

    result: list[ProductFact] = []
    index_by_key: dict[tuple[str, str, str, str | None], int] = {}
    for fact in facts:
        key = (
            fact.facet,
            " ".join(fact.value.casefold().split()),
            fact.polarity.value,
            None if fact.component is None else " ".join(fact.component.casefold().split()),
        )
        previous_index = index_by_key.get(key)
        if previous_index is None:
            index_by_key[key] = len(result)
            result.append(fact)
            continue
        previous = result[previous_index]
        aliases = _unique_aliases((*previous.aliases, *fact.aliases))
        result[previous_index] = ProductFact(
            facet=previous.facet,
            value=previous.value,
            aliases=aliases,
            polarity=previous.polarity,
            component=previous.component,
            meaning=previous.meaning,
            evidence=previous.evidence,
            source_ref=previous.source_ref,
            confidence=max(previous.confidence, fact.confidence),
        )
    return tuple(result)


def _unique_aliases(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = " ".join(value.casefold().split())
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _decode_fact(item: object, *, source_by_ref: dict[str, str]) -> ProductFact:
    if type(item) is not dict:
        raise ValueError("product fact must be an object")
    raw = cast(dict[str, object], item)
    expected = {
        "facet",
        "value",
        "aliases",
        "polarity",
        "component",
        "meaning",
        "evidence",
        "source_ref",
        "confidence",
    }
    if set(raw) != expected:
        raise ValueError("product fact has unexpected fields")
    facet = _string(raw["facet"], name="facet")
    if _FACET_PATTERN.fullmatch(facet) is None:
        raise ValueError("product fact facet must be lower_snake_case")
    aliases_raw = raw["aliases"]
    if type(aliases_raw) is not list:
        raise ValueError("product fact aliases must be an array")
    aliases = _unique_aliases(
        tuple(_string(alias, name="alias") for alias in cast(list[object], aliases_raw))
    )
    polarity_raw = _string(raw["polarity"], name="polarity")
    try:
        polarity = ProductFactPolarity(polarity_raw)
    except ValueError as error:
        raise ValueError("product fact polarity is invalid") from error
    component_raw = raw["component"]
    component = None if component_raw is None else _string(component_raw, name="component")
    source_ref = _string(raw["source_ref"], name="source_ref")
    source_text = source_by_ref.get(source_ref)
    evidence = _string(raw["evidence"], name="evidence")
    grounded_evidence = None if source_text is None else _ground_evidence(evidence, source_text)
    if grounded_evidence is None:
        for candidate_ref, candidate_text in source_by_ref.items():
            candidate_evidence = _ground_evidence(evidence, candidate_text)
            if candidate_evidence is not None:
                source_ref = candidate_ref
                grounded_evidence = candidate_evidence
                break
    if grounded_evidence is None:
        raise ValueError(
            f"product fact evidence {evidence!r} is not an exact substring of {source_ref!r}"
        )
    confidence = raw["confidence"]
    if type(confidence) not in (int, float):
        raise ValueError("product fact confidence must be numeric")
    return ProductFact(
        facet=facet,
        value=_string(raw["value"], name="value"),
        aliases=aliases,
        polarity=polarity,
        component=component,
        meaning=_string(raw["meaning"], name="meaning"),
        evidence=grounded_evidence,
        source_ref=source_ref,
        confidence=float(cast(int | float, confidence)),
    )


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"product fact {name} must be a non-empty string")
    return value


def _ground_evidence(evidence: str, source_text: str) -> str | None:
    if evidence in source_text:
        return evidence
    if not evidence.split():
        return None
    equivalents = {
        "'": "['‘’`]",
        "‘": "['‘’`]",
        "’": "['‘’`]",
        '"': '["“”]',
        "“": '["“”]',
        "”": '["“”]',
        "-": "[-‐‑‒–—]",
        "‐": "[-‐‑‒–—]",
        "‑": "[-‐‑‒–—]",
        "‒": "[-‐‑‒–—]",
        "–": "[-‐‑‒–—]",
        "—": "[-‐‑‒–—]",
    }
    pieces: list[str] = []
    in_whitespace = False
    for character in evidence:
        if character.isspace():
            if not in_whitespace:
                pieces.append(r"\s+")
                in_whitespace = True
            continue
        in_whitespace = False
        pieces.append(equivalents.get(character, re.escape(character)))
    tolerant = re.compile("".join(pieces), flags=re.IGNORECASE)
    match = tolerant.search(source_text)
    return None if match is None else match.group(0)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
