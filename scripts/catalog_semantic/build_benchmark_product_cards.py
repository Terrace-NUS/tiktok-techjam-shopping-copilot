#!/usr/bin/env python3
"""Assemble a zero-API-token product-fact bundle for public benchmark targets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.catalog.product_facts import (  # noqa: E402
    PRODUCT_FACT_SIDECAR_SCHEMA,
    ProductFact,
    ProductFactCard,
    ProductFactPolarity,
    ProductFactRequest,
    ProductSourceItem,
    decode_product_fact_card,
    product_fact_request_from_raw_line,
)
from shopping_copilot.facet_language import FACET_LANGUAGE_VERSION  # noqa: E402

BUNDLE_SCHEMA = "shopping-copilot/benchmark-product-fact-bundle/v1"
LOCAL_EXTRACTOR = "deterministic-grounded-fallback-v1"

_MATERIAL_TERMS = (
    "ethylene vinyl acetate",
    "stainless steel",
    "sterling silver",
    "faux leather",
    "faux fur",
    "polyurethane",
    "polyester",
    "cashmere",
    "elastane",
    "spandex",
    "acrylic",
    "viscose",
    "titanium",
    "tungsten",
    "leather",
    "cotton",
    "nylon",
    "rayon",
    "modal",
    "wool",
    "rubber",
    "latex",
    "lycra",
    "silk",
    "satin",
    "denim",
    "canvas",
    "linen",
    "fleece",
    "suede",
    "ceramic",
    "plastic",
    "synthetic",
    "resin",
    "brass",
    "copper",
    "silver",
    "gold",
    "mesh",
    "pvc",
    "eva",
)
_MATERIAL_ALIASES = {
    "spandex": ("elastane", "lycra"),
    "elastane": ("spandex", "lycra"),
    "lycra": ("spandex", "elastane"),
    "ethylene vinyl acetate": ("EVA",),
    "eva": ("ethylene vinyl acetate",),
    "sterling silver": ("925 silver",),
}
_DETAIL_FACETS = {
    "brand": "brand",
    "brand name": "brand",
    "color": "color",
    "department": "department",
    "fabric": "material",
    "fabric type": "material",
    "material": "material",
    "material type": "material",
    "outer material": "material",
    "sole material": "sole_material",
    "size": "size",
    "size name": "size",
    "style": "style",
    "pattern": "style",
    "closure type": "closure_type",
    "product care instructions": "care_instruction",
    "recommended uses for product": "use_case",
}
_CARE_PATTERNS = (
    (re.compile(r"\bmachine washable\b", re.I), "machine washable", ("machine wash",)),
    (re.compile(r"\bmachine wash\b", re.I), "machine wash", ("machine washable",)),
    (re.compile(r"\bhand[- ]wash only\b", re.I), "hand wash only", ("hand wash",)),
    (re.compile(r"\bhand wash\b", re.I), "hand wash", ("hand wash only",)),
    (re.compile(r"\bdry clean(?: only)?\b", re.I), "dry clean", ("dry cleaning",)),
)
_CLOSURE_PATTERN = re.compile(
    r"\b(button|buckle|drawstring|hook|pull[ -]?on|snap|zipper|zip)\s+closure\b",
    re.I,
)


def main() -> int:
    args = _parse_args()
    target_ids = _target_ids(args.dataset)
    requests, rows = _catalog_products(args.catalog, target_ids)
    output = args.output.resolve()
    existing = _existing_cards(
        args.existing_cards,
        target_ids,
        sidecar_path=output / "product-facts.jsonl",
    )

    output.mkdir(parents=True, exist_ok=True)
    cards_dir = output / "cards" if args.write_individual_cards else None
    if cards_dir is not None:
        cards_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    reused = 0
    fallback = 0
    dropped_existing_facts = 0
    for parent_asin in sorted(target_ids):
        request = requests[parent_asin]
        record = existing.get(parent_asin)
        deterministic = _deterministic_card(request, rows[parent_asin])
        if record is not None and record.get("source_id") == request.source_id:
            model_card = _decode_existing(record, request)
            card = _merge_cards(model_card, deterministic, request=request)
            original_facts = record.get("facts")
            original_count = len(original_facts) if type(original_facts) is list else 0
            dropped_existing_facts += original_count - len(model_card.facts)
            extractor = _extractor_model(record)
            mode = _generation_mode(record)
            if mode == "deterministic_fallback":
                fallback += 1
            else:
                mode = "validated_deepseek_reuse"
                reused += 1
        else:
            card = deterministic
            extractor = LOCAL_EXTRACTOR
            mode = "deterministic_fallback"
            fallback += 1
        normalized = _record(
            request,
            card=card,
            extractor=extractor,
            generation_mode=mode,
        )
        if cards_dir is not None:
            _write_json(cards_dir / f"{parent_asin}.json", normalized)
        records.append(normalized)

    _write_jsonl(output / "product-facts.jsonl", records)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "dataset": _display_path(args.dataset),
        "catalog": _display_path(args.catalog),
        "selected_case_count": len(target_ids),
        "unique_product_count": len(records),
        "validated_deepseek_reuse_count": reused,
        "deterministic_fallback_count": fallback,
        "dropped_ungrounded_existing_fact_count": dropped_existing_facts,
        "api_call_count": 0,
        "reported_token_usage": 0,
        "scope": "known_public_benchmark_target_pool",
        "score_comparability": "diagnostic_only_target_pool_enrichment",
        "warning": (
            "Only known public benchmark targets are enriched. This bundle is for product-card "
            "diagnostics and must not be reported as a comparable 50k retrieval score."
        ),
        "product_fact_sidecar": _display_path(output / "product-facts.jsonl"),
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _target_ids(path: Path) -> frozenset[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = cast(dict[str, object], json.loads(line))
            ground_truth = row.get("ground_truth")
            if type(ground_truth) is not dict:
                raise ValueError(f"dataset row {line_number} has no ground_truth")
            parent_asin = cast(dict[str, object], ground_truth).get("parent_asin")
            if type(parent_asin) is not str or not parent_asin:
                raise ValueError(f"dataset row {line_number} has an invalid target")
            values.add(parent_asin)
    if not values:
        raise ValueError("dataset contains no targets")
    return frozenset(values)


def _catalog_products(
    path: Path,
    target_ids: frozenset[str],
) -> tuple[dict[str, ProductFactRequest], dict[str, dict[str, object]]]:
    requests: dict[str, ProductFactRequest] = {}
    rows: dict[str, dict[str, object]] = {}
    with path.open("rb") as stream:
        for raw_line in stream:
            request = product_fact_request_from_raw_line(raw_line)
            if request.parent_asin not in target_ids:
                continue
            decoded = json.loads(raw_line.decode("utf-8"))
            requests[request.parent_asin] = request
            rows[request.parent_asin] = cast(dict[str, object], decoded)
            if len(requests) == len(target_ids):
                break
    missing = sorted(target_ids - requests.keys())
    if missing:
        raise ValueError(f"catalog is missing targets: {missing[:5]!r}")
    return requests, rows


def _existing_cards(
    directories: list[Path],
    target_ids: frozenset[str],
    *,
    sidecar_path: Path,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for directory in directories:
        for parent_asin in sorted(target_ids):
            if parent_asin in result:
                continue
            path = directory / f"{parent_asin}.json"
            if not path.is_file():
                continue
            decoded: object = json.loads(path.read_text(encoding="utf-8"))
            if type(decoded) is dict:
                result[parent_asin] = cast(dict[str, object], decoded)
    if sidecar_path.is_file():
        with sidecar_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                decoded: object = json.loads(line)
                if type(decoded) is not dict:
                    continue
                record = cast(dict[str, object], decoded)
                parent_asin = record.get("parent_asin")
                if type(parent_asin) is str and parent_asin in target_ids:
                    result.setdefault(parent_asin, record)
    return result


def _decode_existing(record: dict[str, object], request: ProductFactRequest) -> ProductFactCard:
    arguments = json.dumps(
        {
            "parent_asin": request.parent_asin,
            "facts": record.get("facts"),
            "summary": record.get("summary"),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    card = decode_product_fact_card(arguments, request)
    if not card.facts:
        raise ValueError(f"existing card has no grounded facts: {request.parent_asin}")
    return card


def _merge_cards(
    model_card: ProductFactCard,
    deterministic_card: ProductFactCard,
    *,
    request: ProductFactRequest,
) -> ProductFactCard:
    """Keep rich model facts while repairing systematic source-obvious omissions."""

    facts = [
        _fact_payload(fact) for fact in model_card.facts if not _is_positive_negated_material(fact)
    ]
    facts.extend(_fact_payload(fact) for fact in deterministic_card.facts)
    arguments = json.dumps(
        {
            "parent_asin": request.parent_asin,
            "facts": facts,
            "summary": model_card.summary,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    card = decode_product_fact_card(arguments, request)
    if card.warnings:
        raise ValueError(f"merged card is not grounded: {card.warnings[0]}")
    return card


def _is_positive_negated_material(fact: ProductFact) -> bool:
    return (
        fact.facet in {"material", "fabric_type", "upper_material", "sole_material"}
        and fact.polarity is ProductFactPolarity.PRESENT
        and re.search(r"\b(?:no|not|without|non[- ]?)\b", fact.evidence, re.I) is not None
    )


def _fact_payload(fact: ProductFact) -> dict[str, object]:
    return {
        "facet": fact.facet,
        "value": fact.value,
        "aliases": list(fact.aliases),
        "polarity": fact.polarity.value,
        "component": fact.component,
        "meaning": fact.meaning,
        "evidence": fact.evidence,
        "source_ref": fact.source_ref,
        "confidence": fact.confidence,
    }


def _extractor_model(record: dict[str, object]) -> str:
    extractor = record.get("extractor")
    if type(extractor) is dict:
        model = cast(dict[str, object], extractor).get("model")
        if type(model) is str and model.strip():
            return model
    return "unknown-reused-extractor"


def _generation_mode(record: dict[str, object]) -> str:
    extractor = record.get("extractor")
    if type(extractor) is dict:
        mode = cast(dict[str, object], extractor).get("generation_mode")
        if mode == "deterministic_fallback":
            return mode
    return "validated_deepseek_reuse"


def _deterministic_card(
    request: ProductFactRequest,
    row: dict[str, object],
) -> ProductFactCard:
    sources = {item.ref: item for item in request.sources}
    raw_facts: list[dict[str, object]] = []

    title = _source(sources, "title")
    if title is not None:
        raw_facts.append(_fact("product_name", title.text, title))

    categories = tuple(item for item in request.sources if item.field == "categories")
    if categories:
        path = " > ".join(item.text for item in categories)
        raw_facts.append(_fact("category", path, categories[0], aliases=(categories[-1].text,)))
        raw_facts.append(_fact("product_type", categories[-1].text, categories[-1]))

    store = _source(sources, "store")
    if store is not None:
        raw_facts.append(_fact("brand", store.text, store))

    for item in request.sources:
        if item.field == "features":
            value = _concise(item.text, 220)
            raw_facts.append(
                _fact(
                    "feature",
                    value,
                    item,
                    meaning=f"Product feature: {value}",
                )
            )
        elif item.field == "description" and item.text.strip():
            value = _concise(item.text, 260)
            raw_facts.append(
                _fact(
                    "feature",
                    value,
                    item,
                    evidence=value,
                    meaning=f"Product description evidence: {value}",
                    confidence=0.9,
                )
            )

    for item in request.sources:
        if item.field != "details":
            continue
        key, separator, value = item.text.partition(":")
        if not separator or not value.strip():
            continue
        facet = _DETAIL_FACETS.get(" ".join(key.casefold().split()))
        if facet is not None:
            raw_facts.append(_fact(facet, value.strip(), item))

    _append_material_facts(raw_facts, request.sources)
    _append_care_and_closure_facts(raw_facts, request.sources)
    _append_gender_fact(raw_facts, request.sources)

    for field, facet in (
        ("price", "price"),
        ("average_rating", "average_rating"),
        ("rating_number", "rating_number"),
    ):
        item = _source(sources, field)
        if item is not None:
            raw_facts.append(_fact(facet, item.text, item))

    summary = _deterministic_summary(row)
    arguments = json.dumps(
        {
            "parent_asin": request.parent_asin,
            "facts": raw_facts,
            "summary": summary,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    card = decode_product_fact_card(arguments, request)
    if card.warnings:
        raise ValueError(f"deterministic card is not grounded: {card.warnings[0]}")
    return card


def _append_material_facts(
    facts: list[dict[str, object]],
    sources: tuple[ProductSourceItem, ...],
) -> None:
    seen: set[tuple[str, str]] = set()
    for item in sources:
        if item.field not in {"title", "features", "description", "details"}:
            continue
        for term in _MATERIAL_TERMS:
            match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", item.text, flags=re.I)
            if match is None:
                continue
            key = (term, item.ref)
            if key in seen:
                continue
            seen.add(key)
            preceding = item.text[max(0, match.start() - 32) : match.start()]
            absent = re.search(r"\b(?:no|not|without|non[- ]?)\s*(?:virgin\s+)?$", preceding, re.I)
            component = _component(item.text, match.start())
            facts.append(
                _fact(
                    "material",
                    term,
                    item,
                    evidence=match.group(0),
                    aliases=_MATERIAL_ALIASES.get(term, ()),
                    polarity="absent" if absent else "present",
                    component=component,
                    meaning=(
                        f"The product explicitly excludes {term}."
                        if absent
                        else f"The product contains or uses {term}."
                    ),
                )
            )


def _append_care_and_closure_facts(
    facts: list[dict[str, object]],
    sources: tuple[ProductSourceItem, ...],
) -> None:
    for item in sources:
        for pattern, value, aliases in _CARE_PATTERNS:
            match = pattern.search(item.text)
            if match is not None:
                facts.append(
                    _fact(
                        "care_instruction",
                        value,
                        item,
                        evidence=match.group(0),
                        aliases=aliases,
                    )
                )
        imported = re.search(r"\bimported\b|进口", item.text, flags=re.I)
        if imported is None and item.field == "features":
            imported = re.fullmatch(r"\ufffd{2,}", item.text.strip())
        if imported is not None:
            facts.append(
                _fact(
                    "origin",
                    "imported",
                    item,
                    evidence=imported.group(0),
                    aliases=(),
                )
            )
        closure = _CLOSURE_PATTERN.search(item.text)
        if closure is not None:
            value = " ".join(closure.group(0).split())
            aliases = ("button fastening",) if closure.group(1).casefold() == "button" else ()
            facts.append(
                _fact(
                    "closure_type",
                    value,
                    item,
                    evidence=closure.group(0),
                    aliases=aliases,
                )
            )


def _append_gender_fact(
    facts: list[dict[str, object]],
    sources: tuple[ProductSourceItem, ...],
) -> None:
    for item in sources:
        if item.field not in {"title", "categories", "details"}:
            continue
        normalized = item.text.casefold()
        for value, markers in (
            ("women", ("women", "womens", "woman", "girls")),
            ("men", (" men ", "mens", "men's", "boys")),
            ("unisex", ("unisex",)),
        ):
            marker = next(
                (candidate for candidate in markers if candidate in f" {normalized} "), None
            )
            if marker is None:
                continue
            evidence_match = re.search(re.escape(marker.strip()), item.text, re.I)
            if evidence_match is not None:
                facts.append(
                    _fact(
                        "gender",
                        value,
                        item,
                        evidence=evidence_match.group(0),
                    )
                )
                return


def _fact(
    facet: str,
    value: str,
    source: ProductSourceItem,
    *,
    evidence: str | None = None,
    aliases: tuple[str, ...] = (),
    polarity: str = "present",
    component: str | None = None,
    meaning: str | None = None,
    confidence: float = 1.0,
) -> dict[str, object]:
    return {
        "facet": facet,
        "value": value,
        "aliases": list(aliases),
        "polarity": polarity,
        "component": component,
        "meaning": meaning or f"Product {facet.replace('_', ' ')}: {value}",
        "evidence": source.text if evidence is None else evidence,
        "source_ref": source.ref,
        "confidence": confidence,
    }


def _component(text: str, position: int) -> str | None:
    context = text[max(0, position - 45) : position + 45].casefold()
    for component in ("sole", "upper", "lining", "cups", "crotch", "band", "frame"):
        if component in context:
            return component
    return None


def _deterministic_summary(row: dict[str, object]) -> str:
    title = str(row.get("title", "")).strip()
    categories = row.get("categories")
    category = (
        " > ".join(str(value) for value in cast(list[object], categories))
        if type(categories) is list
        else ""
    )
    store = row.get("store")
    features = row.get("features")
    feature_values = (
        [str(value).strip() for value in cast(list[object], features) if str(value).strip()]
        if type(features) is list
        else []
    )
    descriptions = row.get("description")
    description_values = (
        [str(value).strip() for value in cast(list[object], descriptions) if str(value).strip()]
        if type(descriptions) is list
        else []
    )
    parts = [title]
    if category:
        parts.append(f"Category: {category}.")
    if type(store) is str and store.strip():
        parts.append(f"Brand/store: {store.strip()}.")
    highlights = [*feature_values[:5], *description_values[:1]]
    if highlights:
        parts.append("Product evidence: " + " | ".join(highlights))
    if row.get("price") is not None:
        parts.append(f"Price: {row['price']}.")
    if row.get("average_rating") is not None:
        parts.append(f"Average rating: {row['average_rating']}.")
    return _concise(" ".join(parts), 1_600)


def _record(
    request: ProductFactRequest,
    *,
    card: ProductFactCard,
    extractor: str,
    generation_mode: str,
) -> dict[str, object]:
    return {
        "schema": PRODUCT_FACT_SIDECAR_SCHEMA,
        "parent_asin": request.parent_asin,
        "source_id": request.source_id,
        "extractor": {
            "model": extractor,
            "prompt_version": "product_fact_card_v1_1",
            "facet_language_version": FACET_LANGUAGE_VERSION,
            "generation_mode": generation_mode,
        },
        "facts": [
            {
                "facet": fact.facet,
                "value": fact.value,
                "aliases": list(fact.aliases),
                "polarity": fact.polarity.value,
                "component": fact.component,
                "meaning": fact.meaning,
                "evidence": fact.evidence,
                "source_ref": fact.source_ref,
                "confidence": fact.confidence,
            }
            for fact in card.facts
        ],
        "summary": card.summary,
        "warnings": [],
        "trace": {
            "response_id": None,
            "model": extractor,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _source(
    sources: dict[str, ProductSourceItem],
    ref: str,
) -> ProductSourceItem | None:
    return sources.get(ref)


def _concise(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip(" ,;:|-")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    temporary.replace(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--existing-cards",
        action="append",
        type=Path,
        default=None,
        help="existing cards directory; first directory wins",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/benchmark_product_cards/public_200_v1",
    )
    parser.add_argument(
        "--write-individual-cards",
        action="store_true",
        help="also write one inspectable JSON file per product",
    )
    args = parser.parse_args()
    if args.existing_cards is None:
        args.existing_cards = [
            ROOT / "artifacts/catalog-semantic/product-facts-v1-1-expanded/cards",
            ROOT / "artifacts/catalog-semantic/product-facts-v1-1-full/cards",
        ]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
