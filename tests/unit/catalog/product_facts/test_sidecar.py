from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopping_copilot.catalog.product_facts import (
    PRODUCT_FACT_SIDECAR_SCHEMA,
    load_product_fact_sidecar,
    product_fact_request_from_raw_line,
)


def _catalog_row() -> dict[str, object]:
    return {
        "parent_asin": "P1",
        "title": "Black cotton walking shoe",
        "categories": ["Shoes"],
        "store": "Example",
        "features": ["Machine washable cotton upper"],
        "details": {},
        "description": [],
    }


def _write_catalog(path: Path) -> bytes:
    raw = json.dumps(_catalog_row(), ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    return raw


def _record(raw: bytes, *, evidence: str = "cotton") -> dict[str, object]:
    request = product_fact_request_from_raw_line(raw)
    return {
        "schema": PRODUCT_FACT_SIDECAR_SCHEMA,
        "parent_asin": "P1",
        "source_id": request.source_id,
        "extractor": {"model": "test-extractor"},
        "facts": [
            {
                "facet": "material",
                "value": "cotton",
                "aliases": [],
                "polarity": "present",
                "component": "upper",
                "meaning": "The upper contains cotton.",
                "evidence": evidence,
                "source_ref": "features_0",
                "confidence": 1.0,
            }
        ],
        "summary": "A black cotton walking shoe.",
    }


def _write_sidecar(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_sidecar_revalidates_grounding_and_exact_expected_set(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    raw = _write_catalog(catalog)
    sidecar = tmp_path / "cards.jsonl"
    _write_sidecar(sidecar, _record(raw))

    cards = load_product_fact_sidecar(
        sidecar,
        catalog_path=catalog,
        expected_parent_asins={"P1"},
    )

    assert tuple(cards) == ("P1",)
    assert cards["P1"].extractor_model == "test-extractor"
    assert cards["P1"].card.facts[0].value == "cotton"

    with pytest.raises(ValueError, match="differ from expected set"):
        load_product_fact_sidecar(
            sidecar,
            catalog_path=catalog,
            expected_parent_asins={"P1", "P2"},
        )


def test_sidecar_rejects_stale_source_and_ungrounded_fact(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    raw = _write_catalog(catalog)
    sidecar = tmp_path / "cards.jsonl"

    stale = _record(raw)
    stale["source_id"] = "sha256:" + "0" * 64
    _write_sidecar(sidecar, stale)
    with pytest.raises(ValueError, match="source_id is stale"):
        load_product_fact_sidecar(sidecar, catalog_path=catalog)

    _write_sidecar(sidecar, _record(raw, evidence="silk"))
    with pytest.raises(ValueError, match="no grounded facts"):
        load_product_fact_sidecar(sidecar, catalog_path=catalog)
