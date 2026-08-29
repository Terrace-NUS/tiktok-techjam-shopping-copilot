from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SUITE_PATH = REPOSITORY_ROOT / "config" / "retrieval" / "clarity-prompts-v0.json"
EXPECTED_SHA256 = "ed72bac1cab8c5048fb93dd132f2a96e6d419bdb03a338a1cbbc27f39ad23087"


def _suite() -> dict[str, object]:
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def test_frozen_clarity_prompt_suite_has_expected_identity() -> None:
    payload = SUITE_PATH.read_bytes()

    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256


def test_clarity_prompt_suite_shape_and_balance_are_fixed() -> None:
    suite = _suite()

    assert set(suite) == {
        "schema",
        "language",
        "authorship",
        "levels",
        "families",
        "length_controls",
        "invariance_controls",
        "diagnostics",
    }
    assert suite["schema"] == "shopping-copilot/clarity-prompt-suite/v0"
    assert suite["language"] == "en"
    assert suite["levels"] == ["vague", "focused", "specific"]

    families = suite["families"]
    assert isinstance(families, list)
    assert len(families) == 40
    assert all(
        isinstance(item, dict) and set(item) == {"id", "domain", "vague", "focused", "specific"}
        for item in families
    )
    assert Counter(str(item["domain"]) for item in families) == {
        "women_dresses": 4,
        "women_tops_outerwear": 4,
        "women_active_swim": 4,
        "women_shoes": 4,
        "men_clothing": 4,
        "men_shoes": 4,
        "jewelry": 4,
        "bags_travel": 4,
        "watches_accessories": 4,
        "kids_work_costume": 4,
    }

    length_controls = suite["length_controls"]
    assert isinstance(length_controls, list)
    assert len(length_controls) == 10
    assert all(
        isinstance(item, dict)
        and set(item)
        == {
            "id",
            "domain",
            "lower_label",
            "lower_query",
            "higher_label",
            "higher_query",
        }
        for item in length_controls
    )

    invariance_controls = suite["invariance_controls"]
    assert isinstance(invariance_controls, list)
    assert len(invariance_controls) == 10
    assert Counter(str(item["role"]) for item in invariance_controls) == {
        "calibration": 5,
        "audit": 5,
    }
    assert all(
        isinstance(item, dict)
        and set(item) == {"id", "domain", "role", "left_query", "right_query", "reason"}
        for item in invariance_controls
    )

    diagnostics = suite["diagnostics"]
    assert isinstance(diagnostics, list)
    assert len(diagnostics) == 8
    assert all(
        isinstance(item, dict) and set(item) == {"id", "kind", "query", "interpretation"}
        for item in diagnostics
    )


def test_every_prompt_and_record_id_is_unique_and_trimmed() -> None:
    suite = _suite()
    record_ids: list[str] = []
    prompts: list[str] = []

    for item in suite["families"]:  # type: ignore[union-attr]
        record_ids.append(item["id"])
        prompts.extend(item[level] for level in ("vague", "focused", "specific"))
    for item in suite["length_controls"]:  # type: ignore[union-attr]
        record_ids.append(item["id"])
        prompts.extend((item["lower_query"], item["higher_query"]))
    for item in suite["invariance_controls"]:  # type: ignore[union-attr]
        record_ids.append(item["id"])
        prompts.extend((item["left_query"], item["right_query"]))
    for item in suite["diagnostics"]:  # type: ignore[union-attr]
        record_ids.append(item["id"])
        prompts.append(item["query"])

    assert len(record_ids) == len(set(record_ids))
    assert len(prompts) == 168
    assert len(prompts) == len(set(prompts))
    assert all(value and value == value.strip() for value in record_ids + prompts)
