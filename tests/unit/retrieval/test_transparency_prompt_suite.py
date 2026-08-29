from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[3]
SUITE_PATH = ROOT / "config" / "retrieval" / "transparency-prompts-v1.json"
V0_PATH = ROOT / "config" / "retrieval" / "clarity-prompts-v0.json"
ASIN_PATTERN = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{10}(?![A-Z0-9])")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _families() -> list[dict[str, Any]]:
    families = _load(SUITE_PATH)["families"]
    assert isinstance(families, list)
    return families


def test_suite_has_the_frozen_v1_shape() -> None:
    suite = _load(SUITE_PATH)

    assert set(suite) == {"schema", "language", "authorship", "families"}
    assert suite["schema"] == "shopping-copilot/transparency-prompt-suite/v1"
    assert suite["language"] == "en"
    assert isinstance(suite["authorship"], str) and suite["authorship"].strip()
    authorship = suite["authorship"].casefold()
    assert "no target" in authorship
    assert "score" in authorship
    assert len(suite["families"]) == 24

    for family in suite["families"]:
        assert set(family) == {"id", "domain", "split", "vague", "specific"}
        assert set(family["vague"]) == {"q_lex", "q_sem"}
        assert set(family["specific"]) == {"q_lex", "q_sem"}


def test_family_ids_splits_and_domains_are_balanced() -> None:
    families = _families()
    ids = [family["id"] for family in families]
    split_counts = Counter(family["split"] for family in families)
    domain_counts = Counter(family["domain"] for family in families)
    splits_by_domain: defaultdict[str, set[str]] = defaultdict(set)
    for family in families:
        splits_by_domain[family["domain"]].add(family["split"])

    assert len(ids) == len(set(ids)) == 24
    assert split_counts == {"calibration": 12, "audit": 12}
    assert len(domain_counts) >= 8
    assert set(domain_counts.values()) == {2}
    assert all(splits == {"calibration", "audit"} for splits in splits_by_domain.values())


def test_every_query_view_is_nonempty_and_semantic_view_is_complete() -> None:
    for family in _families():
        for level in ("vague", "specific"):
            q_lex = family[level]["q_lex"]
            q_sem = family[level]["q_sem"]
            assert isinstance(q_lex, str) and q_lex == q_lex.strip() and q_lex
            assert isinstance(q_sem, str) and q_sem == q_sem.strip() and q_sem
            assert q_lex != q_sem
            assert len(q_sem) > len(q_lex)


def test_v1_ids_and_prompt_wording_do_not_repeat_v0() -> None:
    v0 = _load(V0_PATH)
    v0_ids: set[str] = set()
    v0_prompts: set[str] = set()
    prompt_keys = {
        "vague",
        "focused",
        "specific",
        "lower_query",
        "higher_query",
        "left_query",
        "right_query",
        "query",
    }
    for section_value in v0.values():
        if not isinstance(section_value, list):
            continue
        for item in section_value:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("id"), str):
                v0_ids.add(item["id"])
            for key in prompt_keys:
                value = item.get(key)
                if isinstance(value, str):
                    v0_prompts.add(value.strip().casefold())

    for family in _families():
        assert family["id"] not in v0_ids
        for level in ("vague", "specific"):
            for view in ("q_lex", "q_sem"):
                assert family[level][view].strip().casefold() not in v0_prompts


def test_suite_contains_no_target_asin_score_or_official_label_fields() -> None:
    suite = _load(SUITE_PATH)
    forbidden_field_fragments = ("asin", "target", "score", "label", "expected")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "authorship":
                    assert not any(
                        fragment in key.casefold() for fragment in forbidden_field_fragments
                    )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            assert ASIN_PATTERN.search(value) is None

    visit(suite)
