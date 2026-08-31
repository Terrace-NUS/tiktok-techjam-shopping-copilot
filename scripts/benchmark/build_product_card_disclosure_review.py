#!/usr/bin/env python3
"""Build review or full-evaluation product-card simulator disclosures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.benchmark import (  # noqa: E402
    DisclosureFact,
    DisclosurePlan,
    project_product_card_disclosures,
)
from shopping_copilot.catalog.product_facts import load_product_fact_sidecar  # noqa: E402

REVIEW_SCHEMA = "shopping-copilot/product-card-disclosure-review/v1"
KNOWN_FAILURES = frozenset(
    {"public_0041", "public_0045", "public_0098", "public_0154", "public_0199"}
)


def main() -> int:
    args = _parse_args()
    samples = _load_samples(args.dataset)
    selection = None if args.all_samples else _load_selection(args.selection)
    selected = list(samples.values()) if selection is None else _select_samples(selection, samples)
    target_ids = {_target_parent_asin(item) for item in selected}
    cards = load_product_fact_sidecar(
        args.sidecar,
        catalog_path=args.catalog,
    )
    missing_cards = target_ids - cards.keys()
    if missing_cards:
        raise ValueError(f"review targets have no product cards: {sorted(missing_cards)!r}")
    raw_cards = _load_raw_cards(args.sidecar, target_ids=target_ids)
    reasons = (
        {sample_id: "full public-set evaluation" for sample_id in samples}
        if selection is None
        else {
            str(item["sample_id"]): str(item["reason"])
            for item in cast(list[dict[str, object]], selection["samples"])
        }
    )

    reviews: list[dict[str, Any]] = []
    for sample in selected:
        sample_id = str(sample["sample_id"])
        scenario = str(sample["scenario_type"])
        target = _target_parent_asin(sample)
        plan = project_product_card_disclosures(
            cards[target].card,
            scenario_type=cast(Any, scenario),
            maximum_facts=args.maximum_facts,
        )
        transcript = _simulate_other_only(plan)
        reviews.append(
            {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "target_parent_asin": target,
                "selection_reason": reasons[sample_id],
                "known_previous_failure": sample_id in KNOWN_FAILURES,
                "source_id": cards[target].source_id,
                "extractor_model": cards[target].extractor_model,
                "plan": asdict(plan),
                "transcript": transcript,
                "full_card_file": f"cards/{sample_id}.json",
            }
        )

    _validate_reviews(
        reviews,
        expected_sample_count=len(selected),
        expected_scenario_counts=Counter(str(item["scenario_type"]) for item in selected),
    )
    _publish(args.output, reviews=reviews, raw_cards=raw_cards)
    print(json.dumps(_summary(reviews), ensure_ascii=False, indent=2), flush=True)
    print(args.output.resolve(), flush=True)
    return 0


def _simulate_other_only(plan: DisclosurePlan) -> list[dict[str, Any]]:
    facts = list(plan.disclosures)
    by_id = {item.id: item for item in facts}
    disclosed: set[str] = set()
    withdrawn: set[str] = set()
    events: list[dict[str, Any]] = []

    initial_fact: DisclosureFact | None = None
    old_override_fact: DisclosureFact | None = None
    if plan.scenario_type == "buying":
        initial_fact = next((item for item in facts if item.commitment == "hard"), None)
    elif plan.scenario_type == "intent_override":
        old_override_fact = next(
            (
                item
                for item in facts
                if item.commitment == "soft"
                and item.ask_attribute in {"style", "color", "use_case"}
            ),
            next((item for item in facts if item.commitment == "soft"), None),
        )
        if old_override_fact is None:
            old_override_fact = next(
                (item for item in reversed(facts) if item.commitment == "soft"),
                facts[-1] if facts else None,
            )
        initial_fact = old_override_fact

    initial_ids = [] if initial_fact is None else [initial_fact.id]
    disclosed.update(initial_ids)
    initial_message = f"I'm looking for {plan.product_type}, but I'm still exploring."
    if initial_fact is not None:
        initial_message = f"I'm looking for {plan.product_type}. {initial_fact.utterance}"
    events.append(
        _user_event(
            turn=1,
            kind="initial",
            message=initial_message,
            fact_ids=initial_ids,
            facts=by_id,
        )
    )

    boundary_refusal_used = False
    next_turn = 2
    while next_turn <= 7:
        events.append(
            {
                "turn": next_turn - 1,
                "role": "assistant",
                "kind": "clarification",
                "ask_attribute": "other",
                "message": "What other requirements or preferences matter to you?",
            }
        )
        if plan.scenario_type == "boundary" and not boundary_refusal_used:
            boundary_refusal_used = True
            events.append(
                _user_event(
                    turn=next_turn,
                    kind="boundary_refusal",
                    message="I don't have a preference for that yet; please use your judgment.",
                    fact_ids=[],
                    facts=by_id,
                )
            )
            next_turn += 1
            continue
        if plan.scenario_type == "intent_override" and next_turn == 3:
            new_fact = next(
                (item for item in facts if item.commitment == "hard" and item.id not in disclosed),
                next((item for item in facts if item.id not in disclosed), None),
            )
            if old_override_fact is not None:
                withdrawn.add(old_override_fact.id)
            override_ids = [] if new_fact is None else [new_fact.id]
            disclosed.update(override_ids)
            old_value = (
                "that earlier preference" if old_override_fact is None else old_override_fact.value
            )
            new_text = (
                "Please use your judgment on the rest." if new_fact is None else new_fact.utterance
            )
            events.append(
                _user_event(
                    turn=next_turn,
                    kind="intent_override",
                    message=f"Actually, don't prioritize {old_value}. {new_text}",
                    fact_ids=override_ids,
                    facts=by_id,
                    withdrawn_fact_ids=(
                        [] if old_override_fact is None else [old_override_fact.id]
                    ),
                )
            )
            next_turn += 1
            continue

        remaining = [
            item for item in facts if item.id not in disclosed and item.id not in withdrawn
        ]
        if plan.scenario_type == "intent_override" and next_turn < 3:
            remaining = [item for item in remaining if item.commitment != "hard"]
        if not remaining:
            events.append(
                _user_event(
                    turn=next_turn,
                    kind="exhausted",
                    message="I don't have any additional preferences right now.",
                    fact_ids=[],
                    facts=by_id,
                )
            )
            break
        revealed = remaining[:2]
        revealed_ids = [item.id for item in revealed]
        disclosed.update(revealed_ids)
        events.append(
            _user_event(
                turn=next_turn,
                kind="attribute_disclosure",
                message=" ".join(item.utterance for item in revealed),
                fact_ids=revealed_ids,
                facts=by_id,
            )
        )
        next_turn += 1
    return events


def _user_event(
    *,
    turn: int,
    kind: str,
    message: str,
    fact_ids: list[str],
    facts: dict[str, DisclosureFact],
    withdrawn_fact_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "role": "user",
        "kind": kind,
        "message": message,
        "disclosed_fact_ids": fact_ids,
        "disclosed_facts": [
            {
                "id": fact_id,
                "facet": facts[fact_id].facet,
                "ask_attribute": facts[fact_id].ask_attribute,
                "value": facts[fact_id].value,
            }
            for fact_id in fact_ids
        ],
        "withdrawn_fact_ids": [] if withdrawn_fact_ids is None else withdrawn_fact_ids,
    }


def _publish(
    output: Path,
    *,
    reviews: list[dict[str, Any]],
    raw_cards: dict[str, dict[str, object]],
) -> None:
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"review output exists and is not a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".product-card-review-", dir=output.parent) as temporary:
        staging = Path(temporary) / "generation"
        cards_dir = staging / "cards"
        sessions_dir = staging / "sessions"
        cards_dir.mkdir(parents=True)
        sessions_dir.mkdir()
        _write_json(staging / "review.json", {"schema": REVIEW_SCHEMA, "sessions": reviews})
        with (staging / "conversations.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for review in reviews:
                stream.write(
                    json.dumps(
                        {
                            "schema": REVIEW_SCHEMA,
                            "sample_id": review["sample_id"],
                            "scenario_type": review["scenario_type"],
                            "target_parent_asin": review["target_parent_asin"],
                            "transcript": review["transcript"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        by_target = {str(item["target_parent_asin"]): item for item in reviews}
        for target, raw_card in raw_cards.items():
            sample_id = str(by_target[target]["sample_id"])
            _write_json(cards_dir / f"{sample_id}.json", raw_card)
        for review in reviews:
            (sessions_dir / f"{review['sample_id']}.md").write_text(
                _render_session(review), encoding="utf-8", newline="\n"
            )
        (staging / "report.md").write_text(_render_report(reviews), encoding="utf-8", newline="\n")
        manifest = {
            "schema": "shopping-copilot/product-card-disclosure-review-manifest/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_count": len(reviews),
            "known_previous_failure_count": sum(
                bool(item["known_previous_failure"]) for item in reviews
            ),
            "api_calls": 0,
            "model_tokens": 0,
            "files": _file_manifest(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        if output.exists():
            _replace_review_generation(staging, output)
        else:
            os.replace(staging, output)


def _render_report(reviews: list[dict[str, Any]]) -> str:
    summary = _summary(reviews)
    lines = [
        "# Product-card disclosure · 20-session review v1",
        "",
        "This packet replaces the legacy four-string hidden intent with a bounded view over "
        "the complete grounded product card. It made zero API calls.",
        "",
        f"- Sessions: {summary['session_count']}",
        f"- Known previous failures: {summary['known_previous_failure_count']}",
        f"- Selected disclosures: {summary['selected_disclosure_count']}",
        f"- Mean disclosures per session: {summary['mean_disclosures_per_session']:.2f}",
        "- Test protocol: every assistant turn asks `ask_attribute=other`.",
        "",
        "## Selection",
        "",
        "| Sample | Scenario | Product type | Disclosures | Previous failure | Reason |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    lines[0] = f"# Product-card disclosure - {summary['session_count']}-session packet v1"
    for item in reviews:
        plan = cast(dict[str, Any], item["plan"])
        lines.append(
            f"| [{item['sample_id']}](sessions/{item['sample_id']}.md) | "
            f"{item['scenario_type']} | {plan['product_type']} | "
            f"{len(plan['disclosures'])} | "
            f"{'yes' if item['known_previous_failure'] else 'no'} | "
            f"{item['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Review questions",
            "",
            "1. Does each selected fact sound like something a shopper could care about?",
            "2. Does component-level wording avoid turning a strap or sole material into the whole product?",
            "3. Are any identifiers, model numbers, ratings, dates, or marketing boilerplate disclosed?",
            "4. Does each session expose more useful state than the legacy four-value cap?",
            "5. Are override and boundary turns understandable before this policy is moved into the benchmark?",
            "",
        ]
    )
    return "\n".join(lines)


def _render_session(review: dict[str, Any]) -> str:
    plan = cast(dict[str, Any], review["plan"])
    decisions = cast(list[dict[str, Any]], plan["decisions"])
    reason_counts = Counter(str(item["reason"]) for item in decisions if not item["selected"])
    lines = [
        f"# {review['sample_id']} · {review['scenario_type']}",
        "",
        f"- Target: `{review['target_parent_asin']}`",
        f"- Product type: {plan['product_type']}",
        f"- Selection reason: {review['selection_reason']}",
        f"- Known previous failure: {review['known_previous_failure']}",
        f"- Complete grounded card: [JSON](../{review['full_card_file']})",
        "",
        "## Card summary",
        "",
        str(plan["summary"]),
        "",
        "## Disclosure plan",
        "",
        "| Order | Commitment | ask_attribute | Facet | Component | Value | Evidence |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for index, fact in enumerate(cast(list[dict[str, Any]], plan["disclosures"]), start=1):
        lines.append(
            f"| {index} | {fact['commitment']} | {fact['ask_attribute']} | "
            f"{fact['facet']} | {fact['component'] or 'item'} | {fact['value']} | "
            f"{fact['evidence']} |"
        )
    lines.extend(["", "## Fixed-other conversation", ""])
    for event in cast(list[dict[str, Any]], review["transcript"]):
        if event["role"] == "assistant":
            lines.append(
                f"**Turn {event['turn']} assistant** (`ask_attribute=other`): {event['message']}"
            )
        else:
            ids = ", ".join(cast(list[str], event["disclosed_fact_ids"])) or "none"
            lines.append(
                f"**Turn {event['turn']} user** ({event['kind']}; facts: `{ids}`): "
                f"{event['message']}"
            )
        lines.append("")
    lines.extend(["## Excluded-fact audit", ""])
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    return "\n".join(lines)


def _summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    disclosure_counts = [
        len(cast(dict[str, list[object]], item["plan"])["disclosures"]) for item in reviews
    ]
    return {
        "session_count": len(reviews),
        "scenario_counts": dict(Counter(str(item["scenario_type"]) for item in reviews)),
        "known_previous_failure_count": sum(
            bool(item["known_previous_failure"]) for item in reviews
        ),
        "selected_disclosure_count": sum(disclosure_counts),
        "mean_disclosures_per_session": sum(disclosure_counts) / len(disclosure_counts),
        "minimum_disclosures": min(disclosure_counts),
        "maximum_disclosures": max(disclosure_counts),
    }


def _validate_reviews(
    reviews: list[dict[str, Any]],
    *,
    expected_sample_count: int,
    expected_scenario_counts: Counter[str],
) -> None:
    if len(reviews) != expected_sample_count:
        raise ValueError(f"review packet must contain exactly {expected_sample_count} sessions")
    scenario_counts = Counter(str(item["scenario_type"]) for item in reviews)
    if scenario_counts != expected_scenario_counts:
        raise ValueError(
            "review scenarios do not match the selected dataset: "
            f"{scenario_counts!r} != {expected_scenario_counts!r}"
        )
    ids = {str(item["sample_id"]) for item in reviews}
    if not KNOWN_FAILURES.issubset(ids):
        raise ValueError("review slice is missing a known previous failure")
    for review in reviews:
        disclosures = cast(dict[str, list[dict[str, object]]], review["plan"])["disclosures"]
        if len(disclosures) < 4:
            raise ValueError(f"review plan has fewer than four disclosures: {review['sample_id']}")
        if len({str(item["id"]) for item in disclosures}) != len(disclosures):
            raise ValueError(f"review plan has duplicate fact IDs: {review['sample_id']}")
    _validate_known_failure_focus(reviews)


def _validate_known_failure_focus(reviews: list[dict[str, Any]]) -> None:
    by_sample = {str(item["sample_id"]): item for item in reviews}
    expectations = {
        "public_0045": (("material", "polyester"), ("closure", "button")),
        "public_0098": (("material", "rubber"), ("material", "polyester")),
        "public_0154": (("material", "cotton"), ("care_instruction", "hand wash")),
        "public_0199": (("care_instruction", "machine wash"),),
        "public_0041": (("material", "polyester"), ("closure", "pull on")),
    }
    for sample_id, required in expectations.items():
        disclosures = cast(dict[str, list[dict[str, object]]], by_sample[sample_id]["plan"])[
            "disclosures"
        ]
        for facet, value_fragment in required:
            if not any(
                (
                    item.get("facet") == facet
                    or (facet == "closure" and item.get("facet") in {"closure", "closure_type"})
                )
                and value_fragment in str(item.get("value", "")).casefold()
                for item in disclosures
            ):
                raise ValueError(
                    f"known-failure review focus is absent: {sample_id}: {facet}={value_fragment}"
                )
    public_0041 = cast(dict[str, list[dict[str, object]]], by_sample["public_0041"]["plan"])[
        "disclosures"
    ]
    if any(item.get("facet") == "origin" for item in public_0041):
        raise ValueError("non-shopping imported metadata must not be disclosed for public_0041")


def _load_selection(path: Path) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if type(decoded) is not dict:
        raise ValueError("selection config must be an object")
    selection = cast(dict[str, object], decoded)
    if selection.get("schema") != "shopping-copilot/product-card-disclosure-review-selection/v1":
        raise ValueError("selection config has an unknown schema")
    values = selection.get("samples")
    if type(values) is not list or len(values) != 20:
        raise ValueError("selection config must contain 20 samples")
    return selection


def _load_samples(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            decoded: object = json.loads(line)
            if type(decoded) is not dict:
                raise ValueError("dataset must contain objects")
            sample = cast(dict[str, object], decoded)
            sample_id = sample.get("sample_id")
            if type(sample_id) is not str:
                raise ValueError("dataset sample_id must be a string")
            result[sample_id] = sample
    return result


def _select_samples(
    selection: dict[str, object],
    samples: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in cast(list[dict[str, object]], selection["samples"]):
        sample_id = str(item["sample_id"])
        if sample_id not in samples:
            raise KeyError(f"selection names an unknown sample: {sample_id}")
        result.append(samples[sample_id])
    return result


def _target_parent_asin(sample: dict[str, object]) -> str:
    ground_truth = sample.get("ground_truth")
    if type(ground_truth) is not dict:
        raise ValueError("dataset ground_truth must be an object")
    target = cast(dict[str, object], ground_truth).get("parent_asin")
    if type(target) is not str or not target:
        raise ValueError("dataset target parent_asin must be a non-empty string")
    return target


def _load_raw_cards(
    path: Path,
    *,
    target_ids: set[str],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            decoded: object = json.loads(line)
            if type(decoded) is not dict:
                continue
            card = cast(dict[str, object], decoded)
            target = card.get("parent_asin")
            if type(target) is str and target in target_ids:
                result[target] = card
    if set(result) != target_ids:
        raise ValueError("raw product-card records do not cover the review targets")
    return result


def _file_manifest(root: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        result.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            }
        )
    return result


def _replace_review_generation(staging: Path, output: Path) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing review output has no manifest")
    existing: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(existing) is not dict or existing.get("schema") != (
        "shopping-copilot/product-card-disclosure-review-manifest/v1"
    ):
        raise ValueError("existing review output is owned by another artifact contract")
    existing_files = {
        item.relative_to(output).as_posix() for item in output.rglob("*") if item.is_file()
    }
    staged_files = {
        item.relative_to(staging).as_posix() for item in staging.rglob("*") if item.is_file()
    }
    if existing_files != staged_files:
        raise ValueError("existing review output has a different file set")
    for relative in sorted(staged_files - {"manifest.json"}):
        os.replace(staging / relative, output / relative)
    os.replace(staging / "manifest.json", output / "manifest.json")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "data/benchmark_product_cards/public_200_v1/product-facts.jsonl",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "config/benchmark/product-card-disclosure-review-v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmark/product-card-disclosure-review-v1",
    )
    parser.add_argument("--maximum-facts", type=int, default=10)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="build disclosures for every dataset row instead of the fixed 20-session review",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
