#!/usr/bin/env python3
"""Stress-test the toy strategy with paraphrased simulator prompt wrappers.

The disclosed catalog attribute values are left byte-for-byte unchanged. Only
the surrounding English is rewritten, so this measures whether the toy parser
links visible catalog facts or memorizes the public evaluator's sentences.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
for source_path in (ROOT, ROOT / "src"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from evaluator import local_evaluator  # noqa: E402
from shopping_copilot.application import ToySimulatorAgent  # noqa: E402


class TracingToySimulatorAgent(ToySimulatorAgent):
    """Capture toy-only parser and recommendation evidence for failed-case review."""

    def __init__(self, catalog_path: str | Path) -> None:
        super().__init__(catalog_path)
        self.traces: dict[str, list[dict[str, object]]] = {}

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        super().reset(session_id, user_profile)
        self.traces[session_id] = []

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        response = super().respond(session_id, user_message, turn, top_k)
        state = self.sessions[session_id]
        self.traces[session_id].append(
            {
                "turn": turn,
                "user_message": user_message,
                "parser_event": state.last_event,
                "category": state.category,
                "active_constraints": [
                    {
                        "text": item.text,
                        "attribute": item.attribute,
                        "source": item.source,
                    }
                    for item in state.active_constraints
                ],
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": response.get("recommendations"),
            }
        )
        return response


def paraphrased_initial_message(sample: dict, category: str, disclosed: set[str]) -> str:
    scenario = sample["scenario_type"]
    hard_constraints = sample["intent_card"].get("hard_constraints") or []
    if scenario == "buying" and hard_constraints:
        constraint = str(hard_constraints[0])
        disclosed.add(constraint)
        return (
            f"I'm shopping within {category}; the non-negotiable catalog detail "
            f"is {constraint}."
        )
    if scenario == "intent_override":
        old_value = str(sample["behavior"]["override"]["old_value"])
        return f"For {category}, my starting preference is {old_value}."
    return f"I'd like to explore {category}; I haven't settled on the details yet."


def paraphrased_customer_reply(
    sample: dict,
    ask_attribute: object,
    disclosed: set[str],
    boundary_used: bool,
) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"Either is fine for {attribute}; use your judgement.", True
    if not attribute:
        return "Those choices miss the mark; ask one concrete attribute next.", boundary_used
    if attribute not in local_evaluator.ALLOWED_ATTRIBUTES:
        attribute = "other"

    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        value
        for value in constraints
        if value not in disclosed
        and (attribute == "other" or local_evaluator.classify_constraint(value) == attribute)
    ][:2]
    if not matches:
        return f"I have no additional preference about {attribute}.", boundary_used
    disclosed.update(matches)
    return "The original catalog details relevant here are " + ", and also ".join(matches) + ".", boundary_used


def paraphrased_materialize_hidden_fields(
    sample: dict,
    products: dict[str, dict],
) -> tuple[dict, dict]:
    card, behavior = _ORIGINAL_MATERIALIZE_HIDDEN_FIELDS(sample, products)
    behavior = copy.deepcopy(behavior)
    override = behavior.get("override")
    if isinstance(override, dict):
        new_value = str(override.get("new_value", ""))
        override["message"] = f"Let's go in a different direction; now I need {new_value}."
    return card, behavior


_ORIGINAL_MATERIALIZE_HIDDEN_FIELDS = local_evaluator.materialize_hidden_fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    samples = local_evaluator.load_jsonl(args.dataset)
    if args.sample_id:
        selected = set(args.sample_id)
        samples = [sample for sample in samples if str(sample.get("sample_id")) in selected]
        missing = selected - {str(sample.get("sample_id")) for sample in samples}
        if missing:
            parser.error("unknown sample ids: " + ", ".join(sorted(missing)))
    catalog_ids, categories, products = local_evaluator.catalog_index(args.catalog)

    # These assignments are process-local and affect only this diagnostic run.
    local_evaluator.initial_message = paraphrased_initial_message
    local_evaluator.customer_reply = paraphrased_customer_reply
    local_evaluator.materialize_hidden_fields = paraphrased_materialize_hidden_fields

    indexed_samples = list(enumerate(samples))
    shards = [indexed_samples[index :: args.workers] for index in range(args.workers)]
    shards = [shard for shard in shards if shard]

    def run_shard(
        shard: list[tuple[int, dict]],
    ) -> tuple[list[tuple[int, dict]], list[tuple[int, list[dict[str, object]]]], dict]:
        agent: ToySimulatorAgent
        if args.include_traces:
            agent = TracingToySimulatorAgent(args.catalog)
        else:
            agent = ToySimulatorAgent(args.catalog)
        shard_samples = [sample for _, sample in shard]
        shard_result = local_evaluator.evaluate(
            agent,
            shard_samples,
            catalog_ids,
            categories,
            products,
        )
        session_pairs = list(
            zip(
                (index for index, _ in shard),
                cast(list[dict], shard_result["sessions"]),
                strict=True,
            )
        )
        trace_pairs: list[tuple[int, list[dict[str, object]]]] = []
        if isinstance(agent, TracingToySimulatorAgent):
            trace_pairs = list(
                zip(
                    (index for index, _ in shard),
                    agent.traces.values(),
                    strict=True,
                )
            )
        return session_pairs, trace_pairs, shard_result

    print(f"evaluating {len(samples)} sessions with {len(shards)} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        shard_results = list(executor.map(run_shard, shards))

    session_pairs = [pair for pairs, _, _ in shard_results for pair in pairs]
    session_pairs.sort(key=lambda item: item[0])
    sessions = [session for _, session in session_pairs]
    result = _summarize_sessions(sessions)
    result["reported_token_usage"] = {
        key: sum(
            int(cast(dict, shard_result["reported_token_usage"])[key])
            for _, _, shard_result in shard_results
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    result["sessions"] = sessions

    trace_pairs = [pair for _, pairs, _ in shard_results for pair in pairs]
    trace_pairs.sort(key=lambda item: item[0])
    traces = {
        str(samples[index]["sample_id"]): trace for index, trace in trace_pairs
    }
    payload = {
        **result,
        "diagnostic": {
            "runtime_mode": "official_simulator",
            "variant": "paraphrased_wrappers_verbatim_catalog_facts_v1",
            "attribute_values_modified": False,
            "workers": len(shards),
        },
    }
    if args.include_traces:
        payload["traces"] = cast(object, traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"sessions", "traces"}
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _summarize_sessions(sessions: list[dict]) -> dict[str, object]:
    overall = local_evaluator.metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": {
            name: local_evaluator.metric_summary(grouped[name])
            for name in sorted(grouped)
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
