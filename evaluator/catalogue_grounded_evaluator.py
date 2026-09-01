from __future__ import annotations

import argparse
import importlib
import json
import re
import statistics
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index, normalize_recommendations, searchable_text


TOP_K = 10
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "for",
    "from",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "use",
    "with",
}


def _tokens(value: object) -> set[str]:
    return {token for token in TOKEN_RE.findall(str(value).casefold()) if token not in STOPWORDS}


def _load_profiles(path: Path) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            profiles[str(row["sample_id"])] = dict(row.get("user_profile") or {})
    return profiles


def _load_journeys(path: Path) -> list[dict[str, object]]:
    journeys: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "shopping-copilot/product-card-disclosure-review/v1":
                raise ValueError("journey file contains an unknown schema")
            target = str(row.get("target_parent_asin") or "")
            user_turns = _user_turns(row)
            fact_ids = {
                str(fact_id)
                for turn in user_turns
                for fact_id in (turn.get("disclosed_fact_ids") or [])
            }
            if not 6 <= len(fact_ids) <= 8:
                raise ValueError("every journey must contain six to eight grounded dimensions")
            if any(target and target.casefold() in str(turn.get("message") or "").casefold() for turn in user_turns):
                raise ValueError("a journey message exposes its target identifier")
            turn_numbers = [int(turn["turn"]) for turn in user_turns]
            if turn_numbers != list(range(1, len(user_turns) + 1)):
                raise ValueError("journey user turns must be contiguous and one-indexed")
            journeys.append(row)
    if len(journeys) != 200:
        raise ValueError(f"catalogue-grounded benchmark expects 200 journeys, got {len(journeys)}")
    if len({str(row["sample_id"]) for row in journeys}) != len(journeys):
        raise ValueError("journey sample IDs must be unique")
    return journeys


def _user_turns(journey: dict[str, object]) -> list[dict[str, object]]:
    transcript = journey.get("transcript")
    if not isinstance(transcript, list):
        raise ValueError("journey transcript must be an array")
    turns = [dict(item) for item in transcript if isinstance(item, dict) and item.get("role") == "user"]
    if not turns:
        raise ValueError("journey must contain at least one user turn")
    return turns


def _active_fact_tokens(events: list[dict[str, object]]) -> set[str]:
    active: dict[str, set[str]] = {}
    for event in events:
        withdrawn = event.get("withdrawn_fact_ids") or []
        if isinstance(withdrawn, list):
            for fact_id in withdrawn:
                active.pop(str(fact_id), None)
        facts = event.get("disclosed_facts") or []
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_id = str(fact.get("id") or "")
            value_tokens = _tokens(fact.get("value") or "")
            if fact_id and value_tokens:
                active[fact_id] = value_tokens
    return set().union(*active.values()) if active else set()


def _category_tokens(product: dict[str, object]) -> set[str]:
    categories = product.get("categories") or []
    if not isinstance(categories, list):
        return set()
    meaningful = [str(item) for item in categories if str(item).strip()][-2:]
    return _tokens(" ".join(meaningful)) - {"clothing", "jewelry", "shoes"}


def _direction(product: dict[str, object]) -> str:
    categories = product.get("categories") or []
    if isinstance(categories, list) and categories:
        value = str(categories[-1]).split(",")[-1].strip().casefold()
        if value:
            return value
    title_tokens = sorted(_tokens(product.get("title") or ""))
    return " ".join(title_tokens[:2]) or "unknown"


def exploration_score(
    ranked: list[str],
    *,
    products: dict[str, dict[str, object]],
    target: str,
    active_fact_tokens: set[str],
) -> tuple[float, dict[str, object]]:
    """Score a target-blind slate against disclosed evidence and set diversity.

    The evaluator may use the target listing to derive the intended product category,
    but the target identifier and this diagnostic never cross the Agent boundary.
    """

    if not ranked:
        return 0.0, {
            "relevance_rate": 0.0,
            "direction_count": 0,
            "diversity_rate": 0.0,
        }
    target_category = _category_tokens(products[target])
    relevant = 0
    directions: set[str] = set()
    for parent_asin in ranked:
        product = products[parent_asin]
        product_tokens = _tokens(searchable_text(product))
        category_match = bool(target_category & _category_tokens(product))
        fact_match = bool(active_fact_tokens & product_tokens)
        if category_match or fact_match:
            relevant += 1
        directions.add(_direction(product))
    relevance_rate = relevant / len(ranked)
    diversity_rate = min(len(directions) / 4.0, 1.0)
    score = 0.6 * relevance_rate + 0.4 * diversity_rate
    return score, {
        "relevance_rate": round(relevance_rate, 6),
        "direction_count": len(directions),
        "diversity_rate": round(diversity_rate, 6),
        "relevance_weight": 0.6,
        "diversity_weight": 0.4,
    }


def evaluate(
    agent: Any,
    *,
    journeys: list[dict[str, object]],
    profiles: dict[str, dict[str, object]],
    catalog_ids: set[str],
    products: dict[str, dict[str, object]],
) -> dict[str, object]:
    sessions: list[dict[str, object]] = []
    satisfaction_sum = 0.0
    satisfaction_total = 0
    prompt_tokens = 0
    completion_tokens = 0
    for journey in journeys:
        sample_id = str(journey["sample_id"])
        scenario = str(journey["scenario_type"])
        target = str(journey["target_parent_asin"])
        turns = _user_turns(journey)
        session_id = f"catalogue-grounded/{sample_id}/{uuid.uuid4().hex}"
        agent.reset(session_id, profiles.get(sample_id, {}))
        first_hit_turn: int | None = None
        best_rank: int | None = None
        seen_events: list[dict[str, object]] = []
        turn_evidence: list[dict[str, object]] = []
        override_applied = scenario != "intent_override"
        for index, event in enumerate(turns):
            turn = int(event["turn"])
            message = str(event["message"])
            seen_events.append(event)
            if event.get("kind") == "intent_override":
                override_applied = True
            try:
                response = agent.respond(session_id, message, turn, TOP_K)
            except Exception as error:
                response = {"recommendations": [], "usage": {}, "error": type(error).__name__}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            usage = response.get("usage") or {}
            if isinstance(usage, dict):
                prompt_tokens += max(0, int(usage.get("prompt_tokens") or 0))
                completion_tokens += max(0, int(usage.get("completion_tokens") or 0))
            rank = ranked.index(target) + 1 if target in ranked else None
            if override_applied and rank is not None and first_hit_turn is None:
                first_hit_turn = turn
                best_rank = rank
            elif override_applied and rank is not None:
                best_rank = rank if best_rank is None else min(best_rank, rank)

            future = turns[index + 1 :]
            another_preference_follows = any(item.get("disclosed_fact_ids") for item in future)
            satisfaction: float | None = None
            diagnostic: dict[str, object] | None = None
            if another_preference_follows:
                satisfaction, diagnostic = exploration_score(
                    ranked,
                    products=products,
                    target=target,
                    active_fact_tokens=_active_fact_tokens(seen_events),
                )
                satisfaction_total += 1
                satisfaction_sum += satisfaction
            turn_evidence.append(
                {
                    "turn": turn,
                    "kind": event.get("kind"),
                    "target_rank": rank,
                    "exploration_satisfaction": satisfaction,
                    "exploration_diagnostic": diagnostic,
                }
            )
        sessions.append(
            {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "turns_to_recall": first_hit_turn,
                "best_rank": best_rank,
                "turns": turn_evidence,
            }
        )

    recall = statistics.fmean(int(bool(item["hit"])) for item in sessions)
    recalled_turns = [
        int(item["turns_to_recall"])
        for item in sessions
        if item["turns_to_recall"] is not None
    ]
    turns_to_recall = statistics.fmean(recalled_turns) if recalled_turns else None
    return {
        "schema": "aperture/catalogue-grounded-benchmark-result/v1",
        "sample_count": len(sessions),
        "recall_at_10": round(recall, 6),
        "turns_to_recall": None if turns_to_recall is None else round(turns_to_recall, 6),
        "exploration_satisfaction": round(
            satisfaction_sum / satisfaction_total if satisfaction_total else 0.0,
            6,
        ),
        "exploration_turn_count": satisfaction_total,
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "sessions": sessions,
    }


def _load_agent_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("agent factory must use module:attribute syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("agent factory must be callable")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--journeys",
        type=Path,
        default=Path("benchmarks/catalogue_grounded_200/journeys.jsonl"),
    )
    parser.add_argument("--profiles", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--agent-factory", default="starter.agent:Agent")
    parser.add_argument("--output", type=Path, default=Path("catalogue-grounded-results.json"))
    args = parser.parse_args()

    catalog_ids, _, products = catalog_index(args.catalog)
    factory = _load_agent_factory(args.agent_factory)
    agent = factory(catalog_path=args.catalog)
    result = evaluate(
        agent,
        journeys=_load_journeys(args.journeys),
        profiles=_load_profiles(args.profiles),
        catalog_ids=catalog_ids,
        products=products,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
