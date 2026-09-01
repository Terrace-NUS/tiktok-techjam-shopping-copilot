#!/usr/bin/env python3
"""Evaluate APERTURE's offline profile with independent session shards in parallel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
for source_path in (ROOT, ROOT / "src"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
)
from shopping_copilot.application import OfflineApertureAgent  # noqa: E402
from shopping_copilot.application.offline.ranker import (  # noqa: E402
    AmbiguityPrior,
    ProductRanker,
)


def main() -> int:
    args = _parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    public_prior = _load_public_prior(args.public_prior_path) if args.prior == "public_like" else {}
    indexed_samples = list(enumerate(samples))
    shards = [indexed_samples[index :: args.workers] for index in range(args.workers)]
    shards = [shard for shard in shards if shard]

    def run_shard(shard: list[tuple[int, dict]]) -> tuple[list[tuple[int, dict]], dict]:
        agent = OfflineApertureAgent(args.catalog, question_mode=args.question_mode)
        prior_scores = _prior_scores(
            source=args.prior,
            asins=agent.catalog.asins,
            review_counts=agent.catalog.rating_count_by_pid,
            public_prior=public_prior,
        )
        if prior_scores is not None:
            agent.ranker = ProductRanker(
                agent.catalog,
                ambiguity_prior=AmbiguityPrior(
                    scores=prior_scores,
                    strength=args.prior_strength,
                    evidence_window=args.ambiguity_window,
                    reorder_depth=args.prior_reorder_depth,
                ),
            )
        result = evaluate(
            agent,
            [sample for _, sample in shard],
            catalog_ids,
            categories,
            products,
        )
        sessions = cast(list[dict], result["sessions"])
        return list(zip((index for index, _ in shard), sessions, strict=True)), result

    print(f"evaluating {len(samples)} sessions with {len(shards)} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        results = list(executor.map(run_shard, shards))

    ordered_pairs = [pair for shard_pairs, _ in results for pair in shard_pairs]
    ordered_pairs.sort(key=lambda item: item[0])
    sessions = [session for _, session in ordered_pairs]
    usage = {
        "prompt_tokens": sum(
            int(cast(dict, result["reported_token_usage"])["prompt_tokens"])
            for _, result in results
        ),
        "completion_tokens": sum(
            int(cast(dict, result["reported_token_usage"])["completion_tokens"])
            for _, result in results
        ),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    payload = {
        **_summarize(sessions),
        "reported_token_usage": usage,
        "workers": len(shards),
        "question_mode": args.question_mode,
        "ranking_prior": {
            "source": args.prior,
            "strength": args.prior_strength,
            "ambiguity_window": args.ambiguity_window,
            "reorder_depth": args.prior_reorder_depth,
        },
        "dataset": str(args.dataset.resolve()),
        "prefix_metrics": {
            str(prefix): _summarize(sessions[:prefix])
            for prefix in args.prefix
            if prefix <= len(sessions)
        },
        "sessions": sessions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in payload.items() if key != "sessions"}, indent=2),
        flush=True,
    )
    return 0


def _summarize(sessions: list[dict]) -> dict[str, object]:
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    turn_values = [
        int(session["first_hit_turn"])
        if session["first_hit_turn"] is not None
        else MAX_TURNS + 1
        for session in sessions
    ]
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "median_turn_to_conversion_with_misses": float(statistics.median(turn_values)),
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--question-mode",
        choices=("typed_first", "other_first"),
        default="typed_first",
    )
    parser.add_argument(
        "--prior",
        choices=("none", "review", "public_like"),
        default="none",
    )
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--ambiguity-window", type=float, default=0.0)
    parser.add_argument("--prior-reorder-depth", type=int, default=None)
    parser.add_argument(
        "--public-prior-path",
        type=Path,
        default=ROOT
        / "artifacts/benchmark/public-like-gradient-v1/catalog-selection-prior.jsonl",
    )
    parser.add_argument("--prefix", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if any(prefix < 1 for prefix in args.prefix):
        parser.error("--prefix must be positive")
    if not math.isfinite(args.prior_strength) or args.prior_strength <= 0.0:
        parser.error("--prior-strength must be positive and finite")
    if not math.isfinite(args.ambiguity_window) or args.ambiguity_window < 0.0:
        parser.error("--ambiguity-window must be finite and non-negative")
    if args.prior_reorder_depth is not None and args.prior_reorder_depth < 1:
        parser.error("--prior-reorder-depth must be positive")
    return args


def _load_public_prior(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"public-likeness prior not found: {path}")
    result: dict[str, float] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["parent_asin"])] = float(row["public_likeness_quantile"])
    return result


def _prior_scores(
    *,
    source: str,
    asins: list[str],
    review_counts: list[int],
    public_prior: dict[str, float],
) -> tuple[float, ...] | None:
    if source == "none":
        return None
    if source == "public_like":
        missing = [asin for asin in asins if asin not in public_prior]
        if missing:
            raise ValueError(f"public-likeness prior is missing {len(missing)} catalog products")
        return tuple(public_prior[asin] for asin in asins)
    if source == "review":
        ordered = sorted(review_counts)
        cap = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        denominator = math.log1p(max(1, cap))
        return tuple(min(1.0, math.log1p(max(0, count)) / denominator) for count in review_counts)
    raise ValueError(f"unknown prior source: {source}")


if __name__ == "__main__":
    raise SystemExit(main())
