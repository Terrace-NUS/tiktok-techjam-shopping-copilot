#!/usr/bin/env python3
"""Replay the public-200 final queries through old and replaced product cards."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from scripts.retrieval.evaluate_candidate_recall_sweep_v0 import (  # noqa: E402
    ReplayCase,
    _load_cases,
)
from shopping_copilot.catalog.product_facts import load_product_fact_sidecar  # noqa: E402
from shopping_copilot.retrieval import (  # noqa: E402
    ProductCardMode,
    create_retrieval_controller,
)

REPORT_SCHEMA = "shopping-copilot/partial-product-card-recall-ab/v1"
STAGES = ("eligible", "dense", "lexical", "facet", "route_union", "fused", "final")


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    cases = tuple(case for case in _load_cases(args.turn_logs) if case.representative)
    if len(cases) != 200:
        raise ValueError(f"expected 200 representative cases, found {len(cases)}")
    transparencies = _load_transparencies(args.turn_logs)

    print("replaying old product cards...", flush=True)
    old = _evaluate_variant(
        cases,
        transparencies=transparencies,
        index_path=args.old_index,
        semantic_release=args.semantic_release,
        catalog=args.catalog,
        device=args.device,
        cards=None,
    )
    gc.collect()

    print("replaying replaced product cards...", flush=True)
    cards = load_product_fact_sidecar(args.sidecar, catalog_path=args.catalog)
    new = _evaluate_variant(
        cases,
        transparencies=transparencies,
        index_path=args.new_index,
        semantic_release=args.semantic_release,
        catalog=args.catalog,
        device=args.device,
        cards=cards,
    )
    by_identity = {item["identity"]: item for item in new["cases"]}
    comparisons = [_compare_case(item, by_identity[item["identity"]]) for item in old["cases"]]
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "turn_selection": "last searchable scored turn per public-200 session",
            "query_understanding_replayed": True,
            "deepseek_called": False,
            "transparency_source": "saved applied_transparency",
            "old_cards": "raw catalog ProductDocument",
            "new_cards": "verified product fact card replaces raw card for covered products",
            "new_dense": "only covered product rows re-embedded; all other rows exact copies",
            "target_visible_to_retrieval": False,
        },
        "old": old,
        "new": new,
        "comparison": _summarize_comparison(comparisons),
        "cases": comparisons,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2), flush=True)
    print(f"json={args.output_json}", flush=True)
    print(f"markdown={args.output_markdown}", flush=True)
    return 0


def _evaluate_variant(
    cases: tuple[ReplayCase, ...],
    *,
    transparencies: dict[str, float],
    index_path: Path,
    semantic_release: Path,
    catalog: Path,
    device: str,
    cards: Any,
) -> dict[str, Any]:
    initialized = time.perf_counter()
    controller = create_retrieval_controller(
        index_path=index_path,
        release_dir=semantic_release,
        catalog_path=catalog,
        device=device,
        local_files_only=True,
        product_fact_cards=cards,
        product_card_mode=ProductCardMode.REPLACE,
    )
    initialization_seconds = time.perf_counter() - initialized
    records: list[dict[str, Any]] = []
    evaluated = time.perf_counter()
    for ordinal, case in enumerate(cases, start=1):
        transparency = transparencies[case.identity]
        result = controller.search(case.compiled, transparency=transparency)
        route_ranks = {
            route.route.value: _rank_of(route.hits, case.target_parent_asin)
            for route in result.routes
        }
        fused_rank = _rank_of(result.fused_candidates, case.target_parent_asin)
        final_rank = _rank_of(result.hits, case.target_parent_asin)
        eligible = case.target_parent_asin in result.hard_mask.eligible_parent_asins
        records.append(
            {
                "identity": case.identity,
                "sample_id": case.sample_id,
                "scenario_type": case.scenario_type,
                "turn": case.turn,
                "target_parent_asin": case.target_parent_asin,
                "transparency": transparency,
                "q_lex": case.compiled.q_lex,
                "q_sem": case.compiled.q_sem,
                "eligible": eligible,
                "dense_rank": route_ranks.get("dense"),
                "lexical_rank": route_ranks.get("lexical"),
                "facet_rank": route_ranks.get("facet"),
                "route_union_hit": any(value is not None for value in route_ranks.values()),
                "fused_rank": fused_rank,
                "final_rank": final_rank,
                "route_counts": {route.route.value: len(route.hits) for route in result.routes},
                "fused_count": len(result.fused_candidates),
                "total_ms": result.timings.total_ms,
            }
        )
        if ordinal % 50 == 0:
            print(f"  {ordinal}/{len(cases)}", flush=True)
    return {
        "index_path": str(index_path.resolve()),
        "index_id": controller.retriever.index.index_id,
        "initialization_seconds": initialization_seconds,
        "evaluation_seconds": time.perf_counter() - evaluated,
        "summary": _summarize_variant(records),
        "cases": records,
    }


def _load_transparencies(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            evaluator = row.get("evaluator")
            transparency = row.get("intent_transparency")
            if not isinstance(evaluator, dict) or not isinstance(transparency, dict):
                continue
            sample_id = evaluator.get("sample_id")
            turn = row.get("turn")
            applied = transparency.get("applied_transparency")
            if (
                isinstance(sample_id, str)
                and isinstance(turn, int)
                and isinstance(applied, (float, int))
            ):
                values[f"{sample_id}/turn-{turn}"] = float(applied)
    return values


def _rank_of(items: Any, target: str) -> int | None:
    for item in items:
        if item.parent_asin == target:
            return int(item.rank)
    return None


def _summarize_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["total_ms"]) for item in records]
    return {
        "case_count": len(records),
        "recall": {stage: _stage_recall(records, stage) for stage in STAGES},
        "mean_retrieval_ms": statistics.fmean(latencies),
        "median_retrieval_ms": statistics.median(latencies),
    }


def _stage_recall(records: list[dict[str, Any]], stage: str) -> float:
    if stage == "eligible":
        hits = sum(bool(item["eligible"]) for item in records)
    elif stage == "route_union":
        hits = sum(bool(item["route_union_hit"]) for item in records)
    else:
        key = f"{stage}_rank"
        hits = sum(item[key] is not None for item in records)
    return hits / len(records)


def _compare_case(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        if stage == "eligible":
            old_hit = bool(old["eligible"])
            new_hit = bool(new["eligible"])
            old_rank = new_rank = None
        elif stage == "route_union":
            old_hit = bool(old["route_union_hit"])
            new_hit = bool(new["route_union_hit"])
            old_rank = new_rank = None
        else:
            old_rank = old[f"{stage}_rank"]
            new_rank = new[f"{stage}_rank"]
            old_hit = old_rank is not None
            new_hit = new_rank is not None
        stages[stage] = {
            "old_hit": old_hit,
            "new_hit": new_hit,
            "old_rank": old_rank,
            "new_rank": new_rank,
        }
    return {
        "identity": old["identity"],
        "sample_id": old["sample_id"],
        "scenario_type": old["scenario_type"],
        "target_parent_asin": old["target_parent_asin"],
        "q_lex": old["q_lex"],
        "q_sem": old["q_sem"],
        "stages": stages,
    }


def _summarize_comparison(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        rows = [item["stages"][stage] for item in comparisons]
        old_count = sum(bool(item["old_hit"]) for item in rows)
        new_count = sum(bool(item["new_hit"]) for item in rows)
        stages[stage] = {
            "old_count": old_count,
            "new_count": new_count,
            "old_recall": old_count / len(rows),
            "new_recall": new_count / len(rows),
            "recall_delta": (new_count - old_count) / len(rows),
            "gained": sum(not item["old_hit"] and item["new_hit"] for item in rows),
            "lost": sum(item["old_hit"] and not item["new_hit"] for item in rows),
        }
    return {"case_count": len(comparisons), "stages": stages}


def _render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    labels = {
        "eligible": "通过硬筛",
        "dense": "Dense 路",
        "lexical": "关键词路",
        "facet": "结构化属性路",
        "route_union": "任一路找到",
        "fused": "合并候选 Top 300",
        "final": "当前轻量排序 Top 10",
    }
    lines = [
        "# Public 200 商品卡完整召回 A/B",
        "",
        "复用保存的 QU 与 T_t，不调用 DeepSeek；新版本对 200 个目标商品完全替换商品卡和 Dense 向量。",
        "",
        "| 阶段 | 旧卡召回 | 新卡召回 | 变化 | 新增 / 丢失 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        values = comparison["stages"][stage]
        lines.append(
            f"| {labels[stage]} | {values['old_recall']:.1%} | {values['new_recall']:.1%} | "
            f"{values['recall_delta']:+.1%} | {values['gained']} / {values['lost']} |"
        )
    lines.extend(
        [
            "",
            "该表评价的是候选生成和仓库当前轻量排序，不包含成本较高的 DeepSeek 最终排序。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-index", type=Path, default=ROOT / "artifacts/retrieval/dense-v0")
    parser.add_argument(
        "--new-index",
        type=Path,
        default=ROOT / "artifacts/retrieval/dense-public-200-replaced-v1",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--semantic-release",
        type=Path,
        default=ROOT / "artifacts/catalog-semantic/release-v0",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "data/benchmark_product_cards/public_200_v1/product-facts.jsonl",
    )
    parser.add_argument(
        "--turn-logs",
        type=Path,
        default=(
            ROOT
            / "artifacts/simulator/deepseek-surface-comparison-20260830/real-other-200/turns.jsonl"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/retrieval/public-200-product-card-recall-ab-v1.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/public-200-product-card-recall-ab-v1.md",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
