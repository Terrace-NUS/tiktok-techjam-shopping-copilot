#!/usr/bin/env python3
"""Compare target-product Dense ranks before and after partial card replacement."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.retrieval import (  # noqa: E402
    SentenceTransformerTextEmbedder,
    load_dense_index,
)

REPORT_SCHEMA = "shopping-copilot/partial-product-card-dense-ab/v1"
DEFAULT_CUTOFFS = (10, 80, 150, 210, 300, 2_000)


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    old_index = load_dense_index(args.old_index)
    new_index = load_dense_index(args.new_index)
    if old_index.parent_asins != new_index.parent_asins:
        raise ValueError("old and new Dense indexes do not contain the same ordered products")
    if old_index.manifest.embedding != new_index.manifest.embedding:
        raise ValueError("old and new Dense indexes use different embedding specifications")

    cases = _load_latest_cases(args.turn_logs)
    embedder = SentenceTransformerTextEmbedder(
        old_index.manifest.embedding,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    row_by_asin = {asin: row for row, asin in enumerate(old_index.parent_asins)}
    asin_keys = np.asarray(old_index.parent_asins, dtype=np.str_)
    results: list[dict[str, Any]] = []

    for case in cases:
        query_vector = embedder.encode_query(case["q_sem"])
        old_scores = np.asarray(old_index.vectors @ query_vector, dtype=np.float32)
        new_scores = np.asarray(new_index.vectors @ query_vector, dtype=np.float32)
        target = case["target_parent_asin"]
        target_row = row_by_asin[target]
        old_rank = _exact_rank(old_scores, asin_keys, target_row)
        new_rank = _exact_rank(new_scores, asin_keys, target_row)
        results.append(
            {
                **case,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "rank_delta": old_rank - new_rank,
                "old_similarity": float(old_scores[target_row]),
                "new_similarity": float(new_scores[target_row]),
                "similarity_delta": float(new_scores[target_row] - old_scores[target_row]),
            }
        )

    summary = _summarize(results)
    report = {
        "schema": REPORT_SCHEMA,
        "source_turn_logs": str(args.turn_logs.resolve()),
        "case_selection": "latest successful compiled query per simulator session",
        "old_index": {
            "path": str(args.old_index.resolve()),
            "index_id": old_index.index_id,
            "document_corpus_id": old_index.manifest.document_corpus_id,
        },
        "new_index": {
            "path": str(args.new_index.resolve()),
            "index_id": new_index.index_id,
            "document_corpus_id": new_index.manifest.document_corpus_id,
        },
        "embedding_model": old_index.manifest.embedding.model_id,
        "summary": summary,
        "cases": results,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"json={args.output_json}", flush=True)
    print(f"markdown={args.output_markdown}", flush=True)
    return 0


def _load_latest_cases(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"turn log line {line_number} is not an object")
            compiled = row.get("compiled_query")
            evaluator = row.get("evaluator")
            session_id = row.get("session_id")
            if not isinstance(compiled, dict) or not isinstance(evaluator, dict):
                continue
            q_sem = compiled.get("q_sem")
            target = evaluator.get("target_parent_asin")
            if not isinstance(session_id, str) or not isinstance(q_sem, str):
                continue
            if not isinstance(target, str) or not target:
                continue
            turn = row.get("turn")
            if not isinstance(turn, int):
                continue
            current = latest.get(session_id)
            if current is None or turn > current["turn"]:
                latest[session_id] = {
                    "session_id": session_id,
                    "sample_id": evaluator.get("sample_id"),
                    "scenario_type": evaluator.get("scenario_type"),
                    "turn": turn,
                    "q_sem": q_sem,
                    "target_parent_asin": target,
                }
    if not latest:
        raise ValueError("turn logs contain no comparable successful cases")
    return [latest[key] for key in sorted(latest)]


def _exact_rank(scores: np.ndarray, asin_keys: np.ndarray, target_row: int) -> int:
    target_score = scores[target_row]
    target_asin = asin_keys[target_row]
    ahead = np.count_nonzero(scores > target_score)
    tied_ahead = np.count_nonzero((scores == target_score) & (asin_keys < target_asin))
    return int(ahead + tied_ahead + 1)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    old_ranks = [int(item["old_rank"]) for item in results]
    new_ranks = [int(item["new_rank"]) for item in results]
    similarity_deltas = [float(item["similarity_delta"]) for item in results]
    return {
        "case_count": len(results),
        "target_rank": {
            "old_mean": statistics.fmean(old_ranks),
            "new_mean": statistics.fmean(new_ranks),
            "old_median": statistics.median(old_ranks),
            "new_median": statistics.median(new_ranks),
            "improved": sum(new < old for old, new in zip(old_ranks, new_ranks, strict=True)),
            "unchanged": sum(new == old for old, new in zip(old_ranks, new_ranks, strict=True)),
            "worse": sum(new > old for old, new in zip(old_ranks, new_ranks, strict=True)),
        },
        "target_similarity_delta": {
            "mean": statistics.fmean(similarity_deltas),
            "median": statistics.median(similarity_deltas),
            "positive": sum(value > 0 for value in similarity_deltas),
            "zero": sum(value == 0 for value in similarity_deltas),
            "negative": sum(value < 0 for value in similarity_deltas),
        },
        "recall": {
            f"at_{cutoff}": {
                "old": sum(rank <= cutoff for rank in old_ranks) / len(results),
                "new": sum(rank <= cutoff for rank in new_ranks) / len(results),
                "delta": (
                    sum(rank <= cutoff for rank in new_ranks)
                    - sum(rank <= cutoff for rank in old_ranks)
                )
                / len(results),
            }
            for cutoff in DEFAULT_CUTOFFS
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    target_rank = summary["target_rank"]
    similarity = summary["target_similarity_delta"]
    lines = [
        "# Public 200 商品卡 Dense A/B",
        "",
        "本实验复用已保存的每个 simulator session 最后一轮 QU 查询，只替换商品卡及其 Dense 向量。",
        "",
        f"- 样本数：{summary['case_count']}",
        f"- 目标排名改善 / 不变 / 变差：{target_rank['improved']} / {target_rank['unchanged']} / {target_rank['worse']}",
        f"- 目标排名中位数：旧 {target_rank['old_median']}，新 {target_rank['new_median']}",
        f"- 目标排名均值：旧 {target_rank['old_mean']:.2f}，新 {target_rank['new_mean']:.2f}",
        f"- 目标相似度平均变化：{similarity['mean']:+.6f}",
        "",
        "## Dense 目标召回率",
        "",
        "| 截断位置 | 旧商品卡 | 新商品卡 | 变化 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for label, values in summary["recall"].items():
        cutoff = label.removeprefix("at_")
        lines.append(
            f"| {cutoff} | {values['old']:.1%} | {values['new']:.1%} | {values['delta']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这 200 个被测目标都使用新商品卡，其他 49,800 个商品保留旧商品卡。"
            "因此它适合判断部分替换方案是否工作，不代表全量商品卡都升级后的最终绝对指标。",
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
        default=ROOT / "artifacts/retrieval/public-200-product-card-dense-ab-v1.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts/retrieval/public-200-product-card-dense-ab-v1.md",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
