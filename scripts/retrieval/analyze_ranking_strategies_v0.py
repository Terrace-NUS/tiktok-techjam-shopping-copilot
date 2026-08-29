"""Produce paired uncertainty and transition diagnostics for ranking-strategy v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SCHEMA = "shopping-copilot/ranking-strategy-analysis/v0"
SEED = 20_260_829
BOOTSTRAP_SAMPLES = 20_000

COMPARISONS = (
    ("relative_score_topk", "rrf_topk"),
    ("combmnz_topk", "rrf_topk"),
    ("qwen_topk", "rrf_topk"),
    ("bge_topk", "rrf_topk"),
    ("bge_topk", "qwen_topk"),
    ("qwen_mmr_low", "qwen_topk"),
    ("qwen_dpp_low", "qwen_topk"),
    ("bge_dpp_low", "bge_topk"),
    ("bge_dpp_high", "bge_topk"),
    ("bge_dpp_low", "rrf_mmr_low"),
    ("bge_dpp_low", "qwen_dpp_low"),
    ("qwen_xquad_low", "qwen_topk"),
)
T_FAMILIES = ("rrf_mmr", "qwen_mmr", "qwen_dpp", "bge_dpp", "qwen_xquad")


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema") != "shopping-copilot/ranking-strategy-evaluation/v0":
        raise ValueError("input is not a ranking-strategy v0 evaluation")
    cases = [case for case in payload["cases"] if case["source"] == "public_simulator"]
    if not cases:
        raise ValueError("input contains no public simulator cases")
    metrics = {
        str(case["case_id"]): {
            str(variant["name"]): variant["metrics"] for variant in case["variants"]
        }
        for case in cases
    }

    comparisons = []
    for left, right in COMPARISONS:
        comparisons.append(
            {
                "left": left,
                "right": right,
                "mrr_at_10": _paired_bootstrap(
                    _values(cases, metrics, left, "reciprocal_rank_at_10"),
                    _values(cases, metrics, right, "reciprocal_rank_at_10"),
                ),
                "mean_pairwise_product_cosine": _paired_bootstrap(
                    _values(cases, metrics, left, "mean_pairwise_product_cosine"),
                    _values(cases, metrics, right, "mean_pairwise_product_cosine"),
                ),
                "unique_reporting_groups": _paired_bootstrap(
                    _values(cases, metrics, left, "unique_reporting_groups"),
                    _values(cases, metrics, right, "unique_reporting_groups"),
                ),
            }
        )

    transitions = []
    for family in T_FAMILIES:
        low = f"{family}_low"
        high = f"{family}_high"
        low_similarity = _values(cases, metrics, low, "mean_pairwise_product_cosine")
        high_similarity = _values(cases, metrics, high, "mean_pairwise_product_cosine")
        low_groups = _values(cases, metrics, low, "unique_reporting_groups")
        high_groups = _values(cases, metrics, high, "unique_reporting_groups")
        transitions.append(
            {
                "family": family,
                "low_minus_high_similarity": _paired_bootstrap(low_similarity, high_similarity),
                "low_is_more_vector_diverse_rate": float(np.mean(low_similarity < high_similarity)),
                "low_minus_high_reporting_groups": _paired_bootstrap(low_groups, high_groups),
                "low_has_at_least_as_many_groups_rate": float(np.mean(low_groups >= high_groups)),
                "low_minus_high_mrr_at_10": _paired_bootstrap(
                    _values(cases, metrics, low, "reciprocal_rank_at_10"),
                    _values(cases, metrics, high, "reciprocal_rank_at_10"),
                ),
            }
        )

    analysis = {
        "schema": SCHEMA,
        "source": str(args.input),
        "case_count": len(cases),
        "bootstrap": {"seed": SEED, "samples": BOOTSTRAP_SAMPLES},
        "comparisons": comparisons,
        "transparency_transitions": transitions,
        "case_audit": {
            "bge_rescues_rrf": _case_ids(cases, metrics, winner="bge_topk", loser="rrf_topk"),
            "bge_harms_rrf": _case_ids(cases, metrics, winner="rrf_topk", loser="bge_topk"),
            "dpp_rescues_qwen": _case_ids(cases, metrics, winner="qwen_dpp_low", loser="qwen_topk"),
            "dpp_harms_qwen": _case_ids(cases, metrics, winner="qwen_topk", loser="qwen_dpp_low"),
            "dpp_rescues_bge": _case_ids(cases, metrics, winner="bge_dpp_low", loser="bge_topk"),
            "dpp_harms_bge": _case_ids(cases, metrics, winner="bge_topk", loser="bge_dpp_low"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render_markdown(analysis), encoding="utf-8")
    print(args.output_json)
    print(args.output_markdown)
    return 0


def _values(
    cases: list[dict[str, object]],
    metrics: dict[str, dict[str, dict[str, object]]],
    variant: str,
    key: str,
) -> NDArray[np.float64]:
    return np.asarray(
        [float(metrics[str(case["case_id"])][variant][key]) for case in cases],
        dtype=np.float64,
    )


def _paired_bootstrap(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> dict[str, float]:
    difference = left - right
    generator = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    block = 1_000
    for start in range(0, BOOTSTRAP_SAMPLES, block):
        count = min(block, BOOTSTRAP_SAMPLES - start)
        indices = generator.integers(0, len(difference), size=(count, len(difference)))
        means[start : start + count] = difference[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean_delta": float(np.mean(difference)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def _case_ids(
    cases: list[dict[str, object]],
    metrics: dict[str, dict[str, dict[str, object]]],
    *,
    winner: str,
    loser: str,
) -> list[str]:
    result = []
    for case in cases:
        if not bool(case["target_in_candidate_pool"]):
            continue
        case_id = str(case["case_id"])
        winner_rank = metrics[case_id][winner]["target_rank"]
        loser_rank = metrics[case_id][loser]["target_rank"]
        if winner_rank is not None and loser_rank is None:
            result.append(case_id)
    return result


def _render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Ranking Strategy v0：配对统计审计",
        "",
        f"案例数：{analysis['case_count']}；bootstrap："
        f"{analysis['bootstrap']['samples']} 次，seed={analysis['bootstrap']['seed']}。",
        "",
        "## 方案对比",
        "",
        "表中均为左方案减右方案；MRR/大类数越高越好，商品相似度越低越好。",
        "",
        "| 左方案 | 右方案 | ΔMRR [95% CI] | Δ商品相似度 [95% CI] | Δ大类数 [95% CI] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in analysis["comparisons"]:
        lines.append(
            f"| {item['left']} | {item['right']} | {_interval(item['mrr_at_10'])} | "
            f"{_interval(item['mean_pairwise_product_cosine'])} | "
            f"{_interval(item['unique_reporting_groups'])} |"
        )
    lines.extend(
        [
            "",
            "## 低 T / 高 T 响应",
            "",
            "| 方法 | 低 T 更分散比例 | Δ相似度 [95% CI] | ΔMRR [95% CI] |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in analysis["transparency_transitions"]:
        lines.append(
            f"| {item['family']} | {item['low_is_more_vector_diverse_rate']:.3f} | "
            f"{_interval(item['low_minus_high_similarity'])} | "
            f"{_interval(item['low_minus_high_mrr_at_10'])} |"
        )
    audit = analysis["case_audit"]
    lines.extend(
        [
            "",
            "## Top-10 救回 / 伤害案例",
            "",
            f"- BGE 相对 RRF 救回：{len(audit['bge_rescues_rrf'])}；"
            f"伤害：{len(audit['bge_harms_rrf'])}。",
            f"- 低 T DPP 相对 Qwen Top-K 救回：{len(audit['dpp_rescues_qwen'])}；"
            f"伤害：{len(audit['dpp_harms_qwen'])}。",
            f"- 低 T DPP 相对 BGE Top-K 救回：{len(audit['dpp_rescues_bge'])}；"
            f"伤害：{len(audit['dpp_harms_bge'])}。",
            "",
        ]
    )
    return "\n".join(lines)


def _interval(value: dict[str, float]) -> str:
    return f"{value['mean_delta']:+.4f} [{value['ci95_lower']:+.4f}, {value['ci95_upper']:+.4f}]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/retrieval/ranking-strategy-v0.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/retrieval/ranking-strategy-analysis-v0.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/retrieval/ranking-strategy-analysis-v0.md"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
