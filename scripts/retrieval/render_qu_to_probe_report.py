"""Render a human-readable report from a QU-to-Probe evaluation artifact."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(report), encoding="utf-8", newline="\n")
    print(f"wrote {output}")
    return 0


def _render(report: dict[str, Any]) -> str:
    turns: list[dict[str, Any]] = report["turns"]
    summary: dict[str, Any] = report["summary"]
    runtime: dict[str, Any] = report["runtime"]
    lines = [
        "# QU → Probe 全链路实测报告",
        "",
        "> 本报告由真实 DeepSeek QU 调用和本地 50k 商品检索运行生成，不是手填示例。",
        "",
        "## 覆盖范围",
        "",
        f"- 模型：`{runtime['model']}`",
        f"- 总 turn：{summary['selected_turn_count']}",
        f"- QU 成功：{summary['qu_success_count']}",
        f"- 完整链路成功：{summary['pipeline_success_count']}",
        f"- 可计算 C_t：{summary['ct_available_count']}",
        f"- 不可计算 C_t：{summary['ct_unavailable_count']}",
        f"- 明确错误：{summary['error_count']}",
        f"- QU token：{summary['successful_turn_token_usage'].get('total_tokens', 0)}",
        "",
        "数据集：",
        "",
    ]
    for suite in report["suite_inventory"]:
        lines.append(
            f"- `{suite['suite_id']}`：{suite['conversation_count']} 段对话，"
            f"{suite['turn_count']} turn"
        )

    lines.extend(["", "## 总体 C_t", ""])
    lines.extend(
        [
            "| 指标 | 实测值 |",
            "|---|---:|",
            f"| 最小值 | {_number(summary['ct_min'])} |",
            f"| 中位数 | {_number(summary['ct_median'])} |",
            f"| 平均值 | {_number(summary['ct_mean'])} |",
            f"| 最大值 | {_number(summary['ct_max'])} |",
            "",
            "| C_t 区间 | 数量 |",
            "|---|---:|",
        ]
    )
    for label, count in summary["ct_bins"].items():
        lines.append(f"| `{label}` | {count} |")

    lines.extend(["", "## 按数据集", ""])
    lines.extend(
        [
            "| 数据集 | turn | 完整成功 | C_t 可用 | C_t 平均 | C_t 中位数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort in ("natural", "simulator"):
        selected = [item for item in turns if item["cohort"] == cohort]
        successes = [item for item in selected if item["status"] == "success"]
        values = _certainty_values(successes)
        lines.append(
            f"| {cohort} | {len(selected)} | {len(successes)} | {len(values)} | "
            f"{_number(statistics.fmean(values) if values else None)} | "
            f"{_number(statistics.median(values) if values else None)} |"
        )

    lines.extend(["", "## Simulator 随对话轮次的变化", ""])
    lines.extend(
        [
            "| turn | C_t 可用 | C_t 平均 | C_t 中位数 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for turn_number in range(1, 5):
        selected = [
            item
            for item in turns
            if item["cohort"] == "simulator"
            and item["turn"] == turn_number
            and item["status"] == "success"
        ]
        values = _certainty_values(selected)
        lines.append(
            f"| {turn_number} | {len(values)} | "
            f"{_number(statistics.fmean(values) if values else None)} | "
            f"{_number(statistics.median(values) if values else None)} |"
        )

    lines.extend(["", "## 手写的‘模糊 → 具体’配对", ""])
    lines.extend(
        [
            "| 对话 | 模糊 C_t | 具体 C_t | 变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in summary["clarity_story_pairs"]:
        lines.append(
            f"| `{pair['conversation_id']}` | {_number(pair['vague_ct'])} | "
            f"{_number(pair['specific_ct'])} | {_signed_number(pair['delta'])} |"
        )

    lines.extend(["", "## 失败、跳过和不可检索", ""])
    abnormal = [item for item in turns if item["status"] != "success"]
    if abnormal:
        lines.extend(
            [
                "| 数据集 | 对话 / turn | 状态 | 阶段 | 用户输入 | 错误 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for item in abnormal:
            error = item.get("error") or {}
            lines.append(
                f"| {item['cohort']} | `{item['conversation_id']} / {item['turn']}` | "
                f"{item['status']} | {item.get('stage') or '—'} | "
                f"{_cell(item['user_message'])} | {_cell(error.get('message') or '—')} |"
            )
    else:
        lines.append("无。")

    unavailable = [
        item
        for item in turns
        if item["status"] == "success" and item["probe"]["certainty"] is None
    ]
    lines.extend(["", "## C_t 不可计算的成功链路", ""])
    if unavailable:
        lines.extend(
            [
                "| 数据集 | 对话 / turn | 硬筛后候选 | 原因 | 用户输入 |",
                "|---|---|---:|---|---|",
            ]
        )
        for item in unavailable:
            lines.append(
                f"| {item['cohort']} | `{item['conversation_id']} / {item['turn']}` | "
                f"{item['mask']['eligible_count']} | "
                f"{_cell(', '.join(item['probe']['diagnostic_reasons']))} | "
                f"{_cell(item['user_message'])} |"
            )
    else:
        lines.append("无。")

    lines.extend(["", "## 全部 turn 的实测值", ""])
    for cohort in ("natural", "simulator"):
        lines.extend(
            [
                f"### {cohort}",
                "",
                "| 对话 / turn | 状态 | C_t | G_mode | 候选 | D_t | 放宽 | 用户输入 |",
                "|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for item in turns:
            if item["cohort"] != cohort:
                continue
            probe = item.get("probe") or {}
            mask = item.get("mask") or {}
            lines.append(
                f"| `{item['conversation_id']} / {item['turn']}` | {item['status']} | "
                f"{_number(probe.get('certainty'))} | {_number(probe.get('mode_coherence'))} | "
                f"{mask.get('eligible_count', '—')} | {probe.get('diagnostic_status', '—')} | "
                f"{'是' if mask.get('hard_filter_relaxed') else '否'} | "
                f"{_cell(item['user_message'])} |"
            )
        lines.append("")

    status_counts = Counter(item["status"] for item in turns)
    lines.extend(
        [
            "## 完整性校验",
            "",
            f"- 状态计数：`{dict(sorted(status_counts.items()))}`",
            f"- 状态合计：{sum(status_counts.values())}",
            f"- 报告总 turn：{summary['selected_turn_count']}",
            "",
        ]
    )
    return "\n".join(lines)


def _certainty_values(items: list[dict[str, Any]]) -> list[float]:
    return [
        item["probe"]["certainty"]
        for item in items
        if item.get("probe") is not None and item["probe"]["certainty"] is not None
    ]


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _signed_number(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
