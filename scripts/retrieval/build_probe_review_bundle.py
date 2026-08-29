"""Build a self-contained review ZIP for the two intent-transparency algorithms."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENSE_RESULT = Path(
    "artifacts/retrieval/qu-to-probe-simulator-other-16x4-audit-v2.json"
)
DEFAULT_LEXICAL_RESULT = Path(
    "artifacts/retrieval/lexical-semantic-coherence-audit-v2.json"
)
DEFAULT_DIAGNOSIS = Path(
    "artifacts/retrieval/transparency-transition-diagnosis-audit-v2.json"
)
DEFAULT_DEPTH_SWEEP = Path(
    "artifacts/retrieval/transparency-probe-depth-sweep-audit-v2.json"
)
DEFAULT_OUTPUT = Path(
    f"artifacts/review-bundles/probe-algorithms-audit-{date.today():%Y%m%d}-v1.zip"
)


def main() -> int:
    args = _parse_args()
    output = _absolute(args.output)
    staging = output.with_suffix("")
    if output.exists() or staging.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {output} or {staging}")

    dense_path = _absolute(args.dense_result)
    lexical_path = _absolute(args.lexical_result)
    diagnosis_path = _absolute(args.diagnosis)
    depth_sweep_path = _absolute(args.depth_sweep)
    dense = _load_json(dense_path)
    lexical = _load_json(lexical_path)
    diagnosis = _load_json(diagnosis_path)
    depth_sweep = _load_json(depth_sweep_path)
    validation = _validate_reports(dense, lexical)

    staging.mkdir(parents=True)
    _write_text(staging / "README.md", _readme(dense, lexical, diagnosis, depth_sweep))
    _write_text(staging / "REPRODUCE.md", _reproduce())
    _write_text(staging / "logs" / "README.md", _logs_readme())
    _write_json(staging / "logs" / "VALIDATION.json", validation)
    _write_combined_logs(staging / "logs", dense=dense, lexical=lexical)

    _copy_result(dense_path, staging / "results" / "dense-probe" / dense_path.name)
    _copy_result(
        lexical_path,
        staging / "results" / "lexical-top80-dense-coherence" / lexical_path.name,
    )
    lexical_markdown = lexical_path.with_suffix(".md")
    if lexical_markdown.exists():
        _copy_result(
            lexical_markdown,
            staging
            / "results"
            / "lexical-top80-dense-coherence"
            / lexical_markdown.name,
        )
    _copy_result(
        diagnosis_path,
        staging / "results" / "diagnostics" / diagnosis_path.name,
    )
    _copy_result(
        depth_sweep_path,
        staging / "results" / "diagnostics" / depth_sweep_path.name,
    )
    _write_text(
        staging / "results" / "SUMMARY.md",
        _results_summary(dense, lexical, diagnosis, depth_sweep),
    )

    _copy_review_sources(staging)
    manifest = _manifest(staging)
    _write_json(staging / "MANIFEST.json", manifest)
    _write_text(staging / "SHA256SUMS.txt", _sha256_sums(staging))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        root_name = staging.name
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            archive.write(path, Path(root_name) / path.relative_to(staging))

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"staging: {staging}")
    print(f"zip: {output}")
    print(f"zip_bytes: {output.stat().st_size}")
    return 0


def _validate_reports(
    dense: dict[str, Any],
    lexical: dict[str, Any],
) -> dict[str, object]:
    turns = dense.get("turns")
    if type(turns) is not list or len(turns) != 64:
        raise ValueError("Dense audit must contain exactly 64 turns")
    lexical_turns = _lexical_turns(lexical)
    if len(lexical_turns) != 64:
        raise ValueError("Lexical audit must contain exactly 64 turns")

    dense_hit_counts: list[int] = []
    lexical_hit_counts: list[int] = []
    for item in turns:
        if item["status"] != "success":
            raise ValueError("all packaged audit turns must be successful")
        identity = (item["conversation_id"], item["turn"])
        lexical_turn = lexical_turns.get(identity)
        if lexical_turn is None:
            raise ValueError(f"Lexical audit is missing turn {identity!r}")
        before = item["session_context_before"]
        after = item["session_context_after"]
        if before["schema"] != "shopping-copilot/session-context/v1":
            raise ValueError(f"invalid before Session Context schema at {identity!r}")
        if after["schema"] != "shopping-copilot/session-context/v1":
            raise ValueError(f"invalid after Session Context schema at {identity!r}")
        after_history = after["payload"]["state"]["interaction"]["turns"]
        if len(after_history) != item["turn"]:
            raise ValueError(f"incomplete Session Context history at {identity!r}")
        request = item["deepseek_request_payload"]
        if request["latest_utterance"] != item["user_message"]:
            raise ValueError(f"DeepSeek input differs from logged message at {identity!r}")
        compiled = item["compiled"]
        if not compiled["q_lex"] or not compiled["q_sem"]:
            raise ValueError(f"compiled search text is missing at {identity!r}")
        dense_hit_counts.append(len(item["probe"]["ranking_hits"]))
        lexical_hit_counts.append(len(lexical_turn["lexical_hit_ids"]))

    summary = dense["summary"]
    lexical_summary = lexical["summary"]
    return {
        "schema": "shopping-copilot/probe-review-bundle-validation/v1",
        "dense_report_schema": dense["schema"],
        "dense_turn_count": len(turns),
        "dense_pipeline_success_count": summary["pipeline_success_count"],
        "dense_ct_available_count": summary["ct_available_count"],
        "lexical_turn_count": lexical_summary["turn_count"],
        "lexical_available_count": lexical_summary["lexical_available_count"],
        "lexical_mode_coherence_available_count": lexical_summary[
            "mode_coherence_available_count"
        ],
        "session_context_v1_before_after_present": True,
        "deepseek_request_payload_present": True,
        "compiled_q_lex_q_sem_present": True,
        "dense_hit_count_range": [min(dense_hit_counts), max(dense_hit_counts)],
        "lexical_hit_count_range": [min(lexical_hit_counts), max(lexical_hit_counts)],
        "credentials_included": False,
    }


def _write_combined_logs(
    destination: Path,
    *,
    dense: dict[str, Any],
    lexical: dict[str, Any],
) -> None:
    lexical_turns = _lexical_turns(lexical)
    combined: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in dense["turns"]:
        identity = (item["conversation_id"], item["turn"])
        log = {
            "schema": "shopping-copilot/probe-search-turn-audit/v1",
            "identity": {
                "suite_id": item["suite_id"],
                "cohort": item["cohort"],
                "conversation_id": item["conversation_id"],
                "turn": item["turn"],
                "scenario_type": item["scenario_type"],
                "response_shape": item["response_shape"],
            },
            "user_message": item["user_message"],
            "session_context_before": item["session_context_before"],
            "query_understanding": {
                "deepseek_request_payload": item["deepseek_request_payload"],
                "provider_attempts": item["qu_attempts"],
                "resolved_turn": item["resolved_turn"],
            },
            "search_input": item["compiled"],
            "hard_mask": item["mask"],
            "algorithm_1_dense_probe": item["probe"],
            "algorithm_2_lexical_top80_dense_coherence": lexical_turns[identity],
            "session_context_after": item["session_context_after"],
            "error": item["error"],
        }
        combined.append(log)
        grouped[item["conversation_id"]].append(log)

    destination.mkdir(parents=True, exist_ok=True)
    jsonl = "\n".join(
        json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for item in combined
    )
    _write_text(destination / "every-search.jsonl", jsonl + "\n")
    for conversation_id, turns in sorted(grouped.items()):
        _write_json(
            destination / "conversations" / f"{conversation_id}.json",
            {
                "schema": "shopping-copilot/probe-conversation-audit/v1",
                "conversation_id": conversation_id,
                "turns": turns,
            },
        )


def _lexical_turns(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (conversation["conversation_id"], turn["turn"]): turn
        for conversation in report["conversations"]
        for turn in conversation["turns"]
    }


def _copy_review_sources(staging: Path) -> None:
    source_roots = (
        Path("src/shopping_copilot/retrieval"),
        Path("src/shopping_copilot/query_compiler"),
        Path("src/shopping_copilot/query_understanding"),
        Path("src/shopping_copilot/session_context"),
        Path("src/shopping_copilot/catalog/semantic"),
    )
    for root in source_roots:
        for source in sorted((_absolute(root)).rglob("*.py")):
            _copy_result(source, staging / "code" / source.relative_to(REPOSITORY_ROOT))

    scripts = (
        "scripts/retrieval/evaluate_qu_to_probe.py",
        "scripts/retrieval/evaluate_lexical_semantic_coherence.py",
        "scripts/retrieval/diagnose_transparency_transitions.py",
        "scripts/retrieval/sweep_transparency_probe_depth.py",
        "scripts/retrieval/build_probe_review_bundle.py",
        "scripts/query_understanding/generate_simulator_other_prompts.py",
    )
    docs = (
        "docs/design/retrieve/probe-v1.md",
        "docs/design/retrieve/transparency-evaluation-v1.md",
        "docs/design/retrieve/simulator-other-evaluation-v1.md",
        "docs/design/retrieve/evidence-hard-mask-v0.md",
        "docs/design/retrieve/contract-v0.md",
        "docs/team_briefing/01-session-context.md",
        "docs/team_briefing/02-facet-system.md",
        "docs/team_briefing/03-query-understanding.md",
        "docs/team_briefing/04-intent-transparency.md",
    )
    configs = (
        "config/retrieval/transparency-calibration-v1.json",
        "config/query_understanding/simulator-other-prompts-v1.json",
        "pyproject.toml",
    )
    for relative in (*scripts, *docs, *configs):
        source = _absolute(Path(relative))
        _copy_result(source, staging / "code" / Path(relative))

    test_roots = (
        Path("tests/unit/retrieval"),
        Path("tests/unit/query_compiler"),
        Path("tests/unit/query_understanding"),
        Path("tests/unit/session_context"),
    )
    for root in test_roots:
        for source in sorted((_absolute(root)).rglob("*.py")):
            _copy_result(source, staging / "code" / source.relative_to(REPOSITORY_ROOT))


def _readme(
    dense: dict[str, Any],
    lexical: dict[str, Any],
    diagnosis: dict[str, Any],
    depth_sweep: dict[str, Any],
) -> str:
    dense_summary = dense["summary"]
    lexical_summary = lexical["summary"]
    return f"""# Probe 两种算法：代码、实验结果与完整逐轮日志

这个压缩包用于交叉 review。两种算法使用同一批 DeepSeek QU 输出与同一批正式 Session Context 快照，因此可以逐 turn 直接比较。

## 数据范围

- 官方 toy simulator 的 `buying` 8 个任务 + `browsing` 8 个任务。
- 每个任务固定交互 4 轮，agent 每轮只返回 `ask_attribute=other`。
- 共 16 个会话、64 个搜索 turn。
- API 密钥、环境文件、原始商品数据集和 Dense 索引没有放入压缩包。

## 算法 1：Dense Probe（当前正式实现）

1. QU 把 Session Context 编译为 `q_sem`、`q_lex` 和 hard constraints。
2. hard mask 在 Top-K 截断前执行。
3. 用 `q_sem` 做 Dense Top-80。
4. 把相似度不低于 0.94 的近重复商品合成 semantic mode。
5. 对 mode centroid 等权计算 `G_mode`，再用冻结的 Dense calibration 映射成 `C_t`。

结果：{dense_summary['pipeline_success_count']}/64 全链路成功，{dense_summary['ct_available_count']}/64 有可用 `C_t`；中位数 `{dense_summary['ct_median']:.6f}`，均值 `{dense_summary['ct_mean']:.6f}`。

## 算法 2：Lexical Top-80 + Dense coherence（实验实现）

1. 使用同一个 hard mask。
2. 用 `q_lex` 做 BM25/FTS5 Top-80。
3. 不再按 query dense score 选商品，只查这些 lexical hit 的商品向量。
4. 用与算法 1 相同的 mode 合并与 `G_mode` 计算。

结果：{lexical_summary['lexical_available_count']}/64 有 Lexical 结果，{lexical_summary['mode_coherence_available_count']}/64 有 `G_mode`。在 {lexical_summary['comparable_attribute_disclosures']} 个可比较的 attribute-disclosure 转移中，方向为 `{lexical_summary['all_disclosure_direction']}`，平均 delta G 为 `{lexical_summary['all_disclosure_mean_delta']:.6f}`。

注意：Lexical 路线里的 `experimental_mapped_ct` 只是借用 Dense calibration 查看量级，不能当作已经校准的正式 `C_t`；有效比较量是 raw `G_mode` 及其变化方向。

## 主要结论（不回避负结果）

- Dense 路线在同一批状态上的 disclosure 方向：`{diagnosis['summary']['full_mode_direction']}`。
- Lexical 路线方向：`{lexical_summary['all_disclosure_direction']}`。
- K=80 的 Dense 平均 delta G：`{depth_sweep['summary']['80']['mean_delta']:.6f}`。
- 当前算法仍没有稳定体现“信息逐轮增加，C_t 持续收敛”的故事；第一轮信息披露后下降尤其常见。日志保留所有 case，未筛选。

## 目录

- `logs/every-search.jsonl`：64 行，每行一个完整搜索 turn。
- `logs/conversations/`：同样日志按 16 个会话拆分，便于人工阅读。
- `results/`：Dense、Lexical、2x2 transition diagnosis、Top-K depth sweep 的原始 JSON。
- `code/`：两种算法、QU、Session Context、hard mask、编译器、实验脚本、相关测试与设计说明。
- `MANIFEST.json` / `SHA256SUMS.txt`：文件清单与哈希。

## 关于“完整 Session Context”

每个 turn 都保存 `session_context_before` 与 `session_context_after`，它们经过仓库正式的 `shopping-copilot/session-context/v1` 编码器和校验器。内容包括完整 IntentState/Preference 字段、累计 interaction history、accepted update、feedback 和 SearchBelief。

toy simulator fixture 只记录 `ask_attribute`，不记录内部 `question_key`；为满足正式 Session Context 三字段不变量，审计 runner 使用可复现的 `ask_attribute:other` 作为 question key。`profile` 在本实验中没有输入，因此明确为 `null`，没有补造用户画像。
"""


def _results_summary(
    dense: dict[str, Any],
    lexical: dict[str, Any],
    diagnosis: dict[str, Any],
    depth_sweep: dict[str, Any],
) -> str:
    return f"""# 实验结果速览

## Dense Probe

```json
{json.dumps(dense['summary'], ensure_ascii=False, indent=2)}
```

## Lexical Top-80 + Dense coherence

```json
{json.dumps(lexical['summary'], ensure_ascii=False, indent=2)}
```

## 2x2 transition diagnosis

```json
{json.dumps(diagnosis['summary'], ensure_ascii=False, indent=2)}
```

## Probe depth sweep

```json
{json.dumps(depth_sweep['summary'], ensure_ascii=False, indent=2)}
```
"""


def _logs_readme() -> str:
    return """# 逐轮日志说明

`every-search.jsonl` 每行都是一个独立 JSON object，字段顺序如下：

- `identity`：suite、会话、turn、buying/browsing、response shape。
- `user_message`：本轮 simulator 返回的自然语言。
- `session_context_before`：搜索前完整、schema-validated Session Context。
- `query_understanding`：DeepSeek 实际读取的 JSON、provider token trace、完整 resolved update。
- `search_input`：完整 compiled query，包括 `q_lex`、`q_sem`、hard constraints、ranking preferences、directives 和 trace。
- `hard_mask`：mask 数量、relax 状态和逐 constraint trace。
- `algorithm_1_dense_probe`：Dense Top-80 ID/score、Lexical diagnostics、semantic modes、coherence、C_t、D_t 与 SearchBelief。
- `algorithm_2_lexical_top80_dense_coherence`：Lexical Top-80 ID、mode coherence、实验映射值及同轮 Dense 对照。
- `session_context_after`：提交本轮 update 并写入 SearchBelief 后的完整 Session Context。

日志没有 raw DeepSeek tool arguments，因为 provider boundary 有意不保存原始参数；保存的是 materializer 接受后的完整 update、最终意图与 trace。日志不包含 API key。
"""


def _reproduce() -> str:
    return """# 复现实验

在仓库根目录、Python 3.10 venv 已安装依赖、`dpskapi` 含 DeepSeek key 的前提下：

```powershell
.\\.venv-3.10\\Scripts\\python.exe scripts/retrieval/evaluate_qu_to_probe.py `
  --tier full `
  --cohort simulator `
  --simulator-suite config/query_understanding/simulator-other-prompts-v1.json `
  --simulator-limit-per-scenario 8 `
  --api-key-file dpskapi `
  --output artifacts/retrieval/qu-to-probe-simulator-other-16x4-audit-v2.json

.\\.venv-3.10\\Scripts\\python.exe scripts/retrieval/evaluate_lexical_semantic_coherence.py `
  --input artifacts/retrieval/qu-to-probe-simulator-other-16x4-audit-v2.json `
  --output artifacts/retrieval/lexical-semantic-coherence-audit-v2.json
```

DeepSeek 输出不是确定性资产，即使 temperature=0，未来重跑仍可能与本包不同。包内两种算法结果来自同一份已保存 QU 状态，逐轮对照是确定的。
"""


def _manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "shopping-copilot/probe-review-bundle-manifest/v1",
        "file_count_excluding_manifest_and_sums": len(files),
        "files": files,
    }


def _sha256_sums(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_result(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_json(path: Path, value: object) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-result", type=Path, default=DEFAULT_DENSE_RESULT)
    parser.add_argument("--lexical-result", type=Path, default=DEFAULT_LEXICAL_RESULT)
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_DIAGNOSIS)
    parser.add_argument("--depth-sweep", type=Path, default=DEFAULT_DEPTH_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
