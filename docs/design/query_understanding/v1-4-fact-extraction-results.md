# Query Understanding v1.4：description fact extraction 结果

- 日期：**2026-08-30**
- 模型：**DeepSeek V4 Flash**
- Prompt：**`query_understanding_v1_4`**
- 调用方式：**native forced function call，thinking disabled，temperature 0**

## 改了什么

v1.4 没有改变 Session Context schema、reducer 或 tool-call 外层协议。改动集中在模型语义协议：

- 自由描述可一次抽出多个商品或商品部件 facet；
- 抽取前判断事实主语和否定范围；
- `meaning` 保存标准化含义，`values/evidence` 保留可验证原文锚点；
- 类别叶子词不再重复生成 feature；
- gender 封闭值规范为 `men/women`；
- `basis=explicit` 与 `strength=hard` 分离；普通描述默认 soft，明确强制或排除才 hard。

这些规则同时写入 system prompt 和 DeepSeek native tool schema descriptions。

## 专项前后对照

固定语料为 10 段 / 11 轮、25 条关键语义断言。v1.3 与最终 v1.4 使用相同语义 case；v1.3
报告中的一条 `cotton` 等值断言后来修正为接受信息更完整的 `100% cotton`，因此表中保留原始留档数字，
不虚构重算后的 artifact。

| 指标 | v1.3 基线 | v1.4 最终 |
| --- | ---: | ---: |
| Tool-call / contract success | 11 / 11 | 11 / 11 |
| Critical semantic assertions | 18 / 25 | **25 / 25** |
| Critical turns | 7 / 11 | **11 / 11** |
| Repair | 0 | 0 |
| 最终状态 hard / soft 数量 | 11 / 6 | **2 / 14** |

v1.4 保留的两条 hard 分别来自明确 `must` 和 `no wool`，不是机械地把所有条件降成 soft。

Artifacts：

- `artifacts/query-understanding/fact-extraction-v1-3-baseline.json`
- `artifacts/query-understanding/fact-extraction-v1-4-final.json`

## 旧能力回归

对 v1.3 曾失败过的 12 段 / 24 轮原样重放：

| 指标 | v1.4 |
| --- | ---: |
| Tool-call / contract success | **24 / 24** |
| Critical semantic assertions | **49 / 49** |
| Critical turns | **9 / 9** |
| Repair | **0** |

这组覆盖撤销条件、部分保留同 facet、数值条件、goal revise/switch 和中英文表达。

Artifact：`artifacts/query-understanding/qu-v1-4-historical-regressions-live.json`。

自然语言 smoke 的首次 v1.4 运行是 20/20 contract、58/60 assertions；失败是 `men's` 未规范为
`men`，以及 `Women's size 8` 漏抽 gender。补充封闭 gender 规则后，两个原对话共 4 轮、18 条断言
原样复测全部通过。Artifact：

- `artifacts/query-understanding/natural-smoke-v1-4.json`
- `artifacts/query-understanding/natural-two-regressions-v1-4.json`

## 原 hard-mask 失败的端到端复测

从 200-task 日志中取出原先最终状态会删除隐藏目标的 7 个任务。评测仍然不向 QU、Probe、Retrieval
或 Ranking 暴露目标；目标只由 evaluator 在输出完成后检查。为避免提前命中掩盖后续状态，本次启用
`continue_after_hit`，每个任务强制走满 5 轮。

| 最终结果 | 数量 |
| --- | ---: |
| 最终没有 hard constraints，目标仍可参与检索 | **5 / 7** |
| 仍被 hard material include 排除 | **2 / 7** |

已恢复的五类包括：description 默认 soft、保留 composition/measurement 原文、Rain 类别不重复生成
feature，以及商品 metadata 的 gender 冲突不再被普通描述升级成 hard。

剩余两项：

- `public_0029`：首轮明确 `key requirement=fabric`；目标的 fabric 证据只在 description；
- `public_0154`：首轮明确 `key requirement=cotton`；目标的 cotton 证据只在 description。

这两项不能靠继续放松 QU 治疗。用户确实表达了 hard，下一步应让商品侧用与 QU 一致的 facet 协议从
description 生成带来源的结构化事实 sidecar，而不是修改原始 catalog。

完整强制五轮日志：`artifacts/simulator/qu-v1-4-hard-mask-seven-forced-five/`。
