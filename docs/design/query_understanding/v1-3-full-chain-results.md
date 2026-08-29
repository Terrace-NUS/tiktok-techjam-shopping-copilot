# Query Understanding v1.3 全量链路结果

- 日期：**2026-08-29**
- 模型：**DeepSeek V4 Flash**
- Prompt：**`query_understanding_v1_3`**
- 语料：**60 段自然语言对话，130 个用户回合**
- QU→Probe artifact：[`../../../artifacts/retrieval/qu-to-probe-intent-space-natural-v3.json`](../../../artifacts/retrieval/qu-to-probe-intent-space-natural-v3.json)
- Intent Volume artifact：[`../../../artifacts/retrieval/fuzzy-intent-volume-natural-v3.json`](../../../artifacts/retrieval/fuzzy-intent-volume-natural-v3.json)
- 自动摘要：[`../../../artifacts/retrieval/fuzzy-intent-volume-natural-v3.md`](../../../artifacts/retrieval/fuzzy-intent-volume-natural-v3.md)

## 1. 实际跑了什么

这不是只测模型能否输出 JSON，也不是手写 Session Context。每个回合都实际经过：

```text
用户自然语言
→ DeepSeek 原生 function call
→ 本地 decoder 与确定性 materializer
→ Session Context Gateway preview / commit
→ Query Compiler
→ hard-mask resolver
→ 50k catalog Probe
→ 离线 Fuzzy Intent Volume / T_t 方向评估
```

本轮使用 CPU 运行本地 embedding 与 50k 商品计算；DeepSeek 请求仍是真实 API 调用。Intent Volume
阶段复用已保存的 Session Context，不会再次调用模型，也不会修改原始 catalog、semantic release 或
dense index。

## 2. QU 与完整链路结果

| 指标 | v1.2 扩大实验 | v1.3 全量重跑 |
| --- | ---: | ---: |
| 选择的回合 | 130 | 130 |
| QU 成功 | 120 | **130** |
| 一次 function call 成功 | — | **130** |
| Repair | — | **0** |
| QU / pipeline error | 9 | **0** |
| 前序失败导致跳过 | 1 | **0** |
| 进入 Query Compiler 与 Probe | 120 | **127** |
| 合理判定为暂不可检索 | 0 | **3** |
| 历史失败语义断言 | 0 / 10 critical turns | **9 / 9 critical turns** |

9 个 critical turn 一共携带 49 条 retrieval-changing assertions；本轮全部通过。它们覆盖数值条件、
未知 don't-care、facet alias、保留同 facet 子条件、批量撤销旧约束和 stale goal 清理。

三个 `not_searchable` 不是协议失败，而是首轮尚未说出可落到商品类别的目标：

- `I need something to wear to a wedding, but I'm still figuring out the look.`
- `I'm after a classic gift for a graduation.`
- `I want something more comfortable to sleep in.`

系统保留这些输入，但不伪造商品 goal；三段对话的后续具体表达都能继续正常处理。

## 3. `T_t` / Intent Volume 方向结果

下表使用随后落成 runtime v1 的配置：
`soft_hybrid_d0.025_q0.850_m0.060_h0.010`。它将结构化条件表示为柔性 evidence membership，
将开放语义条件作为独立因子，以 Product of Experts 求交，并用 catalog density 给重复 listing 降权。

| 对话变化 | 可评分 | 方向正确 |
| --- | ---: | ---: |
| 增加条件，空间应收紧 | 33 | **33 / 33** |
| 撤销条件，空间应放宽 | 10 | **10 / 10** |
| 无关信息或只改展示，空间应稳定 | 7 | **7 / 7** |
| 完全换目标 | 7 | 只观察，不规定升降 |
| 三轮渐进收紧的相邻步骤 | 20 | **20 / 20** |

语料原本有 36 段 narrower 对话；其中 3 段的首轮就是上面的 `not_searchable`，没有可信的首轮商品
空间作为分母，因此不强行把它们计成成功或失败。剩余 50 个首末方向判断全部符合预期。

和 v1.2 相比，最重要的变化是 broader 从只有 4 段可完整评分，提升到全部 **10/10** 完成且方向正确。
这说明 v1.3 修复的不只是 function-call 形状，也修复了撤销后旧条件残留造成的空间方向错误。

## 4. 仍然不能夸大的地方

这次结果足以说明当前 QU→Session Context→Intent Volume 链路在固定语料上已经跑通，而且收紧、
放宽与稳定行为一致。但它还不是最终算法冻结证据：

- 270 组参数是在同一套语料上扫描的，runtime v1 配置不是独立 held-out 选出的；
- Product of Experts 的方向正确部分来自状态代数，不能单独证明商品语义准确；
- 主候选 Top-20 平均满足约 **80.9%** 的可验证 hard facets，但同时满足全部 hard facets 的平均比例
  只有约 **38.8%**；这是柔性空间测量，不应冒充最终严格商品检索；
- 三个首轮模糊表达没有可测空间，说明 `T_t` 必须允许 unavailable，并由 `D_t` 解释原因；
- `T_t` 的 0–1 展示刻度和参数现已冻结为 hackathon runtime v1，但不能外推成通用生产标定。

因此本轮最准确的结论是：

> v1.3 已经消除了扩大语料中已知的 QU 协议与状态更新断层；当前 Fuzzy Intent Volume 能在全部
> 可评分对话上表现正确的空间方向，但参数标定和商品相关性仍需与最终召回分开陈述。

后续运行时实现与协议见
[`../intent_purity/runtime-contract-v1.md`](../intent_purity/runtime-contract-v1.md)。正式组件对全部 127 个
可检索状态重放后，与本实验配置的 transparency 最大绝对误差小于 `4e-7`，没有 parity failure。

## 5. 复现命令

```powershell
.\.venv-3.10\Scripts\python.exe scripts\retrieval\evaluate_qu_to_probe.py `
  --cohort natural `
  --tier full `
  --natural-suite config\query_understanding\intent-space-natural-prompts-v2.json `
  --api-key-file dpskapi `
  --release artifacts\catalog-semantic\release-v0 `
  --dense-index artifacts\retrieval\dense-v0 `
  --calibration config\retrieval\transparency-calibration-v1.json `
  --device cpu `
  --output artifacts\retrieval\qu-to-probe-intent-space-natural-v3.json

.\.venv-3.10\Scripts\python.exe scripts\retrieval\evaluate_fuzzy_intent_volume.py `
  --evaluation artifacts\retrieval\qu-to-probe-intent-space-natural-v3.json `
  --dense-index artifacts\retrieval\dense-v0 `
  --release artifacts\catalog-semantic\release-v0 `
  --density-cache artifacts\retrieval\intent-volume-density-v0.npz `
  --output artifacts\retrieval\fuzzy-intent-volume-natural-v3.json `
  --markdown artifacts\retrieval\fuzzy-intent-volume-natural-v3.md `
  --device cpu `
  --cohort natural
```
