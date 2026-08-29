# Query Understanding v1.3 回归结果

- 日期：**2026-08-29**
- 模型：**DeepSeek V4 Flash**
- Prompt：**`query_understanding_v1_3`**
- 结果 artifact：[`../../../artifacts/query-understanding/qu-v1-3-regressions-live.json`](../../../artifacts/query-understanding/qu-v1-3-regressions-live.json)

## 结果

本轮选择了 v1.2 扩大测试中所有实际失败过的对话，并加入 3 个虽然通过协议、但曾发生 stale goal
约束泄漏的放宽对话，共 12 段对话、24 个真实 DeepSeek 回合。

| 指标 | v1.3 结果 |
| --- | ---: |
| Contract success | 24 / 24 |
| 一次调用成功 | 24 / 24 |
| Repair | 0 |
| Repair exhausted | 0 |
| Critical semantic assertions | 49 / 49 |
| Critical turns | 9 / 9 |

这不是用 fake provider 得到的结果，而是 DeepSeek 原生 function call 经过本地 decoder、materializer 和
真实 Catalog Semantic Gateway preview 后的最终状态。

## 覆盖的原问题

- `40 mm or smaller` 不再触发非法 structured relation，并被完整保存为 semantic-only preference；
- `length`、`heel_height`、`stones` 等未注册 don't-care marker 不再导致整轮失败；
- `metal` 等常见叫法可以映射到正式 facet；
- 取消一个 feature 子条件时，不会顺便删除同 facet 下仍明确保留的 waterproof；
- 放宽颜色、材质、尺寸、功能和预算后，最终 preferences 不再残留旧条件；
- `goal` 使用最短商品任务，或通过 `revise` 清理同一商品任务里的旧约束；
- QU→Probe 错误报告会保留 error code、typed path 和安全 details。

## 一个复测中追加的规则

第一次 v1.3 留档复测中，DeepSeek 曾同时：

1. 在 `keep_active_refs` 中保留 waterproof；
2. 又把整个 `feature` 放进 `dont_care_facets`。

如果让粗粒度 don't-care 获胜，就会误删 waterproof。最终规则因此确定为：

> 同一 frame 中，具体保留 ref 与整 facet don't-care 冲突时，保留 ref 获胜；冲突 marker 被忽略并写入 trace。

正确表达“整个 facet 都无所谓”时，模型仍必须省略该 facet 的全部旧 ref。

## 后续全量结果

原失败样本通过后，又原样重跑了全部 60 段、130 回合自然语言语料。结果为 QU 130/130、
协议与 pipeline error 0、9/9 critical turns 通过；完整结果见
[`v1-3-full-chain-results.md`](v1-3-full-chain-results.md)。
