# Query Understanding

- 契约：**v1.3；针对真实放宽对话补齐了可执行边界**
- 实现：**完整状态 reconciliation、定向修复、Query Compiler / Probe 集成已完成**
- 模型：**DeepSeek V4 Flash 原生 function call**
- 冻结日期：**2026-08-28**

Query Understanding 让 DeepSeek 在每一轮修复完整的购物意图。本地代码把这个完整目标状态
转换成现有 Session Context operation，通过 Gateway preview 后，将被接受的最终意图交给
Query Compiler。

当前模型输出不再使用一个容易混填字段的万能 preference 数组，而是分别提交
`structured`、`price`、`semantic` 三组新增条件；本地 materializer 再统一生成最终状态。

```text
natural language
    -> DeepSeek reconcile_session_intent tool call
    -> complete target intent
    -> deterministic materializer + Gateway preview
    -> compiled query
    -> fixed Probe
    -> C_t
```

## 文档

- [`session-context-flow-example.md`](session-context-flow-example.md)：用一个完整中文案例逐步展示旧
  Session Context、DeepSeek 实际输入、原生 function call、本地 materialization 和最终提交边界。
- [`contract-v1.md`](contract-v1.md)：当前冻结契约，定义完整目标状态工具、宽 structured
  facet、双 authority、operation 顺序、一次修复、事务集成和下游边界。
- [`prompt-evaluation-v0.md`](prompt-evaluation-v0.md)：自然语言与官方 simulator 两套固定语料、
  生成边界、运行方法和 DeepSeek V4 Flash smoke 结果。
- [`failure-analysis-v2.md`](failure-analysis-v2.md)：扩大自然语言实测暴露的协议断层、失败证据和
  v1.3 修复依据。
- [`v1-3-regression-results.md`](v1-3-regression-results.md)：原失败样本的真实 DeepSeek 修复后回归，
  24/24 回合与 49/49 关键语义断言通过。
- [`v1-3-full-chain-results.md`](v1-3-full-chain-results.md)：60 段、130 回合真实全量重跑；QU
  130/130，全部 10 段放宽对话完成且 Intent Volume 方向正确。
- [`contract-v0.md`](contract-v0.md)：已被 v1 取代，保留为设计历史和研究记录。
- [`../query_compiler/contract-v0.md`](../query_compiler/contract-v0.md)：preview 后的最终意图如何
  确定性编译为检索视图，并接入固定 Dense Probe。

## 当前边界

- DeepSeek 决定语义状态变化，但不分配 ID，也不提交 Session Context。
- `price` 和 product category 继续使用 catalog-verified authority。
- brand、material、color、size、style、department、gender、feature 和 use case
  是 structured retrieval-derived facet。
- 未知维度会自动退回 semantic-only，不会导致整轮失败。
- 非价格数值范围也会保留完整含义并退回 semantic-only，而不是丢失或让整轮失败。
- 模型只能从请求给出的合法列表设置 don't-care；未注册的子属性 marker 会被忽略并进入 trace，
  删除单条旧条件仍由省略对应 `active_N` 完成。
- 同一商品任务可以用 `goal.revise` 清理旧 goal 中已经取消的约束；真正换商品才使用 `switch`。
- 同一 categorical facet 上的第二个独立正向需求会保留为 semantic-only，避免因表示能力限制
  丢失整轮意图。
- “增加多样性”等直接行为指令保留为当前轮 sidecar。
- Retrieval 只消费 preview 后的最终意图；Probe 只在 Query Understanding 和查询编译完成后
  计算 `C_t`。

当前代码位于 `src/shopping_copilot/query_understanding/`。普通测试完全离线；真实 DeepSeek
调用必须由应用显式传入 API key。
