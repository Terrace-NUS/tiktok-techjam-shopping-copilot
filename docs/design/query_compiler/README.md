# Query Compiler

- 契约：**v0，已冻结**
- 实现：**确定性编译器与固定 Dense Probe 入口已完成**
- 日期：**2026-08-28**

Query Compiler 是 Query Understanding 和 Retrieval 之间唯一的翻译层。它不调用模型，也不重读
完整对话；它只把已经通过 Gateway preview 的最终意图翻译成检索所需的几个视图。

- [`contract-v0.md`](contract-v0.md)：输入输出、编译规则、trace 和当前实现边界。
- 上游：[`../query_understanding/contract-v1.md`](../query_understanding/contract-v1.md)
- 下游：[`../retrieve/contract-v0.md`](../retrieve/contract-v0.md)

当前代码位于 `src/shopping_copilot/query_compiler/`。固定 Probe 的薄适配器位于
`src/shopping_copilot/retrieval/compiled_probe.py`。
