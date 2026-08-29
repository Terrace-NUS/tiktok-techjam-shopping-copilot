# Query Understanding Prompt Evaluation v0

- 状态：**P0 suite 与 v1.2 live smoke 已完成**
- 日期：**2026-08-28**
- 模型：**DeepSeek V4 Flash**
- 当前 prompt：**`query_understanding_v1_2`**

## 1. 这套测试在回答什么

它把两个问题分开：

1. 面对较自然、多轮、会纠正自己的购物表达，QU 是否保留了会影响后续检索的语义；
2. 面对官方 toy simulator 的固定说话方式，QU 能否稳定生成合法、可被 Gateway 预演的完整
   `IntentState`。

它不测 target recall，也不把 toy simulator 当成真实语言分布。官方 scorer 的拿分路线与这套
story-oriented QU 测试仍是两条独立路径。

## 2. 两个独立 cohort

| Cohort | Conversation | User turns | Smoke | 用途 |
| --- | ---: | ---: | ---: | --- |
| 人工自然语言 | 40 | 72 | 12 组 / 20 轮 | 语义 oracle |
| 官方 simulator | 32 | 128 | 4 组 / 16 轮 | 官方接口与模板兼容 |

人工集覆盖英文、中文和 code-switching，并包含：

- 明确条件、软偏好、排除、多值备选和价格区间；
- 条件替换、撤销、dont-care、换商品目标与有选择地保留旧条件；
- 上一问的省略回答、shown-product 指代、feedback、比较/解释/多样性指令；
- 双重否定、含糊表达、prompt injection 文本，以及“模糊 → 具体”的四组演示链。

关键断言不比较完整 JSON，也不比较 confidence、summary 或 evidence 的逐字文本。`eq(x)` 与
`in([x])`、`neq(x)` 与 `not_in([x])` 按同一语义族判断；hard/soft 只有在用户措辞明确时才作为
关键断言。对目标名称允许等义的中英文或自然变体，避免把“语义正确、措辞不同”误判为失败。

## 3. 官方 prompt 是怎样得到的

[`generate_simulator_prompts.py`](../../../scripts/query_understanding/generate_simulator_prompts.py)
没有仿写官方模板，而是用固定提问策略的 `CaptureAgent` 直接驱动
`evaluator.local_evaluator.evaluate()`，录下 evaluator 实际传给 `Agent.respond()` 的
`user_message`。

- buying、browsing、intent_override、boundary 各稳定选 8 个 session；
- 每个 session 保留前 4 个可见轮次；
- 先按公开的 scenario 分桶；桶内样本选择和 ask schedule 只由 `sample_id` 的 SHA-256
  决定，不查看 hidden intent；
- `--check` 会重新运行 simulator，并要求结果与已提交 fixture 字节完全一致。

安全边界：fixture 中不允许出现 target ASIN、`ground_truth`、商品对象、hidden intent card、
未披露约束、behavior 或 reset profile。`sample_id/scenario/difficulty` 只存在于 runner-side
provenance；单测会确认这些值不进入 `request_payload()`。

当前冻结 identity：

```text
natural-prompts-v0.json
  sha256:b59580be67de6bc503092dfb58827121ddfacd43fe51ede89edee9a57b3ad902

simulator-prompts-v0.json
  sha256:3ef69b2c602251c3218313312e1defeab12fc2a9980eaa070e314b89c706609c
```

## 4. Runner

[`evaluate_prompts.py`](../../../scripts/query_understanding/evaluate_prompts.py) 支持：

- `--validate-only`：只做严格 schema、唯一性和泄漏检查，不读取 API key 或 semantic release；
- `--tier smoke`：只跑 smoke conversation；
- `--tier full`：跑完整 200 轮；
- `--cohort natural|simulator|all`：单独或合并执行；
- API key 只能由显式 `--api-key-file` 或 `--api-key-env` 提供；
- 每个 conversation 从空状态或 fixture 给出的初始 goal 开始，并用模型实际输出连续 rollout；
- 一轮失败只阻断当前 conversation，其他 case 继续；
- 报告记录 suite、prompt、tool schema、model config 和 semantic release identity，以及延迟、
  token、repair、facet/strength 分布和逐轮结果，但不记录 credential 或原始 tool arguments。

## 5. DeepSeek live smoke 结果

### 5.1 v1.1 历史基线

报告：`artifacts/query-understanding/deepseek-smoke-v0.json`（本地生成物，不进 Git）。

一体化 `new_preferences[]` wire 的结果为：

| 指标 | v1.1 |
| --- | ---: |
| 完整 contract + materialization + Gateway success | 33 / 36 = **91.67%** |
| 人工自然语言 end-to-end critical turns | 18 / 20 = **90%** |
| 官方 simulator compatibility | 15 / 16 = **93.75%** |
| repair exhausted | **2** |

两个失败都来自万能 preference 对象的跨字段组合：relation、facet、values 和 numeric value 各自
合法，但组合在一起不合法。普通 JSON Schema 无法简洁表达全部条件依赖，一次 repair 也没有
稳定纠正它们。

### 5.2 v1.2 typed-wire 最终实测

最终报告：`artifacts/query-understanding/deepseek-smoke-v1_2-final.json`（本地生成物，不进 Git）。

配置：

```text
model              deepseek-v4-flash
temperature        0
thinking           disabled
strict_tools       false
selected turns     36
prompt             query_understanding_v1_2
```

协议 identity：

```text
system prompt
  sha256:22c6857d66f77d2b006cb365bad034def41b3a73449416de699a1db6e8831fa4

tool schema
  sha256:ca38347fe0caf78f097e6de7ec0250e6a94ee643dfde7974632eb10651410b29
```

结果：

| 指标 | v1.2 |
| --- | ---: |
| 完整 contract + materialization + Gateway success | 36 / 36 = **100%** |
| 人工自然语言 contract success | 20 / 20 = **100%** |
| 官方 simulator compatibility | 16 / 16 = **100%** |
| 人工自然语言 critical turns | 19 / 20 = **95%** |
| critical assertions | 59 / 60 = **98.33%** |
| 触发一次 repair | 2 / 36 = **5.56%** |
| repair exhausted | **0** |
| token usage | **119,216** |

v1.1 中两个具体的结构失败输入，在 v1.2 都首轮生成了合法 typed frame。最终 36 轮中仍有两轮
使用了一次 repair，但都成功落地：自然语言 `e02_exclude_vs_withdraw` 第 3 轮，以及 simulator
boundary case 第 4 轮。

唯一没有通过的语义断言是 `e05_goal_switch_with_carry` 第 3 轮。用户要求 hiking boots
“must be waterproof”；模型把它保存成 hard semantic `Must be waterproof`，而 oracle 期望
structured `feature=waterproof`。约束语义没有丢失，也不会变成软偏好；差别在于后续 Query
Compiler 采用结构化 filter 还是 semantic query。因此它不是 contract 故障，但仍如实记为
1 个表示层行为偏差。

这不是严格的模型质量 A/B：v1.2 同时改了 wire、prompt 和两条过窄的等义 oracle。它能直接证明
的是新协议消除了已知结构性失败，并在相同的 36 轮 smoke 范围内没有产生新的落地失败。

## 6. v1.2 实际改变了什么

`new_preferences` 不再是允许多种互斥字段组合的万能数组，而是三个封闭数组：

```text
new_preferences.structured  -> facet + eq/neq/in/not_in + values
new_preferences.price       -> lt/le/gt/ge + value_usd
new_preferences.semantic    -> positive/negative + meaning
```

三组都必须存在，没有对应新增条件时使用空数组。模型不再需要根据 relation 猜哪些字段该填
`null`，本地 decoder 也能在更准确的 typed path 上报告错误。

Materializer 仍然允许模型做有价值的语义判断，并增加一条安全落地规则：若多个独立正向需求被
映射到同一个 categorical facet，第二个需求会完整保留为 semantic-only，而不是让 Gateway
因 `multiple_positive_selector` 或空域拒绝整轮。这条 fallback 不修改用户语义，也不额外建立
检索池。

## 7. 当前结论与下一步

typed wire 已经解决这轮讨论中的核心 failure mode；没有必要继续围绕 36 个 smoke 样本堆特殊
规则。保留普通 endpoint + 最多一次带安全本地 reason 的 repair，beta strict 只作为可选能力。

Query Understanding 当前可以交给下一阶段消费。下一步是把
`ResolvedTurnIntent.final_intent + directives` 接给 Query Compiler、fixed Probe 与 $C_t$
可视化，验证故事主线中的完整因果链。
