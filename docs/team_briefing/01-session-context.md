# Session Context：我们的系统怎样保存“记忆”

## 1. 一句话理解

`SessionContext` 是整本会话账本：

- `IntentState` 是“用户现在到底想买什么”的最新状态；
- `InteractionContext` 是每一轮已经发生了什么的流水记录；
- `SearchBelief` 是 Probe 对当前商品空间的观察。

它不是把聊天记录直接拼进 prompt，也不是一个可以被任意模块随手修改的字典。

## 2. 完整结构

```text
SessionContext
├── session_id
├── profile: ProfilePrior | null
└── state: SessionState
    ├── intent: IntentState
    │   ├── goal
    │   ├── preferences
    │   ├── dont_care_facets
    │   └── version
    ├── interaction: InteractionContext
    │   └── turns: TurnRecord[]
    └── search_belief: SearchBelief | null
        ├── based_on_intent_version
        ├── certainty
        ├── certainty_method
        ├── certainty_evidence
        │   ├── probe_id / probe_size
        │   ├── raw_concentration
        │   └── quality_status / quality_reasons
        ├── candidate_modes
        └── facet_stats
```

### 三块状态各自负责什么

| 部分 | 回答的问题 | 不负责什么 |
| --- | --- | --- |
| `IntentState` | 用户现在要找什么、哪些条件有效 | 检索权重、候选商品分布 |
| `InteractionContext` | 每轮说了什么、改了什么、展示了什么 | 重新解释当前意图 |
| `SearchBelief` | 当前意图映射到 catalog 后有多集中、有哪些模式 | 用户真实偏好 |

### reset 后的初始状态

```json
{
  "session_id": "demo-session",
  "profile": null,
  "state": {
    "intent": {
      "goal": null,
      "preferences": [],
      "dont_care_facets": [],
      "version": 0
    },
    "interaction": {
      "turns": []
    },
    "search_belief": null
  }
}
```

基础 Session Context codec 编码 inner snapshot 时，外面会加上：

```json
{
  "schema": "shopping-copilot/session-context/v1",
  "payload": "上面的 SessionContext"
}
```

Catalog-bound 应用边界还会再包一层 release-pinned envelope：

```json
{
  "schema": "shopping-copilot/catalog-bound-session/v0",
  "session_id": "demo-session",
  "catalog_semantic_release_id": "sha256:...",
  "session_snapshot_sha256": "...",
  "session_snapshot_base64url": "..."
}
```

这里描述的是 canonical 编码格式；当前 store 是 in-memory 实现，并没有额外声称已经完成磁盘持久化。

## 3. IntentState：当前购物意图

```python
IntentState(
    goal: str | None,
    preferences: tuple[Preference, ...],
    dont_care_facets: frozenset[str],
    version: int,
)
```

### goal

`goal` 只描述当前商品任务，例如：

```text
hiking boots
black evening dress
laptop backpack
```

颜色、预算、材质等是 preference，不应该全部塞进 goal。

### Preference

一条已提交 preference 的完整信息包括：

```text
id
facet + operator + value
semantic_text + semantic_polarity
commitment
source
source_turn
evidence_text
interpretation_confidence
```

结构化条件例子：

```json
{
  "id": "p_2_1_0",
  "facet": "color",
  "operator": "not_in",
  "value": ["black"],
  "semantic_text": null,
  "semantic_polarity": null,
  "commitment": "hard",
  "source": "user_explicit",
  "source_turn": 2,
  "evidence_text": "不要黑色",
  "interpretation_confidence": 1.0
}
```

开放语义例子：

```json
{
  "id": "p_2_3_0",
  "facet": null,
  "operator": null,
  "value": null,
  "semantic_text": "comfortable enough to walk in all day",
  "semantic_polarity": "positive",
  "commitment": "soft",
  "source": "user_explicit",
  "source_turn": 2,
  "evidence_text": "最好走一天也不累",
  "interpretation_confidence": 0.94
}
```

底层 contract 允许同一个原子条件同时携带相容的 structured 与 semantic 表示；当前 QU 通常选择其中
一种，避免重复表达。

### hard、soft 和 source

```text
hard  = 用户明确要求必须满足
soft  = 用户偏好，但可以权衡
```

来源包括：

```text
user_explicit        用户明确表达
system_inferred      系统推断
behavioral_feedback  从商品反馈得到
```

`system_inferred` 和 `behavioral_feedback` 不能成为 hard。只有用户明确表达的条件可以进入 hard。

### dont-care 不是“没说”

每个 facet 实际有三种状态：

```text
没有 preference，也没有 marker   = 未知 / 尚未表达
存在 active preference            = 用户有要求
位于 dont_care_facets             = 用户明确说这个维度无所谓
```

例如：

```text
“我没说颜色”       -> color 仍是未知
“不要黑色”         -> color NOT_IN [black]
“颜色无所谓”       -> 删除 color preference，并 SetDontCare(color)
```

一个 facet 不能同时有 active preference 又位于 `dont_care_facets`。

## 4. InteractionContext：可回放的历史

每完成一轮，只允许在历史末尾追加一个 `TurnRecord`：

```text
turn
user_message
intent_version_before
accepted_update
intent_version_after
assistant_message
question / question_key / ask_attribute
shown_product_ids
feedback
search_belief_probe_id
```

`accepted_update` 很重要：它保存这一轮真正接受的状态修改。整个会话可以从空 IntentState 开始，
顺序重放所有 `accepted_update`，重新得到当前 IntentState。

因此 InteractionContext 不是一个可以不断改写的“摘要”，而是一份追加式审计日志。

## 5. SearchBelief：catalog observation，不是用户意图

SearchBelief 绑定某一个 intent version：

```text
based_on_intent_version
certainty / C_t
probe size 与质量
candidate modes
每个 facet 的 entropy、coverage、top values
```

例如：

```text
IntentState：用户想看“适合通勤的包”
SearchBelief：候选集中存在 tote、backpack、crossbody 三种明显模式
```

“存在三种模式”是系统观察到的候选分布，不代表用户偏好这三种模式。意图变化以后，旧 SearchBelief
不能假装仍然适用；新的 belief 必须重新绑定新的 intent version。

## 6. IntentState 只能通过 typed operation 改变

Intent reducer 只接受六种操作。`InteractionContext` 通过组装 next SessionContext 追加，
`SearchBelief` 通过同一 transaction 和 Probe token 校验写入，并不经过这六种 operation。

| Operation | 含义 |
| --- | --- |
| `SwitchGoal` | 切换商品任务，并显式携带仍然有效的旧条件 |
| `ReplaceFacet` | 用一组完整条件替换某个 facet 当前状态 |
| `AddPreference` | 添加一个不冲突的条件 |
| `RemovePreference` | 按稳定 ID 删除条件 |
| `ClearFacet` | 清除条件，回到“未知” |
| `SetDontCare` | 清除条件，并标记为“明确无所谓” |

这些操作装在一个 batch 中：

```python
StateUpdateBatch(
    turn=3,
    base_intent_version=2,
    operations=(...),
)
```

### 为什么要有 base version

如果 batch 基于错误或过期的 intent version 2，却被拿去作用于 version 3，必须失败。当前 store
还会在 turn 开始时持有 per-session 锁，避免同一个 session 的两个 turn 穿插提交；base version 继续
负责防止错误 snapshot 绑定、stale batch 和 replay 不一致。

### 为什么一个 batch 只加一次 version

一个用户 turn 可能同时修改预算、材质和颜色，但它仍然是同一个原子意图变更：

```text
version 2 + 一个有效 batch -> version 3
```

无论 batch 中有多少 operation，都只增加一次。逻辑 no-op 不增加 version。

## 7. 最重要的 invariants

### 不可变快照

任何修改都先构造一个新对象。旧 SessionContext 在 commit 成功前保持原样。

### 原子 batch

operation 按顺序作用在临时状态上；任何一步失败，整个 batch 都回滚，不会留下“预算改了但材质
没改”的半成品。

### 本地 ID authority

新 preference ID 固定为：

```text
p_{turn}_{operation_index}_{preference_index}
```

LLM 不能生成、修改或回收这个 ID。

### Canonical facet state

- categorical facet 使用 `eq/neq/in/not_in`；
- numeric facet 使用 `lt/le/gt/ge`；
- 多个备选值是一条 `in`，不是多条互相冲突的 `eq`；
- numeric 上下界的共同区间必须非空；
- structured value 必须经过 facet registry 规范化。

### History append-only

一次 transaction 必须恰好追加一个连续 TurnRecord。旧 history 不能被改写、删减或重排。

### Probe authority

新 SearchBelief 必须来自同一个 transaction 捕获的旧 snapshot，并绑定同一个预期最终 intent。

## 8. 一次 turn 从读取到提交

```text
1. transaction 获取 session 锁并捕获旧 SessionContext
2. 从旧 intent 构造模型安全视图
3. DeepSeek 提出完整目标意图
4. 本地 materializer 生成 StateUpdateBatch
5. Gateway preview 得到 final_intent，但不写 store
6. Query Compiler / Probe / Retrieval / Response 完成本轮工作
7. 应用追加 TurnRecord，组装完整 next SessionContext
8. transaction 再次 replay、比对、验证
9. 全部成功后原子替换旧快照
```

commit 会重新确认：

- session ID 和 profile 没有改变；
- history 只追加一轮；
- `accepted_update` 重放结果与新 intent 完全一致；
- before/after version 正确；
- category、price 和 release authority 合法；
- SearchBelief 与 Probe token 合法。

## 9. 哪些内容故意不放进 Session Context

- 具体使用哪个 retriever；
- fusion 或 ranking 权重；
- 下轮一定问什么问题；
- prompt 内部 chain-of-thought；
- 由 LLM 猜测的 catalog 商品事实；
- 只为了某个 evaluator 临时存在的控制变量。

Session Context 是跨模块共享的语义事实边界，不是所有运行时变量的垃圾桶。

## 10. 当前实现边界

已实现：model、validation、reducer、canonical snapshot、store transaction、catalog-bound Gateway、
QU 到 `final_intent` preview、Query Compiler、Retrieval Evidence / hard-mask resolver，以及固定
multi-view Probe 入口。

未形成正式 production orchestration：$C_t$ 后的正式多路检索、回答、TurnRecord 组装和官方
`starter.Agent.respond()` 还没有串成完整闭环。

## 权威入口

- [Session Context contract](../design/session_context/contract-v1.md)
- [Session Context module status](../design/session_context/README.md)
- [完整 QU / SessionContext 逐字段例子](../design/query_understanding/session-context-flow-example.md)
