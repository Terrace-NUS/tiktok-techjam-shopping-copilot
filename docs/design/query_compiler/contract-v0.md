# Query Compiler Contract v0

- Status: **frozen and implemented**
- Date: **2026-08-28**
- Schema: `shopping-copilot/compiled-query/v0`
- Compiler version: `query_compiler_v0`

## 1. 它解决什么问题

Query Understanding 回答的是“用户现在想要什么”，Retrieval 回答的是“这些信息具体该怎样
参与搜索”。两者中间不能靠 Retrieval 临时重读对话，因此加入一个纯确定性的 Query Compiler：

```text
ResolvedTurnIntent
    -> Query Compiler
       -> q_lex
       -> q_sem
       -> hard_constraints
       -> ranking_preferences
       -> behavioral directives
       -> preference trace
    -> hard-mask resolver（下一阶段）
    -> Fixed Probe
```

编译器不调用 DeepSeek、不扫描商品、不计算 $C_t$，也不写入 Session Context。同一份输入和同一
catalog release 必须产生完全相同的输出。

## 2. 输入

唯一公开输入是已经通过本地 materializer 和 Gateway preview 的：

```python
ResolvedTurnIntent
```

编译器在构造时绑定：

- 当前 `catalog_semantic_release_id`；
- 当前 `CategoryRegistry`，用于把内部 category scope ID 翻译成人能读懂的 label；
- 该 registry 的 `catalog_id` 和 `category_graph_id`。

因此旧 release 的 category preference 不能被悄悄拿到新 catalog 上使用。

## 3. 输出

`CompiledQuery` 的主要字段是：

| 字段 | 用途 |
| --- | --- |
| `q_lex` | 给 BM25 / exact route 的精确商品语言 |
| `q_sem` | 给 Dense Probe、Dense Retrieval 和语义排序的完整自然语言意图 |
| `hard_constraints` | 等待 Evidence Index / verified matcher 生成 eligible mask 的硬条件 |
| `ranking_preferences` | 不可或不应硬筛，但仍需参与打分的偏好 |
| `directives` | 用户明确要求增加/减少多样性、比较或解释 |
| `dont_care_facets` | 明确无所谓的维度，供下游解释和避免反向加权 |
| `trace` | 每条 active preference 实际去了哪些下游通道 |
| `search_ready` | 当前意图是否至少能形成一条非空 `q_sem` |

输出同时固定 `catalog_id`、semantic release、category graph 和 intent version，供后续索引和 mask
做绑定校验。

## 4. 编译规则

### 4.1 `q_lex`

`q_lex` 包含：

- 当前 goal；
- category 的人类可读 label；
- 正向 categorical 条件的规范化值。

它不包含：

- 排除条件，因为把 `not black` 送进 BM25 反而容易召回 `black`；
- price 数字；
- semantic-only 描述。

这些信息不会丢失，全部进入 `q_sem` 或 constraint / ranking view。

### 4.2 `q_sem`

`q_sem` 是从当前完整意图重新生成的稳定自然语言，不直接复用某一轮原始 utterance。这样已经被
用户撤销的旧条件不会因为仍出现在历史文本中而重新进入检索。

例如：

```text
Looking for commuting shoes.
Required category: Footwear.
Required color: red, blue.
Exclude material: plastic.
Required price: at most USD 125.00.
Avoid: does not look cheap.
```

### 4.3 什么能成为 hard constraint

一条 preference 必须同时满足以下条件：

1. 是 structured preference；
2. `commitment == hard`；
3. `source == user_explicit`；
4. facet 属于冻结的 competition hard vocabulary。

编译后保留原 operator 和规范化 value，并显式注明匹配策略：

| Facet | Policy |
| --- | --- |
| product category | `verified_category` |
| price | `conservative_price` |
| 9 个 retrieval-derived facet | `closed_world_retrieval_evidence` |

编译器只描述条件，不在这里生成商品 mask。真正的 mask 必须由相同 release 下的 verified matcher
或 Retrieval Evidence Index 解析，并在 route Top-K 之前应用。

### 4.4 ranking preference 不等于“不重要”

以下条件进入 `ranking_preferences`：

- soft structured preference；
- system inferred / behavioral preference；
- semantic-only preference；
- 不属于当前 hard vocabulary 的 structured 条件。

特别地，用户可能明确说“必须看起来不廉价”。它在 Session Context 中仍是 `hard`，但商品库没有
可证明的布尔字段，所以不能安全地变成 mask。编译器会保留它的 hard commitment，让 ranker 当作
高优先级语义要求，而不是偷偷把它降成普通愿望。

### 4.5 根分类是 no-op

`All products` 根 scope 表示用户取消了原先的分类限制。它不会进入查询文本，也不会生成 hard
constraint；trace 会记录：

```text
root_category_removes_category_restriction -> noop
```

否则用根分类硬筛可能错误排除 category evidence 为 UNKNOWN 的商品。

## 5. 可解释 trace

每一条 active preference 都必须恰好有一条 trace。例如显式的 `color IN [red, blue]` 会记录：

```json
{
  "preference_id": "p_1_1_0",
  "targets": ["q_sem", "q_lex", "hard_constraint"],
  "reason": "explicit_structured_hard_requirement"
}
```

这就是演示时从 Session Context 指向检索行为的因果链，不需要依赖隐藏 prompt 或口头解释。

## 6. Fixed Probe 最小集成

`CompiledDenseProbeRunner` 已经完成以下连接：

1. 拒绝没有 searchable intent 的 query；
2. 校验 compiled query 与 dense index 的 catalog / release 绑定；
3. 接收由上一步解析好的、绑定到 dense index 的 `eligible_mask`；
4. 只编码一次 `q_sem`；
5. 在 mask 之后取固定 `probe_k`；
6. 将同一份 score snapshot 交给现有 `FixedDenseProbe`。

`probe_k` 在 runner 构造时固定，单轮调用不能随 $C_t$ 修改，所以没有
“先认为意图清晰，再用窄 Probe 证明它清晰”的循环。

## 7. 当前明确未做的部分

- Retrieval Evidence Index 到 `eligible_mask` 的 resolver；
- 多个 INCLUDE 导致空集时的确定性 relaxation；
- Probe observation 到 $C_t$ 的校准和 diagnostics $D_t$；
- BM25 / Dense 等 adaptive routes 和最终 ranker；
- `starter.Agent.respond()` 的端到端接线。

因此当前已经跑通的是：

```text
QU accepted intent -> CompiledQuery ->（可选的外部 bound mask）-> Fixed Dense Probe
```

下一步应实现 hard-mask resolver；在它完成前，不应把 `hard_constraints` 假装成已经执行过的筛选。
