# Query Understanding Contract v1

- 状态：**P0 核心已实现**
- 日期：**2026-08-28**
- 模型目标：**DeepSeek V4 Flash**
- Tool protocol：**`query_understanding_v1_3`，typed wire，扩大自然语言回归已补强**
- 上游：**Session Context v1 + Catalog Semantic release v0**
- 下游：**Query Compiler → fixed Probe → $C_t$ → Retrieval**

若本文与 [`contract-v0.md`](contract-v0.md) 冲突，以本文和当前代码为准。v0 只保留为设计过程记录。

## 1. 一句话说明

DeepSeek 负责理解“这一轮结束后，用户的完整购物意图应该是什么”；本地代码负责把这个结果
确定性地翻译成 Session Context 操作，并交给 Gateway 原子预演。

模型有充分的语义判断自由，但没有直接修改状态、分配 ID、绕过 Gateway 或伪造商品事实的权力。

```text
自然语言 + 当前 IntentState
    -> DeepSeek reconcile_session_intent
    -> 完整目标意图
    -> 本地 materializer
    -> StateUpdateBatch + Gateway preview
    -> Query Compiler
    -> fixed Probe
    -> C_t
```

QU 阶段不读取 SearchBelief，不计算 $C_t$，也不访问完整商品目录。

## 2. 为什么返回完整状态

模型返回的不是“增加一条”“删除一条”这类底层命令，而是本轮结束后的完整目标状态：

```text
最终 active preferences
    = keep_active_refs 指向的旧 Preference
    + new_preferences.structured 中的新普通 facet 条件
    + new_preferences.price 中的新价格条件
    + new_preferences.semantic 中的新开放语义条件
```

因此：

- 仍然有效的旧条件必须出现在 `keep_active_refs`；
- 漏掉旧 ref 表示撤销该条件；
- 修改旧条件时不保留旧 ref，而是提交一条新条件；
- `dont_care_facets` 是最终完整集合，不是增量；
- model 不会看到或生成 `p_2_1_0` 这样的内部 ID，只看到 `active_0` 等局部 ref。

这能直接区分三个容易混淆的表达：

```text
“黑色不重要了” -> 删除旧 black 条件
“不要黑色”     -> 新建 color NOT_IN [black]
“颜色无所谓”   -> 删除 color 条件，并把 color 放入 dont_care_facets
```

## 3. Structured facet 与证据边界

### 3.1 P0 词表

```text
category
price
brand
material
color
size
style
department
gender
feature
use_case
```

`category` 是模型侧名称，本地映射到保留字段 `system_product_category`。

- `price` 支持 `lt/le/gt/ge`；模型以 USD 字符串提交，例如 `"99.95"`，本地精确转换为
  整数美分 `9995`。
- 其余普通 facet 支持 `eq/neq/in/not_in`。
- 同一 facet 的多个备选值使用一条 `in` 或 `not_in`，不生成互相冲突的多条 selector。

### 3.2 两种 authority

```python
class FacetAuthority(str, Enum):
    CATALOG_VERIFIED = "catalog_verified"
    RETRIEVAL_DERIVED = "retrieval_derived"
```

| Facet | Authority | 本地保证 |
| --- | --- | --- |
| `system_product_category`、`price` | `catalog_verified` | 继续走 release grounder、scope 和 capability 检查 |
| 其余 P0 facet | `retrieval_derived` | 检查 operator、value shape 和 canonical form |

`retrieval_derived` facet 可以结构化保存，也可以在用户明确要求时成为 hard preference；但某件商品
究竟是不是黑色、丝绸或某种风格，必须由后续 Retrieval Evidence Index 判断。Gateway 不会把这些
字段伪装成已经由原始数据集验证过的 catalog truth。

宽 facet 只叠加在 SessionContext/Gateway 应用边界。现有 Gate A/B、Catalog Semantic release 和
原始数据集保持不变。

### 3.3 unknown fallback

未注册的开放维度不会让整轮失败。只要条件本身可表达，materializer 会保留其完整 `meaning`，
并保存成 `facet=None` 的 semantic-only Preference，同时记录 fallback facet 供演示 trace 使用。

未知 facet 的 `dont-care` 无法形成稳定 marker，因此要求模型修复；它不能偷偷进入 structured 状态。

## 4. 模型输入

当前实现的 `ReconcileRequest` 包含：

```python
turn: int
base_intent_version: int
latest_utterance: str
current_goal: str | None
active_preferences: tuple[ActivePreferenceView, ...]
dont_care_facets: tuple[str, ...]
last_assistant_message: str | None
last_question: str | None
category_options: tuple[CategoryOption, ...]
shown_products: tuple[ShownProductView, ...]
allowed_dont_care_facets: tuple[str, ...]
```

发送给模型时会移除私有字段：

- `Preference.id` 变成 `active_N`；
- category scope ID 变成 `category_N`；
- 商品 ID 变成 `product_N`，模型只看到用于理解的短标签；
- 不发送 raw catalog、SearchBelief、Probe candidates 或 $C_t$。

最新用户话语作为 JSON 数据嵌入固定 prompt。用户文本中的“忽略协议”等内容不会改变工具 schema。

## 5. 唯一模型工具

DeepSeek 每次只允许调用一个修改型工具：

```text
reconcile_session_intent
```

当前 wire shape 如下；实际 JSON Schema 中所有字段均为 required，所有 object 均设置
`additionalProperties=false`。

```json
{
  "base_intent_version": 3,
  "disposition": "ready",
  "goal": {"action": "keep", "value": null},
  "keep_active_refs": ["active_0"],
  "new_preferences": {
    "structured": [
      {
        "facet": "color",
        "relation": "not_in",
        "values": ["black"],
        "strength": "hard",
        "basis": "explicit",
        "meaning": "must not be black",
        "evidence": "not black",
        "confidence": 0.98
      }
    ],
    "price": [
      {
        "relation": "le",
        "value_usd": "120",
        "strength": "hard",
        "basis": "explicit",
        "meaning": "budget at most 120 USD",
        "evidence": "under $120",
        "confidence": 1.0
      }
    ],
    "semantic": []
  },
  "dont_care_facets": [],
  "feedback": [],
  "directives": {
    "diversity": "auto",
    "comparison_requested": false,
    "explanation_requested": false
  },
  "clarification": {
    "needed": false,
    "reason": null,
    "alternatives": []
  },
  "summary": "Keep the old condition and exclude black."
}
```

三个数组有各自封闭的字段和枚举：

```text
structured categorical:       eq, neq, in, not_in
structured non-price numeric: lt, le, gt, ge -> local semantic fallback
price:                       lt, le, gt, ge
semantic:                    positive, negative
```

v1.3 允许非价格 named facet 在 `structured` 数组中使用 `lt/le/gt/ge`，但本地不会把未经注册的
数值维度伪装成 catalog truth，而是将完整 `meaning` 确定性保存为 semantic-only preference。
例如 `case_size/le/["40 mm"]` 会保留为“表盘不超过 40 mm”的语义条件。

规则：

- `new_preferences` 三个数组都必须存在，没有新条件时返回空数组；
- structured 只包含 `facet + relation + values`；`eq/neq` 配多个值时，本地会安全归一为
  `in/not_in`；
- price 只包含价格 relation 和 `value_usd`，本地精确转换为整数美分；
- semantic 只包含 `polarity + meaning` 等公共元数据，不允许同时填写 facet、values 或价格；
- 明确表达使用 `basis=explicit`，可以是 hard 或 soft；
- 模型推断使用 `basis=inferred`，必须是 soft；`inferred + hard` 会触发一次修复，而不是静默降级；
- `evidence` 可以是简短原话或忠实释义，不要求做脆弱的逐字 substring 匹配；
- `needs_clarification` 可以与已经确定的状态修改同时出现。

`dont_care_facets` 只能使用 `turn_input.allowed_dont_care_facets`。取消一条子条件时，模型应省略对应
`active_N`，不能临时发明 `heel_height`、`length` 等 marker。若 provider 仍返回未注册 marker，
materializer 会忽略 marker、保留 ref omission 的删除结果，并在 trace 中记录
`ignored_dont_care_facets`，不再让整轮失败。`metal`、`colour`、`budget` 三个常见叫法会分别映射到
`material`、`color`、`price`。

若同一 frame 同时保留某 facet 的具体 `active_N` 又把整个 facet 标成 don't-care，本地以更具体的
保留 ref 为准，忽略冲突 marker 并记录 trace。正确表达“整个 facet 无所谓”仍然必须省略该 facet
的全部旧 ref。

goal action 为：

```text
keep    当前商品任务及 goal 文案都仍准确
revise  仍是同一商品任务，但需清理 goal 中已经取消的约束
switch  真正改找另一种商品
```

goal 必须是去约束化的最短商品任务；颜色、材质、尺寸、功能、用途和价格属于 preferences。

`directives` 与 `feedback` 是当前轮 sidecar，不写入 `IntentState`。

## 6. DeepSeek V4 Flash 调用

当前 adapter 直接使用 DeepSeek 原生 Chat Completions function calling，不依赖额外 SDK：

```python
model = "deepseek-v4-flash"
stream = False
temperature = 0
tool_choice = {
    "type": "function",
    "function": {"name": "reconcile_session_intent"},
}
thinking = {"type": "disabled"}
```

- 默认 endpoint：`https://api.deepseek.com/chat/completions`；
- `strict_tools=True` 时使用 beta endpoint，并在 function 定义中设置 `strict=true`；
- API key 必须由应用显式传入，adapter 不读取仓库中的 `dpskapi` 或其他文件；
- adapter 要求恰好一个、名称正确的 tool call；普通文本、多个调用或错误函数名都会被拒绝；
- JSON 重复键、NaN/Infinity、字段缺失、未知字段和错误类型都会被本地 decoder 拒绝；
- trace 只记录 response ID、model 和 token usage，不保存 Authorization。

### 一次修复

正常路径调用一次。若 wire、局部 ref、materialization 或 Gateway preview 失败，service 使用同一份
request 和 base version 发起一次全新的修复调用，并附上精简的本地错误码、类型化路径与安全原因，
例如 `new_preferences.structured.0` 下出现了 price 专属字段。

修复反馈还会附带合法 don't-care 列表，并明确“省略旧 ref”“整 facet 无所谓”和 `goal.revise` 的区别。
第二次仍失败时返回 `repair_exhausted`。两次尝试期间都不会修改已提交的 SessionContext。
认证、限流、timeout、provider unavailable 和本地 stale version 不进行语义重试。

## 7. Materializer

Materializer 不联网，不访问 Retriever 或 Probe。它按以下步骤工作：

1. 校验 captured intent、request 和 frame 的 base version；
2. 将 `active_N` 映射回可信旧 Preference；
3. 分别解码 structured、price、semantic 三类新条件；
4. 先确定最终 category scope；
5. catalog-verified facet 走 release grounder；
6. retrieval-derived facet 走 combined registry normalizer；
7. unknown facet 转为 semantic-only；
8. 按 category → structured → price → semantic 的固定顺序合并新条件和完整 dont-care 集合；
9. 将与历史 Preference 逻辑完全相同的新 draft 复用为原对象和原 ID；
10. 计算 current 与 target 的差异；
11. 分配新 ID，构造 `StateUpdateBatch`，调用 Gateway preview。

固定 operation 顺序：

1. `SwitchGoal`（若有）；
2. category `ReplaceFacet`（若有）；
3. 其余 structured facet 按 facet ID 排序，以 `ReplaceFacet`、`SetDontCare` 或 `ClearFacet`
   表达最终状态；
4. 一个按 ID 排序的 semantic-only `RemovePreference`（若有）；
5. 新 semantic-only Preference 的 `AddPreference`。

新 ID 使用既有规则：

```text
p_{turn}_{operation_index}_{preference_index}
```

目标状态没有变化时返回：

```python
update = None
final_intent = current_intent
```

不会构造空 batch。

## 8. 输出与事务

成功结果是尚未提交的 `ResolvedTurnIntent`：

```python
update: StateUpdateBatch | None
final_intent: IntentState
feedback: tuple[ProductFeedback, ...]
directives: BehavioralDirectives
clarification: ClarificationNeed
trace: UnderstandingTrace
```

它不能被称为完整 SessionContext，因为此时还没有 Query Compiler、Probe、assistant response 和
TurnRecord。应用继续完成这些步骤后，再用现有 transaction 一次提交完整 next SessionContext。

`CatalogBoundSessionTransaction.preview_update(batch)` 提供同一 release-bound Gateway 下的公开预演
入口；最终 `commit` 仍会重新验证 batch 与 final intent 完全一致。

## 9. 与 Probe 和 $C_t$ 的边界

顺序固定为：

```text
ResolvedTurnIntent.final_intent + directives
    -> Query Compiler
    -> hard eligible mask + soft preferences + semantic query
    -> fixed Probe
    -> C_t + diagnostics
    -> adaptive retrieval / ranking / response
```

这保证演示中的因果关系清楚：先理解用户，再观察该意图映射到商品空间后的分散程度，最后解释
为什么 $C_t$ 和检索策略发生变化。

## 10. 当前实现文件

```text
src/shopping_copilot/session_context/
  registry.py          # FacetAuthority
  wide_facets.py       # 9 个 retrieval-derived facet

src/shopping_copilot/query_understanding/
  models.py            # provider-independent DTO
  views.py             # active_N/category_N/product_N 安全视图
  wire.py              # 原生 tool schema 与严格 decoder
  prompt.py            # 固定 system prompt
  deepseek.py          # Chat Completions HTTP adapter
  planner.py           # final frame -> StateUpdateBatch -> preview
  service.py           # 一次调用 + 一次修复
```

依赖方向保持为：

```text
session_context <- catalog gateway <- query_understanding <- application
                                                   |
                                                   v
                                              retrieval/compiler
```

DeepSeek adapter 不持有 store 或 transaction；Session Context 和 Gateway 不导入 Query Understanding。

## 11. 测试与尚未包含的内容

离线测试已经覆盖：

- authority 分流和宽 facet Gateway 提交；
- tool schema、严格 decoder、forced tool call 和 provider 错误映射；
- `active_N` 保留、撤销、替换、dont-care、unknown fallback；
- “删除 black 条件”与“排除 black”的差异；
- price USD → integer cents；
- inferred-hard 修复；
- feedback 局部 ref；
- 一次修复；
- 真实 Gateway 下 materialize、transaction preview、commit；
- Catalog Semantic runtime artifact 仍只有 price 与 reserved category。
- 40 组 / 72 轮人工自然语言语义回归；
- 32 组 / 128 轮由官方 toy simulator 确定性生成的兼容性回归；
- opt-in DeepSeek V4 Flash live smoke。固定结果和解释见
  [`prompt-evaluation-v0.md`](prompt-evaluation-v0.md)。

普通测试不需要 API key，也不会联网。

下面这些属于下一阶段，不假装已经完成：

- Query Compiler 与 Retrieval Evidence Index；
- fixed Probe、$C_t$ 计算和可视化 trace；
- 将独立 `session_facet_policy_id` 写入持久化 envelope。当前 policy 由代码版本固定；hackathon P0
  暂不扩大 snapshot schema。

## 12. 参考

- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)
- [DeepSeek Context Caching](https://api-docs.deepseek.com/guides/kv_cache/)
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
- [ShopTalk: A System for Conversational Faceted Search](https://research.google/pubs/shoptalk-a-system-for-conversational-faceted-search/)
