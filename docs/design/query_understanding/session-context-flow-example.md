# 一句话如何变成新的 Session Context

这份文档用一个完整例子解释：

1. 一开始系统的“记忆”是什么；
2. DeepSeek 实际读到了什么；
3. DeepSeek 原生 function call 输出了什么；
4. 本地代码怎样检查和加工这个输出；
5. 最终 Session Context 怎样变化。

示例中的 `category_17`、内部 category scope ID 和商品文案是演示值。字段结构、处理顺序和权限边界
与当前代码一致。

## 0. 先记住整条链

```text
旧 SessionContext
    ↓ 只取允许模型看到的部分
ReconcileRequest 安全视图
    ↓
DeepSeek 原生 function call
    ↓ arguments 是 JSON 字符串
严格 decoder
    ↓
本地 materializer
    ↓
StateUpdateBatch + 新 IntentState 预览
    ↓ Query Compiler / Probe / 回答生成
追加 TurnRecord
    ↓ transaction 再次验证并原子提交
新 SessionContext
```

最重要的权限划分是：

> DeepSeek 只负责提出“这一轮结束后的完整目标意图”。它不能直接读写 Session Context，不能生成
> 内部 Preference ID，不能绕过本地校验，也不能提交记忆。

## 1. 我们所说的“记忆”到底是什么

系统中的完整长期状态叫 `SessionContext`：

```text
SessionContext
├── session_id
├── profile                         用户画像；一次 session 内保持不变
└── state
    ├── intent                      当前用户想买什么、有哪些条件
    ├── interaction                 已处理轮次的追加式历史
    └── search_belief               Probe 对当前商品空间的观察
```

其中，Query Understanding 主要修改的是 `state.intent`。完整类型关系是：

```text
IntentState
├── goal                            当前商品任务
├── preferences                     当前仍然有效的条件
├── dont_care_facets                用户明确表示无所谓的维度
└── version                         意图版本
```

每条 `Preference` 都包含：

```text
id                                  本地生成的稳定 ID
facet                               color、price、material 等；开放语义时为 null
operator                            eq、not_in、le 等；开放语义时为 null
value                               结构化值；开放语义时为 null
semantic_text                       开放语义文本；普通 facet 时为 null
semantic_polarity                   positive/negative；普通 facet 时为 null
commitment                          hard 或 soft
source                              user_explicit、system_inferred 等
source_turn                         来自第几轮
evidence_text                       用户原话或忠实释义
interpretation_confidence           这次解释的置信度
```

### 一个刚刚 reset 的真实空记忆

序列化成 snapshot 后，形状如下：

```json
{
  "schema": "shopping-copilot/session-context/v1",
  "payload": {
    "session_id": "demo-session-001",
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
}
```

这时还没有购物目标、偏好、历史轮次或 Probe 结果。

## 2. 用户说出第一句话

用户输入：

> 我想买一双女士 8 码的徒步靴，100 美元以内，不要真皮，必须防水，最好走一天也不累。

应用不会把上面的完整 Session Context 整包发给 DeepSeek。它先调用
`build_reconcile_request()`，建立一个模型安全视图。

为了这个例子，Catalog Semantic 的 category registry 给出了两个候选：

```text
category_0   All catalog products      根类别
category_17  Hiking Boots              徒步靴类别
```

`category_17` 只是本轮局部代号。DeepSeek 看不到真正的内部 category scope ID。

## 3. DeepSeek 实际读到了什么

DeepSeek 收到的 `turn_input` 是下面这个 JSON：

```json
{
  "turn": 1,
  "base_intent_version": 0,
  "latest_utterance": "我想买一双女士 8 码的徒步靴，100 美元以内，不要真皮，必须防水，最好走一天也不累。",
  "current_intent": {
    "goal": null,
    "active_preferences": [],
    "dont_care_facets": []
  },
  "interaction": {
    "last_assistant_message": null,
    "last_question": null,
    "shown_products": []
  },
  "category_options": [
    {
      "ref": "category_0",
      "label": "All catalog products",
      "is_root": true
    },
    {
      "ref": "category_17",
      "label": "Hiking Boots",
      "is_root": false
    }
  ]
}
```

DeepSeek 同时还会读到：

- 固定的 system prompt；
- `query_understanding_v1_2` 协议版本；
- `reconcile_session_intent` 的 function schema；
- 强制指定这个 function 的 `tool_choice`。

DeepSeek 不会读到：

- `session_id`；
- `ProfilePrior`；
- 完整 `TurnRecord` 历史；
- `SearchBelief` 和 $C_t$；
- 原始商品目录；
- 真实 Preference ID；
- 真实 category scope ID；
- 真实 product ID；
- API key。

如果上一轮展示过商品，模型只能看到类似 `product_0 + 短标签` 的局部引用，真实商品 ID 仍留在
本地映射表中。

## 4. 发给 DeepSeek API 的请求是什么样

下面省略了很长的 system prompt 和完整 JSON Schema，但保留了调用结构：

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {
      "role": "system",
      "content": "你是购物对话中的 Query Understanding 状态编辑器……"
    },
    {
      "role": "user",
      "content": "prompt_version=query_understanding_v1_2\nturn_input={...上面的安全 JSON...}"
    }
  ],
  "stream": false,
  "temperature": 0,
  "max_tokens": 2048,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "reconcile_session_intent",
        "description": "Return the complete intended state after the latest shopping turn.",
        "parameters": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "base_intent_version",
            "disposition",
            "goal",
            "keep_active_refs",
            "new_preferences",
            "dont_care_facets",
            "feedback",
            "directives",
            "clarification",
            "summary"
          ]
        }
      }
    }
  ],
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "reconcile_session_intent"
    }
  },
  "thinking": {
    "type": "disabled"
  }
}
```

所以这里不是让 DeepSeek 在普通聊天文本里随便写 JSON。API 明确要求它调用
`reconcile_session_intent`，而这个 function 的参数恰好使用 JSON Schema 描述。

## 5. DeepSeek 原生 function call 输出了什么

DeepSeek API 返回的外层形状大致如下：

```json
{
  "id": "response-demo-001",
  "model": "deepseek-v4-flash",
  "choices": [
    {
      "message": {
        "content": null,
        "tool_calls": [
          {
            "id": "call-demo-001",
            "type": "function",
            "function": {
              "name": "reconcile_session_intent",
              "arguments": "{...JSON 字符串...}"
            }
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 3200,
    "completion_tokens": 700,
    "total_tokens": 3900
  }
}
```

`arguments` 在 HTTP 响应中是一个 JSON 字符串。把它解析并排版后，本例应该类似：

```json
{
  "base_intent_version": 0,
  "disposition": "ready",
  "goal": {
    "action": "switch",
    "value": "hiking boots"
  },
  "keep_active_refs": [],
  "new_preferences": {
    "structured": [
      {
        "facet": "category",
        "relation": "eq",
        "values": ["category_17"],
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The product category is hiking boots.",
        "evidence": "徒步靴",
        "confidence": 0.99
      },
      {
        "facet": "gender",
        "relation": "eq",
        "values": ["women"],
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The boots are for women.",
        "evidence": "女士",
        "confidence": 1.0
      },
      {
        "facet": "size",
        "relation": "eq",
        "values": ["8"],
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The required size is women's 8.",
        "evidence": "女士 8 码",
        "confidence": 0.98
      },
      {
        "facet": "material",
        "relation": "not_in",
        "values": ["leather"],
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The boots must not be made of leather.",
        "evidence": "不要真皮",
        "confidence": 1.0
      },
      {
        "facet": "feature",
        "relation": "eq",
        "values": ["waterproof"],
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The boots must be waterproof.",
        "evidence": "必须防水",
        "confidence": 1.0
      }
    ],
    "price": [
      {
        "relation": "le",
        "value_usd": "100",
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The price must be at most 100 USD.",
        "evidence": "100 美元以内",
        "confidence": 1.0
      }
    ],
    "semantic": [
      {
        "polarity": "positive",
        "strength": "soft",
        "basis": "explicit",
        "meaning": "Comfortable enough to walk in all day without getting tired.",
        "evidence": "最好走一天也不累",
        "confidence": 0.94
      }
    ]
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
  "summary": "Find women's size 8 hiking boots under 100 USD, excluding leather, requiring waterproofing, with a soft all-day comfort preference."
}
```

这里仍然只是 DeepSeek 提出的“不可信 frame”，还不是 Session Context。

## 6. 本地 decoder 先检查什么

系统先检查 API 外层：

1. `choices` 必须恰好有一个；
2. `tool_calls` 必须恰好有一个；
3. function 名必须是 `reconcile_session_intent`；
4. `arguments` 必须是字符串。

然后检查 arguments：

1. 必须是合法 JSON，不能有重复 key、`NaN` 或 `Infinity`；
2. 根对象不能缺字段，也不能多字段；
3. `new_preferences` 下必须同时有 `structured`、`price`、`semantic`；
4. structured 不能混进价格字段；
5. price 不能混进 facet 或 values；
6. semantic 不能混进 facet、values 或价格；
7. relation、polarity、strength、basis 必须属于各自枚举；
8. `base_intent_version` 必须仍然是 0；
9. `clarification.needed` 必须和 disposition 一致。

校验成功后，JSON 才会变成三个明确的 Python 类型：

```text
StructuredPreferenceFrame
PricePreferenceFrame
SemanticPreferenceFrame
```

## 7. Materializer 怎样把模型提议变成本地状态

Materializer 不联网，也不做检索。它按确定规则处理：

| DeepSeek 提议 | 本地处理 |
| --- | --- |
| `category_17` | 映射回 registry 中可信的内部 category scope ID |
| `price <= "100"` | 使用 Decimal 精确转换成 `10000` 美分 |
| `gender=women` | 使用本地 facet registry 归一化 |
| `size=8` | 使用本地 facet registry 归一化 |
| `material NOT_IN leather` | 保存为 retrieval-derived structured 条件 |
| `feature=waterproof` | 保存为 retrieval-derived structured 条件 |
| all-day comfort | 保存为 `facet=null` 的 soft semantic 条件 |
| `explicit + hard/soft` | 映射成 `user_explicit + Commitment` |

DeepSeek 没有提供任何 Preference ID。本地根据固定规则分配 ID：

```text
p_{turn}_{operation_index}_{preference_index}
```

在本例中，本地得到的操作序列大致是：

```text
operation 0  SwitchGoal("hiking boots")
operation 1  ReplaceFacet(system_product_category)  -> p_1_1_0
operation 2  ReplaceFacet(feature)                  -> p_1_2_0
operation 3  ReplaceFacet(gender)                   -> p_1_3_0
operation 4  ReplaceFacet(material)                 -> p_1_4_0
operation 5  ReplaceFacet(price)                    -> p_1_5_0
operation 6  ReplaceFacet(size)                     -> p_1_6_0
operation 7  AddPreference(all-day comfort)         -> p_1_7_0
```

这些操作一起组成：

```text
StateUpdateBatch
├── turn = 1
├── base_intent_version = 0
└── operations = 上面的 8 个 operation
```

随后 Gateway 会在内存中 preview 整个 batch。任何一个操作不合法，整个 batch 都不会部分生效。

## 8. Preview 后的新 IntentState 是什么样

下面是阅读友好的完整结果。`category-scope:hiking-boots` 代表本地 registry 中的真实内部 scope ID，
实际 ID 由 release 决定，不由 DeepSeek 生成。

```json
{
  "goal": "hiking boots",
  "version": 1,
  "dont_care_facets": [],
  "preferences": [
    {
      "id": "p_1_1_0",
      "facet": "system_product_category",
      "operator": "eq",
      "value": "category-scope:hiking-boots",
      "semantic_text": null,
      "semantic_polarity": null,
      "commitment": "hard",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "徒步靴",
      "interpretation_confidence": 0.99
    },
    {
      "id": "p_1_2_0",
      "facet": "feature",
      "operator": "eq",
      "value": "waterproof",
      "semantic_text": null,
      "semantic_polarity": null,
      "commitment": "hard",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "必须防水",
      "interpretation_confidence": 1.0
    },
    {
      "id": "p_1_3_0",
      "facet": "gender",
      "operator": "eq",
      "value": "women",
      "semantic_text": null,
      "semantic_polarity": null,
      "commitment": "hard",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "女士",
      "interpretation_confidence": 1.0
    },
    {
      "id": "p_1_4_0",
      "facet": "material",
      "operator": "not_in",
      "value": ["leather"],
      "semantic_text": null,
      "semantic_polarity": null,
      "commitment": "hard",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "不要真皮",
      "interpretation_confidence": 1.0
    },
    {
      "id": "p_1_5_0",
      "facet": "price",
      "operator": "le",
      "value": 10000,
      "semantic_text": null,
      "semantic_polarity": null,
      "commitment": "hard",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "100 美元以内",
      "interpretation_confidence": 1.0
    },
    {
      "id": "p_1_6_0",
      "facet": "size",
      "operator": "eq",
      "value": "8",
      "semantic_text": null,
      "semantic_polarity": null,
      "commitment": "hard",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "女士 8 码",
      "interpretation_confidence": 0.98
    },
    {
      "id": "p_1_7_0",
      "facet": null,
      "operator": null,
      "value": null,
      "semantic_text": "Comfortable enough to walk in all day without getting tired.",
      "semantic_polarity": "positive",
      "commitment": "soft",
      "source": "user_explicit",
      "source_turn": 1,
      "evidence_text": "最好走一天也不累",
      "interpretation_confidence": 0.94
    }
  ]
}
```

注意 `version` 从 0 变成 1。一个 batch 不管包含多少 operation，只让 intent version 增加一次。

## 9. 此时记忆已经被修改了吗

还没有。

Query Understanding 当前返回的是 `ResolvedTurnIntent`：

```text
ResolvedTurnIntent
├── update              上面的 StateUpdateBatch；如果没有变化则为 null
├── final_intent        Gateway preview 后的新 IntentState
├── feedback            本轮商品反馈
├── directives          多样性、比较、解释等本轮指令
├── clarification       是否需要澄清
└── trace               调用次数、摘要、semantic fallback 等安全诊断
```

这是一个“已经验证、但尚未提交”的结果。这样设计是因为同一轮还需要完成：

1. Query Compiler；
2. fixed Probe 和 $C_t$；
3. 检索与排序；
4. 生成 assistant response；
5. 记录展示了哪些商品；
6. 组装完整 `TurnRecord`。

只有这些信息都准备好后，应用才应该一次性提交完整的新 Session Context。

## 10. 最终提交的新 Session Context

应用会在旧 history 后追加一个 `TurnRecord`。第一轮的记录形状是：

```text
TurnRecord
├── turn = 1
├── user_message = 用户原话
├── intent_version_before = 0
├── accepted_update = 第 7 节的 StateUpdateBatch
├── intent_version_after = 1
├── assistant_message = 本轮最终回答
├── question = null 或本轮追问
├── question_key = null 或问题的稳定 key
├── ask_attribute = null 或官方协议要求的属性
├── shown_product_ids = 本轮真正展示的内部商品 ID
├── feedback = 本轮解析出的反馈
└── search_belief_probe_id = 本轮 Probe ID；没有则为 null
```

如果暂时不产生 SearchBelief，最终完整状态可理解为：

```text
SessionContext
├── session_id = "demo-session-001"                 不变
├── profile = null                                   不变
└── state
    ├── intent = 第 8 节的新 IntentState             version = 1
    ├── interaction
    │   └── turns = [上面的 TurnRecord]
    └── search_belief = null
```

transaction 在真正写入前还会再次检查：

1. session 和 profile 没有被偷偷替换；
2. history 只追加了恰好一轮；
3. turn 连续；
4. `accepted_update` 从锁内旧状态重放后，结果和 `state.intent` 完全一致；
5. intent version before/after 正确；
6. 如果带有新 SearchBelief，它来自同一个 transaction、同一个旧 snapshot 和同一个最终 intent；
7. catalog release、category 和 price 等 authority 仍然合法。

全部通过后才原子替换旧 SessionContext。任何一步失败，旧记忆保持原样。

## 11. 下一轮怎样读取刚才的记忆

假设用户紧接着说：

> 预算改成 80 美元，真皮现在也可以，其他要求保留。

本地不会把 `p_1_1_0` 等真实 ID 发给 DeepSeek，而会重新映射为：

```text
active_0  category = hiking boots
active_1  feature = waterproof
active_2  gender = women
active_3  material NOT_IN leather
active_4  price <= 10000 cents
active_5  size = 8
active_6  semantic soft: all-day comfort
```

DeepSeek 应该返回的核心部分是：

```json
{
  "base_intent_version": 1,
  "goal": {
    "action": "keep",
    "value": null
  },
  "keep_active_refs": [
    "active_0",
    "active_1",
    "active_2",
    "active_5",
    "active_6"
  ],
  "new_preferences": {
    "structured": [],
    "price": [
      {
        "relation": "le",
        "value_usd": "80",
        "strength": "hard",
        "basis": "explicit",
        "meaning": "The price must be at most 80 USD.",
        "evidence": "预算改成 80 美元",
        "confidence": 1.0
      }
    ],
    "semantic": []
  }
}
```

这里省略了根对象中没有变化的 required sidecar 字段，只突出状态协议：

- `active_3` 被遗漏，所以“不要真皮”被撤销；
- `active_4` 被遗漏，所以旧的 100 美元预算被撤销；
- 新的 80 美元条件放入 `new_preferences.price`；
- 其他 `active_N` 被保留；
- 最终 intent version 从 1 变成 2。

这就是“返回完整目标状态”的含义：

```text
下一轮全部偏好
    = keep_active_refs 中保留的旧偏好
    + 三个 new_preferences 数组中的新增或改写偏好
```

## 12. 如果 DeepSeek 输出错了会怎样

例如 DeepSeek：

- 引用了不存在的 `active_99`；
- 把 `price` 写进 structured；
- 返回 inferred + hard；
- 写了非法 category ref；
- 生成了 Gateway 不接受的最终状态。

系统会：

1. 不修改旧 Session Context；
2. 生成不含敏感数据的错误码、字段路径和原因；
3. 使用完全相同的 `turn_input` 请求 DeepSeek 重新调用一次 function；
4. 第二次仍失败时返回 `repair_exhausted`；
5. 整个过程中都不会留下“改了一半”的记忆。

## 13. 当前仓库真实完成到哪里

已经完成并有测试覆盖：

- Session Context 数据模型、reducer、snapshot 和 transaction；
- Catalog Semantic Gateway preview/commit 边界；
- 模型安全视图；
- DeepSeek V4 Flash 原生 function call；
- typed tool schema 和严格 decoder；
- materializer、ID 分配、grounding、diff 与 Gateway preview；
- 确定性 Query Compiler、逐 preference trace 和固定 Dense Probe 最小入口；
- 一次 repair；
- 自然语言和官方 simulator prompt suite；
- 真实 DeepSeek smoke。

尚未完成：

- 把 QU 接入官方 `starter.Agent.respond()`；
- hard-mask resolver；
- production $C_t$ 和 SearchBelief；
- 检索、回答、TurnRecord 与 QU 的完整应用编排。

因此，当前代码已经真实跑通到“新 IntentState 预览 -> CompiledQuery -> 可选 bound mask -> fixed
Dense Probe”。第 10 节描述的完整 Session Context 原子提交契约已经存在并有集成测试，但还没有被
官方 Agent 的生产流程调用。

## 14. 对应代码位置

```text
src/shopping_copilot/session_context/models.py
    Preference、IntentState、SearchBelief

src/shopping_copilot/session_context/aggregates.py
    TurnRecord、SessionState、SessionContext

src/shopping_copilot/query_understanding/views.py
    Session 状态到模型安全视图

src/shopping_copilot/query_understanding/prompt.py
    DeepSeek system prompt 和 turn_input

src/shopping_copilot/query_understanding/deepseek.py
    原生 function-call HTTP adapter

src/shopping_copilot/query_understanding/wire.py
    function schema 和严格 decoder

src/shopping_copilot/query_understanding/planner.py
    frame 到 StateUpdateBatch 和 IntentState preview

src/shopping_copilot/query_understanding/service.py
    一次正常调用加最多一次 repair

src/shopping_copilot/catalog/semantic/gateway/store.py
    锁内 preview、验证与原子 commit
```
