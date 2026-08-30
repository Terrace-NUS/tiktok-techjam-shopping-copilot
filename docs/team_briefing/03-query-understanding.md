# Query Understanding：一句话怎样变成可靠的购物意图

这篇文档只解释一件事：

> 用户随口说的一句话，怎样被系统安全地写进“当前购物记忆”。

Query Understanding 简称 QU。第一次阅读时，可以先把它理解成一个**购物记忆编辑器**。
它不搜索商品，也不直接推荐商品。

## 1. 先用一句话理解整个过程

DeepSeek 负责理解用户想怎样修改购物意图，本地程序负责检查并完成修改。

```text
用户自然语言 + 旧 IntentState
    ↓
模型安全视图
    ↓
DeepSeek native function call
    ↓
typed intent frame（仍然不可信）
    ↓
本地 decoder + materializer + Gateway preview
    ↓
ResolvedTurnIntent（已验证、未提交）
```

把上面的专业词翻译成白话：

| 术语 | 实际意思 |
| --- | --- |
| `IntentState` | 系统当前记住的购物目标和偏好 |
| 模型安全视图 | 给 DeepSeek 看的简化版购物记忆 |
| native function call | DeepSeek 按我们规定的表格格式交答案 |
| decoder | 检查这份表格的 JSON 和字段是否合法 |
| materializer | 比较新旧状态，算出具体要增加、删除和修改什么 |
| Gateway preview | 在真正保存前，再用本地规则完整检查一次 |
| `ResolvedTurnIntent` | 已经检查通过，但还没有正式写入 Session Context 的结果 |

最重要的责任划分是：

> DeepSeek 提出“修改后的完整意图”；本地代码负责验证、规范化、分配 ID、生成 operation，
> 并决定这次修改能不能生效。

DeepSeek 不会直接修改数据库或 Session Context。

## 2. 为什么不能只把用户最新一句话交给 DeepSeek

假设用户说：

> 黑色也可以了，预算改成 80 美元，其他要求保留。

这句话依赖前文。只看这一句，系统不知道：

- 原来是不是有“不要黑色”；
- 原预算是多少；
- “其他要求”具体指哪些要求；
- 当前找的是鞋还是包；
- 用户在评价哪一轮展示过的商品。

所以 QU 会把最新一句话和当前购物记忆一起交给 DeepSeek：

```text
turn
base intent version
latest utterance
current goal
active preferences
完整 dont-care 集合
上一条 assistant message / question
此前展示商品的安全标签
当前可用 category options
```

可以简单理解为：

```text
用户刚说的话
    +
系统目前记得的内容
    =
DeepSeek 这次需要理解的完整上下文
```

## 3. DeepSeek 看到的是简化、安全的购物记忆

系统不会把全部内部对象和真实 ID 直接交给模型，而是给本轮内容分配临时编号：

```text
真实 Preference ID      -> active_0、active_1
真实 category scope ID  -> category_0、category_1
真实 product ID         -> product_0、product_1
```

这些编号只在当前这一次 QU 调用中使用。它们类似一张表格里的行号，让 DeepSeek 可以说
“保留 active_2”，但不需要接触内部真实 ID。

例如，DeepSeek 实际收到的输入可以是：

```json
{
  "turn": 2,
  "base_intent_version": 1,
  "latest_utterance": "黑色也可以了，预算改成 80 美元，其他要求保留。",
  "current_intent": {
    "goal": "hiking boots",
    "active_preferences": [
      {
        "ref": "active_0",
        "facet": "color",
        "relation": "not_in",
        "value": ["black"],
        "meaning": "must not be black",
        "strength": "hard",
        "source": "user_explicit"
      },
      {
        "ref": "active_1",
        "facet": "price",
        "relation": "le",
        "value": 10000,
        "meaning": "at most 100 USD",
        "strength": "hard",
        "source": "user_explicit"
      },
      {
        "ref": "active_2",
        "facet": "feature",
        "relation": "eq",
        "value": "waterproof",
        "meaning": "must be waterproof",
        "strength": "hard",
        "source": "user_explicit"
      }
    ],
    "dont_care_facets": []
  },
  "interaction": {
    "last_assistant_message": "……",
    "last_question": null,
    "shown_products": []
  },
  "category_options": [
    {
      "ref": "category_0",
      "label": "All catalog products",
      "is_root": true
    }
  ]
}
```

这个例子为了简短，只展示了 root category。真实 runner 会把当前 release 中全部可用 category
scopes 转换成 `category_N` 列表。

DeepSeek 不会看到：

- session ID 和 profile；
- 完整 TurnRecord 历史；
- SearchBelief 和 $C_t$；
- raw catalog；
- 真实 Preference、category 和 product ID；
- simulator hidden target 或 intent card。

这些内部信息与“理解用户这句话”无关，也不应该由模型修改。

## 4. Native function call：让 DeepSeek 填固定表格

这里使用的不是“请输出一段看起来像 JSON 的普通文本”，而是 DeepSeek 原生 function call。

我们只允许它选择一个工具：

```text
reconcile_session_intent
```

HTTP 请求中的关键设置保持如下：

```json
{
  "model": "deepseek-v4-flash",
  "temperature": 0,
  "stream": false,
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

DeepSeek 返回的外层结构是：

```json
{
  "choices": [
    {
      "message": {
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "reconcile_session_intent",
              "arguments": "{...JSON 字符串...}"
            }
          }
        ]
      }
    }
  ]
}
```

这里很容易产生一个误解：

> DeepSeek 没有远程执行我们的 Python 函数。

它只是返回：“我想用这些 arguments 调用 `reconcile_session_intent`。”随后由本地 decoder
读取 `arguments`，再由本地 materializer 计算真正的状态修改。

所以 function call 在这里是一种**严格的答案格式**，不是把系统控制权交给模型。

## 5. DeepSeek 返回的是修改后的完整意图

DeepSeek 不直接输出：

```text
删除 active_0
把 active_1 改成 80 美元
```

它输出的是“本轮结束以后，哪些偏好应该继续存在”。最终 preference 的计算规则是：

```text
本轮结束后的全部 preference
    = keep_active_refs 指向的旧 preference
    + new_preferences.structured
    + new_preferences.price
    + new_preferences.semantic
```

具体规则：

- 旧条件仍然有效：把它的 `active_N` 放进 `keep_active_refs`；
- 删除旧条件：不要保留它的 ref；
- 修改旧条件：不保留旧 ref，在正确的 `new_preferences` 数组中写入新版本；
- 同一个旧条件不能既 keep，又复制进 new；
- `dont_care_facets` 表示修改后的完整集合，不是“本轮新增了哪些 dont-care”。
- `dont_care_facets` 只能使用输入中的合法列表；取消一个子条件时只省略它的旧 ref，不发明新 facet。
- goal 只写最短商品任务；同一种商品需要清理旧 goal 文字时用 `revise`，真正换商品才用 `switch`。

回到前面的例子，DeepSeek 的核心输出是：

```json
{
  "base_intent_version": 1,
  "disposition": "ready",
  "goal": {
    "action": "keep",
    "value": null
  },
  "keep_active_refs": ["active_2"],
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
  "summary": "Remove the black exclusion, replace the budget with 80 USD, and keep waterproofing."
}
```

这份输出的意思是：

- 没有保留 `active_0`：撤销“不要黑色”；
- 没有保留 `active_1`：撤销旧的 100 美元预算；
- 保留 `active_2`：防水要求继续有效；
- 在 `price` 中新增 80 美元预算。

“黑色也可以”只表示撤销“不要黑色”，并不等于“用户明确表示所有颜色都无所谓”。因此这里不会
自动把 color 加进 `dont_care_facets`。

为什么让模型返回完整状态，而不是直接返回增删操作？

因为模型更擅长回答“用户现在完整想要什么”，本地程序更擅长精确计算：

```text
旧状态 -> 新状态
```

需要哪些删除、替换和新增操作。这样 operation 的正确性不依赖模型是否会使用内部更新协议。

## 6. 为什么 new preferences 要拆成三个数组

我们有三种结构不同的偏好：

| Group | 允许的内容 | 简单例子 |
| --- | --- | --- |
| `structured` | `facet + relation + values` | 颜色是黑色、不要丝绸、表盘不超过 40 mm |
| `price` | `lt/le/gt/ge + value_usd` | 不超过 80 美元 |
| `semantic` | `positive/negative + meaning` | 走一天也不要累、不要显得廉价 |

旧设计使用一个万能对象，同时放：

```text
facet
relation
values
numeric value
semantic polarity
```

这会允许模型拼出“每个字段单独看都合法，组合起来却没有意义”的对象，例如 color 条件同时携带
price value。

现在拆成三个数组后，这类错误从数据结构上就无法表达。它不是限制 DeepSeek 理解用户的自由度，
只是让三种不同答案分别填进正确的表格。

非价格数值范围是一个特例：DeepSeek 可以在 `structured` 中写 `case_size/le/["40 mm"]`，本地会把
完整 meaning 保存成 semantic-only。这样既不会假装 catalog 有可信的表盘尺寸列，也不会丢掉用户要求。

### 6.1 自然描述也能抽成 facet，但要先看句子在说谁

QU 不要求用户必须说成 `material: cotton`。普通描述同样可以一次抽出多个事实：

```text
“100% Cotton cups. Colors: White and Black.”
→ material = 100% Cotton
→ color IN [White, Black]
```

但不能只搜关键词：

```text
“nose won't get red and irritated”
→ 希望产品不引起红肿
→ 不是 color = red
```

系统把两个东西分开保存：

- `meaning`：整理后的完整意思；
- `values/evidence`：尽量保留用户原话，供 catalog evidence 匹配。

因此 `95% gossypium, 5% spandex` 不会被随手改成 `cotton blend`，`Heel measures approximately
1.57 inches` 也不会变成 catalog 中从未出现的 `heel height ~1.57 inches`。gender 是封闭枚举，
所以 men's/women's 会规范成 `men/women`，原文仍保存在 evidence。

另外，“用户亲口说了”不等于“不可妥协”：

- `must`、`key requirement`、明确价格上限、`不要/avoid` 等才是 hard；
- 普通商品描述、候选属性摘录、`what matters is ...` 默认 soft。

## 7. Decoder：先检查 DeepSeek 填的表格

API 返回以后，decoder 首先做格式检查。它会确认：

- 恰好有一个 choice 和一个 tool call；
- function 名确实是 `reconcile_session_intent`；
- `arguments` 是合法 JSON；
- 没有重复 key、额外字段、NaN 或 Infinity；
- 三个 preference group 全部存在；
- 每组字段和 enum 正确；
- goal 的 keep/revise/switch 与 value 组合正确；
- clarification 与 disposition 一致。

只有这些检查全部通过，本地程序才会构造：

```text
StructuredPreferenceFrame
PricePreferenceFrame
SemanticPreferenceFrame
```

这些 frame 可以理解为“已经成功读懂格式的提案”。它们仍然不等于可信状态，因为此时还没有检查
category、price、facet 冲突和 Session Context 更新规则。

## 8. Materializer：把提案变成真正的本地修改

Materializer 是模型输出和 Session Context operation 之间的翻译器。

它掌握这些不应该交给模型处理的本地权力：

1. 校验 current、request 和 frame 使用的是同一个 intent version；
2. 把 `active_N/category_N/product_N` 映射回可信本地对象；
3. 把 category 映射到当前 catalog release 的 scope；
4. 用 Decimal 把 USD 精确转换成整数 cents；
5. 让 catalog-verified facet 经过 release-bound grounder；
6. 用本地 registry 规范化 retrieval-derived facet；
7. 把未知或无法可靠 grounding 的内容保留成 semantic-only；
8. 拒绝 `inferred + hard` 这种不允许的组合；
9. 检查 dont-care 和 active preference 是否冲突；
10. 比较旧状态和完整目标状态，计算 diff；
11. 在本地生成 Preference ID；
12. 生成 typed `StateUpdateBatch`，并调用 Gateway preview。

前面的例子最终会得到类似操作：

```text
ClearFacet(color)
ReplaceFacet(price <= 8000 cents)
keep waterproof preference with its original ID
```

所以模型说的是：

```text
“最终请保留防水，删除黑色限制，把预算改成 80 美元。”
```

Materializer 负责把它变成机器可以可靠执行和回放的 operation。Gateway preview 检查通过后，
新的 `IntentState.version` 才会从 1 变成 2。

这里的 version 可以理解为购物意图的修订号。它可以防止两个基于不同旧状态的修改互相覆盖。

## 9. 出错时为什么只让 DeepSeek 重填一次

可能出现的错误包括：

- 使用了不存在的 `active_99`；
- category ref 不合法；
- 把模型推测的条件标成 hard；
- typed frame 不合法；
- Gateway 拒绝最终状态。

遇到这类可以修复的语义或格式错误时，service 会把以下安全信息交给 DeepSeek：

```text
error code + typed path + safe reason
```

可以把它理解为告诉模型：

```text
“你刚才这张表的哪个字段不合格，请重新填写完整表格。”
```

DeepSeek 只有一次重新生成机会。第二次仍然失败，系统才返回 `repair_exhausted`。

在两次尝试期间，Session Context 都不会被修改。认证失败、限流、timeout、provider unavailable
和 stale version 不是“模型理解错了”，所以不会进行这种语义 repair。

## 10. 为什么成功以后仍然没有立刻保存 Session Context

QU 成功后返回：

```text
ResolvedTurnIntent
├── update: StateUpdateBatch | null
├── final_intent
├── feedback
├── directives
├── clarification
└── safe trace
```

它表示：

> 新购物意图已经通过本地检查，但这一轮对话还没有全部处理完。

`final_intent` 现在可以确定性生成 `CompiledQuery`，然后进入固定 Dense Probe。不过完整一轮仍然
缺少：

- hard constraints 对应的 bound eligible mask；
- 校准后的 SearchBelief / $C_t$；
- 检索结果；
- assistant response；
- shown product IDs；
- 完整 TurnRecord。

应用完成这些工作后，transaction 才会重新 preview `accepted_update`，确认它与下一个
`SessionContext.state.intent` 完全一致，然后一次性 commit。

这样做相当于：

```text
理解用户
搜索商品
生成回答
记录这一轮
```

要么一起成功，要么都不写入。不会出现“偏好已经改了，但检索或回答失败，只保存了半轮状态”的
情况。

## 11. 当前测试结果说明什么

DeepSeek V4 Flash v1.4 结果：

| 指标 | 结果 |
| --- | ---: |
| description 专项 tool call | 11 / 11 |
| description 专项语义断言 | 25 / 25 |
| v1.3 历史失败集 tool call | 24 / 24 |
| v1.3 历史关键语义断言 | 49 / 49 |
| 上述两组 repair | 0 |
| 原 hard-mask 失败任务恢复 | 5 / 7 |

专项语料覆盖 description 多事实、否定范围、词法锚点、类别去重和 hard/soft。另有一次 20 轮自然
语言 smoke：20/20 通过协议，第一次为 58/60 断言；补上 gender 封闭值后，对两个失败对话原样复测为
18/18。完整 artifact 和解释见
[v1.4 fact extraction results](../design/query_understanding/v1-4-fact-extraction-results.md)。

原先 7 个“目标被 hard mask 删除”的任务强制走满 5 轮后，5 个最终状态已经没有错误 hard
constraints；剩余两个都是首轮明确 `key requirement=fabric/cotton`，而对应事实只出现在商品
description。它们不再属于 QU 问题，下一步需要商品侧 description fact sidecar。

## 12. 当前已经完成什么，还缺什么

已经实现：

- 模型安全视图；
- DeepSeek native function call；
- typed decoder；
- materializer；
- Gateway preview；
- 一次 repair；
- Query Compiler；
- Retrieval Evidence Index 与 hard-mask resolver；
- 固定 Dense Probe 入口；
- Intent Volume、正式多路召回和实验 simulator adapter；
- 对应的离线测试和 DeepSeek live 回归。

实验 runner 已经跑通 QU → Session Context → $T_t$ → 多路召回 → 排序 → simulator 日志。接下来
仍缺的是商品 description 的 LLM 结构化 sidecar、最终 production 入口，以及之后再讨论的 runtime
分支选择。

## 延伸阅读

- [逐字段完整例子](../design/query_understanding/session-context-flow-example.md)
- [QU contract](../design/query_understanding/contract-v1.md)
- [Prompt evaluation](../design/query_understanding/prompt-evaluation-v0.md)
- [Query Compiler contract](../design/query_compiler/contract-v0.md)
