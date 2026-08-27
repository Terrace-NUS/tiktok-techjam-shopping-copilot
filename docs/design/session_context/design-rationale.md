# Session Context Design Rationale

> **Status: research draft (superseded as an implementation contract).**
> This document preserves the reasoning that led to the session-context
> architecture. Normative schemas, invariants, and transaction semantics live
> in [`contract-v1.md`](contract-v1.md); implementation order and repository
> layout live in [`implementation-plan.md`](implementation-plan.md).

我现在认为应该冻结的不是一个传统 `DialogState`，而是一个很轻的 `SessionState`，里面明确区分三种东西：

```text
SessionState
├── IntentState          # 用户目前要什么：事实源
├── InteractionContext   # 当前对话里发生过什么
└── SearchBelief         # 当前需求投影到商品库后的系统观测
```

概念上三层，工程上完全可以就是一个 Python dataclass，不代表三个复杂服务。

这比简单 slot table 多一点，但每一个字段都能明确对应 Query Understanding、Retrieval、Ranking 或 Asking。

---

# 一、先反推：下游到底需要 State 提供什么？

官方题目要求的不只是 slot accumulation，而是：

- Buying / Browsing 间动态变化；

- incremental information；

- Intent Override；

- Over-Generality 时主动澄清；

- runtime adaptive orchestration；

- search + ranking 联动。


所以至少有四个问题需要回答。

### Query Understanding 需要

看到：

> “Actually brown is fine too.”

它得知道：

- 当前是不是已经有 `color=black`；

- “too” 是 ADD 还是 REPLACE；

- brown 是哪个 facet；

- 这是用户明确表达还是系统推测。


看到：

> “something cheaper”

它必须知道当前 price preference 是什么。

所以 State 不能只有一个 `need: str`。

---

### Retrieval 需要

它至少要区分：

```text
必须满足
最好满足
明确不要
自由语义需求
```

否则：

> “必须低于 $100”

和：

> “最好 $100 左右”

最终都会变成同一个 query string。

---

### Ranking 需要

除了当前 constraint，还要知道：

> 哪些是 hard、哪些只是 preference？

否则无法确定：

```text
filter
vs
boost/penalty
```

还需要知道 Probe 算出的 certainty，才能控制 diversity。

---

### Asking 需要

它要知道：

- 哪些属性用户已经说过；

- 哪些用户明确表示 don't care；

- 什么已经问过；

- 当前候选在哪些维度存在分歧。


因此简单：

```python
need
constraints
semantic_preferences
```

又不够。

---

# 二、研究上最值得借的不是 ShopTalk 的 schema，而是它的三个设计原则

Google ShopTalk 是这里最强的 production precedent：它 2020 年部署到了 Google Assistant Shopping。它不是维护一个自然语言 summary，而是：

```text
latest utterance
→ finite intent operators
→ cumulative structured dialog state
→ fulfillment adapter
→ search query + facet restrictions
```

([Google Research](https://research.google/pubs/shoptalk-a-system-for-conversational-faceted-search/?utm_source=chatgpt.com "ShopTalk: A System for Conversational Faceted Search"))

三个原则特别关键。

## 1. Understanding 输出 **state operation**，不是新的完整 state

ShopTalk 把几十万乃至百万商品属性背后的状态变化压缩成有限操作：

```text
SET VALUE
CLEAR VALUE
CLEAR FACET
NUDGE FACET
ORDER BY
...
```

例如它明确区分：

```text
"I don't care if it's red"
→ clear red

"I don't want red"
→ red != true
```

还支持 `< / ≤ / > / ≥` 等范围关系。([SIGIR eCom](https://sigir-ecom.github.io/ecom22Papers/paper_3793.pdf "ShopTalk: A System for Conversational Faceted Search"))

这非常适合我们。

---

## 2. Dialog State 应该和 Search Backend 解耦

ShopTalk 特别强调 structured state 的价值：它既能展示给用户确认，也能由 adapter 转成不同 fulfillment backend 所需的 query + facets。([SIGIR eCom](https://sigir-ecom.github.io/ecom22Papers/paper_3793.pdf "ShopTalk: A System for Conversational Faceted Search"))

因此我会修改上一版的设计：

**Dialog State 里不要保存 `effect=FILTER/BOOST`。**

因为这是搜索策略，不是用户意图。

应该保存：

```text
用户说的是 hard 还是 soft
positive 还是 negative
= / != / < / >
```

然后 Retrieval Compiler 自己决定：

```text
hard positive    → filter
soft positive    → retrieval/ranking boost
hard negative    → exclusion
soft negative    → ranking penalty
```

这样未来你改 BM25 / Dense / reranker，state 不变。

---

## 3. 结构化不了的需求不能丢

ShopTalk 明确指出真实购物 schema 是 incomplete / dynamic，因此专门维护 **ungrounded spans**，并让文本搜索 backend 继续利用它们。([SIGIR eCom](https://sigir-ecom.github.io/ecom22Papers/paper_3793.pdf "ShopTalk: A System for Conversational Faceted Search"))

Alibaba 2026 conversational recommendation 也把 natural-language intent extraction 和 keyword/exclusion/price constrained retrieval 结合，而不是假设所有用户需求都能变成固定 slot。([阿里云](https://www.alibabacloud.com/help/en/pai/best-practices-of-ai-conversational-recommendation-ai-shopping-guide?utm_source=chatgpt.com "Best practices for AI conversational recommendations - Platform For AI - Alibaba Cloud Documentation Center"))

所以：

> `structured constraint + semantic preference`

这个双表示必须保留。

---

# 三、我现在建议的最终 `IntentState`

这是我认为“够用但不过度”的版本。

```python
class IntentState:
    # 当前购物任务的自由语义表示
    goal: str

    # 当前有效偏好
    preferences: list[Preference]

    # 状态版本
    version: int
```

注意，就三个顶层字段。

复杂度放在一个设计得好的 `Preference` 里面。

---

# 四、核心对象应该叫 `Preference`，而不是 Slot

```python
class Preference:
    id: str

    # 能可靠对应 catalog schema 时填写
    facet: str | None

    # structured 时使用
    operator: Literal[
        "eq", "neq",
        "lt", "le", "gt", "ge"
    ] | None

    value: Any | None

    # 无法可靠结构化时保存原始语义
    semantic_text: str | None

    # 用户对此偏好的约束强度
    commitment: Literal["hard", "soft"]

    # 来源及解析可信度
    source: Literal[
        "user_explicit",
        "user_feedback",
        "system_inferred",
    ]
    confidence: float
```

我认为这些字段一个都不是装饰。

---

# 五、为什么必须有 `facet/operator/value`

因为 Query Understanding 最终必须能稳定处理这种输入：

> black

```text
facet=color
operator=eq
value=black
```

> not red

```text
facet=color
operator=neq
value=red
```

> under $100

```text
facet=price
operator=le
value=100
```

> size 8 or larger

```text
facet=size
operator=ge
value=8
```

这基本继承 ShopTalk 已经生产验证过的 predicate abstraction。([SIGIR eCom](https://sigir-ecom.github.io/ecom22Papers/paper_3793.pdf "ShopTalk: A System for Conversational Faceted Search"))

PEARL 也直接把多轮购物 preference extraction 定义成电商可消费的 key-value filters，而且特别指出实际用户会有非标准表达、隐式属性、否定和范围，以及跨轮修改偏好。([ACL Anthology](https://aclanthology.org/2024.emnlp-industry.112/?utm_source=chatgpt.com "PEARL: Preference Extraction with Exemplar Augmentation and Retrieval with LLM Agents - ACL Anthology"))

所以这部分不是拍脑袋。

---

# 六、为什么还必须有 `semantic_text`

因为用户会说：

> comfortable for walking all day

> not too sporty

> something suitable for a first date

这些可能没有对应的 Amazon facet。

因此：

```python
Preference(
    facet=None,
    semantic_text="not too sporty",
    commitment="soft",
    source="user_explicit",
    confidence=0.98,
)
```

它继续进入：

```text
Dense retrieval
Semantic reranker
Probe representation
```

而不是被 Query Understanding 强行 hallucinate 成：

```text
style = casual
```

这直接继承 ShopTalk 的 ungrounded-span 哲学。([SIGIR eCom](https://sigir-ecom.github.io/ecom22Papers/paper_3793.pdf "ShopTalk: A System for Conversational Faceted Search"))

---

# 七、为什么 `commitment` 还是要保留

这部分不是 ShopTalk 直接给出的标准字段，但对我们的题是必要扩展。

考虑：

### A

> “必须 waterproof。”

### B

> “最好 waterproof。”

如果都存成：

```text
waterproof=true
```

Retrieval/Ranking 无法正确处理。

所以：

```text
A:
commitment = hard

B:
commitment = soft
```

然后后端解释：

|State semantic|Retrieval|Ranking|
|---|---|---|
|hard + eq|必须过滤|不需要额外学习|
|hard + neq|必须排除|—|
|soft + eq|不强制过滤|boost|
|soft + neq|不强制排除|penalty|
|semantic soft|dense query|reranker boost/penalty|

这实际上给 Query Understanding 和 Retrieval 之间建立了一个非常稳定的 contract。

---

# 八、为什么必须有 `source + confidence`

这里不是为了“高级感”，而是为了防止系统自己的推断污染 hard constraints。

例如用户说：

> “something comfortable for work.”

LLM 猜：

```text
style=business_casual
```

如果状态不知道这是 inferred，后面 Structured Retrieval 可能直接把 sneaker 全过滤掉。

正确应该是：

```python
Preference(
    facet="style",
    value="business_casual",
    commitment="soft",
    source="system_inferred",
    confidence=0.58,
)
```

然后下游明确规定：

```text
system_inferred
永远不能直接产生 hard filter
```

这就是 intent transparency 真正在工程上的意义：

> 系统知道哪些是用户说的，哪些是自己猜的。

不是 UI 上显示一个 82% confidence。

---

# 九、Override 怎么做？不需要在 State 里永久保存 superseded graph

这里我仍然建议保持轻量。

Query Understanding 不直接改 `IntentState`，而是输出：

```python
StateDelta
```

P0 只要这些操作：

```python
SET
ADD
REMOVE
CLEAR_FACET
NUDGE
SWITCH_GOAL
```

这几乎就是 ShopTalk operators 的精简现代版。ShopTalk 本身也明确用有限的 operator 作为 Parser 和 DST 的接口，并按照 clear/conflict 等规则合并状态。([SIGIR eCom](https://sigir-ecom.github.io/ecom22Papers/paper_3793.pdf "ShopTalk: A System for Conversational Faceted Search"))

例如：

> “Black, preferably.”

```text
SET color=black soft
```

> “Brown is fine too.”

```text
ADD color=brown soft
```

> “Actually only brown.”

```text
SET color=brown hard
```

更新器移除冲突的 black。

> “Color doesn't matter.”

```text
CLEAR_FACET color
```

> “Something cheaper.”

```text
NUDGE price down
```

> “Actually forget the shoes. Show me jackets.”

```text
SWITCH_GOAL
```

历史变化写进一个很小的 event log：

```python
StateChange(
    turn=4,
    op="CLEAR_FACET",
    facet="color",
)
```

**active IntentState 永远只保存当前有效状态。**

这样 retrieval 不需要处理一堆 `SUPERSEDED`。

---

# 十、但只有 IntentState 仍然不够，因为我们的系统还有 Probe

这就是我们和 ShopTalk 最大的创新差异。

ShopTalk：

```text
Dialog State
→ Search
```

我们：

```text
Intent State
→ Probe
→ Search Belief
→ Adaptive Search
```

所以我认为需要一个非常轻的第二对象：

```python
class SearchBelief:
    based_on_version: int

    certainty: float

    candidate_modes: list[CandidateMode]

    facet_entropy: dict[str, float]
```

只有三个真正的信息。

---

# 十一、为什么 `SearchBelief` 不属于 Intent State？

这是概念上很重要的一刀。

例如：

```text
用户：
"comfortable shoes for work"
```

IntentState 只是：

```yaml
goal: comfortable shoes for work

preferences:
  - semantic: comfortable
```

这是**用户状态**。

Probe 后发现：

```text
loafers             34%
minimal sneakers    32%
walking shoes       28%
```

然后：

```text
certainty = 0.29
```

这是：

> **系统观察到当前需求在当前 catalog 中很分散。**

不能把它写成：

```text
user.certainty = 0.29
```

因为我们没有证明用户心理上只有 29% 确定。

这应该叫：

```text
SearchBelief.certainty
```

更准确。

---

# 十二、`candidate_modes` 也很重要，但不需要复杂 clustering state

例如：

```python
CandidateMode(
    label="loafers",
    mass=0.34,
    representative_ids=[...],
)
```

它解决三个下游问题：

### Retrieval

低 certainty：

```text
每个 mode 分配一定 recall budget
```

避免最大 cluster 吞掉全部结果。

### Ranking

低 certainty：

```text
Top-K 保留 mode diversity
```

### Asking

直接可以：

> “你更偏 loafer、简洁 sneaker，还是 walking shoe？”

所以同一个 Probe 结果支持三个策略。

这正是我们想要的 architecture coherence。

---

# 十三、`facet_entropy` 只负责 Asking

例如：

```text
style     0.90
color     0.31
material  0.45
```

结合：

```text
certainty low
```

系统判断：

> 有必要继续 elicitation。

再用：

```text
argmax entropy = style
```

决定：

> 问 style，而不是 color。

它不需要成为 Dialog State 的一部分。

---

# 十四、还缺最后一个非常小的东西：`InteractionContext`

它只解决对话引用和重复询问：

```python
class InteractionContext:
    last_shown_products: list[str]
    asked_facets: set[str]
    user_feedback: list[ProductFeedback]
```

为什么必须有？

用户会说：

> “第二双挺好。”

> “别给我这种了。”

> “比刚才第一双正式一点。”

Query Understanding 如果不知道刚才展示过什么，无法解析。

而 Asking 如果不知道问过 `color`，会重复提问。

所以这个不是生产系统过度设计，而是最低的 multi-turn correctness。

---

# 十五、最终状态其实就这么大

```python
class SessionState:
    intent: IntentState
    search_belief: SearchBelief | None
    interaction: InteractionContext
```

内部：

```text
IntentState
├── goal
├── preferences[]
└── version

Preference
├── facet/operator/value          # grounded
├── semantic_text                 # ungrounded
├── commitment                    # hard / soft
├── source                        # explicit / feedback / inferred
└── confidence                    # interpretation confidence

SearchBelief
├── certainty
├── candidate_modes
├── facet_entropy
└── based_on_version

InteractionContext
├── last_shown_products
├── asked_facets
└── feedback
```

**这就是我认为当前最合适的复杂度。**

---

# 十六、它和整个系统的接口会非常干净

## Query Understanding

输入：

```text
latest utterance
+ IntentState
+ relevant InteractionContext
```

输出：

```text
StateDelta[]
```

绝不输出完整 query。

---

## Query Compiler

输入：

```text
IntentState
```

输出：

```python
CompiledQuery(
    hard_filters,
    soft_structured_preferences,
    positive_semantic_query,
    negative_semantic_query,
)
```

例如：

```text
IntentState

goal:
  office walking shoes

preferences:
  price <= 120 hard
  sneakers != true hard
  black eq soft
  "comfortable walking all day" soft
```

编译为：

```text
hard_filters:
    price <= 120
    category != sneakers

structured boosts:
    color=black

dense positive:
    comfortable walking all day
    office shoes
```

非常自然。

---

# 十七、Retrieval 直接使用这个 contract

```text
hard filters
      ↓
Structured

goal + grounded text
      ↓
BM25

goal + semantic preferences
      ↓
Dense
```

然后：

```text
SearchBelief.certainty
```

只负责改变：

```text
route K
route weights
discovery breadth
```

例如：

```text
C low:
dense K ↑
mode-balanced recall ↑

C high:
structured/BM25 precision ↑
discovery ↓
```

所以 Dialog State 本身完全不需要知道 retrieval weights。

---

# 十八、Ranking 也能无缝消费

Ranking 收：

```text
Preference
+
SearchBelief.certainty
```

两者分别回答：

### Preference

> 什么东西更相关？

### certainty

> 最终列表应该多集中还是多探索？

于是：

Score(d)=Relevance(d∣IntentState)+PreferenceMatch(d)Score(d) = Relevance(d\mid IntentState) + PreferenceMatch(d)

再：

λ=f(C)\lambda = f(C)

控制 MMR/diversification。

逻辑非常干净。

---

# 十九、Asking 也正好闭环

```text
SearchBelief.certainty
        ↓
是否还有必要问？

SearchBelief.facet_entropy
        ↓
问哪个维度？

InteractionContext.asked_facets
        ↓
有没有已经问过？

IntentState.preferences
        ↓
用户是否已经表达了这个信息？
```

四个问题分别有明确 source of truth。

这比让一个 LLM看完整 transcript 然后决定“问啥好”可解释得多。

---

# 二十、我认为这个版本最符合题目的哲学

官方说的是：

> **Dynamic State Machine**：积累信息，又能 Intent Override；

这对应：

```text
IntentState + StateDelta
```

官方说：

> **Over-Generality**：结果太宽时主动引导；

对应：

```text
Probe → SearchBelief.certainty
```

官方说：

> **Dual-Track Buying/Browsing**；

我们不硬分类，而是：

```text
certainty continuously controls exploration ↔ exploitation
```

官方说：

> **Adaptive Orchestration**；

对应：

```text
SearchBelief → retrieval/ranking/asking policy
```

而 Alibaba 当前生产 recommendation pipeline 本身也明确维护 multi-turn context、constraint caching、candidate-set tracking，并将 intent extraction → constrained multi-channel retrieval → ranking/re-ranking 串起来。([阿里云](https://www.alibabacloud.com/help/en/pai/best-practices-of-ai-conversational-recommendation-ai-shopping-guide?utm_source=chatgpt.com "Best practices for AI conversational recommendations - Platform For AI - Alibaba Cloud Documentation Center"))

所以这不是“为了 architecture 漂亮而增加三个对象”。

它实际上是：

> **用户意图事实、商品空间观测、交互上下文三个语义完全不同的数据源，必须分开；但每层只保存下游真正需要的最少信息。**

---

## 我现在愿意冻结的版本

```text
             latest user turn
                    │
                    ▼
           Query Understanding
                    │
                StateDelta
                    │
                    ▼
            ┌──────────────┐
            │ IntentState  │
            │ current need │
            └──────┬───────┘
                   │
                   ▼
                 Probe
                   │
                   ▼
           ┌───────────────┐
           │ SearchBelief  │
           │ C / modes / H │
           └───────┬───────┘
                   │
          ┌────────┼─────────┐
          ↓        ↓         ↓
      Retrieval  Ranking   Asking
          │        │         │
          └────────┼─────────┘
                   ↓
              products/question
                   │
                   ▼
          InteractionContext
                   │
                   └──────────→ next turn
```

我认为**这个复杂度刚好**：比普通 slot state 多了 `semantic preference + provenance/commitment + SearchBelief + referential context`，因此足以承载题目的哲学；但没有 mission graph、supersession graph、长期 memory subsystem、复杂 belief vector 等当前没有必要的东西。

如果后面 Query Understanding、retrieval、ranking 的具体实现变了，这个 state contract 大概率也不需要推倒重来——这才是判断它是否设计合理的关键。
