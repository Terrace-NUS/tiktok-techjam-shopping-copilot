# Facet：我们怎样确定、构造并从用户语言中抽取属性

## 1. 先把两个问题分开

大家说“facet 抽取”时，实际上经常混在一起的是两件不同的事。

### Catalog-side facet construction

回答：

> catalog 中到底有哪些可以相信的属性？这些值来自哪个原始字段？在哪些商品类别适用？系统允许
> 用它做过滤、Probe 或追问吗？

这是离线、可审计、有人工 review gate 的工作。

### QU-side facet extraction

回答：

> 用户这一轮说了哪些条件？例如“100 美元以内”“不要真皮”“适合走一天”。

这是每个用户 turn 在线执行的 Query Understanding 工作。

两者的关系是：

> Catalog Semantic 发布“可以相信的数据能力”；Query Understanding 抽取“用户本轮表达了什么”。
> QU 不能把模型猜测反向写成 catalog 事实。

## 2. 当前系统不是只有一种 facet authority

为了既保留严格的数据证据，又满足三日 hackathon 的表达能力，当前 registry 明确区分两种 authority。

### Catalog-verified

普通 catalog-verified facet 的商品侧事实经过冻结 catalog、人工 Gate A/B、全量 evidence 和 release
验证。Category 是由 CS1 和 release 单独管理的保留 scope，不经过普通 Gate A/B。

当前包括：

| 名称 | 身份 | 当前能力 |
| --- | --- | --- |
| `price` | 普通 catalog facet | intent commit、保守 retrieval、Probe；不主动追问 |
| `system_product_category` | release 保留的 category scope | category grounding 和 scope authority |

严格地说，当前经过普通 Gate A/B 发布的 catalog facet 只有 `price`；category 是保留的 release
scope，不是普通 Gate-A facet。

### Retrieval-derived

当前为 QU 和后续检索提供九个宽 categorical facet：

```text
brand
color
department
feature
gender
material
size
style
use_case
```

它们允许我们结构化保存“不要黑色”“女士”“真皮”“通勤”等用户要求，并进行确定性文本规范化。

但它们不声称：

- 已经从 50k catalog 的 raw field 经 Gate A/B 审核；
- 每件商品在这些维度上都有可靠值；
- Session Context 中的用户要求本身就证明某商品满足条件。

商品侧真值必须由后续 Retrieval Evidence Index 或检索证据判断。没有证据就是 unknown，不能让
LLM 脱离原文补成 catalog truth。当前新增的 Product Fact Card 会让 DeepSeek 阅读完整商品字段，但每条
事实都必须引用真实 `source_ref + evidence`；它属于 model-derived retrieval evidence，不冒充
Gate-A/B verified fact。

### Semantic-only

不能可靠映射为结构化 facet 的表达，保留成开放语义：

```text
“看起来专业，但不要太正式”
“走一天也不累”
“不要太像游客”
```

semantic-only 不是失败。它是避免模型为了结构化而捏造 ontology 的安全出口。

## 3. Catalog facet 的完整构建主线

```text
冻结 50k catalog
    ↓
CS0 只读 profiling
    ↓
CS1 category 坐标系 + 人工 scope 审核
    ↓
CS2 source profiling + Gate A：批准怎样抽取
    ↓
CS3 全量 EvidenceStore / resolver / index / stats
    ↓
CS4 Gate B：批准运行时怎样使用
    ↓
CS5 runtime registry / lexicon / grounding
    ↓
CS6 不可变 CatalogSemanticRelease
    ↓
CS7 Gateway 与 SessionContext authority
```

自动化负责重放审核决定，不能替代审核决定。

## 4. CS0：先观察真实数据，不先发明 schema

输入是冻结的：

```text
data/catalog.jsonl
```

真实基线：

```text
商品行数                 50,000
raw category prefixes    1,832
details raw keys           287
catalog bytes        60,546,327
```

Profiler 只读统计：

- 所有 top-level key；
- 字段类型、null、empty、support；
- exact category path；
- 每个 `details` raw key；
- category-conditioned coverage；
- 稳定的原始样本。

这一步故意不做：

- 不把 `store` 自动改名为 `brand`；
- 不把 `Material`、`Fabric Type`、`Outer Material` 自动合并；
- 不因为 coverage 高就批准 facet；
- 不从 title、features、description 中让 LLM“发现”catalog truth；
- 不修改原始数据。

核心原则是：

> 先观察，后命名；先保留原始 provenance，后决定语义。

## 5. CS1：先建立 category 语义坐标系

为什么不能只看全 catalog coverage？

```text
ring_size 对 Jewelry 很重要，但全局 coverage 可能很低；
shoe_width 对 Footwear 很重要，但对 Jewelry 完全不适用。
```

所以 facet 必须在 category scope 中评价。

### Pass A：机器只建图

机器从 exact raw path 建立 prefix graph，并输出：

- raw-path mapping；
- lexical collision report；
- 每个节点的 support；
- 一个空的人工选择模板。

它不会自动决定哪些节点应成为面向用户的 scope。

### 人工 review

人工选择稳定、可解释的 category scopes，并写入 source-controlled config。

当前结果：

```text
1,832 个 raw/canonical category nodes
→ 15 个 reviewed category scopes
→ 50,000 个商品 assignment 全部 KNOWN
```

### Pass B：确定性 materialization

根据人工选择生成：

- `CategoryRegistry`；
- 每个 scope 的完整 subtree closure；
- 50k product-category assignments；
- 稳定 scope ID。

## 6. CS2：列出所有候选来源，再通过 Gate A

Source profile 穷举：

```text
top-level price
top-level store
287 个 details raw keys
= 289 个 exact source locators
```

结合 15 个 scope，共产生：

```text
289 × 15 = 4,335 个 scope-source observation rows
```

低 support、空 key 和坏值都保留在 review evidence 中，不在 profiling 阶段偷偷过滤。

### Gate A 回答什么

Gate A 只回答：

> 这个语义概念能否从明确的 raw source 被确定性抽取？

需要人工批准：

```text
facet ID 与人类名称
data type：boolean / categorical / numeric / text
item cardinality
applicable category scopes
exact raw source binding
extractor
catalog value normalizer
source priority / resolver
```

机器可以提出 alias / merge 建议，但不能自行把 raw keys 合并成 canonical facet。建议至少结合：

- key 名相似度；
- value vocabulary overlap；
- category distribution overlap；
- 样本中的真实语义。

### 当前 Gate A 结果

当前只批准了 `price`：

```text
kind                  NUMERIC
cardinality           SINGLE
source                top-level price
applicability         root scope
extractor             top_level_price_usd_v1
catalog normalizer    usd_cent_interval_v1
resolver              priority_exact_v1
```

其余 288 个 source locators 仍然只是待审 evidence，不是已经发布的 facet。

## 7. CS3：把批准的抽取规则跑遍 50k 商品

只对 Gate A 已批准的 binding 执行全量解析。每个商品、每个 facet 都进入四态语义：

```text
KNOWN           有可靠结构化值
UNKNOWN         缺失、无效或证据不足
CONFLICT        多来源之间无法安全合并
NOT_APPLICABLE  对当前 category 不适用
```

这比“有值/没值”更重要，因为 unknown 不是负证据。

### 当前 price 结果

```text
10,410 个 JSON number              → exact interval
     5 个 "from ..." 字符串        → inclusive lower-bound interval
39,473 个 null                     → EMPTY
   112 个 em-dash placeholder      → INVALID
----------------------------------------------------
10,415 KNOWN
39,585 UNKNOWN
0 CONFLICT
0 NOT_APPLICABLE
```

所以“39,473 个商品没有价格”更准确的说法是：

> raw `price` 是明确的 null，因此我们没有可用的结构化价格证据；系统没有替它们补价格。

### CS3 产物

- `FacetEvidenceStore`：50k 商品逐行证据和状态；
- `ProductFacetIndex`：只存 KNOWN / CONFLICT 的稀疏索引；
- category-conditioned facet stats；
- catalog read-only audit。

## 8. CS4：Gate B 决定 facet 可以怎样使用

Gate A 只批准“能抽”。Gate B 才批准“能用来做什么”。

每个 exact category scope 单独决定：

```text
intent_committable
retrieval_eligible
probe_eligible
clarification_eligible
```

### price 的当前决定

15 个 scope 全部批准：

```text
用户明确预算可以写入 IntentState           true
保守价格检索                               true
Probe 可以查看价格分布                     true
系统主动追问预算                           false
```

“保守检索”的意思是：

```text
明确证明超预算    -> 可以删除
明确满足预算      -> 保留
价格 UNKNOWN      -> 也保留
```

公共 200 个目标商品中，178 个价格 KNOWN，22 个 UNKNOWN。审计为每个已知目标构造了与其价格
兼容的合成预算：若错误地要求“必须有已知且满足预算的价格”，会把 22 个 unknown 目标误删；
保守规则能保留全部 200 个兼容目标。

这个实验只证明 UNKNOWN 不应被当成超预算证据。官方 toy set 没有真实用户预算请求，因此它不能
证明真实预算检索效果，也不能证明主动追问预算有收益。

## 9. CS5–CS7：发布为运行时能力

### Runtime projection

CS5 把已批准决定投影为机器可调用的：

- runtime facet registry；
- value lexicon / aliases；
- intent value normalizer；
- release-bound grounding service。

Grounder 接收的是已经抽取好的 candidate：

```text
facet + operator + value + final category scope
```

它不会解析自然语言，也不会分配 Preference ID。它只返回：

```text
GROUNDED
SEMANTIC_ONLY
AMBIGUOUS
```

### Immutable release

CS6 把 catalog、category、Gate A/B、evidence/index/stats、runtime registry 和 reviewed config 等
13 个内容寻址成员装配为一个自包含 release。任何成员缺失、过期、重排或被修改，loader 都会拒绝。

### Gateway

CS7 让 Session Context 只能通过绑定该 release 的 Gateway 提交 catalog-sensitive intent。Gateway
负责验证，不重新解释 catalog。

## 10. 原始数据为什么没有被“修改”

我们没有生成一份“补完价格、清洗属性后替换原数据”的 catalog。

证据包括：

- build 前后 catalog SHA-256 和 `60,546,327` bytes 完全一致；
- derived 输出目录与输入目录分离；
- null price 保持 `EMPTY/UNKNOWN`，没有补值；
- raw value、parse status、canonical value 分栏保存在 evidence；
- ProductFacetIndex 是旁路稀疏索引；
- release 中的 `catalog.jsonl` 是 exact byte copy，不经过解析重写；
- manifest 固定每个成员的 hash 和 size。

因此我们所做的是：

```text
原始数据       保持不变
derived evidence/index/stats   旁路生成
reviewed config                source-controlled
runtime release                内容寻址、可验证
```

## 11. Query Understanding 在线怎样抽取 facet

用户说：

> 女士 8 码徒步靴，100 美元以内，不要真皮，必须防水，最好走一天也不累。

DeepSeek 的 typed function arguments 可以分成：

```text
structured
    category = category_N (General footwear，前提是该 ref 存在于请求)
    gender = women
    size = 8
    material NOT_IN leather
    feature = waterproof

price
    price <= 100 USD

semantic
    positive soft: comfortable enough to walk in all day
```

`hiking boots` 本身保存在 goal 中。当前发布的 15 个 scope 没有任意粒度的 “Hiking Boots” scope，
所以模型不能凭空创建一个；没有可靠 category ref 时可以只保留 goal。

本地 materializer 随后按 authority 分流：

| 条件 | Authority | 本地处理 |
| --- | --- | --- |
| category | release reserved scope | `category_N` 映射回可信 scope ID |
| price | catalog-verified | USD 精确转 cents，再走 release grounder |
| gender/size/material/feature | retrieval-derived | registry 规范化，商品真值留给 retrieval evidence |
| all-day comfort | semantic-only | 完整保留 meaning 和 polarity |

如果 DeepSeek 输出一个 registry 不认识的 facet，例如 `vibe`，系统不会把它偷偷注册为 catalog
facet，而是保留为 semantic-only，并在 trace 中记录 fallback。

## 12. 为什么 official ask_attribute 不是 FacetRegistry

官方 simulator 的 `ask_attribute` 是适配器协议键，不是 catalog ontology。

例如：

```text
other
```

可以是官方交互协议中的选项，但它不是一个合法 structured facet。我们可以把官方键映射到内部
facet 或问题策略，但不能反过来让 evaluator 定义系统的语义 registry。

## 13. Product Fact Card：两侧使用同一种语言

旧 Evidence Index 依赖字段白名单和关键词，因此会漏掉只写在 description 中的 material/color。
现在商品侧与 QU 共享 `shopping_facet_language_v1`：两侧都先判断主语、部件和否定范围，都保留原文证据，
也都能从一段描述抽出多个 facet。

区别是 QU 输出用户偏好和 hard/soft；商品卡输出 present/absent 商品事实。商品卡还保存 aliases，用于连接
`gossypium` 与 `cotton` 等明确同义术语，但 alias 不能代替原文引用。

21 件商品 live test 已达到 21/21 tool call + 本地验证成功，共生成 647 条去重事实；两个原先因
description 未进入 material evidence 而失败的目标商品都恢复了 fabric/cotton 事实。完整协议见
[Product Fact Cards v1](../design/catalog_semantic/product-fact-cards-v1.md)。当前尚未启动 50k 全量生成，
也尚未接入正式 hard mask。

## 14. 如果以后要正式增加 color 或 material

不能只在 prompt 里加一个名字。正确流程是：

1. 从 exact raw source profile 找候选 key；
2. 查看 category-conditioned support、coverage、type 和稳定样本；
3. 分析 raw key alias / merge，但保留 provenance；
4. 人工 Gate A 批准 definition、applicability、binding、extractor、normalizer 和 resolver；
5. 全量构建 EvidenceStore、ProductFacetIndex 和 stats；
6. 人工 Gate B 批准 intent、retrieval、Probe、clarification capability；
7. 生成 runtime registry / lexicon / grounder；
8. 组装新 release；
9. 让 Gateway 绑定新 release；
10. 最后才把它从 `retrieval_derived` 晋升为 `catalog_verified`。

## 15. 常见误解

### “coverage 高就是一个好 facet”

不一定。字段可能接近唯一 ID、值极脏、语义不稳定，或者只适合 ranking。

### “缺失就等于不满足”

错误。UNKNOWN 只能表示无法判断。

### “LLM 能看 title 猜出来，所以就是 catalog truth”

错误。模型推断可以作为 retrieval evidence 或 semantic signal，但不能冒充 Gate-A/B verified fact；
Product Fact Card 还要求每条事实引用原始 source，无法引用的猜测必须丢弃。

### “现在 registry 有 material，所以 material 已经从 50k catalog 发布”

错误。当前 material 是 retrieval-derived 查询词表；正式 catalog release 的普通 facet 仍只有 price。

### “Gate A 批准后就能直接 hard filter”

错误。Gate A 只批准抽取；运行时权限由 Gate B 决定。

## 权威入口

- [Catalog Semantic contract](../design/catalog_semantic/contract-v0.md)
- [实施方法](../design/catalog_semantic/methodology-v0.md)
- [当前真实状态与数据](../design/catalog_semantic/README.md)
- [Facet Registry 研究报告](<../design/facet/TechJam Facet Registry v0：从 50k Catalog 抽取和构建 Facet 的实施规范.md>)
- [当前 retrieval-derived 词表](../../src/shopping_copilot/session_context/wide_facets.py)
- [Catalog read-only audit](../../artifacts/catalog-semantic/resolution-candidate/catalog-read-only-audit.json)
