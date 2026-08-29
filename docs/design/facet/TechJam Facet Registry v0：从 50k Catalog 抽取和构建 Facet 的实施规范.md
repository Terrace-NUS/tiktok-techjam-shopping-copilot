# TechJam Facet Registry v0：从 50k Catalog 抽取和构建 Facet 的实施规范

## 0. 目标

针对 TikTok TechJam 2026 Track 4 的冻结 50,000 商品 catalog，构建一套 **catalog-derived、category-conditioned、typed、versioned 的 FacetRegistry**，供后续：

- Query Understanding 做 grounding；
- IntentState 保存 structured preferences；
- QueryCompiler 生成 structured filters / boosts；
- Probe 统计当前候选的 facet distribution；
- Asking 计算候选属性分歧；
- Retrieval / Ranking 做结构化匹配。

FacetRegistry 不是人工预先写一张购物属性表，也不是让 LLM 自由生成 ontology。

核心原则：

> Facet 必须由当前 catalog 中真实存在、质量可验证的商品属性支撑。

对于无法可靠落到 FacetRegistry 的用户表达，例如：

- "not too sporty"
- "comfortable for walking all day"
- "good for winter commuting"
- "looks professional but not too formal"

后续 Query Understanding 必须保留为 semantic preference，而不是创造一个不存在的 facet。

---

# 1. 开工前先检查真实数据，不要假设 schema

第一步只做 inspection，不修改现有搜索系统。

定位真实：

```text
data/catalog.jsonl
```

验证：

```text
row count == 50_000
```

先读取：

```text
前 100 行
随机 100 行
最后 100 行
```

并统计所有 top-level key：

```text
key
出现商品数
coverage
Python value type distribution
null count
empty count
```

特别检查：

```text
parent_asin
title
categories
features
details
store
description

price
main_category
average_rating
rating_number
```

不要因为 Amazon Reviews 2023 上游有某字段，就假设 TechJam frozen catalog 一定保留。

输出：

```text
artifacts/facets/catalog_schema_profile.json
artifacts/facets/catalog_schema_profile.md
```

---

# 2. Facet 的三个来源

只允许从以下三个来源产生 facet candidate。

## A. 明确的 top-level structured fields

候选包括但不限于：

```text
category
price
```

对于：

```text
store
```

不要自动重命名为 `brand`。

`store` 和 `brand` 语义并不完全相同。

如果：

```text
details["Brand"]
```

真实存在且质量足够，则构造 `brand`。

如果只有 `store`，则仍叫 `store`，除非后续人工审核证明在当前 catalog 中可安全视作 brand proxy。

以下通常是 ranking features，而不是购物 facet：

```text
average_rating
rating_number
```

不要默认加入 FacetRegistry。

---

## B. `details` 中的 structured key-value

这是主要 facet candidate 来源。

扫描所有：

```python
product["details"]
```

必须先检查实际数据类型。

如果是 dict，则统计每个 raw key，例如：

```text
Color
Fabric Type
Material
Department
Size
Outer Material
Pattern
Closure Type
Sole Material
Gem Type
Metal Type
...
```

为每个 raw key 统计：

```text
global_support
global_coverage

support_by_category
coverage_by_category

value_type_distribution
distinct_raw_values
unique_ratio

top_50_raw_values
missing_rate
empty_rate

20~50 个具体 product examples
```

输出：

```text
artifacts/facets/raw_detail_key_stats.csv
artifacts/facets/raw_detail_key_examples.jsonl
```

**这一阶段不要过滤低频 key。**

先完整观测，再筛选。

---

## C. Category hierarchy

解析真实：

```text
categories
```

字段。

不要预设它一定是：

```python
list[str]
```

先 inspection。

构造规范化的：

```text
CategoryPath
```

例如：

```text
Clothing, Shoes & Jewelry
  > Men
  > Shoes
  > Loafers & Slip-Ons
```

具体层级以真实数据为准。

Facet 的 coverage 必须主要计算：

```text
coverage(facet | category)
```

而不是只算：

```text
coverage(facet | whole catalog)
```

因为：

```text
ring_size
```

对 Jewelry 很重要，但全 catalog coverage 可能很低。

同理：

```text
shoe_width
```

只应该在 Shoes 下评价。

---

# 3. Category bucket 的构建

不要对极小 leaf category 单独学习 facet。

先统计每个 category path 的商品数。

建议构造稳定 category bucket：

```text
如果 leaf category support >= MIN_CATEGORY_SUPPORT
    使用 leaf category

否则
    回退到最近的、support 足够的 parent
```

P0 默认：

```text
MIN_CATEGORY_SUPPORT = 100
```

但必须输出分布，再决定该值是否合理。

同时保留：

```text
raw_category_path
resolved_category_bucket
```

不要破坏原始数据。

最终每个 facet 应具有：

```text
applicable_categories
```

而不是假设全局适用。

---

# 4. Raw key normalization：先规范字符串，不先合并语义

第一层 normalization 只能做低风险操作：

```text
Unicode NFKC
casefold
trim
collapse whitespace
统一常见标点
```

例如：

```text
"Fabric Type"
"fabric type"
" Fabric  Type "
```

可以自动归为同一 normalized key。

但是：

```text
Material
Fabric Type
Outer Material
```

**禁止因为名字相似就直接合并成 material。**

原因是不同 category 下它们可能真的表示不同概念。

第一阶段产生：

```text
normalized_raw_key
→ raw source keys
```

---

# 5. Alias / merge proposal：机器可以提议，但不能自动决定

下一步允许系统寻找可能属于同一 facet 的 raw keys。

例如：

```text
Color
Colour
Item Color
```

可能应合并。

但判断依据至少结合：

### 5.1 Key name similarity

例如：

```text
color / colour
fabric / fabric type
```

### 5.2 Value vocabulary overlap

例如：

```text
Color:
black, white, blue, red

Colour:
black, white, navy, red
```

高度重合支持 merge。

### 5.3 Category distribution overlap

如果两个 key 主要出现在相同 category，也支持 merge。

### 5.4 Semantic meaning

可以使用：

- embeddings；
- LLM；
- Shopify/eBay taxonomy；

生成 **merge suggestion**。

但自动工具只允许输出：

```text
MERGE_PROPOSAL
```

不能直接修改 registry。

例如：

```yaml
candidate_group:
  proposed_name: color
  raw_keys:
    - Color
    - Colour
    - Item Color

evidence:
  name_similarity: ...
  value_overlap: ...
  category_overlap: ...

decision:
  NEEDS_REVIEW
```

---

# 6. 外部 taxonomy 只用于参考，不作为数据事实

可选使用 Shopify Standard Product Taxonomy 2026-05 等稳定 release 作为：

```text
canonical naming prior
attribute alias prior
category/attribute relationship prior
```

例如外部 taxonomy 可以帮助判断：

```text
Fabric Type
```

更适合叫：

```text
fabric
```

还是：

```text
material
```

但是：

> 如果 Shopify 有 `neckline`，当前 50k catalog 没有可靠 neckline 数据，不允许因此创造 neckline facet。

外部 taxonomy 的角色只能是：

```text
Current catalog evidence
        ↓
facet candidate already exists
        ↓
external taxonomy helps naming/mapping
```

而不能反过来。

---

# 7. Facet candidate 必须经过数据质量审核

对每个候选 canonical facet 计算：

```text
support
coverage by category
value cardinality
unique ratio
dominant value ratio
type parse rate
missing rate
value examples
source keys
```

重点看以下几个维度。

## 7.1 Coverage

定义：

```text
coverage =
有有效 facet value 的商品数
/
该 applicable category 商品总数
```

重点使用 category-conditioned coverage。

不要因为 global coverage 低就拒绝。

---

## 7.2 Discriminative usefulness

如果：

```text
99.5% 商品 value 完全一样
```

即使 coverage 很高，也不是有价值的 facet。

至少记录：

```text
dominant_value_ratio
entropy
distinct_value_count
```

---

## 7.3 是否接近唯一 ID

例如一个字段：

```text
50,000 商品有 47,000 个不同值
```

通常不适合作为 selectable categorical facet。

记录：

```text
unique_ratio =
distinct_values / non_missing_count
```

高 unique ratio 的字段倾向于：

```text
OPEN / SEARCH_ONLY
```

而不是 CLOSED categorical facet。

---

# 8. 自动推断 facet 数据类型

支持：

```text
BOOLEAN
CATEGORICAL
NUMERIC
TEXT
```

P0 不需要更多类型。

## BOOLEAN

若绝大多数有效值可以稳定映射：

```text
yes/no
true/false
```

才标 boolean。

## NUMERIC

如果有效值中：

```text
>= 90%
```

能解析成数值或同一可转换单位体系，则 candidate type 为 numeric。

但最终 promotion 前人工抽样验证。

例如：

```text
Heel Height: 3 inches
```

可以考虑 numeric + unit normalization。

而：

```text
Size: 8 / Medium / One Size
```

不能因为部分是数字就当 numeric。

## CATEGORICAL

有重复的稳定离散值，例如：

```text
black
white
blue
```

## TEXT

值高度开放：

```text
model name
free-form description
```

不适合作为 closed facet values。

---

# 9. Value cardinality

参考实际商品数据判断一个商品对该 facet 是：

```text
SINGLE
MULTI
```

不要从 key 名猜。

如果真实 value 数据本身是 list，或者大量商品明确具有多个独立值，可以判 MULTI。

注意：

```text
"Black/White"
```

不应该自动 split 成：

```text
["black", "white"]
```

因为它可能是一个组合色描述。

Multi-value splitting 必须由 facet-specific normalizer 决定。

---

# 10. Value mode

每个 facet 再标：

```text
CLOSED
OPEN
HYBRID
```

## CLOSED

值域比较稳定，例如某些 boolean/有限枚举。

可维护：

```text
canonical_values
```

## OPEN

理论值域很大，例如：

```text
brand
```

不维护完整 enumerable whitelist。

## HYBRID

存在一组高频 canonical values，但允许长尾值。

服饰的：

```text
color
material
```

很可能更适合 HYBRID。

具体根据真实数据决定。

---

# 11. Value normalization 必须保留 provenance

不要覆盖 raw value。

每个规范化结果保留：

```text
raw_value
canonical_value
normalizer
confidence
source
```

例如：

```text
BLACK
Black
black
```

可以安全映射：

```text
black
```

但：

```text
Jet Black
Charcoal
Graphite
```

不要第一版强行都压成 black。

宁可少归一化，不要制造 false equivalence。

尤其：

```text
size
material
style
```

应极度保守。

---

# 12. Facet promotion 采用分级，而不是一个简单 coverage threshold

不要写：

```python
if coverage > 0.3:
    facet = True
```

最终给每个 candidate 一个状态：

```text
ACCEPT
SEARCH_ONLY
SEMANTIC_ONLY
REJECT
NEEDS_REVIEW
```

## ACCEPT

表示：

- 含义稳定；
- 有可靠的结构化来源；
- 类型和 value semantics 清楚；
- 至少在某些 category 中 coverage 足够；
- 下游 structured matching 有实际意义。

## SEARCH_ONLY

值结构存在，但不适合 hard filter / clarification。

例如高 cardinality 或数据比较稀疏。

## SEMANTIC_ONLY

概念对用户有价值，但当前 catalog 结构化质量不足。

后续从文本语义处理。

## REJECT

例如：

- seller-specific internal metadata；
- care instruction 等当前系统不打算支持的字段；
- 数据完全脏；
- 几乎恒定；
- 无实际购物决策意义。

## NEEDS_REVIEW

自动流程无法安全判断。

---

# 13. P0 promotion 的建议数据阈值

这些只是初始默认配置，不是科学定律，必须根据 profile report 调整。

### 最小支持量

在至少一个 category bucket：

```text
support >= 100
```

否则默认不进入 P0 structured facet。

### Coverage

如果：

```text
coverage >= 0.30
```

可考虑 `ACCEPT`。

如果：

```text
0.10 <= coverage < 0.30
```

通常考虑：

```text
SEARCH_ONLY / NEEDS_REVIEW
```

但像 ring size / shoe width 等业务重要的 category-specific facet，可以人工例外。

### Constant rejection

如果：

```text
dominant_value_ratio >= 0.98
```

通常拒绝作为搜索 facet。

### Numeric parsing

要作为 NUMERIC：

```text
parse_success >= 0.90
```

最终审核目标建议：

```text
>= 0.95
```

### Clarification eligibility

静态上只允许：

```text
coverage >= 0.50
且
值对人类可理解
且
不是 identifier-like
```

进入：

```text
clarification_eligible = true
```

但真正某一轮问不问仍由 Probe 的动态 FacetStats 决定。

---

# 14. FacetRegistry 最终 schema

建议 P0：

```python
@dataclass(frozen=True)
class FacetDefinition:
    id: str
    name: str

    aliases: tuple[str, ...]
    source_keys: tuple[str, ...]

    applicable_categories: tuple[str, ...]

    data_type: Literal[
        "boolean",
        "categorical",
        "numeric",
        "text",
    ]

    item_cardinality: Literal[
        "single",
        "multi",
    ]

    value_mode: Literal[
        "closed",
        "open",
        "hybrid",
    ]

    allowed_operators: tuple[str, ...]

    canonical_values: tuple[str, ...]

    normalizer_id: str | None

    structured_match: bool
    hard_filter_safe: bool
    clarification_eligible: bool
    searchable_values: bool
```

不要在 `FacetDefinition` 里保存：

```text
当前 entropy
当前 candidate coverage
question utility
retrieval weight
```

这些全部是 runtime 信息。

---

# 15. 另建 `FacetCatalogStats`

静态 ontology 和 catalog 数据统计分开。

```python
@dataclass(frozen=True)
class FacetCatalogStats:
    facet_id: str
    category: str

    product_count: int
    present_count: int
    coverage: float

    distinct_value_count: int
    unique_ratio: float
    dominant_value_ratio: float
    entropy: float

    top_values: tuple[ValueMass, ...]
```

这样以后：

```text
FacetDefinition
```

回答：

> color 是什么？

而：

```text
FacetCatalogStats
```

回答：

> color 在当前 catalog 的 shoes 中数据质量怎么样？

---

# 16. Registry 必须版本化

最终：

```python
@dataclass(frozen=True)
class FacetRegistry:
    version: str
    catalog_hash: str

    facets: Mapping[str, FacetDefinition]
```

版本不能只写：

```text
v1
```

manifest 中至少记录：

```text
catalog SHA256
builder code version / git commit
override config hash
registry version
```

同一个：

```text
catalog
+ code
+ config
```

必须生成完全一致的 registry。

---

# 17. 人工判断必须显式写在配置里

不要在 Python 中偷偷：

```python
if key == "Fabric Type":
    return "material"
```

维护：

```text
configs/facets/overrides.yaml
```

例如：

```yaml
merges:
  color:
    - Color
    - Colour

reject:
  Care Instructions:
    reason: "not in P0 search intent scope"

capabilities:
  brand:
    clarification_eligible: false
```

所有人工决定必须：

```text
可读
可 review
可 diff
可重放
```

这也是 FacetRegistry 能真正成为协议，而不是一次性数据清洗脚本的关键。

---

# 18. 输出一个供人工审核的报告

在正式构建 registry 前生成：

```text
artifacts/facets/review_report.md
```

每个 candidate 展示：

```text
Proposed facet: material

Raw keys:
- Material
- Fabric Type
- Outer Material

Applicable categories:
- Shoes
- Clothing

Coverage:
...

Top values:
...

Example products:
...

Potential external taxonomy mapping:
...

Suggested decision:
NEEDS_REVIEW
```

不要让 Codex 自动接受低置信 merge。

---

# 19. 第一版不要从 title/features/description “发现新 facet”

这是一个重要边界。

## V0 Registry

Facet ontology 只从：

```text
top-level structured fields
+
details structured keys
+
categories
```

产生。

原因：

> 从自由文本中开放式“发现属性”非常容易得到数百个概念，而且无法判断它们究竟是 stable facet 还是 semantic concept。

---

# 20. Registry 建好后，可以做 V1 value enrichment

这和“发现 facet”是两回事。

例如已经知道：

```text
facet = material
```

但大量商品：

```text
details["Material"] = missing
```

可以尝试从：

```text
title
features
description
```

抽取 material value。

顺序：

### A. Exact dictionary matching

如果 closed/hybrid facet 有 canonical value：

```text
"100% cotton shirt"
→ cotton
```

### B. Rule / regex

numeric facet：

```text
"3 inch heel"
→ heel_height = 3 inch
```

### C. 可选模型/LLM extraction

只有 facet 已经定义后，才允许问模型：

> 对于 facet `material`，这个商品文本是否明确支持某个 value？

不要问：

> 这个商品还有哪些属性？

也就是说：

```text
先有 schema
→ 再 extract value
```

不要：

```text
LLM 看文本
→ 自由创造 schema
```

这也符合已有工业 attribute-value extraction 的研究方向。

---

# 21. Inferred values 必须和原始 structured values 分开

最终 normalized product 可以有：

```python
FacetValueEvidence:
    facet_id
    canonical_value

    source:
        structured_details
        top_level
        exact_text_extract
        model_inferred

    evidence_text
    confidence
```

下游可以规定：

```text
structured_details
→ 最高信任

exact_text_extract
→ 高信任

model_inferred
→ soft evidence
```

不要让 LLM inferred value 自动成为 hard-filter truth。

---

# 22. 关于 Query Understanding 的接口

FacetRegistry 建好后，Query Understanding 必须遵守：

```text
用户表达
      ↓
FacetRegistry 是否能可靠 grounding？
      │
    ┌─┴─┐
    │   │
   YES  NO
    │   │
    ↓   ↓
structured preference
        semantic preference
```

禁止：

```text
LLM invent facet
```

例如：

```text
"under $100"
→ price <= 100
```

而：

```text
"not too sporty"
```

如果 Registry 没有可靠 `sportiness`：

```text
facet=None
semantic_text="sporty"
semantic_polarity=negative
```

这属于正常结果，不是 grounding failure。

---

# 23. 关于 official `ask_attribute`

不要使用官方：

```text
category
material
color
size
style
brand
budget
feature
use_case
...
```

来反向定义 FacetRegistry。

这些只是 TechJam protocol adapter 的 coarse labels。

正确方向：

```text
Our FacetRegistry
       ↓
adapter mapping
       ↓
official ask_attribute
```

例如：

```text
fabric
outer_material
sole_material
```

最终都可以映射：

```text
material
```

但 domain model 不应该因此把三个不同 facet 强行合并。

---

# 24. 建议文件结构

先检查现有 repo，再决定精确路径；如果没有对应模块，可使用：

```text
src/catalog/facets/
    models.py
    catalog_profiler.py
    category_profiler.py
    raw_key_profiler.py
    key_normalizer.py
    value_profiler.py
    registry_builder.py
    registry_loader.py
    value_normalizers.py

configs/facets/
    thresholds.yaml
    overrides.yaml

scripts/
    profile_catalog.py
    build_facet_registry.py

artifacts/facets/
    catalog_schema_profile.json
    category_stats.csv
    raw_detail_key_stats.csv
    raw_detail_key_examples.jsonl
    facet_candidates.jsonl
    review_report.md
    facet_registry.json
    facet_catalog_stats.jsonl
    facet_build_manifest.json

tests/catalog/facets/
    test_key_normalization.py
    test_value_typing.py
    test_registry_validation.py
    test_registry_determinism.py
```

不要现在实现：

```text
Query Understanding
LLM parser
Probe
certainty
retrieval
ranking
asking
```

这个任务只负责 catalog → FacetRegistry。

---

# 25. 必须测试的 invariants

至少：

- catalog 所有 50k 行可读取；
- `parent_asin` 唯一性/异常被报告；
- raw details key 统计可重复；
- key normalization deterministic；
- canonical facet ID 唯一；
- alias 不能映射到两个 facet；
- applicable category 必须存在；
- numeric facet 的 allowed operator 合法；
- `IN/NOT_IN` 只允许 categorical-like facet；
- CLOSED facet 必须有 canonical values；
- registry build 两次 byte-for-byte 或 semantic-equivalent；
- invalid override 会使 build fail，而不是静默忽略；
- source key → canonical facet mapping 可追踪；
- 每个 accepted facet 都有 stats；
- 每个 rejected candidate 都记录 rejection reason；
- external taxonomy 不能单独创建当前 catalog 不支持的 facet。

---

# 26. Definition of Done

任务完成时必须给出：

### A. Catalog profile

告诉我真实 50k 中：

```text
有哪些字段
details 有多少种 raw keys
category 分布如何
price 是否存在/coverage
```

### B. Facet candidate report

至少列：

```text
raw key
canonical candidate
coverage by category
type
cardinality
value mode
top values
decision
reason
```

### C. FacetRegistry v0

机器可读：

```text
facet_registry.json
```

### D. Rejected / semantic-only report

明确告诉我：

> 哪些看起来像购物属性，但因为当前数据不足没有进入 structured registry。

### E. 不要直接开始 Query Understanding

先向我汇报：

1. 推荐进入 v0 的 facets；
2. 有争议的 merge；
3. 数据非常脏但业务可能重要的 facet；
4. 哪些 facet 适合 hard filtering；
5. 哪些只适合 soft/search；
6. 哪些静态上适合 clarification；
7. 真实数据中最严重的 coverage / normalization 问题。

等这份结果 review 后再冻结 FacetRegistry v0，并进入 Query Understanding。

---

# 最核心的设计准则

不要为了让状态看起来结构化而制造错误结构。

优先级是：

```text
可靠 structured facet
>
保守 semantic representation
>
错误的 structured grounding
```

如果不确定一个概念是否能稳定成为 facet：

```text
宁可暂时 semantic-only
```

后续 Dense Retrieval / semantic reranking 仍然可以使用它。

FacetRegistry 的目的不是描述人类购物世界中的所有属性，而是描述：

> **当前这 50,000 件商品中，我们能够稳定、可执行、可验证地使用哪些结构化商品维度。**