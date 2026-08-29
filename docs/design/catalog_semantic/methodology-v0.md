# Catalog Semantic Layer 工程方法论 v0

- 状态：**当前 P0 工程实施指南**
- 规范真源：[Catalog Semantic Layer Contract v0](contract-v0.md)
- 实现状态：[Catalog Semantic Layer README](README.md)
- 最后更新：**2026-08-27**

本文回答的是“为什么按这条路线做、每一步做什么、需要谁审核、如何验收”。
它不是新的 runtime schema，也不复制 contract 的全部字段定义。若本文、研究报告、
代码注释或历史讨论与 `contract-v0.md` 冲突，以 contract 为准。

## 1. 文档职责

Catalog semantic 工作有三类文档，各自只承担一种职责：

| 文档 | 回答的问题 | 权威性 |
| --- | --- | --- |
| `contract-v0.md` | 数据结构、算法、hash、状态转换和 invariants 究竟是什么 | 规范性真源 |
| 本文 | 应该以什么顺序实施、审核和验证这些规则 | 工程实施指南 |
| Facet 研究报告 | 为什么需要这些边界、真实数据暴露了什么问题 | 研究输入 |

本文使用独立的 catalog-semantic 里程碑 `CS0` 至 `CS8`，避免与 session-context
implementation plan 已定义的 M4/M5 重名。此前讨论中的“M4A Category Foundation”从本文起
正式对应 `CS1 Category Foundation`；M4A 只保留为历史别名，不再用于后续文件和代码命名。
`CS7` 才真正接触 session store，`CS0` 至 `CS6` 都是离线、确定性的 catalog build 工作。
实时进度只在本目录 README 维护，避免本文成为第二份容易过期的状态真源。

## 2. 问题定义

这个系统不能把 Amazon metadata 直接当成用户可查询的 facet，原因不是代码风格，而是
数据本身有三个事实：

1. 真正用于购物决策的 structured fields 往往稀疏且 category-conditioned，字段缺失通常
   只代表“不知道”；
2. 同名 raw key 在不同类别下可能表达不同含义；
3. 正证据、负证据、冲突和不适用必须被区分，否则 retrieval、Probe 和 Asking 都会收到
   虚假的确定性。

因此系统需要分开四种事实：

| 层 | 它拥有的事实 | 它不能决定的事情 |
| --- | --- | --- |
| Raw profiler | catalog 实际出现了哪些路径、key、值和稀疏度 | facet 语义和 runtime 权限 |
| Catalog Semantic Layer | 商品类别、facet evidence、resolved product facts、runtime capability | 用户这句话想表达什么 |
| Query Understanding | 当前用户表达对应什么候选 facet、operator 和 value | 商品是否真的具有某属性 |
| Session Context | 已被接受的用户需求、交互记录和可信 Probe belief | catalog truth、检索权重和提问策略 |
| Retrieval / Ranking / Asking | 如何消费已发布事实服务当前目标 | 修改 semantic truth 或绕过 Gateway 写状态 |

最终目标不是建立一个“大而全的属性表”，而是建立一条可追溯链：

```text
raw source value
    -> reviewed SourceBinding
    -> typed evidence
    -> resolved product fact
    -> exact-scope runtime capability
    -> grounded user predicate
    -> catalog-bound session commit
```

这些结果不能互相替代：没有 accepted evidence 得到 `UNKNOWN`；同一权威层出现不兼容 valid
evidence 才得到 `CONFLICT`；只有 KNOWN category assignment 加 reviewed FacetApplicability 才能
证明 `NOT_APPLICABLE`；用户表达无法可靠 grounding 时才保留为 semantic-only。任何路径都
不能为了填满结构而猜结论。

## 3. 核心工作原则

### 3.1 先观察，后命名

Profiler 保留 exact raw path、exact raw key 和 exact raw value shape。Facet ID、CategoryScope、
alias 和 normalizer 都在人工 review 后发布，不能由字段名直接自动生成。

例如出现 `details["Brand"]` 并不自动证明：

```text
facet_id = brand
details.Brand 与 top-level store 等价
该字段在全部 category 下语义相同
```

这些都是 Gate A 决策。

### 3.2 Category 是语义坐标系，不是普通字符串

Exact raw category paths 经 contract canonicalization 后形成 `CategoryNode` graph；用户可表达
范围由 `CategoryScope` 表示；商品归属由 `ProductCategoryAssignment` 表示。Session 以后只
保存一个已发布 `CategoryScope.id`，不保存 raw label，也不在 runtime 临时拼 category boolean
expression。

`IntentState.goal` 与 category 始终独立。Goal 是开放的购物任务，category 只是可靠 grounding
到 catalog taxonomy 的 anchor。

### 3.3 Applicability、SourceBinding 和 Capability 不互相代替

这三个概念分别回答：

```text
FacetApplicability
    这个 facet 在该商品类别是否有意义？

FacetSourceBinding
    哪个 raw source 能在这里提供证据？

EffectiveFacetCapability
    runtime 在这个 exact scope 能否 commit、retrieve、probe 或 clarify？
```

Runtime 不从其中一个推断另一个，也不做 parent-scope inheritance。

### 3.4 缺失不是负证据

商品事实使用四种状态：

```text
KNOWN | UNKNOWN | CONFLICT | NOT_APPLICABLE
```

Categorical value 进一步携带 `COMPLETE` 或 `PARTIAL`。PARTIAL 可以证明观察到的值存在，
但不能证明没有其他值。任何 matcher 和统计逻辑都必须保留这一点。

### 3.5 P0 只有一种 catalog truth

P0 的唯一 resolution policy 是：

```text
structured_resolution_v1
```

只有人工审核过的 structured binding 能进入 ProductFacetIndex。Hard filter、soft ranking、
Probe 和 clarification 消费同一份 resolved facts；它们的策略不同，但不能各自发明一套
商品事实。

### 3.6 自动化只能执行审核结果

Profiler 可以自动统计，builder 可以自动校验和物化，resolver 可以按冻结规则自动运行；
但 Gate A 和 Gate B 的语义决定不能由 coverage threshold 自动批准。

### 3.7 所有 build 都必须可重复、可拒绝

同一 catalog、reviewed config 和 `builder_version` 必须产生逐字节相同的 artifacts 和 release
ID。输入不完整、交叉引用错误、实现版本不支持或 hash 不一致时，build/load 必须 fail
closed，不能用 fallback identity normalizer 或运行时修复继续服务。

### 3.8 Gateway 只验证，不重新解释

Category change 造成旧 preference 不再适用时，Gateway 拒绝整个 batch；它不删除旧条件、
不把 apparel size 改成 luggage size，也不把 typo 自动修成 catalog value。语义 reconciliation
属于 Query Understanding。

## 4. 端到端实施路线

```text
Frozen 50k catalog
    |
    v
CS0      Raw profiler
    |
    v
CS1      Category Foundation
    |
    v
CS2      Gate A facet / applicability / binding
    |
    v
CS3      EvidenceStore / resolver / ProductFacetIndex / stats
    |
    v
CS4      Gate B capability
    |
    v
CS5      Runtime lexicon / registry / grounding service
    |
    v
CS6      CatalogSemanticRelease deterministic assembly
    |
    v
CS7      CatalogSemanticGateway + bound session store
    |
    +--------> CS8 downstream handoff:
               Query Understanding / Retrieval / Probe / Ranking / Asking
    |
    v
Thin official Agent adapter
```

每个阶段只消费前一阶段已经验证的 artifact。后续模块不能回头从 raw catalog 建立旁路语义。
在 CS6 完成前，文中以 contract 类型名描述的输出都只是 schema-valid build candidates，
不属于任何可供 runtime 使用的 `CatalogSemanticRelease`。

## 5. CS0：只读 catalog profiling

### 5.1 目标

在做任何语义决定前，建立一份绑定 exact catalog bytes 的可重复数据画像。

### 5.2 输入与输出

输入：

```text
data/catalog.jsonl
```

输出位于被 Git 忽略的 `artifacts/catalog-profile/`：

```text
bundle-manifest.json
profile.json
report.md
category-nodes.jsonl
product-category-assignments.jsonl
detail-keys.jsonl
category-detail-coverage.jsonl
```

### 5.3 方法

- catalog 只以 binary read 模式打开；
- 第一遍计算 exact byte SHA-256，第二遍 streaming profiling；
- 第二遍同时重新计算 hash，检测 profiling 期间的输入变化；
- duplicate JSON keys、非法 UTF-8、非有限数值和结构异常单独诊断；
- sample 由 catalog hash、seed、商品和 key 决定，稳定且有界；
- exact distinct-value 统计允许内存随 distinct raw values 增长，不虚称严格 constant memory；
- 所有 data files 先写 staging，manifest 最后发布；reader 先验证完整 bundle。

### 5.4 当前真实基线

以下基线只对这个 exact input 有效：

```text
profile schema = raw-catalog-profile-v1
catalog sha256 = da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67
file size      = 60,546,327 bytes
```

Profiler 结果为：

| 指标 | 结果 |
| --- | ---: |
| Physical/product rows | 50,000 |
| Unique `parent_asin` | 50,000 |
| Invalid records | 0 |
| Rows with diagnostics | 0 |
| Valid raw category assignments | 50,000 |
| Raw full-path prefix nodes | 1,832 |
| Exact raw details keys | 287 |
| Sparse category × key coverage rows | 24,868 |

两棵 exact raw root 的 subtree support 分别是：

```text
Clothing, Shoes & Jewelry             49,990
Shoe, Jewelry & Watch Accessories         10
```

与首轮 facet review 相关的 raw shapes 包括：

| Raw signal | Observed shape |
| --- | --- |
| top-level `price` | null 39,473；JSON number 10,410；string 117 |
| top-level `store` | string 49,686；null 314；不能因此称为 brand |
| `details` | object 50,000；其中 empty object 1,670 |
| exact `Department` key | support 43,582；175 raw distinct values |
| exact `Color` key | support 2,439；1,165 raw distinct values |
| exact `Brand` key | support 2,328；1,893 raw distinct values |
| exact `Brand Name` key | support 610；435 raw distinct values |
| exact `Material` key | support 2,069；463 raw distinct values |
| exact `Style` key | support 1,752；843 raw distinct values |
| exact `Size` key | support 925；338 raw distinct values |

这些数字是 raw data audit signals，不是 reviewed scope coverage、resolved `KNOWN`、facet 数量
或 runtime capability。全局 raw presence 低也不代表每个 category subtree 都低，因此 Gate A
必须查看 category-conditioned coverage，而不是只看全局比例。

### 5.5 完成标准

- 完整 bundle 通过 hash、size、schema 和交叉字段校验；
- 相同输入重复运行得到相同 bytes；
- profiler 不发布 CategoryScope、facet、binding、canonical runtime value；
- 原始 catalog 和 generated artifacts 均不进入 Git。

`validate_profile_bundle()` 通过只证明 raw observation bundle 自洽，不等于 catalog 已通过
release 的严格 50k schema gate，也不等于任何 semantic assignment 已经 `KNOWN`。

## 6. CS1：Category Foundation（原 M4A）

### 6.1 目标

把 raw category paths 变成可供后续所有 facet 工作引用的稳定类别坐标系。

### 6.2 输入

- exact 50k catalog；
- 已验证 raw profile bundle；
- contract 固定的 canonicalization 和 ID 规则；
- 当前 `builder_version` 的 closed category normalizer。

Profiler output 是 review 辅助数据，不是 semantic artifact source of truth；正式 builder 仍绑定并
验证 exact raw catalog。Profiler 中同名的 raw observation DTO、裸 path digest 和普通
`json.dumps` serializer 不能直接复用为 semantic CategoryNode、`cn_...` ID 或 RFC 8785 JCS
codec。

### 6.3 两遍实施流程

CategoryScope 不能在 canonical graph 之前被“预先审核”，因为 reviewer 需要引用这一版 graph
中真实存在的 `CategoryNode.id`。CS1 因此固定为两遍流程。

第一遍建立可供选择的类别坐标系：

1. 对每个 raw path segment 做 contract 规定的 lexical normalization。
2. 记录 raw path 到 canonical path 的映射和 lexical-collision audit；semantic node identity 仍
   完全由 contract normalizer 决定，reviewer 只能接受或拒绝 build，不能临时覆盖映射。不能
   假设 semantic graph node count 仍等于 raw profiler 的 1,832。
3. 建立完整 canonical prefix graph，生成 deterministic `CategoryNode.id`，并计算
   `category_graph_id`。
4. 验证 parent closure、唯一性和 root 集合，输出包含 canonical path、node ID、parent 和 raw
   provenance 的 graph proposal 与 collision report。
5. Reviewer 先接受这一版 graph，或拒绝 build 并触发 builder/contract change；只有接受后才可
   选择 scope roots。

第二遍把人工选择变成完整 scope：

6. Reviewer 在 source-controlled、versioned 的本地
   `shopping-copilot/category-scope-selection/v0` 输入中填写 `catalog_id`、
   `category_graph_id`、`builder_version`，以及每个 scope 的非空 `label` 和已排序、唯一的
   `root_node_ids`；其中必须有且只有一个选择覆盖全部 graph roots，作为 reviewed root scope。
   这是工程 review 输入，不是 contract artifact，也不能进入 release manifest。
7. Builder 校验选择文件精确绑定当前 graph，展开每个 root 的完整 subtree，确定性地产生
   `member_node_ids` 与 `CategoryScope.id`，并把覆盖整个 graph 的 reviewed scope 设为
   `root_scope_id`。
8. Builder 拒绝 redundant roots、unknown roots、equal-membership duplicate scopes 和 arbitrary
   boolean scopes；scope 之间允许重叠或形成 refinement。
9. Reviewer 检查 builder 生成的完整 `CategoryScope` tuple、closure、ID 和差异报告；只有完整
   materialized scope set 被确认后，才把它作为 source-controlled reviewed config fragment，
   并在 CS6 原样进入 `ReviewedSemanticConfig.category_scopes`。
10. Builder 为每个 `parent_asin` 生成 `KNOWN`、`UNKNOWN` 或 `CONFLICT` assignment；当前官方
    P0 build 额外要求 50,000 个 assignment 全部为 `KNOWN`。
11. 生成 CategoryRegistry 与 ProductCategoryAssignmentSet candidate artifacts，并验证重复 build
    的 bytes。

### 6.4 输出

```text
canonical graph proposal + collision report
source-controlled category-scope-selection/v0 review input
source-controlled reviewed CategoryScope config fragment
CategoryRegistry candidate
ProductCategoryAssignmentSet candidate
category build/audit report
```

这里的 selection 与 reviewed fragment 都是 release 前的工程输入/审核记录，不新增 contract
schema 或 manifest kind。完整 `ReviewedSemanticConfig` 要到所有 reviewed decisions 齐全时才
canonical assembly。这时可以生成并验证 category candidate artifacts，但完整 release artifact
set 尚未齐全，不能生成 `CatalogSemanticReleaseManifest`。

### 6.5 人工审核点

- 一个用户概念是否确实应该跨多个 raw subtrees 形成 union；
- 两个相似 raw labels 是否语义相同；
- normalization collision 是否暴露了不可接受的 lexical policy；若不可接受就停止 build 并走
  builder/contract change review，而不是手工拆改 canonical node；
- scope 是否太宽，导致后续 facet applicability 失去意义；
- 是否需要给真实查询场景补充一个稳定 scope，而不是 runtime 动态拼接。

Raw taxonomy 中存在 merchandising/navigation nodes，例如品牌活动、价格区间、new arrivals、
test color 或 size navigation。它们不能因为出现在树中就自动成为用户 CategoryScope，更不能
从 node label 反推出 product price、brand、color 或 size evidence。商品 raw path 的“terminal
node”也只是该商品路径的终点，不保证是整个 graph 中没有 children 的叶节点。

### 6.6 验收标准

- 所有 1,832 个 raw observed prefix nodes 都进入可审计的 raw-to-canonical mapping；semantic
  graph node count 由冻结 normalization 确定，collision review 只决定接受或拒绝该 build；
- `root_scope_id` 覆盖整个 graph，并匹配所有 50,000 个已知 assignment；
- scope membership 精确等于 roots 的 union-of-subtrees closure；
- category matcher 覆盖 `SATISFIED`、`VIOLATED` 和 `UNKNOWN`；
- 所有错误 scope fixtures 都 fail closed；
- 两次真实 catalog build 逐字节一致。

### 6.7 非目标

CS1 不定义 facet，不提取属性，不生成 ProductFacetIndex，不修改 session-context，也不决定
用户 utterance 如何映射到 scope。

## 7. CS2：Gate A extraction approval

### 7.1 目标

只批准那些能够由 reviewed structured source 稳定抽取的 facet。Gate A 回答的是“能否建立
可信 catalog fact”，不是“runtime 是否值得问这个问题”。

### 7.2 工作流程

对每一个 candidate facet：

1. 从 raw profile 找出 exact source keys、category-conditioned coverage、value shapes 和稳定样本。
2. 定义稳定、不可复用的 facet ID、display name、data type 和 item cardinality。
3. 如果不同 category 的 value domain、单位或语义不同，先拆成不同 facet ID。
4. 人工定义 `FacetApplicability`，明确语义有意义的 scopes。
5. 为每个 exact raw source key 定义独立 `FacetSourceBinding`。
6. 选择 closed `extractor_id`、`catalog_value_normalizer_id` 和 `resolver_id`。
7. 明确较小数字优先的 `priority` 与 categorical `completeness` 上限。
8. 用真实样本检查 false positive、false merge、单位问题和 category leakage。
9. 给出 `EXTRACTION_APPROVED` 或 `REJECT`，不能由阈值自动通过。

### 7.3 首批 candidate 的建议顺序

| Candidate | 为什么先看 | 必须先证明什么 |
| --- | --- | --- |
| `price` | source/type/policy 边界最清楚，适合作为第一条 vertical slice；不是因为 raw coverage 高 | decimal-to-cent 可精确解析；不做 rounding；117 个 raw strings 如何分类；无效值安全降级 |
| `store` / `Brand` / `Brand Name` | 三者高相关但存在真实不等价样本，应作为三个 source lane 审核 | 是否有任何 scope 下可安全绑定到同一 facet；禁止按名字或 overlap 自动 merge |
| `Department` | raw support 高，适合审计 semantic role | 它是否其实是 taxonomy/target audience，而不是一个应独立 commit 的 facet |
| `color` | 用户常表达，部分 structured details 有明确值 | raw key 变体、MULTI/组合色、COMPLETE/PARTIAL 语义 |
| material family | 用户常表达，正证据有价值 | `Material`、`Material Type`、outer/inner/fabric 等不能自动合并；检查 blend/multi-value 和负匹配安全性 |
| category-specific size facets | 价值高但语义风险最高 | 先拆 apparel/shoe/luggage 等 domain、单位和 normalization；不发布全局 `size` |

这只是 review 顺序，不是预先批准的 facet registry。真实样本不支持时，Gate A candidate
应被拒绝并留在研究记录中；未来用户仍可通过 Query Understanding 表达 semantic-only need，
但被 Gate A 拒绝的 facet 不会因此获得 `SEMANTIC_ONLY` capability row。

### 7.4 输出

```text
CatalogFacetSchema
FacetApplicabilitySet
FacetSourceBindingSet
source-controlled reviewed Gate A draft/input
Gate A review report
```

与 category scope draft 一样，这些 reviewed inputs 要到 release assembly 时才与 Gate B/runtime
decisions 一起组成完整 `ReviewedSemanticConfig`。

### 7.5 验收标准

- 每个 approved facet 恰有一个 applicability entry 且至少有一个 binding；
- binding 的 exact source key 在 raw catalog 中真实出现；
- binding scope 不超出 facet applicability；
- equal-priority overlap 不存在 extractor、normalizer、resolver 或 completeness 歧义；
- catalog normalizer 与未来 intent normalizer 没有混用；
- title、features、description、embedding 和 model inference 没有进入 P0 binding；
- rejected candidates 不会通过默认配置出现在下游 artifact。

### 7.6 自动化禁区

- lexical grouping 只能帮助 review，不能自动合并 exact raw keys；
- `store` 不能自动重命名为 `brand`；
- material-family keys 不能按名称自动 union；
- size 不能自动变成 global 或 numeric facet；
- raw missing 不能自动产生 negative 或 `NOT_APPLICABLE`；
- raw support/coverage 不能自动批准 Gate A 或 Gate B；
- raw top values 不能直接复制成 RuntimeValueLexicon；
- external taxonomy 或 LLM 只能给 reviewer 提建议，不能发布 P0 catalog facts。

## 8. CS3：Structured resolution 与 ProductFacetIndex

### 8.1 目标

把每条 reviewed binding 的输出变成可审计 evidence，再按唯一 P0 policy 生成 typed product
facts 和 Gate B statistics。

### 8.2 Evidence build

对 `(product, binding)`：

- 先确认 product category 能激活该 binding；
- 保存 canonical `raw_value_json`；
- 输出 `VALID`、`EMPTY` 或 `INVALID`；
- `VALID` value variant 必须匹配 facet data type/cardinality；
- evidence ID 绑定 product、facet、binding、status、raw value 和 canonical value；
- absence、EMPTY 和 INVALID 都不贡献 resolved truth，但保留各自不同的审计含义。

### 8.3 Resolver

对 `(product, facet)`：

1. 先由 assignment 和 FacetApplicability 判断 applicability。
2. `UNKNOWN`/`CONFLICT` category assignment 产生 product-facet `UNKNOWN`。
3. 已知不适用产生 `NOT_APPLICABLE`。
4. 过滤到 applicable bindings 和 policy-allowed VALID evidence。
5. 选择出现 valid evidence 的最小 priority 数字层。
6. 同层 compatible result 产生 `KNOWN`；incompatible result 产生 `CONFLICT`。对
   `priority_exact_v1` categorical evidence，先比较 canonical value tuple、暂不比较
   completeness；value 相同但 completeness 混合时，有至少一个 COMPLETE 就得到 COMPLETE，
   否则为 PARTIAL。不同 value tuples 才产生 conflict。
7. 低优先级只在所有更高层都没有 accepted valid evidence 时 fallback。
8. MULTI union 只能由显式 resolver ID 的冻结规则执行，generic resolver 不猜 union。

### 8.4 ProductFacetIndex 与 stats

ProductFacetIndex 可以稀疏存储 `KNOWN` 和 `CONFLICT`，但 lookup 必须结合 assignment 与
applicability 精确导出其他两种状态。Stats 从同一个 index 派生，不另用弱 evidence policy。

Category × facet stats 至少覆盖：

```text
scope product count
KNOWN / UNKNOWN / CONFLICT / NOT_APPLICABLE counts
complete known-value payload counts
```

### 8.5 验收标准

- evidence、binding、product、facet 的交叉引用完全一致；
- BOOLEAN/CATEGORICAL/NUMERIC/TEXT value variant 不能串型；
- lower-priority fallback、same-layer conflict 和 evidence support 均有正反测试；
- COMPLETE/PARTIAL 的正负匹配表全部覆盖；
- 某一个 binding 的 source absence、EMPTY 或 INVALID 本身不构成正证据或负证据；其他 binding
  或较低 priority fallback 仍可提供 accepted valid evidence，只有最终没有任何 accepted valid
  evidence 时才解析为 `UNKNOWN`；
- 50k build 可重复，stats count conservation 成立。

## 9. CS4–CS5：Gate B 与 runtime projection

### 9.1 目标

决定 resolved facet 在每个 exact CategoryScope 下是否适合被 runtime commit、retrieve、probe
或 clarify。高 coverage 不是自动批准；Gate B 同时审核冲突率、value distribution、样本和
业务价值。

### 9.2 CS4：决策与 capability

每个 reviewed `(facet, scope)` 发布一个最终决策：

```text
RUNTIME_ACCEPT | SEARCH_ONLY | SEMANTIC_ONLY | REJECT
```

以及 materialized capability booleans：

```text
intent_committable
retrieval_eligible
probe_eligible
clarification_eligible
```

Runtime 只做 exact lookup。Missing row 等价于全部 false，不从 parent、child 或 union scope
继承。与 FacetApplicability 完全 disjoint 的 scope 不能获得任何 true capability。

### 9.3 CS5：RuntimeValueLexicon

- categorical/boolean P0 domain 是 global-per-facet、CLOSED；
- canonical value 必须在该 facet applicability 的某处有 `KNOWN` catalog support；
- 某个 scope 当前没有该值库存，不会让这个全局 canonical value 失效；
- alias 只能映射到已有且有 catalog support 的 canonical value，不能引入新值；
- typo、fuzzy match、embedding 或 LLM 不能直接产生 structured value；
- `catalog_value_normalizer_id` 与 `intent_value_normalizer_id` 保持独立；
- runtime numeric 只允许安全整数 `USD_CENT` 的 `price`。

### 9.4 CS5：Runtime registry projection

只要一个普通 facet 在至少一个 exact scope 为 `intent_committable` 或 `probe_eligible`，它就
需要投影到 session `FacetRegistry`。这样 Probe-only facet 可以合法出现在 SearchBelief 中；
Gateway 仍会拒绝对非 committable scope 的 preference 写入。

`system_product_category` 作为 reserved adapter facet 单独注入，不参加普通 Gate A/B、facet
stats 或 entropy。

### 9.5 CS5：Deterministic grounding service

Grounding 是 catalog-semantic runtime projection 的一部分，不由 Query Understanding 重新实现。
CS5 提供 release-bound 的纯函数/service：普通 facet 消费 extracted candidate、turn 的 final
proposed CategoryScope、verified release 与 exact capability row；reserved category candidate 则
消费 CategoryRegistry，不查普通 capability 或 lexicon row。

该 service 必须：

- 只调用 pinned `intent_value_normalizer_id` 的 closed implementation，执行 exact canonical/alias
  lookup、type/operator validation、exact-scope capability check 与 canonical ordering；
- 对 categorical/boolean 只返回 lexicon 中已有 catalog-supported canonical value，对 `price` 只
  返回安全整数 `USD_CENT`，对 category 只返回 published CategoryScope ID；
- 严格产生 contract 的 `RuntimeValueGroundingResult`，disposition 只有 `GROUNDED`、
  `SEMANTIC_ONLY` 或 `AMBIGUOUS`；
- 对 numeric equality 确定性地产生 inclusive `GE`、`LE`，对 ambiguity 只给同一 facet 内的
  release-valid、canonical-order candidates；
- 不做 facet-language parsing、fuzzy/embedding/LLM repair，也不创建或分配 Preference ID。

这条边界允许未来更换 Query Understanding 技术，同时保证相同 candidate 与相同 release 始终
得到相同 grounding 结果。

### 9.6 验收标准

- CS4 的 capability implication 和 disjoint-applicability 规则全部验证；
- RuntimeFacetDomain、RuntimeFacetSpecRecord 和 reviewed runtime config 的 facet、kind mapping
  与 intent normalizer 一致；
- aliases 与 canonical namespace 无歧义；
- unknown/typo value 不能 ground；
- TEXT 和非 price numeric facet 不会错误投影；
- Probe-only facet 能通过 registry 与 SearchBelief shape validation；
- 普通 facet、reserved category、unsupported operator、non-committable scope、unknown value、
  ambiguity 与 numeric equality 都有 `RuntimeValueGroundingResult` golden tests；
- grounding tests 证明 service 不分配 Preference ID，输出不依赖 candidate discovery order。

## 10. CS6：CatalogSemanticRelease assembly

### 10.1 目标

把同一次 catalog、review、policy 和 builder 产生的 artifacts 绑定成一个不可混搭的 release。

### 10.2 Release artifact 集合

完整 P0 manifest 精确引用 13 类 content-addressed artifact：

```text
catalog
category_registry
product_category_assignment
facet_schema
facet_applicability
facet_source_bindings
facet_evidence_store
product_facet_index
facet_stats
effective_capabilities
runtime_value_lexicon
runtime_registry
reviewed_config
```

Manifest 另外固定：

```text
resolution_policy_id = structured_resolution_v1
builder_version
```

`builder_version` 是当前 P0 信任模型中的不可复用逻辑实现版本，不是额外的代码 hash
artifact。任何 extractor、normalizer、resolver、matcher 或 projection 行为改变，都必须使用
新 builder version。

### 10.3 Build 与 load 流程

1. 读取并保留 raw catalog 的 exact source bytes，校验 format 与 exact catalog hash；绝不把 raw
   JSONL parse 后用 JCS 或其他 serializer 重写。
2. 校验 reviewed config 与所有 generated projections exact equality。
3. 对 raw catalog 之后的 semantic artifacts，在同一 generation staging 区使用 contract 的
   canonical JSON 和 canonical collection order 写入；不在目标 bundle 中逐个暴露半成品。
4. 计算每个 artifact 的 exact bytes hash 和 size。
5. 最后发布 canonical release manifest，并计算 release ID；reader 只接受 manifest 完整验证的
   generation。
6. Loader 重新验证所有 hashes、schemas、cross-references、builder registries 和 runtime
   invariants。
7. 任何 mismatch 在 session reset/decode 前 fail closed。

### 10.4 验收标准

- 相同 exact raw catalog bytes、reviewed config 和 builder version 产生逐字节相同的 generated
  semantic artifacts；manifest 绑定完全相同的 13 个 `ArtifactRef`、manifest bytes 和 release ID；
- 改变任一 artifact byte、reviewed config 或 builder version 会改变或使 release 无效；
- 不完整 candidate build 不能被命名为完整 release；
- 每个 `CatalogBoundSessionStore` 只绑定一个 verified release；同一 process 如需并存不同
  release，必须使用彼此隔离的 wrapper stores，不能共享同一个 existing `InMemorySessionStore`；
- 不实现 hot migration 或 alias 驱动的 silent session upgrade。

## 11. CS7：Gateway 与 session store 集成

### 11.1 目标

让 catalog-domain legality 与 session-context 的机械合法性同时成立，并且不存在“先验证、
后绕过 commit”的时间窗口。

### 11.2 Reserved category

Category 在 IntentState 中只使用：

```text
facet    = system_product_category
operator = EQ
value    = one published CategoryScope.id
```

在显式 targeting `system_product_category` 的 operations 中，Gateway 只接受 `ReplaceFacet`。
普通 `AddPreference`、`RemovePreference`、`ClearFacet`、`SetDontCare`、negative/IN operators、
普通 asking 和 ordinary facet stats 全部拒绝。`SwitchGoal` 不是 facet-targeting operation；它只能
原样 carry 已存在且已验证的 exact-shape category Preference，或 drop 它，不能引入、替换或修改
category。省略 carry 后使用 root effective context，除非紧随其后的是合法 reserved
`ReplaceFacet`。

### 11.3 Atomic authority

`CatalogBoundSessionStore` 私有持有 verified release、projected registry 和底层
`InMemorySessionStore`。Catalog-bound transaction 在同一 per-session lock 内：

1. 捕获 previous context；
2. 从 appended TurnRecord 取得 exact accepted batch；
3. 用 Gateway 调用 unchanged reducer 计算并验证 final IntentState；
4. 验证 category、capability、lexicon 和当前 SearchBelief；
5. 验证 private one-use Probe token；
6. 调用 private raw transaction commit；
7. 成功或失败后释放 lock。

业务代码拿不到 raw store commit handle。Standalone gateway preview 可以辅助规划，但不携带
commit authority，最终检查必须在 locked write 内重新执行。

### 11.4 Category change

Gateway 使用 batch 的 final proposed category 验证所有普通 operations。旧 structured
preference 或 don't-care facet 在新 category 下失效时：

```text
reject whole batch
code = INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE
```

Query Understanding 必须在同一 batch 明确 remove、replace 或 semantic fallback。

### 11.5 Session envelope

外层 envelope 固定 session ID 和 semantic release ID，严格验证 canonical JSON、base64url、
inner hash 和 inner canonical re-encoding。Decode 后用 Gateway replay accepted batches；只验证
当前 SearchBelief，不声称重建已经不存在的历史 belief。

### 11.6 验收标准

- raw reducer 可接受但 Gateway 应拒绝的 operation 有专门反例测试；
- category operation matrix 和 final-state checks 全覆盖；
- gateway validation 与 raw commit 在一个 lock 内，无旁路 handle；
- caller-constructed belief 与错误 Probe token 均无法进入 live state；
- category change rejection 不产生自动 repair 或 partial commit；
- release mismatch、envelope tamper 和 replay mismatch 全部 fail closed；
- 现有 session-context v1 tests 保持不变并继续通过。

## 12. CS8 downstream handoff：Query Understanding

Query Understanding 在完整 semantic release 之后实现。它负责把用户表达变成候选语义，
而不是读 raw catalog、重新定义 facet 或实现第二套 grounding。

### 12.1 正常路径

```text
utterance
    -> semantic parse / candidate facet and operator
    -> call CS5 release-bound exact grounding service
    -> RuntimeValueGroundingResult
    -> trusted Preference construction and ID assignment
    -> StateUpdateBatch
    -> CatalogSemanticGateway
```

Grounding result 只有三种 disposition：

```text
GROUNDED | SEMANTIC_ONLY | AMBIGUOUS
```

- `GROUNDED` 只能使用同一个 recognized facet 的 release-valid canonical values；
- `SEMANTIC_ONLY` 保留 unsupported need，不伪造 facet/value；
- `AMBIGUOUS` 给出同一 facet 内的 release-valid candidates，等待消歧；
- numeric equality 展开成 inclusive `GE` 后 `LE`；
- Preference ID 只在 successful grounding 后分配；
- composite phrase 不复制到两个不等价 atomic predicates 上。

### 12.2 明确边界

Query Understanding 可以使用 model、规则或其他解析技术产生 extracted candidates；随后必须
调用 CS5 已发布的 deterministic grounding service。它不复制 intent normalizer、lexicon、
capability 或 CategoryRegistry 规则。只有 `GROUNDED` result 才能交给 trusted coordinator 分配
Preference ID 并构造 structured update，最终所有输出仍必须经过 Gateway。模型不能直接创建
CategoryScope、canonical value、ProductFacetIndex row 或绕过 release 的 structured preference。

## 13. CS8 downstream handoff：Retrieval、Probe、Ranking 与 Asking

这些模块消费 semantic release，但不能修改它。

### 13.1 Retrieval 与 matcher

- category 使用 assignment-scope matcher；
- categorical/boolean 使用 COMPLETE/PARTIAL 四值表；
- numeric matcher 在同一 `(facet, Commitment)` 内组合 bounds；
- HARD 和 SOFT 不合并成一个 numeric interval；
- matcher 返回四值结果，retrieval policy 再决定 recall/filter/rank 行为；
- `UNKNOWN` 和 `NOT_APPLICABLE` 不会被底层 matcher 偷换成 violation。

### 13.2 Probe 与 SearchBelief

- 只对 active exact scope 中 `probe_eligible=True` 的 facet 生成 stats；
- 使用同一 `structured_resolution_v1` ProductFacetIndex；
- top values 必须通过 runtime lexicon/normalizer；
- belief 由 release-bound private producer 生成，并通过 one-use token commit；
- certainty、entropy 和 question utility 是消费者策略，不回写 catalog facts。

### 13.3 Asking 与 official adapter

Internal facet ID 与官方 `ask_attribute` vocabulary 是两个协议。映射发生在 thin adapter，
不能为了适配 toy simulator 把 internal facet registry 限制成官方字段列表，也不能把 baseline
BM25 的检索接口当成系统架构。

## 14. 人工 review 协议

### 14.1 Gate A review packet

每个 facet candidate 的审核材料至少包含：

```text
proposed facet ID and meaning
data type and cardinality
applicable CategoryScopes
exact raw source keys and SourceKind
category-conditioned presence/non-empty coverage
value type/distribution and stable examples
extractor and catalog normalizer IDs
priority and completeness
resolver ID
known ambiguity / false-positive examples
approval or rejection rationale
```

Reviewer 批准的是一个可执行 binding contract，不是一个听起来合理的名字。

### 14.2 Gate B review packet

每个 `(facet, scope)` 的审核材料至少包含：

```text
resolved KNOWN / UNKNOWN / CONFLICT / NOT_APPLICABLE counts
coverage and conflict rate
canonical value distribution
representative evidence chains
negative-matching safety for completeness
retrieval and user-expression value
proposed runtime decision and four capability booleans
proposed intent normalizer and reviewed aliases
```

Reviewer 必须能选择 SEARCH_ONLY、SEMANTIC_ONLY 或 REJECT；不能因为工程已经实现 extractor 就
默认 RUNTIME_ACCEPT。

### 14.3 Review 变更流程

1. 修改 source-controlled reviewed config。
2. 运行 deterministic builder。
3. 先看 validation result 和人类可读 diff report。
4. 检查 artifact IDs、coverage/conflict/value changes。
5. 审核通过后才组装新 release。
6. 旧 session 不自动迁移；新旧 release 分开路由或停止支持旧 release。

Generated artifact 不能手工编辑；任何修正都回到 raw parser、closed implementation 或 reviewed
config。

## 15. Repository 组织与卫生

推荐按真实实现逐步增加目录，不提前创建空模块：

```text
docs/design/catalog_semantic/
    README.md
    methodology-v0.md
    contract-v0.md

config/catalog_semantic/v0/           reviewed inputs，进入 Git

src/shopping_copilot/catalog/
    profiling/                         已实现，只读 raw observation
    semantic/                          CS1 起按职责增加真实模块

tests/unit/catalog/
    ...                                profiler 与 semantic unit tests

tests/integration/
    ...                                release/gateway/adapter tests

artifacts/catalog-profile/             generated，忽略
artifacts/catalog-semantic/            generated，忽略
```

| 内容 | 是否进入 Git | 原因 |
| --- | --- | --- |
| Contract / methodology / review rationale | 是 | 设计真源与决策历史 |
| Reviewed semantic config | 是 | 可审计的人类决定 |
| Builder、validator、resolver 实现 | 是 | 可测试行为 |
| Small deterministic test fixtures / goldens | 是 | 回归验证 |
| 50k raw catalog | 否 | 大型本地数据与分发边界 |
| Generated profiles / indices / releases | 否，除非以后明确制定发布流程 | 可由 pinned inputs 重建 |
| API keys、private eval data、model cache | 永不 | 安全和比赛规则 |

不要把 semantic logic 放进 `starter/agent.py`、evaluator、retriever 或 session-context reducer。
这些地方只消费已发布接口。

## 16. 测试方法

### 16.1 Contract unit tests

每条 normative invariant 至少有一个正例和一个最小反例。优先测试边界而不是只测试 happy
path：duplicate IDs、wrong type variant、redundant scope roots、unknown scope roots、
equal-membership duplicate scopes、unknown refs、bad ordering、hash mismatch、priority conflict、
partial completeness 和 release mismatch。Scope 之间的 overlap/refinement 是合法能力，必须有
正例覆盖，不能作为通用拒绝条件。

### 16.2 Golden fixtures

用小型人工 catalog 覆盖：

- 多 root category graph；
- union scope；
- unknown/conflict assignment；
- applicable 但 source missing；
- complete/partial categorical values；
- same-priority agreement/conflict；
- lower-priority fallback；
- Probe-only facet；
- category change rejection。

Golden bytes 用于证明 canonical serialization，而不是冻结不重要的 report prose。

### 16.3 Real-catalog acceptance

每个阶段都必须在真实 50k catalog 上运行，并报告：

```text
input catalog hash
builder version and reviewed config hash
row/product counts
diagnostic and rejection counts
artifact hashes and sizes
category/facet status distributions
elapsed time and peak-memory observation
```

真实运行通过不等于语义 Gate 自动通过；它只能证明实现能处理完整数据。

### 16.4 Reproducibility tests

- 同一环境重复 build；
- 支持的 Python 版本重复 build；
- 不同遍历顺序的 fixture build；
- staged publication interruption；
- artifact byte tampering；
- manifest cross-reference substitution。

### 16.5 Integration regression

- session-context 原有 reducer、codec 和 store tests 不得弱化；
- Gateway tests 证明 raw reducer 的合法性不等于 catalog-domain 合法性；
- release-bound snapshot replay 保持 exact final IntentState；
- official evaluator 最终只作为 API compatibility regression，不作为 semantic architecture 定义。

## 17. 失败语义速查

| 情况 | 正确结果 | 禁止行为 |
| --- | --- | --- |
| `KNOWN` category assignment + facet applicable，但所有 applicable bindings 均无 accepted valid evidence | `UNKNOWN` | 把 source 缺失当成不满足用户条件 |
| `KNOWN` assignment 与 facet applicability disjoint | `NOT_APPLICABLE` | 与 UNKNOWN 混合 |
| Category assignment 为 `UNKNOWN` 或 `CONFLICT` | product-facet `UNKNOWN` | 在类别未确定时声称不适用 |
| 同 priority valid evidence 经 pinned resolver 判定 incompatible | `CONFLICT`；`priority_exact_v1` 下不同 canonical value tuples 才 conflict | 按 binding ID 随机选一个；忽略 reviewed custom MULTI resolver |
| 高 priority 无 valid evidence | 尝试较低 priority | 用低 priority 覆盖已有高层 valid evidence |
| PARTIAL 未观察到查询值 | `UNKNOWN` | 证明负命题 |
| 未注册 CategoryScope | `SEMANTIC_ONLY`（`unregistered_category_scope`） | runtime 动态构造 scope；把 clarification 策略冒充 grounding disposition |
| 未知或 typo value | `SEMANTIC_ONLY`（`unknown_value`） | fuzzy 自动写 canonical value |
| 同一 facet 内有多个 release-valid candidates | `AMBIGUOUS` | 随机选择或提交多个互相竞争的值 |
| Category change 使旧条件失效 | 拒绝完整 batch | Gateway 自动删除或重解释 |
| Probe 来源不可证明 | 不允许 live commit | 只凭 DTO shape 接受 caller belief |
| Release 或 artifact 不匹配 | fail closed | 混合加载“看起来兼容”的 artifacts |

## 18. 版本与变更治理

### 18.1 不需要改 contract 的变化

- report 排版；
- profiler 性能优化但输出语义和 bytes 不变；
- 新的 review 辅助视图；
- retrieval/ranking/asking 策略调整，只要不改变 semantic artifacts 或 session contract；
- 本文的解释、顺序说明和工程建议改进。

### 18.2 需要新 reviewed config / release 的变化

- 新增、删除或调整 CategoryScope roots；
- facet approval、applicability、binding、priority 或 completeness 变化；
- Gate B capability 或 aliases 变化；
- catalog bytes 变化；
- builder version 变化。

### 18.3 需要 versioned contract review 的变化

- 改变 ID/hash preimage；
- 改变 CategoryScope 表达能力；
- 增加新的 evidence policy 或弱文本/model evidence；
- 改变四值 matching/completeness 语义；
- 增加 OPEN/HYBRID value domain；
- 增加 runtime numeric facets 或单位协议；
- 增加 capability inheritance；
- 改变 Gateway authority、release pinning 或 session envelope。

### 18.4 P0 当前明确延期

- title、features、description、embedding 或 model-inferred product facts；
- hard、soft、Probe、clarification 各自使用不同 catalog truth policy；
- runtime capability inheritance；
- OPEN/HYBRID runtime value domain；
- `price` 以外的 runtime numeric facets；
- dynamic CategoryScope construction；
- 自动生成 query-language synonyms 或 fuzzy typo repair；
- session 跨 release silent migration；
- 在同一个 `CatalogBoundSessionStore` 或其 existing `InMemorySessionStore` 内混用多个 release；
- 为适配 baseline retriever 或 toy simulator 而改变 semantic architecture。

已发布 ID 不能被复用于不同含义。Display label 可以变，但 artifact/release content hash 会忠实
反映变化。

## 19. 阶段准入顺序

实时完成状态见 [README](README.md)。无论当前进度如何，阶段准入关系保持如下：

| 阶段 | 只有在什么条件下开始 | 此阶段仍然禁止什么 |
| --- | --- | --- |
| CS0 Raw profiler | frozen raw catalog 可只读访问 | 发布 facet/category 语义 |
| CS1 Category Foundation | raw profile bundle 已验证，contract 已冻结 | facet、resolver、runtime 和 session integration |
| CS2 Gate A | category candidates 通过 CS1 两遍审核与验收 | runtime promotion、weak/model evidence |
| CS3 Resolution / Index | Gate A config fragments 与 closed implementation IDs 已审核 | Query Understanding 和 capability 自动晋升 |
| CS4 Gate B | 真实 resolved stats 与 evidence examples 可审查 | capability inheritance、fuzzy grounding、非 price runtime numeric |
| CS5 Runtime / Grounding | Gate B decisions 已审核且 runtime projections 可生成 | Query parsing、Preference ID 分配、fuzzy repair |
| CS6 Release | 13 类 ArtifactRef inputs 与完整 reviewed decisions 齐全 | 不完整 candidate 冒充 release、hot migration |
| CS7 Gateway | verified release 可完整 load | transaction 外验证后 raw commit、自动 category repair |
| CS8 Downstream handoff | verified release、Gateway 和 CS5 grounding service 可调用 | 直接创建 canonical catalog truth，或从 raw catalog 建第二套 facet semantics |

不要为了并行推进在 retriever、Query Understanding 或 official adapter 中预埋临时
facet/category 逻辑；它们会形成未来无法审计的第二套语义系统。

## 20. Catalog Semantic P0 Definition of Done

Catalog Semantic P0 只有在以下条件全部成立时才算完成：

1. 50k catalog 通过全部 category build validation，且官方 P0 assignments 全部 `KNOWN`；
2. 所有 published scopes、facets、bindings 和 capabilities 都有 source-controlled review；
3. 每个 `KNOWN`/`CONFLICT` resolved claim 都能追溯到 reviewed structured evidence IDs；
   `UNKNOWN`/`NOT_APPLICABLE` 能追溯到 category assignment、FacetApplicability 与 sparse
   resolution path，并允许没有 evidence ID；
4. UNKNOWN、CONFLICT、NOT_APPLICABLE 和 completeness 在 matcher 中保持安全语义；
5. 每个 categorical/boolean canonical value 都有 `KNOWN` catalog support；reviewed alias 只能
   指向已有且受支持的 canonical value，不能引入新值；
6. 13 类 artifacts 和 release manifest 可重复、content-addressed、完整验证；
7. Session state 的所有 catalog-sensitive write 都经过 atomic Gateway boundary；
8. SearchBelief 只能由同 release 的 private Probe producer 写入 live state；
9. 原有 session-context v1 contract 和测试保持不变；
10. official adapter 仍然薄，toy simulator 只承担兼容性回归；
11. generated artifacts、catalog、credentials 和 private evaluation data 不进入 Git；
12. 任何 unsupported user need 都能安全保留为 semantic-only，而不是被静默丢弃或错误
    grounding。

完成这些条件之后，团队才拥有一套可以被 Query Understanding、retrieval、Probe、ranking 和
asking 共同消费、同时不会互相篡改语义的 catalog foundation。
