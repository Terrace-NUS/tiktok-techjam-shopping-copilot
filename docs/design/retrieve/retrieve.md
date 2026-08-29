## 结论

我会把召回与排名整体重构成一个 **Catalog-Grounded Active Preference Search** 系统，而不是“Buying 用过滤器、Browsing 用向量检索”的双引擎。

核心定义是：

> **用户意图不是一个等待分类的标签，而是一个需要结合语言反馈与商品空间证据持续更新的 belief state。召回负责搜索，也负责感知；排名负责排序，也负责决定下一步是否值得打扰用户。**

可以把项目命名为：

# **BeliefSearch**

### Retrieve to understand. Rank to decide. Ask only when a turn is worth it.

这比“Hybrid Retrieval + LLM Reranking”高一层，也比我们之前单独强调“需求明确度”再前进一步：**明确度不应该只有一个标量，而应该是一个可诊断的不确定性向量。**

---

# 一、先指出正式题面里最值得利用的张力

正式题面一方面要求：

* 立即识别 Buying / Browsing；
* Buying 进入精确过滤路线；
* Browsing 进入高多样性语义召回路线。

但另一方面又要求：

* Information Accumulation；
* Intent Override；
* Over-Generality detection；
* Personalized Context Distillation；
* Runtime Workflow Re-orchestration；
* 用 MTTC 惩罚无价值对话。

这两组要求实际上存在张力。

“立即识别底层意图”隐含的是：

```text
用户已经属于某个类型
→ 系统把它分类出来
→ 路由到对应搜索器
```

而 Information Accumulation、Intent Override 和 Adaptive Orchestration 隐含的是：

```text
用户意图仍在形成
→ 每轮都有可能改变
→ 搜索结果也会改变我们对意图的判断
→ workflow 必须动态调整
```

所以最强的题目解读不是把双轨做得更准确，而是：

> **Buying / Browsing 不是两个 latent class，而是探索—收缩控制空间中的两个区域。**

更进一步，官方 evaluator 把任务表述为在十轮内找出一个隐藏目标 ASIN，并在目标首次进入 Top 10 时结束。这个设计适合自动评分，但它把购物过程简化成了“恢复一个用户早已知道、只是尚未透露的答案”。([GitHub][1])

因此我的判断是：

* **架构上不要让 toy simulator 定义系统本体；**
* **工程上仍然要保留 evaluator adapter，否则会直接损失进入 Final 的概率。**

也就是：

```text
Realistic Core Policy
        ↓
Official Evaluator Adapter
```

而不是：

```text
Simulator Rules
        ↓
Whole System Architecture
```

---

# 二、不要只维护“需求明确度”，要维护多假设 Belief State

我们之前的 \(C_t\) 仍然有价值，但如果只维护一个 0–1 的 certainty，还是不够。

因为以下几种情况都可能表现为“低确定度”，但需要完全不同的动作：

| 不确定性来源                   | 正确动作           |
| ------------------------ | -------------- |
| 用户语言含糊                   | 可能追问           |
| 语言明确，但 BM25 与 Dense 严重分歧 | 静默改写或调整检索路线    |
| 条件明确，但库存中没有商品满足          | 协商放松约束         |
| 候选很多，但都属于同一需求方向          | 可以直接推荐，不一定要问   |
| 候选分成几个清晰流派               | 展示代表项或提出高信息量问题 |
| Top 结果已明显领先              | 直接推荐，避免多余交互    |

所以建议状态定义为：

$$
B_t=\{H_t,\ C_t,\ U_t,\ P_t,\ V_t\}
$$

其中：

### 1. \(H_t\)：Intent Hypotheses

不是只生成一个 rewritten query，而是维护多个可能方向：

```text
H1: smart-casual loafers for commuting       p=0.45
H2: minimalist walking shoes for office      p=0.30
H3: lightweight ankle boots                  p=0.25
```

### 2. \(C_t\)：Typed Constraints

每个条件都必须包含类型、置信度、来源和版本：

```json
{
  "attribute": "budget",
  "operator": "<=",
  "value": 100,
  "strength": "hard",
  "confidence": 0.98,
  "source_turn": 3,
  "status": "active"
}
```

条件至少区分：

* hard constraint；
* soft preference；
* negative preference；
* don’t-care；
* inferred preference；
* stale / overridden constraint。

### 3. \(U_t\)：Uncertainty Vector

我建议至少保留：

$$
U_t=
[
U_{\text{language}},
U_{\text{catalog}},
U_{\text{route}},
U_{\text{feasibility}},
U_{\text{decision}}
]
$$

分别对应：

* 语言歧义；
* 候选空间分散度；
* 多条召回路线分歧；
* 条件在库存中的可满足性；
* Top 候选之间的决策差距。

### 4. \(P_t\)：Profile Prior

长期偏好只能作为 prior，不能覆盖本轮显式表达。

例如长期偏好黑色，但用户本轮说“这次想看看棕色”，必须以 session state 为准。

### 5. \(V_t\)：State Version / Override Ledger

Intent Override 不应通过简单覆盖 JSON 实现，而应该记录事件：

```text
Turn 2: category = loafer
Turn 4: user says "actually show me boots"
Turn 4: tombstone(category=loafer)
Turn 4: category = boots
```

这样才能验证旧约束有没有错误残留。

一个 scalar certainty 可以保留给 dashboard 和粗粒度 routing，但不能作为全部控制依据。

---

# 三、召回不只是 Candidate Generator，而是环境传感器

完整架构建议如下：

```text
User Dialogue
      │
      ▼
Session Belief State
      │
      ▼
Cheap Retrieval Probe
      │
      ├── candidate snapshot
      ├── route disagreement
      ├── category / hypothesis entropy
      ├── constraint feasibility
      └── discriminative facets
      │
      ▼
Policy Controller
      │
      ├── direct retrieve
      ├── silent rewrite
      ├── broaden
      ├── relax constraint
      └── ask clarification
      │
      ▼
Multi-Hypothesis Multi-Route Retrieval
      │
      ▼
Hard Constraint Gate
      │
      ▼
Route-Aware Unified Ranker
      │
      ├── conversion slate
      └── diagnostic slate
      │
      ▼
Show / Ask / Rewrite / Relax
      │
      └──────────────► State Update
```

这个架构里最关键的变化是：

> **召回结果不是排名器的单向输入，而是 Controller 观察商品世界的 observation。**

JD.com 2026 年的 EASP 已经证明了这种思路具备工业可行性：它先执行轻量 Retrieval Probe，再让 Planner 根据真实结果诊断 entity drift、attribute misalignment 等问题；线上 A/B 报告 UCVR 提升 0.89%、GMV 提升 0.57%，并通过复杂度路由让约 80% 的简单查询直接走 fast path。([arXiv][2])

---

# 四、召回组件具体怎么做

## 4.1 不要做两个搜索器，要做始终存在的多路召回

我建议至少五条路线：

| Route                      | 主要作用                          | 什么时候提高预算            | 主要风险                    |
| -------------------------- | ----------------------------- | ------------------- | ----------------------- |
| Lexical / BM25             | 型号、品牌、材质、精确术语                 | 明确度高、实体词多           | 词汇鸿沟、场景意图弱              |
| Structured / Category      | 类别、价格、颜色、尺码等约束                | hard constraints 增加 | 元数据缺失造成误杀               |
| Dense Semantic             | 场景、风格、功能、隐式语义                 | 查询开放、描述自然语言化        | 语义漂移、违反硬条件              |
| Intent / Facet Route       | occasion、style、use case 等商品意图 | 用户表达高层需求            | 标签生成偏差                  |
| Related-Intent Exploration | 替代品、相邻类别、主题匹配                 | 明确度低且用户在探索          | Recall 上升但 precision 下降 |

这些路线永远共存，变化的是：

* 每条路线的 \(K\)；
* 每个 hypothesis 的召回配额；
* filter 的严格程度；
* 是否允许跨类别扩展；
* 是否启动相关意图路线。

例如：

```text
低确定度：
Dense 80 + Intent 60 + BM25 30 + Structured 20 + Related 30

高确定度：
Structured 80 + BM25 70 + Dense 30 + Intent 20 + Related 0
```

这才是把 Buying / Browsing 解释成连续控制，而不是换名字做二分类。

---

## 4.2 召回应该围绕多个需求假设展开

假设当前 belief 是：

```text
H1: smart-casual loafers       0.50
H2: understated walking shoes  0.30
H3: ankle boots                0.20
```

不能只让 LLM 合成一个平均化 query：

```text
comfortable smart casual walking loafer boots
```

这会导致 semantic soup。

正确做法是：

```text
H1 × lexical / dense / structured
H2 × lexical / dense / structured
H3 × lexical / dense / structured
```

并设置最小 hypothesis quota，防止最高概率方向过早吞掉其他合理方向。

可以形式化为：

$$
K_{r,h}
=
K\cdot
\operatorname{softmax}
\left(
g(r,h,B_t,O_t)
\right)
$$

其中：

* \(r\) 是 retrieval route；
* \(h\) 是 intent hypothesis；
* \(B_t\) 是当前状态；
* \(O_t\) 是 probe observation。

Hackathon 阶段不需要训练神经 router，使用可解释的线性规则或小型 GBDT 即可。

---

## 4.3 商品侧必须做 Intent Card，而不是只 embed 标题

自然语言 query 可能是：

> comfortable elegant shoes for everyday office use

而商品标题通常只写：

> Women’s Leather Slip-On Loafer, Brown

如果只做 query-title embedding，模型需要凭一个标题恢复：

* office use；
* smart casual；
* comfortable walking；
* not overly formal。

建议离线为商品构建：

```json
{
  "product_type": ["loafer"],
  "occasion": ["office", "commuting", "smart casual"],
  "style": ["minimal", "understated"],
  "function": ["walking comfort", "slip-on"],
  "material": ["leather"],
  "weather": [],
  "audience": ["women"],
  "negative_traits": ["not athletic"],
  "evidence": {
    "material": "Leather",
    "function": "Cushioned insole"
  },
  "confidence": {
    "occasion": 0.78,
    "function": 0.91
  }
}
```

关键不是“LLM 生成更多文本”，而是：

1. Query 和 Product 使用相同的 intent schema；
2. 每个推断属性保留 evidence；
3. 低置信度推断只能作为 soft matching；
4. 不能把生成的 use case 当作 hard filter。

Walmart 的 INSPIRE 采用的就是 query-side 与 product-side 对称的结构化 intent augmentation，并在离线实验中报告 Precision@1 提升 4.2%、NDCG@10 提升 2.64%，同时 embarrassing result 减少一半。不过该论文明确说明在线 A/B 仍在计划中，而且固定 schema、teacher bias 与持续维护成本都是局限。([arXiv][3])

所以它适合成为设计依据，但不能把其“工业采用度”讲成已经完全验证。

---

## 4.4 初步融合用 RRF，最终不能停在 RRF

不同路线的原始 score 不可直接比较：

* BM25 的 12.4；
* cosine similarity 的 0.76；
* category match 的 1；
* generated-intent match 的 0.83；

不能直接做：

```python
0.4 * bm25 + 0.4 * dense + 0.2 * filter
```

第一阶段可以使用 rank-based RRF：

$$
RRF(i)=\sum_r\frac{w_r}{k+\operatorname{rank}_r(i)}
$$

但必须保留：

```text
route id
route rank
route score
hypothesis id
query variant
matched constraints
violated constraints
evidence spans
```

然后交给统一 ranker。

Target 2026 的生产方案直接指出，固定 RRF 或 weighted interleaving 无法学习 query-specific channel utility 和跨渠道交互；其 route-aware LTR 将各渠道分数和来源当作特征，在 Target.com 报告 conversion 提升 2.85%，p95 延迟低于 50ms。([arXiv][4])

但这里必须注意数据规模差异：

* Target 使用约 6000 万训练行、50 万查询；
* 你们只有 200 个公开 session。

所以不能照搬其训练方案。对你们更现实的是：

```text
RRF candidate union
        ↓
pretrained semantic reranker
        ↓
small regularized linear / GBDT fusion
```

而不是从 200 个 session 训练一个高容量 LambdaMART。

---

# 五、Probe 必须输出“搜索诊断”，而不只是 Top-K 商品

一次 cheap probe 应该输出：

```json
{
  "route_overlap": 0.18,
  "category_entropy": 1.42,
  "hypothesis_distribution": {
    "loafer": 0.38,
    "walking_shoe": 0.34,
    "ankle_boot": 0.28
  },
  "embedding_coherence": 0.41,
  "top_margin": 0.03,
  "hard_constraint_satisfaction": 0.76,
  "feasible_candidate_count": 183,
  "discriminative_facets": [
    {
      "facet": "formality",
      "values": ["sporty", "smart-casual", "formal"],
      "expected_information_gain": 0.61
    }
  ]
}
```

推荐使用以下信号：

### Candidate-space signals

* category entropy；
* cluster entropy；
* embedding coherence；
* Top-1 / Top-5 score margin；
* Top-K 相似度方差；
* hypothesis coverage。

### Cross-route signals

* BM25 / Dense Jaccard overlap；
* rank-biased overlap；
* 各路线 Top 结果是否支持同一 hypothesis；
* query rewrite 前后 result-set divergence。

### Constraint signals

* 每个 hard constraint 的满足率；
* metadata unknown rate；
* 所有 hard constraint 联合满足的商品数；
* 哪个条件导致 candidate collapse。

### Question signals

* 哪个 facet 能最均衡地切分候选；
* 不同答案是否真正改变最终排名；
* 用户是否有能力回答；
* facet 是否已经被表达过。

eBay/Algolia 的 BoDS 用 engaged documents 在 embedding space 中的集中程度衡量 intent coherence，在约 300 万查询上训练 BERT regression，报告 \(R^2=0.84\)。但论文也明确承认，它把 taxonomic breadth 和 lexical ambiguity 混在一起，并存在 bag size、embedding anisotropy 等问题。([SIGIR eCom][5])

因此 BoDS-like coherence 只能是 \(U_{\text{catalog}}\) 的一个特征，不能直接等同于完整“需求明确度”。

---

# 六、排名不应该只返回一个 item score

我建议排名拆成四层。

## 6.1 第一层：Hard Constraint Gate

任何明确 hard constraint 必须优先于 semantic similarity。

例如用户说：

> brown leather loafers under $100, no patent leather

一个语义上非常相似但价格 180 美元的商品，不应该通过“高 dense similarity”进入第一名。

每个属性状态最好使用三值逻辑：

```text
SATISFIED
VIOLATED
UNKNOWN
```

而不是：

```text
match / no match
```

因为 catalog metadata 缺失不等于商品违反条件。

基本规则：

* 明确 hard + VIOLATED：过滤；
* 明确 hard + UNKNOWN：保留但强降权；
* inferred constraint：不得过滤；
* soft preference：排序加减分；
* negative preference：按置信度决定过滤或惩罚。

Etsy 的生产体系把 semantic relevance 独立于 engagement 建模，并将 lightweight student 同时用于检索后过滤、排名特征、训练 loss weighting 和最终 boosting，额外实时延迟低于 10ms。其 fully relevant listings 比例从 2025 年 8 月的 58% 上升到 10 月的 62%。([Etsy][6])

这里值得借的是：

> **相关性不是一个最后阶段的分数，而应该同时作为 gate、feature 和监控指标。**

---

## 6.2 第二层：Route-Aware Unified Ranker

建议的 item score：

$$
\begin{aligned}
S(i)=&
w_1S_{\text{cross-encoder}}
+w_2S_{\text{lexical}}
+w_3S_{\text{dense}}\\
&+w_4S_{\text{constraint}}
+w_5S_{\text{hypothesis}}
+w_6S_{\text{profile}}\\
&+w_7S_{\text{route interaction}}
-w_8S_{\text{violation}}
\end{aligned}
$$

特征应至少包括：

* lexical rank；
* dense similarity；
* intent-card similarity；
* category / facet match；
* hard / soft / negative constraint status；
* candidate 来自哪些 routes；
* route agreement；
* 支持哪个 intent hypothesis；
* hypothesis posterior；
* cross-encoder semantic relevance；
* current-session preference；
* long-term profile consistency；
* inventory confidence / metadata completeness。

尤其不要丢掉“候选从哪些路线被召回”这一信息。

例如：

```text
商品 A：BM25 #1，Structured #2，Dense #8
商品 B：仅 Dense #1
```

即使两者最终语义分相近，A 的多路线一致性通常是更强证据。

---

## 6.3 第三层：Dual-Slate Ranking

这是我认为最亮眼的设计之一。

排名系统内部同时生成两个 slate：

### A. Conversion Slate

目标是：

* 最大化当前 belief 下的 Top-K 命中；
* 满足 hard constraints；
* 高 MRR；
* 高 posterior relevance。

这是对用户正式返回的推荐列表，也是官方 evaluator 看见的列表。

### B. Diagnostic Slate

目标不是立即成交，而是：

* 覆盖主要 intent hypotheses；
* 找到最具代表性的候选；
* 暴露候选之间最关键的差异；
* 计算哪个问题最能改变排名。

例如：

```text
Representative 1: athletic walking shoe
Representative 2: smart-casual loafer
Representative 3: formal leather shoe
```

Diagnostic Slate 可以只在系统内部使用，避免为了“多样性”牺牲官方 MRR。

其目标可以写成：

$$
\max_{S}
\sum_{h\in H_t}p(h)\mathbf{1}
[\exists i\in S:i\text{ supports }h]
-
\gamma\sum_{i,j\in S}\operatorname{sim}(i,j)
$$

最重要的是：

> **低确定度下的 diversity 应该是 hypothesis coverage，而不是随机挑几个互不相似的商品。**

“红色高跟鞋、男士雨靴、儿童运动鞋”非常多样，但没有决策价值。

---

## 6.4 第四层：Action Ranker

排名组件最终不只对 item 排序，还要对动作排序：

```text
SHOW
SILENT_REWRITE
ASK
BROADEN
RELAX_CONSTRAINT
```

选择动作时优化：

$$
U(a)=
\mathbb E[\Delta Q\mid a]
+\eta\,IG(a)
-\lambda\,TurnCost
-\rho\,DriftRisk
-\kappa\,LatencyCost
$$

其中：

* \(\Delta Q\)：预期 Hit / MRR / relevance 改善；
* \(IG(a)\)：信息增益；
* TurnCost：增加一轮对话的成本；
* DriftRisk：改写后偏离原意的风险；
* LatencyCost：LLM 或复杂检索成本。

不同诊断对应不同动作：

| 观察                     | 动作                              |
| ---------------------- | ------------------------------- |
| Top 结果稳定、路线一致          | SHOW                            |
| BM25 失败但 Dense 聚焦      | SILENT_REWRITE 或调整路线            |
| 候选分成多个可解释方向            | ASK                             |
| hard constraints 导致零结果 | RELAX_CONSTRAINT                |
| 用户主动重新探索               | BROADEN                         |
| 查询简单且精确                | fast path，禁止额外 LLM intervention |

“Route, Don’t Guess”在 Amazon ESCI 上观察到：困难查询经过 result-aware intervention 后 nDCG@10 可提升 0.202，而简单查询反而下降 0.060；其 adaptive router 在接近最佳固定策略的同时减少约 30% token cost。([SIGIR eCom][7])

但必须同时讲清它的局限：

* 使用的是模拟用户；
* clarification simulator 可以访问目标；
* 删除 probe features 后 holdout 性能没有下降；
* 作者自己承认普通 score-distribution features 在其规模下已经捕获了主要信号。

因此正确借法是：

> 借“intervene selectively”的结论，而不是把 agentic probe 本身包装成已被充分验证的 magic feature。

---

# 七、追问应该由 Counterfactual Ranking 驱动

传统追问逻辑：

```text
material missing
→ ask material
```

建议改成：

```text
如果知道 material，
最终 Top-K 会不会显著变化？
```

对一个 candidate facet \(f\)，计算：

$$
VOI(f)=
\sum_vP(v)
\left[
Q(B_t\oplus f=v)-Q(B_t)
\right]
-\lambda_{\text{turn}}
$$

只有满足以下条件才问：

1. 能显著改变 Top-K；
2. 能有效减少 hypothesis entropy；
3. 各答案在 catalog 中都有真实供给；
4. 用户能理解并回答；
5. 不是已经表达过的信息；
6. 信息增益大于增加一轮对话的成本。

例如候选主要分成：

```text
运动型通勤鞋        38%
商务休闲 loafer     34%
轻户外 walking shoe 28%
```

此时：

> “你希望更偏运动休闲，还是稍微正式一些？”

比：

> “你有颜色偏好吗？”

更有价值，即使 color 是 missing slot。

另外，追问后的答案必须直接回写 retrieval plan，而不是只在原 candidate set 上 rerank。

JD 的 GenFacet 正是把 facet generation 与 intent-driven query rewriting 耦合起来：用户选择 facet 后，不只过滤旧结果，而是重写 query 并重新执行 retrieval。其正式 SIGIR 2026 工作报告了 facet CTR 相对提升 42%、facet users 的 UCVR 提升 2%，并已部署在 JD Search。([arXiv][8])

这是很重要的工业启示：

> **Clarification 必须改变召回空间，否则它只是 UI ornament。**

---

# 八、召回组和排名组之间应该定义什么接口

无论你说的“组间”是组件之间还是两名队员之间，我都会把 contract 明确冻结。

## 8.1 组件职责

| 组件                 | 负责                       | 不负责               |
| ------------------ | ------------------------ | ----------------- |
| State / Controller | 维护 belief、决定 workflow    | 不直接给商品打最终分        |
| Retriever          | 候选覆盖、route budget、商品空间诊断 | 不决定是否追问           |
| Constraint Gate    | 检查明确 hard constraints    | 不推断新用户偏好          |
| Ranker             | 相关性、统一排序、置信度、slate       | 不重新解释对话           |
| Question Selector  | 基于诊断 slate 计算 VOI        | 不机械补 missing slot |

最重要的边界：

* Ranker 不得偷偷重新 parse dialogue；
* Retriever 不得把 inferred preference 当 hard filter；
* Controller 不得只看 slot count；
* Question generator 不得在没有 catalog evidence 时自由发挥。

---

## 8.2 请求接口

```json
{
  "state_version": 4,
  "intent_hypotheses": [
    {
      "id": "h1",
      "description": "smart casual loafers for commuting",
      "probability": 0.52
    }
  ],
  "hard_constraints": [],
  "soft_preferences": [],
  "negative_preferences": [],
  "route_budget": {
    "lexical": 50,
    "structured": 40,
    "dense": 80,
    "intent": 60,
    "related": 20
  },
  "probe_only": false
}
```

## 8.3 Candidate 接口

```json
{
  "parent_asin": "B0...",
  "route_hits": ["lexical", "dense", "intent"],
  "route_ranks": {
    "lexical": 4,
    "dense": 1,
    "intent": 7
  },
  "hypothesis_support": {
    "h1": 0.81,
    "h2": 0.19
  },
  "constraint_status": {
    "budget": "SATISFIED",
    "material": "UNKNOWN"
  },
  "evidence_spans": [
    "cushioned insole",
    "genuine leather"
  ]
}
```

## 8.4 Retrieval Diagnostics 接口

```json
{
  "category_entropy": 1.21,
  "route_disagreement": 0.63,
  "top_margin": 0.04,
  "feasible_candidate_count": 247,
  "candidate_clusters": [],
  "discriminative_facets": []
}
```

## 8.5 Ranker 输出

```json
{
  "conversion_slate": [],
  "diagnostic_slate": [],
  "top10_hit_confidence": 0.71,
  "hypothesis_coverage": 0.88,
  "hard_violation_count": 0,
  "recommended_action_features": {
    "show_value": 0.74,
    "rewrite_value": 0.31,
    "ask_value": 0.48
  }
}
```

这样各模块可以独立测试、替换和做 ablation。

---

# 九、我认为最亮眼的两个核心贡献

## 亮点一：Retrieval as Sensing

传统设计：

```text
Understand → Retrieve → Rank
```

你们的设计：

```text
Retrieve → Observe Catalog Reality
        → Update Intent Belief
        → Retrieve Better
```

项目故事可以概括为：

> **The query is not fully known before search. Search itself is part of understanding the query.**

这能统一：

* dual-track routing；
* over-generality；
* proactive guidance；
* intent override；
* adaptive orchestration；
* MTTC。

---

## 亮点二：Ranking as Decision Policy

传统 ranker 只回答：

> 哪个商品更相关？

你们的 ranker还回答：

> 当前应该直接推荐、静默改写、扩展搜索、放松约束，还是花一轮对话追问？

而 Diagnostic Slate 与 Conversion Slate 的分离解决了一个非常真实的冲突：

* 为理解用户而展示多样候选；
* 为成交而把最可能商品排在最前。

这两个目标不能粗暴地揉进一个 item score。

---

# 十、哪些工业方案值得借，哪些不能照搬

| 工作                        |                     证据等级 | 借什么                                       | 不照搬什么                     |
| ------------------------- | -----------------------: | ----------------------------------------- | ------------------------- |
| JD EASP / Probe-then-Plan |         A：正式会议、部署、线上 A/B | 轻量 probe、结果感知 planning、fast path          | 4B Planner 与 RL 训练        |
| Target Unified LTR        |            A：生产部署、线上 A/B | route-aware fusion、跨渠道特征                  | 6000 万行数据规模的训练 recipe     |
| Etsy Semantic Relevance   |                A-：正式工程部署 | relevance gate、LLM teacher → student、持续监控 | 把 58→62% 直接解释成单一组件的因果提升   |
| JD GenFacet               |              A：部署、线上 A/B | clarification → rewrite → retrieval 闭环    | H800 上的 4B 在线生成系统         |
| Walmart INSPIRE           |             B：大规模离线、生产形态 | query/product symmetric intent            | 固定 schema 当作完整用户意图        |
| BoDS Specificity          |      B-：Workshop、日志与公开复现 | catalog coherence 特征                      | 把 coherence 当完整 certainty |
| Route, Don’t Guess        |         C+：Workshop、模拟实验 | selective intervention                    | 把 probe 贡献讲成已充分验证         |
| Instacart Related Intent  | B-/C+：工业研究、离线 session 分析 | 低确定度下扩展 substitutes/complements           | 全流量开启探索路线                 |

Instacart 的 Related Intent Generation 将 discovery coverage 从约 60% 扩展到 80%，但同时观察到 recall 提高伴随 precision 略降，且其部署经验明确列出了尾部 query 失效、hallucinated brands、维护成本和 counterfactual offline bias。([arXiv][9])

因此 Related Intent 应该：

* 只在低确定度下启用；
* 有明确 route budget；
* 不进入 hard constraint path；
* 可以随时 fallback；
* 不作为所有 Browsing query 的默认扩展。

---

# 十一、72 小时内真正可落地的版本

## P0：先建立可靠搜索底座

1. Event-sourced session state；
2. BM25；
3. structured filter；
4. dense exact retrieval；
5. RRF union；
6. hard constraint gate；
7. compact cross-encoder rerank；
8. evaluator adapter。

50,000 个商品不需要重型向量数据库。

如果 embedding 是 384 维 float32：

$$
50,000\times384\times4
\approx 76.8\text{ MB}
$$

直接使用 NumPy matrix multiplication + `argpartition` 就足够，通常比引入 HNSW、FAISS 服务化、外部 vector DB 更简单、更可复现。

---

## P1：加入题目哲学

1. 多 hypothesis query generation；
2. cheap probe；
3. catalog uncertainty vector；
4. dynamic route budget；
5. intent override tombstone；
6. counterfactual facet scoring；
7. selective ask / rewrite / show。

LLM 最多负责：

* 每轮 state diff；
* 生成 2–4 个 intent hypotheses；
* 将选中的 facet 写成自然问题。

不要让 LLM：

* 遍历 50,000 个商品；
* 在线给所有商品排序；
* 决定 hard constraint 是否满足；
* 生成不存在的商品属性。

---

## P2：加入真正的展示亮点

1. Offline product intent cards；
2. Diagnostic Slate；
3. Conversion Slate；
4. 每轮 search trace；
5. ablation dashboard；
6. inventory infeasibility recovery。

Demo 中最好展示：

```text
Turn 1:
“想找适合上班走路穿的鞋，舒服一点。”

Probe:
- sneakers 37%
- loafers 34%
- walking shoes 29%
- route disagreement high

Action:
Ask “更偏运动休闲，还是稍微正式一些？”
```

```text
Turn 2:
“别太运动，棕色，100 美元以内。”

State:
- sporty = negative
- color brown = soft/high confidence
- budget <= 100 = hard
- smart-casual probability rises

Retrieval:
- structured + lexical budget rises
- related-intent disabled
- hard gate removes over-budget items
```

```text
Turn 3:
“算了，我也想看看靴子。”

Override:
- old category preference tombstoned
- uncertainty rises
- exploration budget reopens
- no stale loafer constraint survives
```

这一个 demo 同时覆盖：

* Information Accumulation；
* Probe Retrieval；
* Buying/Browsing 连续转换；
* Intent Override；
* Dynamic Orchestration；
* 高价值 clarification；
* hard constraint enforcement。

---

# 十二、评价体系也要超越 toy simulator

官方分数仍然要报告，但同时增加一套更符合设计哲学的 internal benchmark：

| 指标                             | 测什么               |
| ------------------------------ | ----------------- |
| Candidate Recall@K by route    | 每条路线是否真的增加覆盖      |
| Hard Constraint Violation Rate | 是否推荐明确不满足条件的商品    |
| Easy-query Intervention Rate   | 简单 query 是否被无意义打扰 |
| Override Recovery Rate         | 改口后旧约束是否残留        |
| Rewrite Drift Rate             | 改写是否偏离原需求         |
| Information Gain per Turn      | 每次追问减少了多少不确定性     |
| ΔMRR per Question              | 追问是否真正改善排序        |
| Inventory-void Recovery        | 无商品满足时能否正确协商放宽    |
| Confidence Calibration         | “已经很确定”是否真的对应高命中  |
| p50 / p95 Latency              | 是否符合工程可行性         |
| LLM Calls / Tokens per Session | 成本与复杂度            |

必须做的 ablation：

```text
固定 Buying/Browsing 二分类
vs
continuous belief routing
```

```text
无 Probe
vs
result-aware Probe
```

```text
固定 RRF
vs
route-aware ranker
```

```text
missing-slot question
vs
VOI question
```

```text
single slate
vs
conversion + diagnostic dual slate
```

```text
title-only product representation
vs
product intent cards
```

同时给出反证条件：

* 如果 probe diagnostics 无法预测哪种 action 更好，删除 probe；
* 如果 dual slate 不减少问题轮数或不改善 override 场景，不能称为贡献；
* 如果 intent cards 只提高 simulator 分数，却增加 constraint violation，应降级为 soft feature；
* 如果 route-aware ranker在 held-out session 上不胜过 RRF，就保留简单方案；
* 如果 clarification 的平均 ΔMRR 小于 turn cost，就应该更少提问。

---

# 最终我会拍板的系统定义

## **BeliefSearch: Catalog-Grounded Active Preference Search**

它的五个核心模块是：

```text
1. Result-Aware Intent Belief
2. Multi-Hypothesis Multi-Route Retrieval
3. Route-Aware Constraint-Safe Ranking
4. Diagnostic / Conversion Dual Slate
5. Value-of-Information Action Controller
```

对应的一句话故事：

> **The user’s intent is not a label to classify, but a belief to update. We use retrieval as a sensor, ranking as a decision policy, and clarification only when the expected information gain exceeds the cost of another turn.**

这套设计不是偏离题面，而是把题面中看似分散的 Dual Track、State Machine、Proactive Guidance、Context Programming 和 MTTC，重新解释成一个统一问题：

# **Managing preference uncertainty over time**

你们现在最应该先冻结的不是具体 embedding model，而是 **Belief State、Retrieval Diagnostics、Candidate Provenance 与 Ranker Output 这四个组间 contract**；这四个接口一旦正确，后续模型和规则都可以替换。

[1]: https://github.com/TechJam2026/techjam-conversational-search "https://github.com/TechJam2026/techjam-conversational-search"
[2]: https://arxiv.org/html/2603.15262v2 "https://arxiv.org/html/2603.15262v2"
[3]: https://arxiv.org/html/2606.23889v1 "https://arxiv.org/html/2606.23889v1"
[4]: https://arxiv.org/html/2602.23530v3 "https://arxiv.org/html/2602.23530v3"
[5]: https://sigir-ecom.github.io/eCom26Papers/paper_753.pdf "https://sigir-ecom.github.io/eCom26Papers/paper_753.pdf"
[6]: https://www.etsy.com/codeascraft/how-etsy-uses-llms-to-improve-search-relevance "https://www.etsy.com/codeascraft/how-etsy-uses-llms-to-improve-search-relevance"
[7]: https://sigir-ecom.github.io/eCom26Papers/paper_765.pdf "https://sigir-ecom.github.io/eCom26Papers/paper_765.pdf"
[8]: https://arxiv.org/html/2603.19665v1 "https://arxiv.org/html/2603.19665v1"
[9]: https://arxiv.org/html/2607.27172v1 "https://arxiv.org/html/2607.27172v1"
