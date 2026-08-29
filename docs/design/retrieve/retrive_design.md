可以。现在我会把它收敛成一套**可以直接冻结接口并开始实现的 Retrieval + Ranking v1**。核心不变：

> **Probe 负责观察当前需求在 catalog 中呈现出的结构，得到连续确定度 \(C_t\)；\(C_t\) 不是 Buying/Browsing 标签，而是连续控制召回宽度、route allocation、ranking emphasis 和 diversity 的旋钮。**

正式题面本身要求 multi-route retrieval、keyword/category/vector similarity、dynamic truncation、slot decay，并同时考 Coverage、MRR 和 MTTC，所以这种连续 orchestration 是在题面允许范围内对 Dual-Track Routing 的泛化。

但如果要严谨，我会对我们之前的设计做一个小修正：

> **不要把所有 probe signal 都硬塞进 \(C_t\)。**

例如 BM25/Dense 不一致，既可能因为“用户需求模糊”，也可能因为“retriever 出错”。JD 的 EASP 恰恰把 inventory void、recall failure、precision failure 区分开来。([arXiv][1])

因此最终状态应该是：

$$
\boxed{
\text{Probe}\rightarrow
(C_t,\ D_t)
}
$$

其中：

* \(C_t\)：**Catalog-Grounded Intent Clarity**，用户当前需求在商品空间中的明确程度；
* \(D_t\)：Retrieval Diagnostics，告诉系统“为什么搜索表现成这样”。

**整个系统仍然只有 \(C_t\) 这一个 exploration↔precision 主轴。** \(D_t\) 只是防止我们把“检索器坏了”误诊成“用户不知道自己想要什么”。

---

# 一、我建议冻结的整体算法

```text
                SessionContext
                      │
               Query Compiler
          ┌───────────┼────────────┐
          ▼           ▼            ▼
        q_lex       q_sem        constraints
          │           │            │
          └──────┬────┴────────────┘
                 ▼
          Fixed Neutral Probe
                 │
        ┌────────┴─────────┐
        ▼                  ▼
   Clarity C_t        Diagnostics D_t
        │
        └────────┬─────────┘
                 ▼
      Adaptive Multi-Route Retrieval
                 │
      ┌──────────┼─────────────┐
      ▼          ▼             ▼
    BM25F      Dense       Intent/Facet
      │          │             │
      └──────────┼─────────────┘
                 ▼
        C-aware Candidate Fusion
                 │
          Hard Constraint Gate
                 │
                 ▼
             Pre-Rank
                 │
                 ▼
        Semantic Relevance Rank
                 │
                 ▼
       Channel-Aware Final Rank
                 │
                 ▼
        C-aware light reranking
                 │
                 ▼
              Top-10
```

这里最重要的两个原则：

1. **Probe 本身不受 \(C_t\) 控制。**
   否则会产生循环：因为觉得用户明确 → 用精确检索 → 结果自然集中 → 更觉得用户明确。

2. **真正搜索才受 \(C_t\) 控制。**

JD 的 Probe-then-Plan 就是一个固定、廉价、只保留核心 retrieval/relevance 的 probe，然后把 snapshot 交给 planner；其 JD 实现报告 probe 相比完整搜索 tp99 latency 降约 75%。([arXiv][1])

---

# 二、Query Understanding 给 Retrieval 的输入

Retrieval **不要再自己读整段 conversation**。

QU 应编译成三个 query view。

### 1. `q_lex`

只保留可精确匹配的信息：

```text
brown mens leather loafer waterproof
```

适合：

* BM25；
* brand；
* exact category；
* material；
* model；
* color。

### 2. `q_sem`

保留自然语言意图：

```text
comfortable smart-casual men's shoes for commuting,
prefer brown leather, not sporty, budget around $100
```

适合 Dense / Semantic Ranker。

### 3. Structured constraints

例如：

```json
{
  "category": {
    "value": "loafers",
    "source": "explicit",
    "confidence": 0.97
  },
  "price": {
    "op": "<=",
    "value": 100,
    "strength": "hard"
  },
  "color": {
    "value": "brown",
    "strength": "soft",
    "confidence": 0.91
  },
  "sporty": {
    "value": false,
    "strength": "soft",
    "confidence": 0.88
  }
}
```

这里我会非常严格地区分：

### `filterable_hard`

可以真的过滤：

* price；
* canonical brand；
* canonical size；
* gender；
* 明确且 catalog coverage 足够高的 category；
* 可靠的 structured attributes。

### `semantic_hard`

用户语气上是“必须”，但商品库没有可靠结构字段：

> “不能太正式”
> “适合长时间走路”
> “不要看起来廉价”

**不能拿 LLM 猜出来的属性做 hard filter。**

它们只能进入 semantic relevance / strong penalty。

这是一个很重要的工程边界。

---

# 三、商品侧索引：四条 Route

我建议最终是四路，不是两个搜索器。

## Route L — Lexical / BM25F

不要用裸 BM25，而用 field-aware BM25：

$$
S_L(q,i)=
\sum_f w_f\,BM25(q_f,d_{i,f})
$$

初始 field weight：

| Field                 | weight |
| --------------------- | -----: |
| title                 |    4.0 |
| category              |    2.5 |
| structured attributes |    2.0 |
| description           |    1.0 |

这几个值是**我们的初始工程值，不是论文数字**，后续做 ablation。

目的：高 specificity 时非常重要，而且型号、颜色、材质、品牌等 lexical information，Dense 不应该替代。

---

# 四、Route D — Raw Dense Semantic

离线给每个商品生成：

```text
title
category
important attributes
compact description
```

的 embedding：

$$
e_i^D=Encoder(x_i)
$$

查询：

$$
e_q^D=Encoder(q_{\text{sem}})
$$

然后：

$$
S_D(q,i)=\cos(e_q^D,e_i^D)
$$

50k catalog 很小。

即使使用 1024-d float32：

$$
50000\times1024\times4
\approx 205MB
$$

完全可以直接做 exact in-memory dot product，没有任何必要为了“工业感”上向量数据库。

这也符合题面禁止 heavy external vector DB 的要求。

---

# 五、Route I — Intent-Aware Semantic Retrieval

这是我认为值得加的“企业方案借鉴”。

Raw product：

```text
Women's leather loafers, cushioned insole, brown
```

Intent view：

```text
product_type: loafers
style: smart casual
occasion: office, commuting
function: walking comfort
appearance: understated
material: leather
```

然后独立 embedding：

$$
e_i^I=Encoder(IntentCard_i)
$$

和：

$$
e_q^I=Encoder(IntentView(q_t))
$$

这个 route 的意义不是“再做一次 Dense”，而是专门解决：

> **用户使用 scenario language，而 catalog 使用 product language。**

Walmart 2026 的 INSPIRE 正是将 query 和 product 都增强为结构化、多维 intent representation，再进入 bi-encoder retrieval；其离线实验报告 Precision@1 +4.2%、NDCG@10 +2.64%，embarrassing results 减半。([arXiv][2])

但证据边界要讲清楚：INSPIRE 当前论文只有 offline results，论文自己说 A/B test 尚待部署后进行；固定 intent schema 也被作者列为局限。([arXiv][2])

所以我们的做法应该是：

> **IntentCard 是一个 soft semantic view，而不是真实属性表。**

绝对不能：

```text
LLM觉得商品适合commuting
→ commuting = true
→ hard filter
```

---

# 六、Route F — Structured Facet / Category Retrieval

不是简单 SQL filter。

它输出一个 structural relevance：

$$
S_F(i)=
\frac{
\sum_{j\in A_t}a_j(t)\cdot m_j(i)
}{
\sum_{j\in A_t}a_j(t)
}
$$

其中：

* \(A_t\)：active structured preferences；
* \(m_j(i)\in[0,1]\)：商品对 facet \(j\) 的匹配程度；
* \(a_j(t)\)：当前 slot 权重。

明确 hard constraints 会进入 filter gate。

soft constraint 则进入这个 score。

---

# 七、Slot Decay 应该具体定义

题面明确写了 slot decay。

我建议：

### Explicit hard constraint

$$
a_j(t)=1
$$

**永不自动衰减。**

只有用户明确：

* override；
* relax；
* erase；

才改变。

例如：

> “必须 100 以下”

不能过三轮之后系统自己变成 120。

---

### Explicit soft preference

$$
a_j(t)
=
a_j(t_j)
e^{-\lambda_s(t-t_j)}
$$

但衰减要慢。

### Inferred / profile preference

衰减更快：

$$
\lambda_{\text{profile}}>
\lambda_{\text{explicit-soft}}
$$

因此优先级：

```text
explicit hard
    >
explicit soft
    >
session inferred
    >
long-term profile prior
```

这部分直接进入 Route F 和 Ranking。

---

# 八、Probe：我建议固定成什么样

每轮 Query Understanding 完成后，固定跑：

```text
BM25F top 40
Dense top 40
Intent Dense top 40
```

不动态变。

Structured constraints 不作为单独 probe route，但同时计算：

```text
constraint selectivity
known violation ratio
unknown ratio
feasible count
```

为什么不能先 C-aware probe？

因为会出现：

```text
C高
↓
只搜窄结果
↓
结果很集中
↓
C更高
```

这是一种 self-fulfilling certainty。

所以：

$$
Probe(q_t)
$$

必须是 stationary observation operator。

---

# 九、我会怎么严谨定义 \(C_t\)

这里是整个算法的核心。

首先我会明确：

> \(C_t\) **不是**“用户心理确定性的概率”。

它是一个 operational score：

### **Catalog-Grounded Intent Clarity**

即：

> 当前 session state 在当前 catalog 上对应的商品需求空间有多集中。

---

## 9.1 Semantic Coherence

借 BoDS，但不能直接抄原论文。

对 Probe Dense Top-\(n\) 商品 embedding：

先用全 catalog embedding 均值中心化：

$$
\tilde e_i=
\frac{e_i-\mu_{\text{catalog}}}
{\|e_i-\mu_{\text{catalog}}\|}
$$

然后：

$$
R=
\left\|
\frac1n
\sum_i\tilde e_i
\right\|
$$

对应平均 pairwise cosine：

$$
G=
\frac{nR^2-1}{n-1}
$$

这是 BoDS 的核心几何关系。

BoDS 原论文确实用 engaged documents 的 mean resultant length 定义 query intent coherence，并强调 broad↔specific 是连续谱；但原论文的 bag 来源是 historical clicked/converted products，不是 probe top-K。([Speaker Deck][3])

因此我们应该明确叫：

> **Probe Result Coherence, BoDS-inspired**

而不是“我们实现了 BoDS”。

---

## 9.2 为什么还要 mean-center

BoDS 自己讨论了 embedding anisotropy：即随机商品 embedding 之间 cosine 本来就可能明显大于零。

否则：

```text
完全随机的30件商品
```

也可能表现得“挺集中”。

所以要用 corpus baseline 校正。

离线预计算：

$$
\rho_0 =
E_{i,j\sim Catalog}
[\cos(\tilde e_i,\tilde e_j)]
$$

以及 coherence 的高分位：

$$
\rho_{95}
$$

定义：

$$
C_{\text{sem}}
=
clip
\left(
\frac{G-\rho_0}
{\rho_{95}-\rho_0},
0,1
\right)
$$

这里 \(\rho_0,\rho_{95}\) **从你们自己的 catalog 自动估计**，不手写 cosine threshold。

这一点比：

```python
if avg_cosine > 0.7:
```

严谨很多。

---

# 十、Category Coherence

Dense Top-\(n\) 的 category 分布：

$$
p(c)=
\frac{\#\{i:category(i)=c\}}n
$$

entropy：

$$
H_{\text{cat}}
=
-\sum_cp(c)\log p(c)
$$

归一化：

$$
C_{\text{cat}}
=
1-
\frac{H_{\text{cat}}}
{\log |\mathcal C_{\text{observed}}|}
$$

于是：

```text
80% loafers
15% flats
5% dress shoes
```

比：

```text
20% sneakers
20% boots
20% loafers
20% flats
20% dress shoes
```

更加确定。

---

# 十一、State Specificity：绝对不要数 slot

以前我们已经反复强调：

```python
certainty = filled_slots / total_slots
```

是伪创新。

真正应该计算一个条件**到底缩小了多少 catalog**。

对于 constraint \(j\)，设：

$$
p_j=
\frac{
|\{i:i\text{ satisfies }j\}|
}{
|\mathcal I_{\text{scope}}|
}
$$

信息量：

$$
I_j=-\log(p_j+\epsilon)
$$

例如：

```text
gender = men
```

可能还剩 45%。

信息量不大。

而：

```text
brand = Dr. Martens
```

可能只剩 0.3%。

非常 specific。

综合 active state：

$$
I_{\text{state}}
=
\sum_j
r_j\,
conf_j\,
I_j
$$

其中：

```text
explicit hard       r = 1.0
explicit soft       r = 0.6
session inferred    r = 0.3
profile prior       r = 0.15
```

这些也是**我们的初始化值**。

再压到 \([0,1]\)：

$$
C_{\text{state}}
=
1-e^{-I_{\text{state}}/\tau}
$$

\(\tau\) 用 catalog 分布的 median / percentile 自动校准。

---

# 十二、最终 Clarity

第一版我建议：

$$
\boxed{
C_t=
0.55C_{\text{sem}}
+
0.20C_{\text{cat}}
+
0.25C_{\text{state}}
}
$$

这三个权重不是论文数字。

这是我建议的 **v1 prior**。

理由是：

* 核心创新必须来自真实 retrieval space，所以 retrieval geometry 占 75%；
* session state 不能被忽略；
* 又不能让 slot count 主导。

后续不要直接“为了 public evaluator 最优”任意调 20 个参数。

只允许小范围：

$$
w_{\text{sem}},
w_{\text{cat}},
w_{\text{state}}
$$

三参数搜索。

---

# 十三、那我们以前说的 route agreement / score margin 去哪里了？

我现在建议：

**从 \(C_t\) 主体移出去，放入 \(D_t\)。**

这是我这轮最想修正的一个点。

因为：

```text
BM25和Dense不一致
```

不一定说明：

> 用户不确定。

也可能说明：

> retriever vocabulary mismatch。

同理：

```text
Top1 score 很低
```

也不等于：

> 用户不知道自己要什么。

可能只是：

> catalog 没货。

JD EASP 正是明确区分 Effective、Recall Failure、Inventory Void 和不同类型的 Precision Failure。([arXiv][1])

所以定义：

$$
D_t=
\{
A_{\text{route}},
M_{\text{score}},
F_{\text{feasible}},
N_{\text{candidate}},
...
\}
$$

---

## Route Agreement

BM25 与 Dense Top20：

我倾向用 RBO 而不是 Jaccard，因为排名位置也重要：

$$
A_{\text{route}}
=
RBO(
L_{BM25}^{20},
L_{Dense}^{20};
p=0.9
)
$$

---

## Score sharpness

每条 route 内部做 robust normalization：

$$
z_i=
\frac{s_i-\operatorname{median}(s)}
{\operatorname{MAD}(s)+\epsilon}
$$

然后看：

* top1 margin；
* top10 std；
* kurtosis。

Route, Don’t Guess 的 router 就使用 top-1、top-10 std、kurtosis、brand entropy、semantic coherence 等连续 retrieval signals。([SIGIR eCom][4])

但有一个重要反证：论文删除其五个“agentic probe”特征后，held-out router 性能没下降，所以我们不能声称“probe divergence feature 已被证明必要”；真正有力的结论是**continuous result-aware signals 是有价值的，而不是某一个 feature 是 magic**。([SIGIR eCom][4])

---

# 十四、于是 Retrieval Controller 的输入是

$$
(C_t,D_t,State_t)
$$

不是：

```text
BUYING / BROWSING
```

然后真正搜索时才动态改变 route。

---

# 十五、召回预算：连续变化，不设 Buying threshold

设：

$$
E_t=1-C_t
$$

代表 exploration strength。

我建议起步使用：

$$
K_L=80+60C_t
$$

$$
K_D=100+60(1-C_t)
$$

$$
K_I=40+100(1-C_t)
$$

$$
K_F=40+100C_t
$$

取整数。

于是：

| \(C_t\) | BM25 | Dense | Intent | Facet |
| ------: | ---: | ----: | -----: | ----: |
|     0.0 |   80 |   160 |    140 |    40 |
|     0.5 |  110 |   130 |     90 |    90 |
|     1.0 |  140 |   100 |     40 |   140 |

注意：

* 没有 active facet 时 F 不跑；
* 没有可靠 IntentCard 时 I 不跑；
* 未使用的 budget 可以给 Dense；
* 这是**fetch budget**，不是最终 ranking weight。

这就真正实现了题目里的 **custom dynamic truncation**。

---

# 十六、为什么高 C 仍然保留 Dense

这是设计上非常重要的一点。

绝不能变成：

```python
if C > .7:
    BM25 only
```

否则只是把 Buying/Browsing 二分类藏起来了。

即便 \(C=1\)：

```text
BM25    140
Dense   100
Intent   40
Facet   140
```

四路仍然存在。

只是 allocation 改变。

所以系统真正实现的是：

$$
\textbf{soft routing}
$$

而不是：

$$
\textbf{hard routing}
$$

---

# 十七、召回 fusion：RRF 可以用，但只能当 candidate fusion

不同 route 的 raw score 不可直接比较：

```text
BM25 = 14.2
cosine = .73
facet = .9
```

第一轮 candidate fusion：

$$
S_{RRF}(i)
=
\sum_r
\frac{w_r(C_t)}
{k_0+rank_r(i)}
$$

取：

$$
k_0=60
$$

初始 route weights 用两个 endpoint 插值：

### Explore endpoint \(C=0\)

```text
BM25      .15
Dense     .35
Intent    .35
Facet     .15
```

### Precision endpoint \(C=1\)

```text
BM25      .35
Dense     .20
Intent    .10
Facet     .35
```

所以：

$$
w_r(C)
=
(1-C)w_r^{browse}
+
Cw_r^{precise}
$$

然后 candidate union 截到例如：

$$
K_{\text{union}}=300-160C
$$

即：

```text
C=0 → 300
C=1 → 140
```

再进入 ranker。

---

# 十八、为什么 RRF 不能作为最终 Ranker

Target 2026 的生产系统正好给出了很好的证据。

他们指出固定 RRF / weighted interleaving：

* 使用 global channel weight；
* 不理解 query-specific channel utility；
* 不理解 cross-channel interaction。

因此他们把：

```text
candidate source
retrieval score
channel signals
item signals
```

一起交给统一 LTR。

在 Target.com 上：

* Weighted Interleaving NDCG@8 = 0.6620；
* Unified Ranking = 0.7169；
* 加 engagement features = 0.7799；
* 最终版本 = 0.7994；
* conversion +2.85%；
* p95 < 50ms；
* 已部署。([arXiv][5])

所以我们应该：

> **RRF 用来召回合并和 cheap pre-truncation，不能让 RRF 当最终 ranking architecture。**

---

# 十九、Hard Constraint Gate 必须在真正 Ranker 前

每个 candidate 对每个 hard constraint 有三个状态：

$$
m_j(i)\in
\{
SATISFIED,
VIOLATED,
UNKNOWN
\}
$$

规则：

### Known VIOLATED

直接过滤：

```text
price <= $100
product price = $159
→ DROP
```

### UNKNOWN

不能过滤：

```text
material = leather
product material missing
→ KEEP + penalty
```

### Semantic Hard

例如：

> “绝对不要太 sporty”

如果 sporty 是 LLM 推断属性：

```text
不能 DROP
```

只能：

```text
large semantic penalty
```

这个设计能避免一个工业系统很常见的灾难：

> metadata missing 被错误解释成 false。

---

# 二十、Ranking：我建议三阶段

---

## Stage A：Cheap Pre-Ranker

输入 RRF union，大概 140–300 items。

Feature：

```text
BM25 rank
BM25 robust-z score
Dense rank
Dense cosine
Intent rank
Intent cosine
Facet score
number of routes hit
RRF score

exact attribute matches
soft preference matches
negative preference matches
hard UNKNOWN count

category distance
C_t
```

Cheap score：

$$
S_{\text{pre}}(i)
$$

筛到：

$$
Top80
$$

---

# 二十一、Stage B：Semantic Relevance

这里才应该落实题面的 “LLM Semantic Ranking”。

但我不建议生产路径：

```text
每一轮
80个商品
全部调用大LLM
```

企业现在基本也不是这么做。

Etsy 的最新生产架构是：

```text
human labels
→ expensive LLM annotator
→ Qwen3-VL-4B teacher
→ BERT two-tower student
→ real-time relevance signal
```

其轻量 student 增加的实时延迟低于 10ms，并被同时用于：

* retrieval 后 filtering；
* downstream ranker feature；
* loss weighting；
* final relevance boosting。

该框架已部署。([Etsy][6])

DoorDash SIGIR 2026 更直接：把 semantic relevance 做成 ordinal head，并直接纳入最终 value function；其三周线上 A/B 报告 ATCR +1.16%、CVR +1.10%、GOV +0.50%。([arXiv][7])

---

# 二十二、我们的 Hackathon 版本怎么办

72 小时没必要重新训练 Etsy 那套。

定义一个统一接口：

```text
SemanticRelevance(q_sem, product)
→ score ∈ [0,1]
```

实现可以首先用：

### P0

local cross-encoder。

例如 Route, Don't Guess 使用：

```text
ms-marco-MiniLM-L-12-v2
```

对 Dense Top100 rerank，使 nDCG@10 从 0.492 → 0.514，而且不需要 LLM tokens。([SIGIR eCom][4])

之后再比较：

* BGE reranker；
* Qwen reranker；
* small local LLM；
* external LLM pointwise/listwise。

接口不变。

---

# 二十三、为什么不是让 LLM 直接给 1～80 排序

因为 listwise LLM ranking：

* 对排列位置敏感；
* 输出稳定性差；
* token 高；
* latency 高；
* 很难 debug；
* 不能很好处理严格 hard constraints。

我们真正要的是：

$$
semantic\ relevance
$$

作为独立 feature。

这也是 Etsy / DoorDash 当前工业方案更值得借鉴的点。

---

# 二十四、Stage C：Channel-Aware Final Ranker

最终每个商品产生：

$$
x_i=
[
C_t,
S_L,
S_D,
S_I,
S_F,
S_{RRF},
S_{sem},
route\_mask,
preferences,
constraints,
...
]
$$

尤其加入 interaction feature：

$$
C_tS_L
$$

$$
C_tS_F
$$

$$
(1-C_t)S_D
$$

$$
(1-C_t)S_I
$$

所以 ranker可以显式知道：

> 同一个 Dense score，在 low clarity 与 high clarity 状态下意义并不相同。

这是我们的 architecture 最核心的一步。

---

# 二十五、最终 Ranker 我建议两级方案

## Ranker V1：解析式 C-conditioned ranker

先实现这个，保证系统可运行。

所有 route score 先做 per-query robust normalization：

$$
z_r(i)
=
clip
\left(
\frac{s_r(i)-median(s_r)}
{MAD(s_r)+\epsilon},
-3,3
\right)
$$

语义分也 normalize。

然后：

$$
\begin{aligned}
S(i)=&
w_L(C)z_L(i)
+w_D(C)z_D(i)\\
&+
w_I(C)z_I(i)
+w_F(C)S_F(i)\\
&+
w_S(C)z_{sem}(i)\\
&+
P_t(i)
-
N_t(i)
-
U_t(i)
\end{aligned}
$$

---

## 初始 endpoint ranking weights

### \(C=0\)

```text
lexical          .10
dense            .25
intent           .25
facet            .10
semantic ranker  .30
```

### \(C=1\)

```text
lexical          .25
dense            .15
intent           .05
facet            .30
semantic ranker  .25
```

中间线性插值：

$$
w_r(C)
=
(1-C)w_r^{explore}
+
Cw_r^{precise}
$$

这组数字依然只是：

> **可运行且符合 inductive bias 的 initial policy。**

不是论文结论。

---

# 二十六、Preference score

soft preference：

$$
P_t(i)
=
\sum_j
a_j(t)
\cdot match_j(i)
$$

negative preference：

$$
N_t(i)
=
\sum_j
b_j(t)
\cdot violation_j(i)
$$

unknown hard attribute：

$$
U_t(i)
=
\lambda_u
\sum_j
\mathbf1[
status_j(i)=UNKNOWN
]
$$

而：

```text
KNOWN hard violation
```

已经在 gate 阶段移除了。

---

# 二十七、Ranker V2：Shallow LambdaMART

这个是我建议最终提交版本尝试的。

结构直接借 Target：

> 把 multi-route candidate fusion 重新表述成 channel-aware Learning-to-Rank。

Target 用的是 GBDT + LambdaMART，并强调 retrieval scores、channel source、missing channel signals 都保留下来作为 features。([arXiv][5])

但 Target 有：

* 约 60M rows；
* 500k queries。

我们只有 200 public sessions。([arXiv][5])

所以绝不能：

> “Target 用 LambdaMART，所以我们也堆一个 1000-tree GBDT。”

---

# 二十八、我们的小数据 LambdaRank 应该这样训练

每个：

```text
(session_id, turn)
```

是一组 ranking group。

candidate 中：

```text
target ASIN → label 1
others      → label 0
```

如果 target 根本没有被 recall：

> **这轮不用于 ranker 训练。**

因为这是 retrieval failure，不能让 ranker承担。

Split 必须：

```text
GroupKFold(session_id)
```

绝不能：

```text
随机切 turn
```

否则同一 session 的相邻 turn 会泄漏。

---

## 推荐参数范围

不是固定死值：

```text
objective       = lambdarank
metric          = ndcg
ndcg_at         = [10]

num_leaves      = 7 ~ 15
max_depth       = 3 ~ 4
learning_rate   = 0.03 ~ 0.05
lambda_l2       = strong
min_data_leaf   = conservative
early_stopping  = true
```

而且：

### 禁止放进去

```text
ASIN
user_id
session_id
turn-specific target identity
```

否则太容易学 evaluator artifact。

只允许：

```text
retrieval / relevance / state features
```

---

# 二十九、什么条件下我们才采用 LambdaMART

它必须同时满足：

$$
MRR_{CV}^{LTR}>
MRR_{CV}^{analytic}
$$

并且：

$$
Hit@10_{CV}
$$

不降低，以及：

$$
HardViolationRate=0
$$

否则：

> **直接提交解析式 ranker。**

这不是退步。

在 200 session 的 hackathon 中，一个稳定的 structured ranker 比过拟合的“工业 LTR”更专业。

---

# 三十、最后一个 rerank：C-aware diversity

低 \(C\) 时，题面要求的 diverse browsing 不能只停留在 retrieval。

但是不能为了 diversity 砸掉 MRR。

所以：

### Rank 1 不动

然后 rank 2–10 做轻量 MMR：

$$
MMR(i)
=
S(i)
-
\lambda(C)
\max_{j\in Selected}
\cos(e_i^I,e_j^I)
$$

其中：

$$
\lambda(C)
=
\lambda_{\max}(1-C)^2
$$

建议开始：

$$
\lambda_{\max}=0.08\sim0.12
$$

于是：

```text
C = 1
λ = 0
```

完全 precise ranking。

```text
C = 0
λ ≈ .1
```

才允许适度 diversity。

而不是一个固定：

```text
diversity=true
```

---

# 三十一、这里的 diversity 不是随机“商品长得不一样”

我们真正想要的是：

### intent-space diversity

例如：

```text
smart-casual loafer
comfortable flat
minimal walking shoe
```

而不是：

```text
鞋
项链
帽子
```

所以 MMR 最好使用：

$$
e_i^I
$$

IntentCard embedding，

而不是原始 title embedding。

---

# 三十二、最终整个 Retrieval 算法可以写成

$$
\boxed{
\begin{aligned}
&State_t=QU(Dialogue_{\le t})\\
&(C_t,D_t)=Probe(State_t)\\
&K_r=K_r(C_t)\\
&R_r=Retrieve_r(q_r,K_r)\\
&R=\bigcup_r R_r\\
&S_{rrf}(i)=\sum_r\frac{w_r(C_t)}{60+rank_r(i)}\\
&R'=\operatorname{Top}_{K(C_t)}(R,S_{rrf})\\
&R''=HardConstraintGate(R')\\
&Candidates=\operatorname{PreRank}(R'')
\end{aligned}
}
$$

然后 Ranking：

$$
\boxed{
\begin{aligned}
S_{\text{sem}}(i)&=SemanticRank(q_t,i)\\
x_i&=
[
C_t,
route\ scores,
route\ ranks,
facet,
sem,
prefs,
constraints
]\\
S_i&=f_\theta(x_i)\\
Top10&=C\text{-aware-Rerank}(S)
\end{aligned}
}
$$

其中：

* \(f_\theta\) 第一版是解析式；
* 第二版是 shallow LambdaMART。

---

# 三十三、还有一个很重要的 diagnostic matrix

这是 Probe-then-Plan 真正值得抄的地方。

| \(C_t\) | Retrieval diagnostics | 含义                       | Retrieval 策略                             |
| ------- | --------------------- | ------------------------ | ---------------------------------------- |
| 高       | routes 一致，score sharp | 需求明确且搜索健康                | preserve，precision-heavy                 |
| 高       | routes 冲突             | **用户清楚，但 retriever 有问题** | result-aware rewrite / semantic rescue   |
| 高       | feasible = 0          | **库存不满足**                | 不假装用户模糊，不静默放松 hard constraint            |
| 低       | routes 较稳定            | genuine browsing         | semantic/intention expansion + diversity |
| 低       | routes 混乱             | broad + vocabulary gap   | widest recall + semantic expansion       |

这一步很重要，因为它避免了：

$$
\text{retrieval failure}
\Rightarrow
\text{low user certainty}
$$

这种错误推断。

这正是 JD EASP 里“环境感知”比盲目 rewrite 更成熟的地方。其 offline ablation 中，没有 retrieval snapshot 的 Blind Rewriter HR@30 为 28.6%，EASP 为 31.0%；之后两周线上 A/B 的总体 UCVR +0.89%、GMV +0.57%，并已部署。([arXiv][1])

---

# 三十四、召回组与 Ranking 组真正应该传什么

绝对不要只传：

```python
[(asin, score), ...]
```

应该：

```json
{
  "asin": "...",

  "routes": {
    "lexical": {
      "hit": true,
      "rank": 4,
      "raw_score": 12.84,
      "norm_score": 1.37
    },
    "dense": {
      "hit": true,
      "rank": 2,
      "raw_score": 0.734,
      "norm_score": 1.81
    },
    "intent": {
      "hit": false
    }
  },

  "rrf_score": 0.034,

  "constraint_status": {
    "price": "SATISFIED",
    "material": "UNKNOWN"
  },

  "facet_score": 0.81
}
```

再另传：

```json
{
  "clarity": 0.62,

  "diagnostics": {
    "semantic_coherence": 0.71,
    "category_coherence": 0.56,
    "route_agreement": 0.38,
    "feasible_count": 124,
    "inventory_void": false
  }
}
```

Target 的结果非常支持“保留 channel provenance”：其 unified ranker 就是把**所有 channel 的 retrieval score 和 source signal**暴露给最终模型，让模型学习 channel utility 和 cross-channel interaction。([arXiv][5])

---

# 三十五、证据等级

| 设计                              | 来源                 |                                    证据强度 | 我们怎么用                                     |
| ------------------------------- | ------------------ | --------------------------------------: | ----------------------------------------- |
| Retrieval Probe                 | JD EASP            | **A**：SIGIR + production A/B + deployed | 固定 cheap probe                            |
| Channel-aware final ranking     | Target             |                **A**：大规模 A/B + deployed | 保留 route provenance，最终 unified ranking    |
| Semantic relevance 独立建模         | Etsy               |           **A-**：production engineering | cross-encoder/relevance scorer            |
| Relevance 作为明确 rank objective   | DoorDash           |                **A**：SIGIR + online A/B | semantic relevance 作为 first-class feature |
| Structured query/product intent | Walmart INSPIRE    |    **B**：industrial data + offline only | IntentCard route                          |
| Continuous specificity          | eBay/Algolia BoDS  |            **B-**：workshop + large logs | Probe coherence 定义                        |
| Result-aware continuous signals | Route, Don’t Guess |          **C+/B-**：workshop + simulator | retrieval diagnostics，不当 production proof |

这也是为什么我没有把系统直接抄成某一篇论文。

---

# 三十六、最重要的 ablation

你们要能够证明“确定度真的有用”，而不是 presentation story。

### Retrieval

```text
R1 BM25
R2 BM25 + Dense fixed hybrid
R3 + Intent route
R4 + fixed four-route
R5 + C-aware route allocation
```

报告：

* Recall@50；
* Recall@100；
* Hit@10；
* 每条 route 的 unique recall；
* 按 clarity quintile 分层结果。

最关键的一张表应该是：

| Clarity | Fixed Hybrid | Precision-heavy | Exploration-heavy | C-aware |
| ------- | -----------: | --------------: | ----------------: | ------: |
| Q1 low  |              |                 |                   |         |
| Q2      |              |                 |                   |         |
| Q3      |              |                 |                   |         |
| Q4      |              |                 |                   |         |
| Q5 high |              |                 |                   |         |

如果我们的故事成立，应该看到：

```text
low C:
exploration-heavy > precision-heavy

high C:
precision-heavy > exploration-heavy

C-aware:
overall best / near-best
```

**如果看不到这个 pattern，我们的核心 hypothesis 就没有被验证。**

---

# 三十七、Ranking ablation

```text
RRF
↓
C-conditioned analytic ranker
↓
+ Semantic cross-encoder
↓
+ Channel-aware LambdaRank
↓
+ C-aware diversity
```

分别报告：

* target conditional MRR@10：

  * 只统计 target 已经被 recalled 的 turn；
* end-to-end MRR；
* Hit@10；
* hard violation rate；
* p50 / p95 latency。

这个 distinction 很重要：

$$
P(target\in TopK)
$$

是 Retrieval 的责任。

而：

$$
rank(target)\mid target\in Candidates
$$

才是 Ranking 的责任。

否则两个组会互相甩锅。

---

# 三十八、我目前会真正冻结的 v1

如果现在让我让 Codex 开工，我会冻结：

```text
Product index
├── BM25F index
├── raw dense matrix
├── intent dense matrix
└── structured facet index

Per turn
├── Query Compiler
├── Fixed Probe
│   ├── BM25 top40
│   ├── Dense top40
│   └── Intent top40
│
├── Clarity
│   ├── semantic coherence
│   ├── category coherence
│   └── selectivity-weighted state specificity
│
├── Diagnostics
│   ├── route RBO
│   ├── score sharpness
│   ├── feasible ratio
│   └── inventory void
│
├── Adaptive Retrieval
│   ├── dynamic K_lex
│   ├── dynamic K_dense
│   ├── dynamic K_intent
│   └── dynamic K_facet
│
├── weighted RRF candidate union
├── hard constraint gate
├── cheap pre-rank → Top80
├── semantic cross-encoder
├── C-conditioned analytic rank
└── light C-aware MMR → Top10
```

**LambdaMART 放在这套 baseline 跑通以后再上。**

这套设计我认为已经足够严谨地把你们之前的“需求确定度”从一个漂亮概念变成一个真正作用于 Retrieval 和 Ranking 的**连续 control variable**。

而且它有一个很漂亮的工业叙事：

> **BoDS 告诉我们 broadness 是连续的；JD 告诉我们应先看真实 retrieval environment；Target 告诉我们 multi-route 不应靠固定 fusion；Etsy 和 DoorDash 告诉我们 semantic relevance 应成为 ranking 的 first-class signal。我们把这四件事统一成了一个 multi-turn、clarity-conditioned search pipeline。**

我觉得下一步最值得做的已经不是继续 brainstorm，而是把 **\(C_t\) 计算、四路 retrieval、candidate contract、rank features** 四个东西具体写成 class/API 级 specification，然后就可以直接实现了。

[1]: https://arxiv.org/abs/2603.15262 "Probe-then-Plan: Environment-Aware Planning for Industrial E-commerce Search"
[2]: https://arxiv.org/html/2606.23889 "INSPIRE: Intent-aware Neural Sponsored Product Retrieval for E-commerce"
[3]: https://speakerdeck.com/dtunkelang/a-bag-of-documents-model-for-query-specificity "A Bag-of-Documents Model for Query Specificity - Speaker Deck"
[4]: https://sigir-ecom.github.io/eCom26Papers/paper_765.pdf "Route, Dont Guess: Adaptive Interaction Strategy Selection for Agentic Product Search"
[5]: https://arxiv.org/html/2602.23530 "Unified Learning-to-Rank for Multi-Channel Retrieval in Large-Scale E-Commerce Search"
[6]: https://www.etsy.com/uk/codeascraft/how-etsy-uses-llms-to-improve-search-relevance?utm_source=chatgpt.com "Etsy Engineering | How Etsy Uses LLMs to Improve Search Relevance"
[7]: https://arxiv.org/html/2605.27704 "Joint Optimization of Relevance and Engagement in Multi-Task Ranking for E-Commerce with Efficient LLM Supervision"
