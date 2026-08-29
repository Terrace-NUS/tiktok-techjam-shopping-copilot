# Ranking Strategy Evaluation v0

- 状态：**实验完成，尚未替换正式 RetrievalController**
- 完整结果：`artifacts/retrieval/ranking-strategy-v0.{json,md}`
- 配对统计：`artifacts/retrieval/ranking-strategy-analysis-v0.{json,md}`
- 实现：[`../../../src/shopping_copilot/retrieval/ranking.py`](../../../src/shopping_copilot/retrieval/ranking.py)

## 1. 这轮到底比较什么

正式检索 v0 已经能执行 hard mask、Dense/Lexical/Facet 三路召回、RRF 和向量 MMR。这轮没有改变
召回，也没有拿 target 帮任何算法挑商品。每个排序方法收到同一个 RRF Top-80 候选池，只比较：

1. 三路信号怎样形成相关性顺序；
2. cross-encoder 能不能把更合适的商品提到前面；
3. $T_t$ 怎样把相关候选组成一个聚焦或分散的 Top-10。

因此候选池之外的 target 不属于 ranking 失败。报告同时给整体 MRR 和只在 target 已进入候选池时计算的
conditional MRR。

## 2. 实测的方法

### 融合方法

- **RRF**：只用每路名次，当前正式 baseline；
- **Relative Score Fusion**：先在每路内部做 min-max，再相加；Lexical BM25 会正确反向；
- **CombMNZ-style Fusion**：在 Relative Score 上增加多路共同命中的乘数。

RRF 的依据是异构系统的 raw score 不必位于同一数值空间；原始工作见
[Cormack、Clarke 与 Büttcher, 2009](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)。

### 相关性模型

- **Qwen3-Reranker-0.6B**：带 shopping instruction，model score 与 25% RRF prior 混合；
- **BGE-reranker-v2-m3**：相同候选池和文本输入，也与 25% RRF prior 混合。

两者都是 cross-encoder：同时读取 query 和一个商品文本，逐对产生相关性分数，而不是重新召回。
[Sentence Transformers 文档](https://www.sbert.net/docs/cross_encoder/usage/usage.html)也把这种模型定位为对首阶段
Top-K 做较慢但更精细的重排。实验固定版本为：

```text
Qwen/Qwen3-Reranker-0.6B@e61197ed45024b0ed8a2d74b80b4d909f1255473
BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
max_length = 384
batch_size = 32
```

官方模型说明分别见
[Qwen3 Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)和
[BGE reranker v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。

### Top-10 集合优化

- **Top-K**：直接采用相关性顺序；
- **MMR**：逐个选择“相关但不重复”的商品，参考
  [Goldstein 与 Carbonell, 1998](https://aclanthology.org/X98-1025/)；
- **Greedy DPP**：把商品质量和向量集合的行列式体积放在同一个 kernel 里；低 $T_t$ 增强排斥，
  高 $T_t$ 增强质量。快速 greedy MAP 的工程依据见
  [Chen、Zhang 与 Zhou, 2017](https://arxiv.org/abs/1709.05135)；
- **Latent xQuAD**：从候选商品向量自动找最多 6 个潜在方向，再尝试覆盖尚未覆盖的方向。它不读取
  category。概念来自
  [Explicit Query Aspect Diversification](https://theses.gla.ac.uk/4106/)。

## 3. 统一实验协议

- catalog：正式 50k 商品，原文件只读；
- story cases：6 个冻结的自然语言请求；
- simulator：80 Buying + 80 Browsing；Intent Override / Boundary 不参加；
- simulator query：官方首轮可见消息，不调用 target-aware QU；
- 候选池：所有方法共享 RRF Top-80；
- Top-10：每种方法输出同样数量；
- $T_t$：自然案例使用冻结锚点；官方 160 例对每个请求同时运行 `T=0.2` 与 `T=0.8`，而不是用
  Buying/Browsing 标签生成 $T_t$；
- category 只在结果完成后统计，不参与 MMR/DPP/xQuAD；
- 配对统计：20,000 次 bootstrap，seed `20260829`。

官方 target 只在排序完成后计算 rank。统一 Top-80 的 target recall 是 `0.425`，所以 conditional 指标
的分母是 68 个已召回任务。

## 4. 核心结果

| 方法 | MRR@10 | conditional MRR@10 | 商品两两 cosine↓ | 平均大类数↑ |
| --- | ---: | ---: | ---: | ---: |
| RRF Top-K | 0.068 | 0.161 | 0.772 | 1.64 |
| Relative Score Top-K | 0.080 | 0.188 | 0.763 | 1.82 |
| CombMNZ Top-K | 0.083 | 0.195 | 0.773 | 1.68 |
| Qwen Top-K | 0.102 | 0.240 | 0.805 | 1.12 |
| BGE Top-K | **0.145** | **0.342** | 0.808 | 1.20 |
| RRF MMR, low T | 0.060 | 0.142 | **0.734** | **2.28** |
| Qwen MMR, low T | 0.096 | 0.226 | 0.773 | 1.21 |
| Qwen DPP, low T | 0.100 | 0.235 | 0.767 | 1.21 |
| Qwen xQuAD, low T | 0.100 | 0.234 | 0.806 | 1.14 |

这个表体现了真正的冲突：cross-encoder 提高了 target 排名，却会把开放结果收缩为更相似、更少类的商品；
RRF MMR 最能展开空间，但牺牲了 simulator target rank。一个指标不能代表两种目标。

## 5. 配对统计给出的结论

### BGE 是这轮最强的纯相关性 ranker

相对 RRF，BGE 的 MRR 平均提高 `+0.0770`，95% bootstrap 区间
`[+0.0423, +0.1151]`；相对 Qwen 仍提高 `+0.0433 [+0.0110, +0.0768]`。

68 个 target 已进入候选池的任务中：

- RRF 把 26 个 target 放进 Top-10；
- Qwen 是 36 个；
- BGE 是 43 个，其中 rank 1 有 14 个；
- BGE 相对 RRF 救回 19 例，伤害 2 例。

所以如果目标只是 toy simulator 的 target rank，BGE 是明确赢家。但它把平均大类数从 `1.64` 降到
`1.20`，不能单独承担开放探索。

### DPP 打出了最稳定的 $T_t$ 响应

Qwen+DPP 从高 T 改为低 T 后：

- 99.4% 的请求变得更分散；
- 平均商品 cosine 降低 `0.0334`，区间 `[-0.0363, -0.0305]`；
- MRR 只改变 `-0.0012`，区间 `[-0.0149, +0.0127]`，没有观察到可靠损失。

相对 Qwen Top-K，低 T DPP 把 cosine 降低 `0.0385`，同时 MRR 差值
`-0.0024 [-0.0174, +0.0127]`。它比 Qwen+MMR 的向量展开更强，MRR 损失更小。

### xQuAD 在当前向量空间失败

xQuAD 只有 40% 的请求满足低 T 比高 T 更分散；平均 cosine 甚至轻微上升，区间跨 0。自动找出的
局部向量方向并不等于用户能感知的商品方向。因此 v0 淘汰 xQuAD，不继续调参包装结果。

### Score Fusion 暂时不能替换 RRF

Relative Score 和 CombMNZ 的 MRR 点估计都高于 RRF，但 95% 区间跨 0。Relative Score 确实增加了
大类数并降低了 cosine，不过它依赖每次 Top-K 内部的 min-max，候选分布变化时含义会漂移。它们保留
为 shadow ablation，不进入正式默认链路。

## 6. 延迟

在 RTX 4070 Ti 12GB、Top-80、长度 384、batch 32 下：

| 模型 | median | p95 | max |
| --- | ---: | ---: | ---: |
| Qwen3 0.6B | 3.32 s | 6.54 s | 15.89 s（含首次 warm-up） |
| BGE v2-m3 | 3.00 s | 4.91 s | 6.97 s |

两者都能稳定运行，但这个延迟不适合不加条件地进入每一轮请求。DPP/MMR 本身只处理 80 个已存在向量，
成本远小于 cross-encoder。

## 7. 当前决策

这轮只冻结**实验结论**，不冻结新的 production ranking contract：

- RRF 继续作为安全、无训练的候选池边界；
- MMR 继续作为当前正式 $T_t$ 控制点；
- DPP 通过实验，成为下一版 slate selector 的首选候选；
- BGE 通过实验，成为精确相关性重排的首选候选；
- Qwen 保留为模型消融，不优先于 BGE；
- Relative Score / CombMNZ 只保留 shadow；
- latent xQuAD 淘汰。

下一次只需回答一个尚未实测的组合问题：**BGE relevance + DPP slate** 能否同时保留 BGE 的 target-rank
收益和 DPP 的稳定 $T_t$ 响应。得到这组数据后，再决定是否修改 `RetrievalController`，不要现在凭组件
各自最好就假设组合一定最好。

## 8. 复现

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/evaluate_ranking_strategies_v0.py
.\.venv-3.10\Scripts\python.exe scripts/retrieval/analyze_ranking_strategies_v0.py
```

完整 JSON 保留每个请求、路线状态、候选位置、模型分数、13 套 Top-10、category 审计和 target rank。
生成物位于 ignored `artifacts/`；代码、固定模型版本和分析协议进入 Git。
