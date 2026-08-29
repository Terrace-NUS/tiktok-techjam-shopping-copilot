# Thresholded Semantic Space 实验 v0

- 状态：**完成第一轮真实 50k 数据分析；参数未冻结**
- 日期：**2026-08-29**
- 实验脚本：`scripts/retrieval/analyze_semantic_thresholds.py`
- 原始结果：`artifacts/retrieval/semantic-threshold-analysis-v0.json`
- 自然语言完整聚合结果：`artifacts/retrieval/semantic-space-merge-natural-v0.json`

## 1. 这轮验证什么

本轮验证下面这条候选算法是否具备计算可行性，以及阈值大致应当落在哪个区域：

```text
accepted Session Context
    -> 现有 Hard Mask
    -> q_sem 与全部 50,000 件商品计算 cosine similarity
    -> 删除相似度过低的商品
    -> 对剩余商品计算交叉相似度
    -> 合并相似度过高的商品
    -> 每个合并组等权
    -> 在组代表上计算原有 mean-centered coherence
```

这里没有要求指标逐轮严格单调。用户撤销条件、切换目标或改变主意时，候选空间和最终指标都可以显著变化。

## 2. 数据与实现

查询来自已经完成的 QU-to-Probe 全量运行，不重新调用 DeepSeek：

| Cohort | 保存的 turn | 完整成功且可搜索 |
| --- | ---: | ---: |
| 自然语言测试 | 72 | 70 |
| 官方 simulator 测试 | 128 | 122 |
| 合计 | 200 | 192 |

每个成功 turn 使用当时已经接受的 `Session Context`、`q_sem` 和 hard constraints。实验重新构造 Hard Mask，并要求每一轮的 `eligible_count` 与原日志完全相同。

商品侧使用当前 50k Dense R0 索引：

- 模型：`BAAI/bge-small-en-v1.5`
- 向量维度：384
- 商品数：50,000
- 向量已做 L2 normalization
- query-to-catalog 和 candidate-to-candidate 矩阵均使用 PyTorch CUDA
- GPU：NVIDIA GeForce RTX 4070 Ti

本实验不修改官方 catalog，也不修改已有 dense index。

## 3. 全量打分成本

一次性计算 192 个 query 对 50,000 件商品的完整矩阵：

```text
shape = 192 × 50,000
elapsed = 0.0297 seconds
```

自然语言 cohort 的 70 个 turn 完成全候选交叉矩阵、5 个 merge threshold、2 个 keep window 和聚合后 coherence，总计：

```text
full merge stage = 16.03 seconds
average           = 0.229 seconds / turn
```

实验脚本中的约 38 秒 Hard Mask 时间主要来自从原始 catalog 临时重建 evidence index。正式 runtime 若加载预构建 index，不需要每轮支付该成本。

因此，全量 50k query similarity 不是性能瓶颈。候选交叉矩阵在当前 4070 Ti 上也足以支持实验和 demo，但宽泛 query 的 runtime 上限仍应显式控制。

## 4. 保留阈值分析

### 4.1 固定绝对阈值不稳

192 个成功 turn 的结果：

| Absolute cosine | 空集合 | 候选中位数 | 候选 p90 | 超过 4,096 个的 turn |
| ---: | ---: | ---: | ---: | ---: |
| 0.600 | 3 | 3,211.5 | 9,410.8 | 85 |
| 0.625 | 4 | 1,005 | 5,215 | 26 |
| 0.650 | 4 | 464.5 | 2,393.7 | 2 |
| 0.675 | 7 | 186 | 988.1 | 0 |
| 0.700 | 15 | 47.5 | 387 | 0 |

自然语言和 simulator 的 Top-5 平均分也存在明显偏移：

```text
natural median Top-5 mean   = 0.7229
simulator median Top-5 mean = 0.7517
```

所以固定绝对 cosine 会把 query 类型差异错误地变成候选规模差异。目前没有证据支持冻结一个绝对阈值。

### 4.2 Query-relative window 更合适

本轮测试：

$$
\tau_{keep}(q)=\operatorname{mean}(Top5(q))-\Delta
$$

全体 192 turn：

| Delta | 空集合 | 候选中位数 | 候选 p90 | 超过 4,096 | 超过 10,000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | 0 | 84 | 266.9 | 0 | 0 |
| 0.075 | 0 | 229.5 | 904.3 | 1 | 0 |
| 0.100 | 0 | 633 | 2,636.7 | 11 | 1 |
| 0.125 | 0 | 1,507.5 | 5,635.9 | 38 | 11 |

自然语言 70 turn：

| Delta | 候选中位数 | 候选 p90 | 最大候选数 |
| ---: | ---: | ---: | ---: |
| 0.075 | 261.5 | 1,352.1 | 4,644 |
| 0.100 | 617 | 3,337.4 | 17,206 |
| 0.125 | 983 | 8,800.9 | 34,780 |

`Delta=0.075` 和 `0.100` 是当前合理的继续实验区间：

- `0.075` 计算规模稳定，更接近严格 relevance support；
- `0.100` 保留更宽的语义边界，更适合观察模糊 query；
- `0.125` 已经会在宽泛 query 上产生 30k 以上候选，不适合作为当前默认值。

这仍是实验区间，不是冻结参数。

## 5. Merge threshold 分析

在 `Top-5 mean - 0.100` 的 retained set 中，每轮最多从完整分数范围等距采样 1,536 件商品。普通商品 pair 的跨 turn 中位分布为：

```text
pair cosine p50   = 0.7237
pair cosine p90   = 0.7804
pair cosine p95   = 0.8012
pair cosine p99   = 0.8472
pair cosine p99.9 = 0.9004
```

不同 merge threshold 下，能够找到至少一个可合并邻居的商品比例：

| Merge threshold | 比例中位数 | 比例 p90 | 含义 |
| ---: | ---: | ---: | --- |
| 0.850 | 66.3% | 81.1% | 太宽，已经在合并普通相似商品 |
| 0.875 | 41.2% | 59.9% | 强聚合 |
| 0.900 | 22.5% | 39.2% | 中等语义聚合 |
| 0.925 | 11.3% | 23.9% | 保守聚合 |
| 0.940 | 8.1% | 17.3% | 更接近近重复消除 |
| 0.950 | 5.9% | 14.0% | 非常保守 |

在自然语言完整 retained set 上，实际 greedy merge 的中位缩减比例：

| Merge threshold | Delta 0.075 | Delta 0.100 |
| ---: | ---: | ---: |
| 0.875 | 22.36% | 22.42% |
| 0.900 | 10.96% | 11.89% |
| 0.925 | 5.81% | 6.03% |
| 0.940 | 3.98% | 4.02% |
| 0.950 | 2.86% | 2.84% |

如果目标是只消除近重复 listing，`0.940` 是当前更安全的中心点；如果希望把语义相近的 SKU 也视作同一个选择方向，则应继续测试 `0.900–0.925`。仅靠数值分布不能决定产品语义边界，需要抽查实际被合并的商品对。

## 6. 聚合后 coherence 的实际结果

三个自然语言模糊→具体对照：

### Wedding dress

`Delta=0.100, merge=0.940`：

```text
retained candidates       3334 -> 10
semantic representatives 3155 -> 8
coherence               0.1199 -> 0.3719
```

方向符合预期。

### Carry-on luggage

`Delta=0.100, merge=0.940`：

```text
retained candidates      639 -> 15
semantic representatives 625 -> 15
coherence              0.2338 -> 0.2970
```

方向符合预期。

### Running shoes

`Delta=0.100, merge=0.940`：

```text
retained candidates       1696 -> 107
semantic representatives 1516 -> 106
coherence               0.2135 -> 0.1701
```

候选支持空间和去重代表数量都大幅收缩，但原 coherence 反而下降。所有 12 个完整配置（两个 keep delta，raw 加五个 merge threshold）在三个 clarity pair 上均为 `2/3`，没有一组参数修复 running-shoes failure。

这说明：

1. 全量阈值筛选成功观察到了相关空间收缩；
2. 高相似聚合成功降低了重复 listing 的影响；
3. 原 coherence 只衡量剩余代表的平均方向一致性；
4. 如果最终只保留 coherence，而丢掉 representative count，running-shoes 中最明显的收缩信号会被丢失。

## 7. 当前结论

该方案在工程上可行，CUDA 足以支撑 50k 全量打分和当前规模的交叉矩阵。它也比固定 Top-80 更诚实：不再强行补满无关尾部，并显式降低重复商品的权重。

但本轮数据不支持下面这个更强结论：

> threshold + merge 之后，原有 coherence 就可以单独充当 Intent Transparency。

当前最重要的观测不是某个最佳阈值，而是 retained semantic representative count 与 coherence 提供了不同信息：

- representative count 表达“还有多少受支持的语义选择”；
- coherence 表达“这些选择的方向是否集中”。

下一轮若继续此路线，应先检查如何定义一个同时保留“剩余支持规模”和“空间分散程度”的 semantic volume，而不是继续微调 merge threshold 试图修复 coherence 本身。

## 8. 可复现命令

全体 192 turn 的 score distribution 与 pairwise sample：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/analyze_semantic_thresholds.py
```

自然语言 70 turn 的完整 retained-set merge：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/analyze_semantic_thresholds.py `
  --cohort natural `
  --run-merge `
  --output artifacts/retrieval/semantic-space-merge-natural-v0.json `
  --markdown artifacts/retrieval/semantic-space-merge-natural-v0.md
```
