# 高维语义空间指标对比 v0

> 后续扩大实验见 [Intent-space 扩大测试 v1](expanded-natural-evaluation-v1.md)。
> v1 增加了 24 段自然语言轨迹、三轮 simulator 截断，以及忽略 hard mask 的
> semantic-only 对照。v0 的三组自然样本结论不能单独用于冻结算法。

- 状态：**完整实验；候选方案已排序，尚未冻结 runtime metric**
- 日期：**2026-08-29**
- 前置实验：[Thresholded Semantic Space 实验 v0](thresholded-semantic-space-experiment-v0.md)
- 实验脚本：`scripts/retrieval/analyze_semantic_thresholds.py`
- 全量结果：`artifacts/retrieval/semantic-dispersion-metrics-v0.json`
- 最新 simulator-only 结果：`artifacts/retrieval/semantic-dispersion-metrics-simulator-other-v0.json`

## 1. 公平比较设置

所有指标使用相同的：

- accepted Session Context；
- `q_sem`；
- Hard Mask；
- 50,000 件商品 BGE 向量；
- query-relative relevance window；
- 商品向量 normalization。

比较四个候选空间：

```text
Top-5 mean - 0.075，未聚合
Top-5 mean - 0.075，cosine >= 0.94 greedy merge
Top-5 mean - 0.100，未聚合
Top-5 mean - 0.100，cosine >= 0.94 greedy merge
```

自然语言主判据是三个明确的模糊→具体 pair。用户撤销条件、切换目标和其他 override 不要求单调。

Simulator 补充测试严格使用最新的：

```text
qu-to-probe-simulator-other-16x4-audit-v2.json
```

它只有 buying / browsing，共16个 task，每个四轮、64/64 turn 全链路成功。下面比较每个 task 的第一轮与第四轮。旧 full suite 混有 intent-override 和 boundary，不用于算法选择。

## 2. 被测试的指标

### 2.1 当前 coherence

当前 mean-centered average pairwise cosine。清晰方向为数值升高。

### 2.2 Pairwise angular distance

商品对距离：

$$
d_{ij}=\sqrt{2-2\cos(x_i,x_j)}
$$

测试中位数与 p90。清晰方向为距离降低。

### 2.3 Covariance spectrum

对候选向量协方差矩阵的特征值测试：

- total variance；
- stable rank；
- Rényi-2 effective rank；
- Shannon effective rank。

清晰方向为有效维度降低。

### 2.4 Regularized log-det

$$
V_\beta=\log\det(I+\beta C)
$$

测试 `beta = 10, 50, 100, 500`。清晰方向为体积降低。

### 2.5 kNN entropy proxy

测试 `k = 3, 5, 10` 的邻居距离与 Kozachenko-Leonenko 风格 entropy proxy。清晰方向假定为 entropy 降低。

### 2.6 Kernel Effective Number

$$
K_{ij}=\exp\left(\frac{\cos(x_i,x_j)-1}{\tau}\right)
$$

$$
N_{eff}=\frac{(\operatorname{tr}K)^2}{\operatorname{tr}(K^2)}
=\frac{n^2}{\sum_{i,j}K_{ij}^2}
$$

测试：

```text
tau = 0.025, 0.050, 0.075, 0.100, 0.150, 0.200
```

它表示当前候选集合相当于多少个彼此不同的语义方向。清晰方向为有效方向数降低。

## 3. 主配置结果

主配置：

```text
keep delta = 0.100
merge threshold = 0.940
```

| 指标 | 自然语言具体方向正确 | Simulator 第4轮更收缩 |
| --- | ---: | ---: |
| 当前 coherence | 2/3 | 6/14 |
| Representative count | 3/3 | 13/16 |
| Kernel effective number, tau 0.025 | 3/3 | 13/16 |
| Kernel effective number, tau 0.050 | 3/3 | 13/16 |
| Kernel effective number, tau 0.075 | 3/3 | 10/16 |
| Kernel effective number, tau 0.100 | 3/3 | 10/16 |
| Covariance Rényi-2 effective rank | 3/3 | 7/14 |
| Covariance Shannon effective rank | 3/3 | 11/14 |
| Covariance stable rank | 3/3 | 8/14 |
| Log-det, beta 100 | 3/3 | 8/14 |
| Median pairwise angular distance | 3/3 | 6/14 |
| kNN entropy, k=5 | 0/3 | 3/14 |

Simulator 中 covariance、log-det、pairwise 与 kNN 只有14个可比较 task，是因为两个末轮候选/代表不足两件。这不是把缺失值计算成失败；Kernel Effective Number 与 representative count 对单点集合仍然有定义，因此是16个。

## 4. 三个自然语言 clarity pair

以下使用 `delta=0.100, merge=0.940`。

### 4.1 Kernel Effective Number，tau=0.050

| Pair | 模糊 | 具体 | 方向 |
| --- | ---: | ---: | --- |
| Wedding dress | 2,608.25 | 7.96 | 正确 |
| Running shoes | 1,244.14 | 104.73 | 正确 |
| Carry-on | 600.20 | 14.92 | 正确 |

Running-shoes failure 被明确修复。它没有要求具体集合内部的平均 cosine 必须更高，而是识别到有效语义支持从约1,244个方向缩小到约105个方向。

### 4.2 Covariance Shannon effective rank

| Pair | 模糊 | 具体 | 方向 |
| --- | ---: | ---: | --- |
| Wedding dress | 176.37 | 6.66 | 正确 |
| Running shoes | 138.86 | 67.06 | 正确 |
| Carry-on | 140.32 | 12.54 | 正确 |

它也修复了三个 pair，但 simulator 为 `11/14`，弱于 Kernel。

### 4.3 Log-det，beta=100

| Pair | 模糊 | 具体 | 方向 |
| --- | ---: | ---: | --- |
| Wedding dress | 25.32 | 8.66 | 正确 |
| Running shoes | 23.97 | 22.36 | 正确但差距很小 |
| Carry-on | 25.18 | 14.12 | 正确 |

Running-shoes 只减少约1.61，参数敏感性较高。

### 4.4 Pairwise distance

| Pair | 模糊 | 具体 | 方向 |
| --- | ---: | ---: | --- |
| Wedding dress | 0.7530 | 0.6407 | 正确 |
| Running shoes | 0.7474 | 0.7470 | 几乎不变 |
| Carry-on | 0.7698 | 0.7460 | 正确 |

虽然形式上是 `3/3`，running-shoes 的变化只有约 `-0.00048`，不能认为已经可靠修复。

### 4.5 kNN

kNN 系列在自然语言 clarity pair 上为 `0/3`。候选数降低后，有限样本中的近邻距离反而变大，384维 entropy proxy 被样本密度效应主导。当前数据明确不支持使用 kNN 作为主指标。

## 5. Hard merge 是否必要

Kernel Effective Number 本身已经会对相似商品进行软聚合：

- 完全相同的商品几乎不增加有效方向数；
- 高度相似的商品只增加一部分；
- 明显不同的商品接近增加一个方向。

在最新 simulator 测试中：

| 配置 | tau=0.050 首末轮更收缩 |
| --- | ---: |
| delta 0.075，raw | 12/16 |
| delta 0.075，merge 0.94 | 12/16 |
| delta 0.100，raw | 13/16 |
| delta 0.100，merge 0.94 | 13/16 |

`0.94` hard merge 没有提高 Kernel 结果。它还引入一个不连续边界，并增加工程复杂度。因此，如果选择 Kernel Effective Number，当前数据更支持先取消 hard merge，只保留 Kernel 的 soft deduplication。

## 6. Simulator 的三个 Kernel failure

在 `delta=0.100, raw, tau=0.050` 下，13/16 task 收缩。三个未收缩 task 是：

```text
official_simulator_other_browsing_006
official_simulator_other_browsing_007
official_simulator_other_buying_007
```

共同特征：

1. 后两轮没有继续改变 Session Context；
2. 新增内容主要是商品原始 feature/material 长句；
3. Hard Mask 的 eligible pool 虽然缩小，但 query-relative threshold 下的语义支持方向反而增加；
4. 两个 browsing case 首轮 category 只存在于 goal 文本，没有成为结构化 category mask。

因此这三条不是 Kernel 数值失效，而是 retained relevance support 的构造仍有问题。下一轮应检查 keep-threshold 与 category grounding，而不是用后处理强行让指标下降。

## 7. 当前排序

### 第一名：Kernel Effective Number

建议继续验证：

```text
keep delta = 0.100
hard semantic merge = off
kernel tau = 0.050
```

理由：

- 自然语言 clarity `3/3`；
- 最新 buying/browsing simulator `13/16`；
- 修复 running-shoes；
- 同时表达支持数量和语义相似度；
- 自动软去重；
- CUDA 上只需 elementwise exponential 与 reduction；
- 可以分块计算，不需要保留整个相似度矩阵。

`tau=0.025` 也达到 `13/16`，但数值已经非常接近 representative count，语义折扣较弱；`tau=0.050` 保留了更明显的连续语义作用，因此更适合作为下一轮中心点。

### 第二名：Covariance Shannon Effective Rank

自然语言表现良好，且不依赖候选数量的直接计数；但 simulator 稳定性较弱，更适合作为 shape diagnostic。

### 第三名：Regularized log-det

能够表达高维体积，但对 `beta` 和小样本更敏感，running-shoes margin 较小。

### 不建议作为主指标

- 当前 coherence；
- median/p90 pairwise distance；
- kNN entropy。

## 8. 性能

完整 192-turn 多指标参数扫描：

```text
query-to-50k CUDA matrix = 0.039 seconds
full merge + all metric suites = 56.47 seconds
```

这个56.47秒同时计算了四个候选空间、五个 hard-merge threshold、六个 Kernel 温度、四个 log-det beta、三个 kNN 和多种 covariance 指标，不代表最终单一 Kernel metric 的在线开销。

最终 Kernel 版本可以对候选矩阵分块累计：

$$
\sum_{i,j}K_{ij}^2
$$

因此不需要构造或保存完整 $n\times n$ 矩阵。

## 9. 结论边界

本轮足以淘汰明显不合适的方法，并选出 Kernel Effective Number 作为下一轮主候选；但三个自然语言 pair 和16个 toy-simulator task 仍不足以冻结最终 `tau`、keep delta 或 Transparency 映射。

下一步应针对 Kernel 方案补：

1. 更大的自然语言 vague/refine/specific/override suite；
2. `delta` 与 `tau` 的联合稳定区，而不是单点最优；
3. category grounding 后重跑三个 simulator failure；
4. 将原始 `N_eff` 映射为 demo 可解释的剩余意图空间与 `T_t`。
