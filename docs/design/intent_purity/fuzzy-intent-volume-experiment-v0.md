# Fuzzy Intent Volume 实验 v0

状态：第一轮实验完成；可以选择算法方向，但不冻结数值参数。更大的 60 段 / 130 turn 复现实验见 [扩大实验 v2](expanded-fuzzy-intent-volume-v2.md)。

## 1. 这轮实验想回答什么

上一版直接把完整 `q_sem` 与 5 万件商品做向量相似度，再对留下的商品计算分散程度。它有一个根本问题：查询写得更具体以后，整句话也变长了，cosine 分布会整体移动，因此“条件更多”不一定留下更少的商品。

这轮测试两个替代想法：

1. 把 Session Context 拆成多个独立条件，每个条件分别计算商品兼容度，再用乘法求交集。
2. 用全 catalog 的向量密度给重复或近似商品降权，避免同一种热门商品的许多 listing 把空间虚假放大。

## 2. 实验数据

自然语言集：

- 24 段双轮对话，共 48 个用户 turn；
- 46 个 turn 完成 `DeepSeek QU → Session Context → compiler`；
- 2 个 turn 因 `repair_exhausted` 保留为 QU 失败；
- 得到 19 个可以判断方向的完整 pair：15 个应变窄、2 个应变宽、2 个应保持稳定；
- 另有 3 个完全换目标的 pair，只观察，不规定升降方向。

官方 toy simulator 集：

- 只使用 buying / browsing；
- 只发送 `ask_attribute="other"`；
- 16 个 task，每个保留前三轮，共 48 个 turn；
- 比较 turn 1 与 turn 3。

Catalog 与运行环境：

- 50,000 件商品；
- 384 维商品向量；
- 全 catalog 计算在 RTX 4070 Ti 上完成；
- 实验只读取 catalog、semantic release、dense index 和已有 QU 日志，不修改比赛数据。

## 3. 比较了哪些算法

### 3.1 整句语义 `qsem`

把 compiler 生成的完整 `q_sem` 当成一个向量条件。

这是旧思路的直接对照。

### 3.2 原子语义乘积 `fuzzy`

将 goal 和每条 preference 拆成独立文本条件。商品 $i$ 对条件 $c$ 的 membership 为：

$$
m_{ic}=\sigma\left(\frac{s_{ic}-b_c}{\tau_m}\right)
$$

其中 $s_{ic}$ 是 cosine，$b_c$ 是该条件在全 catalog 上的相似度分位点。排除条件使用 $1-m_{ic}$。

所有条件通过 Product of Experts 合并：

$$
a_i=\prod_c m_{ic}^{\lambda_c}
$$

硬偏好权重为 1，软偏好权重为 0.5。

### 3.3 整句锚点加原子乘积 `anchored`

在 `fuzzy` 上再乘一个较弱的完整 `q_sem` membership。它用于测试整句语义能否帮助短条件找到更相关的商品。

### 3.4 严格结构化交集 `hybrid`

每个可解析 hard facet 使用 evidence mask，并把所有 mask 直接求交集。交集外 membership 为 0。

### 3.5 柔性结构化交集 `soft_hybrid`

结构化 hard facet 不再直接删除商品：

$$
m^{hard}_{ic}=\begin{cases}
1,&\text{商品满足条件}\cr
\epsilon,&\text{商品不满足或证据不足}
\end{cases}
$$

随后与 goal、软偏好以及无法结构化解析的条件共同相乘。

本轮扫描：

$$
\epsilon\in\{0.01,0.05,0.20\}
$$

它仍然只有一个商品池，不需要另建 fallback pool。存在完全合格商品时，它们自然占优；不存在完整交集时，只差一个条件的商品仍可保留少量质量。

### 3.6 随机打乱对照

`fuzzy_shuffled`、`qsem_shuffled` 和 `anchored_shuffled` 保留完全相同的分数分布，但随机打乱“分数属于哪件商品”。

如果真实算法与随机对照一样好，说明结果只是公式结构造成的，不是语义真的找对了商品。

## 4. 密度修正与最终数值

商品 $i$ 的全 catalog 密度为：

$$
d_i=\sum_j \exp\left(\frac{\cos(x_i,x_j)-1}{\tau_d}\right)
$$

剩余意图体积：

$$
N_t=\sum_i \frac{a_i}{d_i}
$$

展示用透明度：

$$
T_t=1-\frac{\log(1+N_t)}{\log(1+N_{catalog})}
$$

这不是对 Top-K 商品再做两两大矩阵，而是预先为每件商品计算一次密度。每轮只需要对 5 万个 membership 做加权求和。

## 5. 怎样判断结果是否真的有效

使用三种互补检查：

1. **方向正确率**：新增限制时 $N_t$ 是否降低，撤销限制时是否升高，无关对话时是否不变。
2. **Top-20 全条件命中率**：前 20 件商品中，有多少同时满足所有可验证 hard facet。
3. **Top-20 平均 facet 命中率**：对前 20 件商品逐项检查 category、color、material、price、size 等条件，计算平均满足比例。

第三项很重要。若 catalog 中没有一件商品同时满足所有条件，第二项必然是 0；平均 facet 命中率仍能区分“只差一项”和“完全搜偏”。

## 6. 方向结果

### 6.1 自然语言集

| 算法 | 符合预期的 pair |
|---|---:|
| 严格结构化交集 | **19/19** |
| 柔性结构化交集 | **19/19** |
| 原子语义乘积 | **19/19** |
| 整句锚点 + 原子乘积 | **19/19** |
| 完整 `q_sem` | 9–11/19 |
| 随机打乱的原子乘积 | **19/19** |

### 6.2 Toy simulator

| 算法 | turn 1 → turn 3 变窄 |
|---|---:|
| 严格结构化交集 | **16/16** |
| 柔性结构化交集 | **16/16** |
| 原子语义乘积 | **16/16** |
| 整句锚点 + 原子乘积 | **16/16** |
| 完整 `q_sem` | 6–7/16 |
| 随机打乱的原子乘积 | **16/16** |

这里必须诚实解释：原子乘积的方向满分来自它的集合求交结构。新增一个 $0\ldots1$ 的 factor，体积天然不会增加；随机打乱商品以后也一样。因此方向满分证明“状态更新的数学结构对了”，但不证明 embedding 找对了商品。

## 7. 商品相关性结果

### 7.1 自然语言集

38 个 turn 含有至少一个可验证 hard facet。

| 算法 | Top-20 同时满足全部 hard facet | Top-20 平均满足的 facet |
|---|---:|---:|
| 柔性结构化交集 | **41.6%–43.4%** | **82.2%–83.9%** |
| 完整 `q_sem` | 22.0% | 63.7% |
| 整句锚点 + 原子乘积 | 19.3%–20.5% | 55.0%–58.7% |
| 原子语义乘积 | 19.1%–19.6% | 54.4%–57.3% |
| 随机语义对照 | 3.3%–4.1% | 25.7%–26.9% |

严格结构化交集在有结果的 turn 上当然是 100%，但 38 个 hard-facet turn 中有 **15 个交集为空**。因此该 100% 只覆盖剩下 23 个 turn，不能与其他算法直接比较。

### 7.2 Toy simulator

39 个 turn 含有可验证 hard facet，且严格交集都非空。

| 算法 | Top-20 同时满足全部 hard facet | Top-20 平均满足的 facet |
|---|---:|---:|
| 柔性结构化交集 | **86.5%–87.9%** | **94.1%–94.8%** |
| 原子语义乘积 | 43.5%–45.3% | 61.1%–63.7% |
| 整句锚点 + 原子乘积 | 44.0%–45.4% | 61.7%–64.2% |
| 完整 `q_sem` | 39.4% | 59.1% |
| 随机语义对照 | 4.5%–6.9% | 11.1%–13.5% |

纯语义方法明显优于随机对照，说明 embedding 确实包含有效信息；但自然语言样本上的绝对准确度仍不足以让它单独负责颜色、价格、材质等明确条件。

## 8. 柔性结构化版本的真实例子

以下使用一个便于观察的临时配置：

```text
density temperature = 0.025
semantic quantile = 0.85
semantic temperature = 0.06
hard mismatch floor = 0.05
```

| 对话 | 第一轮 N / T | 第二轮 N / T | 现象 |
|---|---:|---:|---|
| 运动鞋：泛泛想恢复运动 → 女款 8 码宽楦白色公路跑鞋 | 2832.05 / 0.246 | 0.67 / 0.951 | 明显收窄；严格交集为空 |
| 行李：需要 luggage → 海军蓝 20 寸硬壳 TSA 登机箱 | 14719.99 / 0.090 | 3.40 / 0.860 | 明显收窄；严格交集为空 |
| 红色皮质闭口高跟鞋 → 只保留尺码和高跟鞋 | 0.46 / 0.964 | 199.61 / 0.497 | 撤销条件后重新变宽 |
| 黄色防水女童雨衣 → 只要求女童外套 | 0.57 / 0.957 | 662.12 / 0.384 | 旧算法失败的 broader case 已恢复正确方向 |
| 只改变展示要求 | 34.83 / 0.661 | 34.83 / 0.661 | Session Context 未变，数值完全不变 |

当“男款轻量防水连帽深绿色雨衣、无羊毛、100 美元内”的严格交集为空时，柔性版本的前列商品包括：

- `Common District Men's Waterproof Lightweight Rain Jacket ... Hooded Raincoat`
- `Wantdo Men's Waterproof Raincoat Light Hooded Windbreaker ...`
- `Men's All-Terrain ... Waterproof ... Jackets for Men`

它们不是全部条件完美命中，但明显处在正确的购物区域。这正是非零惩罚相对直接清空结果的价值。

## 9. 参数扫描揭示的风险

### 9.1 密度温度极其敏感

| $\tau_d$ | 50k catalog 的有效体积 |
|---:|---:|
| 0.025 | 38123.16 |
| 0.050 | 914.26 |
| 0.100 | 33.51 |

三个值相差三个数量级。当前数据无法证明 38,123、914 或 34 哪一个才是真实的“意图方向数”。因此密度修正的概念可保留，但 `tau_d` 不能冻结。若现在需要做 demo，`0.025` 最不容易把整个 catalog 过度压成几十个方向。

### 9.2 Hard mismatch floor 不影响 Top 排名，却显著影响 T

在 $\epsilon=0.01,0.05,0.20$ 之间，最佳 Top-20 facet 命中率几乎不变；但同一个具体跑鞋状态的 $T$ 分别约为：

```text
0.995 / 0.951 / 0.766
```

所以 `0.05` 只能作为视觉上不过度极端的中间演示值，不能称为已校准真值。

### 9.3 原子语义存在重复计票和 veto 问题

一个跑鞋状态可能同时产生 goal、category、use case、gender、size、wide、road-running 等十个短 factor。它们并不独立，而且某个非常泛化或质量差的短语也会通过乘法强烈否决商品。

这解释了为何自然语言集上完整 `q_sem` 的商品相关性反而高于纯原子语义乘积，而原子乘积的体积方向更稳定。

## 10. 当前结论

最可行的数学结构不是“完整 query 向量 → 单阈值 → Top-K 分散度”，也不是“所有 facet 都交给 embedding”。当前数据支持：

```text
Session Context
    → 每个可靠 structured facet 产生一个柔性 evidence membership
      满足 = 1；不满足或未知 = ε；不直接清空
    → goal、软偏好、开放文本产生 semantic membership
    → Product of Experts 得到每件商品的整体兼容度 a_i
    → inverse-density weight 对重复 listing 软降权
    → 对全 catalog 求和得到剩余体积 N_t
    → 映射成用于展示和策略控制的 T_t
```

这套结构同时保留了故事的核心：

> 用户每提供一个有效条件，就削弱一部分仍然合理的购物空间；撤销条件时，这部分空间重新获得质量；换目标时，整个空间可以迁移或剧变。

但现在只冻结方向，不冻结参数：

- 可以冻结“按条件分解、柔性求交、密度降重、全 catalog 求体积”的结构；
- 不能冻结 `density temperature`、semantic quantile、semantic temperature、hard mismatch floor 或最终 0–1 映射；
- 纯原子 embedding 不应替代 structured facet evidence；
- `T_t` 必须附带诊断信息，例如严格交集是否为空、evidence 覆盖率和 QU 状态。

## 11. 可复现实验产物

- 实验程序：`scripts/retrieval/evaluate_fuzzy_intent_volume.py`
- 自然语言完整结果：`artifacts/retrieval/fuzzy-intent-volume-natural-v0.json`
- 自然语言摘要：`artifacts/retrieval/fuzzy-intent-volume-natural-v0.md`
- Simulator 完整结果：`artifacts/retrieval/fuzzy-intent-volume-simulator-16x3-v0.json`
- Simulator 摘要：`artifacts/retrieval/fuzzy-intent-volume-simulator-16x3-v0.md`
- 全 catalog 密度缓存：`artifacts/retrieval/intent-volume-density-v0.npz`

每个 JSON turn 都保留用户原话、goal、原子 factors、完整 `q_sem`、严格交集数量、Top 商品 ASIN、每组参数的 $N_t$、$T_t$ 与 Top-20 条件命中率。
