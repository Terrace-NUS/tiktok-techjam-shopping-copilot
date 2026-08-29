# Fuzzy Intent Volume 扩大实验 v2

状态：实验完成。算法结构得到复现，但 Query Understanding 的批量撤销能力与数值刻度仍未冻结。

## 1. 这次扩大了什么

第一轮自然语言实验只有 24 段双轮对话、48 个 turn。v2 保留全部旧样本并新增 36 段，最终包含：

| 预期状态变化 | 对话数 |
|---|---:|
| 新增条件，空间应变窄 | 36 |
| 撤销条件，空间应变宽 | 10 |
| 只改展示或无关信息，空间应稳定 | 7 |
| 完全换目标，只观察空间迁移 | 7 |
| 合计 | **60** |

共 130 个用户 turn，其中 10 段是三轮渐进式收窄，因此除了比较首尾，还能检查 20 个相邻步骤。

语言与领域：

- 英语 46 段、简体中文 8 段、中英混合 6 段；
- 服装、鞋、珠宝、箱包、手表、配饰和跨品类换目标；
- 覆盖否定、预算、尺码、材质、颜色、feature、批量撤销、无关闲聊和展示指令；
- 没有 catalog target、simulator hidden state 或手写的预期商品。

## 2. 完整链路成功率

本轮重新真实运行：

```text
用户自然语言
→ DeepSeek V4 Flash 原生 tool call
→ 本地修复与 materialization
→ 完整 Session Context
→ Query Compiler
→ 50k catalog Probe
→ Fuzzy Intent Volume
```

结果：

| 项目 | 数量 |
|---|---:|
| 选择的 turn | 130 |
| QU 与 pipeline 成功 | **120** |
| `repair_exhausted` | 9 |
| 因前一轮失败而跳过 | 1 |
| 完整成功的对话 | **51/60** |

Turn 级成功率为 **92.3%**，Wilson 95% 区间约为 86.4%–95.8%。旧版为 46/48；样本扩大和难度增加后，QU 失败率有所上升。

按状态变化拆开：

| 类型 | 完整成功对话 |
|---|---:|
| narrower | 33/36 |
| broader | **4/10** |
| stable | 7/7 |
| override | 7/7 |

9 段失败对话全部是 `repair_exhausted`。其中 6 段来自 broader 样本，主要表达是“一次撤销颜色、材质、价格、feature 等多项条件”。

因此扩大实验后的第一个新结论不是指标问题，而是：

> 当前端到端系统的主要短板已经变成 QU 批量撤销。指标只能忠实读取成功提交的 Session Context，无法替上游修复失败状态。

## 3. 参与比较的算法

与第一轮相同：

- `hybrid`：所有可解析 hard masks 严格求交；
- `soft_hybrid`：hard facet 满足时 membership 为 1，违反时为 $\epsilon$，再与开放语义因子相乘；
- `fuzzy`：goal 与所有 preference 都使用独立 semantic membership；
- `qsem`：只使用 compiler 生成的完整 `q_sem`；
- `anchored`：原子语义乘积再加一个较弱的完整 `q_sem` 锚点；
- `*_shuffled`：保持分数分布，但随机打乱分数与商品的对应关系。

剩余体积仍定义为：

$$
N_t=\sum_i \frac{a_t(i)}{d_i}
$$

其中 $a_t(i)$ 是全部条件的 Product-of-Experts 兼容度，$d_i$ 是该商品在全 catalog embedding 中的局部密度。

## 4. 状态变化方向

51 段完整对话中，7 段 override 不规定数值升降，剩余 44 段可评分。

| 方法 | 首轮到末轮符合预期 | 三轮样本的相邻收窄 |
|---|---:|---:|
| 严格 hard 交集 | 43/44 | 20/20 |
| 柔性结构化 PoE | **44/44** | **20/20** |
| 纯原子语义 PoE | **44/44** | **20/20** |
| q_sem 锚点 + 原子 PoE | **44/44** | **20/20** |
| 完整 `q_sem` | 22–24/44 | 8/20 |
| 随机打乱原子语义 | **44/44** | **20/20** |

柔性结构化版 44/44 的 Wilson 95% 下界约为 92.0%；渐进式 20/20 的下界约为 83.9%。

严格交集唯一失败的是 `b07_release_dress_exclusions`：第一轮和撤销条件后的第二轮都为空，体积保持 `0 → 0`，无法表现“变宽”。

### 方向满分应该怎样解释

随机打乱商品以后，原子 PoE 仍然是 44/44 和 20/20。这证明：

- 把 Session Context 拆成独立条件并做乘法求交，确实能稳定表达“增加条件 / 删除条件”；
- 但方向满分主要是状态代数的性质，不是 embedding 商品理解准确率；
- 语义是否正确必须用商品条件命中率单独检查。

## 5. 商品相关性

120 个成功状态中，有 97 个包含至少一个可由当前 evidence 验证的 hard facet。

| 方法 | Top-20 同时满足全部 hard facet | Top-20 平均满足的 facet |
|---|---:|---:|
| 柔性结构化 PoE，$\epsilon=0.01$ | **39.1%** | **81.7%** |
| 柔性结构化 PoE，$\epsilon=0.05$ | **38.7%** | **81.5%** |
| 柔性结构化 PoE，$\epsilon=0.20$ | **37.0%** | **80.6%** |
| 完整 `q_sem` | 16.3% | 59.3% |
| q_sem 锚点 + 原子 PoE | 12.1%–14.6% | 50.6%–55.2% |
| 纯原子语义 PoE | 11.9%–14.2% | 50.0%–54.6% |
| 随机语义对照 | 1.4%–1.7% | 25.6%–26.2% |

使用暂定演示值 $\epsilon=0.05$ 时：

- 在 97 个可比较状态里，柔性结构化版有 91 个胜过完整 `q_sem`，6 个持平；
- 平均 facet 命中率提高 22.2 个百分点；
- 按对话聚类 bootstrap 的 95% 区间为 **+19.2 到 +25.3 个百分点**。

所以结构化 evidence 的贡献不是随机波动，也不是单纯由乘法方向造成的。

## 6. 严格 hard 交集仍然不可用

97 个带可验证 hard facet 的状态中：

```text
41 个严格交集为空
空集率 = 42.3%
```

第一轮为 15/38，空集率 39.5%。扩大样本后空集比例几乎没有下降，说明这不是小样本偶然现象。

严格 hard 方案在非空状态中的 Top 商品当然是 100% hard-compliant，但它无法描述四成以上的状态。柔性 membership 保留了同一个商品空间，并让“只差一个条件”的商品仍有小但非零的质量，不需要额外 fallback pool。

## 7. 分语言和领域观察

使用 `soft_hybrid`、$\epsilon=0.05$：

| 语言 | 可验证状态 | 平均 facet 命中率 |
|---|---:|---:|
| 英语 | 73 | 81.5% |
| 中文 | 13 | 75.3% |
| 中英混合 | 11 | 88.9% |

方向结果在完整可评分对话上分别为英语 35/35、中文 5/5、中英混合 4/4。

中文与混合语言样本仍然很小，不能据此声称混合语言优于英语。中文较低值得继续观察，可能同时受 QU 规范化、商品英文 evidence 和样本领域分布影响。

按领域的平均 facet 命中率：

| 领域 | 命中率 |
|---|---:|
| apparel | 75.3% |
| jewelry | 78.0% |
| luggage | 79.8% |
| footwear | 82.4% |
| accessories | 82.5% |
| handbags | 85.0% |
| cross-domain override states | 88.4% |
| watches | 94.4% |

手表只有 3 个可验证状态，不能单独做强结论。服装是当前较明显的弱项，原因包括 category / gender evidence 覆盖不足，以及尺寸、款式和 feature 同时出现时没有完整交集。

## 8. 最差案例说明了什么

柔性版的低命中状态主要包括：

- `b03_release_jacket_constraints` 第二轮：只保留“女童外套”，但旧 goal 与 gender evidence 的覆盖问题仍在，facet 命中 20%；
- `b07_release_dress_exclusions` 第二轮：保留 size 8 与 midi dress，但严格交集仍为空，命中 50%；
- 男士正式 Oxford、皮带、儿童校鞋、网球手链等多条件组合：大量字段只能从稀疏文本 evidence 推断；
- 服装中的 size、department、style、feature 经常不能在同一 listing 上得到完整证据。

这说明 $D_t$ 诊断仍然必要。`T_t` 很高可能表示意图真的很具体，也可能表示 catalog evidence 稀疏或条件组合在数据中没有完整覆盖。

## 9. 与第一轮对比

| 指标 | 第一轮 | 扩大后 |
|---|---:|---:|
| 自然语言 turn | 48 | **130** |
| QU 成功 | 46/48 | **120/130** |
| 可评分完整 pair | 19 | **44** |
| 柔性结构化方向 | 19/19 | **44/44** |
| 完整 q_sem 方向 | 9–11/19 | **22–24/44** |
| 柔性版平均 facet 命中 | 82.2%–83.9% | **80.6%–81.7%** |
| 严格交集空集率 | 39.5% | **42.3%** |

扩大后商品命中率略有下降，但保持在约 81%，仍明显高于 `q_sem` 的 59%。方向、纯语义失败模式和严格空集率都与第一轮一致，因此核心结论得到复现，而不是被新样本推翻。

## 10. 当前可以与不可以得出的结论

可以得出：

1. `q_sem → cosine threshold → 分散度` 不适合作为主指标；扩大后仍只有约一半方向正确。
2. Session Context 条件分解加 Product of Experts 能稳定表达状态空间的增加与删除。
3. structured facet 应使用 evidence membership，而不应全部退化成短文本 embedding。
4. 绝对 hard intersection 在当前 50k 数据和 evidence 上空集率约四成，不能作为唯一空间。
5. 柔性结构化版是当前最好的工程候选：方向稳定、商品条件命中显著更高，而且没有第二个 fallback pool。

还不能得出：

1. 不能说 44/44 证明 embedding 准确；随机对照同样有方向满分。
2. 不能说 broader 端到端已经可靠；10 段中只有 4 段完整通过 QU。
3. 不能冻结 `density temperature`、semantic threshold、$\epsilon$ 或最终 0–1 的 `T_t` 刻度。
4. 不能把高 `T_t` 自动解释为“用户意图一定清晰”，必须结合 evidence coverage、空交集和 QU 状态。

## 11. 当前推荐结构

```text
完整 Session Context
    → 可验证 structured facet：满足为 1，违反为 ε
    → goal / soft / open-text preference：semantic membership
    → Product of Experts 得到商品兼容度
    → inverse-density weight 给近似 listing 降权
    → 全 catalog 加权体积 N_t
    → T_t 用于故事展示和策略控制
    → D_t 同时公开 QU、evidence 与空交集诊断
```

`epsilon=0.05` 可以继续作为 demo 中间值，但只是视觉折中。`0.01` 的商品命中略高，三个候选值的排名差异很小，而最终 `T_t` 刻度差异很大。

## 12. 下一步优先级

在继续调 Intent Volume 公式之前，优先修 QU 的批量撤销：

1. 为 complete-state tool contract 增加专门的 multi-release examples；
2. 记录 repair 失败的结构化原因，而不只保留 `repair_exhausted`；
3. 对本轮 6 个 broader 失败案例做固定回归；
4. 修复后原样重跑 v2，目标是至少 9/10 broader 对话完整提交；
5. 再用完整 broader 集判断空间是否可靠回升。

## 13. 可复现产物

- 扩展 suite：`config/query_understanding/intent-space-natural-prompts-v2.json`
- suite 构建器：`scripts/query_understanding/build_intent_space_suite_v2.py`
- 完整 QU / Session Context / compiler / Probe 日志：`artifacts/retrieval/qu-to-probe-intent-space-natural-v2.json`
- 270 组参数与对照的完整结果：`artifacts/retrieval/fuzzy-intent-volume-natural-v2.json`
- 自动摘要：`artifacts/retrieval/fuzzy-intent-volume-natural-v2.md`
- 实验程序：`scripts/retrieval/evaluate_fuzzy_intent_volume.py`

实验没有修改原始 catalog、semantic release 或 dense index。
