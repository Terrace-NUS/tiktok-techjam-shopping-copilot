# Intent-space 扩大测试 v1

状态：实验报告，不冻结算法或参数。

## 1. 为什么要扩大

旧的自然语言集虽然包含 40 段对话，但旧分析器只把 4 段 `c0*` 对话识别为
“模糊到具体”的指标样本。因此上一轮的 `3/3` 只能用于快速筛选算法，不能作为
冻结依据。

本轮新增：

```text
config/query_understanding/intent-space-natural-prompts-v1.json
```

它包含 24 段双轮对话、48 个真实自然语言输入：

| 预期变化 | 对话数 | 含义 |
|---|---:|---|
| `expected_narrower` | 16 | 用户补充了真正缩小购买空间的条件 |
| `expected_broader` | 3 | 用户明确撤销原有条件 |
| `expected_stable` | 2 | 只改变展示方式或说了无关信息 |
| `expected_override` | 3 | 用户换了目标；只观察，不要求数值升降 |

覆盖服装、鞋、箱包、手表、首饰、童装，以及英文、中文和中英混合表达。

## 2. Simulator 的轮次修正

本轮使用现有 16 个 buying / browsing task，但只保留每段的前 3 个可见 turn：

```text
--max-turn 3
```

不再把第 4 个通常没有新隐藏信息的回复用于首尾比较。原始四轮 fixture 没有被修改，
只是在运行和分析时截断，因此仍然可以复查。

## 3. 两组必须分开的实验

### 3.1 Hybrid feasible space

先应用 Session Context 编译出的 hard facet，得到结构上可行的商品；再在其中执行：

```text
score(q_sem, product) >= eligible Top-5 mean - 0.10
```

最后用 `Kernel Effective Number(tau=0.05)` 衡量剩余意图空间。

### 3.2 Semantic-only space

完全忽略 hard facet，对全部约 5 万件商品做同样的语义阈值与 Kernel 计算。

这组对照用于回答：指标表现是否真的来自 embedding 空间，而不是先被结构化筛选做完了
大部分工作。

两组实验都不修改 catalog、semantic release 或 dense index。

## 4. 端到端运行结果

48 个自然语言 turn 中：

- 46 个完成 `DeepSeek QU → Session Context → compiler → retrieval`；
- 2 个在 QU 阶段 `repair_exhausted`；
- 因此 24 段对话中有 22 段形成完整指标 pair。

两个 QU 失败是：

1. `n13_beach_swimwear` 的具体化 turn；
2. `b02_release_necklace_constraints` 的批量撤销条件 turn。

失败被保留为 QU 失败，没有计入指标的成功或失败。

## 5. Kernel 主候选的扩大结果

设置：`keep delta=0.10`、不做 hard merge、`tau=0.05`。

| 测试 | Hybrid：符合预期 | Semantic-only：符合预期 |
|---|---:|---:|
| 自然语言，应收窄 | **15/15** | 9/15 |
| 自然语言，应放宽 | 1/2 | 0/2 |
| 自然语言，应稳定 | **2/2** | **2/2** |
| Simulator，turn 1 → turn 3 | **13/16** | 3/16 |

Hybrid 在所有可用的自然语言收窄 pair 上方向正确。示例：

| 轨迹 | 第一轮 | 第二轮 |
|---|---:|---:|
| 婚礼穿搭 | 7920.20 | 2.99 |
| 重新开始运动的鞋 | 678.54 | 1.00 |
| 旅行箱 | 607.83 | 15.92 |
| 日常手表 | 926.81 | 7.06 |
| 通勤背包（中文） | 1337.36 | 104.64 |
| 家居拖鞋（中英混合） | 1359.95 | 9.86 |

两个 stable pair 的数值完全不变，说明展示数量、解释请求和明确声明“条件不变”的闲聊
没有污染商品意图空间。

三个 override pair 分别出现下降、下降和上升。这是合理现象：换目标表示意图空间移动，
新目标可能比旧目标更宽或更窄，不能强制一个方向。

## 6. 最重要的新结论：纯向量方案目前不成立

Semantic-only 版本在 6/15 个自然语言收窄 pair 上反而变大。例如：

| 轨迹 | 第一轮 | 第二轮 | 错误变化 |
|---|---:|---:|---:|
| 跑步鞋 | 889.47 | 2482.34 | +1592.87 |
| 冬季外套 | 1281.37 | 2116.43 | +835.07 |
| 男童校鞋 | 1318.46 | 4948.68 | +3630.21 |
| 珍珠耳钉 | 192.11 | 605.93 | +413.82 |

原因不是 CUDA、Kernel 公式或商品两两距离计算错误，而是候选边界本身不稳定：

1. 具体 `q_sem` 更长，包含更多可与不同商品局部匹配的词；
2. 查询变化后 cosine 分数分布整体改变；
3. `Top-5 mean - delta` 是每个查询各自的相对门槛；
4. “更具体”不保证跨过该相对门槛的商品更少。

固定 cosine 阈值也没有解决该问题。在本轮 natural semantic-only 数据上，扫描的绝对阈值
没有任何一个能让超过 2/15 的收窄 pair 得到正确的候选数量方向。

所以当前数据明确否定下面这条过于简单的链路：

```text
q_sem → 全 5 万商品 cosine → 单一阈值 → 意图空间
```

## 7. Hybrid 唯一的自然语言方向失败

失败轨迹是 `b03_release_jacket_constraints`：

```text
第一轮：黄色、防水、女童、130 码、可拆帽、$60 内的雨衣
第二轮：只要女童外套，其他都不限制
```

正确预期是空间变宽，但 Kernel 从 `17.59` 降到 `4.00`。

审计表明这不是 Kernel 的几何判断：

- QU 正确撤销了 color、feature、material、price、size；
- 但 goal 仍保留为 `yellow waterproof raincoat for child`；
- `gender=girl` 的 closed-world hard mask 只留下 4 件商品；
- 第一轮的多条件 mask 反而留下 24 件。

因此本例暴露的是上游两个问题：goal 没有随“雨衣 → 外套”更新，以及 gender evidence
覆盖不完整。指标忠实地报告了错误的 feasible space，无法自行修复它。

## 8. 各类高维指标在扩大数据上的表现

Hybrid 自然语言收窄 pair：

| 指标 | 正确 / 可用 |
|---|---:|
| Kernel Effective Number | **15/15** |
| 去重前代表数量 | **15/15** |
| Covariance Shannon effective rank | 11/11 |
| LogDet (`beta=100`) | 11/11 |
| Pairwise median angular distance | 10/11 |
| 旧 coherence | 9/11 |
| kNN entropy (`k=5`) | 1/6 |

Kernel 仍然是表达“不同意图数量”最自然、覆盖单点集合也最完整的指标；但它与简单
representative count 在当前自然集和 Simulator 上方向得分相同。当前实验还没有证明
Kernel 在实际数据上优于计数，只证明它比 coherence、pairwise 和 kNN 更稳定且定义更完整。

## 9. 当前结论

现在可以保留的工程方向是：

```text
Session Context
    ↓
结构化 hard facet 限定 feasible space
    ↓
q_sem 在 feasible space 内保留语义支持区域
    ↓
Kernel Effective Number 描述剩余的不同意图数
```

不能声称纯 embedding 已经可靠解决意图透明度。更准确的故事是：

> 结构化约束负责排除确定不可能的区域；语义空间负责描述剩余区域内部还有多少不同的
> plausible shopping intents。

下一轮如果要证明 Kernel 相比商品计数真正有额外价值，需要专门构造“商品数近似相同，
但重复 listing 程度或多模态结构不同”的对照，而不是继续增加普通 vague-to-specific 句子。

## 10. 可复现实验产物

- QU 完整日志：`artifacts/retrieval/qu-to-probe-intent-space-natural-v1.json`
- Natural Hybrid：`artifacts/retrieval/semantic-dispersion-natural-expanded-v1.json`
- Natural Semantic-only：`artifacts/retrieval/semantic-dispersion-natural-expanded-unmasked-v1.json`
- Simulator 16×3 Hybrid：`artifacts/retrieval/semantic-dispersion-simulator-other-16x3-v1.json`
- Simulator 16×3 Semantic-only：`artifacts/retrieval/semantic-dispersion-simulator-other-16x3-unmasked-v1.json`

所有产物包含每轮自然语言、完整 `q_sem`、候选数量和各指标值。
