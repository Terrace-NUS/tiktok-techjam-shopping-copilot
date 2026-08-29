# Formal Multi-route Retrieval v0

- 状态：**已实现并完成 50k catalog 实测。**
- 实现入口：[`../../../src/shopping_copilot/retrieval/controller.py`](../../../src/shopping_copilot/retrieval/controller.py)
- 完整实验：[`../../../artifacts/retrieval/multi-route-v0.md`](../../../artifacts/retrieval/multi-route-v0.md)

## 1. 一句话说明

系统不会先把用户硬分成 Buying 或 Browsing，再选择两套搜索器。每次搜索都执行同一条流程：

```text
完整 Session Context
  → Query Understanding 生成 CompiledQuery
  → hard mask 先排除明确不合格商品
  → Dense / Lexical / Facet 三路各自找候选
  → RRF 合并成 80 个仍然相关的候选
  → T_t 控制纯向量 MMR
  → 最终 Top-10
```

`T_t` 低时，在相关候选中主动展开不同方向；`T_t` 高时，结果更贴近融合排序。它控制的是
“探索多少”，不是预测用户会不会购买。

## 2. 三条召回路线

### Dense

输入 `q_sem`，对完整 50k 商品向量计算 cosine，再在 hard mask 合格集合内取 Top-80。

它主要解决场景和自然语言表达，例如：

```text
冬天去北海道，可能需要穿或携带的商品，商品类型仍然开放
```

### Lexical

输入 `q_lex`，用 SQLite FTS5 匹配标题、category、feature、detail、store 和 description。字段有固定
权重，精确词、品牌、型号、颜色和材质不会完全交给 embedding。

它先遍历完整命中流、应用 hard mask，再截 Top-80。因此不允许“先取 80 条，再从里面删违规商品”。

### Facet

输入 QU 已经生成的结构化 soft preference，以及 hard include 因空集合而放松下来的条件。

v0 支持 catalog evidence 能验证的：

```text
brand / color / department / feature / gender
material / size / style / use_case
```

这条路线不会重新阅读对话，也不会自己猜新 facet：

- 至少有一个正向、可验证的结构化偏好才启动；
- 正向命中产生候选；
- soft negative 命中只降权 `0.35`，不会变成第二道 hard gate；
- hard exclusion 已经在共享 hard mask 阶段执行，绝不在这里放松；
- 只有负向条件时不生成一大批任意“未命中”商品，Facet 路直接记为 unavailable。

Facet 路的 `raw_score` 是：

$$
\frac{\text{matched positive}}{\text{positive count}}
-0.35\frac{\text{matched negative}}{\max(1,\text{negative count})}
$$

这个分数只负责 Facet 路内部排序，不会直接与 Dense cosine 或 FTS BM25 相加。

## 3. hard mask 为什么必须在 Top-K 前

三条路线共享同一个 `ResolvedHardMask`：

1. 明确排除条件先执行，而且永不放松；
2. 明确 include 有非空交集时继续收窄；
3. include 若会清空候选，则转为 ranking evidence；
4. 每路只在剩余商品中取自己的 Top-80。

这同时满足两个目标：

- “不要黑色”不会因为某路的 Top-80 恰好全是黑色而让搜索失效；
- 尺码等 metadata 缺失导致 include 空集合时，系统仍能返回结果，但日志会明确记录
  `relaxed_to_ranking`，不能假装约束已经满足。

## 4. 三种分数怎样合并

Dense cosine、FTS BM25 和 Facet match ratio 不在同一个数值空间，直接加权相加没有可信含义。v0 使用
Reciprocal Rank Fusion：

$$
RRF(i)=\sum_{r\in available\ routes}\frac{1}{60+rank_r(i)}
$$

规则很简单：

- 某条路线没有证据时不参加，不制造假分数；
- 三路当前等权；
- 同一个商品被多路靠前找到，会自然排到融合列表前面；
- 融合后只保留 Top-80，防止多样化阶段从过宽候选池捡到纯噪声。

## 5. `T_t` 怎样影响正式结果

先把融合分数除以当前最大融合分数，得到 `[0,1]` relevance。然后对融合 Top-80 做纯向量 MMR：

$$
MMR(i)=\lambda(T_t)Rel_{RRF}(i)
-(1-\lambda(T_t))\max_{j\in Selected}Cosine(i,j)
$$

$$
\lambda(T_t)=0.30+0.60T_t
$$

| `T_t` | relevance weight | 默认行为 |
| ---: | ---: | --- |
| 0.10 | 0.36 | 在相关候选里积极寻找不同商品方向 |
| 0.50 | 0.60 | 相关性与差异平衡 |
| 0.90 | 0.84 | 基本遵循多路融合的聚焦结果 |

多样性惩罚只使用商品 embedding 的 product-product cosine，不读 category，不做鞋/帽子/衣服配额。
因此跨品类是向量空间自然产生的结果，不是演示用的手写规则。

显式指令仍可覆盖默认值：

- `INCREASE diversity`：在当前 `λ` 上减 `0.15`；
- `DECREASE diversity`：在当前 `λ` 上加 `0.15`；
- 最终截断到 `[0,1]`。

v0 刻意不让 `T_t` 开关路线或改变 RRF 权重。所有可用路线始终存在，`T_t` 只有一个清楚、可展示、
可消融的控制点。

## 6. 50k catalog 实测

实验对 6 个自然语言请求比较：

```text
Dense only
Dense + Lexical
Dense + Lexical + Facet
```

所有商品都来自正式 50k catalog；category 只在实验结束后统计，不参与 MMR。实验中的 `T_t` 是固定锚点，
目的是隔离检索行为，不冒充 Intent Volume 的重新测量值。

### 模糊搜索

| 请求 | Dense 大类数 | 三路大类数 | Dense pair cosine | 三路 pair cosine |
| --- | ---: | ---: | ---: | ---: |
| 北海道冬季旅行 | 4 | 4 | 0.768 | 0.756 |
| 夏季婚礼 | 2 | 5 | 0.823 | 0.651 |
| 新办公室工作 | 5 | 4 | 0.677 | 0.679 |
| 红色婚礼配饰 | 3 | 6 | 0.753 | 0.695 |

北海道 Top-10 实际包含雪地靴、滑雪裤、帽子/围巾/手套、雪服和保暖手套。这里 Facet 路没有可用的
结构化条件，所以真实执行的是 Dense + Lexical；系统不会为了声称“三路”而伪造 Facet 证据。

夏季婚礼中，Facet evidence 生效后从 Dense 的 2 个审计大类扩展到 5 个，平均商品两两 cosine 从
`0.823` 降到 `0.651`。红色婚礼配饰从 3 个大类扩展到 6 个。

办公室请求是一个重要反例：Dense 本身已经有 5 个大类，加入其他路线后是 4 个，pair cosine 也没有
下降。这证明“多路一定更多样”不是实现假设；多路的职责是补足不同证据，不是机械提高 category 数。

### 精确搜索

- 男士黑色防水雪地靴：hard mask 从 50k 收到 29 条，最终 10 条全部是 footwear；
- 女士红色皮质高跟鞋且排除黑色：先执行排除，再经 category、gender、color、material 收到 24 条，
  最终 10 条全部是 footwear；
- size=10 和 closed-toe 在对应交集中没有可靠 evidence，因此按协议转为 ranking evidence，并在 trace
  中留下记录。

## 7. 当前边界

- Facet evidence 能补充覆盖，但广义词也可能引入边缘商品；夏季婚礼结果中就出现了偏弱的
  lightweight/ceremony 命中。这是后续 relevance ranker 的问题，不应用 category quota 掩盖。
- FTS 当前是进程内 SQLite 索引，Facet postings 也在启动时从 raw catalog 构建；正式服务应预热并复用
  `RetrievalController`，不能每轮调用 factory。
- v0 没有 LLM rerank，也没有个性化 profile 分；这两者可以成为融合后的附加排序证据，但不能改变
  hard-mask 结果。
- `RetrievalController.search()` 当前接收已经算好的 `T_t`。QU → Intent Volume → Controller 的应用层
  orchestration 仍需接线，但检索组件本身已经完成。

## 8. 冻结的 v0 参数

```text
route_k = 80
RRF rank constant = 60
fusion_k = 80
final_k = 10
λ(T) = 0.30 + 0.60T
explicit diversity adjustment = ±0.15
```

这些是 Hackathon v0 工程参数。它们可以在后续真实结果上调整，但禁止在没有日志的情况下临场改变算法。
