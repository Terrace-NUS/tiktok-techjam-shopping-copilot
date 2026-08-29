# Intent Transparency：系统怎样判断“用户还可能想要多少种东西”

## 1. 一句话理解

我们的核心故事不是把用户硬分成 Buying 或 Browsing，也不是猜他最终会不会购买。

> 每轮对话后，catalog 中仍然存在一片合理的购物意图空间。用户补充条件会排除一部分空间；撤销
> 条件会让空间重新变宽；改变目标会让整片空间移动。Intent Transparency `T_t` 把这件事变成一个
> 可展示、可供系统使用的数字。

```text
空间很大  → T_t 低 → 继续探索、保留多样性
空间变小  → T_t 高 → 收紧检索、强化偏好匹配
```

`T_t` 不要求逐轮严格上升。系统不会把“聊天轮数更多”假装成“理解一定更深”。

## 2. 它位于整个系统的哪里

```text
用户自然语言
    ↓
DeepSeek Query Understanding
    ↓
完整、修复后的 Session Context
    ↓
Query Compiler
    ↓
Fuzzy Intent Volume
    ├── structured hard evidence
    ├── goal / semantic preferences
    └── catalog duplicate density
    ↓
N_t：还剩多少有效意图空间
    ↓
T_t：0–1 意图透明度
    +
D_t：这次测量是否健康、为什么
```

Intent Volume 不重新理解聊天，也不读取一轮孤立的用户原话。它只消费 QU 已经修复好的完整
`IntentState + CompiledQuery`。

## 3. `N_t` 怎样计算

### 3.1 结构化条件

颜色、材质、价格、尺码、feature 等可由 catalog evidence 检验的明确条件，不直接把违反商品删成
零质量：

```text
商品满足条件   → membership = 1
商品违反条件   → membership = ε
```

runtime v1 使用 `ε = 0.01`。这样“只差一个条件”的商品仍有极小质量，意图空间不会因为当前 50k
商品 evidence 稀疏而轻易坍缩成空集。

### 3.2 开放语义条件

goal、soft preference 和 semantic-only preference 各自形成独立 embedding factor。对每个 factor：

```text
factor 与全部 50k 商品计算 cosine
→ 以 catalog 分位数作为中心
→ sigmoid 变成 0–1 membership
```

负向语义使用 `1 - positive membership`；soft preference 的影响弱于 hard preference。

### 3.3 条件合并

所有条件使用 Product of Experts 相乘：

$$
a_t(i)=\prod_j m_j(i)
$$

只有同时兼容多个条件的商品才能保留较大质量。这也是为什么新增条件通常让空间自然缩小，而不是靠
事后强行修改数字。

### 3.4 重复 listing 降权

同一类热门商品可能有许多近似 listing。若每条 listing 都算一个完整方向，指标会被商家数量污染。

系统预先根据商品 embedding 计算 catalog density `d_i`，给密集区域中的商品更低权重：

$$
N_t=\sum_i\frac{a_t(i)}{d_i}
$$

因此 `N_t` 更接近“还剩多少有效购物方向”，而不是“还剩多少条 listing”。

## 4. `N_t` 怎样变成 `T_t`

令 `N_catalog` 为同一 release 下整个 catalog 的 density-corrected reference volume：

$$
T_t=1-\frac{\log(1+N_t)}{\log(1+N_{catalog})}
$$

并限制在 `[0,1]`。

- `N_t` 大：剩余空间广，`T_t` 低；
- `N_t` 小：剩余空间窄，`T_t` 高；
- `N_t = N_catalog`：`T_t = 0`；
- `N_t → 0`：`T_t → 1`。

不用“本轮 goal-only 空间”作为零点，因为 goal 本身可能已经非常具体，例如 `pearl stud earrings`。
若把每个 goal 强制定为 0，相同用户意图会因 session 起点不同而得到不同透明度。

## 5. 每轮变化怎样解释

方向根据原始 `N_t` 判断，不根据已经压缩成 0–1 的展示值判断：

| direction | 白话解释 |
| --- | --- |
| `initial` | session 第一笔可用测量 |
| `narrower` | 同一购物任务下空间明显缩小 |
| `broader` | 用户撤销条件，空间明显变宽 |
| `stable` | 空间相对变化不超过 10% |
| `moved` | 上游明确确认用户换商品目标 |
| `unavailable` | 当前没有可信商品空间可测 |

两个 goal 字符串不一样，不代表一定换目标：`footwear → boots` 可能只是细化。运行时只有收到上游明确
的 switch 证据才标记 `moved`，否则继续依据实际体积变化判断。

## 6. `D_t` 是什么

`D_t` 是体检报告，不是另一个用户画像分数，也不参与偷偷修正 `T_t`。

它包括：

| 诊断 | 含义 |
| --- | --- |
| status | `healthy`、`degraded` 或 `unavailable` |
| semantic factor count | 本轮有多少语义因子 |
| hard factor count | 有多少可验证结构化条件 |
| relaxed hard IDs | 哪些正向 hard 条件因 evidence 空结果退回语义 |
| Top all-hard compliance | Top 商品同时满足全部 hard facet 的比例 |
| Top mean-hard compliance | Top 商品平均满足多少 hard facets |
| active / don't-care / open facets | 已明确、不在乎、由未来提问模块显式提供的开放维度 |

测量降级时仍可以展示 `T_t`，但 UI 应同时显示警告；`unavailable` 时 `T_t = null`，不能用 0 或 0.5
冒充真实测量。

## 7. 运行时输出长什么样

```json
{
  "schema": "shopping-copilot/intent-transparency/v1",
  "policy_id": "soft_hybrid_intent_volume_v1",
  "intent_version": 3,
  "goal": "shoes",
  "transparency": 0.99996,
  "change": 0.07852,
  "direction": "narrower",
  "remaining_intent_volume": 0.00039,
  "catalog_reference_volume": 38123.16,
  "diagnostics": {
    "status": "degraded",
    "reason_codes": ["low_all_hard_top_compliance"],
    "semantic_factor_count": 3,
    "hard_factor_count": 5
  }
}
```

完整 JSON 还包含 release/index binding、goal reference、active facets 和 hard compliance。

## 8. 三类真实演示轨迹

以下数值来自真实 DeepSeek QU 保存的 Session Context，再通过正式 runtime v1 组件重放。

### 8.1 持续增加条件

```text
weekend trail shoes
T_t = 0.476

→ men's trail-running shoes, size 10
T_t = 0.921  narrower

→ waterproof, wide, dark grey, strong grip, under $150
T_t = 1.000  narrower
```

### 8.2 撤销条件

```text
red leather closed-toe heels, size 7, under $80
T_t = 0.996

→ only size and heels matter; color/material/toe/budget are open
T_t = 0.544  broader
```

### 8.3 改变商品目标

```text
women's cushioned running shoes under $120
T_t = 0.463

→ forget the shoes; pearl stud earrings under $50
T_t = 0.680  moved
```

数值可以上涨或下降；`moved` 告诉展示层不要把这次 delta 解释成“对旧目标理解更深”。

## 9. 全量实测结果

60 段自然语言对话、130 个用户 turn：

- DeepSeek QU：130/130 成功；
- 127 个可检索状态进入 Intent Volume；
- narrower：33/33；
- broader：10/10；
- stable：7/7；
- 10 段三轮收紧对话的 20 个相邻变化：20/20；
- 7 段 override 只观察迁移，不规定数值升降。

正式 runtime 对全部 127 个状态逐项对比离线实验：

- parity failure：0；
- transparency 最大绝对误差 `< 4e-7`；
- remaining volume 最大相对误差 `< 5e-6`；
- CPU 热路径平均约 94ms/turn，最慢约 308ms。

完整冷启动约 77 秒，主要是加载 release、构建 retrieval evidence 和加载 embedding 模型。因此正式
应用必须在进程启动时预热并复用这些对象，不能每轮重新构造 estimator。

## 10. `T_t` 后续怎样控制系统

`T_t` 已经接入最终召回控制器。当前 v0 不用它关闭任何召回路线，而是在三路候选完成 RRF 融合后，
连续控制纯向量 MMR 的 relevance weight：

```text
T_t 低
→ 融合 Top-80 中更强地惩罚相似商品
→ 保留更多商品方向和多样性
→ 优先问能明显切分空间的问题

T_t 中
→ 平衡融合相关性和商品间差异

T_t 高
→ 更接近多路融合的相关性顺序
→ 降低无意义多样性
→ 减少继续追问
```

具体映射是 `λ(T_t)=0.30+0.60T_t`。它不能反过来修改本协议的测量值。完整流程见
[Formal Retrieval 说明](05-formal-retrieval.md)。

## 11. 不能夸大的地方

- Product of Experts 的方向正确部分来自状态代数，不等于商品语义 100% 正确；
- runtime v1 参数来自同一套开发语料扫描，不是独立 held-out 的通用生产标定；
- 主候选 Top-20 平均满足约 80.9% 的可验证 hard facets，但同时满足全部 hard facets 的平均比例约
  38.8%；Intent Volume 不是最终严格召回；
- `open_facets` 必须由未来 clarification planner 提供，Intent Volume 不会自己猜。

## 12. 权威入口

- [Intent Transparency 运行时协议 v1](../design/intent_purity/runtime-contract-v1.md)
- [QU v1.3 全量链路结果](../design/query_understanding/v1-3-full-chain-results.md)
- [Fuzzy Intent Volume 扩大实验 v2](../design/intent_purity/expanded-fuzzy-intent-volume-v2.md)
- [Query Understanding 说明](03-query-understanding.md)
- [Facet System 说明](02-facet-system.md)
- [Session Context 说明](01-session-context.md)
- [Formal Retrieval 说明](05-formal-retrieval.md)
