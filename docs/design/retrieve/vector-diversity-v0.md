# Category-Blind Vector Diversity v0

状态：**实验完成；证明路线可行，参数尚未冻结。**

- 实现：[`../../../src/shopping_copilot/retrieval/vector_diversity.py`](../../../src/shopping_copilot/retrieval/vector_diversity.py)
- 完整结果：[`../../../artifacts/retrieval/vector-diversity-v0.md`](../../../artifacts/retrieval/vector-diversity-v0.md)

## 1. 这轮要回答什么

模糊请求不应该只返回十个相似 listing。例如：

> 我想去北海道，帮我找找有什么能买的。

我们希望结果自然覆盖冬季配饰、雪地鞋、保暖外套、雪裤等不同商品方向。但是第一版刻意不使用：

- category quota；
- facet quota；
- 手写的“北海道应当推荐帽子、鞋和棉衣”规则；
- LLM 生成的商品类型列表。

category 仅用于实验结束后审计结果，完全不参加选品。

## 2. 算法

```text
q_sem
  → 对完整 50k catalog 计算 query-product cosine
  → 截取一个仍然相关的候选窗
  → 使用相同商品向量计算 product-product cosine
  → MMR 逐个选择 Top-10
```

每次选择：

$$
MMR(i)=
\lambda(T_t)\,Sim(q,i)
-
(1-\lambda(T_t))
\max_{j\in Selected}Sim(i,j)
$$

实验映射为：

$$
\lambda(T_t)=0.30+0.60T_t
$$

因此：

| `T_t` 实验锚点 | relevance weight | 行为 |
| ---: | ---: | --- |
| 0.10 | 0.36 | 强调商品间差异 |
| 0.50 | 0.60 | 平衡相关性和差异 |
| 0.90 | 0.84 | 强调 query relevance |

这些值只是扫描锚点，不是冻结的 runtime threshold。

## 3. 北海道结果

### 3.1 直接使用字面查询

`q_sem` 只表达“去北海道旅行，可以买不同种类商品”，但不提供 cold/snow 上下文时：

- Dense Top-10 有 4 个审计大类，平均两两 cosine 为 `0.751`；
- MMR、`K=80`、`T=0.10` 有 7 个审计大类，平均两两 cosine 降至 `0.687`；
- 但是结果出现 kimono cosplay、普通首饰、手表和沙滩 cover-up。

这说明 MMR 确实制造了向量差异，但 query 自己没有可靠地表达“为什么去北海道会需要这些商品”。多样化不能修复一个缺少场景知识的检索 query。

### 3.2 使用经过场景补全的语义查询

`q_sem` 加入冬季、寒冷和降雪语境，但仍不枚举鞋、帽子、外套等商品类型：

```text
products that could be useful to wear or bring for a winter trip to Hokkaido,
Japan, with cold weather and snow; open to different kinds of products
```

Dense Top-10：

- 3 个审计大类；
- 平均两两 cosine `0.777`；
- 结果主要集中在冬季手套、帽子和围巾。

MMR、`K=80`、`T=0.10`：

- 4 个审计大类、9 个叶子类别；
- 平均两两 cosine 降至 `0.717`；
- 实际出现冬季帽子/围巾/手套、雪地靴、滑雪裤、羽绒服、厚外套和 neck gaiter。

这已经接近预期的“鞋子、帽子、棉衣来自同一次模糊搜索”，且选品阶段没有读 category。

## 4. 其他请求是否也有相同行为

在 `K=80` 下比较普通 Dense Top-10 和低透明度 MMR：

| 请求 | Dense 大类数 | MMR 大类数 | Dense pair cosine | MMR pair cosine |
| --- | ---: | ---: | ---: | ---: |
| 北海道，字面 query | 4 | 7 | 0.751 | 0.687 |
| 北海道，场景补全 | 3 | 4 | 0.777 | 0.717 |
| 夏季婚礼 | 1 | 3 | 0.859 | 0.770 |
| 新办公室工作 | 2 | 4 | 0.762 | 0.641 |
| 优雅风格礼物 | 4 | 8 | 0.770 | 0.725 |

同时，明确请求在高透明度锚点下仍然会自然聚焦：

- 男士防水保暖雪地靴：Top-10 全部回到 footwear 主方向；
- 小号纯银珍珠耳钉：Top-10 全部保持 jewelry；
- 这不是 category gate 的结果，而是较高 `\lambda(T_t)` 让 query relevance 主导。

## 5. Top-80 是否太窄

对这一轮“结果多样性”实验来说，**没有证据表明 Top-80 太窄**。

- `K=80` 已经能产生目标中的跨商品类型结果；
- `K=500` 会继续增加类别，但开始引入语义擦边商品；
- `K=2000 + 低 λ` 会选到仅仅因为标题含 `Snow` 而相关的耳钉、cosplay、普通裙子等商品；
- MMR 只能在候选池中做 relevance–novelty 交换，候选池过宽时会把“不相关但很不同”误当成有价值多样性。

所以之前 Top-80 Probe 不收敛的问题，不能直接推导出正式多样化召回也需要极宽候选池。两者是不同任务。

## 6. 当前结论

第一版正式检索可以先保持纯向量方案：

1. 以 `K=80` 作为当前工程起点；
2. 使用 category-blind cosine MMR；
3. `T_t` 连续提高 relevance weight；
4. category/facet 只做 hard constraint，不做 diversity quota；
5. 用输出日志继续观察跨类别效果，不提前加入类别规则。

但必须保留一条边界：

> MMR 负责在已经相关的商品中展开不同方向；`q_sem` 必须先把用户场景表达成商品可匹配的语义。

对于“北海道”这种依赖常识的请求，下一步要测试的是 QU 的 `q_sem` 是否能稳定做出恰当而不过度的场景补全。如果不能，再决定把 contextual retrieval expansion 放进召回层；这和是否使用 category diversity 是两个独立问题。

## 7. 已知噪声

- catalog 中存在标题和 category 不一致或文本污染的商品，例如被归在 cold-weather 类别下的 summer sandal；
- lexical coincidence 会产生 `Snow` 耳钉、snow-globe card holder 等 false positive；
- MMR 不会识别“不同向量来自不同商品需求”还是“不同向量来自坏商品”。

这些应进入 retrieval diagnostics / 后续 relevance rerank，不应通过偷偷增加 category quota 来掩盖。
