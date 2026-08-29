# Formal Retrieval：系统最后怎样找出 10 个商品

## 1. 先看完整流程

```text
用户自然语言
  → QU 更新完整 Session Context
  → 编译出 q_lex、q_sem、hard constraints、soft preferences
  → Intent Volume 算出 T_t
  → 三路召回
  → 合并候选
  → T_t 控制结果要多样还是聚焦
  → Top-10
```

这里没有 Buying 搜索器和 Browsing 搜索器。请求模糊还是明确，使用的是同一套系统，只是 `T_t` 不同。

## 2. 先处理绝对不能违反的条件

例如用户说：

> 女士红色皮质高跟鞋，不要黑色。

系统先从 50k 商品里排除 evidence 明确是黑色的商品，再用类别、性别、红色和皮质继续收窄。这个动作发生
在任何路线取 Top-80 之前。

明确排除永远不放松；明确 include 如果会让结果变成空集合，则改成排序偏好，并在日志里说明。这样不会
因为 catalog 缺少一个尺码字段就彻底搜不到商品。

## 3. 三条路各自擅长什么

### Dense 路

看 `q_sem` 的整体意思，擅长理解场景。例如“冬天去北海道有什么可以买”。

### Lexical 路

看 `q_lex` 中的明确词，擅长品牌、型号、颜色、材质、`waterproof` 这类字面证据。

### Facet 路

看 QU 已经抽取好的结构化偏好，例如：

```json
{"facet": "color", "operator": "eq", "value": "red"}
```

Facet 路不会自己再猜一次用户意思。没有可靠结构化正向条件时，这条路就不生效。

## 4. 为什么不能直接把三种分数相加

Dense 的 cosine、Lexical 的 BM25 和 Facet 命中比例是三种不同单位。系统只看每路名次：同一个商品如果被
多条路排在前面，它在合并结果里就更靠前。这叫 RRF，但组员只需要理解成：

> 多种独立证据都支持的商品，优先级更高。

三路各取 80 条，合并后仍只留 80 条。

## 5. `T_t` 在最后做什么

系统从合并后的 80 条里逐个选 10 条：

- `T_t` 低：更在意“下一条和已经选的商品是否太像”；
- `T_t` 高：更在意“下一条在融合排名里是否足够靠前”。

商品之间是否相像只看 embedding，不读取 category，也没有“必须放一双鞋、一顶帽子、一件衣服”的规则。

所以同一句北海道搜索在低 `T_t` 下，真实结果里自然出现了雪地靴、滑雪裤、帽子围巾手套和雪服；精确的
男士黑色防水雪地靴在高 `T_t` 和 hard mask 下则全部回到 footwear。

## 6. 现在已经做到什么

- 三路正式召回已实现；
- hard mask 在每路 Top-K 前执行；
- RRF 已实现；
- `T_t` 控制的向量 MMR 已接在融合后；
- 6 个自然语言请求已在真实 50k catalog 上完成三组对照；
- 单元测试、完整日志和每件商品的 route contribution 都已保留。

详细算法和实测数据见：

- [Formal Multi-route Retrieval v0](../design/retrieve/formal-multi-route-v0.md)
- [50k 完整实验](../../artifacts/retrieval/multi-route-v0.md)
