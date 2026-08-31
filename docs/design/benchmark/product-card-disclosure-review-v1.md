# 商品卡驱动 Benchmark · 20-session Review v1

Status: review fixture implemented, not yet promoted into the benchmark evaluator.

## 目的

当前 benchmark 的 `intent_card()` 从原始商品字段机械截取：

```python
hard_constraints = cleaned[:2]
soft_preferences = cleaned[2:4]
```

因此 simulator 最多只能披露四个字符串。本 review 用完整、来源可验证的新商品卡替代该来源，
但不会把商品卡中的所有后台事实都说成用户偏好。

## 20 条如何选择

- Buying、Browsing、Intent Override、Boundary 各 5 条；
- 包含之前重点调查的 `public_0041`、`public_0045`、`public_0098`、
  `public_0154`、`public_0199`；
- 其余样本覆盖珠宝、鞋、上衣、包、帽子、旅行用品和冷天配件等类型。

固定选择在
[`product-card-disclosure-review-v1.json`](../../../config/benchmark/product-card-disclosure-review-v1.json)。

## 投影规则

完整商品卡不会被修改。系统在它上面生成一个 simulator 专用披露视图：

- 每条披露保留原 facet、value、component、polarity、evidence 和 source reference；
- 商品型号、完整名称、评分、上架日期、产地和后台尺寸等不会变成用户要求；
- 部件事实保持部件边界，例如 `band=leather` 不会被说成整件商品都是 leather；
- 显式的 absent material 可以生成“鞋面不应使用 polyester”一类排除偏好；
- `ask_attribute` 直接来自 fact facet，不再对自然语言做关键词猜测；
- 每个 session 最多选择 10 条购物相关事实，不再有四条上限；
- Buying 和 Override 有 hard/soft commitment，Browsing 全部从 soft 开始；
- 固定模板只负责展示，没有调用 DeepSeek。

## 当前结果

- session：20；
- 每个 scenario：5；
- 历史失败样本：5；
- 总披露事实：182；
- 平均每 session：9.1；
- 最少 / 最多：6 / 10；
- API 调用和模型 token：0。

五个历史失败点有构建期断言：

| Sample | 必须进入披露视图的事实 |
| --- | --- |
| `public_0041` | polyester composition、pull-on closure；`imported` 被明确排除 |
| `public_0045` | polyester composition、button closure |
| `public_0098` | 100% rubber、upper excludes polyester |
| `public_0154` | cotton、hand wash |
| `public_0199` | machine wash |

## 如何查看与重建

入口报告：

[`artifacts/benchmark/product-card-disclosure-review-v1/report.md`](../../../artifacts/benchmark/product-card-disclosure-review-v1/report.md)

其中每条 session 都有：

- 完整原始新商品卡 JSON；
- 最终披露计划；
- 每条事实的 commitment、facet、component、value 和 evidence；
- 固定 `ask_attribute=other` 的完整多轮对话；
- 被排除事实的原因统计。

重建：

```powershell
python scripts/benchmark/build_product_card_disclosure_review.py
```

## 尚未做的事

本轮只生成供人工审阅的 fixture，没有修改 benchmark 仓库的正式 evaluator。审阅通过后，
再把同一投影器移入 benchmark，并保留 `legacy` / `product_facts` 两种模式做 A/B。
