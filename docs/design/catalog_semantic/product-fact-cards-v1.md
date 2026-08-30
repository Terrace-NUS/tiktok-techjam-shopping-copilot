# Product Fact Cards v1：让商品侧和用户侧说同一种 facet 语言

- 状态：**抽取器与 smoke/expanded live test 已实现；尚未接入正式 hard-mask**
- 日期：**2026-08-30**
- 模型：**DeepSeek V4 Flash**
- Product prompt：**`product_fact_card_v1_1`**
- Shared facet language：**`shopping_facet_language_v1`**

## 1. 为什么要增加这一层

用户侧 QU 已经会理解完整自然语言：它会判断事实在说商品、商品部件还是佩戴者，也会识别否定范围，
并从一段描述中抽取多个 facet。

旧商品侧却仍主要使用字段白名单和关键词：例如 material/color 没有完整读取 description。于是会发生：

```text
用户：fabric 是关键要求
目标商品 description：soft, stretchy, fabric straps
旧 Evidence Index：material 中没有 fabric
结果：正确商品被 hard mask 删除
```

解决方法不是把用户的 hard 要求偷偷改成 soft，而是让商品侧也使用相同的事实语言。

## 2. “相同逻辑”具体指什么

QU 和商品卡不是同一个任务，不能复用同一种输出对象；但它们共享同一份 normative rules：

- 先判断主语、商品部件和否定范围，不能只搜关键词；
- 同一段文本可以产生多个独立 facet；
- 保留 composition、尺寸、单位和限定词；
- `meaning` 表达标准化含义，`evidence` 保存连续原文；
- gender 使用同一封闭概念；
- facet 使用相同 lower_snake_case 语言，并允许确有证据的具体属性如 `heel_height`。

共享协议由
[`facet_language.py`](../../../src/shopping_copilot/facet_language.py) 单点定义，同时进入 QU prompt 和
商品卡 prompt。这样两侧将来修改主语、否定或 facet 命名规则时，不需要靠人工保持两份文档同步。

两侧的差异是：

| 用户侧 QU | 商品侧 Product Fact Card |
| --- | --- |
| 编辑多轮 Session Context | 描述一件固定商品 |
| 有 hard/soft、include/exclude | 有 present/absent 商品事实 |
| evidence 来自用户最新话语 | evidence 来自 title/features/description/details 等原字段 |
| 需要保留、替换、撤销旧偏好 | 不涉及状态更新 |

## 3. DeepSeek 读取什么

每个商品单独做一次 native function call。输入不按 token 成本截断，包含原始行中除 `parent_asin`
外的所有非空字段：

```text
title
categories[*]
store
features[*]
description[*]
details.*
price / rating / 其他 metadata
```

数组项和 details 项会获得稳定 `source_ref`，但文本保持完整。原始 `catalog.jsonl` 只读且不修改；
每行原始 bytes 的 SHA-256 写入 `source_id`，用于断点续跑和过期检测。

这里明确采用质量优先策略：不限制只抽取 top facets，不丢弃 description，也不为了少用 DeepSeek token
把多个商品塞进一次容易相互污染的调用。`max_tokens` 默认 8192。

## 4. DeepSeek 输出什么

每条 atomic fact 包含：

```json
{
  "facet": "material",
  "value": "100% Cotton",
  "aliases": ["cotton"],
  "polarity": "present",
  "component": "cups",
  "meaning": "The bra cups are made of 100% cotton.",
  "evidence": "100% Cotton cups",
  "source_ref": "description_0",
  "confidence": 0.99
}
```

`value` 保留有辨识力的来源措辞；`aliases` 只保存明确等价的购物叫法，用来连接用户常用词和 catalog
术语，例如 `gossypium → cotton`。alias 不是原文证据，原文仍必须由 `evidence/source_ref` 证明。

`polarity=absent` 只表示来源明确说商品不含或不具备某属性。未来构建 hard-mask index 时，absent
事实不能被当成 present 命中。

## 5. 本地为什么还要验证

DeepSeek 有语义判断自由，但没有权伪造引用。本地 decoder 会检查：

- tool name、字段类型和 parent_asin；
- facet 必须是 lower_snake_case；
- source_ref 必须属于本次商品输入；
- evidence 必须能恢复为对应 source 中的真实连续原文；
- 同一 facet/value/polarity/component 的重复事实会合并，aliases 取并集。

只对普通空白与 NBSP 等价做无损恢复，最后写入卡片的仍是 source 中的真实字符；释义不能通过引用校验。

## 6. Live test

扩大测试从官方 public set 跨间隔选择 21 个目标商品，包含服装、鞋履和珠宝，并包括原 hard-mask
失败的 `B01IAKCZEK` 与 `B00CYNKSTE`。

| 指标 | 结果 |
| --- | ---: |
| native tool call + local card validation | **21 / 21** |
| 生成 atomic facts | **647** |
| 每张卡平均 facts | **30.81** |
| 来自 description 的 facts | **64** |
| 本地去重后 exact duplicate groups | **0** |
| reported total tokens | **97,681** |

两个关键事实均已恢复：

- `B01IAKCZEK`：`material=fabric`，`component=straps`，证据来自 description；
- `B00CYNKSTE`：`material=100% Cotton` + alias `cotton`，并从同一 description 抽出 White/Black
  和尺码信息。

完整产物位于 `artifacts/catalog-semantic/product-facts-v1-1-expanded/`。

## 7. 50k 生成方式

入口：

```powershell
.venv-3.10\Scripts\python.exe `
  scripts/catalog_semantic/extract_product_fact_cards.py `
  --workers 12 `
  --resume
```

生成器使用有界并发，不会把 50k 完整商品文本同时放进内存。每个商品先原子写入独立 card，进程中断后
按 `source_id + model + prompt version + facet language version` 精确续跑。全量完成后再把 sidecar
绑定进新的 Retrieval Evidence Index；当前 `CatalogSemanticRelease v0` 与原始数据集均不变。

## 8. 尚未完成

- 还没有实际启动 50k 全量调用；
- 还没有把 sidecar 的 present value/aliases 合并进 hard-mask evidence；
- 还没有为模型升级或 prompt 变化定义正式 release migration；
- sidecar 属于可追溯的 model-derived retrieval evidence，不自动升级为 Gate-A/B
  `catalog_verified` truth。

