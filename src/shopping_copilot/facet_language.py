"""Shared semantic language for user intent and catalog product facts."""

from __future__ import annotations

FACET_LANGUAGE_VERSION = "shopping_facet_language_v1"

CORE_PRODUCT_FACT_FACETS = (
    "category",
    "brand",
    "material",
    "color",
    "size",
    "style",
    "department",
    "gender",
    "feature",
    "use_case",
)

CLOSED_GENDER_VALUES = ("men", "women", "unisex", "boys", "girls", "kids", "baby")

SHARED_FACT_EXTRACTION_RULES = """\
共享的 shopping facet 事实协议：
- 先判断事实的主语、商品部件和否定范围，再决定 facet；不能只做关键词命中。
- 只把商品或商品部件的属性写成商品事实。佩戴者反应、包装文字、保养说明、品牌故事、比较对象
  和广告场景中的词，不能冒充商品自身属性。
- 一段文本可以包含多个独立事实；必须逐项抽取，不能抽到第一个 material/color/size 就停止。
- meaning 保存标准化的完整含义；evidence 必须尽量保留输入中的最短连续原文，不能用释义冒充引文。
- composition、尺寸、单位和限定词不能被概括丢失，例如必须保留“95% gossypium, 5% spandex”
  和“approximately 1.57 inches”中的精度。
- facet 使用 lower_snake_case。优先使用 category、brand、material、color、size、style、department、
  gender、feature、use_case；确有清楚的商品属性时可以使用更具体的 facet，例如 heel_height。
- gender 是封闭概念：men's/male -> men，women's/female -> women；原始措辞仍保留在 evidence。
- 明确否定的商品事实必须保留否定极性，不能反转成肯定事实。例如 “not waterproof” 不是
  feature=waterproof；“nose won't get red”也不是 color=red。
"""
