# Public 200 商品卡完全替换实验 v1

Status: implemented and measured, 2026-08-31.

## 1. 这次到底替换了什么

本实验只替换 public benchmark 对应的 200 个目标商品：

- 这 200 个商品的旧搜索文本不再参与检索；
- Lexical 使用新卡里的摘要、事实和同义表达；
- Facet 与 hard mask 使用新卡里的结构化事实；
- BGE reranker 收到新卡文本；
- Dense 使用同一个 `BAAI/bge-small-en-v1.5` 重新编码新卡；
- 其他 49,800 个商品的文本和 Dense 向量不变。

这里的“完全替换”不是把新卡追加到旧卡后面。对于有新卡的商品，运行时会构造一个新的
`ProductDocument`；没有新卡的商品则原样返回旧 `ProductDocument`。

原始 [`catalog.jsonl`](../../../data/catalog.jsonl) 没有被修改。新卡仍然是独立 sidecar，
Dense 也是一个独立的新索引目录，因此旧系统可以随时复现。

## 2. Dense 为什么不需要重算 5 万件商品

旧索引和新索引使用相同的模型、版本、维度和商品顺序。构建器执行下面四步：

1. 复制旧的 `50,000 × 384` 向量矩阵；
2. 只编码 200 张新商品卡；
3. 按 `parent_asin` 覆盖对应的 200 行；
4. 用新的文本全集哈希和向量文件哈希发布一个完整、可校验的新索引。

实测结果：

- 200 个目标行全部发生变化；
- 其余 49,800 行逐元素完全等于旧索引；
- GPU 构建约 7.3 秒，总流程约 10.0 秒。

构建命令：

```powershell
python scripts/retrieval/build_partial_product_card_dense.py --device cuda
```

输出索引是 `artifacts/retrieval/dense-public-200-replaced-v1/`。

## 3. 测试方法

为了不再消耗 DeepSeek token，也不让一次新的 QU 输出影响比较，本实验复用已保存的
public 200 sessions：

- 每个 session 取最后一个可搜索、可评分的查询；
- 复用当时生成的 `CompiledQuery`；
- 完整召回还复用当时的 `T_t`；
- 对旧卡和新卡分别运行相同检索算法；
- 检索结束后才读取目标 ASIN 计算召回率。

因此 A/B 之间唯一有意义的变化是商品卡及其对应检索数据。它不是一次新的 simulator
对话评分，也不包含 DeepSeek 最终排名。

复现命令：

```powershell
python scripts/retrieval/evaluate_partial_product_card_dense.py --device cuda
python scripts/retrieval/evaluate_partial_product_card_recall.py --device cuda
```

## 4. Dense 单路结果

Dense 对 5 万商品精确计算目标排名：

| 截断位置 | 旧卡 | 新卡 | 变化 |
| ---: | ---: | ---: | ---: |
| Top 10 | 31.5% | 31.5% | +0.0% |
| Top 80 | 67.0% | 65.0% | -2.0% |
| Top 150 | 73.5% | 77.0% | +3.5% |
| Top 210 | 79.0% | 80.0% | +1.0% |
| Top 300 | 81.5% | 84.0% | +2.5% |
| Top 2,000 | 97.0% | 97.0% | +0.0% |

新卡不是在每个截断位置都更好。它救回了一些原本离查询较远的商品，但较长的摘要也会稀释
部分非常精确的词面匹配：200 个目标中 86 个排名改善、22 个不变、92 个变差。这说明
Dense 商品卡模板仍有调优空间，不能仅凭“文本更丰富”推断向量一定更好。

## 5. 完整多路召回结果

| 阶段 | 旧卡召回 | 新卡召回 | 变化 |
| --- | ---: | ---: | ---: |
| 通过 hard mask | 92.5% | 94.5% | +2.0% |
| Dense 路 | 76.5% | 78.5% | +2.0% |
| Lexical 路 | 71.5% | 82.5% | +11.0% |
| Facet 路 | 18.5% | 18.5% | +0.0% |
| 任一路找到 | 88.0% | 91.0% | +3.0% |
| 合并候选 Top 300 | 88.0% | 91.0% | +3.0% |
| 当前轻量 Top 10 | 53.5% | 56.5% | +3.0% |

最明确的收益来自 Lexical：新卡把原始字段里隐含或写法不统一的信息变成了更容易搜索的
事实和同义表达。Dense 有小幅净收益；Facet 总数持平但具体命中发生变化。最终候选召回
提升 3 个百分点，说明新卡整体可用，但还不应把 v1 模板视为最终版本。

## 6. 如何扩展到现有几千张卡

局部构建器不写死 200。后续把 sidecar 换成已有的几千张经验证商品卡即可：

- sidecar 里有卡的商品完全换新；
- 只重新编码这些商品；
- 没有卡的四万多商品继续使用旧卡和旧向量；
- 生成一个新的、与 sidecar 内容绑定的 Dense 索引。

扩展前建议先改进 Dense 专用文本模板：把最能区分商品的名称、品类、材质、颜色、闭合方式
和用途放在前面，把较长解释放在后面。Lexical/BGE 可以继续读取完整卡，不要求三个通道使用
完全相同的字符串视图。

## 7. 结果边界

这 200 张卡来自已知 public target pool，所以存在“目标池泄漏”：系统不知道某个 session
对应哪个目标，但知道这 200 件商品可能成为目标。结果只能标为商品卡诊断实验，不能与
完全不使用目标池信息的 50k benchmark 分数直接比较。

机器可读结果保存在：

- `artifacts/retrieval/public-200-product-card-dense-ab-v1.json`；
- `artifacts/retrieval/public-200-product-card-recall-ab-v1.json`。
