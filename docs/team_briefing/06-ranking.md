# 06：Ranking 怎么选

## 一句话结论

我们实测后没有用一个“更高级”的模型把 RRF 全部替掉：

> RRF 负责给出可信候选边界，cross-encoder 负责判断单件商品是否更相关，MMR/DPP 负责让整个
> Top-10 在低 $T_t$ 时展开、在高 $T_t$ 时聚焦。

这三件事是不同问题。

## 测了哪些办法

同一个 Top-80 商品池上，我们比较了：

1. RRF、相对分数相加、CombMNZ；
2. Qwen3-Reranker-0.6B、BGE-reranker-v2-m3；
3. 直接 Top-K、MMR、DPP、自动语义方向 xQuAD。

所有方法都在 hard mask 之后运行；“不要黑色”之类的约束不会被 ranking 模型重新放回来。

## 最容易理解的结果

### BGE 最会把正确商品往前提

target 已经在 Top-80 时，BGE 把它放进最终 Top-10 的比例是 `63.2%`；RRF 是 `38.2%`。但是 BGE
给出的十件商品也更像彼此，平均只覆盖 `1.20` 个审计大类。

所以 BGE 很适合回答“我已经很清楚要什么，请帮我排准”，不适合独自回答“去北海道有什么可以买”。

### DPP 最能体现 $T_t$

DPP 把 Top-10 看成一个整体：十件都不错但几乎相同，不是一个好集合。低 $T_t$ 时它加强商品间的
排斥，高 $T_t$ 时加强单件相关性。

在 160 个官方请求上，99.4% 都满足“低 T 比高 T 更分散”，而 MRR 没有可靠下降。这是目前最强的
故事结果。

### xQuAD 没成功

我们尝试只靠商品向量自动找 6 个潜在方向，再覆盖不同方向。实际只有 40% 的请求符合低 T 更分散，
所以这个版本淘汰。数学名字更 fancy，不代表在我们的商品向量空间里更好。

## 当前系统是否已经换成 BGE+DPP

还没有。当前 production-like controller 仍是：

```text
RRF Top-80 → T-aware MMR → Top-10
```

实验分别证明了“BGE 相关性最好”和“DPP 的 T 响应最好”，但还没有直接跑 **BGE+DPP** 组合。下一轮
把这个组合与当前链路同池比较，胜出后才修改正式 controller。

详细数字、置信区间和失败案例见
[`ranking-strategy-evaluation-v0.md`](../design/retrieve/ranking-strategy-evaluation-v0.md)。
