# Shopping Copilot 组内讲解材料

这组文档用于向组员解释当前系统的五块核心基础设施：

1. 我们怎样保存多轮购物记忆；
2. 我们怎样从 50k catalog 建立可信 facet，又怎样从用户语言抽取 facet；
3. Query Understanding 怎样把一句自然语言变成可提交的状态更新。
4. Fuzzy Intent Volume 怎样从商品空间计算意图透明度 $T_t$，并用 $D_t$ 说明测量质量；
5. 正式检索怎样执行 hard mask、三路召回、RRF 和 $T_t$ 控制的向量多样化。

这些是讲解材料，不替代各模块的 normative contract。发生冲突时，以链接的设计 contract 和当前代码
为准。

## 推荐阅读与讲解顺序

| 顺序 | 文档 | 核心问题 | 建议时间 |
| ---: | --- | --- | ---: |
| 1 | [`01-session-context.md`](01-session-context.md) | 系统到底记住什么，状态怎样安全变化 | 8 分钟 |
| 2 | [`02-facet-system.md`](02-facet-system.md) | facet 从哪里来，什么算 catalog truth | 12 分钟 |
| 3 | [`03-query-understanding.md`](03-query-understanding.md) | DeepSeek 读什么、输出什么、本地怎样落地 | 10 分钟 |
| 4 | [`04-intent-transparency.md`](04-intent-transparency.md) | Intent Volume 怎样观察 catalog，$T_t/D_t$ 分别表示什么 | 10 分钟 |
| 5 | [`05-formal-retrieval.md`](05-formal-retrieval.md) | 三路怎样找候选、怎样融合，$T_t$ 怎样改变 Top-10 | 8 分钟 |

QU 的逐字段长例子仍保留在：
[`session-context-flow-example.md`](../design/query_understanding/session-context-flow-example.md)。

## 一句话系统故事

> Catalog Semantic 告诉系统“哪些商品属性可以相信”；Query Understanding 判断“用户这一轮表达了
> 什么”；Session Context 保存“到这一轮为止用户真正想要什么”；Probe 再观察这个意图在商品空间中
> 还剩多少有效空间并形成意图透明度 $T_t$；正式检索再用同一个 $T_t$ 决定结果应该展开还是聚焦。

## 四块模块如何连接

```text
冻结的 50k catalog
    ↓ 只读 profiling、人工 Gate A/B、runtime release
Catalog Semantic
    │
    ├── 可信 category scopes
    ├── catalog-verified facet capability
    └── grounding + Gateway authority
                         ┌───────────────────────────────┐
用户自然语言             │                               │
    ↓                    ▼                               │
Query Understanding → ResolvedTurnIntent                │
    ↑                    │                               │
旧 SessionContext ───────┘                               │
                         ↓                               │
                  Query Compiler / Probe / Retrieval     │
                         ↓                               │
                  回答 + TurnRecord                      │
                         ↓                               │
                  新 SessionContext ─────────────────────┘
```

## 讲解时最重要的五个辨析

### 1. Session Context 不是聊天文本拼接

它是不可变、可校验、可回放的结构化快照。历史只能追加，当前意图只能通过带版本号的 typed batch
更新。

### 2. 用户偏好不等于商品事实

“用户不要皮革”是一个 intent；“商品 A 是否真的是皮革”是 catalog/retrieval evidence。前者由 QU
理解，后者不能由模型猜测。

### 3. Catalog-side facet construction 不等于 QU-side facet extraction

- construction：离线决定 catalog 中哪些 facet 有证据、来源在哪里、在哪些类别适用、允许怎样使用；
- extraction：在线判断用户这句话表达了哪些条件。

QU 只能使用当前 SessionContext facet policy 已声明的 structured vocabulary：catalog-verified facet
走 release grounding，retrieval-derived facet 走本地 registry 规范化；未知或不可靠内容保留为
semantic preference。后者可以结构化表达用户要求，但不等于 catalog truth。

### 4. DeepSeek function call 不等于直接执行本地函数

DeepSeek 返回的是一个原生 `tool_call`，其中 `arguments` 是 JSON 字符串。本地 decoder 消费参数，
materializer 规划状态更新，Gateway 再验证转换；它不是远程 RPC 自动执行同名 Python 函数。

### 5. Preview 不等于 commit

QU 当前产出的是经过 Gateway 验证的新 `IntentState` 预览。它现在可以继续生成 `CompiledQuery`
并进入检索视图；应用仍要完成 $T_t$、正式检索、回答和 `TurnRecord`，最后才能在
transaction 中原子提交新的 Session Context。

## 当前真实实现状态

已经实现：

- Session Context 数据模型、reducer、snapshot、校验与 transaction；
- 50k catalog 的只读 profiling、category build、price Gate A/B、runtime release；
- Catalog Semantic Gateway 与 release-bound grounding；
- DeepSeek V4 Flash 原生 function call；
- structured / price / semantic typed QU 协议；
- materializer、Gateway preview 和一次 repair；
- 确定性 Query Compiler、逐 preference trace；
- 固定 Top-80 Lexical + Dense + semantic-mode Probe；
- 50k Retrieval Evidence Index、hard-mask resolver，以及和固定 Probe 的同 mask 接线；
- 旧 Top-80 mode-coherence $C_t$ 兼容路径；
- 完整 catalog 上的 Fuzzy Intent Volume $T_t$ runtime v1，以及独立健康诊断 $D_t$；
- hard-mask-first 的 Dense / Lexical / Facet 三路正式召回、RRF 和 $T_t$-aware 向量 MMR；
- 真实 50k catalog 上的单路、两路、三路消融与逐商品 route contribution 日志；
- 自然语言与官方 simulator prompt suite。

尚未形成 production 闭环：

- `starter.Agent.respond()` 尚未调用 QU；
- Intent Volume / $T_t$ 已形成独立 vertical slice，但尚未进入本轮 transaction；
- QU、Intent Volume、正式检索、回答、`TurnRecord` 与 commit 的应用层 orchestration 尚待实现。

## 权威资料入口

- Session Context：[`design/session_context/contract-v1.md`](../design/session_context/contract-v1.md)
- Catalog Semantic：[`design/catalog_semantic/contract-v0.md`](../design/catalog_semantic/contract-v0.md)
- Catalog 实施流程：[`design/catalog_semantic/methodology-v0.md`](../design/catalog_semantic/methodology-v0.md)
- Query Understanding：[`design/query_understanding/contract-v1.md`](../design/query_understanding/contract-v1.md)
- Query Compiler：[`design/query_compiler/contract-v0.md`](../design/query_compiler/contract-v0.md)
- Intent Transparency：[`design/intent_purity/runtime-contract-v1.md`](../design/intent_purity/runtime-contract-v1.md)
- Formal Retrieval：[`design/retrieve/formal-multi-route-v0.md`](../design/retrieve/formal-multi-route-v0.md)
- 官方问题边界：[`official_problem/README.md`](../official_problem/README.md)
