# Query Understanding 现状诊断（v2 实测后）

> 后续状态：本文记录的是 `query_understanding_v1_2` 的失败基线。文中 P0/P1/P2 修复已在
> `query_understanding_v1_3` 落地；原失败样本复测达到 24/24 回合与 49/49 关键语义断言通过。
> 原始数字仍保留，供修复前后复测对照。详见 [`v1-3-regression-results.md`](v1-3-regression-results.md)。

## 结论先行

目前 QU 的总体方向没有问题：DeepSeek 负责理解整轮自然语言并提出“本轮结束后的完整意图”，本地系统负责校验、生成状态操作并提交 Session Context。

但现在还不能把 QU 当作已经完成。主要问题不是 DeepSeek 不会调用工具，而是**模型看到的协议与本地真正接受的状态之间有几处断层**。这些断层在普通的“不断增加条件”对话中不明显，却会集中破坏“取消条件、放宽条件、非价格数值条件”——而这些恰好是意图空间扩大或收缩实验最需要测准的行为。

这轮只做诊断，没有修改 QU 实现。

## 这次实际测到了什么

扩大后的自然语言测试集包含：

- 60 段对话；
- 130 个用户回合；
- 其中包括逐步收紧、主动放宽、保持稳定和更换购买目标。

首次完整运行结果：

- 120 / 130 个回合通过本地协议并跑通 Probe；
- 9 个回合在一次修复重试后仍失败；
- 1 个后续回合因为前一轮失败而跳过；
- 60 段对话中有 51 段完整跑完；
- “主动放宽”对话只有 4 / 10 完整跑完。

这里的 120 / 130 只是“结构合法并被系统接受”，**不是 120 / 130 都理解正确**。目前这套扩大测试没有为每个自然语言回合填写语义断言，因此它发现不了“模型把一句条件悄悄漏掉”或“被取消的条件仍藏在 goal 中”。

## 真正的问题

### 1. `dont_care_facets` 的公开协议和本地协议不一致

模型工具 schema 只说 `dont_care_facets` 是字符串数组，Prompt 还允许模型使用“其他可命名 facet”。但本地提交状态时，只接受 registry 中已经注册的 facet ID。

因此用户说“长度无所谓”“不要求低跟了”时，模型很自然地生成：

```json
{
  "dont_care_facets": ["length", "heel_height"]
}
```

工具调用在 JSON/schema 层完全合法，到了本地才被拒绝，因为 `length` 和 `heel_height` 并不是当前注册 facet。

实测失败包括：

| 对话 | 模型生成的非法 facet | 本地结果 |
| --- | --- | --- |
| 放宽项链条件 | `stones`，修复后又生成 `metal`、`motif` | 两次都失败 |
| 放宽靴子条件 | `heel_height` | 修复后原样重犯 |
| 放宽项链宝石/长度 | `length`，修复后生成 `gemstone` | 两次都失败 |

这里还有一个重要语义点：取消一条条件不一定等于“整个 facet 都无所谓”。例如 `feature` 中同时记录了 `waterproof` 和 `low heel`，用户只取消 `low heel` 时，正确操作应是删掉对应 preference，而不是把整个 `feature` 设为 don't-care，否则会连 `waterproof` 一起覆盖掉。

所以这不是简单加几个别名就能彻底解决的问题。协议必须明确区分：

1. 删除某一条旧 preference：从 `keep_active_refs` 中省略它；
2. 用户声明整个已注册维度都无所谓：才写入 `dont_care_facets`；
3. 不得为子属性临时发明新的 don't-care facet。

### 2. 非价格数值条件没有合法表示法

用户说“表盘 40 mm 或更小”时，模型生成 `<=` 是合理的。但当前协议规定：

- structured preference 只允许 `eq / neq / in / not_in`；
- `lt / le / gt / ge` 只允许用于 price。

所以模型想表达的意思是对的，输出却必然被 wire decoder 拒绝为 `structured_relation_required`。普通模式下一次修复仍然重犯；开启 DeepSeek strict tool schema 后也连续两次失败。

这证明 strict tool call 不能解决该问题：schema 只能限制“输出长什么样”，不能替系统补出一个本来不存在的合法语义通道。

另一次重放中，这个回合虽然通过了协议，但 `40 mm` 条件被直接丢掉。这比显式失败更危险，因为系统会把它记为成功。

这里需要二选一：

- 正式支持非价格数值 facet 和范围运算；或
- 明确要求这类条件进入 semantic preference，并给模型一个原生示例。

Hackathon 阶段第二种更小、更稳，但应由我们正式决定，不能让模型临场猜。

### 3. `goal` 会偷偷保留已经取消的条件

当前 Prompt 要求：

- 第一次确定商品任务时使用 `goal.action=switch`；
- 仍在找同一种商品时只能 `goal.action=keep`；
- 颜色、材质、价格等变化不算 goal switch。

问题是，模型第一轮经常把约束也写进 goal，例如：

- `red leather closed-toe heels`
- `yellow waterproof raincoat for child`
- `navy midi dress with sleeves`

下一轮用户取消颜色、材质、防水或袖子条件时，preferences 的确删掉了，但 goal 按协议只能 `keep`，旧文字不能改。Query Compiler 又会把 goal 放进语义查询，于是被取消的条件继续影响检索。

在 4 个表面上完整成功的“放宽”对话中：

- 3 个存在这种 stale goal（过期目标文字）泄漏；
- 只有 `crossbody bag` 那个 goal 本身足够干净。

因此“放宽场景 4 / 10 跑完”仍然高估了真实语义正确率。就这 4 个成功样本而言，只有 1 个是干净的放宽。

根因不是 compiler，而是 goal 协议只有 `keep / switch`，没有“仍是同一类商品，但重新表述一个去约束化目标”的能力。

### 4. 当前 repair 信息不足以让模型修对

本地修复反馈大致只有：

```text
local_error=invalid_final_state;
path=dont_care_facets.2;
details=facet=stones
```

它没有告诉模型：

- 合法 facet 到底有哪些；
- `stones` 是否应映射为 `feature`；
- 这个场景其实应该只省略旧 ref，而不应新增 don't-care；
- 同一数组中是否还有其他非法值。

所以实测会出现“删掉一个非法词，再换成另一个非法词”，或者原样重复。当前校验还倾向于一次只报告第一个问题，唯一的一次 repair 机会很容易浪费。

### 5. 现有评测和日志会掩盖问题

QU service 在最终异常中保留了 `last_error`、`last_path` 和 `last_detail_*`。但 QU→Probe 评测脚本把这些内容丢掉了，只记录异常类型和一句 `repair_exhausted`，同时把 `qu_attempts` 写成空数组。

因此原始端到端 artifact 只能看到“失败”，看不到模型究竟提交了什么、第一次为什么失败、第二次如何修坏。

此外，扩大测试集目前没有 critical semantic assertions，所以：

- 漏掉 `40 mm` 仍可能算成功；
- stale goal 仍可能算成功；
- 数值指标按错误的 Session Context 继续运行，最后看起来仍然“方向正确”。

## 模型随机性有多大

把首次失败的 9 段对话单独重放后，有 4 段恢复；后续单独重放时，还有一个腕表样本也曾恢复，但漏掉了 `40 mm` 条件。

这说明当前失败由两部分组成：

1. DeepSeek 输出存在不可完全消除的波动，即使 temperature 为 0；
2. 上述协议断层会让某些自然表达稳定地没有正确出口。

不能靠“多重试几次”代替修协议。重试可以缓解随机错误，但对于错误的合法空间，只会在失败、误修和静默漏条件之间随机选择。

## 建议的修复顺序

下面只是下一步建议，本轮没有实施。

### P0：先让失败可见

- QU→Probe 日志保留结构化错误详情和每次尝试的安全元数据；
- 对测试用 provider 记录 decoded frame，避免只看到最终 `repair_exhausted`；
- 不记录 API key 等秘密信息。

这是低风险改动，否则后面每次改 Prompt 都无法可靠判断改好了什么。

### P1：修三个协议断层

1. 在 turn input 或 tool schema 中明确给出合法 `dont_care_facets`；
2. Prompt 明确“删除 preference”和“整个 facet 无所谓”的区别；
3. 让 goal 保持去约束化，或增加同任务下的 `revise/restate` 行为；
4. 明确非价格数值范围进入 structured 还是 semantic，不能继续悬空。

### P2：让 repair 变成定向修复

- 一次返回所有非法 don't-care 值；
- 附合法 facet 列表；
- 对“应省略 ref”这种错误给出明确动作；
- 再考虑是否把总尝试次数从 2 增加到 3。

### P3：补语义回归测试

至少把本轮暴露的失败固定成回归用例：

- 取消项链长度/宝石条件；
- 取消靴子颜色、预算和低跟，但保留防水；
- 腕表 40 mm 或更小；
- 取消条件后，goal 和最终编译查询中不得残留旧约束。

下一轮不能只看 tool-call/contract success，还要看最终 Session Context 和编译查询是否保留、删除了正确的信息。

## 对当前状态的准确判断

QU 的基础架构可以保留，不需要推倒重做；Session Context reducer 也不是主要问题。真正该修的是 DeepSeek 与本地 reducer 之间的契约边界。

简单说：**模型大多数时候听懂了人话，但我们还没有把所有常见意思都给它准备好合法、唯一、可执行的写法。**
