# QU → Probe 真实全链路评测（v1）

日期：2026-08-29

> Simulator scope correction: the original 128-turn simulator cohort rotated
> `ask_attribute` across many fields. The public toy simulator does not interpret
> the assistant's natural-language question and its attribute classifier is too
> shallow for that trace to be the primary `C_t` convergence test. The corrected
> buying/browsing, `ask_attribute="other"` evaluation is documented in
> [Simulator other-only C_t 实测](./simulator-other-evaluation-v1.md). Natural-suite
> measurements and engineering-chain results in this document remain valid.

这次评测回答两个问题：

1. DeepSeek 能否把自然语言稳定地更新成 Session Context，并继续走完编译、硬筛和 Probe？
2. 当前的 `C_t` 是否真的能区分“意图模糊”和“意图具体”？

这里所有数字都来自真实运行，不是为了说明公式而手写的示例。

## 1. 测了多少

我们把现有两套 QU 数据全部送进了同一条链路：

- `natural-prompts-v0`：40 段对话，72 turn；
- `official-simulator-prompts-v0`：32 段四轮对话，128 turn；
- 合计：72 段对话，200 turn。

每一条实际执行：

```text
用户自然语言
  → DeepSeek v4 Flash
  → 更新 Session Context
  → Query Compiler 生成 q_lex / q_sem / hard constraints
  → hard-mask resolver
  → lexical + dense Probe
  → G_mode
  → C_t 与 D_t
```

多轮对话不是拆成独立句子运行的。每一轮都会读到该对话上一轮产生的 Session Context。

## 2. 运行成功率

| 项目 | 实测值 |
|---|---:|
| 数据集总 turn | 200 |
| 实际发起 QU 的 turn | 196 |
| QU 成功 | 193 / 196（98.5%） |
| 完整走到 Probe | 192 / 200（96.0%） |
| 成功得到可用 `C_t` | 186 / 200（93.0%） |
| QU `repair_exhausted` | 3 |
| 因同一会话上一轮失败而跳过 | 4 |
| 合法但不可检索 | 1 |
| 成功 QU trace 的 token 总数 | 614,577 |

自然语言 fixture 的严格逐 turn 语义断言为 `53 / 72` 通过；已执行并留下明细的单项断言为 `197 / 216` 通过。前一个标准更严格：同一 turn 只要有一项不符，整个 turn 就记为失败。

这说明两件事不能混为一谈：

- **协议成功率较高**：DeepSeek 大多数时候能给出可解析、可落地的状态更新；
- **行为并非全部符合 fixture**：目标名、偏好强弱、替换语义等细节仍有偏差。

## 3. `C_t` 总体实测值

在 186 个可计算样本中：

| 指标 | 实测值 |
|---|---:|
| 最小值 | 0.000 |
| 中位数 | 0.513 |
| 平均值 | 0.440 |
| 最大值 | 1.000 |

| 区间 | 数量 |
|---|---:|
| `[0.0, 0.2)` | 66 |
| `[0.2, 0.4)` | 10 |
| `[0.4, 0.6)` | 43 |
| `[0.6, 0.8)` | 28 |
| `[0.8, 1.0]` | 39 |

它不是所有样本都挤在同一个数附近，说明当前算法确实能产生区分度。但“有数值分布”不等于“数值含义正确”。

## 4. 一个完整、表现符合预期的真实例子

以下两轮来自 `c01_clarity_wedding_dress`。

### 第 1 轮：信息很少

用户说：

> I need something to wear to a wedding.

DeepSeek 更新后的 Session Context 是：

```yaml
goal: wedding attire
preferences:
  - semantic_text: Something appropriate to wear to a wedding
    strength: soft
version: 1
```

Query Compiler 生成：

```yaml
q_lex: wedding attire
q_sem: Looking for wedding attire. Preference: Something appropriate to wear to a wedding.
hard_constraints: []
```

没有硬条件，所以 50,000 个商品都有资格进入 Probe。Probe 的真实结果是：

```yaml
eligible_count: 50000
dense_count: 80
lexical_count: 80
mode_count: 79
G_mode: 0.270045
C_t: 0.068336
D_t: healthy
```

候选的语义模式很分散，因此 `C_t` 很低。

### 第 2 轮：用户补足细节

用户继续说：

> Make it a navy knee-length wrap dress with sleeves, no sequins, under $120.

DeepSeek 没有重建一份无关记忆，而是在上一轮 Session Context 上增加：

- `color = navy`，hard；
- `price < 12000` cents，hard；
- `style = wrap dress`，hard；
- knee-length、sleeves，语义 hard；
- no sequins，语义 negative hard。

编译后的核心内容是：

```yaml
q_lex: wedding attire navy wrap dress
q_sem: >-
  Looking for wedding attire. Preference: Something appropriate to wear to a wedding.
  Required color: navy. Required price: below USD 120.00.
  Required style: wrap dress. Requirement: Knee-length dress.
  Requirement: Dress with sleeves. Avoid: No sequins on the dress.
hard_constraints:
  - color = navy
  - price < 12000
  - style = wrap dress
```

硬筛过程：

```text
50,000
  → navy: 993
  → under $120: 987
  → wrap dress: 10
```

Probe 的真实结果是：

```yaml
eligible_count: 10
dense_count: 10
lexical_count: 10
mode_count: 8
G_mode: 0.371878
C_t: 0.600280
D_t: degraded
D_t_reasons: [dense_probe_underfilled]
```

因此这一组的 `C_t` 从 `0.068` 上升到 `0.600`，变化 `+0.532`。`D_t=degraded` 没有篡改 `C_t`，只是如实说明硬筛后只有 10 个候选，Probe 没取满 80 个。

## 5. 关键反例

手写的四组“模糊 → 具体”对照，实测如下：

| 对话 | 模糊 `C_t` | 具体 `C_t` | 变化 |
|---|---:|---:|---:|
| wedding dress | 0.068 | 0.600 | +0.532 |
| running shoes | 0.456 | 0.000 | -0.456 |
| carry-on | 0.516 | 0.000 | -0.516 |
| daily watch | 1.000 | QU 失败 | — |

Simulator 四轮对话的总体趋势也没有按故事预期上升：

| 对话轮次 | 可用样本 | `C_t` 平均 | `C_t` 中位数 |
|---:|---:|---:|---:|
| 1 | 32 | 0.484 | 0.553 |
| 2 | 29 | 0.459 | 0.543 |
| 3 | 29 | 0.441 | 0.507 |
| 4 | 28 | 0.422 | 0.510 |

还有两个直观反例：

- “I'm just looking around for something to wear to a summer wedding—nothing specific yet.” 得到 `C_t=0.910`；
- “Find me a navy silk wrap dress for women, under $120, and absolutely no sequins.” 得到 `C_t=0.395`。

所以目前不能对外声称“用户越具体，`C_t` 就稳定越高”。

## 6. 为什么会这样

当前 `C_t` 只看 Probe 结果内部的语义聚合程度。它没有直接计算“用户已经说明了多少偏好”，也没有直接使用候选数量。

这保持了我们原来的设计哲学，但当前实现暴露出一个问题：

> **剩余商品是否彼此相似，不总是等价于用户意图是否透明。**

硬筛后的商品可能很少但互相很不相似；反过来，一句很模糊的话也可能刚好召回一批文本非常相似的热门商品。商品重复上架、catalog 文本噪声、QU 对 category 的处理和 dense top-k 都会影响聚合度。

此外，当前标定区间很窄：

```text
low_anchor  = 0.256963
high_anchor = 0.448398
```

这使得轻微的 `G_mode` 变化容易被放大到接近 `0` 或 `1`。

## 7. 失败样本

三次 QU 错误都是 `repair_exhausted`：

- 具体手表描述 1 次；
- simulator 的 “I don't have a preference for category” 2 次。

后两次说明 `category` 的“无所谓”语义与当前系统协议存在边界冲突。因为失败发生在四轮 simulator 对话的第 2 轮，后续共 4 turn 没有使用不可信状态继续运行。

另有 6 条完整链路成功，但硬筛后只剩 1 个商品。单个商品不存在“候选之间的分散程度”，因此 `C_t` 正确地返回 unavailable，而不是伪造为 `0` 或 `1`。

## 8. 当前结论

QU → Session Context → Query Compiler → hard mask → Probe 的工程链路已经真实跑通，且协议成功率足够高，可以继续作为演示系统基础。

但是，当前 `C_t v1` 只能称为：

> **检索结果的语义收敛度**

它还不能直接称为已经验证过的“意图透明度”。下一轮设计需要保留“分散程度是核心”这个故事，同时解决结果聚合度与用户意图透明度不总一致的问题，再重新做成对测试。

## 9. 结果文件与复现

完整逐 turn 报告：

- `artifacts/retrieval/qu-to-probe-full-v1.md`
- `artifacts/retrieval/qu-to-probe-full-v1.json`

运行真实全链路：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/evaluate_qu_to_probe.py `
  --tier full `
  --cohort all `
  --api-key-file dpskapi `
  --output artifacts/retrieval/qu-to-probe-full-v1.json
```

把 JSON 渲染成逐条 Markdown 表：

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/render_qu_to_probe_report.py `
  --input artifacts/retrieval/qu-to-probe-full-v1.json
```
