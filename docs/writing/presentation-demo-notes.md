# PPT / Demo 展示备忘录

本文件记录 Devpost 正文中不会完整展开，但必须通过 PPT、录屏或现场 Demo 让评委直观看到的内容。

## 1. Intent Transparency 随购物过程变化

### 要展示的核心现象

1. 用户从模糊需求逐步补充条件时，合理商品空间收窄，`T_t` 随之提高。
2. 用户删除条件或改变商品目标时，系统根据新的完整 Session Context 重新计算 `T_t`。
3. `T_t` 不随对话轮数强制单调上升；真正的意图改变可以使它下降或迁移到新的商品空间。

### 建议演示脚本

| 轮次 | 用户输入 | 预期系统变化 | 画面重点 |
| --- | --- | --- | --- |
| 1 | “I need something for a summer wedding, but I’m not sure what.” | `T_t` 较低；保留多个语义方向 | 多样化商品、较宽的意图空间、多个召回中心 |
| 2 | “Something comfortable that I can wear all day. Nothing gold.” | Session Context 增加舒适度要求和排除条件；`T_t` 上升 | 偏好状态 diff、商品空间收窄、方向数减少 |
| 3 | “Simple silver earrings under $80.” | 意图进一步集中；`T_t` 继续上升 | 召回转向聚焦、Ranking 更重视偏好匹配 |
| 4 | “Actually, forget the earrings. I need waterproof boots.” | 识别目标切换，移除过时方向并重新计算 `T_t` | transition 标记为 `moved`；曲线发生跳变；新商品空间重新展开或聚焦 |

### PPT / Demo 画面

- 一条随轮次变化的 `T_t` 折线或动态仪表盘。
- 每轮并排展示简化后的 Session Context diff。
- 用商品点云、语义中心或商品卡分组表示意图空间从分散到集中。
- 第四轮不要画成算法失效；明确标记为 **Intent Changed / Recomputed**。
- 同时展示召回方向数随 `T_t` 改变，例如 `6 → 4 → 1 → recompute`。

### 现场讲解句

> More conversation does not automatically mean more understanding. Transparency rises when the shopping space genuinely narrows—and is recomputed when the shopper changes direction.

### 演示准备注意事项

- 最终录制前使用真实运行值替换“较低 / 上升”等描述，不手工伪造 `T_t`。
- 提前固定一条能够稳定体现 `narrower → narrower → moved` 的演示轨迹。
- 保留每轮完整日志，以便技术问答时展示 Session Context、`N_t`、`T_t`、方向数和最终结果。
