# 讨论：推理时参考图-主图注意力演化可视化

> 状态：▶️ 首轮已跑完（负结果），curve 拆 head 后待二次上机 | 轮次：R6 | 日期：2026-07-29 起，R6 于 2026-07-31

## ⚠️ 首轮结果（#R6）：管线全绿，但 D02 那把尺子被证伪

32 次录制 0 失败，工程侧全部达标。但**"late-block ref 份额失衡比"测不出主体是否真的出现**：
8 个样本里有 7 个，唯一确定渲染正确的 teacher 被判为四变体中最差；
S1_022 上 `ours_kv_pre` 肉眼确认丢了主体，失衡比却是 0.986（"均衡"）。
→ curve 拆到 head 维度重跑一轮，并预登记了证伪判据 → [D04](./decisions/D04-per-head-recording.md)

副产品（正面）：`ours_iso_nocache` 与 `ours_kv_pre` 整条曲线逐点最大差 0.014，
**缓存在注意力份额这一层无罪**，归因指向隔离/训练 → [D04](./decisions/D04-per-head-recording.md)

## ▶️ 解冻说明（#R5）

暂缓条件是"等训练结束后启动"，**已满足**：M3 蒸馏续训完成（`log/ref_distill/checkpoint-4000`），
M4 评测集已建成并复核。解冻时按 D01/D02 逐条核对代码，核心技术判断成立
（`math.py:attention()` 确为 57 个 block 的唯一汇聚点，两条路径在记录点结构归一），
但有四处需修正 → [D03](./decisions/D03-implementation-corrections.md)。

恢复时要重新确认的两项均已关闭：

- [x] 蒸馏后 ckpt 的实际路径与步数 → `log/ref_distill/checkpoint-4000`
- [x] held-out DreamBooth 多 ref 组合的具体 prompt 清单
      → **不必实现期生成**，直接从 `datasets/eval_multiref/eval_set.json` 按原 seed 取，
      于是诊断样本与 M4 Stage B 的产物是同一张图 → [D03](./decisions/D03-implementation-corrections.md)

## ⚪ 待讨论（下一轮）

- [ ] **二次上机结果的判读**：per-head 是否证实"少数身份头被平均摊薄"。
      证伪则转去查 K/V **内容**（比对 ref token 的 value 向量），不再拆更细
      → 判据已预登记在 [D04](./decisions/D04-per-head-recording.md)
- [ ] `--verify_identical` 仍未跑（需 Stage B 的 711 张 PNG，只在 H800 上）
- [ ] 扩量：首轮管线验证通过后，按 D02 扩到 15–20 样本
- [ ] 反选样本：M4 Stage B 人工判定出来后，按「丢失 / 复制 / 正常」各取若干条重跑，
      用已知人工标签反过来验证"注意力份额"这把尺子准不准（换 `--tasks_json` 即可，不改代码）
- [ ] 展示版形式：GIF 动画 / timestep×block 静态网格 / 交互式 HTML
      （等诊断结论后定；当前 HTML 报告是**诊断用**，不是展示版）
- [ ] MultiBanana 难例泛化压力测试（D02 顺延）

## ✅ 已确认

- 总体方向：诊断+展示两阶段复用，覆盖标准与 kv_cache 双路径 → [D01](./decisions/D01-attention-visualization-direction.md) (#R2)
- hook 位置：`math.py:attention()` 统一汇聚点，在线降维存储 → [D01](./decisions/D01-attention-visualization-direction.md) (#R2)
- 样本来源：held-out DreamBooth 多 ref 组合，每变体 5-8 个 → [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 启动时机：训练结束后再启动，决策文档备查 (#R4)
- **记录开关的接入方式**：零侵入 monkey-patch（替换 `layers.attention` 的模块级绑定）
  + 布局外部注册 + 每次调用硬断言 → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- **对比矩阵改为四件套**：补 `ours_iso_nocache`（隔离无缓存，挂 pre bank），
  否则 D02 自己的验收产物 #3 不可达 → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- **首轮 8 个样本**：4×S1 + 2×S2 + 2×S4，全部从 eval_set.json 按原 seed 取；
  受控对照保槽位对齐、弃模板对齐 → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- **存储改两级**：曲线全录（标量），热力图只在指定 (block, step) 留网格；
  全轮 10.7 MB 而非 400 MB → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- **曲线拆到 head 维度**：`(step, block, seg)` → `(step, block, head, seg)`，
  热力图维持 head 平均；全轮 28 MB → [D04](./decisions/D04-per-head-recording.md) (#R6)

## ❌ 已否决

- 存完整注意力矩阵（57 blocks × 50 steps 体量不可行）→ [D01](./decisions/D01-attention-visualization-direction.md) (#R2)
- 两件套（无 teacher 上界参照）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- ~~首轮四件套（ours_full 留作二轮补充）~~ → **#R5 推翻**：D02 的验收产物 #3
  在三件套下不可达（write 只发生在 step 0），第四变体必须进首轮
  → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- 首轮用 MultiBanana（偏离训练分布，留作二轮泛化验证）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 首轮 15+ 样本（管线未验证前不值得扩量）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 改 `math.py` / `layers.py` 加记录钩子（既有代码产出过 M1–M4 全部结果，改它等于改历史）
  → 走 monkey-patch，既有 `.py` diff 为 0 → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- 用 matplotlib 出图（本地无该依赖）→ 内联 SVG + PNG 的自包含 HTML，只需 numpy + pillow
  → [D03](./decisions/D03-implementation-corrections.md) (#R5)
- 汇总指标对全部 timestep 取均值（会把只发生在后段的塌陷稀释一半，
  而"早期布局没放置 vs 后期细化衰减"正是 D02 的核心问题）→ 改早/晚各三分之一分别报
  → [D03](./decisions/D03-implementation-corrections.md) (#R5)

## 📁 归档

| 问题 | 结论 | 详情 |
|------|------|------|
| 首要目的 | 诊断+展示两阶段复用 | [→ D01](./decisions/D01-attention-visualization-direction.md) |
| 覆盖路径 | 标准 + kv_cache 双路径，记录点结构归一 | [→ D01](./decisions/D01-attention-visualization-direction.md) |
| 首轮实验配置 | 三件套 × held-out DreamBooth × 5-8 样本 | [→ D02](./decisions/D02-first-round-experiment-config.md) |
| 启动时机 | 训练结束后启动，文档备查 | #R4 |
| 解冻与四处修正 | per-ref 边界／序列长度／第四变体／新失败模式 | [→ D03](./decisions/D03-implementation-corrections.md) |
| 实现与本地验证 | 4 个新文件；录制器 30 项 + 出图 15 项测试全通 | [→ D03](./decisions/D03-implementation-corrections.md) |
| 首轮 32 次录制 | 工程全绿；份额失衡比与肉眼判定系统性反号 | [→ D04](./decisions/D04-per-head-recording.md) |
| 缓存归因 | 份额层面缓存无罪（逐点最大差 0.014），指向隔离/训练 | [→ D04](./decisions/D04-per-head-recording.md) |
| 二次改动 | curve 拆 head；旧 npz 自动识别重跑；判据预登记 | [→ D04](./decisions/D04-per-head-recording.md) |
