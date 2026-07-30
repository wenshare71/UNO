# 讨论：推理时参考图-主图注意力演化可视化

> 状态：⏸️ 已暂缓（等训练结束后启动）| 轮次：R4 | 日期：2026-07-29

## ⏸️ 暂缓说明

方向（D01）与首轮实验配置（D02）均已确认。用户决定等训练结束后再启动实现，
届时按 D01/D02 直接转入实现规划即可。恢复时需重新确认：

- [ ] 蒸馏后 ckpt 的实际路径与步数
- [ ] held-out DreamBooth 多 ref 组合的具体 prompt 清单（实现期生成）

## ⚪ 待讨论（恢复时）

- [ ] 记录开关的接入方式（外部注册布局 + attention() 内记录的两段式结构，实现期确定）
- [ ] 展示版形式：GIF 动画 / timestep×block 静态网格 / 交互式 HTML（等诊断结论后定）

## ✅ 已确认

- 总体方向：诊断+展示两阶段复用，覆盖标准与 kv_cache 双路径 → [D01](./decisions/D01-attention-visualization-direction.md) (#R2)
- hook 位置：`math.py:attention()` 统一汇聚点，在线降维存储 → [D01](./decisions/D01-attention-visualization-direction.md) (#R2)
- 对比矩阵：三件套（official_full / 蒸馏前 ours_kv / 蒸馏后 ours_kv）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 样本：held-out DreamBooth 10 subject 多 ref 组合，每变体 5-8 个 → [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 启动时机：训练结束后再启动，决策文档备查 (#R4)

## ❌ 已否决

- 存完整注意力矩阵（57 blocks × 50 steps 体量不可行）→ [D01](./decisions/D01-attention-visualization-direction.md) (#R2)
- 两件套（无 teacher 上界参照）/ 首轮四件套（ours_full 留作二轮补充）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 首轮用 MultiBanana（偏离训练分布，留作二轮泛化验证）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)
- 首轮 15+ 样本（管线未验证前不值得扩量）→ [D02](./decisions/D02-first-round-experiment-config.md) (#R3)

## 📁 归档

| 问题 | 结论 | 详情 |
|------|------|------|
| 首要目的 | 诊断+展示两阶段复用 | [→ D01](./decisions/D01-attention-visualization-direction.md) |
| 覆盖路径 | 标准 + kv_cache 双路径，记录点结构归一 | [→ D01](./decisions/D01-attention-visualization-direction.md) |
| 首轮实验配置 | 三件套 × held-out DreamBooth × 5-8 样本 | [→ D02](./decisions/D02-first-round-experiment-config.md) |
| 启动时机 | 训练结束后启动，文档备查 | 本轮 (#R4) |
