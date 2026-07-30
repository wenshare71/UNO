# 注意力演化可视化：总体方向确认

**决策时间**：#R2
**状态**：✅ 已确认
**关联大纲**：[返回大纲](../outline.md)

---

## 📋 背景

### 问题/需求

训练（蒸馏续训）结束后进入评测阶段。现有定量指标（CLIP/DINO 分数、MultiBanana VLM judge）只能回答"丢没丢第二主体"，无法回答"什么时候、在哪一层、怎么丢的"。希望在模型推理时将参考图与主图之间的注意力演化过程可视化。

### 约束条件

- 模型为 FLUX DiT：19 double + 38 single blocks，24 heads，去噪步数通常 25~50
- 存在两条推理路径：标准 full/isolation 路径（ref token 在序列内）与 kv_cache 路径（step 0 后 ref token 不在序列内，主图查询缓存 K/V）
- 不能显著拖慢正常推理，可视化必须是可选开关
- kv_cache 下必须开 eager 计算才能拿到注意力矩阵，只能用于少量诊断样本

---

## 🎯 目标

1. **诊断**：验证/证伪"丢第二主体"的注意力层面假设——第二主体是布局阶段（早期 step）就没被放置，还是后期身份细化阶段衰减
2. **展示**：产出直观的图/动画，用于汇报与论文 supplementary
3. **附加验证**：确认 kv_cache 本身是否引入注意力行为差异（理论上数学无损，实测验证）

---

## 📊 方案对比

| 维度 | 候选 | 决策 |
|------|------|------|
| 首要目的 | 诊断 / 展示 / 两者都要 | ✅ 两者都要（先诊断后展示，一次投入两处复用） |
| 推理路径 | 只标准 / 只 kv_cache / 两条都要 | ✅ 两条都要（同时验证缓存是否引入行为差异） |
| hook 位置 | 逐 processor 注入 / `math.py:attention()` 统一汇聚点 | ✅ `attention()` 统一汇聚点 |
| 存储策略 | 存完整注意力矩阵 / 在线降维只存分段统计 | ✅ 在线降维（完整矩阵 57 块 × 50 步不可行） |

### hook 位置的关键依据

`uno/flux/math.py` 的 `attention()` 是全部 57 个 block 的唯一汇聚点，且：

- `cache_key` / `ref_len` / `attn_mask` / `ref_kv` 已经穿参到位
- read 模式下缓存 K/V 在函数内 `torch.cat` 拼回（math.py:47），**两条路径在记录点处结构归一**——同一套记录代码天然覆盖 write/full/read 三种情况
- RoPE 已在记录点前施加（math.py:38），记录的注意力含位置信息，可直接解释

### 存储策略的关键依据

512² 主图 + 2 张 ref 的序列长约 2300 token，单 head 单 block 注意力矩阵约 10 MB(fp16)，57 blocks × 50 steps 全存不可行。记录时在线计算 `softmax(QK^T/√d + mask)` 后立即按段聚合：

- 输出 = 每个主图 token(query) 对 [txt, ref_1..ref_N] 各段的注意力质量，heads 取均值
- 主图 query 可 reshape 回 2D 网格 → 每张 ref 一张"主图哪些区域在看它"的热力图
- 单 block 单 step 仅 ~img_tokens × (N+1) 个 float，全 57 块 50 步也只有几十 MB

---

## ✅ 最终决策

### 选定方案

**两阶段投入**：

1. **诊断版**：少量 held-out 样本，对比蒸馏前/后 ckpt（及 official_full 基线），输出 per-ref 注意力份额曲线（timestep × block 二维）+ 代表层热力图
2. **展示版**：诊断结论明确后，挑典型样本做 GIF 动画/拼图，进入报告

### 决策理由

- 注意力汇聚点单一（`math.py:attention()`），hook 成本低、对正常路径零侵入（可选开关）
- ref_isolation 架构下 ref 段边界明确，per-ref 归因无歧义——这是相比普通 full-attention 模型做注意力可视化的结构性优势
- 双路径记录点归一，kv_cache 无损性验证是顺手产物

### 预期效果

- 把"丢主体"从定性观察变为可量化的时序指标（per-ref 注意力份额曲线）
- 蒸馏前/后对比可直接展示修复机制，而不只是分数变化

---

## ⚠️ 解读边界（写入报告的措辞红线）

- 注意力权重高 ≠ 身份信息被转移（身份保真可能主要靠 K/V 内容），注意力图是诊断线索而非因果证据
- kv_cache 可视化必须开 eager 注意力，"可视化状态下的行为"与"线上 flash 行为"理论上等价但数值有微小差异，措辞需注明

---

## 🔗 后续待议

- 对比矩阵：哪些 ckpt/变体进入对比（official_full / 蒸馏前 ours_kv / 蒸馏后 ours_kv / ours_full）
- 首批诊断样本：来源（held-out DreamBooth vs MultiBanana）与规模
- 展示版形式：GIF 动画 / timestep×block 静态网格 / 交互式 HTML
