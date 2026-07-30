# 解冻与实现落地：对 D01/D02 的四处修正

**决策时间**：#R5（2026-07-30）
**状态**：✅ 已确认，代码已落地
**关联大纲**：[返回大纲](../outline.md) | 前置决策：[D01](./D01-attention-visualization-direction.md) · [D02](./D02-first-round-experiment-config.md)

---

## 📋 背景

D01/D02 于 2026-07-29 确认后暂缓，解冻条件是"等训练结束后启动"。**该条件已满足**：
M3 蒸馏续训完成（`log/ref_distill/checkpoint-4000`，6h14min，final step_loss 0.192），
M4 评测集已建成并复核（`datasets/eval_multiref/eval_set.json`，232 任务）。

解冻时按 D01/D02 逐条核对代码，核心技术判断**成立**：
`uno/flux/math.py:attention()` 确实是全部 57 个 block 的唯一汇聚点
（6 个调用点中 4 个是活路径且都带 ref 上下文；另 2 个所在的 processor
没有被任何 block 装配，是死代码），RoPE 在记录点前施加，
read 模式缓存 K/V 在函数内拼回，两条路径在记录点结构归一。

但有四处需要修正，其中两处会直接导致实验跑错或跑挂。

---

## ⚠️ 修正 ①：per-ref 边界在 hook 点拿不到 —— 外部注册从"待议"变为唯一解

D01 写"ref_isolation 架构下 ref 段边界明确，per-ref 归因无歧义"。
这在 `RefContext` 层面成立（`ref_attention.py:67` 有 `ref_lens: list[int]`），
**但在记录点不成立**：`_ref_attn_kwargs`（`layers.py:28-36`）只传了
`ref_len=sum(ref_lens)`。所以 `attention()` 内部只知道"末尾这一坨是 ref"，
**切不开 ref_1 / ref_2**。

而补传 `ref_lens` 属于修改既有 `.py`——M1–M4 全部结果都是这份代码产出的。
所以 outline 里列为"待讨论"的**外部注册布局**方案不是可选项，是唯一解：
driver 预先算出 `(txt_len, img_len, ref_lens)` 注册给录制器，
录制器每次调用硬断言 q/k 长度。

**为什么必须是硬断言而不是尽力而为**：段边界错一位，曲线照样画得出来，
而且看着挺合理。**静默的错误归因比崩溃危险得多。**

---

## ⚠️ 修正 ②：序列长度算错 2.5 倍，存储策略随之改为两级

D01 按 L≈2300 估算，那对应 `ref_size=320`。但本项目全线 `ref_size=512`
（`eval_multiref.py:477`、`smoke_eval.py:136`、`gen_data.py:525`），
且 T5 `max_length=512`（`pipeline.py:114`）：

| | txt | img | ref | **L** | fp16 概率矩阵/head | ×24 heads |
|---|---|---|---|---|---|---|
| D01 假设 | 512 | 1024 | 2×400 | 2336 | 10.6 MB | 0.25 GB |
| **实际 2-ref** | 512 | 1024 | 2×1024 | **3584** | 24.5 MB | **0.59 GB** |
| **实际 3-ref** | 512 | 1024 | 3×1024 | **4608** | 42.5 MB | **1.0 GB** |

D01 的结论（全存不可行）不变，但**显存峰值是它估计的 2.5 倍**，
这是唯一能把实验跑挂的数。实现上两条应对：

- **只算 img query 行**，并按 head 分块累加（`head_chunk=4`）→ 瞬态压到 ~2×59 MB
- **存储改两级**：曲线（每 step×block×seg 一个标量，~17 KB）必录；
  热力图只在指定 (block, step) 上留 img 网格（~245 KB）。
  D01 设想的"全 57 块 × 全步 × 全分辨率"是 17 MB/样本/变体 → 全轮 400 MB，进不了 git；
  拆开存后全轮约 **10.7 MB**，可直接提交。

---

## ⚠️ 修正 ③：D02 的验收产物 #3 在三件套下不可达 —— 补第四变体

D02 的预期产出 #3 是"kv_cache 无损性旁证：同 ckpt 同 seed 下 write/full 记录
vs read 记录的曲线一致性"。但 `sampling.py:239` 是
`mode = "write" if i == 0 else "read"`——**write 只发生在 step 0**，
没有"write 曲线"可比。要拿同 ckpt 的全程 full 曲线，必须另跑一条
`ref_isolation=True, kv_cache=False`，即 D02 明确否决、顺延二轮的第四变体。

**决策：补成四件套。** （`pipeline.py:254-255` 两个 flag 独立可控，
只有 `kv_cache=True` 单向强制 `ref_isolation=True`，所以这条路技术上通。）

```
official_full      full attention，无 cache     ← teacher 上界
ours_kv_pre        隔离 + cache，ckpt-20000     ← 故障现场
ours_kv_post4000   隔离 + cache，ckpt-4000      ← 修复效果
ours_iso_nocache   隔离，无 cache               ← 新增
```

注：D02 称第四变体为 `ours_full`，但它其实**不是 full attention**，
而是"隔离但不缓存"。按后者实现并改名，避免与 `official_full` 混淆。

**第四条线默认挂 `pre` bank**，因为 pre 能同时回答两个问题：

1. 缓存本身改没改变注意力行为（与 `ours_kv_pre` 同 ckpt 同 seed 逐点比）
2. 失败该归因到"隔离"还是"缓存"——**这个只有在故障现场问才有意义**

若 `ours_iso_nocache` 同样丢 ref_2 → 缓存无罪，归因到隔离/训练；
若不丢 → 缓存有嫌疑，那 M1–M4 的结论都要重新审视。
（缓存无损性由构造可证：ref 行掩码只看自己、`vec_ref` 固定 t=0。
但实测能抓实现 bug，而全部既有结果都出自这份代码。）

---

## ⚠️ 修正 ④：失败模式在 D01/D02 之后新增了"主体复制"

D01/D02 全篇假设的失败模式是"丢第二主体"。M4 冒烟复核后新识别出**主体复制**：
模型学到的 count prior（"该出现 N 个物体"）强于 binding（"ref j → slot j"）。

这对可视化是利好，但需要样本设计配合：
**丢失与复制在份额曲线上可能都表现为"ref_2 份额低"，难分；
但在空间热力图上签名不同**——复制会看到两处空间热区都热在 ref_1 上。

所以首轮样本必须含 **S2（同类对 bear_plushie + grey_sloth_plushie）**：
同类对上肉眼最难判、注意力最好判，正是可视化的独特价值区。

---

## ✅ 新增性质：D01/D02 的一条解读红线消失了

D01/D02 都写了"eager 记录状态与线上 flash 行为理论等价但数值有微小差异，
措辞需注明"。实现采用 **monkey-patch + 前向旁路**后这条不再适用：

wrapper 先原样调用真 `attention()` 拿前向输出，录制用的概率矩阵**另算**、
不参与前向。所以**开录与不开录出的图逐比特相同**——不用注明，因为没有差异。
代价是注意力算了两遍（D01 已预期 1.5–2x）。

本地已实测：patch 激活时输出与原函数 `torch.equal` 为真。
上机后 `run_attn_diag.py --verify_identical` 会与 M4 Stage B 的同名产物逐像素比，
max 必须为 0；非零则说明 patch 泄漏进了数值路径，那这条红线要复活。

另外顺带实证了 D01 的一个推断：**隔离掩码对 img 行的 softmax 影响为 0**
（`build_isolated_attn_mask` 对 txt/img 行全 True），
所以录制路径可以完全忽略 `attn_mask`——少一个出错点。

---

## ✅ 首轮样本（D02 说 5–8，取 8）

全部从 `eval_set.json` 按**原 seed** 取，因此与 M4 Stage B 的产物**是同一张图**。
这把 D02 隐含的"等评测结果再定样本"依赖消掉了——先后顺序不影响对齐。

| 样本 | 组合 | 作用 |
|---|---|---|
| S1_000_s0 | backpack_dog + bear_plushie | 异类对基线 |
| S1_011_s0 | bear_plushie + candle | 异类对，**对照组** |
| S1_022_s0 | berry_bowl + grey_sloth_plushie | 异类对 |
| S1_033_s0 | candle + grey_sloth_plushie | 异类对 |
| S2_000_s0 | bear_plushie + grey_sloth_plushie | 同类对，**复制探针** |
| S2_001_s0 | 同上，换模板 | 同类对 |
| S4_000_s0 | backpack_dog + bear_plushie + berry_bowl | 3-ref |
| S4_001_s0 | can + candle + clock | 3-ref |

S1 的 4 个按组合序号等距取（k = 0/11/22/33），落在 4 个不同模板上——
等距是为了跨 subject、跨模板铺开，不是挑好看的。

### 受控对照：槽位对齐优先于模板对齐

`S1_011`（bear_plushie + candle）vs `S2_000`（bear_plushie + grey_sloth_plushie）：
**同一 subject、同一槽位（都在 slot 0）**，只差搭档是否同类。

槽位必须对齐，因为**槽位本身就是被研究的变量**（ref_2 是否被丢/被复制），
槽位不同的对照会把"槽位效应"和"类别碰撞"混在一起。
而模板对不齐是结构性的：`template_id = k % 20`，bear_plushie 落在 slot 0 的组合是
k = 9..15（模板 9..15），S2 的组合只带模板 0..4，**eval_set 里不存在同模板配对**。
要同模板就得离开 eval_set 另造任务，那会丢掉"与 Stage B 同一张图"的 seed 对齐——
那个性质更值钱。故保槽位、弃模板；**解读时按对照说，不按严格控制变量说**。

---

## 📦 实现

| 文件 | 职责 | 运行位置 |
|---|---|---|
| `distill/attn_record.py` | 录制器：monkey-patch、布局注册、段归约、两级存储 | H800 |
| `distill/build_attn_tasks.py` | 从 eval_set.json 按固定规则挑 8 个样本 | 本地（已跑） |
| `distill/run_attn_diag.py` | 四变体 × 8 样本推理与录制 | H800 |
| `distill/plot_attn.py` | 渲染自包含 HTML 报告（内联 SVG + PNG） | 本地 |

**既有 `.py` 的 diff 为 0。** 靠的是 `layers.py:23` 的
`from ..math import attention` 是模块级名字绑定，运行时替换
`uno.flux.modules.layers.attention` 即可拦下全部活调用点。

出图脚本不依赖 matplotlib（只用 numpy + pillow），产物是单个 `.html`，
可直接提交、直接发人看。

### 本地验证（不需要 GPU）

- **录制器**：30 项行为测试全通。段归约与独立 float64 参考实现差 7e-8；
  各段份额之和恰为 1.000000；write/read 两条路径分别验证；
  6 条硬断言全部确认会抛；`head_chunk` 1/2/3/5 结果完全一致。
- **出图**：合成 npz 端到端 15 项全通，含 3-ref、复制签名、write/read 标注。

### 两处被测试抓出来的问题

1. 合成数据 fixture 把 ref2 的衰减写成与变体无关，导致 teacher 也失衡——fixture 自己的 bug。
2. **汇总指标本身有问题**：原先"late block 份额对全部 timestep 取均值"会把
   只发生在后段的塌陷稀释掉一半（合成数据里 pre 被摊成 0.507，几乎和 teacher 分不开）。
   而 D02 要回答的恰恰是"早期布局阶段没放置，还是后期细化阶段衰减"，
   把时间轴平均掉这个问题就答不了。**改为早/晚各三分之一分别报一次**，
   报告直接给出形态判定（布局阶段就没放置 / 细化阶段衰减 / 早段低但后期追回 / 均衡）。

---

## 🚀 上机执行

```bash
git pull
python distill/run_attn_diag.py --dry_run     # 先看计划：8 任务 × 4 变体 = 32 次
python distill/run_attn_diag.py               # 单卡约 11 min（含 ~7 min 加载）
python distill/run_attn_diag.py --verify_identical   # 纯 CPU，验录制未改前向
```

产物 `output/attn_diag/`（32 png + 32 npz + results.json，约 10.7 MB）提交回来，
本地跑 `python distill/plot_attn.py` 出报告。

---

## ⏸️ 顺延事项

- **扩量**：首轮 8 个样本的首要目标是验证记录管线（D02 原意）。管线通过后按 D02 扩到 15–20。
- **MultiBanana 难例**：沿用 D02，二轮泛化压力测试。
- **展示版形式**（GIF / 静态网格 / 交互式 HTML）：等诊断结论出来、知道要讲什么故事后再定。
  当前 HTML 报告是**诊断用**的，不是展示版。
- **反选样本**：M4 Stage B 的人工判定出来后，可按"丢失 / 复制 / 正常"各取若干条重跑
  （`run_attn_diag.py --tasks_json` 换一份 JSON 即可，不用改代码）。
  那时每条曲线都有已知人工标签作对照，能反过来验证"注意力份额"这把尺子准不准。

---

## ⚠️ 解读边界（沿用 D01，报告里已原样印出）

- 注意力份额高 **≠** 身份信息被转移。身份保真可能主要由 K/V **内容**承载，
  份额只反映"看了多久"，不反映"看到了什么"。结论措辞限于"注意力层面观察到…"。
- "eager vs flash 数值微差"这条按上面「新增性质」一节不再适用——但**前提是
  `--verify_identical` 实测 max = 0**。没实测过就不能这么说。
