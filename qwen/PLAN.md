# 用隔离注意力 + ref KV 缓存蒸馏 Qwen-Image-Edit-2511 · 执行计划

> 目标、背景、上一轮的账在 `GOAL.md`。这里讲怎么做。
> 2026-08-12 定稿,2026-08-13 按 Q1-B 结果修订(3-ref 的处置)。
>
> **这份文件给的是路线和已知的坑,不是规格。** 除了 §4 预登记那几条,
> 其余都是默认值——有更好的理由就改,把理由记在报告里即可。
> 与 `../distill/M4_EVAL_SPEC.md` §8/§9、`../distill/DISTILL_PLAN.md` §11 冲突时以它们为准。

---

## 0. 这一轮是什么

**一条直线,不是消融树。**

> 用「隔离注意力 + ref KV 缓存」的 Qwen 架构,把原始的、不带缓存的 Qwen-Image-Edit-2511
> 蒸馏成一个带缓存的学生;数据用现有的 UNO 数据集;然后用盲评量出质量代价。

teacher 和 student **是同一份权重**,只差三件事:student 多一组 LoRA、注意力多一张 mask、
推理时 ref 的 K/V 只算一次。所以这不是跨模型蒸馏,是**架构改造后的自我对齐**。

### 明确不做的

写下来是为了以后不用重新讨论。想做的话请先说服自己上一轮为什么砍掉它。

| 不做 | 理由 |
|---|---|
| M1 重新生成蒸馏数据 | 速度匹配不需要新目标图 |
| M2 人工过滤 | 速度匹配不断言目标图"好",只要 teacher 在该点的函数值 |
| 块对角 mask 臂 | 只训整段自注意一条臂。开关照写,消融不进本轮 |
| 分层隔离扫描 / 近似缓存 / ref K/V 漂移测量 | 它们是「免训练能不能用」那棵决策树的枝,树砍了 |
| 重跑 stage-1、加样本救判据、跨批相减 | `DISTILL_PLAN.md` §11.11(f) 已列明,继续有效 |

### 已翻案的一条:3-ref 回来了

原计划照搬 UNO 的决定把 3-ref 数据排除在外,理由是"teacher 自己在 3-ref 上
系统性不行"(`DISTILL_PLAN.md` §4.2,人工通过率 3.5%)。**Q1-B 证明这是上一个
teacher 的毛病,不是任务的毛病:**

| | UNO teacher | Qwen-2511(Q1 / Q1-B) |
|---|---|---|
| 1-ref | 90.0% | 12/12 |
| 2-ref | 51.0% | 21/28 = 75.0% |
| 3-ref | **3.5%** | **89/122 = 73.0%** |
| 2-ref → 3-ref | **掉一个数量级** | **平的**(Fisher p = 1.0) |

绝对值跨批不可比(不同分辨率、题面、判读场次,§8.5-3 照样管这里),
**但每列的斜率是各自批内测的,斜率可比**。详见 `../reports/20260812-q1b-3ref/REPORT.md`。

⇒ **3-ref 进训练集。** 详见 §3.2。

---

## 1. 地基:为什么在 Qwen 上只差一条边

隔离注意力要同时成立两条:**(a)** ref 只自注意(mask);**(b)** ref 段用固定 t=0 调制、
不跟去噪步走。两条缺一条,ref 的 K/V 就仍随步变化,缓存从无损降级成近似。

**(b) 在 2511 上是原生的,不用我们做。**

`Qwen-Image-Edit-2511/transformer/config.json` 里 `"zero_cond_t": true`。对应实现:

```python
# QwenImageTransformer2DModel.forward
if self.zero_cond_t:
    timestep = torch.cat([timestep, timestep * 0], dim=0)
    modulate_index = torch.tensor(
        [[0]*prod(sample[0]) + [1]*sum(prod(s) for s in sample[1:]) for sample in img_shapes], ...)
```

`_modulate(x, mod, index)` 按 index 逐 token 在两份调制参数里选:0 → 真实 t,1 → t=0。
而 pipeline 里 `img_shapes[0]` 是目标噪声图、`img_shapes[1:]` 是各参考图
(`pipeline_qwenimage_edit_plus.py:753`)。

⇒ **2511 训练时 ref 段就是 t=0 调制。这不是补丁,是权重认得的结构。**
上一轮在 UNO 上 (b) 是硬贴的,`official_iso` 身份归零(0/51),很可能相当一部分记在它头上。

### 由此得到的三条(会诊已独立复核)

1. **无损缓存的归纳成立,只差 mask。** 第 0 层 ref hidden = `img_in(VAE latent)`,步不变;
   调制 t=0;加 mask 后 ref 只在 ref 段内注意 ⇒ 输出步不变 ⇒ 逐层归纳,ref K/V 步不变。
   复核过的旁路:LayerNorm 全部 `elementwise_affine=False`;QK 是 per-token RMSNorm,
   无跨 token 统计量;`controlnet_block_samples` 不传就不生效;`guidance_embeds: false`
   且 `QwenTimestepProjEmbeddings` 里 `hidden_states` 只供 dtype ——
   **调制里没有 pooled text、没有 guidance 项**。
2. **cond / uncond 共享一份缓存。** ref 段看不见 txt ⇒ 两次前向的 ref K/V 逐位相同。
   Q1 口径 `true_cfg_scale=4.0`、40 步 = 80 次前向,**1 次写、79 次读**。
3. **两种 mask 写法都可无损缓存**(ref 段对注意力封闭即可)。所以 `GOAL.md` §6 那个叉口
   在新 teacher 上是纯质量取舍,不是"能不能缓存"的问题。

> 顺带记一句,本轮不做:UNO 的 `vec_ref` 含 `vector_in(y)`(pooled prompt,`uno/flux/model.py:214`),
> ref K/V 依赖 prompt;Qwen 的调制完全不含文本 ⇒ **ref K/V 真正与 prompt 无关**,
> 同一组参考图跨 prompt 复用缓存在这个架构上是成立的。

---

## 2. 主线

```
P0 上机前必查  →  P1 实现 + 等价性自检  →  P2 训练  →  P3 一批盲评
  (无 GPU)        (无 GPU / 1 张卡)        (8×H800)     (出图 + 人工判读)
```

P0 已完成(`../reports/20260812-q1b-3ref/REPORT.md`):`zero_cond_t` 在 env 里确实生效,
Q1 与 Q1-B 的产物已回仓。剩下的三步下面各讲一节。

---

## 3. 三步各自要做什么

### 3.1 P1 · 实现

四个新文件放本目录,**不动 `../uno/` 和 `../distill/` 的既有 `.py`**(R0)。
文件怎么切随意,下面只是一个够用的切法:

| 文件 | 职责 |
|---|---|
| `iso_attn.py` | attention processor;mask 构建;`RefKVCache` |
| `pipeline_iso.py` | 继承 `QwenImageEditPlusPipeline`,加 write / read 两种模式 |
| `train_iso.py` | 速度匹配训练 |
| `infer_iso.py` | 评测出图 |

**序列布局**:processor 内部是 `cat([txt, img])`,而 `img = cat([latents, image_latents])`,
所以有效序列是 `[ txt │ noise │ ref₁ │ ref₂ … ]` —— 与 UNO 同构,mask 的写法可以照搬语义。

**mask 默认整段自注意**(refs 互相可见)。理由:约束更弱、同样可无损缓存、
与 BFL `FLUX.2-klein-9b-kv` 同构可比。块对角留个开关,本轮不训。

#### 两个坑,踩上去很贵

1. `QwenDoubleStreamAttnProcessor2_0.__call__` 里有
   `raise ValueError("... does not accept an external attention_mask ...")`,
   只认 `encoder_hidden_states_mask` 那个 key 侧 padding mask。所以得换 processor,
   换的时候记得把原来那条 padding mask 合并进新的 `(B,1,L,L)`,别丢了。
2. **RoPE 平移。** `QwenEmbedRope.forward` 里
   `max_vid_index = max(h//2, w//2, max_vid_index)` 是对 `img_shapes` 里**所有**图取的,
   而 `txt_freqs = pos_freqs[max_vid_index : max_vid_index + txt_len]`。
   缓存读模式下序列里没有 ref,**若顺手把 ref 从 `img_shapes` 摘掉,`max_vid_index` 会变,
   txt 的 RoPE 整体平移,写/读两侧对不上**。这是静默 bug,只体现成质量下降。
   ⇒ 读模式仍按完整 `img_shapes` 算 `max_vid_index`,训练侧同理。

#### 等价性自检 —— 这是唯一的硬门禁

同一 seed 下,`隔离-无缓存` 与 `隔离-有缓存` 的输出应当在数值误差内相同。
上一轮有先例(`../scripts/bench_kv_cache.py`,iso vs kv 像素差 max=58 / mean=0.4681)。

**这个自检不过,后面所有数字都没有意义**——"缓存无损"是 §1 的结论,自检是它的实验对应物。

### 3.2 P2 · 训练

**目标函数 —— 速度匹配,不是图像 SFT。**

```
teacher = 同一份权重, disable_adapters(), 全注意力, no_grad
student = 同一份权重, LoRA 开,   隔离 mask

L = ‖ v_iso(x_t, t) − v_full(x_t, t).detach() ‖²      只在噪声图 token 位置上取
```

ref 位置的输出本来就被 `noise_pred[:, :latents.size(1)]` 丢掉,不进 loss。
权重共用一份(LoRA 开关切换身份),显存里**不多一份 40 GB 权重**,
只多一次 no_grad 前向的激活。

#### 数据

用 `../datasets/distill_multiref/manifest_raw.json` 的**全部 9000 条**
(1-ref 1000 + 2-ref 4000 + 3-ref 4000)。不用 `manifest_filtered.json`——
速度匹配不断言 `x₀` 好,M2 那 34.2% 的通过率在这里不构成约束。

**3-ref 为什么要带上**(2026-08-13 翻案,见 §0):不是因为那批数据变好了,
而是因为 **mask 的扰动强度随 ref 段长度单调涨**——1-ref 的 ref 段 4096 token,
3-ref 是 12288。只拿 1/2-ref 训练然后在 3-ref 上部署,等于在我们改动最大的
那个维度上外推。以前不敢用是因为那个 regime teacher 自己都不行、练了没意义;
Q1-B 之后它行了。

采样比自己定。一个合理的起点是照评测分布压 3-ref 的权重(比如 1-ref : 2-ref : 3-ref
= 3 : 5 : 2),但**别压到 0**——覆盖到就行,不需要均衡。

两条已知的将就,标定时留意,不用现在解决:

- **`x₀` 是 UNO 的 512² 产物,评测在 1024²。** 默认上采样到 1024² 训练
  (保评测分辨率一致,不引入第二个变量)。100 步标定时看 loss 会不会被放大伪影主导。
  但**不许为此改评测口径**。
- **3-ref 那 4000 条的 `x₀` 是 UNO 的坏图**,低 σ 端会落在 Qwen 流形外一点
  (高 σ 端几乎全是噪声,不受影响)。想彻底干净就用 Qwen 在 **train 20 主体**上补跑一批
  3-ref `x₀`——8 卡跑 500 张约 1.8 小时。Q1-B 那 122 张是 held-out,一张都不能进训练集。

refs 只能来自 dreambooth **TRAIN 20 主体**,held-out 10 个严格不进。
这条切分沿用 `DISTILL_PLAN.md` §2,**建议启动时加断言,泄漏即退出**。

#### 起手配置(都是默认值,标定后按实测调)

| | |
|---|---|
| init | stock 2511 权重 + 新建 LoRA(不从任何 checkpoint 续) |
| LoRA rank | 64 起。上一轮的 512 是 UNO 官方配方,不适用于这里的窄改造 |
| target modules | `to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out` |
| 分辨率 | 1024²(与评测一致) |
| t 采样 | 从推理用的 sigma 网格取(`FlowMatchEulerDiscreteScheduler`,按 seq_len 动态 shift,40 步)。理由:训练分布 = 部署分布 |
| prompt_embeds | 离线预算并缓存。它依赖 ref 图(VL 模板把 384² ref 编进去)但在去噪循环外只算一次 ⇒ 训练时不用挂 7B VL,省 ~14 GB |
| 梯度检查点 | 开 |
| 机器 | 直接申请 8×H800,**不走 infer_hub**(那是推理队列) |
| 断点续跑 | 长任务会被打断——上一轮臂 B 被 SIGHUP 打断两次,`nohup` 挡不住 `torchrun` |

**先跑 100 步标定,再定 batch / accum / 步数。** 量三样:**峰值显存**
(20B bf16 权重 40 GB + 12.7k token × 60 层的激活,80 GB 单卡偏紧)、**s/it**、
**loss 量级与下降形状**。

粗估供对照(**是估的,以标定为准**):按上一轮 UNO 12B @3584 token 的 5.4 s/it 折算,
params ×1.67、token ×3.55、多一次 teacher 前向 ×1.33 ⇒ 有效 batch 16 时约 40 s/it,
batch 8 时约 20 s/it。**速度匹配大概率比上一轮的 SFT 收敛快**——不是教新能力,
是补一条被切断的通路。所以先按 **1000–2000 步**规划,不要一上来就排 4000。

### 3.3 P3 · 评测

**三个变体,同批同 seed:**

| 变体 | 是什么 |
|---|---|
| `qwen_full` | stock 权重 + 全注意力 = teacher = Q1 口径 |
| `qwen_iso_pre` | stock 权重 + mask + 缓存,**未训练** = 训练的第 0 步基线 |
| `qwen_iso_post` | 训练后的 LoRA + mask + 缓存 |

`qwen_iso_pre` 直接对上一轮 P-probe 的 `official_iso` = 0/51 可比 —— 它回答
「(b) 原生之后,单加 mask 还会不会归零」。**它是对照,不是闸门**:无论结果如何,P2 照跑。

#### 评测集大小 —— 这里的数字有理由,别随便调小

默认取 `m6_tasks.json` 里每层 `i % 4 != 3`,得 S1 165 + S3 75 = **240 条**。
已核:这 240 条完整包含 Q1 的全部 40 条,也完整包含 §5.1 的 7 条豁免集。

> **为什么是 240 不是 192**:§8.2 要 `n_nontie ≥ 94`,上一轮**两次**卡在 93 和 89
> (臂 B 93、边 ③ 89)。192 在平局率 50% 时 `E[n_nontie] = 96`,达标概率只有六成上下;
> 240 在 50–55% 平局率下才稳。多花约 0.6 GPU·h,**别第三次演 93/94**。

**同批加 30 条 `run_floor`**(同权重、同 seed、异 run)。不做就读不出差异是模型差异
还是会话漂移——上一轮每一批都做,臂 B 那次批内天花板 30/30 正是它把结论钉住的。

**可以考虑加一个 3-ref 层。** `m6_tasks.json` 里没有 3-ref,但既然训练带上了 3-ref、
Q1-B 又证明 teacher 在这个 regime 能用,不测就等于放着最大的扰动维度不看。
成本比看起来低:`q1b_3ref_tasks.json` 那 122 条的 `qwen_full` 臂**已经渲染好了**
(`output/qwen_3ref/`),只需要补两条 iso 臂。做不做自己判,做的话记得在预登记里
单列这一层,别和 S1/S3 混进同一个判据。

#### 两把尺子

- **§8 偏好盲评**(`M4_EVAL_SPEC.md` §8.2):主判 `qwen_iso_post` vs `qwen_full`,
  240 对 + 30 条 run_floor 同批混判。判据见 §4。分层 S1/S3 必报但单层不作判据;
  S1 要报组合级 ICC 与 `deff`。
- **§9 客观身份留存计数**(`M4_EVAL_SPEC.md` §9):`qwen_iso_pre` 与 `qwen_iso_post` 都算。
  崩坏一眼可辨时偏好盲评失效(§8.5-2),这一层必须用客观计数。

**§5.1 那 7 条豁免集**只用于 **§9 客观计数**的解读(teacher 自己就做不到的,不算 student 的错),
**不用于 §8 偏好盲评**——偏好是成对比较,teacher 图差的时候两边同样受影响,是对称的。
另外要声明:豁免集只在 Q1 的 40 条上标注过,240 条里其余 200 条的 teacher 失效没标。
不为此扩大标注,写进局限。

---

## 4. 预登记 —— 全文只有这一节是硬的

**写下就不许改,只能加带日期的订正注记。** 这是上一轮 D02「份额失衡比」的教训:
判据一旦能在看到结果之后调整,整批数字就不再是证据。

1. **达标判据**:`M4_EVAL_SPEC.md` §8.2 原文,**一字不改地沿用**——
   非平局胜率 Wilson 95% CI 下界 ≥ 0.40 且 `n_nontie ≥ 94`。**非劣性判据,不是优越性判据。**
2. **样本不足时结论是「判据不适用」,不是「不达标」。** 不许事后追加样本。
3. **平局率单列,不并进主指标。** 参照点是本批的 `run_floor`,不是上一轮的数字
   (§8.5-3:跨批次不得并排引用)。
4. **速度预测,出任何图之前写死在这里:**

   | | 理论 | 预测实测 |
   |---|---|---|
   | 2-ref @1024² | **2.74×** | **1.9–2.0×** |
   | 1-ref @1024² | 1.94× | ~1.4× |

   推导:token 账 = txt(估 400–600,标定时实测)+ 噪声 4096 + ref 2×4096 ⇒ L ≈ 12.7k;
   缓存读的 79 次前向 query 只有 txt+噪声 ≈ 4.5k、key 仍 12.7k ⇒ 每层开销比 0.357;
   80 /(1 + 79×0.357) = 2.74×。折算系数取上一轮实测/理论 = 1.672/2.33 = 0.72。
   若加 3-ref 层,那一层的预测在出图前补登记。

5. 若平局率高到判据不适用,读数落回上一轮的**「平局率 vs 批内 run_floor 天花板」**口径。
   现在就登记,免得到时候临时找说法。

---

## 5. 边界

- **不动 `../uno/` 与 `../distill/` 的既有 `.py`/`.sh`**(R0)。要复用就 import,要改就在本目录新写。
- **不改 Q1 口径**(steps 40 / true_cfg 4.0 / 1024² / negative_prompt `" "` / prompt 原样)。
  P3 沿用同一口径,改了基线就废了。
- **`.gitignore` 白名单模式**:`output/` 默认全忽略,逐批显式放行且写明理由。
  **未判读的批次,带变体名的拼图不许进 git**(盲评纪律)。

远程 agent 按 `../distill/REMOTE_AGENT_HANDBOOK.md` 的红/黄/绿三档走。
**本计划不构成黄档规格**——黄档要求常量、枚举规则、seed 公式写死到没有自由度
(参照 `M4_EVAL_SPEC.md` 的粒度),这份文件是刻意留白的,给不了那个粒度。
要放权给远程 agent 写 `infer_iso.py`,得先单独补一份 `QWEN_EVAL_SPEC.md`。
