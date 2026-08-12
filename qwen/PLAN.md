# 用隔离注意力 + ref KV 缓存蒸馏 Qwen-Image-Edit-2511 · 执行计划

> **目标、背景、上一轮的账在 `GOAL.md`。这里只讲怎么做。**
> 定稿 2026-08-12,经第四次会诊复核。
> 规则性条文沿用 `../distill/M4_EVAL_SPEC.md` §8 / §9 与 `../distill/DISTILL_PLAN.md` §11,
> **本文件与它们冲突时以它们为准**;本文件只新增本轮特有的规格。

---

## 0. 这一轮是什么

**一条直线,不是消融树。**

> 用「隔离注意力 + ref KV 缓存」的 Qwen 架构,把原始的、不带缓存的 Qwen-Image-Edit-2511
> **蒸馏**成一个带缓存的学生;数据用现有的 UNO 数据集;然后用盲评量出质量代价。

teacher 和 student **是同一份权重**,只差三件事:student 多一组 LoRA、注意力多一张 mask、
推理时 ref 的 K/V 只算一次。所以这不是跨模型蒸馏,是**架构改造后的自我对齐**。

### 明确不做的

上一轮的方法学里,下面这些在本轮**不做**,写下来是为了以后不用重新讨论:

| 不做 | 理由 |
|---|---|
| M1 重新生成蒸馏数据 | 用现有数据。速度匹配不需要新目标图 |
| M2 人工过滤 | 速度匹配不断言目标图"好",只要 teacher 在该点的函数值 |
| 块对角 mask 臂 | 只训整段自注意一条臂(§3.1)。开关照写,消融不进本轮 |
| 分层隔离扫描 / 近似缓存 / ref K/V 漂移测量 | 它们是「免训练能不能用」那棵决策树的枝,树砍了 |
| 3-ref 训练数据 | 评测里没有 3-ref;那批目标图 UNO teacher 通过率只有 3.5%。⚠️ **Q1-B 若显示 Qwen 的 3-ref 明显能打,这条要重议**——那说明"3-ref 不行"是上一个 teacher 的毛病,不是任务的毛病 |
| 重跑 stage-1、加样本救判据、跨批相减 | `DISTILL_PLAN.md` §11.11(f) 已列明,继续有效 |

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

## 2. 主线四步

```
P0 取回 Q1 凭据  →  P1 实现 + 等价性自检  →  P2 训练  →  P3 一批盲评
   (无 GPU)          (无 GPU / 1 张卡)        (8×H800)     (1 张卡出图 + 人工判读)
```

**P0 必须排在写训练代码之前**:Q1 是所有 student 数字的锚点,豁免集也出自它,
而"比 UNO 好不少"目前在仓库里零凭据(`GOAL.md` §9)。

---

## 3. 规格

### 3.0 P0 · 取回 Q1 凭据 + 两条上机前必查

**取回**:H 机 `/kaimm-distill/wuwenxuan/output/qwen_baseline` 下的 `results.json`
与拼图,落进仓库;补一份 Q1 结果报告(跑了什么口径、40 条是哪 40 条、§5.1 那 7 条的图)。
拼图按 `.gitignore` 白名单逐批显式放行并写明理由(白名单已开好,见根 `.gitignore` 末尾)。

**顺带一件事:Q1-B —— teacher 的 3-ref 能力探底。** 122 条(held-out 全部 112 个合法
3-组合),8 卡分片,任务表 `../datasets/eval_multiref/q1b_3ref_tasks.json` 已在 git 里。
上一轮在 UNO 上 3-ref 是系统性失效的(M1 人工通过率 3.5%、M4 的 S4 层 75% 平局且
"大概率是一样烂"),所以「3-ref 能不能做」至今只有一个坏答案。这是**探底不是验收**:
没有判据、不进 §8.2、不做盲评,产出是图和拼图。执行单:`Q1B_3REF_RUN.md`。
它同时是 P3 那 720 张图要用的 8 卡分片脚手架的第一次实跑。

**必查一(阻塞级)**:env 里的 diffusers 认不认 `zero_cond_t`。
`/kaimm-distill/wuwenxuan/envs/qwen-edit` 装的是 0.40.0.dev0 的某个快照,而 `zero_cond_t`
是较晚合入的。**若它不认这个 config 键,`from_pretrained` 会静默丢弃,ref 就不是 t=0 调制**
——那样 §1 的地基在实际运行环境里不成立,而且 **Q1 基线本身就是在错的语义下跑出来的**。

```bash
python - <<'PY'
import inspect, diffusers
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as M
print(diffusers.__version__, 'zero_cond_t' in inspect.signature(M.__init__).parameters)
PY
# 加载后再确认一次实例上真的是 True:
#   print(pipe.transformer.config.zero_cond_t, pipe.transformer.zero_cond_t)
```

不通过 ⇒ 先升 diffusers,并**重跑 Q1 基线**,再谈后面。

**必查二**:训练机。**infer_hub 是推理队列,训练不走它**——P2 直接申请机器训。
要确认的是:拿到的是不是 8×H800 80GB、UNO 仓库和 `distill_multiref/images/` 在不在盘上、
env 是不是同一个 `qwen-edit`(§3.0 必查一 那条对训练机同样要跑一遍)。
断点续跑仍然是硬要求,但理由从"队列超时"变成"长任务本来就会被打断"
——上一轮臂 B 就被 SIGHUP 打断过两次(`nohup` 挡不住 `torchrun`)。

### 3.1 P1 · 实现

四个新文件,都在本目录,**不动 `../uno/` 也不动 `../distill/` 的既有 `.py`**:

| 文件 | 职责 |
|---|---|
| `iso_attn.py` | 替换 attention processor;mask 构建;`RefKVCache` |
| `pipeline_iso.py` | 继承 `QwenImageEditPlusPipeline`,加 write / read 两种模式 |
| `train_iso.py` | 速度匹配训练 |
| `infer_iso.py` | 评测出图(三变体,断点续跑,逐样本容错) |

**序列布局**:processor 内部是 `cat([txt, img])`,而 `img = cat([latents, image_latents])`,
所以有效序列是 `[ txt │ noise │ ref₁ │ ref₂ ]` —— 与 UNO 同构,mask 的写法可以照搬语义。

**mask**:`build_ref_mask(txt_len, img_len, ref_lens, mode)`,`mode ∈ {"segment", "block"}`,
**默认 `"segment"`(整段自注意,refs 互相可见)**。选它的理由:约束更弱、同样可无损缓存、
且与 BFL `FLUX.2-klein-9b-kv` 同构可比。`"block"` 只是留个开关,本轮不训。

**两个陷阱(实现时必须处理,不是提醒)**:

1. `QwenDoubleStreamAttnProcessor2_0.__call__` 里有
   `raise ValueError("... does not accept an external attention_mask ...")`,
   只认 `encoder_hidden_states_mask` 那个 key 侧 padding mask。**必须换 processor**,
   而且换的时候要把原来那条 padding mask 合并进新的 `(B,1,L,L)` 里,不能丢。
2. **RoPE 平移**。`QwenEmbedRope.forward` 里
   `max_vid_index = max(h//2, w//2, max_vid_index)` 是对 `img_shapes` 里**所有**图取的,
   而 `txt_freqs = pos_freqs[max_vid_index : max_vid_index + txt_len]`。
   缓存读模式下序列里没有 ref,**若顺手把 ref 从 `img_shapes` 摘掉,`max_vid_index` 会变,
   txt 的 RoPE 整体平移,写/读两侧对不上**。这是个静默 bug,只体现成质量下降。
   ⇒ **读模式必须仍按完整 `img_shapes` 算 `max_vid_index`。训练侧同理**
   (训练时 ref 在场、推理读模式 ref 不在场,两边必须用同一份完整 `img_shapes`)。

**等价性自检(门禁,不过不许往下走)**:同一 seed 下,`隔离-无缓存` 与 `隔离-有缓存`
的输出应当在数值误差内相同。上一轮有先例(`../scripts/bench_kv_cache.py`,iso vs kv
像素差 max=58 / mean=0.4681)。**这个自检不过,后面所有数字都没有意义**——
因为"缓存无损"是 §1 的结论,自检就是它的实验对应物。

### 3.2 P2 · 训练

**目标函数 —— 速度匹配,不是图像 SFT。**

```
teacher = 同一份权重, disable_adapters(), 全注意力, no_grad
student = 同一份权重, LoRA 开,   隔离 mask

L = ‖ v_iso(x_t, t) − v_full(x_t, t).detach() ‖²      只在噪声图 token 位置上取
```

ref 位置的输出本来就被 `noise_pred[:, :latents.size(1)]` 丢掉,不进 loss。
权重共用一份(LoRA 开关切换身份),所以显存里**不多一份 40 GB 权重**,
只多一次 no_grad 前向的激活。

**数据 —— 现有 UNO 数据集,不新生成、不人工过滤。**

`../datasets/distill_multiref/` 的 `manifest_raw.json`,取 **2-ref 4000 条 + 1-ref 1000 条
= 5000 条**;3-ref 4000 条不用(理由见 §0)。采样比按评测分布 **2-ref : 1-ref ≈ 69 : 31**
(与 P3 的 S1:S3 = 165:75 一致)。

- refs 来自 dreambooth **TRAIN 20 主体**,held-out 10 个严格不进——这条切分沿用
  `DISTILL_PLAN.md` §2,启动断言,泄漏即 `sys.exit`;
- `x₀` 是上一轮 UNO teacher 生成的 512² 目标图。**不用 `manifest_filtered.json`**:
  速度匹配不断言 `x₀` 好,只需要它落在合理的图像流形附近,所以 M2 那 34.2% 的通过率
  在这里不构成约束。
- ⚠️ **分辨率错配是本轮已知的最大数据瑕疵**:`x₀` 是 512²,而评测在 1024²。
  默认处置是**上采样到 1024² 训练**(保评测分辨率一致,不引入第二个变量)。
  100 步标定时看 loss 量级和曲线是否被放大伪影主导;若明显是,再议——
  但**不许为此改评测口径**。

**配置**:

| | |
|---|---|
| init | stock 2511 权重 + **新建** LoRA(不从任何 checkpoint 续) |
| LoRA rank | 64(标定项;上一轮的 512 是 UNO 官方配方,不适用于这里的窄改造) |
| target modules | `to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out` |
| 分辨率 | 1024²(与评测一致) |
| t 采样 | 从推理用的 sigma 网格取(`FlowMatchEulerDiscreteScheduler`,按 seq_len 动态 shift,40 步)。理由:训练分布 = 部署分布 |
| prompt_embeds | **离线预算并缓存**。它依赖 ref 图(VL 模板把 384² ref 编进去)但在去噪循环外只算一次 ⇒ 训练时不用挂 7B VL,省 ~14 GB |
| 梯度检查点 | 开 |
| 机器 | **直接申请 8×H800,不走 infer_hub**(那是推理队列) |
| 断点续跑 | **硬要求**。长任务会被打断——上一轮臂 B 被 SIGHUP 打断两次 |

**先跑 100 步标定,再定 batch / accum / 步数。** 这是这个项目一贯的做法,不是形式。
标定要量三样:**峰值显存**(20B bf16 权重 40 GB + 12.7k token × 60 层的激活,
80 GB 单卡偏紧)、**s/it**、**loss 量级与下降形状**。

粗估供标定时对照(**是估的,以标定为准**):按上一轮 UNO 12B @3584 token 的 5.4 s/it
折算,params ×1.67、token ×3.55、多一次 teacher 前向 ×1.33 ⇒ 有效 batch 16 时约 40 s/it,
batch 8 时约 20 s/it。**预期速度匹配比上一轮的 SFT 收敛快得多**——不是教新能力,
是补一条被切断的通路。所以先按 **1000–2000 步**规划,不要一上来就排 4000。

### 3.3 P3 · 评测

**评测集 240 条,规则确定性且已本地核算过**:

```
在 m6_tasks.json 内,每层按数组原始顺序取  i % 4 != 3
  S1: 220 × 3/4 = 165        S3: 100 × 3/4 = 75        合计 240
```

已核:这 240 条**完整包含 Q1 的全部 40 条**,也**完整包含 §5.1 的 7 条豁免集**。

> **为什么是 240 不是 192**:§8.2 要 `n_nontie ≥ 94`,上一轮**两次**卡在 93 和 89
> (臂 B 93、边 ③ 89)。192 在平局率 50% 时 `E[n_nontie] = 96`,达标概率只有六成上下;
> 240 在 50–55% 平局率下才稳。多花约 0.6 GPU·h 和几分钟判读,**别第三次演 93/94**。

**同批加 30 条 `run_floor`**(同权重、同 seed、异 run)。不做就读不出差异是模型差异
还是会话漂移——上一轮每一批都做,臂 B 那次批内天花板 30/30 正是它把结论钉住的。

**三个变体,同批同 seed**:

| 变体 | 是什么 |
|---|---|
| `qwen_full` | stock 权重 + 全注意力 = teacher = Q1 口径 |
| `qwen_iso_pre` | stock 权重 + mask + 缓存,**未训练** = 训练的第 0 步基线 |
| `qwen_iso_post` | 训练后的 LoRA + mask + 缓存 |

`qwen_iso_pre` 直接对上一轮 P-probe 的 `official_iso` = 0/51 可比 —— 它回答的是
「(b) 原生之后,单加 mask 还会不会归零」。**它是对照,不是闸门**:无论结果如何,P2 照跑。

**两把尺子分工**:

- **§8 偏好盲评**(`M4_EVAL_SPEC.md` §8.2):主判 `qwen_iso_post` vs `qwen_full`,
  240 对 + 30 条 run_floor 同批混判。判据 = **非平局胜率 Wilson 95% CI 下界 ≥ 0.40
  且 n_nontie ≥ 94**;分层 S1/S3 必报但单层不作判据;S1 必须报组合级 ICC 与 `deff`。
- **§9 客观身份留存计数**(`M4_EVAL_SPEC.md` §9):`qwen_iso_pre` 与 `qwen_iso_post` 都算。
  崩坏一眼可辨时偏好盲评失效(§8.5-2),这一层必须用客观计数。

**§5.1 那 7 条豁免集的用法**:只用于 **§9 客观计数**的解读(teacher 自己就做不到的,
不算 student 的错),**不用于 §8 偏好盲评**——偏好是成对比较,teacher 图差的时候
两边同样受影响,是对称的,不需要豁免。
⚠️ 并且要声明:豁免集只在 Q1 的 40 条上标注过,**240 条里其余 200 条的 teacher 失效没标**。
不为此扩大标注,写进局限。

**速度验收(预登记的预测,判读前写死)**:

| | 理论 | 预测实测 |
|---|---|---|
| 2-ref @1024² | **2.74×** | **1.9–2.0×** |
| 1-ref @1024² | 1.94× | ~1.4× |

推导:token 账 = txt(估 400–600,**标定时实测**)+ 噪声 4096 + ref 2×4096 ⇒ L ≈ 12.7k;
缓存读的 79 次前向 query 只有 txt+噪声 ≈ 4.5k、key 仍 12.7k ⇒ 每层开销比 0.357;
80 /(1 + 79×0.357) = 2.74×。折算系数取上一轮的实测/理论 = 1.672/2.33 = 0.72。

---

## 4. 判据与预登记

**写下就不许改,只能加带日期的订正注记。** 这条是上一轮 D02「份额失衡比」的教训。

1. **达标判据**:`M4_EVAL_SPEC.md` §8.2 原文,**一字不改地沿用**——
   CI 下界 ≥ 0.40 且 `n_nontie ≥ 94`。非劣性判据,不是优越性判据。
2. **样本不足时结论是「判据不适用」,不是「不达标」。** 不许事后追加样本。
3. **平局率单列,不并进主指标。** 参照点是本批的 `run_floor`,不是上一轮的数字
   (§8.5-3:跨批次不得并排引用)。
4. **速度预测**(§3.3 那张表)在出任何图之前写死在这里。
5. 若平局率高到判据不适用,读数落回上一轮的**「平局率 vs 批内 run_floor 天花板」**口径。
   ——这一条现在就预登记,免得到时候临时找说法。

---

## 5. 边界

沿用 `GOAL.md` §10,不重复,只强调三条在本轮特别容易破的:

- **不动 `../uno/` 与 `../distill/` 的既有 `.py`/`.sh`**(R0)。要复用就 import,
  要改就在本目录新写。上一轮 `infer_qwen_edit.py` 就是这么做的。
- **不改 Q1 口径**(steps 40 / true_cfg 4.0 / 1024² / negative_prompt `" "` / prompt 原样)。
  P3 的 240 条沿用同一口径,改了基线就废了。
- **`.gitignore` 白名单模式**:`output/` 默认全忽略,逐批显式放行且每条写明理由。
  **未判读的批次,带变体名的拼图不许进 git**(盲评纪律)。

远程 agent 按 `../distill/REMOTE_AGENT_HANDBOOK.md` 的红/黄/绿三档走。
本计划**不构成黄档规格**——黄档要求"常量、枚举规则、seed 公式、超参写死到没有自由度"
(参照 `M4_EVAL_SPEC.md` 的详细程度),本文件还没到那个粒度。
要放权给远程 agent 写 `infer_iso.py`,得先按那个标准补一份 `QWEN_EVAL_SPEC.md`。
