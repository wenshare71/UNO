# P1 侦察:把写 `iso_attn.py` 需要的东西取回本地 · 临时执行单

> 给远程 agent。**先读 `../distill/REMOTE_AGENT_HANDBOOK.md` §2.0(红/黄/绿三档)。**
> 上下文:`qwen/PLAN.md` §3.1(P1 要写什么)、`qwen/GOAL.md` §8(机器与环境)。
>
> **这是一次性的侦察单,做完即弃。** 本轮**只取东西、只清点、不跑 GPU、不写新脚本**。
> 全部动作落在绿档:读文件、`cp`、`ls`、`pip freeze`、写 `reports/**.md`。
> **一个既有 `.py` / `.sh` 都不动(R0),一张图都不生成。**

---

## 0. 为什么是这一轮,以及为什么它这么小

我在本地要写 `qwen/iso_attn.py`(注意力 processor + mask + `RefKVCache`),
而本地机器 **没有 diffusers、没有 torch**,仓库里也没有 vendored 的 Qwen 源码。
`PLAN.md` §1 和 §3.1 引用的所有行号、`_modulate` 的签名、
`QwenDoubleStreamAttnProcessor2_0.__call__` 的参数表——**我一行都读不到**。
不先把源码取回来,写出来的就是盲写。

所以本轮**刻意不含任何需要加载模型的探针**:那些探针的 API 我现在猜不准,
写出来大概率在远程崩在第 40 行,白烧一个来回。**读完源码之后我会写第二份单子**
(带 GPU 探针:`named_modules` 实名、txt token 实测、sigma 网格、注意力后端),
那时候脚本可以写死到没有自由度。

**本轮零 GPU、零风险、不可能失败。** 有做不到的项就照 §7 报"未取到"+ 原因,
**不要猜、不要替代、不要顺手跑别的**。

远程仓库路径以下记作 `$R`(上一轮是 `/kaimm-distill/wuwenxuan/UNO`,以实际为准),
env 记作 `$E = /kaimm-distill/wuwenxuan/envs/qwen-edit`,
权重记作 `$W = /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511`。

---

## 1. A · diffusers 源码快照 → `qwen/_vendor/`

**为什么进 git**:`PLAN.md` §1 那三条结论(无损缓存的归纳、cond/uncond 共享缓存、
两种 mask 都可缓存)是**逐行读这份源码得出的**,而 env 里是 `0.40.0.dev0` 的某个快照,
不是任何 release。快照一换,论证就可能不成立。把它钉进仓库,那三条才是可审计的。
diffusers 是 Apache 2.0,与本仓库同许可,可以 vendored。

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
SP=$E/lib/python3.11/site-packages
D=$R/qwen/_vendor/diffusers_0.40.0.dev0
mkdir -p $D/models/transformers $D/models $D/pipelines/qwenimage $D/schedulers
```

**必取(五个文件,路径相对 `$SP/diffusers/`,目录结构原样保留)**:

| 文件 | 我要在里面看什么 |
|---|---|
| `models/transformers/transformer_qwenimage.py` | `forward` 的 `zero_cond_t` 分支、`_modulate`、`modulate_index`、`QwenEmbedRope`、`QwenDoubleStreamAttnProcessor2_0`、`QwenTimestepProjEmbeddings` |
| `pipelines/qwenimage/pipeline_qwenimage_edit_plus.py` | `img_shapes` 怎么拼、`latents` / `image_latents` 的 cat 顺序、去噪循环、`true_cfg` 两次前向 |
| `schedulers/scheduling_flow_match_euler_discrete.py` | 按 seq_len 的动态 shift、40 步的 sigma 网格怎么算(`PLAN.md` §3.2 的 t 采样要用) |
| `models/normalization.py` | 复核「LayerNorm 全部 `elementwise_affine=False`、QK 是 per-token RMSNorm」 |
| `models/attention_dispatch.py` | 注意力后端怎么选、`attention_mask` 怎么往下传 |

最后一个**可能不存在**(该模块是较晚版本才有的)。不存在就跳过,在报告里写一行
「`models/attention_dispatch.py` 不存在」,**不要去找替代文件、不要拷 `attention.py` 顶包**。

**另外三样(都是文本,直接贴进 REPORT,不落文件)**:

1. 前两个文件顶部的全部 `import` 行原样(`sed -n '1,60p'` 即可)。
   我据此决定第二轮还要取哪些模块——**你不要替我判断该多拷什么**。
2. `ls -1` 的两份目录清单:`$SP/diffusers/pipelines/qwenimage/`、
   `$SP/diffusers/models/transformers/`。
3. diffusers 的**确切来源**。`0.40.0.dev0` 是 dev 版,要能定位到 commit:
   ```bash
   ls -d $SP/diffusers*.dist-info && cat $SP/diffusers*.dist-info/METADATA | head -20
   cat $SP/diffusers*.dist-info/direct_url.json 2>/dev/null
   ls -la $SP/diffusers/_version.py 2>/dev/null && cat $SP/diffusers/_version.py
   git -C $SP/diffusers log -1 2>/dev/null || echo "(不是 git checkout)"
   ```
   四条都跑,有什么贴什么,**没有就写"没有"**。

---

## 2. B · 权重侧的 config → `qwen/_vendor/qwen2511_config/`

只取 JSON,**一个权重分片都不要动**。

```bash
W=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
C=$R/qwen/_vendor/qwen2511_config
mkdir -p $C
```

| 从 `$W/` 取 | 拷成 |
|---|---|
| `model_index.json` | `model_index.json` |
| `transformer/config.json` | `transformer_config.json` |
| `vae/config.json` | `vae_config.json` |
| `scheduler/scheduler_config.json` | `scheduler_config.json` |
| `text_encoder/config.json` | `text_encoder_config.json` |

缺哪个就报哪个不存在,不要找近似的顶替。

另贴进 REPORT:`find $W -maxdepth 2 -not -name "*.safetensors" | sort` 与 `du -sh $W/*`。
(我要知道 VAE 的下采样倍率和 latent 通道数——它决定 1024² 下噪声图是不是 4096 token,
而 `PLAN.md` §4.4 的速度预测**已经预登记了这个数**,对不上就得在出图前订正。)

---

## 3. C · 环境与机器清点 → `reports/20260813-p1-recon/`

```bash
$E/bin/python -m pip freeze  > reports/20260813-p1-recon/pip_freeze.txt   # 全量,不过滤
nvidia-smi                   > reports/20260813-p1-recon/nvidia-smi.txt
```

REPORT 里另列一张表,**逐项给版本号或"未安装"**:

`torch` / `diffusers` / `transformers` / `accelerate` / `peft` / `deepspeed` /
`bitsandbytes` / `safetensors` / `xformers` / `flash-attn` / `numpy` / `Pillow`

> `peft` 与 `accelerate` 决定 LoRA 那条路能不能直接走(`PLAN.md` §3.2 的
> target modules 是 peft 的写法);`deepspeed` / `bitsandbytes` 决定全参微调
> 那条备选路可不可行。这两行不许省。
>
> **为什么这条现在就要问清楚**:`qwen-edit` 是**推理** env。8×H800 要等一切就绪
> 才申请得下来,所以「训练 env 缺什么」必须在申请之前就查明并补齐——
> 到了机器上才发现装不了包(4090 那台的公共 env 有 GLIBC 前科),等于白占一台卡。
> **本轮只查、只报,不要装任何东西**(装包是绿档 G1,但那是下一轮的事)。

再加:`$E/bin/python -V`、`ldd --version | head -1`、`uname -a`。

### 3.1 你现在这台机器(4090 开发调试机)

**本轮全部在 4090 开发机上执行**(`GOAL.md` §8:`aiplatform-bjy-ge47-391`,
8× RTX 4090 24 GB,Ubuntu 20.04,glibc 2.31)。8×H800(143771 MiB/卡)是训练机,
**要等一切就绪、真正开训那一刻才申请得下来**,本轮碰不到它,也不要去试。

所以这一节要的是**当前这台机器的实况**:

1. `nvidia-smi` 原样(已在 §3 落文件),外加 `nvidia-smi --query-gpu=index,name,memory.total --format=csv`;
2. `df -h` 三行:`/kaimm-distill/wuwenxuan`(共享盘)、本机盘、`/dev/shm`;
3. `free -g` 一行(前向做 CPU offload 时吃主存,`Q1B_3REF_RUN.md` §3.4 的 `sleep 20` 就是这个坑);
4. `$E/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"`
   —— 确认 `qwen-edit` 这个 env 在 **4090 上** import 得动
   (`GOAL.md` §8:`aio_n26` / `v4moe` 在 4090 上 import torch 直接挂,kling-mini 系才行。
   qwen-edit 是 kling-mini 底座,预期能过,但**没实测过就不算数**)。

**H800 的规格本轮不核**——`docs/H800_REBUILD.md` 的 143771 MiB 作为规划假设先用着,
落实成 `train_iso.py` 的启动断言,开训第一秒验。**不要为此去申请机器。**

---

## 4. D · 训练数据清点(**大文件不要拷**)

`PLAN.md` §3.2 要用 `datasets/distill_multiref/manifest_raw.json` 的全部 9000 条。
本地仓库里这个目录只有 16 张抽样图,manifest 不在 git 里。

### 4.1 先定位,再决定拷不拷

```bash
find /kaimm-distill/wuwenxuan -name "manifest_raw.json" -o -name "manifest_filtered.json" 2>/dev/null
ls -la <找到的路径>
```

**判据(不许自己改)**:`manifest_raw.json` **≤ 20 MB** ⇒ 按 §6 放行进 git,我要用它写
采样规则和 held-out 断言;**> 20 MB** ⇒ **不拷**,只交 §4.2 的统计 + 前 3 条原样。
`manifest_filtered.json` **一律不拷**,只报条数(本轮不用它,只是要知道它在)。

### 4.2 统计(无论拷不拷都要,写进 REPORT)

用 `$E/bin/python` 现场算,**结果原样贴,不要转述**:

1. 顶层结构:是 list 还是 `{"...": [...]}`,总条数;
2. **前 3 条原样**(完整 JSON,含 `meta`);
3. 按 `meta.n_refs` 的条数分布(应为 1-ref 1000 / 2-ref 4000 / 3-ref 4000,**对不上照实报**);
4. `meta.subjects` 展平后的主体频次表(全部,不要截断);
5. **held-out 泄漏断言**。名单逐字取自 `distill/DISTILL_PLAN.md` §2:

   ```
   HELD_OUT = backpack_dog, bear_plushie, berry_bowl, can, candle, clock,
              colorful_sneaker, duck_toy, fancy_boot, grey_sloth_plushie
   TRAIN    = backpack, cat, cat2, dog, dog2, dog3, dog5, dog6, dog7, dog8,
              monster_toy, pink_sunglasses, poop_emoji, rc_car, red_cartoon,
              robot_toy, shiny_sneaker, teapot, vase, wolf_plushie
   ```

   报两个数:落在 HELD_OUT 里的条数(**期望 0**)、不在 TRAIN∪HELD_OUT 里的主体名。
   **不为零不要自己处理、不要过滤、不要修**——原样报上来,这是 R1,我判。

### 4.3 `x₀` 目标图

从 manifest 的 `image_tgt_path` 解析出实际目录后:

- 绝对路径、文件张数、`du -sh` 总大小、扩展名分布;
- **抽 5 张**(排序后取第 1 / 2250 / 4500 / 6750 / 9000 张)报
  `PIL` 读出的 `size` 与 `mode`。
  (`PLAN.md` §3.2 登记了这批是 UNO 的 512² 产物、训练要上采样到 1024²,
  这 5 张是那句话的实测凭据。)
- 所在挂载点的 `df -h` 一行。

**这些图一张都不要拷回来。**

---

## 5. 明确不要做的

写下来是因为它们看起来都"顺手":

| 不要 | 为什么 |
|---|---|
| 加载模型、跑任何推理、`from_pretrained` | 本轮零 GPU。探针在第二份单子里,那时 API 已确定 |
| 拷任何 `.safetensors` / `.png` / `.jpg` | 仓库靠 git pull 同步代码,推图会把它拖垮 |
| 写任何新 `.py` | 本轮没有新脚本。要写就是第二份单子的事 |
| 改既有 `.py` / `.sh` | R0,无例外 |
| 改 `.gitignore` 里 §6 给定块之外的任何一行 | 白名单是逐批显式放行的 |
| 对数据/环境做任何"顺手修复" | 清点就是清点。发现异常照实报,我判 |
| 写"够不够用 / 行不行"的结论 | 你交清单、原始输出和现象 |

---

## 6. 交付物

```
qwen/_vendor/diffusers_0.40.0.dev0/**.py        ← §1(五个文件,目录结构原样)
qwen/_vendor/qwen2511_config/*.json             ← §2
reports/20260813-p1-recon/REPORT.md             ← 你写,§1–§4 的全部文字产物
reports/20260813-p1-recon/pip_freeze.txt        ← §3
reports/20260813-p1-recon/nvidia-smi.txt        ← §3
datasets/distill_multiref/manifest_raw.json     ← §4.1 判据为真时才放
```

`qwen/**` 和 `reports/**` 本来就不在 `.gitignore` 里,直接 `git add` 即可。
**只有 manifest 需要放行**,且**仅在 §4.1 判据为真时**才追加这一块到 `.gitignore` 末尾:

```gitignore
# ─── P1 侦察(2026-08-13,qwen/P1_RECON_RUN.md §4.1)────────────────────────
# manifest_raw.json 单独放行:P2 的采样比、held-out 断言、9000 条的分层都要按它写,
# 而本地机器上没有共享盘。图**不**放行——放行的只有这一个 json。
!/datasets/distill_multiref/manifest_raw.json
```

commit message:`chore(qwen): P1 侦察 — diffusers 源码快照 + 环境/数据清点`

---

## 7. 报告怎么写

`reports/20260813-p1-recon/REPORT.md` 一节对一节(§1 A / §2 B / §3 C / §4 D),
每节里:**命令原样 + 输出原样**。不要转述、不要摘要、不要美化对齐。

取不到的项单开一节 **「未取到」**,逐条写:哪一项、报错原文(带 traceback)、
你试了什么。**取不到不是失败,猜一个填上去才是。**

规格自相矛盾时:报告 + 按优先级执行,禁止沉默修复(手册 §2.0 黄档义务)。
规格没写到的语义决定一律红档——**规格留白 ≠ 授权你填空**。
