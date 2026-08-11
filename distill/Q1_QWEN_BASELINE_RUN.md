# Q1 执行单 — Qwen-Image-Edit-2511 裸基线上机

> **执行者**:4090 机器(`aiplatform-bjy-ge47-391`)上的 Claude Code。
> **档位:全程 🟢 绿档——你不写任何代码。**
> `scripts/infer_qwen_edit.py` 已随本执行单进主线,本地 dry_run 验过(见 §4)。
> 你只做三件事:装环境、下权重、跑命令。**任何 `.py` / `.sh` 一个字都不许改(R0)。**
>
> 目标不是做实验,是**看一眼 Qwen-Image-Edit-2511 在我们现有多参考任务上裸跑成什么样**。
> 所以口径全部锁死在官方默认值,不调参、不改 prompt、不挑样本。
>
> 总耗时估计:阶段 A 里人要盯的约 30 min,其余是无人值守下载(时长未知,见 §2);
> 阶段 C 的 GPU 占用约 1 卡 × 30–60 min。

---

## 0. 这一步在干什么(先看懂再跑)

我们要换一个更强的底座继续做隔离注意力 + KV cache 的蒸馏,候选是
**Qwen-Image-Edit-2511**(20B MMDiT,Apache 2.0,参考图 latent 沿 sequence 维
concat 进 joint attention——结构上和 UNO 的 `[txt, img, ref_1..ref_N]` 同构,
我们的 `ref_attention.py` 能平移)。

换底座之前必须先回答一件事:**它在我们自己的任务分布上,不加任何改造,原本是什么水平。**
这是后面一切对比的零点。没有这个零点,以后说"隔离掉了多少"就没有参照。

所以这一轮**只做三件事**:

1. 权重、环境、脚本全部在 4090 上就位并验通(不占 H 卡);
2. 在 4090 上单例冒烟一张图,确认通路对(不占 H 卡);
3. 投一个 infer_hub 任务,在 H 卡上跑完 40 条子集,出拼图。

**明确不做的事**(想做先问,属于 R13):

- 不改 prompt(不加指令式前缀、不重写句式);
- 不调 steps / cfg / 分辨率;
- 不挑样本(子集规则是机械的,见 §6.2);
- 不训练、不加 LoRA、不动 `ref_attention.py`;
- 不跑 `judge.py` 打分——这一轮用眼睛看。

---

## 1. 机器分工与两个硬约束

| 阶段 | 在哪 | 为什么 |
|---|---|---|
| A. 下权重 / 建环境 / 写脚本 / dry_run | **4090** | 训练机有 GPU 利用率考核,纯 CPU 长任务会被强杀(M6 P1 实测第 5 片就被杀) |
| B. 单例冒烟 | **4090**,`enable_model_cpu_offload()` | 24 GB 装不下 20B bf16(约 40 GB),靠 offload 过一遍,慢但只为验通路 |
| C. 40 条正式跑 | **infer_hub**,`--gpus 1 --cluster h` | 20B bf16 单卡 H 装得下,不需要 8 卡 |

**硬约束 1:所有路径必须在 `/kaimm-distill/` 下。**
`infer_submit` 会直接拒绝 `/home/<user>` 和本地盘的路径(推理机看不到那些挂载)。

**硬约束 2:`infer_submit` 只认已 push 的 40 位 commit。**
所以 §6 那个脚本写完必须先进主线。4090 能不能 push 未知(H800 是不能的,见手册 §3.5),
按 §7 的两条路走。

---

## 2. 步骤 A1 — 起权重下载(最长的一步,**第一件事就做**)

### 走哪条线

**不要走日本 squid**(`oversea-squid1.jp.txyun:11080`)。`docs/H800_REBUILD.md` §1 实测
HuggingFace 经它是 **0.66 MB/s**,几十 GB 要下十几个小时。

走 `scripts/DOWNLOAD_RUNBOOK.md` §一/§二里已经验过的那套:
`HF_ENDPOINT=https://hf-mirror.com` + **国内代理**(`10.66.29.113` / `10.66.37.111` /
`10.66.72.150`,各 `:11080`)+ curl 256 MB 分块。RUNBOOK §二的五条经验全部适用,
尤其:

- **代理会截断 >1 GB 的 range 请求** → 必须 256 MB 固定小块 + 每块校验大小;
- **`hf_transfer` 走国内代理会 D 状态挂死** → 只能用 curl;
- **必须 `unset HF_HUB_OFFLINE`**。

### 落哪

```
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/
```

⚠️ **落 ceph,不落本地 NVMe。** 这和 H800 的"权重放本地盘"惯例相反,原因是
`infer_submit --weights` 要求路径在共享盘上,推理机才看得见。ceph 读 136 MB/s,
20B 权重首次加载约 5 min,可接受(FLUX 从 ceph 加载实测 96.1 s)。

### 先算总量,再起下载

**不要假设是 40 GB。** 20B 的 DiT 之外还打包了 Qwen2.5-VL 文本编码器和 VAE,
实际更大。先把文件清单和总字节数拉出来:

```bash
export HF_ENDPOINT=https://hf-mirror.com; unset HF_HUB_OFFLINE
# 列文件 + 大小(走代理,几 KB 的请求)
curl -s -x http://10.66.29.113:11080 \
  "https://hf-mirror.com/api/models/Qwen/Qwen-Image-Edit-2511?blobs=true" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); s=[(f['rfilename'],f.get('size',0)) for f in d['siblings']]; [print(f'{n}\t{z/1e9:.2f} GB') for n,z in sorted(s,key=lambda x:-x[1])[:30]]; print('TOTAL', sum(z for _,z in s)/1e9, 'GB')"
```

**把这个 TOTAL 和文件清单原样报回来**,再按 RUNBOOK 的三代理分区起下载。
分区规则同 RUNBOOK:**每个进程的文件列表必须不相交**。

按 RUNBOOK 实测的 10–100 MB/s,先给一个 ETA 报回来。

> 手册 §4.0:后台跑 + 日志 + 每 5 分钟一行心跳。不要 `-q`。

---

## 3. 步骤 A2 — 建推理环境(与下载并行,**这是最可能翻车的一步**)

### 为什么单列出来

Qwen-Image-Edit-2511 的 model card 要求的是
`pip install git+https://github.com/huggingface/diffusers`——**main 分支,不是 release**。
而我们的机器是 torch 2.4.0 / py3.10 / **无 nvcc**。diffusers main 很可能要更新的 torch,
内网源有没有对应轮子是未知数。

**这一步不通,后面全废。所以它必须和下载并行,不能等 40 GB 下完才发现装不上。**

### 怎么建(用 cp -a,不要从零建 venv)

`docs/infer_hub/USAGE.md` §0 明说公共环境是只读共识,要改就 `cp -a` 一份。
**照做,不要自己 `python -m venv`** ——公共环境的 python 版本和推理机是配套的,
自己建的很可能对不上(4090 是 py3.12,H800 只有 py3.10.12,**跨机直接炸**)。

```bash
mkdir -p /kaimm-distill/wuwenxuan/envs
cp -a /kaimm-distill/infer_hub/envs/aio_n26 /kaimm-distill/wuwenxuan/envs/qwen-edit
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python -V        # 记下来,报回
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

> `aio_n26` 只是起点。如果它的 torch 太老装不上 diffusers main,换
> `kling-mini` / `v4moe` 再试一次——三个都试过还不行就是红灯,按 §8 报回,别自己编译。

### 装依赖

pip 走**内网源**(241 MB/s),但 `git+https://` 那条必须走代理:

```bash
P=/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/pip
IDX="--index-url https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/"

$P install $IDX -U transformers accelerate safetensors
# diffusers main 走代理
https_proxy=http://10.66.29.113:11080 $P install "git+https://github.com/huggingface/diffusers"
```

### 门禁 A2(通不过就停,不要往下走)

```bash
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python -c "
from diffusers import QwenImageEditPlusPipeline
import diffusers, torch, transformers
print('diffusers', diffusers.__version__)
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('OK')"
```

打出 `OK` 才算过。**把这四行原样报回来。**

---

## 4. 步骤 A3 — 验收 runner 脚本(不是写,是跑一遍对答案)

脚本已经在仓库里。`git pull` 之后直接跑 dry_run(不需要权重、不需要 GPU、几秒钟):

```bash
cd <4090 上的 UNO 仓库> && git pull
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python scripts/infer_qwen_edit.py \
  --dry_run --out /tmp/q1dry
```

### 预期输出(本地 2026-08-11 实测,你那边必须一字不差)

自检行:

```
[自检] 全量 320 条 | 子集(i%8==0)40 条 | 待跑 40 | 已跳过 0 | 输出 /tmp/q1dry | 权重 (dry_run)
```

40 个 `task_id`,**顺序也要一致**:

```
M6_S1_000_s0  M6_S1_001_s3  M6_S1_003_s1  M6_S1_004_s4  M6_S1_006_s2
M6_S1_008_s0  M6_S1_009_s3  M6_S1_011_s1  M6_S1_012_s4  M6_S1_014_s2
M6_S1_016_s0  M6_S1_017_s3  M6_S1_019_s1  M6_S1_020_s4  M6_S1_022_s2
M6_S1_024_s0  M6_S1_025_s3  M6_S1_027_s1  M6_S1_028_s4  M6_S1_030_s2
M6_S1_032_s0  M6_S1_033_s3  M6_S1_035_s1  M6_S1_036_s4  M6_S1_038_s2
M6_S1_040_s0  M6_S1_041_s3  M6_S1_043_s1  M6_S3_00c2_s0  M6_S3_01c1_s0
M6_S3_02c0_s0  M6_S3_02c4_s0  M6_S3_03c3_s0  M6_S3_04c2_s0  M6_S3_05c1_s0
M6_S3_06c0_s0  M6_S3_06c4_s0  M6_S3_07c3_s0  M6_S3_08c2_s0  M6_S3_09c1_s0
```

子集构成(已核对**与全量同比例**,是忠实抽样不是偏样):

| | S1(2-ref) | S3(1-ref) | 合计 |
|---|---|---|---|
| 全量 | 220 | 100 | 320 |
| 子集 | 28 | 12 | 40 |
| 抽样率 | 12.7% | 12.0% | 12.5% |

40 条里有 **38 组不同的 subject 组合**,基本没有重复。

> ⚠️ 注意 **12/40 是 1-ref**。M6 的结论是 1-ref 上测不出隔离的差别,但这一轮量的是
> Qwen 的**绝对水平**不是隔离代价,1-ref 同样是零点的一部分,所以照跑。
> 看图时按 S1 / S3 分开看。

### 门禁 A3

报回两样:

1. 自检行原文 + 40 个 `task_id`(和上面逐条对,**有任何一条对不上就是红灯,停**);
2. 拼图分了几张 part、`results.json` 的绝对路径。

对上了就直接进 §5,不用等我回复。

---

## 5. 步骤 B — 4090 单例冒烟(权重下完之后)

只跑**第 1 条任务**,`enable_model_cpu_offload()`,不求快只求通:

```bash
QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python scripts/infer_qwen_edit.py \
  --limit 1 --offload --out output/q1_smoke
```

`--out` 单独开一个目录:冒烟产物别和正式跑的输出混在一起,否则断点续跑会把
offload 下生成的那张当成已完成跳过(两者噪声一致但不是同一次运行,混着不干净)。

### 门禁 B(**这是人工确认点,等我回复再投 infer_hub**)

报回:

- 单张耗时(offload 下预计 3–10 min,慢是正常的,**卡住不动才是问题**);
- 峰值显存;
- 输出图的绝对路径,并**用文字描述你看到了什么**:两个主体在不在、有没有丢、
  身份像不像参考图、有没有明显崩坏(手册 §3.5——图我看不到,文字描述才是交付物);
- 完整 stdout 的头 30 行 + 尾 20 行。

---

## 6. `scripts/infer_qwen_edit.py` 的口径(**已实现,这一节是给你对账用的**)

下面是脚本里已经写死的东西。列出来不是让你实现,是让你在看结果时知道口径是什么、
以及**哪些东西动了就等于换了一个实验**。

### 6.1 常量(写死在文件顶部 CONSTANTS 段,**不做 CLI 可调**)

| 项 | 值 | 来源 |
|---|---|---|
| 任务表 | `datasets/eval_multiref/m6_tasks.json` | 已有 320 条,`m6_iso`/`m6_full` 在同 seed 下的产出也在 |
| pipeline | `QwenImageEditPlusPipeline` | 官方 model card |
| dtype | `torch.bfloat16` | 官方 model card |
| `num_inference_steps` | `40` | 官方示例值,不调 |
| `true_cfg_scale` | `4.0` | 官方示例值,不调 |
| `negative_prompt` | `" "`(一个空格) | 官方示例值 |
| `height` / `width` | `1024` / `1024` | Qwen 原生分辨率 |
| 权重路径 | 环境变量 `QWEN_WEIGHTS`,无默认值,缺失即 `raise` | 硬约束 1 |
| 输出目录 | 环境变量 `INFER_OUTPUT_DIR`,缺省 `output/qwen_baseline` | infer_hub 注入 |

> ⚠️ **1024×1024 和我们 UNO 侧的 512×512 不是一个尺寸**(`inference.py:51-52`,
> `infer_multibanana.py` 默认 512)。这是刻意的——Qwen 在 512 上不是原生分辨率,
> 压到 512 比的就不是它的真实水平了。**并排看图时必须知道两边尺寸不同**,
> 这条要写进最后的报告。

### 6.2 子集规则(机械的,**唯一的选样规则**)

按 `m6_tasks.json` 里 `tasks` 数组的**原始顺序**(不排序、不打乱),取下标满足
`i % 8 == 0` 的条目 ⇒ 320 / 8 = **40 条**。

`--limit N` 只截取这 40 条的**前 N 条**,不改变选取规则。

### 6.3 每条任务怎么跑

```
images  = [Image.open(p).convert("RGB") for p in task["image_paths"]]   # 顺序原样,不重排
prompt  = task["prompt"]                                                # 原样,不加前缀、不改写
gen     = torch.Generator(device=<device>).manual_seed(task["seed"])    # 用任务自带的 seed
out     = pipe(image=images, prompt=prompt, negative_prompt=" ",
               num_inference_steps=40, true_cfg_scale=4.0,
               height=1024, width=1024, generator=gen).images[0]
out.save(f"{OUT}/{task['task_id']}.png")
```

`image_paths` 是相对 `datasets/eval_multiref/` 的相对路径(形如 `../dreambooth/...`),
按该 JSON 文件所在目录解析。

### 6.4 CLI(只有这四个,**都不改变实验语义**,属绿档 G3)

| 参数 | 作用 |
|---|---|
| `--dry_run` | 不加载模型,用 1024×1024 纯色占位图走完整流程 |
| `--limit N` | 只跑前 N 条 |
| `--offload` | 调 `pipe.enable_model_cpu_offload()`(4090 冒烟用) |
| `--out DIR` | 覆盖输出目录 |

### 6.5 已实现的行为(手册 §4.1,你验收时对照看)

- **断点续跑**:`<task_id>.png` 已存在即跳过,自检行里报"已跳过 N 条"(本地已验)。
  所以任务被 infer_hub 超时杀掉后**重投同一条命令即可**,不会重跑已完成的;
- **启动自检行**:全量数 / 子集数 / 待跑 / 已跳过 / 输出目录 / 权重路径,第一秒打印;
- **进度行**,每条一行:
  ```
  [HH:MM:SS] 12/40 (30.0%) | 24.1 s/img | ETA 11m | fail 0 | M6_S1_020_s4
  ```
- **失败当场打印**并计入 fail,**继续跑不中断**;失败条目在 `results.json` 里
  **保留**,只填 `error` 字段——不会被悄悄从子集里删掉;
- 全程 `flush=True`(重定向到日志时不会几分钟不更新);
- **`results.json`**:`meta` 段含子集规则、四个采样常量、失败数、总耗时、
  `diffusers` / `torch` 版本;`tasks` 段每条含 `task_id` / `n_refs` / `seed` /
  `elapsed_s` / `peak_mem_gb` / `error`;
- **拼图**:`ALL_COMPARISON.png`,每行 `[ref_1 | ... | ref_N | 红线 | qwen2511]`。
  超 2 MB 自动分批成 `ALL_COMPARISON_partNN.png`(本地 dry_run 实测分成 5 张)。
  拼图用**全子集**重建,断点续跑时也出完整总览。

### 6.6 红线(动了就等于换了实验)

- 改 §6.1 任何常量、改 §6.2 子集规则 → 🔴;
- 显存不够就改小分辨率或 steps → 🔴,报上来;
- 因为某条老失败就把它从任务表里去掉 → 🔴(手册"自作聪明"专条);
- 改 `infer_qwen_edit.py` 本身或任何既有 `.py` / `.sh` → 🔴(R0,无例外)。
  脚本有 bug 就按手册 §3.3 出 REPORT,我改完你 pull。

---

## 7. 步骤 C — 投 infer_hub

### C1. 拿 commit sha

脚本和本执行单是**用户在本地 push 的**,你只需要:

```bash
git pull && git rev-parse HEAD          # 40 位 sha,给下面的 --commit 用
git status --short                      # 必须干净;有本地改动先报上来别投
```

**你不需要 commit,也不需要 push。** 工作区脏了说明有人动过代码,那是红灯。

### C2. 投

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH

infer_submit --owner wuwenxuan --project qwen-edit-baseline \
  --repo <本仓库 https 地址> --commit <40位 sha> \
  --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --output-dir /kaimm-distill/wuwenxuan/outputs/q1_qwen_baseline \
  --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
  --gpus 1 --cluster h --timeout 60 \
  --label q1_qwen2511_baseline_n40 \
  --cmd 'QWEN_WEIGHTS=$INFER_WEIGHTS_DIR python scripts/infer_qwen_edit.py'
```

三个参数的理由:

- **`--gpus 1`**:20B bf16 约 40 GB,单卡 H(80 GB)装得下,不要占 8 卡;
- **`--timeout 60`**:默认 30 分钟不够——40 条 × 40 步 × 1024²,加上从 ceph
  首次加载权重的几分钟。infer_hub 到点会看 GPU 利用率决定杀还是延,但别卡着线投;
- **不用 `--prep-cmd`**:我们没有切权重步骤,`from_pretrained` 直接读。

**先 `--dry-run` 打印一遍 job json 报回来**,确认无误再真投。

### C3. 跑完报什么

1. `results.json` 的 `meta` 段全文;
2. 失败条目的 `task_id` + `error`(如果有);
3. `ALL_COMPARISON.png` 的**绝对路径** + 一段文字描述:
   主体保留情况、身份相似度的主观印象、有没有系统性的失败模式
   (比如是不是总丢第二个主体、是不是背景压过主体);
4. 与 `m6_full` / `m6_iso` 在同 `task_id` 上的**主观对比**——
   那些图在机器上已经有了,并排看一眼,用文字说清 Qwen 是明显更好、差不多、还是更差。

---

## 8. 红黄绿速查(本执行单特化)

| 情形 | 档 | 动作 |
|---|---|---|
| 装包失败、版本冲突、换公共环境起点 | 🟢 | 自己修,报一行 |
| 代理超时、下载中断 | 🟢 | 重试(下载器自带断点续跑) |
| infer_hub 任务超时被杀 | 🟢 | 重投同一条命令(脚本自带断点续跑,不会重跑已完成的) |
| dry_run 的 40 个 task_id 和 §4 对不上 | 🔴 | 停 —— 说明代码或数据不是我这份 |
| 改 §6.1 任何一个常量 / §6.2 子集规则 | 🔴 | 停,报上来 |
| 显存不够想降分辨率或降 steps | 🔴 | 停,报上来 |
| 想改 prompt 让效果变好 | 🔴 | 停 —— 这一轮要的就是裸基线 |
| **写或改任何 `.py` / `.sh`** | 🔴 | 停(R0,无例外)。本执行单全程绿档,你不该动代码 |
| 三个公共环境都装不上 diffusers main | 🔴 | 停,按手册 §3.3 出 REPORT |
| 同一个问题修了 3 次没过 | 🔴 | R10,停 |
| 机器上有别人的任务 | 🔴 | 如实报告占用,由用户拍板(R12) |

**两个门禁要停下等回复:A2(环境验通)和 B(冒烟)。**
A3 对上了就自己往下走,不用等。

---

## 9. 交付物清单

| 门禁 | 交付 |
|---|---|
| A1 | 权重仓库文件清单 + TOTAL GB + 下载 ETA |
| A2 | `diffusers` / `torch` / `transformers` 版本三行 + `OK` ⟵ **停,等回复** |
| A3 | 自检行 + 40 个 `task_id`(与 §4 逐条对)+ 拼图分了几张 |
| B | 单张耗时 / 峰值显存 / 图的绝对路径 / **文字描述看到了什么** |
| C | job json(dry-run)→ 投递确认 → `results.json` meta + 拼图路径 + 文字对比 |

沉默上限 10 分钟(手册 §5)。长任务后台 + 日志 + 每 5 分钟一行心跳。
