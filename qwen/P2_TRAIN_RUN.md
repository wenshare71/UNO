# P2 训练 · 完整执行单

> 给远程 agent。机器:已申请到的 8×H800。
> 起点 commit `aa2a97a`(含 code review 修的 6 条,其中两条是**只有 8 卡才发作**的静默错)。
> **§1–§4 一路做下去,做完停下等我判读。§5 正式跑要等我说。**

---

## 0. 先对一下状态

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
R=/kaimm-distill/wuwenxuan/UNO
cd $R && git pull && git log --oneline -1        # 应当是 aa2a97a 或更新
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
df -h /kaimm-distill/wuwenxuan | tail -1        # embeds 要写 ~32 GB,ckpt 另 ~12 GB
```

已知状态:`prompt_embeds` 只算了 512/9000,**而且全是 2-ref**(manifest 按 n_refs 排序)。

---

## 1. 重跑训练路径自检(纯 CPU,几秒)

```bash
$E/bin/python -u qwen/test_train_smoke.py
```

**应当 18/18。** 比上次多了四项,都是刚修那几条的回归:

| | 钉的是什么 |
|---|---|
| S10 | 训练的 t 与推理逐位相同(不是只舍入一次那个) |
| S11 | `set_lora_state` 遇到对不上的 key 会 raise,不静默丢 |
| S12a | 评测侧 `apply_lora_ckpt` 吃得下 `train_iso.save()` 的格式 |
| S12b | 未训练的 ckpt(`lora_B` 全 0)会被拦下,不会冒充 `iso_post` |

不过就停,贴输出。

---

## 2. 补完 prompt_embeds(GPU 0,约 1 h)

```bash
CUDA_VISIBLE_DEVICES=0 nohup $E/bin/python -u qwen/train_iso.py precompute \
    > /tmp/precompute.log 2>&1 &
```

已存在的会跳过,从第 512 条续。**这条和 §3 可以同时跑**(不同卡,互不抢)。

跑完最后一行是:

```
txt token 实测:min … / 中位 … / max …
```

**这个数要。** `PLAN.md` §4.4 的 token 账里它至今写的是「估 400–600」,这是唯一一个还没实测的量。

跑完确认:

```bash
ls cache/prompt_embeds | wc -l      # 应当是 9000
du -sh cache/prompt_embeds
```

---

## 3. 5 步冒烟(GPU 1,约 5 分钟)

**和 §2 并行。** 只用已算好的那 512 条 2-ref —— 冒烟看的是能不能跑起来,不是数据分布。
2-ref 恰好是最长序列、最大 mask 那一档,压力测试意义上是对的那一档。

```bash
CUDA_VISIBLE_DEVICES=1 $E/bin/python -u qwen/train_iso.py train \
    --steps 5 --log_every 1 --allow_partial_embeds \
    --out /tmp/smoke_iso 2>&1 | tee /tmp/smoke5.log
```

### 开头三行自检,逐个核

```
[自检] embeds 覆盖 512/9000 | 1-ref 0/1000 2-ref 512/4000 3-ref 0/4000
[自检] LoRA rank 64 | 可训参数 188.7 M | dtype torch.float32 | seed 20260813(各 rank 一致) | target […]
[自检] …
```

- `dtype torch.float32` —— 修的第 ② 条(bf16 下 AdamW 的更新会被舍入吃掉)。
  **打出 bfloat16 就是没修进去,停。**
- `seed 20260813(各 rank 一致)` —— 修的第 ① 条。单卡看不出效果,但这行必须在。
- `可训参数 188.7 M` —— rank 64 × 8 模块 × 60 层 × 2 个张量。差很多说明 target 没挂全。

### 三个数

| 数 | 期望 | 不对说明 |
|---|---|---|
| **峰值显存** | 完全没数,这就是来量的 | >130 GB 就危险,3-ref 更长,正式跑会 OOM |
| **s/it** | 粗估 20–30 s(accum=1、单样本) | 差一倍以上说明有别的问题 |
| **loss** | 从一个**小但非零**的值开始,5 步内应当有下降趋势 | 见下表 |

loss 的三种坏情况:

- **恒为 0** ⇒ mask 没生效,student 和 teacher 是同一个函数,训练在学恒等;
- **NaN / inf** ⇒ 停,贴 traceback;
- **不降反升** ⇒ 5 步太短看不准,记下数字往下走,§4 再看。

---

## 4. 100 步标定(单卡,约 40–50 分钟)

§2 和 §3 都过了再跑。这时 embeds 应当已经 9000 条齐了,**不要再带 `--allow_partial_embeds`**
—— 带了就只在那 512 条 2-ref 上标定,读出来的 s/it 和显存都不代表真实分布。

```bash
CUDA_VISIBLE_DEVICES=0 $E/bin/python -u qwen/train_iso.py train \
    --steps 100 --log_every 10 --out /tmp/calib_iso 2>&1 | tee /tmp/calib100.log
```

跑完 stdout 末尾会打:

```
训练结束 | 100 步 | … min | … s/it
峰值显存 … GB
loss 首 … → 末 …
```

**这四个数是我决定正式跑多少步的全部依据。** 另外从 `/tmp/calib100.log` 里把
10 个 `loss` 采样点原样抄出来(每 10 步一行),我要看下降的**形状**不只是首末。

### 顺带留意

- 100 步里会混进 1-ref 和 3-ref。**3-ref 是序列最长的一档**(4096 噪声 + 3×4096 ref + txt
  ≈ 16.8k),显存尖峰大概率出现在它身上。如果日志里 3-ref 那几步 OOM 而 2-ref 没事,
  照实记,别调 batch 绕开。
- write 模式那份 `(1,1,L,L)` bool mask 让 SDPA 走不了 flash。上一单实测**时间上慢 1.77 倍**,
  显存那一侧到现在没量过——就是这一步在量。

---

## 5. ⏸ 停在这里,等我判读

§4 跑完**不要**自己接着跑正式训练。把 §1–§4 的结果回报,我看过之后给步数和 `--accum`。

理由:s/it 和显存决定 `--accum` 该取多少、1000 步现实不现实。粗估修正后约 60 s/it,
1000 步 ≈ 17 h;要是标定量出来是 90 s/it,那就得先谈缩短,而不是排一个 25 小时的任务。

---

## 6. 8 卡正式跑(等我说了再执行)

```bash
cd $R && export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
export NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

setsid nohup $E/bin/torchrun --nproc_per_node=8 qwen/train_iso.py train \
    --steps <我给> --accum <我给> \
    --out $R/output/train_iso 2>&1 | tee -a $R/output/train_iso/train.log &
```

三处不是随手写的:

- **`setsid`** —— `PLAN.md` §3.2:上一轮臂 B 被 SIGHUP 打断两次,`nohup` 挡不住 `torchrun`。
- **`NCCL_P2P_DISABLE=0` / `NCCL_IB_DISABLE=0`** —— 仓库里为 4090 打的补丁默认是禁掉的,
  这台 NV18 全互联 + 12 张 IB 网卡,禁掉就退回 PCIe(`docs/H800_REBUILD.md` §2)。
- **`PYTORCH_CUDA_ALLOC_CONF` 在命令行 export** —— 1/2/3-ref 三种序列长度交替会造成
  明显碎片,而代码里那处 `setdefault` 在 `import torch` 之后,不保证生效。

### 断点续跑

每 200 步存一次(`--save_every`),一份约 2.3 GB(LoRA fp32 755 MB + AdamW 两个动量)。
被打断就:

```bash
# 找最新的
ls -t $R/output/train_iso/step*.pt | head -1
# 同一条命令加 --resume
… --resume $R/output/train_iso/step000400.pt
```

`--resume` 现在会把抽样流一起快进(修的第 ⑤ 条),日志里会打
`抽样流已快进 N 个样本`。**没打这一行就是版本不对,停。**

---

## 回报

新建 `reports/20260813-p2-train/REPORT.md`,四节对应 §1–§4:

1. 每节的命令原样 + **stdout 原样全文**;
2. §2 单独列出 `txt token 实测` 那一行 + `ls | wc -l` + `du -sh`;
3. §3 单独列出开头那两行自检(`dtype` / `seed` 那行必须在);
4. §4 单独列出:`s/it`、`峰值显存`、`loss 首/末`,以及 10 个 loss 采样点;
5. 有 traceback 就原样贴,**不要转述、不要自己修**。

commit message:`test(qwen): P2 冒烟 + 100 步标定`。
`cache/` 和 `output/` 都不进 git(`.gitignore` 白名单模式)。

---

## 红线

| 不要 | 为什么 |
|---|---|
| 改 `qwen/*.py` 任何一行 | 报错了我改。贴 traceback,停下等我 |
| §4 跑完自己接着跑正式训练 | 步数和 accum 要按标定数定,见 §5 |
| OOM 了自己调 batch / accum / 分辨率绕开 | 那是把问题藏起来。照实记,我来定 |
| §2 没跑完就跑 §4 标定 | 只有 512 条 2-ref,标出来的 s/it 和显存都不作数 |
| 正式跑带 `--allow_partial_embeds` | 那 512 条是单一桶,拿它出正式结果是自欺 |
| 改 `LORA_TARGETS` / `REF_MIX` / `RESOLUTION` / `NUM_INFERENCE_STEPS` | 都是判据的一部分 |
| loss 不好看就调 lr 重跑 | 标定是来读数的,不是来调参的。数字交上来 |
