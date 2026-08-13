# 申请 H800 之前的三件 + 到手之后的第一件 · 执行单

> 给远程 agent。**§1–§3 在 4090 上跑,不碰 infer_hub、不碰 H800。**
> §4 是 H800 到手之后的顺序,先别执行,等我说。
> 上下文:`train_iso.py` 至今一行没跑过,而 H800 是「一切就绪才申请得下来」的资源。
> 这三件的唯一目的是:**别让 8 张卡陪着调 `add_adapter` 的参数名。**

---

## 1. 训练路径自检(纯 CPU,几秒)· 绿档

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
cd $R && git pull
$E/bin/python qwen/test_train_smoke.py
```

不需要 `QWEN_WEIGHTS`,不上卡。2 层随机权重,和 P1 门禁同一套路子。

测 13 项。**不过就别往下走**,把输出原样贴回来,我改代码。

| | 断言 | 不过说明 |
|---|---|---|
| S1 | `add_adapter` 认得 `LORA_TARGETS` 那 8 个名字 | 目标模块名写错,LoRA 挂了个空 |
| S2 | `enable_gradient_checkpointing` 落到顶层 | fork 的 forward 读的是 `transformer.gradient_checkpointing`,读不到就是没开 |
| S3 | LoRA 初始恒等,disable/enable 逐位相同 | 教师/学生共用一份权重这个前提不成立 |
| S4 | 第 0 步 loss 非零 | 差异应当来自 mask;为零说明 mask 没生效,训练在学恒等 |
| S5 | 检查点下能反传,梯度到每层 LoRA | `_gradient_checkpointing_func` 那 8 个位置参数传错了 |
| S6 | 反传之后 cache 仍为空 | 每步漏 60 层 ref K/V(真模型 2-ref 约 6 GB) |
| S7 | step 后参数动了,且 teacher 回原样 | LoRA 没接进前向,或 disable 是近似 |
| S8 | LoRA 存/读往返一致 | 断点续跑会静默丢权重 |
| S9 | sigma 网格 = 推理那 40 个 | 训练分布 ≠ 部署分布 |

## 2. 真 pipeline 段(CPU 加载 54 GB,不上卡)· 绿档

```bash
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
$E/bin/python qwen/test_train_smoke.py --pipe
```

补上 §1 覆盖不到的那截:`prepare_latents` / `_encode_vae_image` / `_pack_latents` /
`img_shapes`。这几个我是照着 vendored 源码写的,**一次都没执行过**。

CPU 上跑 bf16 VAE 会慢(单张 1024² 约一两分钟),正常。内存要 60 GB 上下,
这台机器 1007 GB,够。

## 3. prompt_embeds 预算(1 张卡,约 1–1.5 h)· 绿档

```bash
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
$E/bin/python qwen/train_iso.py precompute 2>&1 | tee /tmp/precompute.log
```

**为什么现在做:** 之前我判它「不是申请机器的闸门」,那个判断在「先投 infer_hub 冒烟」
的方案下成立。现在改成直接申请 H800,它就上了关键路径——`train_iso.py train`
在缓存目录不存在时**直接退出**,机器到手才发现就是白等。

- 只挂 VL 7B(约 16 GB),4090 装得下,transformer / vae 不上卡。
- 断点续跑:文件已存在就跳过,中断了重跑同一条命令即可。
- 输出 `cache/prompt_embeds/{000000..008999}.pt`,**约 32 GB 写到 ceph**(67 MB/s)。
  磁盘不够就先说,别跑一半炸。
- 跑完最后一行会打印 `txt token 实测:min / 中位 / max` —— **这个数要**,
  §4.4 的 token 账里它至今还是「估 400–600」。

---

## 4. H800 到手之后(先别做,等我说)

**顺序不能换。** 一上来就排长任务,崩了就是 8 张卡空转。

```bash
# ① 单卡 5 步 —— 只为确认它能跑起来
CUDA_VISIBLE_DEVICES=0 python qwen/train_iso.py train --steps 5 --log_every 1

# ② 单卡 100 步标定 —— 量峰值显存 / s-it / loss 下降形状
CUDA_VISIBLE_DEVICES=0 python qwen/train_iso.py train --steps 100

# ③ 8 卡,步数按 ② 的实测定
torchrun --nproc_per_node=8 qwen/train_iso.py train --steps 1000 --accum 2
```

① 要盯三样,任何一样不对就停:

- **峰值显存**。write 模式那份 `(1,1,L,L)` bool mask 让 torch SDPA 不走 flash,
  退到哪个 kernel 是它自己挑的。P2 前置那一单实测**带 mask 比不带慢 1.77 倍**,
  显存那一侧还没量过。
- **s/it**。`PLAN.md` §3.2 估的 40 s/it 没算 mask 那 1.77×,修正后约 60 s/it。
  差太远就是有别的问题。
- **loss 量级**。速度匹配的 loss 应当从一个小但非零的数开始降,不是从 0 开始
  (从 0 开始 = mask 没生效)。

---

## 明确不要做的

| 不要 | 为什么 |
|---|---|
| 改 `qwen/*.py` 任何一行 | 报错了我改。贴 traceback 原文,停下等我 |
| 投 infer_hub 跑训练 | 那是推理队列 |
| 申请 H800 / 执行 §4 | 等 §1–§3 都过了我再说 |
| §1 不过就接着跑 §2、§3 | §1 是最便宜的一层,它不过说明接线错了 |
| 判据没过就调参数重跑 | 交输出,判读是我的事 |

## 回报

新建 `reports/20260813-p2-launch/REPORT.md`(绿档),三节对应 §1–§3:

1. 命令原样 + **stdout 原样全文**;
2. §3 那节额外单独列出 `txt token 实测` 那一行,和 `du -sh cache/prompt_embeds`;
3. 有 traceback 就原样贴,不要转述、不要自己修。

commit message:`test(qwen): P2 训练路径自检 + prompt_embeds 预算`。
`cache/` 不进 git(`.gitignore` 白名单模式)。
