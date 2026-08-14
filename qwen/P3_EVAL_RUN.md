# P3 出图 · 执行清单

> 给在 4090 开发机上的 claude 会话。你可能没有这个项目的上下文,§0 先看完。
> **这一单只出图,不做盲评判读,不建拼图。**
>
> **本清单 2026-08-14 重写过一次。** 之前的版本里有两条判据是作者立错的
> (`mean < 0.5`、floor `≈0`),各害得执行方停下来上报一次。那些段落已删除,
> 不是隐藏,`git log qwen/P3_EVAL_RUN.md` 能看到全过程。
> `reports/20260814-p3-eval/REPORT.md` 里引用的 §5.1 / §5.3 编号指的是旧版,
> 别拿旧编号来对现在这份。
>
> **进度**:§6.0 ✅ 已过 · §6.3 floor ✅ 已跑完 60 张 ·
> §6.1 判据 第一轮已跑,收窄到「ref K/V 随文本段变」;成因已在本地判定为良性的长度效应,
> 第二轮只是真机确认,**不阻塞主批** ·
> §6.2 主批 ⏳ 待投。

---

## 0. 你在做什么

一句话:把 **Qwen-Image-Edit-2511**(20B 图像编辑模型)蒸馏成一个用
「**隔离注意力 + 参考图 KV 缓存**」的学生,然后用盲评量出这套改造的质量代价。

- **隔离注意力**:参考图的 token 只自注意,不看噪声图和文本。这样参考图的 K/V 在
  40 步去噪里逐步不变 ⇒ 算一次缓存起来,后面 79 次前向直接读 ⇒ 提速。
- **代价**:切断 ref→img 的通路会掉质量。P2 已经用 LoRA 把这个缺口补了一部分
  (速度匹配蒸馏,2000 步,loss 降 2.6×),**掉多少、补回来多少,就是这一单要出的图去回答的**。

三个臂,同一批任务、同一个 seed:

| 变体 | 是什么 |
|---|---|
| `full` | stock 权重 + 全注意力 = teacher = 基线 |
| `iso_pre` | stock 权重 + 隔离 + 缓存,**未训练** = 训练的第 0 步 |
| `iso_post` | 训练后的 LoRA + 隔离 + 缓存 |

代码已经写好了,就是 `qwen/infer_iso.py`,三个臂只差 `--variant`。你**不用写代码**,
这一单的难点全在「怎么把它投到远程推理集群上并且不被中途杀掉」。

想看全貌:`qwen/PLAN.md`(§3.3 是这一单)、`qwen/GOAL.md`(项目背景)。
不用全读,卡住了再去查。

---

## 1. 三条硬约束

1. **R0 —— `uno/` 和 `distill/` 下既有的 `.py` / `.sh` 一个字不动。** import 可以,改不行。
   `qwen/` 下的也一样:这一单预期不需要改任何代码,真要改先说。
2. **不许改 Q1 口径**:steps 40 / true_cfg 4.0 / 1024² / negative_prompt `" "` /
   prompt 原样 / seed 取自任务表。这些已经写死在 `infer_iso.py` 顶部,别传参绕过去。
   改了基线就废了,整批图作废。
3. **带变体名的拼图不许进 git,也不要自己建 board。** 盲评要求判读的人看不到哪张是哪个臂,
   拼图和配对由盲评那一侧(`../distill/blind_eval/`)按它自己的纪律做。
   你只产出 `<task_id>.png` 和 `results_shard*.json`。
   **单臂图也别再往 git 里放**(已经漏进去 63 张,见 §8)。

---

## 2. 什么时候停、什么时候不停 ← 先读这一节

**默认动作是继续。** 这份清单里出现的任何"预期值""大概是多少",都是作者的猜测,
**不是门禁**。数没对上就记进报告接着跑,不要为它停下来等回复。猜错是作者的问题,
不该由你付一个来回的代价。

前两版清单在这上面翻过两次车(`mean < 0.5` 抄自另一套栈;floor `≈0` 的对照
根本没有扰动源),两次都是判据错、执行对。所以现在把停线收窄到四条:

🔴 **真要停的只有这四条**

1. `qwen/diag_kv.py` 打出 **FAIL_LEAK / FAIL_STEP**(§6.1)—— 缓存有逻辑错,720 张全废。
   它自己打四选一,**只有这两个要 kill 主批**;`INCONCLUSIVE` 是探针自己的问题,不影响出图。
2. ckpt 自检对不上(§6.0)—— `iso_post` 会静默退化成 `iso_pre`,事后看不出来。
3. 需要改任何 `.py`。
4. 同一个原因连续失败两次 —— 别投第三次,把归因贴过来。

🟡 **其余全部是"记下来,继续"**,包括但不限于:速度不符合预登记的预测、
显存比预估高、加速比和上一轮对不上、某个数落在清单没写到的区间。
这些都写进报告,不要停。

⚪ 还有一条一直有效:**清单不是权威,机器上看到的才是。**
发现清单里某个前提是错的,以机器为准,继续做,然后在报告里指出来。

---

## 3. 机器、网络、存储(已知条件,不用你摸)

**4090 开发机跑不了这个模型。** 20B transformer bf16 ≈ 40 GB > 24 GB 显存。
所有出图必须投远程推理集群(infer_hub,H800 80GB)。4090 上你能做的是:git、CPU 侧的验算、
看结果、写报告。别试图在本地生成,那是死路。

| | |
|---|---|
| env | `/kaimm-distill/wuwenxuan/envs/qwen-edit`(py3.11,torch 2.5.1+cu124,diffusers 0.40.0.dev0) |
| 权重 | `/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511` |
| 仓库 | `/kaimm-distill/wuwenxuan/UNO` |
| LoRA ckpt | `/kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt`(2.26 GB) |

**别碰 `aio_n26` / `v4moe` 这两个公共 env**——它们的 torch 是给 H 机(glibc ≥ 2.34)编的,
4090 是 Ubuntu 20.04 / glibc 2.31,`import torch` 直接挂在 `libucs.so.0` 上。
上一轮在这上面烧了大半天,细节在 `reports/20260811-0951-q1-env-glitch/REPORT.md`。

**存储**:`/kaimm-distill` 是 ceph 共享盘(215 T),推理机也挂着,所以是唯一的交接通道。
`/home/<user>` 是各人独立挂载,**推理机上看不到**,路径写它会被提交端直接拒。
ceph 读约 136 MB/s、写约 67 MB/s —— 比本地 NVMe 慢 25 倍,§5.2 那个坑就出在这。

**网络**:出网走日本代理,全通但全慢——HF 0.66 MB/s、PyPI 官方 0.05 MB/s;
内网源 `pypi.corp.kuaishou.com` 命中 `no_proxy` 直连,241 MB/s。要装包只走内网源。
**本地跑任何 import diffusers 的东西先 `export HF_HUB_OFFLINE=1`**:不设的话
`from_pretrained` 每次去 HF 探 etag,过日本代理能拖到两分钟。远程 worker 侧已处理,不受影响。

---

## 4. infer_hub 是什么

多人共享的推理队列。心智模型三条:

1. **状态即目录**:共享盘上一个目录,`pending/` → `claimed/<机器>/` → `done/`/`failed/`。
   没有中心调度器,活着的 worker 自己来抢活。
2. **派活是拉不是推**,你 ssh 不上推理机,也不需要。
3. **代码只能来自已 push 的 commit**,参数走环境变量。改了代码不 push 就投,
   跑的是旧代码——这是最容易踩、而且最难发现的坑。

手册在 `docs/infer_hub/`(`USAGE.md` 是全参数 + 硬规矩 + 常见坑)。

**上一轮实际跑通的提交模板,照抄**(§5 的坑都已经编进去了):

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
SHA=$(git rev-parse HEAD)          # 必须已 push

sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h \
    --repo https://github.com/wenshare71/UNO.git \
    --commit $SHA \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/<本批> \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label <实验名>_Iter2000 \
    --gpus 8 --timeout 180 \
    --prep-cmd 'true' \
    --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --cmd '...'
```

看状态:`infer_status --owner wuwenxuan`;看日志:
`tail -f /kaimm-distill/infer_hub/queues/default/logs/<job_id>.log`。
`--dry-run` 只打印 job json 不真投,试探提交端检查用它。

---

## 5. 已知的坑(前人踩过,直接抄,别重新发现)

### 5.1 提交端

- **必须 `sudo -E env PATH=...`** —— 锁文件在 `/kaimm-distill/infer_hub/locks/`,属 root,
  普通用户直接投是 `PermissionError`。模板 §4 里已经写好了。
- **`submit_require_prep=true`** —— 不带 `--prep-cmd` 直接拒收。本单没有切分步骤,
  兜底写 `--prep-cmd 'true' --prep-marker <权重目录>` 过门槛。
- **`--cmd` 里的共享盘绝对路径检查是一刀切**(`hubcore.py:124` 的 `cmd_shared_disk_leaks`):
  正则匹配所有 `/kaimm-distill/...`,不在 `[weights, output_dir, uv_env, prep_marker]`
  声明的根下就 `die`。所以 `iso_post` 的 LoRA 路径不能直写进 `--cmd`。
  **已验证可行的解法**:把 LoRA 文件路径填进 `--prep-marker`(它在共享盘、已存在,
  marker 存在 ⇒ `prep_done=true` 直接排卡),路径就合法声明进来了。
  语义上 `--prep-marker` 本该是"切分完成标志",这里是权宜,真实切分场景别照抄。

### 5.2 运行端

- **八卡要自己 fan out。** `infer_iso.py` 不起多进程,靠 `--shard_idx i --num_shards N`
  (`i % N == shard_idx`,断点续跑靠已存在的 png 跳过)。别写死卡号,读注入的
  `CUDA_VISIBLE_DEVICES`。已验证能跑的写法:

  ```bash
  export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR
  IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"
  pids=()
  for i in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$i]} python qwen/infer_iso.py --variant <臂> \
        --shard_idx $i --num_shards ${#G[@]} --out $INFER_OUTPUT_DIR & pids+=($!)
  done
  st=0; for p in "${pids[@]}"; do wait $p || st=1; done; exit $st
  ```

- **`QWEN_WEIGHTS` 必须 `export`,不能只做命令前缀。** 写成
  `QWEN_WEIGHTS=$INFER_WEIGHTS_DIR IFS=, read ...` 的话它只修饰 `read` 那一条,
  传不到后面的 python,8 个进程齐刷刷报「环境变量未设置」,job 1 秒挂掉。已经踩过一次。
- **权重加载慢,而且会拖 GPU 利用率。** 8 进程并发从 ceph 读实测 490 s,这段时间
  GPU 利用率是 0,而 infer_hub 的 `infer_gpu_idle` 会杀低利用率的任务。
  实测 `--timeout 180` 没被误杀,但**第一个 job 盯一眼日志**。
- **显存 66.7 GB / 卡**(1 卡跑 cache_check 时的峰值),H800 是 80 GB,余量 13 GB。
  8 卡 fan-out 每卡一个进程不叠加。
- **H 集群经常满**,4 台机器(ge90-10/26/49/70)常被别人的 8 卡任务占着。
  `klingai-wlf2-ge124-node194` 是 5kpro 集群,`--cluster h` 不接。排队十几分钟到两小时都可能。

---

## 6. 做什么

### 6.0 ckpt 自检 ✅ 已过(纯 CPU,几秒)

```bash
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python - <<'PY'
import torch
ck = torch.load("/kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt",
                map_location="cpu")
lora = ck["lora"]
nz_b = sum(1 for k, v in lora.items() if "lora_B" in k and v.abs().sum().item() > 0)
bad  = [k for k, v in lora.items() if not torch.isfinite(v).all()]
print(f"step {ck['step']} | rank {ck['rank']} | targets {len(ck['targets'])} | "
      f"张量 {len(lora)} | lora_B 非零 {nz_b} | 非有限 {len(bad)}")
PY
```

期望也实得 `step 2000 | rank 64 | targets 8 | 张量 960 | lora_B 非零 478 | 非有限 0`。
(`478` = 8 模块 × 60 层 − 末层的 `add_q_proj` 和 `to_add_out`,那两个永远拿不到梯度。)
**🔴 对不上就停** —— 尤其 `lora_B 非零 0`,那等于 `iso_post` 退化成 `iso_pre`。

### 6.1 ⏳ 缓存判据 `diag_kv.py` 第二轮(1 卡,约 6 分钟)—— 与主批并行,不阻塞

**第一轮的结论**(你跑的,`diag_kv.json` v1)把问题定位得很干净,分成了两半:

| 问的是 | 结果 |
|---|---|
| ref K/V 随**去噪步**变吗 | **不变**。cond 支路前向 20/40 重算 vs 第 0 步缓存,60/60 层逐位相同 ⇒ 缓存的地基在 bf16+GPU 下成立 ✅ |
| ref K/V 随**文本段**变吗 | **变**。uncond 支路 59/60 层不等(max\|Δ\| 1216,相对 0.246),只有第 0 层——不过注意力那层——相等 |

第二个结果有两种成因,**信号一模一样,但一个是 bug 一个是良性**:

- **(a) 真漏** —— ref 行实际看见了 txt。那 `build_isolated_attn_mask` 就没起作用,
  隔离的地基塌了,缓存跨 cond/uncond 共享是错的。
- **(b) 长度效应** —— cond 与 uncond 的 txt 段长度不同 ⇒ 联合序列总长不同 ⇒
  ref 段相对注意力 kernel 分块边界的对齐不同 ⇒ 累加顺序不同 ⇒ bf16 舍入不同。
  数学上 ref 仍只看 ref。良性。

(倾向 (b) 的一个旁证是你已经测到的:ref K/V 差 0.246 的那次,噪声速度只差
3.6e-3——和 K/V 完全逐位相同那两次的 4.3e-3 同量级。真差 25% 的话输出不可能纹丝不动。
但这是旁证不是判据。)

**成因已经在本地(CPU,不需要模型)判掉了 —— 是 (b)。**(2026-08-14,作者)

从 `qwen/iso_attn.py` 里 AST 抽出真的 `build_isolated_attn_mask` + `IsoContext.prepare`,
接 `F.scaled_dot_product_attention` 跑玩具尺寸(txt 7 / noise 5 / ref 3+3):

| 本地判据 | 结果 |
|---|---|
| 换 txt **内容**(同长度)→ ref 行输出 | `max\|Δ\| = 0`,fp32 与 bf16 都是 |
| 换 noise 内容 → ref 行输出 | `max\|Δ\| = 0` |
| mask 里 ref 行在 txt 列上 | 全 `False` |
| 同长度(64)不同内容 | `max\|Δ\| = 0.000e+00` |
| **不同长度(7 vs 64)** | **`max\|Δ\| = 3.576e-07`** |

最后两行把长度和内容拆开了:**差异 100% 来自长度,与 txt 内容无关** ⇒ (a) 死、(b) 活。
量级也对得上:fp32 下 3.576e-7 ≈ 3 ulp(fp32 eps 1.19e-7);bf16 eps 是 3.9e-3,
同样 3 ulp 即单层 ~1.2e-2,经 60 层累积到 0.246 正常。

真机 kernel 会不会不忠实执行 mask?**你自己的数据已经答了** —— cond 支路 60/60 逐位相同,
说明 mask 确实挡住了 noise;同一个 mask、同一次调用,挡得住 noise 就挡得住 txt。

**所以第二轮不再是阻塞门禁。** 它现在只是把本地结论在真机 bf16 上钉一遍:

**分辨方法不看任何阈值,只看两个逐位判断**,脚本已经加进去了:

- **A(机制自检)**:在前向 1 用**存下来的 cond 文本**重跑 write 路。前向 1 和前向 0
  是同一步、同一个 latent,所以这等于把前向 0 原样重算 —— **必须逐位等于缓存**。
  不等说明探针自己有问题,别去怪缓存。
- **C(判据)**:同样在前向 1,把 cond 文本**沿 token 维翻转**再跑。长度一模一样、
  内容完全不同。逐位不变 ⇒ ref 对 txt 内容是盲的 ⇒ 成因是 (b);变了 ⇒ 成因是 (a)。

提交(和上次一样,1 卡):

```bash
--gpus 1 --timeout 30
--output-dir /kaimm-distill/wuwenxuan/UNO/output/p3_diag_kv
--cmd 'export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; python qwen/diag_kv.py --out $INFER_OUTPUT_DIR'
```

**这一轮和 §6.2 主批同时投,不要串行等。** diag_kv 6 分钟就出结果,主批光排队+加载就
远不止;先后顺序天然对。脚本自己打四选一:

| 它打的 | 含义 | 你做什么 |
|---|---|---|
| `✅ PASS` | 换掉 txt 内容 ref K/V 逐位不变,和本地结论一致 | 记进报告,**主批照跑** |
| `❌ FAIL_LEAK` | 同长度换内容就变了 —— 与本地结论冲突 | **kill 主批**,发给作者 |
| `❌ FAIL_STEP` | cond 支路不再步不变(第一轮没出现) | **kill 主批**,发给作者 |
| `⚠️ INCONCLUSIVE` | A 复算都重现不了缓存 ⇒ 探针机制有问题 | 主批继续跑,把输出发给作者(探针的问题不影响出图) |

顺带多记了 mean\|Δ\| 和不等元素占比(第一轮只有 max)。这些是**记录不是门禁**,
连同噪声速度相对 L2 一起写进报告。

**上一轮的主批处置**:你在门禁出结论前先排队、FAIL 后 kill,没浪费资源,做法很对,
这轮就按这个来。

### 6.2 ⏳ 主批:三个臂 × 240 条

任务表 `datasets/eval_multiref/m6_tasks.json`,`infer_iso.py` 自动按「每层 `i % 4 != 3`」
取 240 条(S1 165 + S3 75),取不到 240 它自己报错停下。

```
python qwen/infer_iso.py --variant full
python qwen/infer_iso.py --variant iso_pre
python qwen/infer_iso.py --variant iso_post --lora /kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt
```

(上面是逻辑,实际按 §5.2 拆成 8 个分片进程;LoRA 路径按 §5.1 走 `--prep-marker`。)

**三个臂的输出目录必须分开**(`output/p3_full` / `p3_iso_pre` / `p3_iso_post`)——
`output_exists` 的续跑逻辑只看文件在不在,撞在一起会静默跳过。

粗估:`full` 约 67 s/img(floor 那轮 8 卡实测)、两个 iso 臂更快。每个臂 8 卡分片下
十几到三十几分钟纯计算,加上 8 分钟权重加载。**这些只是让你估排队时间用的,不是门禁。**

**为什么是 240 不是更少**:盲评判据要求 `n_nontie ≥ 94`,上一轮两次卡在 93 和 89。
192 条在平局率 50% 时只有六成概率达标。**别把它调小**,这个数是算过的。

### 6.3 ✅ floor 30 条 —— 已完成,不用重跑

`--tasks m6_floor --variant full`,渲两遍到 `output/p3_floor/a` 和 `b`,已跑完。
30 对像素差全部 `mean=0.0000 / max=0`(逐位相同)。

**这个结果的含义**:流水线是位级确定的。所以 floor 测到的不是"渲染噪声"
(答案是 0),而是"盲评法官对着两张完全一样的图还会不会选出赢家",也就是判读侧的硬底噪。
这比原设想更有用。**判读口径怎么改是作者和盲评侧的事,你不用管,也不用重渲。**

---

## 7. 要记的数

`report_speed` 会打。全部是**记录,不是门禁**:

| 数 | 说明 |
|---|---|
| 每个臂的**中位 s/img** | 三个臂分开记,按 1-ref / 2-ref 分行 |
| **加速比** | `full` ÷ `iso` |
| **前向次数** | `write N / read M`,每张图应当是 1 写 79 读 |
| `diag_kv` 的**每步扰动相对 L2** | §6.1 那个数 |

三条读数上的提醒:

- 预登记的速度预测是 2-ref 1.9–2.0×、1-ref ~1.4×,**已经被 P2 预检的 2.37× 打脸了**。
  这不用修,预登记就是用来被证伪的,**照实记数,不要去调预测也不要调代码**。
- 8 个进程挤一台机器,绝对 s/img 会比 1 卡时高。**加速比要拿同样并发下的两个臂相比**才公平,
  别把这次的绝对值直接跟上一轮的 1 卡数并排引用。
- 早先在 `DIFFUSERS_ATTN_BACKEND=_native_efficient` 下测到的 2.76–3.28× **不作数**,
  那次 read 路被人为从 flash 拽到了 mem-efficient,分子分母都不是部署时的样子。
  真实加速比只能用**默认 backend** 的主批来测。

---

## 8. 明确不做

- **不做盲评判读、不建拼图、不做配对**。出完图这一单就结束。
- **不加 3-ref 那一层**。要先把它的速度预测写进预登记才能出图,那是作者的事。
- **不改 `PLAN.md`**。§4 是预登记,只能加带日期的订正注记。有要订正的数
  (比如 txt token 实测中位 426,§4.4 账上还写着「估 400–600」)**写进报告,作者来加**。
- **不删已经漏进 git 的单臂图**(`p3_cachecheck` 3 张 `iso_pre` + `p3_floor` 60 张 `full`)。
  删了也不改变已经可见的事实。在报告的局限里声明一句即可。
  **但后续批次不要再往 git 里放单臂图。**

---

## 9. 回报

写进 `reports/20260814-p3-eval/REPORT.md`(已存在,续写):

1. 每个 infer_hub 任务:**提交命令原样** + job_id + 最终状态 + 关键日志段;
2. `diag_kv` 的 stdout 原样 + `diag_kv.json`;
3. 每个臂:出图张数 + `report_speed` 的数 + 输出目录路径;
4. §7 那张表里的数;
5. **踩到的坑原样写**,有 traceback 就贴原文不要转述。§5 里没写到的都算新坑,
   下一个人要靠它。

`output/` 不进 git(`.gitignore` 是白名单模式),图留在 ceph 上。
commit message:`eval(qwen): P3 三臂出图`。
