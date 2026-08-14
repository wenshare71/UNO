# P3 出图 · 执行清单

> 给在 4090 开发机上新开的 claude 会话。你没有这个项目的上下文,§0 先看完。
> 起点 commit `585e720`(或更新的 HEAD)。
> **这一单只出图,不做盲评判读,不建拼图。** 理由见 §1 第三条。

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

代码已经写好了,就是 `qwen/infer_iso.py`,三个臂只差 `--variant`。你**基本不用写代码**,
这一单的难点全在「怎么把它投到远程推理集群上并且不被中途杀掉」。

想看全貌:`qwen/PLAN.md`(§3.3 是这一单)、`qwen/GOAL.md`(项目背景与上一轮的坑)。
不用全读,卡住了再去查。

---

## 1. 三条硬约束

1. **R0 —— `uno/` 和 `distill/` 下既有的 `.py` / `.sh` 一个字不动。** import 可以,改不行。
   `qwen/` 下的也一样:这一单预期不需要改任何代码,真要改先跟我说。
2. **不许改 Q1 口径**:steps 40 / true_cfg 4.0 / 1024² / negative_prompt `" "` /
   prompt 原样 / seed 取自任务表。这些已经写死在 `infer_iso.py` 顶部,别传参绕过去。
   改了基线就废了,整批图作废。
3. **带变体名的拼图不许进 git,你也不要自己建 board。** 盲评要求判读的人看不到哪张是哪个臂,
   拼图和配对由盲评那一侧(`../distill/blind_eval/`)按它自己的纪律做。
   你只产出 `<task_id>.png` 和 `results_shard*.json`。

---

## 2. 机器和网络(已知条件,不用你摸)

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
ceph 读约 136 MB/s、写约 67 MB/s —— 比本地 NVMe 慢 25 倍,后面 §4 有个坑就出在这。

**网络**:出网走日本代理,全通但全慢——HF 0.66 MB/s、PyPI 官方 0.05 MB/s;
内网源 `pypi.corp.kuaishou.com` 命中 `no_proxy` 直连,241 MB/s。要装包只走内网源。
**记得 `export HF_HUB_OFFLINE=1`**:权重全在本地,不设的话 `from_pretrained` 每次都去
HF 探 etag,过日本代理白等十几秒,乘以 8 个进程 × 3 个臂就很可观了。

---

## 3. infer_hub 是什么

多人共享的推理队列。心智模型三条:

1. **状态即目录**:共享盘上一个目录,`pending/` → `claimed/<机器>/` → `done/`/`failed/`。
   没有中心调度器,活着的 worker 自己来抢活。
2. **派活是拉不是推**,你 ssh 不上推理机,也不需要。
3. **代码只能来自已 push 的 commit**,参数走环境变量。改了代码不 push 就投,
   跑的是旧代码——这是最容易踩、而且最难发现的坑。

手册在 `docs/infer_hub/`(`USAGE.md` 是全参数 + 硬规矩 + 常见坑,`FUNCTION.md` 是机制)。
**投之前把 `USAGE.md` §6「硬规矩」和 §7「常见坑」读一遍**,那七条每一条都有人踩过。

上一轮跑通过的命令,拿它当起点(**注意这一单要 `--gpus 8`,上一轮那次是 1 卡冒烟**):

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
SHA=$(git rev-parse HEAD)          # 必须已 push

infer_submit --owner wuwenxuan --project default --cluster h --gpus 8 --timeout 180 \
  --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
  --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --output-dir /kaimm-distill/wuwenxuan/UNO/output/<本批> \
  --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
  --label <实验名>_Iter2000 \
  --cmd '...'
```

看状态:`infer_status --owner wuwenxuan`;看日志:
`tail -f /kaimm-distill/infer_hub/queues/default/logs/<job_id>.log`。

---

## 4. 三个已知的坑,想清楚再投

这一单的难点全在这里。我把我知道的写下来,**具体怎么解你自己定**。

### 4.1 八张卡要你自己 fan out

`infer_iso.py` 自己不起多进程,它靠 `--shard_idx i --num_shards 8` 做分片
(`i % num_shards == shard_idx`,断点续跑靠已存在的 png 跳过)。所以 `--cmd` 里得起 8 个
进程再 `wait`。

**别写死卡号**(`USAGE.md` 硬规矩 2)。worker 注入的 `CUDA_VISIBLE_DEVICES` 具体长什么样
我没验过,先打出来看一眼再决定怎么切。大概是这个意思:

```bash
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} python qwen/infer_iso.py --variant full \
      --shard_idx $i --num_shards ${#G[@]} --out $INFER_OUTPUT_DIR &
done
wait
```

### 4.2 权重加载可能把任务害死

每个进程要从 ceph 读约 60 GB 权重。**8 个进程同时读会互抢带宽**,而且这段时间 GPU 利用率是 0
—— infer_hub 有个 `infer_gpu_idle` 机制,发卡若干分钟后利用率过低会**提前杀掉任务**。
上一轮 1 卡加载约 7 分钟,8 卡并发多久没人量过。

几个方向(自己判断,也可以有别的想法):错开启动让 page cache 先热起来;
`--timeout` 给足;**第一个 job 先只投一个臂,盯着日志确认没被误杀,再投后面两个**。
被杀了别硬重试同一条命令,先看 `failed/` 里的归因。

### 4.3 LoRA ckpt 的路径可能被提交端拒

`iso_post` 臂要 `--lora /kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt`,
而硬规矩 3 是「`--cmd` 里不许出现未声明的共享盘绝对路径」。规矩的原文举的例子是
`cd` 到个人代码目录、`PYTHONPATH` 指向个人仓库,**用意是代码必须来自 commit**,
数据路径未必在管辖范围内——但我没验过提交端的检查是不是一刀切。

先直接试。被拒了的话方向有:把 `--weights` 指到 ckpt 所在目录、用 `$INFER_WEIGHTS_DIR`
拼路径,再用环境变量把模型权重路径传进去。`--dry-run` 可以只打印 job json 不真投,拿它试探。

---

## 5. 出什么

### 5.0 先验一下 ckpt(纯 CPU,几秒,在 4090 上做)

这 10 份 ckpt 到现在一次都没被读回来过。`infer_iso.py` 里的 `apply_lora_ckpt` 自带门槛
(结构对不上、`lora_B` 全 0 都会 raise),但那要等到跑起来才触发,别拿一个远程任务去试。

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

期望 `step 2000 | rank 64 | targets 8 | 张量 960 | lora_B 非零 478 | 非有限 0`。
`478` = 8 模块 × 60 层 − 末层的 `add_q_proj` 和 `to_add_out`(最后一个 block 的
`encoder_hidden_states` 在进 `norm_out` 前就被丢了,那两个永远拿不到梯度)。
**对不上就停下来告诉我**,尤其 `lora_B 非零 0` —— 那等于 `iso_post` 退化成 `iso_pre`。

### 5.1 先做一个判定实验(约 10 分钟,值得单独跑)

P1 门禁里有一条判据**没过**:真权重 bf16 下,「隔离+缓存」和「隔离+每步重算」的图
应当几乎逐像素相同(判据 `mean < 0.5`),实测 1.83–4.26。

我的诊断是**核不对称**而不是缓存逻辑错:写缓存那一步带 `(1,1,L,L)` 的 bool mask,
SDPA 走不了 flash 只能走 mem-efficient;读缓存那 79 步 mask 是 `None`,走 flash。
两个 kernel 的 bf16 累加顺序不同,差异就出在这。旁证是同样 80 次前向、同样序列长度下,
「隔离无缓存」118.0 s/img vs「全注意力」66.7 s/img = 纯 mask 代价 1.77 倍。

判定方法:**把两条路强制压到同一个 kernel 上,看差异塌不塌**。

```bash
DIFFUSERS_ATTN_BACKEND=_native_efficient \
python qwen/infer_iso.py --variant iso_pre --cache_check 3 --limit 3
```

看 stdout 里 `[缓存确认]` 那行的 `mean=`:

- 掉到 0.5 以下 ⇒ 确认是核不对称,缓存逻辑是精确的,P3 的两个 iso 臂可以放心解读;
- 仍在 2–4 ⇒ 缓存本身有实质问题,**停下来告诉我**,后面 720 张图先别跑。

⚠️ **先确认这个环境变量在这个 diffusers 版本上真的生效**——名字对不上的话它会被静默忽略,
那这个实验什么都没证明,却看起来像证明了。去 diffusers 源码里确认变量名和可选值,
或者对比一下有无该变量时 s/img 有没有变化。这是个陷阱,别掉进去。

### 5.2 主批:三个臂 × 240 条

任务表 `datasets/eval_multiref/m6_tasks.json`,`infer_iso.py` 会自动按
「每层 `i % 4 != 3`」取出 240 条(S1 165 + S3 75),取不到 240 它会自己报错停下。

```
python qwen/infer_iso.py --variant full
python qwen/infer_iso.py --variant iso_pre
python qwen/infer_iso.py --variant iso_post --lora <上面那个 step002000.pt>
```

(上面是逻辑,实际要按 §4.1 拆成 8 个分片进程。)

**输出目录三个臂必须分开**,别让它们互相覆盖——`output_exists` 的续跑逻辑只看文件在不在,
撞在一起会静默跳过。

粗估:`full` 约 63 s/img、两个 iso 臂约 28 s/img(上一轮 1 卡实测)。8 卡分片下
每个臂 15–35 分钟纯计算,加上权重加载。

**为什么是 240 不是更少**:判据要求 `n_nontie ≥ 94`,上一轮两次卡在 93 和 89。
192 条在平局率 50% 时只有六成概率达标。**别把它调小**,这个数是算过的。

### 5.3 run_floor 30 条

`datasets/eval_multiref/m6_floor_tasks.json`,`--tasks m6_floor`。

用途:同权重、同 seed、**不同 run** 再渲一遍,量「同一个模型两次生成之间的差异」,
也就是这一批盲评的**批内天花板**。没有它,读不出差异是模型差异还是会话漂移。
上一轮正是靠它把结论钉住的。

我查了一下:这 30 条里有 22 条落在主批那 240 条里(seed 一致),8 条不在。
所以**最稳的做法是把 30 条完整地再渲一遍到一个独立目录**,这样 30 对配对都成立,
是个安全的超集。

用哪个臂渲 floor,我倾向 `full`(它是参照系),但**这属于盲评那侧的口径,你去
`M4_EVAL_SPEC.md` §8 和 `../distill/blind_eval/` 里确认上一轮的做法,别猜**。
查到什么写进报告。

---

## 6. 要记的数

盲评之前,这一单本身要产出几个数,`report_speed` 会打:

| 数 | 说明 |
|---|---|
| 每个臂的 **中位 s/img** | 三个臂分开记 |
| **加速比** | `full` ÷ `iso`。预登记的预测是 2-ref 1.9–2.0×、1-ref ~1.4× |
| **前向次数** | `write N / read M`,每张图应当是 1 写 79 读 |

两条读数上的提醒:

- 预登记那个 1.9–2.0× **已经被打脸了**——P2 预检在 1 卡上实测 2.37×。这不用修,
  预登记就是用来被证伪的,**照实记数,不要去调预测也不要调代码**。
- 8 个进程挤在一台机器上,绝对 s/img 会比 1 卡时高。**加速比要拿同样并发下的两个臂相比**才公平,
  别把这次的绝对值直接跟上一轮的 1 卡数并排引用。

---

## 7. 本轮明确不做

- **不做盲评判读、不建拼图、不做配对**。出完图这一单就结束。
- **不加 3-ref 那一层**。`PLAN.md` §3.3 提到可以加(`q1b_3ref` 那 122 条的 `full` 臂
  已经渲好在 `output/qwen_3ref/`),但加一层要先把它的速度预测写进预登记才能出图,
  那是我的事不是你的。这一轮先把 240 条这条主线走完。
- **不改 `PLAN.md`**。§4 是预登记,只能加带日期的订正注记,不能改。有要订正的数
  (比如 txt token 实测中位 426,§4.4 账上写的还是「估 400–600」)**写进报告告诉我,我来加**。

---

## 8. 回报

新建 `reports/<日期>-p3-eval/REPORT.md`:

1. §5.0 ckpt 自检的输出原样;
2. §5.1 判定实验:完整命令 + `[缓存确认]` 那行原样 + 你怎么确认环境变量生效的;
3. 每个 infer_hub 任务:**提交命令原样** + job_id + 最终状态 + 关键日志段;
4. 每个臂:出图张数 + `report_speed` 的三个数 + 输出目录路径;
5. run_floor 用了哪个臂、依据是在哪份文档哪一节查到的;
6. **踩到的坑原样写**——尤其 §4 那三个我没验过的地方,你的实际做法和失败过的尝试都记下来,
   下一个人要靠它。有 traceback 就贴原文,不要转述。

`output/` 不进 git(`.gitignore` 是白名单模式),图留在 ceph 上。
commit message:`eval(qwen): P3 三臂出图`。

---

## 9. 什么时候停下来问我

- ckpt 自检对不上;
- §5.1 判定实验 `mean` 仍在 2–4;
- 需要改任何 `.py`;
- 任务连续两次以同一个原因失败——别第三次投同一条命令,把归因贴给我;
- 你发现这份清单里某个前提是错的。**我写的东西不是权威,机器上看到的才是**,
  冲突的时候以机器为准,然后告诉我哪里错了。
