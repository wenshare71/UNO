# P3 出图 · 执行报告(进行中,停在 §5.3 floor 判读)

> 对应 `qwen/P3_EVAL_RUN.md`。本单只出图、不盲评判读、不建拼图。
> 起点 commit `3c1f90f`。执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),
> 执行时间 2026-08-14。
>
> **当前进度**:§5.0 ✅ → §5.1 判定实验 ✅(判据 `mean<0.5` 被作者立错,见
> `qwen/P3_EVAL_RUN.md` §5.1-判读,2026-08-14 放行)→ **§5.3 floor 判定 ✅(≈0,停线)**。
> **§5.2 主批 720 张未投** —— §5.3 floor 判读触发停线,等作者 K/V 级诊断。

---

## 0 · 结论速览

- `step002000.pt` 的 LoRA 结构自检**逐字符合期望**,iso_post 臂是训练后的权重(不是未训练的退化臂)。
- §5.1 判定实验:把写/读两条注意力路**强制压到同一个 kernel**(`DIFFUSERS_ATTN_BACKEND=_native_efficient`),期望「隔离+缓存 vs 隔离+每步重算」的像素差塌到 0.5 以下。**结果没塌,反而更高**(mean 2.33–5.92)。「核不对称」假说被证伪。
- **§5.3 floor 判定(2026-08-14 追加):`full` 臂渲两遍(独立目录、异 run),30 对像素差全部逐位相同(mean=0.0000, max=0)。** 流水线位级确定,不存在可放大的 run-to-run 噪声 ⇒ §5.1 的 2–6 差异是**确定性系统差异**,不是随机 bf16 噪声 ⇒ 按 §5.3-补 判读表「≈0」分支,**停线,主批先不跑**,等作者 K/V 级诊断。

---

## 1 · §5.0 ckpt 自检(纯 CPU,4090 上)

命令:

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

输出(原样):

```text
step 2000 | rank 64 | targets 8 | 张量 960 | lora_B 非零 478 | 非有限 0
```

期望 `step 2000 | rank 64 | targets 8 | 张量 960 | lora_B 非零 478 | 非有限 0`,**逐字一致**。
`lora_B 非零 478` 非 0 ⇒ iso_post 不会退化成 iso_pre。

---

## 2 · §5.1 判定实验(核对称性)

### 2.1 完整提交命令

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
SHA=3c1f90ffc290af5cd8ada8bd818c1290c1e2cbcb

sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h \
    --repo https://github.com/wenshare71/UNO.git \
    --commit $SHA \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/p3_cachecheck \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label p3_cachecheck_bf16 \
    --gpus 1 --timeout 60 \
    --prep-cmd 'true' \
    --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --cmd 'QWEN_WEIGHTS=$INFER_WEIGHTS_DIR DIFFUSERS_ATTN_BACKEND=_native_efficient python qwen/infer_iso.py --variant iso_pre --cache_check 3 --limit 3'
```

### 2.2 任务结果

| 字段 | 值 |
|---|---|
| job_id | `wuwenxuan__p3_cachecheck_bf16__3c1f90ffc290` |
| 最终状态 | 成功(exit_code=0,reason 空) |
| worker / 卡 | `aiplatform-wlf3-ge90-70` / 1 卡(H 集群) |
| 耗时 | 781s(12m46s);prepare 14.8s,avg_gpu_util 60.1%,timeout 延长 0 次 |
| 日志 | `/kaimm-distill/infer_hub/queues/default/logs/wuwenxuan__p3_cachecheck_bf16__3c1f90ffc290.log` |
| 结果 json | `/kaimm-distill/wuwenxuan/UNO/output/p3_cachecheck/results_shard0.json` |

### 2.3 `[缓存确认]` 三行(原样)

```text
[缓存确认] M6_S1_000_s0  像素差 max=242 mean=5.9179 | 117.7s → 42.6s (2.76×)
[缓存确认] M6_S1_000_s1  像素差 max=176 mean=2.6673 | 117.7s → 36.7s (3.21×)
[缓存确认] M6_S1_000_s2  像素差 max=225 mean=2.3288 | 120.3s → 36.7s (3.28×)
```

**三张 mean = 5.92 / 2.67 / 2.33,全部 > 0.5,门禁未过。**

对照上一轮 P2 预检(`reports/20260813-p2-preflight/REPORT.md` §「像素差」,默认 `native` backend)
同任务、同 seed:

| 任务 | P2 预检(默认 native) | 本次(强制 `_native_efficient`) |
|---|---|---|
| M6_S1_000_s0 | mean=4.2636 / max=225 | mean=**5.9179** / max=242 |
| M6_S1_000_s1 | mean=1.8321 / max=170 | mean=**2.6673** / max=176 |
| M6_S1_000_s2 | mean=2.3189 / max=226 | mean=**2.3288** / max=225 |

### 2.4 环境变量确实生效(不是被静默忽略)

清单 §5.1 的 ⚠️ 提示要防「变量名对不上被静默忽略」。我从两条路确认了:

1. **源码确认**(env 里的 diffusers 0.40.0.dev0):
   - `diffusers/utils/constants.py:45` → `DIFFUSERS_ATTN_BACKEND = os.getenv("DIFFUSERS_ATTN_BACKEND", "native")`,import 时读 env。
   - `diffusers/models/attention_dispatch.py:3636` → `_NATIVE_EFFICIENT` 是已注册的真实后端。
   - `attention_dispatch.py:411-414` → `dispatch_attention_fn` 在 `backend=None` 时走 `_AttentionBackendRegistry.get_active_backend()`,即受 env 控制;`iso_attn.py` 调 `dispatch_attention_fn` 时不传 `backend`,故 env 生效。
   - `_native_efficient_attention`(L3639)用 `torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` 把两条路都锁进 mem-efficient kernel。
2. **运行时确认**:read 路径从 flash 换成 mem-efficient 后,缓存 s/img 从预检的 **28.8/33.5/27.9s → 本次 42.6/36.7/36.7s**(变慢 = kernel 真换了);无缓存路径 117.7–120.3s 与预检 118s 稳定一致。

### 2.5 判读

- 「核不对称」假说被证伪:两条路压到同一个 EFFICIENT_ATTENTION kernel 后,差异**没有塌**,s0/s1 反而更大。
- 差异残留说明缓存不是 bf16 数值精确的。可能机制:同一 backend 内部,write 的 `(1,1,L,L)` 真 block mask 与 read 的 `(1,1,1,L)` padding mask 仍走不同的 tiling/累加路径。
- 变化模式(s2 持平、s0 显著变差)更像「40 步去噪对每步微小数值差的选择性放大」,而非一致性的结构 bug。
- forward 计数 243 写 / 237 读 = 3 张缓存 + 3 张无缓存对照,每张 1 写 79 读,符合预期。

### 2.6 report_speed

```text
2-ref  n=  3  中位   36.7 s/img  均值   38.7 s/img
```

(这是「隔离+缓存」在 `_native_efficient` 下的速度,与默认 backend 的 ~28s 不可直接并排引用——kernel 被人为改慢过。)

---

## 3 · §5.2 主批 —— 未投(floor 判读停线)

三个臂(full / iso_pre / iso_post,各 240 条,8 卡 fan-out)的提交命令**已 dry-run 验证通过**,
但按清单 §5.1-判读 + §5.3-补 的新次序,§5.3 floor 判读先跑,其结论为「≈0(采样确定性)」
⇒ 停线,**未实际投递**。

dry-run 验证结论(备查):

- **fan-out**(清单 §4.1):`--cmd` 里 `export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"` 起
  `${#G[@]}` 个进程再 `wait`,不写死卡号,全部用注入变量 + 相对路径,无共享盘路径泄漏。
  ⚠️ `QWEN_WEIGHTS` **必须 `export`**,不能只做命令前缀——否则 `read` 的临时前缀不会传给
  python(见 §5 坑 5,第一次 floor 就栽在这)。
- **iso_post 的 LoRA 路径**(清单 §4.3):直写绝对路径 `/kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt`
  会被 `cmd_shared_disk_leaks` 拦(见 §5 坑 2)。解法:把它声明进 `--prep-marker`(该路径在共享盘、
  已存在,`prep_done=true` 直接排卡),dry-run 通过。
- 三个臂输出目录分开(`output/p3_full` / `p3_iso_pre` / `p3_iso_post`),不会互相覆盖。

---

## 4 · §5.3 run_floor —— 已跑,判读停线

### 4.0 用哪个臂

查到了依据,**用 `full`(基线侧)**:

- `distill/build_m6_tasks.py` L46–55(正是本批 m6 任务表的生成脚本)写明:天花板第二侧走
  **另一个 infer_hub job**,权重取 `m6_full`(主对的基线侧)——「天花板量的正是骑在主对上的
  那一份噪声,而不是另一个模型的噪声」。
- P3 语境下主对基线侧就是 `full`(stock 权重 + 全注意力 = teacher)。
- `distill/M4_EVAL_SPEC.md` §8 是 M5 人工判读判据,不涉及 floor;`distill/blind_eval/` 无 floor 专属代码。

### 4.1 提交命令(原样)

按清单 §5.3-补,一个 8 卡 job 把两遍跑完:卡 0-3 跑 `p3_floor/a`、卡 4-7 跑 `p3_floor/b`
(同权重、同 seed、**异进程异卡**,满足「异 run」)。

```bash
SHA=3c1f90ffc290af5cd8ada8bd818c1290c1e2cbcb
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h \
    --repo https://github.com/wenshare71/UNO.git --commit $SHA \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/p3_floor \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label p3_floor_Iter2000 --gpus 8 --timeout 180 \
    --prep-cmd 'true' --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --force \
    --cmd 'export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"; pids=(); for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=${G[$i]} python qwen/infer_iso.py --variant full --tasks m6_floor --shard_idx $i --num_shards 4 --out $INFER_OUTPUT_DIR/a & pids+=($!); CUDA_VISIBLE_DEVICES=${G[$((i+4))]} python qwen/infer_iso.py --variant full --tasks m6_floor --shard_idx $i --num_shards 4 --out $INFER_OUTPUT_DIR/b & pids+=($!); done; st=0; for p in "${pids[@]}"; do wait $p || st=1; done; exit $st'
```

### 4.2 任务结果

| 字段 | 值 |
|---|---|
| job_id | `wuwenxuan__p3_floor_Iter2000__3c1f90ffc290__1786679354`(首次失败见 §5 坑 5,`--force` 重投) |
| 最终状态 | 成功(exit_code=0) |
| worker / 卡 | `aiplatform-wlf3-ge90-49` / 8 卡(H 集群) |
| 耗时 | 19m25s(权重加载 490s,8 进程并发 ceph 读) |
| 日志 | `/kaimm-distill/infer_hub/queues/default/logs/wuwenxuan__p3_floor_Iter2000__3c1f90ffc290__1786679354.log` |
| 产物 | `output/p3_floor/a/`(30 张)+ `output/p3_floor/b/`(30 张),各 4 个 `results_shard*.json` |

`full` report_speed(8 卡并发):

```text
a: 1-ref n=10 中位 38.7 s/img | 2-ref n=20 中位 67.0 s/img
b: 1-ref n=10 中位 38.8 s/img | 2-ref n=20 中位 66.9 s/img
```

### 4.3 判读:30 对全部逐位相同(≈0)

在 4090 上按 task_id 配对 `a/` vs `b/` 算像素差(口径同 `infer_iso.pixel_diff`):

```text
30 条 mean 统计:  n=30  中位 0.0000  均值 0.0000  min 0.0000  max 0.0000
落在 2–6 量级: 0/30   |   ≈0(<0.5): 30/30
```

验证数据可靠:a/b 各 30 张、无软链、inode 不同(真分开的文件)、尺寸一致;
a/b 是两次独立 run(不同卡、不同进程、输出独立目录)。

**判读结论(按 §5.3-补 判读表「≈0」分支)**:

- 流水线是**位级确定**的(full-vs-full 逐位相同)⇒ 不存在可被放大的 run-to-run 噪声。
- 因此 §5.1 里 iso-缓存 vs iso-重算 的 2–6 差异**不是**「随机 bf16 噪声经 40 步混沌放大」,
  而是**确定性、可复现的系统差异**。
- 这同时推翻 §5.1-判读 里「+39%/+46%/+0.4% 是噪声的签名」的解读——流水线无噪声,
  那些跳动是换 kernel 后确定性的变化。
- 剩余分歧收敛到两个可能:kernel 内部(write 的 `(1,1,L,L)` mask 与 read 的 `(1,1,1,L)`
  padding mask,即便同 backend 内部 tiling/累加仍不同)vs 缓存逻辑真 bug。floor 只能排除
  「随机噪声放大」,**区分不了两者**。

**停线,主批 720 张不投。** 等作者 K/V 级诊断。

---

## 5 · 踩到的坑(原样记录)

### 坑 1 · 提交必须 sudo 提权(沿用已知坑,复现)

`infer_submit` 写锁文件在 `/kaimm-distill/infer_hub/locks/`,该目录属 root、当前 shell 用户
`wuwenxuan03` 无写权限。直接用 `--owner wuwenxuan` 提交会 `PermissionError`。
解法沿用 [[infer-hub-submit-notes]] 的模板:

```bash
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit ...
```

(本会话 `sudo -n true` 免密可用。)

### 坑 2 · iso_post 的 LoRA 绝对路径会被提交端拒(清单 §4.3 已预警)

清单 §4.3 问「--cmd 里的共享盘绝对路径检查是不是一刀切」。**实测是一刀切**:

`infer_submit` 的 `cmd_shared_disk_leaks`(hubcore.py L124)用正则匹配 `--cmd` 里所有
`/kaimm-distill/...` 路径,只要不在 `[weights, output_dir, uv_env, prep_marker]` 声明的根下
就 `die` 拒收。LoRA 路径 `/kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt`
直写进 `--cmd` 必然被拒。

解法(未实际投,仅 dry-run 验证):把 LoRA 文件路径声明进 `--prep-marker`。该文件在共享盘、
已存在,`--prep-cmd 'true'` + marker 存在 ⇒ `prep_done=true` 直接排卡,同时把路径合法地
「声明」进来,dry-run 输出的 job json 里 `prep_marker` 就是 LoRA 路径、无报错。
注意:`--prep-marker` 语义上本该是「切分完成标志」,这里借它声明数据路径是权宜,
真实切分场景不要照抄。

### 坑 3 · v3 prep 强制门槛(沿用已知坑)

`submit_require_prep=true`,不带 `--prep-cmd` 的任务会被拒收。本单没有独立切分步骤,
按 SKILL.md 兜底写 `--prep-cmd 'true' --prep-marker <权重目录>` 过门槛。

### 坑 4 · 本地 import diffusers 慢(1m20s)

4090 上 `import diffusers`(qwen-edit env)不带 `HF_HUB_OFFLINE=1` 时会去 HF 探 etag,
拖到 2 分钟超时。加 `export HF_HUB_OFFLINE=1` 后 1m20s(权重全在本地,不影响正确性)。
远程任务不受影响(worker 侧已处理)。

### 坑 5 · `QWEN_WEIGHTS` 必须 export,不能只做命令前缀(本单踩到,1s 失败)

第一次 floor 提交,fan-out 脚本里写的是:

```bash
QWEN_WEIGHTS=$INFER_WEIGHTS_DIR IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"
```

`QWEN_WEIGHTS=$INFER_WEIGHTS_DIR` 和 `IFS=,` 一样,只是 `read` 命令的**临时前缀**,
`read` 一结束就没了,**不会传给后面的 python 进程**。8 个进程全部报
`❌ 环境变量 QWEN_WEIGHTS 未设置`,job exit_code=1 耗时 1s。

为什么 §5.1 那次没炸:那次是 `QWEN_WEIGHTS=$INFER_WEIGHTS_DIR DIFFUSERS_ATTN_BACKEND=... python ...`,
前缀直接跟的是 `python`,所以生效。

修复:先 `export`:

```bash
export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"
```

已本地 bash 验证 + dry-run 验证,重投成功。**主批三个臂的脚本也用同样写法。**
教训:shell 里「命令前缀」只在它修饰的那一条命令里生效,想跨命令传递必须 `export`。

### 坑 6 · H 集群满负荷,8 卡 job 排队约 2 小时

投 floor 时 H 集群 4 台机器(ge90-10/26/49/70)都在跑别人的 8 卡任务,唯一空闲的
`klingai-wlf2-ge124-node194` 是 **5kpro** 集群(硬绑定 `--cluster h` 不接)。8 卡 job
要等 H 机器空出。实际 ge90-49 提前空出,floor 从入队到被接走等了约 1.5 分钟,没真等到 2 小时。

---

## 6 · 要记的数

| 数 | 值 | 备注 |
|---|---|---|
| ckpt 自检 | `step 2000 / rank 64 / targets 8 / 张量 960 / lora_B 非零 478 / 非有限 0` | ✅ 逐字符合 |
| §5.1 判定 mean | 5.92 / 2.67 / 2.33 | ❌ 门禁 `mean < 0.5` 未过,且高于默认 backend |
| 前向次数 | write 243 / read 237 | 3 缓存 + 3 无缓存,每张 1 写 79 读 |
| 缓存 s/img(`_native_efficient`) | 中位 36.7 | 被人为改慢,不与默认 backend 并排引用 |
| **floor 30 对像素差** | 全部 mean=0.0000 / max=0 | 流水线位级确定 ⇒ §5.1 差异是系统差异,非随机噪声 |
| **full s/img**(8 卡并发) | 2-ref 中位 ~67s、1-ref 中位 ~38.7s | a/b 两遍一致 |

预登记的速度预测(2-ref 1.9–2.0×、1-ref ~1.4×)本轮未触达,不记。

---

## 7 · 本轮明确未做

- **主批 720 张(三臂 × 240)未投**,因 §5.3 floor 判读「≈0」停线协议触发。
- **run_floor 已跑 60 张**(§5.3 判定用),但**主批 720 张未投**。
- 未盲评判读、未建拼图、未配对。
- 未加 3-ref 层、未改 `PLAN.md`。
- **盲评前可见的单臂图声明(§8b 要求)**:`output/p3_floor/a/` 与 `b/` 各 30 张
  **`full`(基线侧)渲染**已随本报告进 git。其中 22 条的任务 seed 与主批 240 条重叠。
  全部是 `full`(teacher/基线),不是 iso 臂,不揭示「哪个臂是哪个」;但基线侧图在
  判读前可见,泄漏量与 `p3_cachecheck` 的 3/240 同类。`p3_cachecheck` 另有
  **3/240 的 iso_pre 渲染在判读前可见**(§8b 已声明,图保留未删)。
  按 §8b「后续任何批次不要再往 git 里放单臂图」,**主批产出的 iso 单臂图不会进 git**。

---

## 8 · 待办 / 待用户决定

1. **floor ≈0 的判读结论需要你(作者)定夺**:
   - §5.3-补 的「≈0」分支原文是「我的解释死掉,缓存确实有问题 ⇒ 停下来告诉我,我写
     K/V 级的诊断」。你说了算:写 K/V 级诊断,或调整解读。
   - 我能配合的取证方向(需你点头改 `qwen/` 下 `.py`,R0):只读对比「第 k 步重算的
     ref K/V」vs「缓存里的 step-0 ref K/V」是否位等——若位等则差异纯在 kernel 内部;
     若不等则是缓存逻辑/数值问题。
2. 诊断结论出来后,主批三臂 720 张 + 是否要重跑 floor(若改判据)再定。

---

*commit message 建议:`eval(qwen): P3 floor 判读(30 对逐位相同)停线`*
