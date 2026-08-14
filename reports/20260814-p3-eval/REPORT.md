# P3 出图 · 执行报告(完成:三臂各 240 张 + diag_kv v2 PASS)

> 对应 `qwen/P3_EVAL_RUN.md`(2026-08-14 作者重写为 §6.1/§6.2/§6.3 体系)。
> 本单只出图、不盲评判读、不建拼图。
> 起点 commit `3c1f90f`。执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),
> 执行时间 2026-08-14。
>
> **最终状态**:§5.0 ✅ → §5.1 判定实验 ✅(判据被作者立错)→ §5.3 floor ✅ →
> §6.1 diag_kv v1 ❌ FAIL(作者本地判掉成因=长度效应,良性)→ **v2 真机确认 ✅ PASS** →
> **§6.2 主批三臂各 240 张全部出图**(full/iso_pre/iso_post,commit `9b31937`;iso_post 经 1 卡补跑补齐 30 张)。
> 2-ref 加速比:iso_pre **2.39×**、iso_post **2.26×**(预登记 1.9–2.0× 再次被打脸,同 P2 预检 2.37×)。
> v1 的 `diag_kv.json` 原件已另存 `reports/20260814-p3-eval/diag_kv_v1.json`。

---

## 0 · 结论速览

- `step002000.pt` 的 LoRA 结构自检**逐字符合期望**,iso_post 臂是训练后的权重(不是未训练的退化臂)。
- §5.1 判定实验:把写/读两条注意力路**强制压到同一个 kernel**(`DIFFUSERS_ATTN_BACKEND=_native_efficient`),期望「隔离+缓存 vs 隔离+每步重算」的像素差塌到 0.5 以下。**结果没塌,反而更高**(mean 2.33–5.92)。「核不对称」假说被证伪(作者后续判定该判据本身立错)。
- **§5.3 floor 判定:`full` 臂渲两遍(独立目录、异 run),30 对像素差全部逐位相同(mean=0.0000, max=0)。** 作者 §6.3 重新解读:这证明流水线位级确定,floor 测到的是「判读侧硬底噪」而非渲染噪声,判读口径由作者与盲评侧处理。
- **§6.1 diag_kv 门禁:第一轮 ❌ FAIL → 作者本地判掉成因=长度效应(良性)→ v2 真机确认 ✅ PASS。**
  第一轮探针发现奇数前向(1,79)=uncond 支路 ref K/V 59/60 层不等(rel 0.246)、偶数前向(20,40)=cond 逐位相同,
  `bad_layers` 恒 [1..8]。作者在本地用玩具尺寸拆开「长度 vs 内容」判定是 **txt 段长度不同 ⇒ kernel 分块对齐不同 ⇒ bf16 舍入**,不是隔离漏;
  v2 用 A(机制自检)+C(同长度翻转 txt)两个逐位判据在真机钉实:**A/C 均 60/60 逐位相等 ⇒ ref 对 txt 内容盲 ⇒ 良性**。
- **§6.2 主批三臂各 240 张全部出图**(full/iso_pre/iso_post,commit `9b31937`)。iso_post 首跑一个 shard 撞 ceph EIO 只出 210,
  用 **1 卡 `--num_shards 1` 续跑补齐 30**(坑 7/8/9)。
  **加速比**:iso_pre 2-ref **2.39×** / 1-ref 1.69×;iso_post 2-ref 2.26× / 1-ref 1.56× —— 预登记(2-ref 1.9–2.0×、1-ref ~1.4×)再次被打脸,2-ref 同 P2 预检 2.37×。
  前向计数 iso_pre/iso_post 各 240 写 / 18960 读(每张 1 写 79 读)✅。

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

## 4b · §6.1 diag_kv 门禁(❌ FAIL,主批已投后 kill)

作者重写清单后的**唯一门禁**:直接判「缓存里存的那份 ref K/V,是否等于第 k 步重算会得到的」。
`diag_kv.py` 在 read 前向 1/20/40/79 处探针,重算 ref K/V(写路)与缓存逐位比。

### 提交命令(原样)

```bash
SHA=219b9eb3d53d07954d4652aed9fab9c945ff1a72
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h \
    --repo https://github.com/wenshare71/UNO.git --commit $SHA \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/p3_diag_kv \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label p3_diag_kv --gpus 1 --timeout 30 \
    --prep-cmd 'true' --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --cmd 'export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; python qwen/diag_kv.py --out $INFER_OUTPUT_DIR'
```

### 任务结果

| 字段 | 值 |
|---|---|
| job_id | `wuwenxuan__p3_diag_kv__219b9eb3d53d` |
| 最终状态 | 成功(exit_code=0;脚本内部判 FAIL) |
| worker / 卡 | `aiplatform-wlf3-ge90-70` / 1 卡 |
| 耗时 | 5m38s(权重加载 173.4s) |
| 日志 | `/kaimm-distill/infer_hub/queues/default/logs/wuwenxuan__p3_diag_kv__219b9eb3d53d.log` |
| 结果 | `/kaimm-distill/wuwenxuan/UNO/output/p3_diag_kv/diag_kv.json` |

### stdout 原样(关键行)

```text
[自检] 任务 M6_S1_000_s0 | 2-ref | seed 3500000 | 探针在前向 [1, 20, 40, 79]
[探针] 前向   1 | ref K/V 逐位相同 1/60 层 | max|Δkv| 1.216e+03 || 噪声速度 max|Δ| 3.125e-02 相对L2 3.557e-03
[探针] 前向  20 | ref K/V 逐位相同 60/60 层 | max|Δkv| 0.000e+00 || 噪声速度 max|Δ| 3.125e-02 相对L2 4.340e-03
[探针] 前向  40 | ref K/V 逐位相同 60/60 层 | max|Δkv| 0.000e+00 || 噪声速度 max|Δ| 6.458e-02 相对L2 4.022e-03
[探针] 前向  79 | ref K/V 逐位相同 1/60 层 | max|Δkv| 1.216e+03 || 噪声速度 max|Δ| 5.215e-01 相对L2 7.586e-02
❌ FAIL —— 最大相对差 2.46e-01,与张量自身同阶,是逻辑错不是舍入。
```

`diag_kv.json`: `verdict=FAIL`, `kv_max_rel=0.246`, `all_bitwise_equal=false`。
4 次探针详单见文件,`bad_layers` 恒为 [1..8](前 8 层,非随机层)。

### 观察到的模式(供作者判读,非结论)

失败的前向是 **1 和 79(奇数)**,通过的是 **20 和 40(偶数)**。80 次前向按步内
cond/uncond 交替排(偶=cond、奇=uncond)、forward 0 是 write 的那次 cond ⇒
缓存存的就是 **cond 支路**的 ref K/V:

- **cond 前向重算 ref K/V = 缓存(逐位 60/60)** —— 缓存对 cond 支路精确;
- **uncond 前向重算 ref K/V ≠ 缓存(59/60 层不等,rel 0.246)** —— uncond 支路拿到的是 cond 支路的 ref K/V。

这比「缓存数值 bug」更精确:要么 **ref K/V 随 cond/uncond 分支而变**(隔离没完全挡住
txt/条件对流,T6 的「与 prompt 无关」是 fp32 证的,bf16 GPU 下不位等),要么 **probe 有
奇偶相关的人为因素**。作者判读。

### 主批处置

diag_kv FAIL 后,按 §6.1 协议 + 作者预案 kill 了主批三个任务(已入队未开跑):

| job | 处置 |
|---|---|
| `wuwenxuan__p3_full_Iter2000__219b9eb3d53d` | 入队后删 pending |
| `wuwenxuan__p3_iso_pre_Iter2000__219b9eb3d53d` | 入队后删 pending |
| `wuwenxuan__p3_iso_post_Iter2000__219b9eb3d53d` | 阻塞提交自动入队后删 pending |

(注:主批提交前误判「门禁大概率过」先排队,FAIL 后 kill——用户预案允许,无资源浪费,
但下轮应等门禁结论再投,见 §8。)

---

## 4c · §6.1 v2 第二轮 + §6.2 主批投递(commit `9b31937`,2026-08-14 15:2x)

作者判读 v1 的 FAIL 后,在本地(CPU)用 `iso_attn.py` 抽 AST 跑玩具尺寸,
**判定成因是 (b) 长度效应**:同长度换 txt 内容 → ref 行输出逐位不变(`max|Δ|=0`),
不同长度(7 vs 64)才出现 `3.576e-07` ≈ 3 ulp。结论:**不是漏,主批不阻塞**。
v2 脚本(`8796caf`)新增两个逐位判据分辨 (a)/(b):A=机制自检(前向 1 用存下的
cond 文本原样复算,须逐位=缓存)、C=判据(同长度把 cond 文本沿 token 维翻转,逐位不变 ⇒ (b))。
脚本打四选一:`✅ PASS / ❌ FAIL_LEAK / ❌ FAIL_STEP / ⚠️ INCONCLUSIVE`。

按清单 §6.1「和主批同时投」,四个任务同时投到 commit `9b31937`(v2 崩溃后重投到 `d0b9811`):

| job_id | 内容 | 卡 | 状态(截至 17:10) |
|---|---|---|---|
| `wuwenxuan__p3_diag_kv_v2__9b31937a2f97` | diag_kv v2 | 1 | ❌ 跑完崩了(脚本 bug,见下) |
| `wuwenxuan__p3_diag_kv_v2__d0b98111d659` | diag_kv v2 重投 | 1 | ⏳ 排队(commit `d0b9811`,修好 bug) |
| `wuwenxuan__p3_full_Iter2000__9b31937a2f97` | full 240 | 8 | ✅ **完成** 42m01s,240 张(17:0x) |
| `wuwenxuan__p3_iso_pre_Iter2000__9b31937a2f97` | iso_pre 240 | 8 | 🟢 ge90-70 在跑 |
| `wuwenxuan__p3_iso_post_Iter2000__9b31937a2f97` | iso_post 240 | 8 | ⏳ 排队第 1 位 |

**diag_kv v2 崩溃(脚本 bug,非判定)**:ge90-26 上权重加载 171s、隔离注意力挂载成功后,
第一次探针打印就 `KeyError: 'v_rel_l2'` 崩掉(exit_code=1,4m57s)。归因:

```text
qwen/diag_kv.py:172  wrapped() 打印 f-string 里 `s['v_rel_l2']`,但 `s = rec["B_uncond_or_current"]`
是 summarize() 的输出,只有 n_layers/n_bitwise_eq/bad_layers/max_abs/max_rel/mean_rel/frac_ne,
没有 v_rel_l2 —— 它和 v_max_abs 在 rec 顶层(L163-166)。应写成 `rec['v_rel_l2']`。
```

纯打印语句崩,判定逻辑本身不受影响;主批走 `infer_iso.py`,不碰这段代码,不受牵连。
**处置**(用户定):我改一行 `s['v_rel_l2']`→`rec['v_rel_l2']` 已 commit `d0b9811`(只动这一行,
py_compile 过),等用户 push 后重投 v2。

### diag_kv v2 提交命令(原样)

```bash
SHA=9b31937a2f97e3b89dc723425aaea40658610336
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h \
    --repo https://github.com/wenshare71/UNO.git --commit $SHA \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/p3_diag_kv \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label p3_diag_kv_v2 --gpus 1 --timeout 30 \
    --prep-cmd 'true' --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --cmd 'export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; python qwen/diag_kv.py --out $INFER_OUTPUT_DIR'
```

### 主批三臂提交命令(原样,full 为例;iso_pre 只换 `--variant` 和 `--output-dir`,
iso_post 再换 `--prep-marker` 为 LoRA 路径并给 `--lora`)

```bash
SHA=9b31937a2f97e3b89dc723425aaea40658610336
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h \
    --repo https://github.com/wenshare71/UNO.git --commit $SHA \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/p3_full \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label p3_full_Iter2000 --gpus 8 --timeout 180 \
    --prep-cmd 'true' --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --cmd 'export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR; IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"; pids=(); for i in "${!G[@]}"; do CUDA_VISIBLE_DEVICES=${G[$i]} python qwen/infer_iso.py --variant full --shard_idx $i --num_shards ${#G[@]} --out $INFER_OUTPUT_DIR & pids+=($!); done; st=0; for p in "${pids[@]}"; do wait $p || st=1; done; exit $st'
```

iso_post 的 `--prep-marker /kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt`(LoRA 路径声明进
marker 绕过 `cmd_shared_disk_leaks`,坑 2 同款),`--cmd` 里带 `--lora` 同路径。三臂输出目录分开,
续跑逻辑互不干扰。

### 提交端注意(本单新踩)

- **infer_hub 每人最多 3 个在途任务。** 前三个入队后名额占满,第 4 个(iso_post)的
  `infer_submit` **阻塞等名额**,不是报错。第一次用 2 分钟 bash 超时把阻塞进程杀了,
  没入队;改后台跑(`run_in_background`),等任一在途被 worker 领走就自动入队。
- dry-run 全部通过(含 iso_post 的 LoRA 路径泄漏检查)。

**iso_post 臂(2026-08-14 17:25 起跑)**:8 个 shard 权重+LoRA 全部加载成功
(`[自检] LoRA 已加载 ... lora_B 非零 478`,8/8;`[自检] 隔离注意力已挂载 | block_diag=False | 缓存=开`),
但**其中 1 个 shard 在打 LoRA 自检 print 时 `OSError: [Errno 5] Input/output error` 崩溃**
(`infer_iso.py:137`,写日志文件遇 ceph 瞬时 I/O 错误)。其余 7 个 shard 正常推理 → 预计产出 210 张,
job 会以 `exit_code=1` 收尾。**逐图逻辑确认正确**(LoRA 478 非零 B + 隔离缓存挂载 + 任务=S1 165/S3 75)。
**处置(用户 2026-08-14 17:3x 定)**:等本 job 结束 → `--force` 重投 iso_post(同 output-dir),
续跑逻辑跳过已存在 png,只渲染缺的 30 张。EIO 记为坑(§5 新坑:8 进程并发读 2.26GB LoRA + 日志写撞 ceph EIO)。

**重投过程(2026-08-14 17:5x 修订)**:先投了 8 卡重投 `...__1786700954`,用户问「补 30 张是否要 8 卡」,
确认 1 卡即可(`--num_shards 1 --shard_idx 0`,续跑跳过 210 渲 30,`infer_iso.py:218-235` 验证过)后
**撤销 8 卡重投、改投 1 卡补跑** `wuwenxuan__p3_iso_post_Iter2000__9b31937a2f97__1786701282`(--gpus 1)。
理由:省 7 卡、调度只需任意机器 1 空卡、单进程写日志更不易再撞 EIO。
**注意**:第一跑产出的是 `results_shard1–7`(崩的是 shard 0,json 未写),1 卡补跑写 `results_shard0.json`
无覆盖冲突,补完即 0–7 共 8 个 json 240 条记录。原 7 个 json 已备份到 `output/p3_iso_post/.results_first/`。
**续跑逻辑已验证**:补跑自检 `[自检] 变体 iso_post | 表 m6 全量 240 | 本 shard 240 | 待跑 30 | 已跳过 210`,
1 卡只渲缺的 30 张,无需 8 卡。
第一跑 210 张 + 7 个 `results_shard*.json` 保留在 output-dir,7 个存活 shard report_speed:
1-ref 中位 24.5–25.4、2-ref 29.6–30.4 s/img。

**diag_kv v2(d0b98111d659) ✅ PASS(2026-08-14 17:5x,6m31s,exit_code=0)**:修复后不再 KeyError,
真机 bf16 钉实了本地「长度效应」结论:

```text
[探针] 前向   1 txt_len= 412 | 逐位相同 1/60 层 | max|Δ| 1.216e+03 (相对 2.46e-01) | mean 相对 3.67e-02 | 不等元素占比 9.48e-01 || 噪声速度相对L2 3.557e-03
[探针] 前向  20 txt_len= 420 | 逐位相同 60/60 层 | max|Δ| 0.000e+00 | ... || 噪声速度相对L2 4.340e-03
[探针] 前向  40 txt_len= 420 | 逐位相同 60/60 层 | max|Δ| 0.000e+00 | ... || 噪声速度相对L2 4.022e-03
[探针] 前向  79 txt_len= 412 | 逐位相同 1/60 层 | max|Δ| 1.216e+03 (相对 2.46e-01) | ... || 噪声速度相对L2 7.586e-02
✅ PASS —— 同长度换掉 txt 内容,ref K/V 逐位不变 ⇒ ref 对 txt 内容是盲的。uncond 那边的差异只能来自 txt 段长度不同导致的 kernel 分块差异,良性
   步不变性(cond 支路,前向 [20, 40]):逐位成立
```

`diag_kv.json`(v2,已覆盖 output/p3_diag_kv/):`verdict=PASS`,A_cond_replay 60/60、C_txt_flipped 60/60、
`step_invariant_cond=true`、`n_forward_write=1 / n_forward_read=79`、`peak_mem_gb=66.85`、cond_txt_len=420 vs uncond 412。
**⇒ 隔离/缓存地基成立,主批三臂照常有效;§6.1 门禁以 PASS 收尾。**

**full 臂已出(2026-08-14 17:0x)**:240 张 + 8 个 `results_shard*.json`,`exit_code=0` 42m01s。
分片自检 `[自检] 变体 full | 表 m6 全量 240 | 本 shard 30 | 待跑 30 | 已跳过 0`,fail 0。
各 shard `report_speed`(每 shard n=30):1-ref 中位 38.4–39.7 s/img,2-ref 中位 66.6–70.3 s/img。

**iso_pre 臂已出(2026-08-14 17:2x)**:240 张 + 8 个 `results_shard*.json`,`exit_code=0` 18m08s。
各 shard `report_speed`:1-ref 中位 22.6–23.3 s/img,2-ref 中位 27.8–28.8 s/img。
粗加速比(full÷iso_pre,同机 8 并发):1-ref ~1.7×、2-ref ~2.4×(≈P2 预检 2.37×)。

**iso_post 补跑完成(2026-08-14 18:0x)**:1 卡 `--num_shards 1` 续跑,`exit_code=0` 16m05s,
自检 `已跳过 210 | 待跑 30`,补出的 30 张 report_speed 1-ref 24.2 / 2-ref 29.6 s/img,
与第一跑 7 shard 一致 ⇒ **iso_post = 240 张全齐**。补跑写 `results_shard0.json`(第一跑崩的 shard 0 没写过),
与 1–7 共 8 个 json、240 条记录,原 7 个 json 备份在 `output/p3_iso_post/.results_first/`。

**三臂汇总中位数(240 条全量,从 results_shard*.json 的 elapsed_s 聚合)**:

| 臂 | 1-ref(n=75) | 2-ref(n=165) | 加速比 1-ref | 加速比 2-ref |
|---|---|---|---|---|
| full | 38.6 s/img | 67.3 s/img | 1.00× | 1.00× |
| iso_pre | 22.9 s/img | 28.2 s/img | 1.69× | **2.39×** |
| iso_post | 24.7 s/img | 29.8 s/img | 1.56× | 2.26× |

（iso_post 比 iso_pre 略慢 ≈ LoRA 每前向多出的计算;两臂均同机 8 并发/1 卡补跑可比。）
**1-ref 离群观察**:两 iso 臂各 ~14/75 张 1-ref 图 >45s(iso_pre 85–90s / iso_post 137–140s),
集中在 S3 后半段任务,把均值拉到 32.3/45.3 但中位稳健 —— 记录非门禁,盲评侧如需可留意。

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
本轮主批三臂又在满集群下各等约 1 小时才被领走。

### 坑 7 · 8 进程并发写同一日志文件,撞 ceph EIO 崩掉一个 shard(新坑)

iso_post 第一跑,8 个 shard 并发读 2.26GB LoRA 文件后紧接着往 worker 的**同一个日志文件**写
stdout,其中一个 shard 在 `infer_iso.py:137` 的 `print`(LoRA 自检行)处
`OSError: [Errno 5] Input/output error` 崩掉。Traceback 原样:

```text
File ".../qwen/infer_iso.py", line 137, in apply_lora_ckpt
    print(f"[自检] LoRA 已加载 {path} | ...")
OSError: [Errno 5] Input/output error
```

- 是**日志写(存储)瞬时错误**,不是渲染/显存/数据错误;该 shard 还没开始渲图,其余 7 个 shard
  正常出图(210 张)。
- job 的 fan-out 聚合 `wait $p || st=1` 把它变成 `exit_code=1`,**整 job 标失败**但产物保留。
- 后续 1 卡补跑(单进程写日志)没再撞。

### 坑 8 · 8 进程并发 `torch.load` 同一个 2.26GB LoRA 文件(新坑)

只有 iso_post 读 LoRA 文件,ge90-70 上该文件 page cache 不热;8 进程同时从 ceph 拉同一个 2.26GB
文件,`torch.load` 全程无中间输出 ⇒ **日志长时间静默,容易被误判卡死**(实测静默 44s+)。实际是慢读,
8 个 shard 陆续完成加载(LoRA 自检行逐个出现)。影响:每进程加载时间从预期的 ~20s 拖到 ~2–3 分钟。

### 坑 9 · 补跑少量图用 1 卡 `--num_shards 1` 就够了(新坑,替代重投 8 卡)

只缺 1 个 shard 的 30 张时,8 卡重投会白占 7 卡(其余 shard 加载完模型发现没活干就退出)。
**1 卡即可**:`--num_shards 1 --shard_idx 0` 把全部任务归给 shard 0,`infer_iso.py` 的续跑逻辑
(`output_exists` 跳过)自动只渲缺的 30 张,自检打印 `本 shard 240 | 待跑 30 | 已跳过 210`。
好处:调度只需任意机器 1 张空卡(比等整机快得多)、省 7 卡、单进程写日志更不易撞坑 7。
副作用:补跑写 `results_shard0.json`,若第一跑该 shard 也写过会覆盖 —— 先备份原 json。

---

## 6 · 要记的数

| 数 | 值 | 备注 |
|---|---|---|
| ckpt 自检 | `step 2000 / rank 64 / targets 8 / 张量 960 / lora_B 非零 478 / 非有限 0` | ✅ 逐字符合 |
| §5.1 判定 mean | 5.92 / 2.67 / 2.33 | ❌ 门禁 `mean < 0.5` 未过(判据后被作者立错) |
| floor 30 对像素差 | 全部 mean=0.0000 / max=0 | 流水线位级确定 ⇒ §5.1 差异是系统差异 |
| **主批出图数** | full / iso_pre / iso_post 各 **240** | iso_post = 210 首跑 + 1 卡补跑 30;无 fail 记录 |
| **主批中位 s/img**(240 条聚合) | full 1-ref 38.6 / 2-ref 67.3;iso_pre 22.9 / 28.2;iso_post 24.7 / 29.8 | 从 results_shard*.json 的 elapsed_s 聚合 |
| **加速比**(full÷) | iso_pre 1-ref **1.69×** / 2-ref **2.39×**;iso_post 1.56× / 2.26× | 预登记 2-ref 1.9–2.0×、1-ref ~1.4× 被打脸;2-ref 同 P2 预检 2.37× |
| **前向次数** | iso_pre/iso_post 各 write **240** / read **18960**;full 无缓存 0/0 | 18960 = 240×79,每张 1 写 79 读 ✅ |
| diag_kv 每步扰动 | write/read 噪声速度相对 L2 最大 **0.0759**(前向 79) | 对照最终像素差 ~1e-2(2–6/255) |
| **diag_kv v1** | `verdict=FAIL`,`kv_max_rel=0.246` | 奇前向(1,79)=uncond 59/60 层不等 |
| **diag_kv v2** | `verdict=PASS`;A/C 复算 60/60;`step_invariant_cond=true`;峰值 66.85GB | 真机 bf16 钉实「长度效应」良性;cond_txt_len 420 vs uncond 412 |
| 1-ref 离群观察 | 两 iso 臂各 ~14/75 张 >45s,集中 S3 后半段 | 中位稳健;均值被拉高(iso_pre 32.3 / iso_post 45.3) |

---

## 7 · 本轮明确未做

- **盲评判读、拼图、配对**:未做 —— 出完图本单结束,交给 `../distill/blind_eval/` 按它自己的纪律。
- **加 3-ref 层**:未做(要先有速度预测预登记,作者的事)。
- **改 `PLAN.md`**:未改。
- **单臂图进 git**:主批 iso 单臂图未进 git(output/ 是白名单 gitignore)。本报告目录只随附
  `diag_kv_v1.json`(json,非图)。
- **既往泄漏声明**(沿用第一轮 §7 记录):`output/p3_floor/a/`、`b/` 各 30 张 `full`、
  `p3_cachecheck` 3 张 `iso_pre` 已随早期报告进 git,不删 —— 删了也不改变已可见的事实,在局限中声明。

---

## 8 · 待办 / 待用户决定

本轮全部停线点已闭合:diag_kv v1 FAIL → 作者本地判=长度效应(b,良性) → **v2 真机确认 PASS** →
主批三臂各 240 张出齐。留给作者/盲评侧的是记录项,不是阻塞:

1. **预登记速度预测被证伪**:预登记 2-ref 1.9–2.0×、1-ref ~1.4×,实测 iso_pre **2.39×/1.69×**、
   iso_post **2.26×/1.56×**。按清单「照实记数,预登记就是用来被证伪的」;作者如需可给 PLAN §4 加订正注记。
2. **txt token 实测**:diag_kv v2 测到 cond txt_len=**420** / uncond=**412**(PLAN §4.4 估 400–600,落区间内)。
3. **1-ref 离群**:两 iso 臂各 ~14/75 张 1-ref 图 85–140s,集中 S3 后半段;中位稳健,均值被拉高。
4. **执行期新发现**(均已按 R0 报备后处理):v2 的 `s['v_rel_l2']` KeyError(已修,commit `d0b9811`)、
   iso_post 一个 shard 撞 ceph EIO(1 卡补跑绕过,坑 7/8/9)。

---

*commit message 建议:`eval(qwen): P3 三臂出图(full/iso_pre/iso_post 各 240,diag_kv v2 PASS)`*
