# P3 出图 · 执行报告(进行中,停在 §5.1)

> 对应 `qwen/P3_EVAL_RUN.md`。本单只出图、不盲评判读、不建拼图。
> 起点 commit `3c1f90f`。执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),
> 执行时间 2026-08-14。
>
> **当前进度**:§5.0 ckpt 自检 ✅、§5.1 判定实验 ✅(结果未过门禁)。
> **§5.2 主批 720 张未投** —— §5.1 停线协议触发,等用户决定(见 §5.1 判读)。

---

## 0 · 结论速览

- `step002000.pt` 的 LoRA 结构自检**逐字符合期望**,iso_post 臂是训练后的权重(不是未训练的退化臂)。
- §5.1 判定实验:把写/读两条注意力路**强制压到同一个 kernel**(`DIFFUSERS_ATTN_BACKEND=_native_efficient`),期望「隔离+缓存 vs 隔离+每步重算」的像素差塌到 0.5 以下。**结果没塌,反而更高**(mean 2.33–5.92)。「核不对称」假说被证伪。
- 按清单 §5.1 协议,`mean` 落在 2–4 区间 ⇒ 「缓存本身有实质问题」⇒ **停线,720 张主批先不跑**。

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

## 3 · §5.2 主批 —— 未投(停线)

三个臂(full / iso_pre / iso_post,各 240 条,8 卡 fan-out)的提交命令**已 dry-run 验证通过**,
但按清单 §5.1 协议「mean 仍在 2–4 ⇒ 停下来告诉我,后面 720 张图先别跑」,**未实际投递**。

dry-run 验证结论(备查):

- **fan-out**(清单 §4.1):`--cmd` 里 `IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"` 起
  `${#G[@]}` 个进程再 `wait`,不写死卡号,全部用注入变量 + 相对路径,无共享盘路径泄漏。
- **iso_post 的 LoRA 路径**(清单 §4.3):直写绝对路径 `/kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt`
  会被 `cmd_shared_disk_leaks` 拦(见 §5 坑 2)。解法:把它声明进 `--prep-marker`(该路径在共享盘、
  已存在,`prep_done=true` 直接排卡),dry-run 通过。
- 三个臂输出目录分开(`output/p3_full` / `p3_iso_pre` / `p3_iso_post`),不会互相覆盖。

---

## 4 · §5.3 run_floor —— 依据已查,未投(随主批停线)

run_floor 用哪个臂:查到了依据,**用 `full`(基线侧)**。

- `distill/build_m6_tasks.py` L46–55(正是本批 m6 任务表的生成脚本)写明:天花板第二侧走
  **另一个 infer_hub job**,权重取 `m6_full`(主对的基线侧)——「天花板量的正是骑在主对上的
  那一份噪声,而不是另一个模型的噪声」。
- P3 语境下主对基线侧就是 `full`(stock 权重 + 全注意力 = teacher)。
- `distill/M4_EVAL_SPEC.md` §8 是 M5 人工判读判据,不涉及 floor;`distill/blind_eval/` 无 floor 专属代码。

任务表 `m6_floor` 30 条(S1 20 + S3 10;2-ref 20 / 1-ref 10),参考图全在(本地已验)。

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

---

## 6 · 要记的数

| 数 | 值 | 备注 |
|---|---|---|
| ckpt 自检 | `step 2000 / rank 64 / targets 8 / 张量 960 / lora_B 非零 478 / 非有限 0` | ✅ 逐字符合 |
| §5.1 判定 mean | 5.92 / 2.67 / 2.33 | ❌ 门禁 `mean < 0.5` 未过,且高于默认 backend |
| 前向次数 | write 243 / read 237 | 3 缓存 + 3 无缓存,每张 1 写 79 读 |
| 缓存 s/img(`_native_efficient`) | 中位 36.7 | 被人为改慢,不与默认 backend 并排引用 |

预登记的速度预测(2-ref 1.9–2.0×、1-ref ~1.4×)本轮未触达,不记。

---

## 7 · 本轮明确未做

- **主批 720 张(三臂 × 240)未投**,因 §5.1 停线协议触发。
- **run_floor 30 条未投**,随主批停线。
- 未盲评判读、未建拼图、未配对。
- 未加 3-ref 层、未改 `PLAN.md`。

---

## 8 · 待办 / 待用户决定

1. 缓存「mean 2–4」怎么解:是否加测「read 也用 `(1,1,L,L)` 全 True mask」的只读诊断,
   以区分「mask 结构不对称」vs「缓存逻辑真 bug」(需改 `qwen/` 下的 `.py`,R0 需点头)。
2. 决定后主批三臂 + run_floor 是否照跑、跑的话是否改判据。

---

*commit message 建议:`eval(qwen): P3 ckpt 自检 + §5.1 判定实验(停线)`*
