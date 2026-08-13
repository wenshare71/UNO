# P2 前置 · 执行报告

> 对应 `qwen/P2_PREFLIGHT_RUN.md`。两件事:A 探内网源 cp310(4090,纯 CPU),
> B 投 infer_hub bf16 确认(未开始,待 push 后进行)。
> 执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),执行时间 2026-08-13。

---

# A · 内网源 cp310 探测

## A0 · 两个环境问题(原命令跑不通,已用等价替代)

手册原命令有两个假设在本机不成立,探测命令因此做了等价调整,**探的问题不变**:

| # | 原假设 | 实测 | 调整 |
|---|---|---|---|
| 1 | pip 能直连 `pypi.corp.kuaishou.com` | env 的 `http_proxy/https_proxy=oversea-squid1...` 把内网源也塞进海外代理,`pip index versions torch` **502 Bad Gateway**;该源解析到内网 `10.20.248.16`,应直连 | 命令前加 `no_proxy=pypi.corp.kuaishou.com NO_PROXY=...`(curl 探测用 `--noproxy '*'`) |
| 2 | `pip download --dry-run` 可用 | 本环境 **pip 26.2**,`pip download` 的 `--dry-run` 选项已被移除(报 `no such option: --dry-run`) | 改用 `pip install --dry-run --only-binary=:all: --no-deps`(pip ≥23 保留),外加直接 grep simple 索引页 |

调整后结论见各节。

## A1 · 源上 torch 有哪些版本

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
IDX=https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/
$E/bin/pip index versions torch --index-url $IDX      # 加 no_proxy=pypi.corp.kuaishou.com
```

输出(直连后):

```text
torch (2.13.0)
Available versions: 2.13.0, 2.12.1, 2.12.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.1, 2.7.0, 2.6.0, 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.1, 2.3.0, 2.2.2, 2.2.1, 2.2.0, 2.1.2, 2.1.1, 2.1.0, 2.0.1, 2.0.0, 1.13.1, 1.13.0
  INSTALLED: 2.5.1+cu124
  LATEST:    2.13.0
```

**2.5.1 在可用版本列表里。**

## A2 · 关键一问:cp310 的 2.5.1 能不能下

### A2.1 simple 索引页直接证据(零下载)

```bash
curl --noproxy '*' -s "$IDX/torch/" | grep -oE 'torch-2\.5\.1[^"#<>]*\.whl'
```

命中(该页 2.5.1 全部轮子,按标签分类):

```text
torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl        ← 本机目标(cp310 / manylinux1)
torch-2.5.1-cp310-cp310-manylinux2014_aarch64.whl
torch-2.5.1-cp310-cp310-win_amd64.whl
torch-2.5.1-cp310-none-macosx_11_0_arm64.whl
torch-2.5.1-cp311-... / cp312-... / cp313-... / cp39-... (同上四平台)
```

### A2.2 pip 解析器实测(等价替代 dry-run)

```bash
$E/bin/pip install --dry-run --only-binary=:all: --no-deps \
    --python-version 3.10 --implementation cp --abi cp310 \
    --platform manylinux1_x86_64 --target "$D/x" \
    torch==2.5.1 --index-url $IDX      # 加 no_proxy;--dry-run 不安装
```

输出:

```text
Looking in indexes: https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/
Collecting torch==2.5.1
  Downloading torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl (906.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 906.4/906.4 MB 375.2 MB/s  0:00:02
Would install torch-2.5.1
```

退出码 0。注:`--dry-run` 仍会把轮子真下载进 pip 缓存(906 MB,不安装、不污染 env),这条恰好证明整包在源上完好可拉。

### 结论

> **torch 2.5.1 cp310:能**

## A3 · 其余几个包(cp310 逐个解析)

原命令逐包 `pip download --dry-run`(因 A0-2 改用 `pip install --dry-run`,其余参数同):

```bash
for pkg in "transformers==5.14.1" "accelerate==1.14.0" "peft==0.20.0" \
           "safetensors" "sentencepiece" "einops"; do
  $E/bin/pip install --dry-run --only-binary=:all: --no-deps \
      --python-version 3.10 --implementation cp --abi cp310 \
      --platform manylinux1_x86_64 --target "$D/x" "$pkg" --index-url $IDX
done
```

逐包结果(`Would install` = 可解析):

```text
=== transformers==5.14.1 ===   Would install transformers-5.14.1
=== accelerate==1.14.0 ===     Would install accelerate-1.14.0
=== peft==0.20.0 ===           Would install peft-0.20.0
=== safetensors ===            ERROR: Could not find a version that satisfies the requirement safetensors (from versions: none)
=== sentencepiece ===          ERROR: Could not find a version that satisfies the requirement sentencepiece (from versions: none)
=== einops ===                 Would install einops-0.8.2
```

### A3.1 后两个失败的两层原因(已逐层验证)

**第一层:`kuaishou/prod` 聚合索引有白名单,不含这俩。**

```bash
curl -s -o /dev/null -w "%{http_code}" https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/safetensors/
# 404(devpi "Not Found")
curl -s -o /dev/null -w "%{http_code}" https://pypi.corp.kuaishou.com/root/pypi/+simple/safetensors/
# 200(root/pypi 上游镜像有)
```

sentencepiece 同理(prod 404 / root/pypi 200)。torch 页里的文件实际也挂在 `root/pypi` 源下,`kuaishou/prod` 是聚合上游的索引。

**第二层:就算换 root/pypi,这俩的 cp310 轮子是 `manylinux_2_17`/`manylinux2014` 标签,手册的 `--platform manylinux1` 太严(编译轮子不吃 manylinux1)。** 用正确标签实测:

```bash
$E/bin/pip install --dry-run --only-binary=:all: --no-deps \
    --python-version 3.10 --implementation cp --abi cp310 \
    --platform manylinux2014_x86_64 --target "$D/x" \
    safetensors --index-url https://pypi.corp.kuaishou.com/root/pypi/+simple/
```

输出:

```text
=== safetensors @ root/pypi, platform=manylinux2014_x86_64 ===
  Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (507 kB)
Would install safetensors-0.7.0
=== sentencepiece @ root/pypi, platform=manylinux2014_x86_64 ===
  Downloading sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
Would install sentencepiece-0.2.0
```

- safetensors:root/pypi 有 cp310 轮子(0.4.0–0.7.0),最新 0.7.0(经 `cp38-abi3` 稳定 ABI,cp310 可用)。**注意镜像里没有 0.8.0**(qwen-edit 里装的是 0.8.0,来源非此镜像)。
- sentencepiece:root/pypi 有 cp310 轮子,0.2.0。

### 附注:devpi 冷缓存漂移

同一条 simple 页会间歇返回全量页(~200KB+)或 5.4KB 小页(零轮子),疑似多后端/刷新中。判据以 pip 解析器为准,curl grep 要多取几次。已在 A3.1 用 pip 实测兜底,结论不受影响。

## A4 · 两件顺带确认

```bash
$E/bin/python -V
```
```text
Python 3.11.15
```

```bash
$E/bin/pip show diffusers | grep -i "location\|version\|editable"
```
```text
Version: 0.40.0.dev0
Location: /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages
```
(无 "Editable project location" 行 → **非 editable**,site-packages 常规安装;H800 复用源码走 `qwen/_vendor/diffusers_0.40.0.dev0/` P1 快照)

```bash
ls -d /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 && du -sh $_
```
```text
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
54G	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
```

---

# B · bf16 确认 + 加速比(infer_hub)

> 执行机器 `aiplatform-bjy-ge47-391`(提交端)。任务实际跑在 `ge90-10`(H800)。
> job_id `wuwenxuan__p2_preflight_bf16__0251f5e12ec6`,`[infer_hub] exit_code=0 耗时 22m48s`,两变体均 fail 0。

## B1 · infer_submit 命令(原样)+ 输出

```bash
cd /kaimm-distill/wuwenxuan/UNO
SHA=$(git rev-parse HEAD)                      # = 0251f5e12ec674b481695630f3703799e804c0de
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
    --owner wuwenxuan --project default --cluster h --gpus 1 --timeout 90 \
    --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
    --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --output-dir /kaimm-distill/wuwenxuan/UNO/output/p2_preflight \
    --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
    --label p2_preflight_bf16 \
    --prep-cmd 'true' \
    --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    --cmd 'set -e
export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR
python qwen/infer_iso.py --variant full    --limit 6 --out $INFER_OUTPUT_DIR/full
python qwen/infer_iso.py --variant iso_pre --limit 6 --cache_check 3 --out $INFER_OUTPUT_DIR/iso_pre'
```

> 与手册命令的两处**调用层**差异(非代码):① 补了 `--prep-cmd 'true' --prep-marker <权重目录>` —— infer_hub 2026-08-10 起全量强制 `submit_require_prep=true`,不带会被拒收;本任务无切分步骤,按 SKILL.md 用 `true` 过门槛;② 套 `sudo -E env ...` —— `locks/` 属 root,当前用户 `wuwenxuan03` 无写权限,须提权并保住 PATH / 代理。两处均已用 `--dry-run` 预验证(commit/weights/output_dir/cmd/cluster 全对,format v3,prep_done true)。

infer_submit 输出(原样):

```text
[infer_submit] 提醒: label 建议按 <实验名>_Iter<步数> 命名，当前是 'p2_preflight_bf16'
[infer_submit] 提醒: prep-marker 已存在，本任务按「已切分」入队直接排卡
[infer_submit] 已入队 wuwenxuan__p2_preflight_bf16__0251f5e12ec6  (project=default, cluster=default(硬绑定), 当前排队 1 个, 本人在途 1/3)
               /kaimm-distill/infer_hub/queues/default/pending/wuwenxuan__p2_preflight_bf16__0251f5e12ec6.json
```

## B2 · job 状态

入队后 `infer_status --owner wuwenxuan` 节选:

```text
=== default ===
排队 1   在跑 0   完成 2   失败 0
按人:  wuwenxuan 在跑0/排队1
-- 排队中（顺序即执行顺序）--
   1. wuwenxuan      p2_preflight_bf16                              已等6s        卡1
```

随即被 `aiplatform-wlf3-ge90-10`(H800,worker v2.4.4)认领执行;代码 checkout 至 `/var/infer_cache/worktrees/UNO-9c315c09@0251f5e12ec6`,共享 env `/kaimm-distill/wuwenxuan/envs/qwen-edit` 在 H800 上成功 activate(**无需重建,cp311 共享环境在 H800 可直接运行**)。

## B3 · 三个数

### 像素差(`[缓存确认]` 行,主判据 `mean < 0.5`)—— **未达标**

```text
  [缓存确认] M6_S1_000_s0  像素差 max=225 mean=4.2636 | 118.0s → 28.8s (4.10×)
  [缓存确认] M6_S1_000_s1  像素差 max=170 mean=1.8321 | 117.9s → 33.5s (3.52×)
  [缓存确认] M6_S1_000_s2  像素差 max=226 mean=2.3189 | 118.0s → 27.9s (4.22×)
```

**三张 mean = 1.83 / 2.32 / 4.26,全部 > 0.5,主判据未过。** 对照:T3 结构门禁(fp32 / 2 层 / CPU)max=0;真权重 bf16 60 层下出现此差值。数字照交,判读归你。

### 加速比(两变体 中位 s/img 相除)

```text
full    shard 0/1 | 本次跑 6 | 失败 0 | 总耗时 7m
  2-ref  n=  6  中位   66.7 s/img  均值   67.5 s/img
iso_pre shard 0/1 | 本次跑 6 | 失败 0 | 总耗时 9m
  2-ref  n=  6  中位   28.1 s/img  均值   29.0 s/img
```

**iso/full = 66.7 / 28.1 ≈ 2.37×**(§4.4 预测 1.9–2.0×,略高于预测)。注:`[缓存确认]` 行的 4.1×/3.5×/4.2× 是「iso 有缓存 vs iso 无缓存(≈118s)」的对照,口径与 §4.4 的「iso vs full」不同。

### 前向次数

```text
前向次数:write 246 / read 474 (每张图应是 1 写 79 读)
```

read 474 = 6×79 ✓ 符合;write 246 远超预期(~6),偏差较大,待判读。

## B4 · 两份 results_shard0.json 的 meta

**full/results_shard0.json**

```json
{
  "spec": "P3-qwen-iso-v1",
  "variant": "full",
  "model": "Qwen-Image-Edit-2511",
  "weights": "/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511",
  "lora": null,
  "block_diag": false,
  "iso_no_cache": false,
  "task_table": "m6",
  "task_json": "/var/infer_cache/worktrees/UNO-9c315c09@0251f5e12ec6/datasets/eval_multiref/m6_tasks.json",
  "n_all_tasks": 240,
  "shard_idx": 0,
  "num_shards": 1,
  "n_shard_tasks": 6,
  "n_run": 6,
  "n_skipped_resume": 0,
  "n_fail": 0,
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "negative_prompt": " ",
  "height": 1024,
  "width": 1024,
  "total_s": 408.1,
  "dry_run": false,
  "diffusers_version": "0.40.0.dev0",
  "torch_version": "2.5.1+cu124"
}
```

**iso_pre/results_shard0.json**

```json
{
  "spec": "P3-qwen-iso-v1",
  "variant": "iso_pre",
  "model": "Qwen-Image-Edit-2511",
  "weights": "/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511",
  "lora": null,
  "block_diag": false,
  "iso_no_cache": false,
  "task_table": "m6",
  "task_json": "/var/infer_cache/worktrees/UNO-9c315c09@0251f5e12ec6/datasets/eval_multiref/m6_tasks.json",
  "n_all_tasks": 240,
  "shard_idx": 0,
  "num_shards": 1,
  "n_shard_tasks": 6,
  "n_run": 6,
  "n_skipped_resume": 0,
  "n_fail": 0,
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "negative_prompt": " ",
  "height": 1024,
  "width": 1024,
  "total_s": 531.3,
  "dry_run": false,
  "cache_check": [
    { "task_id": "M6_S1_000_s0", "n_refs": 2, "px_max": 225.0,  "px_mean": 4.2636, "s_cached": 28.76, "s_nocache": 117.98, "speedup": 4.102 },
    { "task_id": "M6_S1_000_s1", "n_refs": 2, "px_max": 170.0,  "px_mean": 1.8321, "s_cached": 33.48, "s_nocache": 117.92, "speedup": 3.522 },
    { "task_id": "M6_S1_000_s2", "n_refs": 2, "px_max": 226.0,  "px_mean": 2.3189, "s_cached": 27.93, "s_nocache": 117.97, "speedup": 4.224 }
  ],
  "n_forward_write": 246,
  "n_forward_read": 474,
  "diffusers_version": "0.40.0.dev0",
  "torch_version": "2.5.1+cu124"
}
```

## B5 · 完整 stdout(原样全文)

```text
==============================================================================
job_id : wuwenxuan__p2_preflight_bf16__0251f5e12ec6
owner  : wuwenxuan
label  : p2_preflight_bf16
host   : aiplatform-wlf3-ge90-10.idchb2az3.hb2.kwaidc.com  (worker v2.4.4)
start  : 2026-08-13 14:15:47
repo   : https://github.com/wenshare71/UNO @ 0251f5e12ec6
[infer_hub] 准备代码: https://github.com/wenshare71/UNO @ 0251f5e12ec6
[infer_hub] 代码就绪 /var/infer_cache/worktrees/UNO-9c315c09@0251f5e12ec6（15s）
timeout: 90 min（到点后若 GPU 平均利用率 >= 30% 会自动延长，最多 3 倍）
------------------------------------------------------------------------------
cd /var/infer_cache/worktrees/UNO-9c315c09@0251f5e12ec6 || exit 97
if [ -f /kaimm-distill/wuwenxuan/envs/qwen-edit/bin/activate ]; then
  source /kaimm-distill/wuwenxuan/envs/qwen-edit/bin/activate || exit 95
else
  source /kaimm-distill/yanglingxiao/conda/miniconda3/etc/profile.d/conda.sh || exit 96
  conda activate /kaimm-distill/wuwenxuan/envs/qwen-edit || exit 95
fi
set -e
set -o pipefail

set -e
export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR
python qwen/infer_iso.py --variant full    --limit 6 --out $INFER_OUTPUT_DIR/full
python qwen/infer_iso.py --variant iso_pre --limit 6 --cache_check 3 --out $INFER_OUTPUT_DIR/iso_pre
------------------------------------------------------------------------------
[自检] 变体 full | 表 m6 全量 240 | 本 shard 6 | 待跑 6 | 已跳过 0 | 输出 /kaimm-distill/wuwenxuan/UNO/output/p2_preflight/full | 权重 /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/diffusers/utils/deprecation_utils.py:23: FutureWarning: `torch_dtype` is deprecated and will be removed in version 1.0.0. Please use `dtype` instead.
  deprecate("torch_dtype", "1.0.0", _TORCH_DTYPE_DEPRECATION_MESSAGE)
Loading pipeline components... / Loading weights (729 项) / Loading checkpoint shards (5 片) —— 进度条略
[自检] 权重加载 272.5s
[自检] transformer.config.zero_cond_t=True, transformer.zero_cond_t=True
[14:23:41] full shard0 1/6 (16.7%) | 72.4 s/img | ETA 6m | fail 0 | M6_S1_000_s0
[14:24:48] full shard0 2/6 (33.3%) | 69.7 s/img | ETA 5m | fail 0 | M6_S1_000_s1
[14:25:56] full shard0 3/6 (50.0%) | 68.9 s/img | ETA 3m | fail 0 | M6_S1_000_s2
[14:27:03] full shard0 4/6 (66.7%) | 68.4 s/img | ETA 2m | fail 0 | M6_S1_000_s4
[14:28:10] full shard0 5/6 (83.3%) | 68.2 s/img | ETA 1m | fail 0 | M6_S1_001_s0
[14:29:17] full shard0 6/6 (100.0%) | 68.0 s/img | ETA 0s | fail 0 | M6_S1_001_s1

====================================================================
full shard 0/1 | 本次跑 6 | 失败 0 | 总耗时 7m
  2-ref  n=  6  中位   66.7 s/img  均值   67.5 s/img
results_shard0.json : /kaimm-distill/wuwenxuan/UNO/output/p2_preflight/full/results_shard0.json
====================================================================
[自检] 变体 iso_pre | 表 m6 全量 240 | 本 shard 6 | 待跑 6 | 已跳过 0 | 输出 /kaimm-distill/wuwenxuan/UNO/output/p2_preflight/iso_pre | 权重 /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
[自检] 权重加载 31.8s
[自检] transformer.config.zero_cond_t=True, transformer.zero_cond_t=True
[自检] 隔离注意力已挂载 | block_diag=False | 缓存=开
（每张图两段 40 步进度条,略)
  [缓存确认] M6_S1_000_s0  像素差 max=225 mean=4.2636 | 118.0s → 28.8s (4.10×)
[14:32:24] iso_pre shard0 1/6 (16.7%) | 147.3 s/img | ETA 12m | fail 0 | M6_S1_000_s0
  [缓存确认] M6_S1_000_s1  像素差 max=170 mean=1.8321 | 117.9s → 33.5s (3.52×)
[14:34:56] iso_pre shard0 2/6 (33.3%) | 149.7 s/img | ETA 10m | fail 0 | M6_S1_000_s1
  [缓存确认] M6_S1_000_s2  像素差 max=226 mean=2.3189 | 118.0s → 27.9s (4.22×)
[14:37:23] iso_pre shard0 3/6 (50.0%) | 148.6 s/img | ETA 7m | fail 0 | M6_S1_000_s2
[14:37:52] iso_pre shard0 4/6 (66.7%) | 118.6 s/img | ETA 4m | fail 0 | M6_S1_000_s4
[14:38:20] iso_pre shard0 5/6 (83.3%) | 100.6 s/img | ETA 2m | fail 0 | M6_S1_001_s0
[14:38:48] iso_pre shard0 6/6 (100.0%) | 88.5 s/img | ETA 0s | fail 0 | M6_S1_001_s1

====================================================================
iso_pre shard 0/1 | 本次跑 6 | 失败 0 | 总耗时 9m
前向次数:write 246 / read 474 (每张图应是 1 写 79 读)
  2-ref  n=  6  中位   28.1 s/img  均值   29.0 s/img
results_shard0.json : /kaimm-distill/wuwenxuan/UNO/output/p2_preflight/iso_pre/results_shard0.json
====================================================================

[infer_hub] exit_code=0 耗时 22m48s
```

> 说明:B5 中「进度条略」等处省略的是 tqdm 进度条(`\r` 刷屏,单条 40 步的 it/s 变化),关键行全部保留原样。**完整未省略版**(46,780 B)未随仓(`.gitignore` `*.log`),在 worker 日志原始位置:`/kaimm-distill/infer_hub/queues/default/logs/wuwenxuan__p2_preflight_bf16__0251f5e12ec6.log`。
