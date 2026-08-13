# Q1-B 3-ref 能力探底 · 执行报告

> 本报告对应 `qwen/Q1B_3REF_RUN.md` §2 P0 交付与 §1 必查门禁。
> 只记录数字、日志与现象，不做“3-ref 行不行”的判读。

---

## 1. §1 必查：`zero_cond_t`

执行命令与原始输出：

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
$E/bin/python - <<'PY'
import inspect, diffusers
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as M
sig = inspect.signature(M.__init__).parameters
print("diffusers:", diffusers.__version__)
print("zero_cond_t in __init__:", "zero_cond_t" in sig)
import json, os
cfg = json.load(open(os.path.join(os.environ["QWEN_WEIGHTS"], "transformer/config.json")))
print("config zero_cond_t:", cfg.get("zero_cond_t"))
PY
```

输出：

```text
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
diffusers: 0.40.0.dev0
zero_cond_t in __init__: True
config zero_cond_t: True
```

结论：两个条件均为 `True`，G1 通过，继续执行。

---

## 2. Q1 产物回仓

### 2.1 来源

共享盘目录：`/kaimm-distill/wuwenxuan/output/qwen_baseline/`

已取回仓库：

- `output/qwen_baseline/results.json`
- `output/qwen_baseline/ALL_COMPARISON_part01.png`
- `output/qwen_baseline/ALL_COMPARISON_part02.png`
- `output/qwen_baseline/ALL_COMPARISON_part03.png`
- `output/qwen_baseline/ALL_COMPARISON_part04.png`
- `output/qwen_baseline/ALL_COMPARISON_part05.png`

全分辨率单图（40 张 1024² PNG，约 60 MB）仍保留在共享盘，未入 git。

### 2.2 `results.json` 的 `meta` 原样

```json
{
  "meta": {
    "spec": "Q1-qwen-baseline-v1",
    "model": "Qwen-Image-Edit-2511",
    "weights": "/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511",
    "task_json": "/var/infer_cache/worktrees/UNO-9c315c09@4a9a034521e0/datasets/eval_multiref/m6_tasks.json",
    "n_all_tasks": 320,
    "subset_stride": 8,
    "n_subset": 40,
    "n_run": 40,
    "n_skipped_resume": 0,
    "n_fail": 0,
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "height": 1024,
    "width": 1024,
    "total_s": 2412.0,
    "dry_run": false,
    "offload": false,
    "diffusers_version": "0.40.0.dev0",
    "torch_version": "2.5.1+cu124"
  }
}
```

口径说明：Q1 在 `m6_tasks.json` 共 320 条任务上，按 `i % 8 == 0` 取子集，实际跑 40 条，全部成功。

### 2.3 40 条 task_id 列表（`i % 8 == 0`）

```python
['M6_S1_000_s0', 'M6_S1_001_s3', 'M6_S1_003_s1', 'M6_S1_004_s4',
 'M6_S1_006_s2', 'M6_S1_008_s0', 'M6_S1_009_s3', 'M6_S1_011_s1',
 'M6_S1_012_s4', 'M6_S1_014_s2', 'M6_S1_016_s0', 'M6_S1_017_s3',
 'M6_S1_019_s1', 'M6_S1_020_s4', 'M6_S1_022_s2', 'M6_S1_024_s0',
 'M6_S1_025_s3', 'M6_S1_027_s1', 'M6_S1_028_s4', 'M6_S1_030_s2',
 'M6_S1_032_s0', 'M6_S1_033_s3', 'M6_S1_035_s1', 'M6_S1_036_s4',
 'M6_S1_038_s2', 'M6_S1_040_s0', 'M6_S1_041_s3', 'M6_S1_043_s1',
 'M6_S3_00c2_s0', 'M6_S3_01c1_s0', 'M6_S3_02c0_s0', 'M6_S3_02c4_s0',
 'M6_S3_03c3_s0', 'M6_S3_04c2_s0', 'M6_S3_05c1_s0', 'M6_S3_06c0_s0',
 'M6_S3_06c4_s0', 'M6_S3_07c3_s0', 'M6_S3_08c2_s0', 'M6_S3_09c1_s0']
```

### 2.4 实测汇总

| 指标 | 值 |
|---|---|
| `total_s` | 2412.0 |
| `n_run` | 40 |
| `s/img` | 60.3 |
| `peak_mem_gb` 范围 | 57.97 – 57.98 |
| `n_fail` | 0 |

---

---

## 3. G2：`qwen/infer_qwen_3ref.py` `--dry_run` 8 分片

脚本路径：`qwen/infer_qwen_3ref.py`（新写，结构照 `scripts/infer_qwen_edit.py`，常量与 Q1 逐字相同，只加了 8 卡分片）。

本地完整 `--dry_run`（不加 `--limit`）结果：

| shard_idx | 条数 | 第一个 task_id | 最后一个 task_id |
|---|---|---|---|
| 0 | 16 | `S4_000_s0` | `Q1B_109_s0` |
| 1 | 16 | `S4_000_s1` | `Q1B_110_s0` |
| 2 | 15 | `S4_001_s0` | `Q1B_101_s0` |
| 3 | 15 | `S4_001_s1` | `Q1B_103_s0` |
| 4 | 15 | `S4_002_s0` | `Q1B_104_s0` |
| 5 | 15 | `S4_002_s1` | `Q1B_105_s0` |
| 6 | 15 | `S4_003_s0` | `Q1B_106_s0` |
| 7 | 15 | `S4_003_s1` | `Q1B_107_s0` |

合并后 `results.json` 总任务 **122**，拼图按 8 行分批输出 16 张 `ALL_COMPARISON_partNN.png`。

G2 通过。

---

## 4. G3：正式 job 已提交

提交命令（经 `--dry-run` 验证后正式执行）：

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
sudo -E env PATH=/kaimm-distill/infer_hub/lib:$PATH \
  http_proxy=http://oversea-squid1.jp.txyun:11080 \
  https_proxy=http://oversea-squid1.jp.txyun:11080 \
  /kaimm-distill/infer_hub/lib/infer_submit \
  --owner wuwenxuan --project default --cluster h --gpus 8 --timeout 60 \
  --repo https://github.com/wenshare71/UNO.git \
  --commit 16c7a672b14a385dae3af1bdd912e3abcef38c92 \
  --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --output-dir /kaimm-distill/wuwenxuan/output/qwen_3ref \
  --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
  --label qwen2511_3ref_122 \
  --prep-cmd 'true' \
  --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --cmd 'E=/kaimm-distill/wuwenxuan/envs/qwen-edit; SP=$E/lib/python3.11/site-packages; export LD_LIBRARY_PATH=$E/lib:$SP/torch/lib:$(echo $SP/nvidia/*/lib | tr " " :):$LD_LIBRARY_PATH; for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i QWEN_WEIGHTS=$INFER_WEIGHTS_DIR $E/bin/python qwen/infer_qwen_3ref.py --shard_idx $i --num_shards 8 --out $INFER_OUTPUT_DIR > $INFER_OUTPUT_DIR/shard$i.log 2>&1 & sleep 20; done; wait; QWEN_WEIGHTS=$INFER_WEIGHTS_DIR $E/bin/python qwen/infer_qwen_3ref.py --merge --out $INFER_OUTPUT_DIR'
```

入队信息：

```text
[infer_submit] 已入队 wuwenxuan__qwen2511_3ref_122__16c7a672b14a  (project=default, cluster=default(硬绑定), 当前排队 1 个, 本人在途 1/3)
               /kaimm-distill/infer_hub/queues/default/pending/wuwenxuan__qwen2511_3ref_122__16c7a672b14a.json
```

任务参数摘要：

| 字段 | 值 |
|---|---|
| job_id | `wuwenxuan__qwen2511_3ref_122__16c7a672b14a` |
| owner | `wuwenxuan` |
| project | `default` |
| cluster | `default`（H 卡主集群，硬绑定） |
| gpus | 8 |
| timeout | 60 min |
| commit | `16c7a672b14a385dae3af1bdd912e3abcef38c92` |
| weights | `/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511` |
| output_dir | `/kaimm-distill/wuwenxuan/output/qwen_3ref` |
| uv_env | `/kaimm-distill/wuwenxuan/envs/qwen-edit` |

---

## 5. G3 结果

infer_hub 状态：**成功**，耗时 42m26s，机器 `@aiplatform-wlf3-ge90-70`。

### 5.1 产物

已取回仓库 `output/qwen_3ref/`：

- `results.json`
- `results_shard{0..7}.json`
- `ALL_COMPARISON_part01.png` … `ALL_COMPARISON_part16.png`

全分辨率单图 122 张仍保留在共享盘 `/kaimm-distill/wuwenxuan/output/qwen_3ref/`，未入 git。

### 5.2 `results.json` 的 `meta` 原样

```json
{
  "meta": {
    "spec": "Q1B-qwen-3ref-v1",
    "model": "Qwen-Image-Edit-2511",
    "weights": "/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511",
    "task_json": "/var/infer_cache/worktrees/UNO-9c315c09@16c7a672b14a/datasets/eval_multiref/q1b_3ref_tasks.json",
    "n_all_tasks": 122,
    "n_shards": 8,
    "shard_totals": [
      {"shard_idx": 0, "total_s": 1627.3, "n_run": 16, "n_fail": 0},
      {"shard_idx": 1, "total_s": 1646.3, "n_run": 16, "n_fail": 0},
      {"shard_idx": 2, "total_s": 1512.0, "n_run": 15, "n_fail": 0},
      {"shard_idx": 3, "total_s": 1549.9, "n_run": 15, "n_fail": 0},
      {"shard_idx": 4, "total_s": 1502.2, "n_run": 15, "n_fail": 0},
      {"shard_idx": 5, "total_s": 1637.3, "n_run": 15, "n_fail": 0},
      {"shard_idx": 6, "total_s": 1512.3, "n_run": 15, "n_fail": 0},
      {"shard_idx": 7, "total_s": 1553.3, "n_run": 15, "n_fail": 0}
    ],
    "n_run": 122,
    "n_fail": 0,
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "height": 1024,
    "width": 1024,
    "total_s": 12540.6,
    "dry_run": false,
    "diffusers_version": "0.40.0.dev0",
    "torch_version": "2.5.1+cu124"
  }
}
```

### 5.3 实测汇总

| 指标 | Q1 (2-ref, 40 条) | Q1-B (3-ref, 122 条) |
|---|---|---|
| `total_s` | 2412.0 | 12540.6 |
| `n_run` | 40 | 122 |
| `s/img` | 60.3 | **102.8** |
| `peak_mem_gb` | 57.97–57.98 | 57.98 |
| `n_fail` | 0 | 0 |

3-ref 单张耗时约为 2-ref 的 **1.70×**（102.8 / 60.3），在 §3.4 估计的 1.6–1.8× 范围内。

### 5.4 任务覆盖

- 总任务 122 条，无失败，无 error。
- `elapsed_s` 范围：98.47 – 114.74。
- 段 A（`S4_*`）20 条 + 段 B（`Q1B_*`）102 条全部完成。

---

## 6. 后续

Q1-B 产物已齐：数字、日志、拼图和 122 张全分辨率图。是否做判读由你决定；本报告只记录现象与数字，不写"3-ref 行不行"的结论。
