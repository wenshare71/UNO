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

## 3. 下一步

待 G2（`qwen/infer_qwen_3ref.py` 的 `--dry_run` 8 分片）通过并确认后，提交正式 job 跑 Q1-B 122 条 3-ref。
