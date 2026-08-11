"""Qwen-Image-Edit-2511 在我们 M6 多参考任务子集上的**裸基线**推理。

规格:`distill/Q1_QWEN_BASELINE_RUN.md` §6。这一轮的目的不是做实验,是量一个零点——
换底座之前先看清楚 Qwen 不加任何改造在我们自己的任务分布上是什么水平。
所以口径全部锁死在官方 model card 的默认值:**不调参、不改 prompt、不挑样本**。

§6.1 的常量写死在下面 CONSTANTS 段,**不做 CLI 可调**。CLI 只有四个开关,
都不改变实验语义(执行手册绿档 G3)。要改任何常量都是红档,报上来。

用法:
  # dry_run:不加载模型、不用 GPU,验证子集规则 / 落盘 / 拼图
  python scripts/infer_qwen_edit.py --dry_run

  # 4090 单例冒烟(24G 装不下 20B bf16,靠 offload 过一遍,慢但只为验通路)
  QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
    python scripts/infer_qwen_edit.py --limit 1 --offload

  # infer_hub 正式跑(单卡 H,worker 注入 INFER_WEIGHTS_DIR / INFER_OUTPUT_DIR)
  QWEN_WEIGHTS=$INFER_WEIGHTS_DIR python scripts/infer_qwen_edit.py
"""
import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# 复用 multibanana_eval/board.py 的拼图逻辑(标题按像素宽度换行,不会把长 prompt 截掉)。
# 只 import 不改动——既有文件一个字不动(R0)。
sys.path.insert(0, os.path.join(_REPO_ROOT, "multibanana_eval"))

from PIL import Image

import board

# ------------------------------------------------------------------ CONSTANTS
# 全部来自 Q1_QWEN_BASELINE_RUN.md §6.1。改任何一个都是红档。

TASK_JSON = os.path.join(_REPO_ROOT, "datasets/eval_multiref/m6_tasks.json")
SUBSET_STRIDE = 8            # 按 tasks 数组原始顺序取 i % 8 == 0 ⇒ 320/8 = 40 条
NUM_INFERENCE_STEPS = 40     # 官方示例值
TRUE_CFG_SCALE = 4.0         # 官方示例值
NEGATIVE_PROMPT = " "        # 官方示例值(一个空格)
HEIGHT = WIDTH = 1024        # Qwen 原生分辨率。⚠️ 与我们 UNO 侧的 512×512 不同,
                             # 是刻意的——压到 512 比的就不是 Qwen 的真实水平。
                             # 并排看图时必须知道两边尺寸不同,这条要写进报告。
DEFAULT_OUT = os.path.join(_REPO_ROOT, "output/qwen_baseline")
BOARD_MAX_BYTES = 2 * 1024 * 1024   # 超过就分批出图(执行手册 §3.2:单文件 >2MB 别进 reports)
BOARD_ROWS_PER_PART = 8


def load_subset(limit: int | None) -> tuple[list[dict], int]:
    """读 m6_tasks.json,按 §6.2 的机械规则取子集。返回 (子集, 全量条数)。

    规则只有一条:**按数组原始顺序**(不排序、不打乱)取下标 i % SUBSET_STRIDE == 0。
    --limit 只截前 N 条,不改变选取规则——这样产物能在本地按规格独立重算逐条 diff,
    这是黄档放权的依据,不是形式主义。
    """
    with open(TASK_JSON, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    all_tasks = doc["tasks"]
    subset = [t for i, t in enumerate(all_tasks) if i % SUBSET_STRIDE == 0]
    if limit is not None:
        subset = subset[:limit]
    return subset, len(all_tasks)


def resolve_refs(task: dict) -> list[str]:
    """image_paths 是相对 datasets/eval_multiref/ 的相对路径(形如 ../dreambooth/...)。"""
    base = os.path.dirname(TASK_JSON)
    return [os.path.normpath(os.path.join(base, p)) for p in task["image_paths"]]


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def save_board(rows: list[Image.Image], out_dir: str) -> list[str]:
    """先整拼一张;超过 2MB 就删掉改为按 BOARD_ROWS_PER_PART 分批。"""
    full = os.path.join(out_dir, "ALL_COMPARISON.png")
    board.stack_board(rows).save(full)
    if os.path.getsize(full) <= BOARD_MAX_BYTES:
        return [full]
    os.remove(full)
    paths = []
    for k in range(0, len(rows), BOARD_ROWS_PER_PART):
        p = os.path.join(out_dir, f"ALL_COMPARISON_part{k // BOARD_ROWS_PER_PART + 1:02d}.png")
        board.stack_board(rows[k:k + BOARD_ROWS_PER_PART]).save(p)
        paths.append(p)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true",
                   help="不加载模型,用纯色占位图跑通子集规则 / 落盘 / 拼图")
    p.add_argument("--limit", type=int, default=None, help="只跑子集的前 N 条")
    p.add_argument("--offload", action="store_true",
                   help="enable_model_cpu_offload():24G 卡装不下 20B bf16 时用,慢")
    p.add_argument("--out", default=None, help="覆盖输出目录")
    args = p.parse_args()

    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    weights = os.environ.get("QWEN_WEIGHTS")
    if not args.dry_run and not weights:
        raise SystemExit(
            "❌ 环境变量 QWEN_WEIGHTS 未设置(无默认值,防止误读到别的权重)。\n"
            "   infer_hub 上写 QWEN_WEIGHTS=$INFER_WEIGHTS_DIR;\n"
            "   本地写 QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511")

    tasks, n_all = load_subset(args.limit)
    # 断点续跑:已有产物的直接跳过(执行手册 §4.1 硬性要求)
    todo = [t for t in tasks if not os.path.exists(os.path.join(out_dir, f"{t['task_id']}.png"))]
    n_skipped = len(tasks) - len(todo)

    # 启动自检行:第一秒就能发现"任务数不对"这种致命错误,而不是等跑完
    print(f"[自检] 全量 {n_all} 条 | 子集(i%{SUBSET_STRIDE}==0)"
          f"{len(tasks)} 条 | 待跑 {len(todo)} | 已跳过 {n_skipped} | "
          f"输出 {out_dir} | 权重 {weights or '(dry_run)'}", flush=True)

    missing = [q for t in tasks for q in resolve_refs(t) if not os.path.exists(q)]
    if missing:
        raise SystemExit(f"❌ {len(missing)} 张参考图不存在,例如:{missing[:3]}")

    # ---------- 初始化 ----------
    pipe = torch = None
    if not args.dry_run:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        t0 = time.perf_counter()
        pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        print(f"[自检] 权重加载 {time.perf_counter() - t0:.1f}s", flush=True)

    # ---------- 生成 ----------
    records: list[dict] = []
    n_fail = 0
    t_start = time.perf_counter()

    for idx, task in enumerate(todo, 1):
        tid = task["task_id"]
        ref_paths = resolve_refs(task)
        images = [Image.open(q).convert("RGB") for q in ref_paths]
        err = None
        peak_gb = None
        t_task = time.perf_counter()

        if args.dry_run:
            out_img = Image.new("RGB", (WIDTH, HEIGHT), (200, 220, 240))
        else:
            torch.cuda.reset_peak_memory_stats()
            try:
                # seed 用任务自带的,和 m6_iso / m6_full 的产出同源,便于三方并排。
                # generator 固定用 cuda:Philox 序列只取决于 seed 与形状,跨卡型一致,
                # 所以 4090 冒烟和 H 卡正式跑拿到的是同一份噪声。
                gen = torch.Generator(device="cuda").manual_seed(task["seed"])
                out_img = pipe(
                    image=images,                      # 顺序原样,不重排
                    prompt=task["prompt"],             # 原样,不加前缀、不改写
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    true_cfg_scale=TRUE_CFG_SCALE,
                    height=HEIGHT, width=WIDTH,
                    generator=gen,
                ).images[0]
            except Exception as e:                     # 外部模型 + 外部图,是真会失败的边界
                err = f"{type(e).__name__}: {e}"
                out_img = None
                n_fail += 1
                # 当场打印,不攒到最后——攒起来的后果是跑两小时才发现前十分钟就全在失败
                print(f"  ❌ {tid}  {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        elapsed = time.perf_counter() - t_task
        if out_img is not None:
            out_img.save(os.path.join(out_dir, f"{tid}.png"))

        # 失败的条目**保留**,只记 error。不许因为老失败就把它从子集里删掉。
        records.append({"task_id": tid, "n_refs": task["meta"]["n_refs"], "seed": task["seed"],
                        "prompt": task["prompt"], "image_paths": ref_paths,
                        "elapsed_s": round(elapsed, 2),
                        "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else None,
                        "error": err})

        avg = (time.perf_counter() - t_start) / idx
        print(f"[{time.strftime('%H:%M:%S')}] {idx}/{len(todo)} ({idx / len(todo) * 100:.1f}%) | "
              f"{avg:.1f} s/img | ETA {fmt_eta(avg * (len(todo) - idx))} | "
              f"fail {n_fail} | {tid}", flush=True)

    # ---------- 汇总 ----------
    total_s = time.perf_counter() - t_start
    meta = {
        "spec": "Q1-qwen-baseline-v1",
        "model": "Qwen-Image-Edit-2511",
        "weights": weights,
        "task_json": TASK_JSON, "n_all_tasks": n_all,
        "subset_stride": SUBSET_STRIDE, "n_subset": len(tasks),
        "n_run": len(todo), "n_skipped_resume": n_skipped, "n_fail": n_fail,
        "num_inference_steps": NUM_INFERENCE_STEPS, "true_cfg_scale": TRUE_CFG_SCALE,
        "negative_prompt": NEGATIVE_PROMPT, "height": HEIGHT, "width": WIDTH,
        "total_s": round(total_s, 1), "dry_run": args.dry_run, "offload": args.offload,
    }
    if not args.dry_run:
        import diffusers
        meta["diffusers_version"] = diffusers.__version__
        meta["torch_version"] = torch.__version__

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tasks": records}, f, indent=2, ensure_ascii=False)

    # ---------- 拼图 ----------
    # 用全子集(含本次跳过的)重建,断点续跑时也能出完整总览。
    rows = []
    for task in tasks:
        gen_path = os.path.join(out_dir, f"{task['task_id']}.png")
        if not os.path.exists(gen_path):
            continue
        refs = [Image.open(q).convert("RGB") for q in resolve_refs(task)]
        rows.append(board.build_row(task["task_id"], task["prompt"], refs,
                                    {"qwen2511": Image.open(gen_path)}))
    board_paths = save_board(rows, out_dir) if rows else []

    print("\n" + "=" * 68)
    print(f"子集 {len(tasks)} 条 | 本次跑 {len(todo)} | 失败 {n_fail} | 总耗时 {fmt_eta(total_s)}")
    print(f"results.json : {os.path.join(out_dir, 'results.json')}")
    for q in board_paths:
        print(f"拼图         : {q}")
    print("=" * 68)
    if n_fail:
        print(f"⚠️  有 {n_fail} 条失败,条目已保留在 results.json 里(error 字段),未从子集删除。")


if __name__ == "__main__":
    main()
