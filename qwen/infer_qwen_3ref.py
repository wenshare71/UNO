"""Qwen-Image-Edit-2511 在 Q1-B 3-ref 任务表上的推理。

规格:`qwen/Q1B_3REF_RUN.md` §3.3。只新建本文件,既有文件一个字不动(R0)。
"""
import argparse
import glob
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "multibanana_eval"))

from PIL import Image

import board

# ------------------------------------------------------------------ CONSTANTS
# 来自 qwen/Q1B_3REF_RUN.md §3.3。改任何一个都是红档。

TASK_JSON = os.path.join(_REPO_ROOT, "datasets/eval_multiref/q1b_3ref_tasks.json")
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = " "
HEIGHT = WIDTH = 1024
DEFAULT_OUT = os.path.join(_REPO_ROOT, "output/qwen_3ref")
BOARD_MAX_BYTES = 2 * 1024 * 1024
BOARD_ROWS_PER_PART = 8


def load_tasks() -> list[dict]:
    """读 q1b_3ref_tasks.json,返回全部 122 条任务(不改顺序)。"""
    with open(TASK_JSON, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["tasks"]


def resolve_refs(task: dict) -> list[str]:
    """image_paths 是相对 datasets/eval_multiref/ 的相对路径。"""
    base = os.path.dirname(TASK_JSON)
    return [os.path.normpath(os.path.join(base, p)) for p in task["image_paths"]]


def output_exists(path: str) -> bool:
    """断点续跑:文件存在且 Image.open(...).load() 不抛异常才算有效。"""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


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


def shard_tasks(tasks: list[dict], shard_idx: int, num_shards: int) -> list[dict]:
    """按数组原始顺序分片:只取 i % num_shards == shard_idx。"""
    return [t for i, t in enumerate(tasks) if i % num_shards == shard_idx]


def run_shard(args):
    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    weights = os.environ.get("QWEN_WEIGHTS")
    if not args.dry_run and not weights:
        raise SystemExit(
            "❌ 环境变量 QWEN_WEIGHTS 未设置(无默认值,防止误读到别的权重)。\n"
            "   infer_hub 上写 QWEN_WEIGHTS=$INFER_WEIGHTS_DIR;\n"
            "   本地写 QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511")

    all_tasks = load_tasks()
    num_shards = max(1, args.num_shards)
    shard_idx = args.shard_idx
    if not (0 <= shard_idx < num_shards):
        raise SystemExit(f"❌ shard_idx {shard_idx} 超出 [0, {num_shards})")

    shard = shard_tasks(all_tasks, shard_idx, num_shards)
    if args.limit is not None:
        shard = shard[:args.limit]

    # 断点续跑:产物有效才跳过
    todo = []
    skipped = 0
    for t in shard:
        p = os.path.join(out_dir, f"{t['task_id']}.png")
        if output_exists(p):
            skipped += 1
        else:
            todo.append(t)

    # 启动自检行
    print(f"[自检] 全量 {len(all_tasks)} | 本 shard {len(shard)} 条 | "
          f"待跑 {len(todo)} | 已跳过 {skipped} | 输出 {out_dir} | "
          f"权重 {weights or '(dry_run)'}", flush=True)

    missing = [q for t in shard for q in resolve_refs(t) if not os.path.exists(q)]
    if missing:
        raise SystemExit(f"❌ {len(missing)} 张参考图不存在,例如:{missing[:3]}")

    # ---------- 初始化 ----------
    pipe = torch = None
    if not args.dry_run:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        t0 = time.perf_counter()
        pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        print(f"[自检] 权重加载 {time.perf_counter() - t0:.1f}s", flush=True)

        # §1 实例层确认 + §3.3 硬断言
        cfg_zct = getattr(pipe.transformer.config, "zero_cond_t", None)
        inst_zct = getattr(pipe.transformer, "zero_cond_t", None)
        print(f"[自检] transformer.config.zero_cond_t={cfg_zct}, transformer.zero_cond_t={inst_zct}", flush=True)
        assert cfg_zct is True, f"zero_cond_t 没生效,停 (config={cfg_zct})"
        assert inst_zct is True, f"zero_cond_t 没传到实例,停 (instance={inst_zct})"

    # ---------- 生成 ----------
    records: list[dict] = []
    n_fail = 0
    t_start = time.perf_counter()

    for idx, task in enumerate(todo, 1):
        tid = task["task_id"]
        ref_paths = resolve_refs(task)
        err = None
        peak_gb = None
        elapsed = None
        t_task = time.perf_counter()

        try:
            if args.dry_run:
                out_img = Image.new("RGB", (WIDTH, HEIGHT), (200, 220, 240))
            else:
                images = [Image.open(q).convert("RGB") for q in ref_paths]
                torch.cuda.reset_peak_memory_stats()
                gen = torch.Generator(device="cuda").manual_seed(task["seed"])
                out_img = pipe(
                    image=images,
                    prompt=task["prompt"],
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    true_cfg_scale=TRUE_CFG_SCALE,
                    height=HEIGHT, width=WIDTH,
                    generator=gen,
                ).images[0]
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            out_img = None
            n_fail += 1
            print(f"  ❌ {tid}  {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)

        elapsed = time.perf_counter() - t_task
        if out_img is not None:
            out_img.save(os.path.join(out_dir, f"{tid}.png"))

        records.append({
            "task_id": tid,
            "n_refs": task["meta"]["n_refs"],
            "seed": task["seed"],
            "prompt": task["prompt"],
            "image_paths": ref_paths,
            "elapsed_s": round(elapsed, 2),
            "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else None,
            "error": err,
            "shard_idx": shard_idx,
        })

        avg = (time.perf_counter() - t_start) / idx
        print(f"[{time.strftime('%H:%M:%S')}] shard{shard_idx} {idx}/{len(todo)} "
              f"({idx / len(todo) * 100:.1f}%) | {avg:.1f} s/img | "
              f"ETA {fmt_eta(avg * (len(todo) - idx))} | fail {n_fail} | {tid}", flush=True)

    # ---------- 写 shard 结果 ----------
    total_s = time.perf_counter() - t_start
    meta = {
        "spec": "Q1B-qwen-3ref-v1",
        "model": "Qwen-Image-Edit-2511",
        "weights": weights,
        "task_json": TASK_JSON,
        "n_all_tasks": len(all_tasks),
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "n_shard_tasks": len(shard),
        "n_run": len(todo),
        "n_skipped_resume": skipped,
        "n_fail": n_fail,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": HEIGHT, "width": WIDTH,
        "total_s": round(total_s, 1),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        import diffusers
        meta["diffusers_version"] = diffusers.__version__
        meta["torch_version"] = torch.__version__

    out_path = os.path.join(out_dir, f"results_shard{shard_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tasks": records}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print(f"shard {shard_idx}/{num_shards} | 本 shard {len(shard)} 条 | "
          f"本次跑 {len(todo)} | 失败 {n_fail} | 总耗时 {fmt_eta(total_s)}")
    print(f"results_shard{shard_idx}.json : {out_path}")
    print("=" * 68)


def run_merge(args):
    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or DEFAULT_OUT
    if not os.path.isdir(out_dir):
        raise SystemExit(f"❌ 输出目录不存在:{out_dir}")

    all_tasks = load_tasks()
    order = {t["task_id"]: i for i, t in enumerate(all_tasks)}

    shard_files = sorted(glob.glob(os.path.join(out_dir, "results_shard*.json")))
    if not shard_files:
        raise SystemExit(f"❌ 在 {out_dir} 未找到 results_shard*.json")

    merged_tasks: list[dict] = []
    shard_metas: list[dict] = []
    for sf in shard_files:
        with open(sf, "rt", encoding="utf-8") as f:
            doc = json.load(f)
        shard_metas.append(doc.get("meta", {}))
        for t in doc.get("tasks", []):
            # 合并后不再保留 shard_idx(每条任务唯一),按原表顺序排
            merged_tasks.append({k: v for k, v in t.items() if k != "shard_idx"})

    merged_tasks.sort(key=lambda t: order.get(t["task_id"], len(order)))

    n_shards = len(shard_metas)
    total_s = sum(m.get("total_s", 0) for m in shard_metas)
    n_run = sum(m.get("n_run", 0) for m in shard_metas)
    n_fail = sum(m.get("n_fail", 0) for m in shard_metas)

    meta = {
        "spec": "Q1B-qwen-3ref-v1",
        "model": "Qwen-Image-Edit-2511",
        "weights": shard_metas[0].get("weights") if shard_metas else None,
        "task_json": TASK_JSON,
        "n_all_tasks": len(all_tasks),
        "n_shards": n_shards,
        "shard_totals": [
            {
                "shard_idx": m.get("shard_idx"),
                "total_s": m.get("total_s"),
                "n_run": m.get("n_run"),
                "n_fail": m.get("n_fail"),
            }
            for m in shard_metas
        ],
        "n_run": n_run,
        "n_fail": n_fail,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": HEIGHT, "width": WIDTH,
        "total_s": round(total_s, 1),
        "dry_run": all(m.get("dry_run", False) for m in shard_metas),
    }
    # 非 dry_run 时从任意 shard meta 抄版本号(应一致)
    for key in ("diffusers_version", "torch_version"):
        vals = [m.get(key) for m in shard_metas if m.get(key)]
        if vals:
            meta[key] = vals[0]

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tasks": merged_tasks}, f, indent=2, ensure_ascii=False)

    # ---------- 拼图 ----------
    rows = []
    for task in all_tasks:
        gen_path = os.path.join(out_dir, f"{task['task_id']}.png")
        if not os.path.exists(gen_path):
            continue
        refs = [Image.open(q).convert("RGB") for q in resolve_refs(task)]
        rows.append(board.build_row(task["task_id"], task["prompt"], refs,
                                    {"qwen2511": Image.open(gen_path)}))
    board_paths = save_board(rows, out_dir) if rows else []

    print("\n" + "=" * 68)
    print(f"合并 {n_shards} 个 shard | 总任务 {len(merged_tasks)} | 失败 {n_fail} | 总耗时 {fmt_eta(total_s)}")
    print(f"results.json : {os.path.join(out_dir, 'results.json')}")
    for q in board_paths:
        print(f"拼图         : {q}")
    print("=" * 68)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true",
                   help="不加载模型、不用 GPU,用纯色占位图验证分片 / 落盘 / 合并 / 拼图")
    p.add_argument("--shard_idx", type=int, default=0, help="分片下标,默认 0")
    p.add_argument("--num_shards", type=int, default=1, help="分片总数,默认 1")
    p.add_argument("--merge", action="store_true", help="只合并 + 出拼图,不推理")
    p.add_argument("--out", default=None, help="覆盖输出目录")
    p.add_argument("--limit", type=int, default=None, help="只跑本分片的前 N 条(冒烟用)")
    args = p.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
