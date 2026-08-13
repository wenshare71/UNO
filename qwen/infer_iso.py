"""P3 评测出图:`qwen_full` / `qwen_iso_pre` / `qwen_iso_post` 三个变体。

规格:`qwen/PLAN.md` §3.3(三变体、240 条口径)、§4(预登记的速度预测)、
§5(不改 Q1 口径)。**只新建本文件,既有 `.py` 一个字不动(R0)。**

口径全部沿用 Q1 —— steps 40 / true_cfg 4.0 / 1024² / negative_prompt " " / prompt 原样 /
seed 从任务表取。三个变体之间**只差 transformer 内部**:隔离臂靠
`IsoPipelineHook` 换掉 `transformer.forward`,去噪循环、sigma 网格、CFG 范数重标定、
scheduler 全是 stock 代码(为什么这么挂见 `pipeline_iso.IsoPipelineHook` 的 docstring)。

**不出拼图。** 带变体名的拼图在盲评判读完成前不许进 git(§5),拼图交给盲评那一侧
按它自己的纪律做。这里只产出 `<task_id>.png` 和 `results_shard*.json`。

用法:
    # 1. bf16 确认(真权重下 隔离-无缓存 vs 隔离-有缓存),顺带量加速比
    python qwen/infer_iso.py --variant iso_pre --cache_check 3 --limit 3

    # 2. 三个臂各出一批
    python qwen/infer_iso.py --variant full
    python qwen/infer_iso.py --variant iso_pre
    python qwen/infer_iso.py --variant iso_post --lora <ckpt>

    # 3. 合并分片
    python qwen/infer_iso.py --variant iso_pre --merge
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import Image

# ------------------------------------------------------------------ Q1 口径
# 与 qwen/infer_qwen_3ref.py 逐字相同。改任何一个都是红档(PLAN §5)。
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = " "
HEIGHT = WIDTH = 1024

VARIANTS = ("full", "iso_pre", "iso_post")

# 任务表。m6 那 240 条是 PLAN §3.3 的默认口径:每层取 i % 4 != 3。
TASK_TABLES = {
    "m6": "datasets/eval_multiref/m6_tasks.json",
    "m6_floor": "datasets/eval_multiref/m6_floor_tasks.json",
    "q1b_3ref": "datasets/eval_multiref/q1b_3ref_tasks.json",
}


def load_tasks(table: str) -> tuple[list[dict], str]:
    path = os.path.join(_REPO_ROOT, TASK_TABLES[table])
    with open(path, "rt", encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    if table == "m6":
        # PLAN §3.3:每层 i % 4 != 3 ⇒ S1 165 + S3 75 = 240。
        # 分层各自计数,不是对全表取模——两层的条数不是 4 的倍数,混算会偏。
        kept, seen = [], {}
        for t in tasks:
            s = t["stratum"]
            i = seen.get(s, 0)
            seen[s] = i + 1
            if i % 4 != 3:
                kept.append(t)
        if len(kept) != 240:
            raise SystemExit(f"❌ m6 子集应为 240 条,实得 {len(kept)}。任务表变了,停下来看。")
        tasks = kept
    return tasks, path


def resolve_refs(task: dict, task_json: str) -> list[str]:
    base = os.path.dirname(task_json)
    return [os.path.normpath(os.path.join(base, p)) for p in task["image_paths"]]


def output_exists(path: str) -> bool:
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


def build_pipe(weights: str, variant: str, lora: str | None, block_diag: bool,
               always_write: bool):
    """返回 (pipe, hook, torch)。`full` 臂 hook 为 None,走完全未改动的 stock 路径。"""
    import torch
    from diffusers import QwenImageEditPlusPipeline

    t0 = time.perf_counter()
    pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print(f"[自检] 权重加载 {time.perf_counter() - t0:.1f}s", flush=True)

    # §1 的地基:没有 zero_cond_t,ref 段就不是固定 t=0 调制,缓存从无损降级成近似
    cfg_zct = getattr(pipe.transformer.config, "zero_cond_t", None)
    inst_zct = getattr(pipe.transformer, "zero_cond_t", None)
    print(f"[自检] transformer.config.zero_cond_t={cfg_zct}, "
          f"transformer.zero_cond_t={inst_zct}", flush=True)
    assert cfg_zct is True, f"zero_cond_t 没生效,停 (config={cfg_zct})"
    assert inst_zct is True, f"zero_cond_t 没传到实例,停 (instance={inst_zct})"

    if variant == "iso_post":
        if not lora:
            raise SystemExit("❌ iso_post 必须给 --lora。不给就是在拿未训练的权重冒充训练后的臂。")
        pipe.load_lora_weights(lora)
        print(f"[自检] LoRA 已加载 {lora}", flush=True)
    elif lora:
        raise SystemExit(f"❌ --variant {variant} 不该带 --lora。"
                         f"full / iso_pre 都是 stock 权重(PLAN §3.3)。")

    hook = None
    if variant != "full":
        from pipeline_iso import IsoPipelineHook
        hook = IsoPipelineHook(pipe, block_diag=block_diag, always_write=always_write)
        print(f"[自检] 隔离注意力已挂载 | block_diag={block_diag} | "
              f"缓存={'关(每步重算)' if always_write else '开'}", flush=True)
    return pipe, hook, torch


def generate(pipe, hook, torch, task: dict, ref_paths: list[str]):
    """跑一张图,返回 (PIL.Image, peak_gb, elapsed_s)。"""
    if hook is not None:
        hook.reset()          # 不 reset 就会读到上一张图的 ref K/V,而且不报错
    images = [Image.open(q).convert("RGB") for q in ref_paths]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(task["seed"])
    out = pipe(
        image=images,
        prompt=task["prompt"],
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        true_cfg_scale=TRUE_CFG_SCALE,
        height=HEIGHT, width=WIDTH,
        generator=gen,
    ).images[0]
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() / 1024**3, time.perf_counter() - t0


def pixel_diff(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """与 `../scripts/bench_kv_cache.py` 同口径:uint8 像素的 max / mean 绝对差。"""
    import numpy as np
    d = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    return float(d.max()), float(d.mean())


def run_shard(args):
    variant = args.variant
    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or \
        os.path.join(_REPO_ROOT, f"output/qwen_{variant}_{args.tasks}")
    os.makedirs(out_dir, exist_ok=True)

    weights = os.environ.get("QWEN_WEIGHTS")
    if not args.dry_run and not weights:
        raise SystemExit(
            "❌ 环境变量 QWEN_WEIGHTS 未设置(无默认值,防止误读到别的权重)。\n"
            "   本地写 QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511")

    all_tasks, task_json = load_tasks(args.tasks)
    num_shards = max(1, args.num_shards)
    if not (0 <= args.shard_idx < num_shards):
        raise SystemExit(f"❌ shard_idx {args.shard_idx} 超出 [0, {num_shards})")
    shard = [t for i, t in enumerate(all_tasks) if i % num_shards == args.shard_idx]
    if args.limit is not None:
        shard = shard[:args.limit]

    todo, skipped = [], 0
    for t in shard:
        if output_exists(os.path.join(out_dir, f"{t['task_id']}.png")):
            skipped += 1
        else:
            todo.append(t)

    print(f"[自检] 变体 {variant} | 表 {args.tasks} 全量 {len(all_tasks)} | "
          f"本 shard {len(shard)} | 待跑 {len(todo)} | 已跳过 {skipped} | 输出 {out_dir} | "
          f"权重 {weights or '(dry_run)'}", flush=True)

    missing = [q for t in shard for q in resolve_refs(t, task_json) if not os.path.exists(q)]
    if missing:
        raise SystemExit(f"❌ {len(missing)} 张参考图不存在,例如:{missing[:3]}")

    pipe = hook = torch = None
    if not args.dry_run:
        pipe, hook, torch = build_pipe(weights, variant, args.lora, args.block_diag,
                                       args.iso_no_cache)

    records: list[dict] = []
    cache_checks: list[dict] = []
    n_fail = 0
    t_start = time.perf_counter()

    for idx, task in enumerate(todo, 1):
        tid = task["task_id"]
        ref_paths = resolve_refs(task, task_json)
        err = peak_gb = elapsed = None
        t_task = time.perf_counter()

        try:
            if args.dry_run:
                out_img = Image.new("RGB", (WIDTH, HEIGHT), (200, 220, 240))
            else:
                out_img, peak_gb, elapsed = generate(pipe, hook, torch, task, ref_paths)

                # bf16 确认:同一 seed 再跑一遍无缓存版,比像素差(PLAN §3.1 引的先例口径)
                if idx <= args.cache_check and hook is not None and not hook.always_write:
                    hook.always_write = True
                    ref_img, _, elapsed_nc = generate(pipe, hook, torch, task, ref_paths)
                    hook.always_write = False
                    mx, mn = pixel_diff(out_img, ref_img)
                    cache_checks.append({
                        "task_id": tid, "n_refs": task["meta"]["n_refs"],
                        "px_max": mx, "px_mean": round(mn, 4),
                        "s_cached": round(elapsed, 2), "s_nocache": round(elapsed_nc, 2),
                        "speedup": round(elapsed_nc / elapsed, 3),
                    })
                    print(f"  [缓存确认] {tid}  像素差 max={mx:.0f} mean={mn:.4f} | "
                          f"{elapsed_nc:.1f}s → {elapsed:.1f}s ({elapsed_nc / elapsed:.2f}×)",
                          flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            out_img = None
            n_fail += 1
            print(f"  ❌ {tid}  {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)

        if elapsed is None:
            elapsed = time.perf_counter() - t_task
        if out_img is not None:
            out_img.save(os.path.join(out_dir, f"{tid}.png"))

        records.append({
            "task_id": tid,
            "stratum": task.get("stratum"),
            "n_refs": task["meta"]["n_refs"],
            "seed": task["seed"],
            "prompt": task["prompt"],
            "image_paths": ref_paths,
            "elapsed_s": round(elapsed, 2),
            "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else None,
            "error": err,
            "shard_idx": args.shard_idx,
        })

        avg = (time.perf_counter() - t_start) / idx
        print(f"[{time.strftime('%H:%M:%S')}] {variant} shard{args.shard_idx} {idx}/{len(todo)} "
              f"({idx / len(todo) * 100:.1f}%) | {avg:.1f} s/img | "
              f"ETA {fmt_eta(avg * (len(todo) - idx))} | fail {n_fail} | {tid}", flush=True)

    total_s = time.perf_counter() - t_start
    meta = {
        "spec": "P3-qwen-iso-v1",
        "variant": variant,
        "model": "Qwen-Image-Edit-2511",
        "weights": weights,
        "lora": args.lora,
        "block_diag": args.block_diag,
        "iso_no_cache": args.iso_no_cache,
        "task_table": args.tasks,
        "task_json": task_json,
        "n_all_tasks": len(all_tasks),
        "shard_idx": args.shard_idx,
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
    if cache_checks:
        meta["cache_check"] = cache_checks
    if hook is not None:
        meta["n_forward_write"] = hook.n_write
        meta["n_forward_read"] = hook.n_read
    if not args.dry_run:
        import diffusers
        meta["diffusers_version"] = diffusers.__version__
        meta["torch_version"] = torch.__version__

    out_path = os.path.join(out_dir, f"results_shard{args.shard_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tasks": records}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print(f"{variant} shard {args.shard_idx}/{num_shards} | 本次跑 {len(todo)} | "
          f"失败 {n_fail} | 总耗时 {fmt_eta(total_s)}")
    if hook is not None:
        print(f"前向次数:write {hook.n_write} / read {hook.n_read} "
              f"(每张图应是 1 写 79 读)")
    report_speed(records)
    print(f"results_shard{args.shard_idx}.json : {out_path}")
    print("=" * 68)


def report_speed(records: list[dict]) -> None:
    """按 n_refs 分组报 s/img —— §4.4 的速度预测就是按 1-ref / 2-ref 分行登记的。"""
    ok = [r for r in records if r["error"] is None and r["elapsed_s"]]
    if not ok:
        return
    by: dict[int, list[float]] = {}
    for r in ok:
        by.setdefault(r["n_refs"], []).append(r["elapsed_s"])
    for n in sorted(by):
        v = by[n]
        print(f"  {n}-ref  n={len(v):3d}  中位 {statistics.median(v):6.1f} s/img  "
              f"均值 {statistics.mean(v):6.1f} s/img")


def run_merge(args):
    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or \
        os.path.join(_REPO_ROOT, f"output/qwen_{args.variant}_{args.tasks}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"❌ 输出目录不存在:{out_dir}")

    all_tasks, task_json = load_tasks(args.tasks)
    order = {t["task_id"]: i for i, t in enumerate(all_tasks)}

    shard_files = sorted(glob.glob(os.path.join(out_dir, "results_shard*.json")))
    if not shard_files:
        raise SystemExit(f"❌ 在 {out_dir} 未找到 results_shard*.json")

    merged, metas = [], []
    for sf in shard_files:
        with open(sf, "rt", encoding="utf-8") as f:
            doc = json.load(f)
        metas.append(doc.get("meta", {}))
        merged.extend({k: v for k, v in t.items() if k != "shard_idx"}
                      for t in doc.get("tasks", []))
    merged.sort(key=lambda t: order.get(t["task_id"], len(order)))

    variants = {m.get("variant") for m in metas}
    if len(variants) > 1:
        raise SystemExit(f"❌ 同一目录下混了多个变体 {variants}。分开放,别混判。")

    meta = dict(metas[0])
    for k in ("shard_idx", "n_shard_tasks", "n_skipped_resume"):
        meta.pop(k, None)
    meta["n_shards"] = len(metas)
    meta["n_run"] = sum(m.get("n_run", 0) for m in metas)
    meta["n_fail"] = sum(m.get("n_fail", 0) for m in metas)
    meta["total_s"] = round(sum(m.get("total_s", 0) for m in metas), 1)

    missing = [t["task_id"] for t in all_tasks
               if not os.path.exists(os.path.join(out_dir, f"{t['task_id']}.png"))]
    meta["n_missing_png"] = len(missing)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tasks": merged}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print(f"{meta.get('variant')} | 合并 {len(metas)} 个 shard | 任务 {len(merged)} | "
          f"失败 {meta['n_fail']} | 缺图 {len(missing)}")
    if missing:
        print(f"  缺图示例:{missing[:5]}")
    report_speed(merged)
    print(f"results.json : {os.path.join(out_dir, 'results.json')}")
    print("=" * 68)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--tasks", default="m6", choices=sorted(TASK_TABLES))
    p.add_argument("--lora", default=None, help="iso_post 的 LoRA 目录,其余变体不许给")
    p.add_argument("--block_diag", action="store_true",
                   help="ref 之间也隔离(本轮不训,只留开关,PLAN §3.1)")
    p.add_argument("--iso_no_cache", action="store_true",
                   help="隔离但不缓存,每步重算 ref K/V。只用于对照,不是评测臂")
    p.add_argument("--cache_check", type=int, default=0, metavar="N",
                   help="前 N 条同 seed 再跑一遍无缓存版,报像素差与加速比(bf16 确认)")
    p.add_argument("--dry_run", action="store_true", help="不加载模型,用占位图验证分片/落盘/合并")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="只合并,不推理")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None, help="只跑本分片前 N 条(冒烟用)")
    args = p.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
