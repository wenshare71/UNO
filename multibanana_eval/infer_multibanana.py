"""用我们的 UNO ref_isolation LoRA 在 MultiBanana 数据集上推理。

复用 scripts/smoke_ref_isolation.py 里在远程真跑通的加载/OOM/计时路径(拼图与
swap_lora 直接复制过来,让本目录自包含、远程 git pull 后无需依赖 scripts/):
  - init 后把 t5/clip 踢回 CPU 给 DiT 腾显存(否则 24G 卡首个 forward 会 OOM)
  - swap_lora 硬校验 key 匹配,避免 LoRA 静默加载失败(否则比的是"官方 vs 官方")
  - 只给 denoise 循环计时,offload 搬运不算进加速比

对每个 MultiBanana 任务(读 <num>_<i> 参考图 + <num>_prompt.txt):
  - 逐变体生成,拼一张 [ref… | 各变体] 的对比图,肉眼看 LoRA 在没见过的多主体组合上的泛化
  - 主变体(默认 ours_kv)额外写回任务目录 <num>_generated.jpg,遵循 MultiBanana 约定,
    可直接跑它自带的 judge.py 用 VLM 打分(数据集无 GT,定量只能靠 judge 或身份相似度)

用法(远程单卡,先跑 download_multibanana.py):
  python multibanana_eval/infer_multibanana.py \
      --lora_path log/ref_isolation/checkpoint-7000/dit_lora.safetensors \
      --model_type flux-dev-fp8
  python multibanana_eval/infer_multibanana.py --dry_run   # 不用 GPU,验证任务发现/拼图/落盘
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

# 必须在 import torch 之前设置才生效(与 smoke 脚本一致):24G 卡放 22.4G 的 DiT,
# 余量全靠碎片管理,PyTorch OOM 时自己就建议开这个。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 让脚本不管从哪个目录运行都能找到 uno 包(uno/ 缺 __init__.py,走 namespace fallback)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont

# name: (用我们的 LoRA?, ref_isolation, kv_cache) —— 语义同 smoke_ref_isolation.py
VARIANTS = {
    "official_full": (False, False, False),  # 官方 UNO + 全注意力:质量金标准 & 速度基线
    "ours_full":     (True,  False, False),  # 我们的 LoRA + 全注意力:诊断项
    "ours_kv":       (True,  True,  True),   # 我们的 LoRA + KV-Cache:待验证方案
    "ours_iso":      (True,  True,  False),  # 纯隔离注意力(不开 cache),验缓存实现
}
DEFAULT_VARIANTS = ["official_full", "ours_kv"]

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------- 任务发现

def discover_tasks(data_dir: str) -> list[dict]:
    """递归发现 MultiBanana 任务:每个 <num>_prompt.txt 对应一个任务。

    用 prompt 文件(而不是 judge.py 那种靠 _generated 反查)来枚举,因为我们还没生成
    任何图。参考图用 glob 收全所有数字后缀 <num>_<i>,不 hardcode 起始下标——
    judge.py 从 _1 开始数、README 示例从 _0 开始,两者矛盾,收全最保险。
    """
    tasks = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            m = re.match(r"(\d+)_prompt\.txt$", fn)
            if not m:
                continue
            number = m.group(1)
            with open(os.path.join(root, fn), "rt", encoding="utf-8") as f:
                prompt = f.read().strip()
            # 收集同任务的参考图:<number>_<数字>.<ext>,排除 _generated/_prompt
            refs = []
            for cand in files:
                rm = re.match(rf"{number}_(\d+)\.(\w+)$", cand)
                if rm and cand.lower().endswith(_IMG_EXTS):
                    refs.append((int(rm.group(1)), os.path.join(root, cand)))
            refs = [p for _i, p in sorted(refs)]
            if not refs or not prompt:
                continue
            # 任务名带上目录名,避免不同目录的相同编号撞车
            task_name = f"{os.path.basename(root)}_{number}"
            tasks.append({"name": task_name, "dir": root, "number": number,
                          "prompt": prompt, "image_paths": refs})
    return sorted(tasks, key=lambda t: t["name"])


# ---------------------------------------------------------------- 拼图(复制自 smoke)

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _labeled(img: Image.Image, text: str, cell: int) -> Image.Image:
    bar = 24
    canvas = Image.new("RGB", (cell, cell + bar), (255, 255, 255))
    w, h = img.size
    scale = min(cell / w, cell / h)
    resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas.paste(resized, ((cell - resized.width) // 2, bar + (cell - resized.height) // 2))
    ImageDraw.Draw(canvas).text((4, 5), text[:38], fill=(0, 0, 0), font=_font(14))
    return canvas


def make_comparison(task: dict, refs: list[Image.Image], results: dict,
                    times: dict | None = None, cell: int = 256) -> Image.Image:
    """一行:[ref1..refN | 红线 | 各变体],变体标签带该例 denoise 耗时。"""
    tiles = [_labeled(r, f"ref{i + 1}", cell) for i, r in enumerate(refs)]
    for name, img in results.items():
        t = (times or {}).get(name)
        tiles.append(_labeled(img, f"{name}  {t:.1f}s" if t else name, cell))
    gap, sep_at = 8, len(refs)
    row = Image.new("RGB", (len(tiles) * cell + gap, cell + 24 + 20), (255, 255, 255))
    x = 0
    for i, t in enumerate(tiles):
        if i == sep_at:
            x += gap
        row.paste(t, (x, 20))
        x += cell
    draw = ImageDraw.Draw(row)
    prompt_preview = task["prompt"].replace("\n", " ")[:70]
    draw.text((4, 3), f'{task["name"]}  |  "{prompt_preview}"', fill=(0, 0, 0), font=_font(14))
    if sep_at:
        lx = sep_at * cell + gap // 2
        draw.line([(lx, 20), (lx, row.height)], fill=(200, 0, 0), width=2)
    return row


# ---------------------------------------------------------------- LoRA 切换(复制自 smoke)

def swap_lora(model, state_dict: dict, tag: str) -> None:
    """就地替换 LoRA 权重,硬校验 key 真的匹配上——对不上就 raise,绝不静默失效。"""
    model_sd = model.state_dict()
    unexpected = [k for k in state_dict if k not in model_sd]
    if unexpected:
        raise SystemExit(
            f"❌ [{tag}] checkpoint 有 {len(unexpected)} 个 key 在模型中不存在,"
            f"加载会静默失效。例如:{unexpected[:3]}")
    dev = next(model.parameters()).device
    aligned = {k: v.to(device=dev, dtype=model_sd[k].dtype) for k, v in state_dict.items()}
    model.load_state_dict(aligned, strict=False, assign=True)


# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/multibanana",
                   help="download_multibanana.py 下载到的目录")
    p.add_argument("--lora_path", default="log/ref_isolation/checkpoint-7000/dit_lora.safetensors")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    p.add_argument("--ref_size", type=int, default=512,
                   help="参考图长边。默认 512:上一轮 ckpt7000 对照实验里 512 的多主体保留"
                        "明显好于 320(训练 resolution_ref 未设、ref 按 512 分桶,512 对齐)")
    p.add_argument("--max_refs", type=int, default=4,
                   help="只处理参考图数 <= 此值的任务;超的跳过(显存 + 超训练分布)。"
                        "MultiBanana 最多 8 参考,4 已接近 UNO 常用上限")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--lora_rank", type=int, default=512)
    # 远程 24G 卡实测用 fp8(bf16 DiT 22.4G 余量太小),故默认 fp8;有大卡可换 flux-dev
    p.add_argument("--model_type", default="flux-dev-fp8", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write_generated", action=argparse.BooleanOptionalAction, default=True,
                   help="把主变体结果写回任务目录 <num>_generated.jpg,供 MultiBanana judge.py 评分")
    p.add_argument("--save_path", default="output/multibanana_eval")
    p.add_argument("--dry_run", action="store_true", help="不加载模型,用占位图跑通流程")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"❌ 数据目录不存在:{args.data_dir}\n   先跑 download_multibanana.py")
    tasks = discover_tasks(args.data_dir)
    if not tasks:
        raise SystemExit(f"❌ {args.data_dir} 下没发现任务(找不到 *_prompt.txt)")
    skipped = [t["name"] for t in tasks if len(t["image_paths"]) > args.max_refs]
    tasks = [t for t in tasks if len(t["image_paths"]) <= args.max_refs]
    if skipped:
        print(f"⚠️  跳过 {len(skipped)} 个参考图 > {args.max_refs} 的任务:{skipped[:5]}"
              f"{' …' if len(skipped) > 5 else ''}(--max_refs 可调)")
    if not tasks:
        raise SystemExit(f"❌ 没有参考图数 <= {args.max_refs} 的任务可跑")
    os.makedirs(args.save_path, exist_ok=True)
    print(f"任务 {len(tasks)} 个 × 变体 {len(args.variants)} 个 = {len(tasks) * len(args.variants)} 次生成")

    # 主变体:优先 ours_kv,否则取列表第一个。它的结果写回 <num>_generated.jpg。
    primary = "ours_kv" if "ours_kv" in args.variants else args.variants[0]

    # ---------- 初始化 ----------
    if args.dry_run:
        pipeline = preprocess_ref = None
    else:
        import torch
        from uno.flux.pipeline import UNOPipeline, preprocess_ref
        import uno.flux.pipeline as pipeline_mod
        from safetensors.torch import load_file

        if not os.path.exists(args.lora_path):
            raise SystemExit(f"❌ LoRA 不存在:{args.lora_path}")

        # 只给 denoise 循环计时(offload 每次搬运整个 DiT,与注意力结构无关,混进来会掩盖加速比)
        _orig_denoise = pipeline_mod.denoise
        denoise_times: list[float] = []

        def timed_denoise(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _orig_denoise(*a, **kw)
            torch.cuda.synchronize()
            denoise_times.append(time.perf_counter() - t0)
            return out

        pipeline_mod.denoise = timed_denoise

        pipeline = UNOPipeline(args.model_type, torch.device("cuda"), offload=args.offload,
                               only_lora=True, lora_rank=args.lora_rank)
        # __init__ 把 t5/clip 放 GPU 且只在 forward 里 offload,首个 forward 搬 DiT 上卡时
        # 它们还占着约 10G。先踢回 CPU 腾地方。
        if args.offload:
            pipeline.t5.cpu()
            pipeline.clip.cpu()
            torch.cuda.empty_cache()

        ours_sd = load_file(args.lora_path, device="cpu")
        if not ours_sd:
            raise SystemExit(f"❌ {args.lora_path} 是空的(大概率被 kill 时写坏的残档)")
        model_sd = pipeline.model.state_dict()
        unknown = [k for k in ours_sd if k not in model_sd]
        if unknown:
            raise SystemExit(f"❌ checkpoint key 与模型不匹配,例如 {unknown[:3]}")
        official_sd = {k: model_sd[k].detach().clone().cpu() for k in ours_sd}
        print(f"LoRA:{len(ours_sd)} 个张量,已备份官方权重用于对照")

    # ---------- 生成(外层按变体循环,切 LoRA 只搬一次权重) ----------
    all_results: dict[str, dict] = {t["name"]: {} for t in tasks}
    task_times: dict[str, dict] = {t["name"]: {} for t in tasks}
    timing: dict[str, dict] = {}
    current_lora = None
    warmed = False

    for variant in args.variants:
        use_ours, ref_isolation, kv_cache = VARIANTS[variant]
        if not args.dry_run and current_lora != use_ours:
            swap_lora(pipeline.model, ours_sd if use_ours else official_sd, variant)
            current_lora = use_ours

        per_task = []
        for task in tasks:
            refs = [Image.open(pth) for pth in task["image_paths"]]
            n_refs = len(refs)
            if args.dry_run:
                img, t_denoise, t_e2e = Image.new("RGB", (args.width, args.height),
                                                  (200, 220, 240)), 0.0, 0.0
            else:
                ref_imgs = [preprocess_ref(r, args.ref_size) for r in refs]

                def run():
                    return pipeline(
                        prompt=task["prompt"], width=args.width, height=args.height,
                        guidance=args.guidance, num_steps=args.num_steps, seed=args.seed,
                        ref_imgs=ref_imgs, pe="d",
                        ref_isolation=ref_isolation, kv_cache=kv_cache,
                    )

                if not warmed:
                    print("warmup …")
                    try:
                        run()
                    except torch.OutOfMemoryError:
                        raise SystemExit(
                            f"❌ 显存不足。改用 fp8 或调小 --ref_size / --max_refs:\n"
                            f"   python multibanana_eval/infer_multibanana.py "
                            f"--lora_path {args.lora_path} --model_type flux-dev-fp8")
                    warmed = True

                denoise_times.clear()
                t0 = time.perf_counter()
                img = run()
                t_e2e = time.perf_counter() - t0
                t_denoise = denoise_times[-1]

            all_results[task["name"]][variant] = img
            task_times[task["name"]][variant] = t_denoise
            per_task.append({"task": task["name"], "n_refs": n_refs,
                             "denoise_s": t_denoise, "e2e_s": t_e2e})
            # 原始单图存到 save_path(自包含,不污染下载目录)
            img.save(os.path.join(args.save_path, f'{task["name"]}__{variant}.png'))
            # 主变体额外写回任务目录,遵循 MultiBanana 约定,供其 judge.py 评分
            if args.write_generated and variant == primary and not args.dry_run:
                img.save(os.path.join(task["dir"], f'{task["number"]}_generated.jpg'))
            print(f"  [{variant}] {task['name']:<24} {n_refs}ref  "
                  f"denoise {t_denoise:6.2f}s  e2e {t_e2e:6.2f}s")

        timing[variant] = {
            "per_task": per_task,
            "denoise_mean_s": statistics.mean(x["denoise_s"] for x in per_task),
            "denoise_median_s": statistics.median(x["denoise_s"] for x in per_task),
            "e2e_mean_s": statistics.mean(x["e2e_s"] for x in per_task),
        }
        if not args.dry_run:
            import torch
            timing[variant]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

    # ---------- 汇总 ----------
    rows = []
    for task in tasks:
        refs = [Image.open(pth) for pth in task["image_paths"]]
        row = make_comparison(task, refs, all_results[task["name"]], task_times[task["name"]])
        row.save(os.path.join(args.save_path, f'compare__{task["name"]}.png'))
        rows.append(row)
    width = max(r.width for r in rows)
    board = Image.new("RGB", (width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        board.paste(r, (0, y))
        y += r.height
    board_path = os.path.join(args.save_path, "ALL_COMPARISON.png")
    board.save(board_path)

    print("\n" + "=" * 68)
    print(f"{'变体':<16}{'denoise均值':>12}{'denoise中位':>12}{'e2e均值':>10}{'vs baseline':>14}")
    print("-" * 68)
    base = timing.get("official_full", {}).get("denoise_mean_s")
    for variant, t in timing.items():
        speedup = f"{base / t['denoise_mean_s']:.2f}x" if base and t["denoise_mean_s"] else "-"
        print(f"{variant:<16}{t['denoise_mean_s']:>11.2f}s{t['denoise_median_s']:>11.2f}s"
              f"{t['e2e_mean_s']:>9.2f}s{speedup:>14}")
    print("=" * 68)

    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump({"config": vars(args), "timing": timing,
                   "tasks": [{k: t[k] for k in ("name", "prompt", "image_paths")} for t in tasks]},
                  f, indent=2, ensure_ascii=False)

    print(f"\n对比总览:{board_path}")
    print(f"单例对比:{args.save_path}/compare__*.png")
    if args.write_generated and not args.dry_run:
        print(f"已写回 <num>_generated.jpg 到各任务目录,可跑 MultiBanana 的 judge.py 打分")


if __name__ == "__main__":
    main()
