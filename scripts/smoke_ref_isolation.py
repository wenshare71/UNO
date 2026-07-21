"""隔离注意力 LoRA 冒烟实验：对比 full attention 与 kv_cache 的「速度 + 质量」。

跑三个变体（同 prompt / 同 seed / 同参考图，逐例对齐）：
  official_full : 官方 UNO LoRA + 全注意力   —— 质量金标准 & 速度基线
  ours_full     : 我们的 LoRA  + 全注意力   —— 诊断项，见下
  ours_kv       : 我们的 LoRA  + KV-Cache   —— 待验证的方案

为什么要有 ours_full 这个看似没用的组合：它能把「质量差」的两种原因分开。
  ours_kv 差 + ours_full 好  → 隔离注意力/缓存这套结构有问题
  ours_kv ≈ ours_full 都差   → 结构没问题，只是训练步数不够（2000 步本来就早期）
少了它，看到糊图无法判断该继续训还是该改结构。

计时只统计 denoise 循环，不含 offload 搬运：
  pipeline 在 offload 模式下每次生成都要把 DiT 在 CPU↔GPU 之间搬一个来回（十几 GB，
  好几秒），而隔离注意力省下的只是 attention 的计算量。把搬运算进去会把加速比稀释到
  看不出来。end-to-end 时间同时也记录，但主指标看 denoise。

用法（远程单卡即可，别用 accelerate launch）：
  python scripts/smoke_ref_isolation.py --lora_path log/ref_isolation/checkpoint-2000/dit_lora.safetensors
  python scripts/smoke_ref_isolation.py --dry_run     # 不用 GPU，只验流程/拼图/例子文件是否齐全
"""
import argparse
import json
import os
import statistics
import sys
import time

# 必须在 import torch 之前设置才生效。24 GiB 卡放 22.4 GiB 的 DiT，余量全靠碎片管理，
# 上次那次 OOM 的报错里 PyTorch 自己就建议了这个——设成默认值省得用户每次记着导出。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 让脚本不管从哪个目录运行都能找到 uno 包（uno/ 缺 __init__.py，走 namespace fallback）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 变体定义

# name: (用我们的 LoRA?, ref_isolation, kv_cache)
VARIANTS = {
    "official_full": (False, False, False),
    "ours_full":     (True,  False, False),
    "ours_kv":       (True,  True,  True),
    # 可选：不开 cache 的纯隔离注意力。与 ours_kv 应逐像素接近（cache 只是省重算），
    # 用来验证缓存实现本身没写错；默认不跑，加 --variants 里带上才跑。
    "ours_iso":      (True,  True,  False),
}
DEFAULT_VARIANTS = ["official_full", "ours_full", "ours_kv"]

# ---------------------------------------------------------------- 测试例子

# 内置例子用仓库自带的 assets，不依赖任何 submodule，保证一定能跑。
# 覆盖 1/2/3 张参考图——参考图越多，隔离注意力省掉的 attention 计算越多，
# 加速比应当越明显，所以这个维度必须铺开测。
BUILTIN_CASES = [
    {
        "name": "figurine_in_ball",
        "prompt": "The figurine is in the crystal ball",
        "image_paths": ["assets/examples/3two2one/ref1.png", "assets/examples/3two2one/ref2.png"],
    },
    {
        "name": "logo_on_cup",
        "prompt": "The logo is printed on the cup",
        "image_paths": ["assets/examples/4two2one/ref1.png", "assets/examples/4two2one/ref2.png"],
    },
    {
        "name": "clock_and_cup",
        "prompt": "a clock and a cup on a wooden table, product photo, soft daylight",
        "image_paths": ["assets/clock.png", "assets/cup.png"],
    },
    {
        "name": "figurine_and_clock_street",
        "prompt": "The figurine stands next to the clock on a cobblestone street in Paris",
        "image_paths": ["assets/figurine.png", "assets/clock.png"],
    },
    {
        "name": "dress_bag_flowers",
        "prompt": "A woman wears the dress and holds a bag, in the flowers.",
        "image_paths": [
            "assets/examples/5many2one/ref1.png",
            "assets/examples/5many2one/ref2.png",
            "assets/examples/5many2one/ref3.png",
        ],
    },
    {
        "name": "single_clock_beach",  # 单参考对照组：kv_cache 能省的最少，加速比应最小
        "prompt": "A clock on the beach is under a red sun umbrella",
        "image_paths": ["assets/examples/1one2one/ref1.jpg"],
    },
]


def load_dreambench_cases(n: int) -> list[dict]:
    """从 dreambench_multiip.json 挑 n 组「主体组合互不相同」的双参考例子。

    这些是真实的 benchmark 数据（不是 UNO 自家 demo 图），避免只在官方精选样例上看着好。
    依赖 datasets/dreambooth 这个 submodule，没 init 就静默跳过——冒烟实验不该因为
    可选数据缺失而整个跑不起来。
    """
    json_path = "datasets/dreambench_multiip.json"
    if n <= 0 or not os.path.exists(json_path):
        return []
    with open(json_path, "rt") as f:
        data = json.load(f)

    root = os.path.dirname(json_path)
    cases, seen = [], set()
    for item in data:
        combo = tuple(item["image_paths"])
        if combo in seen:
            continue
        paths = [os.path.normpath(os.path.join(root, p)) for p in item["image_paths"]]
        if not all(os.path.exists(p) for p in paths):
            continue  # submodule 没 init
        seen.add(combo)
        subjects = "_".join(os.path.basename(os.path.dirname(p)) for p in paths)
        cases.append({"name": f"db_{subjects}", "prompt": item["prompt"], "image_paths": paths})
        if len(cases) >= n:
            break
    return cases


# ---------------------------------------------------------------- 拼图

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _labeled(img: Image.Image, text: str, cell: int) -> Image.Image:
    """把图缩放到 cell 见方（保持比例，居中留白）并在顶部加标签条。"""
    bar = 24
    canvas = Image.new("RGB", (cell, cell + bar), (255, 255, 255))
    w, h = img.size
    scale = min(cell / w, cell / h)
    resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas.paste(resized, ((cell - resized.width) // 2, bar + (cell - resized.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), text[:38], fill=(0, 0, 0), font=_font(14))
    return canvas


def make_comparison(case: dict, refs: list[Image.Image], results: dict,
                    times: dict | None = None, cell: int = 256) -> Image.Image:
    """一行：[ref1..refN | 各变体生成结果]，参考图和结果之间留一条分隔线。

    变体标签直接带上该例的 denoise 耗时——看图时质量和速度得同时在眼前，
    否则要在图和表格之间来回对照。
    """
    tiles = [_labeled(r, f"ref{i + 1}", cell) for i, r in enumerate(refs)]
    for name, img in results.items():
        t = (times or {}).get(name)
        tiles.append(_labeled(img, f"{name}  {t:.1f}s" if t else name, cell))

    gap = 8
    sep_at = len(refs)
    width = len(tiles) * cell + gap
    row = Image.new("RGB", (width, cell + 24 + 20), (255, 255, 255))
    x = 0
    for i, t in enumerate(tiles):
        if i == sep_at:
            x += gap
        row.paste(t, (x, 20))
        x += cell
    draw = ImageDraw.Draw(row)
    draw.text((4, 3), f'{case["name"]}  |  "{case["prompt"]}"', fill=(0, 0, 0), font=_font(14))
    if sep_at:  # 分隔线：左边是输入，右边是输出
        lx = sep_at * cell + gap // 2
        draw.line([(lx, 20), (lx, row.height)], fill=(200, 0, 0), width=2)
    return row


# ---------------------------------------------------------------- LoRA 切换

def swap_lora(model, state_dict: dict, tag: str) -> None:
    """就地替换 LoRA 权重，并硬校验 key 真的匹配上了。

    这里必须硬失败而不是打印警告：如果 checkpoint 的 key 和推理端模型对不上
    （比如训练侧残留 "module." 前缀），load_state_dict(strict=False) 会安安静静地
    什么都不改，模型里还是官方权重——那样整个实验对比的就是「官方 vs 官方」，
    结论完全错误且很难看出来。
    """
    model_sd = model.state_dict()
    unexpected = [k for k in state_dict if k not in model_sd]
    if unexpected:
        raise SystemExit(
            f"❌ [{tag}] checkpoint 里有 {len(unexpected)} 个 key 在模型中不存在，"
            f"key 命名对不上，加载会静默失效。\n   例如: {unexpected[:3]}\n"
            f"   模型侧 LoRA key 形如: {[k for k in model_sd if 'lora' in k][:2]}"
        )
    dev = next(model.parameters()).device
    # dtype 对齐：fp8 模式下主干权重会被量化，LoRA 层通常仍是 bf16，
    # 直接 assign 一个 dtype 不同的张量会让后续 matmul 报类型错。
    aligned = {k: v.to(device=dev, dtype=model_sd[k].dtype) for k, v in state_dict.items()}
    model.load_state_dict(aligned, strict=False, assign=True)


# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_path", default="log/ref_isolation/checkpoint-2000/dit_lora.safetensors")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    p.add_argument("--num_dreambench", type=int, default=4, help="额外从 dreambench 取几个例子（0=不取）")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--lora_rank", type=int, default=512)
    p.add_argument("--ref_size", type=int, default=-1,
                   help="参考图长边；-1 = 沿用官方惯例（单图 512 / 多图 320）。"
                        "注意我们训练时 resolution_ref 没设，ref 是按 512 分桶的，"
                        "多主体质量若明显偏差，值得用 --ref_size 512 再跑一遍看是不是尺度不一致导致")
    p.add_argument("--save_path", default="output/smoke_ref_isolation")
    # 默认 bf16：fp8 只省显存不提速（uno/flux 各层没有 _scaled_mm，autocast 每次把 fp8
    # 权重转回 bf16 再算），那份固定开销会把 kv_cache 的加速比稀释得偏小。
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry_run", action="store_true", help="不加载模型，用占位图跑通流程")
    args = p.parse_args()

    cases = BUILTIN_CASES + load_dreambench_cases(args.num_dreambench)
    missing = [pth for c in cases for pth in c["image_paths"] if not os.path.exists(pth)]
    if missing:
        raise SystemExit(f"❌ 参考图缺失: {missing[:5]}\n   请在 UNO 仓库根目录运行本脚本")
    os.makedirs(args.save_path, exist_ok=True)
    print(f"例子 {len(cases)} 个 × 变体 {len(args.variants)} 个 = {len(cases) * len(args.variants)} 次生成")

    # ---------- 初始化 ----------
    if args.dry_run:
        pipeline = preprocess_ref = None
    else:
        import torch
        from uno.flux.pipeline import UNOPipeline, preprocess_ref
        import uno.flux.pipeline as pipeline_mod
        from safetensors.torch import load_file

        if not os.path.exists(args.lora_path):
            raise SystemExit(f"❌ LoRA 不存在: {args.lora_path}")

        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if args.model_type == "flux-dev" and total_gb < 26:
            # 24 GiB 卡跑 bf16 DiT（权重 22.4 GiB）余量只有 1 GiB 出头，能不能站住取决于
            # 序列长度。上次 bench 在这里 OOM 的直接原因是 T5/CLIP 还占着显存（下面已修），
            # 所以这次不硬拦，让 warmup 去试——真炸了那里会给出回退命令。
            print(f"⚠️  单卡 {total_gb:.1f} GiB 跑 bf16 DiT(22.4 GiB) 余量很小，"
                  f"若 warmup 处 OOM 请改用 --model_type flux-dev-fp8")

        # 只给 denoise 循环计时。offload 模式下 pipeline 每次调用都要搬运整个 DiT，
        # 那部分开销与注意力结构无关，混进来会掩盖真实加速比。
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
        # UNOPipeline.__init__ 把 t5/clip 直接放 GPU 且只在 forward 里才 offload，
        # 首次 forward 搬 DiT 上卡时它们还占着显存。先踢回 CPU，给 DiT 腾地方。
        if args.offload:
            pipeline.t5.cpu()
            pipeline.clip.cpu()
            torch.cuda.empty_cache()

        ours_sd = load_file(args.lora_path, device="cpu")
        if not ours_sd:
            raise SystemExit(f"❌ {args.lora_path} 是空的（大概率是被 kill 时写坏的残档）")
        # 备份官方 LoRA，用于在变体间来回切换（只存 ckpt 涉及的那些 key）
        model_sd = pipeline.model.state_dict()
        unknown = [k for k in ours_sd if k not in model_sd]
        if unknown:
            raise SystemExit(f"❌ checkpoint key 与模型不匹配，例如 {unknown[:3]}")
        official_sd = {k: model_sd[k].detach().clone().cpu() for k in ours_sd}
        print(f"LoRA: {len(ours_sd)} 个张量，已备份官方权重用于对照")

    def ref_size_for(n_refs: int) -> int:
        return args.ref_size if args.ref_size > 0 else (512 if n_refs == 1 else 320)

    # ---------- 生成 ----------
    # 外层按变体循环：切换 LoRA 要搬一遍权重，按例子循环会切 N 倍次数。
    all_results: dict[str, dict] = {c["name"]: {} for c in cases}
    case_times: dict[str, dict] = {c["name"]: {} for c in cases}
    timing: dict[str, dict] = {}
    current_lora = None
    warmed = False

    for variant in args.variants:
        use_ours, ref_isolation, kv_cache = VARIANTS[variant]
        if not args.dry_run and current_lora != use_ours:
            swap_lora(pipeline.model, ours_sd if use_ours else official_sd, variant)
            current_lora = use_ours

        per_case = []
        for case in cases:
            refs = [Image.open(pth) for pth in case["image_paths"]]
            if args.dry_run:
                img, t_denoise, t_e2e = Image.new("RGB", (args.width, args.height),
                                                  (200, 220, 240)), 0.0, 0.0
            else:
                size = ref_size_for(len(refs))
                ref_imgs = [preprocess_ref(r, size) for r in refs]

                def run():
                    return pipeline(
                        prompt=case["prompt"], width=args.width, height=args.height,
                        guidance=args.guidance, num_steps=args.num_steps, seed=args.seed,
                        ref_imgs=ref_imgs, pe="d",
                        ref_isolation=ref_isolation, kv_cache=kv_cache,
                    )

                if not warmed:
                    # 首次前向含 kernel 编译/权重换入，计进去会把第一个变体污染成最慢的
                    print("warmup ...")
                    try:
                        run()
                    except torch.OutOfMemoryError:
                        raise SystemExit(
                            "❌ 显存不足。24 GiB 卡放 bf16 DiT(22.4 GiB) 余量太小。\n"
                            "   改用: python scripts/smoke_ref_isolation.py "
                            f"--lora_path {args.lora_path} --model_type flux-dev-fp8\n"
                            "   （fp8 三个变体条件一致，相对比较仍然成立，只是加速比会偏保守）"
                        )
                    warmed = True

                denoise_times.clear()
                t0 = time.perf_counter()
                img = run()
                t_e2e = time.perf_counter() - t0
                t_denoise = denoise_times[-1]

            all_results[case["name"]][variant] = img
            case_times[case["name"]][variant] = t_denoise
            per_case.append({"case": case["name"], "n_refs": len(refs),
                             "denoise_s": t_denoise, "e2e_s": t_e2e})
            img.save(os.path.join(args.save_path, f'{case["name"]}__{variant}.png'))
            print(f"  [{variant}] {case['name']:<28} denoise {t_denoise:6.2f}s  e2e {t_e2e:6.2f}s")

        timing[variant] = {
            "per_case": per_case,
            "denoise_mean_s": statistics.mean(x["denoise_s"] for x in per_case),
            "denoise_median_s": statistics.median(x["denoise_s"] for x in per_case),
            "e2e_mean_s": statistics.mean(x["e2e_s"] for x in per_case),
        }
        if not args.dry_run:
            import torch
            timing[variant]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

    # ---------- 汇总 ----------
    rows = []
    for case in cases:
        refs = [Image.open(pth) for pth in case["image_paths"]]
        row = make_comparison(case, refs, all_results[case["name"]], case_times[case["name"]])
        row.save(os.path.join(args.save_path, f'compare__{case["name"]}.png'))
        rows.append(row)

    width = max(r.width for r in rows)
    board = Image.new("RGB", (width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        board.paste(r, (0, y))
        y += r.height
    board_path = os.path.join(args.save_path, "ALL_COMPARISON.png")
    board.save(board_path)

    print("\n" + "=" * 74)
    print(f"{'变体':<16}{'denoise均值':>12}{'denoise中位':>12}{'e2e均值':>10}{'vs baseline':>14}")
    print("-" * 74)
    base = timing.get("official_full", {}).get("denoise_mean_s")
    for variant, t in timing.items():
        speedup = f"{base / t['denoise_mean_s']:.2f}x" if base and t["denoise_mean_s"] else "-"
        print(f"{variant:<16}{t['denoise_mean_s']:>11.2f}s{t['denoise_median_s']:>11.2f}s"
              f"{t['e2e_mean_s']:>9.2f}s{speedup:>14}")
    print("=" * 74)

    # 计时可信度自检：official_full 和 ours_full 结构完全相同（同 rank 的 LoRA，只是权重
    # 数值不同），denoise 耗时理应几乎一致。两者差得多就说明这台机器上的计时被别的负载
    # 干扰了（比如另一张卡在跑训练、或 offload 搬运抢 PCIe），此时加速比数字不能信。
    if "official_full" in timing and "ours_full" in timing:
        a = timing["official_full"]["denoise_mean_s"]
        b = timing["ours_full"]["denoise_mean_s"]
        drift = abs(a - b) / max(a, b) if max(a, b) else 0
        verdict = "✅ 计时稳定" if drift < 0.05 else "⚠️  计时受干扰，加速比仅供参考"
        print(f"\n计时自检: official_full vs ours_full 相差 {drift:.1%}（同结构，应 <5%）{verdict}")

    # 按参考图数量拆开看：隔离注意力省掉的是 ref 之间/ref 对主图的 attention，
    # ref 越多省得越多，这个趋势本身就是方案是否按预期工作的证据。
    if base:
        print("\n按参考图数量的加速比（denoise）：")
        n_refs_set = sorted({x["n_refs"] for x in timing["official_full"]["per_case"]})
        for n in n_refs_set:
            line = f"  {n} 张 ref: "
            for variant, t in timing.items():
                if variant == "official_full":
                    continue
                b = statistics.mean(x["denoise_s"] for x in timing["official_full"]["per_case"]
                                    if x["n_refs"] == n)
                v = statistics.mean(x["denoise_s"] for x in t["per_case"] if x["n_refs"] == n)
                line += f"{variant} {b / v:.2f}x   " if v else ""
            print(line)

    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump({"config": vars(args), "timing": timing,
                   "cases": [{k: c[k] for k in ("name", "prompt", "image_paths")} for c in cases]},
                  f, indent=2, ensure_ascii=False)

    print(f"\n对比总览: {board_path}")
    print(f"单例对比: {args.save_path}/compare__*.png")
    print(f"原始结果: {args.save_path}/<case>__<variant>.png + results.json")


if __name__ == "__main__":
    main()
