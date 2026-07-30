"""M4 冒烟评测:蒸馏前 vs 蒸馏后 vs teacher 三方对比(单张拼图)。

目的:用最小成本肉眼判断 M3 蒸馏是否让隔离注意力 LoRA 在多参考图上少丢主体。
做法:从 dreambench_multiip.json 的 held-out 2-ref 组合里确定性挑几条,每条用同一
seed 跑三个变体并排拼图——
  - official_full:官方 UNO + 全注意力(teacher 金标准,蒸馏数据就是它生成的)
  - ours_kv_pre  :我们的 LoRA(ckpt-20000,蒸馏前)+ 隔离注意力 + KV cache
  - ours_kv_post :我们的 LoRA(ckpt-4000,蒸馏后)+ 隔离注意力 + KV cache
若蒸馏有效,ours_kv_post 应比 ours_kv_pre 更接近 official_full(两个主体都在)。

只读:不改任何训练产物、不动 train_mixed.json、不 commit。单卡 bf16 不 offload
(H800 143 GB 余量充足,与计划 §D-1 teacher bf16 一致)。

用法:
  python distill/smoke_eval.py --dry_run    # 不碰 GPU,验证任务发现/拼图/落盘
  python distill/smoke_eval.py              # 默认 5 个 held-out 2-ref 案例
  python distill/smoke_eval.py --n_cases 10
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "multibanana_eval"))

from PIL import Image

import board  # noqa: E402  (复用 multibanana_eval/board.py 的拼图)

# held-out 10 个(DISTILL_PLAN §2,严禁进蒸馏数据,这里用作评测)
HELD_OUT = {
    "backpack_dog", "bear_plushie", "berry_bowl", "can", "candle",
    "clock", "colorful_sneaker", "duck_toy", "fancy_boot", "grey_sloth_plushie",
}

# 默认 5 条:每个 held-out 2-ref 组合各取一条,场景错开便于肉眼区分。
DEFAULT_CASES = [
    {"prompt": "a backpack and a stuffed animal in the jungle",
     "image_paths": ["./dreambooth/dataset/backpack_dog/02.jpg",
                     "./dreambooth/dataset/bear_plushie/03.jpg"]},
    {"prompt": "a bowl and a can on the beach",
     "image_paths": ["./dreambooth/dataset/berry_bowl/02.jpg",
                     "./dreambooth/dataset/can/01.jpg"]},
    {"prompt": "a candle and a clock in the snow",
     "image_paths": ["./dreambooth/dataset/candle/02.jpg",
                     "./dreambooth/dataset/clock/03.jpg"]},
    {"prompt": "a sneaker and a toy in the jungle",
     "image_paths": ["./dreambooth/dataset/colorful_sneaker/01.jpg",
                     "./dreambooth/dataset/duck_toy/01.jpg"]},
    {"prompt": "a boot and a stuffed animal on the beach",
     "image_paths": ["./dreambooth/dataset/fancy_boot/02.jpg",
                     "./dreambooth/dataset/grey_sloth_plushie/04.jpg"]},
]

# 变体:(标签, 用我们的LoRA?, ref_isolation, kv_cache, LoRA bank)
# LoRA bank: "official"=官方权重备份, "pre"=蒸馏前 ckpt, "post"=蒸馏后 ckpt
VARIANTS = [
    ("official_full", False, False, False, "official"),
    ("ours_kv_pre",    True,  True,  True, "pre"),
    ("ours_kv_post",   True,  True,  True, "post"),
]


def pick_cases(data_json: str, n: int) -> list[dict]:
    """从 dreambench_multiip.json 选 n 条 held-out 2-ref 案例,确定性。

    优先取 DEFAULT_CASES(已人工挑好场景错开);不足再从 json 里按顺序补,
    保证每个组合不连续出现。
    """
    cases = list(DEFAULT_CASES)
    if n <= len(cases):
        return cases[:n]
    # 需要更多:从 json 补 held-out 2-ref,跳过已选
    data = json.load(open(data_json))
    seen = {(c["prompt"], tuple(c["image_paths"])) for c in cases}

    def subj(p):
        return p.split("/")[-2]

    extra = []
    for x in data:
        if len(x["image_paths"]) != 2:
            continue
        if not all(subj(p) in HELD_OUT for p in x["image_paths"]):
            continue
        key = (x["prompt"], tuple(x["image_paths"]))
        if key in seen:
            continue
        seen.add(key)
        extra.append(x)
        if len(cases) + len(extra) >= n:
            break
    return cases + extra[:n - len(cases)]


def resolve_path(p: str, data_dir: str) -> str:
    """dreambench_multiip.json 的路径以 ./dreambooth/... 开头,相对 datasets/ 目录。"""
    if p.startswith("./"):
        p = p[2:]
    return os.path.join(data_dir, p)


def swap_lora(model, state_dict: dict, tag: str) -> None:
    """就地替换 LoRA 权重,硬校验 key 匹配(复制自 infer_multibanana.py:91)。"""
    model_sd = model.state_dict()
    unexpected = [k for k in state_dict if k not in model_sd]
    if unexpected:
        raise SystemExit(
            f"❌ [{tag}] checkpoint 有 {len(unexpected)} 个 key 在模型中不存在,"
            f"加载会静默失效。例如:{unexpected[:3]}")
    dev = next(model.parameters()).device
    aligned = {k: v.to(device=dev, dtype=model_sd[k].dtype) for k, v in state_dict.items()}
    model.load_state_dict(aligned, strict=False, assign=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pre_lora", default="log/ref_isolation/checkpoint-20000/dit_lora.safetensors",
                   help="蒸馏前 LoRA(隔离注意力续训起点)")
    p.add_argument("--post_lora", default="log/ref_distill/checkpoint-4000/dit_lora.safetensors",
                   help="蒸馏后 LoRA(M3 产物)")
    p.add_argument("--data_json", default="datasets/dreambench_multiip.json")
    p.add_argument("--n_cases", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--ref_size", type=int, default=512)
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"],
                   help="H800 用 bf16(flux-dev);4090 才用 fp8")
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False,
                   help="H800 143GB 无需 offload,默认关")
    p.add_argument("--lora_rank", type=int, default=512)
    p.add_argument("--save_path", default="output/smoke_eval")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    data_dir = os.path.dirname(os.path.abspath(args.data_json))
    cases = pick_cases(args.data_json, args.n_cases)
    # 启动断言:所有 case 必须是 held-out(防蒸馏数据泄漏到评测)
    for c in cases:
        for pth in c["image_paths"]:
            subj = pth.split("/")[-2]
            assert subj in HELD_OUT, f"❌ {subj} 不是 held-out,评测集泄漏蒸馏数据!"
    print(f"案例 {len(cases)} 个 × 变体 {len(VARIANTS)} 个 = "
          f"{len(cases) * len(VARIANTS)} 次生成")
    os.makedirs(args.save_path, exist_ok=True)

    # ---------- 初始化 ----------
    if args.dry_run:
        pipeline = preprocess_ref = None
        lora_banks = {}
    else:
        import torch
        from uno.flux.pipeline import UNOPipeline, preprocess_ref
        from safetensors.torch import load_file

        for lp in (args.pre_lora, args.post_lora):
            if not os.path.exists(lp):
                raise SystemExit(f"❌ LoRA 不存在:{lp}")

        pipeline = UNOPipeline(args.model_type, torch.device("cuda"), offload=args.offload,
                               only_lora=True, lora_rank=args.lora_rank)
        if args.offload:  # 把 t5/clip 踢回 CPU 给 DiT 腾地方(非 offload 模式无需)
            pipeline.t5.cpu()
            pipeline.clip.cpu()
            torch.cuda.empty_cache()

        model_sd = pipeline.model.state_dict()
        # 三个 LoRA bank:official=模型当前已加载的官方 LoRA 备份,pre/post=两个 ckpt
        official_sd = {k: model_sd[k].detach().clone().cpu() for k in model_sd
                       if any(k.endswith(s) for s in (".lora_A.weight", ".lora_B.weight",
                                                      ".lora_A.default.weight", ".lora_B.default.weight"))}
        if not official_sd:
            # only_lora 模式下官方 LoRA 已挂载,直接备份全部可训练 key 兜底
            official_sd = {k: v.detach().clone().cpu() for k, v in model_sd.items()
                           if "lora" in k.lower()}
        if not official_sd:
            raise SystemExit("❌ 没找到 LoRA key,无法备份官方权重做对照")
        pre_sd = load_file(args.pre_lora, device="cpu")
        post_sd = load_file(args.post_lora, device="cpu")
        for sd, tag in [(pre_sd, "pre"), (post_sd, "post")]:
            unknown = [k for k in sd if k not in model_sd]
            if unknown:
                raise SystemExit(f"❌ [{tag}] checkpoint key 与模型不匹配,例如 {unknown[:3]}")
        lora_banks = {"official": official_sd, "pre": pre_sd, "post": post_sd}
        print(f"LoRA bank:official {len(official_sd)} / pre {len(pre_sd)} / post {len(post_sd)} 张量")

    # ---------- 生成(外层变体,切 LoRA 只搬一次) ----------
    all_results = {f"case{i}": {} for i in range(len(cases))}
    all_times = {f"case{i}": {} for i in range(len(cases))}
    timing = {}
    current_bank = None
    warmed = False

    for variant_name, use_ours, ref_iso, kv_cache, bank in VARIANTS:
        if not args.dry_run and current_bank != bank:
            swap_lora(pipeline.model, lora_banks[bank], variant_name)
            current_bank = bank

        per_case = []
        for i, case in enumerate(cases):
            ref_pths = [resolve_path(p, data_dir) for p in case["image_paths"]]
            if args.dry_run:
                img = Image.new("RGB", (args.width, args.height), (200, 220, 240))
                t_denoise = 0.0
            else:
                ref_imgs = [preprocess_ref(Image.open(rp), args.ref_size) for rp in ref_pths]

                def run():
                    return pipeline(
                        prompt=case["prompt"], width=args.width, height=args.height,
                        guidance=args.guidance, num_steps=args.num_steps, seed=args.seed,
                        ref_imgs=ref_imgs, pe="d",
                        ref_isolation=ref_iso, kv_cache=kv_cache,
                    )

                if not warmed:
                    print("warmup ...")
                    run()
                    warmed = True

                t0 = time.perf_counter()
                img = run()
                t_denoise = time.perf_counter() - t0

            key = f"case{i}"
            all_results[key][variant_name] = img
            all_times[key][variant_name] = t_denoise
            per_case.append(t_denoise)
            img.save(os.path.join(args.save_path, f"case{i:02d}__{variant_name}.png"))
            print(f"  [{variant_name}] case{i}  denoise {t_denoise:5.2f}s")

        timing[variant_name] = {
            "mean_s": statistics.mean(per_case) if per_case else 0.0,
            "median_s": statistics.median(per_case) if per_case else 0.0,
        }
        if not args.dry_run:
            import torch
            timing[variant_name]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

    # ---------- 拼图 ----------
    rows = []
    for i, case in enumerate(cases):
        ref_pths = [resolve_path(p, data_dir) for p in case["image_paths"]]
        refs = [Image.open(rp) for rp in ref_pths]
        key = f"case{i}"
        title = f"case{i}  ({'/'.join(p.split('/')[-2] for p in case['image_paths'])})"
        row = board.build_row(title, case["prompt"], refs,
                              all_results[key], all_times[key], cell=256)
        row.save(os.path.join(args.save_path, f"row_case{i:02d}.png"))
        rows.append(row)
    board_path = os.path.join(args.save_path, "smoke_compare.png")
    board.stack_board(rows).save(board_path)

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print(f"{'变体':<18}{'denoise均值':>14}{'中位':>10}{'peak GB':>10}")
    print("-" * 60)
    for vname, _, _, _, _ in VARIANTS:
        t = timing[vname]
        mem = t.get("peak_mem_gb", 0.0)
        print(f"{vname:<18}{t['mean_s']:>13.2f}s{t['median_s']:>9.2f}s{mem:>10.1f}")
    print("=" * 60)
    print(f"\n对比总览:{board_path}")
    print(f"单行对比:{args.save_path}/row_case*.png")

    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump({"config": vars(args), "timing": timing,
                   "cases": [{"prompt": c["prompt"], "image_paths": c["image_paths"]}
                             for c in cases]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
