"""multibanana 子集上 teacher vs student 对比(单张拼图)。

用途:在 multibanana 的某个子集(默认 `add`——主体合成)上,对每个任务用同一
seed 跑三个变体并排拼图,肉眼判断蒸馏是否让 student 在多主体合成上更接近 teacher:
  - official_full :官方 UNO + 全注意力(teacher 金标准)
  - ours_kv_pre   :我们的 LoRA(ckpt-20000,蒸馏前)+ 隔离注意力 + KV cache
  - ours_kv_post  :我们的 LoRA(ckpt-4000,蒸馏后)+ 隔离注意力 + KV cache

与 smoke_eval.py 的区别:任务发现用 multibanana 的目录格式(<num>_prompt.txt +
<num>_<i>.jpg),不走 dreambench_multiip.json。multibanana 的 ref 是真人/真物照片
(from_where.csv 标 real/generated),与蒸馏用的 dreambooth 玩具/动物分布不同,
结论按"泛化到 OOD 主体"读。

只读:不改任何训练产物、不 commit。单卡 bf16 不 offload(H800 143 GB 充足)。

用法:
  python distill/eval_multibanana.py --dry_run                 # 不碰 GPU
  python distill/eval_multibanana.py --subset add              # 默认 8 个任务
  python distill/eval_multibanana.py --subset add --n_tasks 16
  python distill/eval_multibanana.py --subset 3_back           # 3-ref 子集
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "multibanana_eval"))

from PIL import Image

import board  # noqa: E402  (复用 multibanana_eval/board.py)

# 默认从 `add` 子集挑 8 个任务,覆盖不同主体类型(人/动物/物品/虚构角色/动物+人)。
# 人工挑过,确保 prompt 句式和主体种类多样。
DEFAULT_ADD_TASKS = ["004", "009", "024", "064", "072", "092", "128", "143"]

# 变体:(标签, 用我们的LoRA?, ref_isolation, kv_cache, LoRA bank)
VARIANTS = [
    ("official_full", False, False, False, "official"),
    ("ours_kv_pre",    True,  True,  True, "pre"),
    ("ours_kv_post",   True,  True,  True, "post"),
]

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def discover_tasks(subset_dir: str, task_nums: list[str] | None) -> list[dict]:
    """发现 multibanana 子集里的任务。

    每个任务:<num>_prompt.txt + <num>_<i>.{jpg,png,...}。若给 task_nums,只取
    指定的(用于 DEFAULT_ADD_TASKS);否则全取。
    """
    tasks = []
    for fn in sorted(os.listdir(subset_dir)):
        m = re.match(r"(\d+)_prompt\.txt$", fn)
        if not m:
            continue
        num = m.group(1)
        if task_nums and num not in task_nums:
            continue
        with open(os.path.join(subset_dir, fn), "rt", encoding="utf-8") as f:
            prompt = f.read().strip()
        refs = []
        for cand in sorted(os.listdir(subset_dir)):
            rm = re.match(rf"{num}_(\d+)\.(\w+)$", cand)
            if rm and cand.lower().endswith(_IMG_EXTS):
                refs.append(os.path.join(subset_dir, cand))
        if not refs or not prompt:
            continue
        tasks.append({"name": f"{os.path.basename(subset_dir)}_{num}",
                      "dir": subset_dir, "number": num,
                      "prompt": prompt, "image_paths": refs})
    return tasks


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
    p.add_argument("--subset", default="add", help="multibanana 子集名(add/3_back/...)")
    p.add_argument("--data_root", default="data/multibanana")
    p.add_argument("--task_nums", nargs="+", default=None,
                   help="只跑指定任务号(如 004 009);默认用 DEFAULT_ADD_TASKS(add 子集)或全部")
    p.add_argument("--n_tasks", type=int, default=8,
                   help="没显式给 task_nums 时,取前 N 个任务")
    p.add_argument("--pre_lora", default="log/ref_isolation/checkpoint-20000/dit_lora.safetensors")
    p.add_argument("--post_lora", default="log/ref_distill/checkpoint-4000/dit_lora.safetensors")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--ref_size", type=int, default=512)
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lora_rank", type=int, default=512)
    p.add_argument("--save_path", default="output/multibanana_eval_distill")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    subset_dir = os.path.join(args.data_root, args.subset)
    if not os.path.isdir(subset_dir):
        raise SystemExit(f"❌ 子集目录不存在:{subset_dir}")

    # 决定 task_nums:显式 > 默认(add) > None(全取前 N)
    if args.task_nums:
        task_nums = args.task_nums
    elif args.subset == "add":
        task_nums = DEFAULT_ADD_TASKS
    else:
        task_nums = None  # 全取,后面按 n_tasks 截断

    tasks = discover_tasks(subset_dir, task_nums)
    if task_nums is None:
        tasks = tasks[:args.n_tasks]
    if not tasks:
        raise SystemExit(f"❌ {subset_dir} 下没发现任务(找不到 *_prompt.txt)")

    print(f"子集={args.subset}  任务 {len(tasks)} 个 × 变体 {len(VARIANTS)} 个 = "
          f"{len(tasks) * len(VARIANTS)} 次生成")
    for t in tasks:
        print(f"  [{t['number']}] refs={len(t['image_paths'])} | {t['prompt'][:90]}")
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
        if args.offload:
            pipeline.t5.cpu()
            pipeline.clip.cpu()
            torch.cuda.empty_cache()

        model_sd = pipeline.model.state_dict()
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

    # ---------- 生成 ----------
    all_results = {t["name"]: {} for t in tasks}
    all_times = {t["name"]: {} for t in tasks}
    timing = {}
    current_bank = None
    warmed = False

    for variant_name, use_ours, ref_iso, kv_cache, bank in VARIANTS:
        if not args.dry_run and current_bank != bank:
            swap_lora(pipeline.model, lora_banks[bank], variant_name)
            current_bank = bank

        per_task = []
        for task in tasks:
            refs = [Image.open(pth) for pth in task["image_paths"]]
            if args.dry_run:
                img = Image.new("RGB", (args.width, args.height), (200, 220, 240))
                t_denoise = 0.0
            else:
                ref_imgs = [preprocess_ref(r, args.ref_size) for r in refs]

                def run():
                    return pipeline(
                        prompt=task["prompt"], width=args.width, height=args.height,
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

            all_results[task["name"]][variant_name] = img
            all_times[task["name"]][variant_name] = t_denoise
            per_task.append(t_denoise)
            img.save(os.path.join(args.save_path, f'{task["name"]}__{variant_name}.png'))
            print(f"  [{variant_name}] {task['name']}  denoise {t_denoise:5.2f}s")

        timing[variant_name] = {
            "mean_s": statistics.mean(per_task) if per_task else 0.0,
            "median_s": statistics.median(per_task) if per_task else 0.0,
        }
        if not args.dry_run:
            import torch
            timing[variant_name]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

    # ---------- 拼图 ----------
    rows = []
    for task in tasks:
        refs = [Image.open(pth) for pth in task["image_paths"]]
        row = board.build_row(task["name"], task["prompt"], refs,
                              all_results[task["name"]], all_times[task["name"]], cell=256)
        row.save(os.path.join(args.save_path, f'row_{task["name"]}.png'))
        rows.append(row)
    board_path = os.path.join(args.save_path, f"compare_{args.subset}.png")
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
    print(f"单行对比:{args.save_path}/row_*.png")

    with open(os.path.join(args.save_path, f"results_{args.subset}.json"), "w") as f:
        json.dump({"config": vars(args), "timing": timing,
                   "tasks": [{"name": t["name"], "prompt": t["prompt"],
                              "image_paths": t["image_paths"]} for t in tasks]},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
