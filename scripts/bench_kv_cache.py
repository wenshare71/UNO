# KV-Cache 快速验证（不依赖我们自己训的 LoRA，官方权重即可跑）：
#   1. 一致性: ref_isolation(不开cache) vs kv_cache 数学上等价（都是隔离注意力 + t=0 调制，
#      cache 只是省掉每步重算 ref K/V），逐像素比对即可验证缓存实现没写错——这与权重无关。
#   2. 加速比: full attention / isolation / kv_cache 三种模式计时，计算量差异只取决于结构。
#   注意: 官方权重是全注意力训的，isolation/kv_cache 模式下"生成质量变差"是预期行为，
#         这正是需要 ref_isolation LoRA 的原因；本脚本不评质量，只验机制和测速。
#
# 用法（远程、有一张空闲 GPU 时）:
#   python scripts/bench_kv_cache.py
#   python scripts/bench_kv_cache.py --lora_path log/ref_isolation/checkpoint-XXXX/dit_lora.safetensors
import argparse
import json
import os
import sys
import time

# 让脚本不管从哪个目录运行都能找到 uno 包（uno/ 缺 __init__.py，走 namespace fallback）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from uno.flux.pipeline import UNOPipeline, preprocess_ref

MODES = {
    # name: (ref_isolation, kv_cache)
    "full": (False, False),
    "iso": (True, False),
    "kv": (True, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a clock and a cup on a wooden table, product photo")
    parser.add_argument("--image_paths", nargs="+", default=["assets/clock.png", "assets/cup.png"])
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=512)
    parser.add_argument("--lora_path", default=None,
                        help="加载指定 LoRA（如我们训出的 checkpoint）；不传则用默认加载逻辑(官方 UNO)")
    parser.add_argument("--save_path", default="output/bench_kv_cache")
    # 单张 4090(24GB) 放不下 bf16 FLUX-dev + T5 + AE，默认 offload（denoise 期间 DiT 常驻 GPU，
    # 不影响计时公平性）；若还 OOM 改 --model_type flux-dev-fp8
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    args = parser.parse_args()

    device = torch.device("cuda")
    if args.model_type == "flux-dev":
        # FLUX.1-dev DiT 是 12B 参数（hidden_size=3072, depth=19+38），bf16 权重本身就要
        # ~22.4 GiB，即使 --offload 把 T5/CLIP/AE 全挪回 CPU，单张 24GB 卡也几乎装不下——
        # 在这里提前拦，别等 T5/CLIP 都加载编码完了才在 model.to(device) 那步 OOM 浪费时间。
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if total_gb < 24.5:
            raise SystemExit(
                f"❌ GPU 总显存 {total_gb:.1f} GiB，装不下 bf16 FLUX-dev DiT(~22.4 GiB)+ CUDA 上下文\n"
                f"   请改用: python scripts/bench_kv_cache.py --model_type flux-dev-fp8"
            )
    pipeline = UNOPipeline(args.model_type, device, offload=args.offload,
                           only_lora=True, lora_rank=args.lora_rank)
    if args.lora_path:
        pipeline.load_ckpt(args.lora_path)

    ref_size = 512 if len(args.image_paths) == 1 else 320
    ref_imgs = [preprocess_ref(Image.open(p), ref_size) for p in args.image_paths]

    os.makedirs(args.save_path, exist_ok=True)

    def run(ref_isolation, kv_cache, seed):
        return pipeline(
            prompt=args.prompt, width=args.width, height=args.height,
            guidance=4, num_steps=args.num_steps, seed=seed,
            ref_imgs=ref_imgs, pe="d",
            ref_isolation=ref_isolation, kv_cache=kv_cache,
        )

    # 首次前向包含 CUDA kernel 编译等一次性开销，热身一轮不计时
    run(False, False, args.seed)
    torch.cuda.synchronize()

    results = {}
    for name, (ref_isolation, kv_cache) in MODES.items():
        torch.cuda.reset_peak_memory_stats()
        times = []
        for r in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = run(ref_isolation, kv_cache, args.seed)  # 固定 seed，保证 iso/kv 可逐像素比对
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        img.save(os.path.join(args.save_path, f"{name}.png"))
        results[name] = {
            "mean_s": float(np.mean(times)),
            "std_s": float(np.std(times)),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        print(f"[{name:4s}] {results[name]['mean_s']:.2f}s ± {results[name]['std_s']:.2f}s "
              f"| peak mem {results[name]['peak_mem_gb']:.1f} GiB")

    iso = np.asarray(Image.open(os.path.join(args.save_path, "iso.png")), dtype=np.int16)
    kv = np.asarray(Image.open(os.path.join(args.save_path, "kv.png")), dtype=np.int16)
    diff = np.abs(iso - kv)
    # bf16 下算子归约顺序不同（拼接序列 vs 读缓存）会有微小误差，25 步迭代会放大一些，
    # 所以用 mean 判等价：mean<0.5 视为等价；个别边缘像素 max 偏大不算失败，但 mean 大说明实现有错
    results["iso_vs_kv_diff"] = {"max": int(diff.max()), "mean": float(diff.mean())}
    results["speedup_kv_vs_full"] = results["full"]["mean_s"] / results["kv"]["mean_s"]
    results["config"] = {k: v for k, v in vars(args).items()}

    print(f"\niso vs kv 像素差: max={diff.max()} mean={diff.mean():.4f} "
          f"({'✅ 等价，缓存机制正确' if diff.mean() < 0.5 else '❌ 差异过大，kv_cache 实现可能有问题'})")
    print(f"加速比 (full → kv): {results['speedup_kv_vs_full']:.2f}x")

    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果与图片已保存到 {args.save_path}/")


if __name__ == "__main__":
    main()
