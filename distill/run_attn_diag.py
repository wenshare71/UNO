"""注意力演化诊断 —— 首轮推理与录制(在 H800 上跑)。

上游:`.codeflicker/discuss/2026-07-29/attention-visualization/`(D01/D02) + D03 修正。
任务单:`datasets/eval_multiref/attn_diag_tasks.json`(由 `build_attn_tasks.py` 生成)。
录制器:`distill/attn_record.py`(monkey-patch,既有 `.py` diff 为 0)。

推理配置与 M4 完全一致(否则跨实验不可比),全部写成 default:
  flux-dev bf16 / 不 offload / 512×512 / ref_size 512 / 25 步 / guidance 4.0 /
  lora_rank 512 / pe="d"

## 四件套(D03 修正 ③)

  official_full      full attention,无 cache      —— teacher 上界
  ours_kv_pre        隔离 + cache,ckpt-20000      —— 故障现场
  ours_kv_post4000   隔离 + cache,ckpt-4000       —— 修复效果
  ours_iso_nocache   隔离,**无 cache**            —— 新增第四条线

第四条线默认挂 **pre** bank。因为它要回答两个问题,而 pre 能同时回答:
  1. 缓存本身改没改变注意力行为?(与 ours_kv_pre 同 ckpt 同 seed 逐点对比)
  2. 失败该归因到"隔离"还是"缓存"?——**这个只有在故障现场问才有意义**。
     pre_nocache 若同样丢 ref_2 → 缓存无罪,归因到隔离/训练;
     若不丢 → 缓存有嫌疑,那 M1–M4 的结论都要重新审视。
`--nocache_bank post4000` 可切到修复后的 ckpt(那样只回答问题 1)。

## 用法(全在 H800 上)

  python distill/test_attn_record.py                 # 上机预检,几秒;先跑这个
  python distill/run_attn_diag.py --dry_run          # 不碰 GPU,打计划
  python distill/run_attn_diag.py                    # 单卡跑全部 32 次(约 11 min)
  python distill/run_attn_diag.py --verify_identical # 纯 CPU:与 M4 产物逐像素比

`--verify_identical` 是不变量 1 的实证:录制不参与前向,所以本脚本产出的图应当与
`output/eval_multiref/` 里 M4 Stage B 的同名图**逐比特相同**(max diff = 0)。
Stage B 还没跑就跳过,不算失败。

**必须在 H800 上跑,不能在本地跑**:Stage B 的 711 张 PNG 按 .gitignore 留在 H800
不进 git,本地只有 results.json 和拼图。在本地跑只会得到"没有可比对的图对",
那是参照物不在,不是通过。
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# (标签, 用我们的 LoRA?, ref_isolation, kv_cache, bank);bank=None 表示由 CLI 决定
VARIANTS = [
    ("official_full",    False, False, False, "official"),
    ("ours_kv_pre",      True,  True,  True,  "pre"),
    ("ours_kv_post4000", True,  True,  True,  "post4000"),
    ("ours_iso_nocache", True,  True,  False, None),
]

# M4 Stage B 的产物目录,--verify_identical 拿它做参照
M4_DIR = "output/eval_multiref"

# --dry_run 估存储用。与 attn_record 的同名常量重复,是为了让 dry_run 不必 import torch;
# run() 里 AR 到手后会硬校验两边一致(常量漂了只会让估算悄悄失真)。
N_BLOCKS, N_HEADS = 57, 24


def out_stem(save_path: str, task_id: str, variant: str) -> str:
    return os.path.join(save_path, f"{task_id}__{variant}")


def resolve(p: str, json_dir: str) -> str:
    return os.path.normpath(os.path.join(json_dir, p))


def decodable(path: str) -> bool:
    """纯谓词,不动文件(与 eval_multiref.py:84 同语义)。"""
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def curve_ndim(path: str) -> int:
    """npz 里 curve 的维数;读不出来算 0。3 = D04 之前(head 已平均),4 = 带 head 维度。"""
    try:
        with np.load(path) as z:
            return int(z["curve"].ndim)
    except Exception:  # noqa: BLE001 — 坏文件与旧格式一样都得重跑
        return 0


def done(stem: str) -> bool:
    """图与 npz 都在、且 npz 是当前格式才算完成;否则删掉重跑。

    WHY 要查 curve 维数:D04 把曲线拆到了 head 维度。旧 npz 躺在目录里会被当成
    "已完成"跳过,跑完得到 3 维 4 维混着的一批,而 plot_attn 两种格式都吃——
    报告照样出得来,只是一部分样本没有 head 维度,**且没有任何提示**。
    """
    png, npz = stem + ".png", stem + ".npz"
    if decodable(png) and os.path.exists(npz):
        nd = curve_ndim(npz)
        if nd == 4:
            return True
        print(f"  ↻ {os.path.basename(stem)}:npz 是旧格式(curve {nd} 维),重跑", flush=True)
    for p in (png, npz):
        if os.path.exists(p):
            os.remove(p)
    return False


def swap_lora(model, state_dict: dict, tag: str) -> None:
    """就地替换 LoRA,硬校验 key(抄 eval_multiref.py:115)。"""
    model_sd = model.state_dict()
    unexpected = [k for k in state_dict if k not in model_sd]
    if unexpected:
        raise SystemExit(
            f"❌ [{tag}] checkpoint 有 {len(unexpected)} 个 key 在模型中不存在,"
            f"加载会静默失效。例如:{unexpected[:3]}")
    dev = next(model.parameters()).device
    aligned = {k: v.to(device=dev, dtype=model_sd[k].dtype) for k, v in state_dict.items()}
    model.load_state_dict(aligned, strict=False, assign=True)


def load_tasks(path: str) -> tuple[list[dict], str]:
    with open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["tasks"], os.path.dirname(os.path.abspath(path))


def resolved_variants(args) -> list[tuple]:
    """填掉 ours_iso_nocache 的 bank,并**按 bank 分组**——每个 bank 只搬一次 LoRA。"""
    out = []
    for name, use_ours, iso, kv, bank in VARIANTS:
        out.append((name, use_ours, iso, kv, bank or args.nocache_bank))
    order = {"official": 0, "pre": 1, "post4000": 2, "post2000": 3}
    return sorted(out, key=lambda v: (order.get(v[4], 9), v[0]))


# ------------------------------------------------------------------ 生成 + 录制

def run(args, tasks, json_dir):
    import torch
    from safetensors.torch import load_file
    from uno.flux.pipeline import UNOPipeline, preprocess_ref

    import attn_record as AR

    if (AR.N_BLOCKS, AR.N_HEADS) != (N_BLOCKS, N_HEADS):
        raise SystemExit(
            f"❌ 本文件的 (N_BLOCKS, N_HEADS)=({N_BLOCKS}, {N_HEADS}) 与 attn_record 的 "
            f"({AR.N_BLOCKS}, {AR.N_HEADS}) 不一致——--dry_run 的存储估算会失真,先对齐。")

    ckpts = {"pre": args.pre_lora, "post4000": args.post4000_lora}
    for tag, lp in ckpts.items():
        if not os.path.exists(lp):
            raise SystemExit(f"❌ LoRA 不存在({tag}):{lp}")

    t_load = time.perf_counter()
    pipeline = UNOPipeline(args.model_type, torch.device("cuda"), offload=args.offload,
                           only_lora=True, lora_rank=args.lora_rank)
    print(f"[{datetime.now():%H:%M:%S}] 模型就绪,耗时 {time.perf_counter() - t_load:.1f}s",
          flush=True)

    model_sd = pipeline.model.state_dict()
    official_sd = {k: v.detach().clone().cpu() for k, v in model_sd.items()
                   if "lora" in k.lower()}
    if not official_sd:
        raise SystemExit("❌ 没找到 LoRA key,无法备份官方权重做对照")
    lora_banks = {"official": official_sd}
    for tag, lp in ckpts.items():
        sd = load_file(lp, device="cpu")
        unknown = [k for k in sd if k not in model_sd]
        if unknown:
            raise SystemExit(f"❌ [{tag}] checkpoint key 与模型不匹配,例如 {unknown[:3]}")
        lora_banks[tag] = sd
    print("LoRA bank:" + " / ".join(f"{t} {len(sd)}" for t, sd in lora_banks.items())
          + " 张量", flush=True)

    spatial_blocks = tuple(int(x) for x in args.spatial_blocks.split(","))
    spatial_steps = tuple(int(x) for x in args.spatial_steps.split(","))
    for b in spatial_blocks:
        if not 0 <= b < AR.N_BLOCKS:
            raise SystemExit(f"❌ spatial_block {b} 超出 [0, {AR.N_BLOCKS})")
    for s in spatial_steps:
        if not 0 <= s < args.num_steps:
            raise SystemExit(f"❌ spatial_step {s} 超出 [0, {args.num_steps})")

    records, fails = [], []
    times, peak_mem = {}, {}
    current_bank = None
    warmed = False
    n_done = n_skip = 0
    t_start = time.perf_counter()

    variants = resolved_variants(args)
    total = sum(1 for t in tasks for v in variants if v[0] in t["variants"])
    print(f"[{datetime.now():%H:%M:%S}] === 注意力诊断启动 | "
          f"{len(tasks)} 任务 × {len(variants)} 变体 = {total} 次录制 ===", flush=True)
    print(f"  spatial: blocks {spatial_blocks} × steps {spatial_steps}", flush=True)

    for vname, use_ours, ref_iso, kv_cache, bank in variants:
        todo = [t for t in tasks if vname in t["variants"]]
        if not todo:
            continue
        if current_bank != bank:
            swap_lora(pipeline.model, lora_banks[bank], vname)
            current_bank = bank
            print(f"[{datetime.now():%H:%M:%S}] LoRA → {bank}(为 {vname})", flush=True)
        torch.cuda.reset_peak_memory_stats()
        times.setdefault(vname, [])

        for t in todo:
            stem = out_stem(args.save_path, t["task_id"], vname)
            if done(stem):
                n_skip += 1
                records.append({"task_id": t["task_id"], "variant": vname,
                                "status": "skipped"})
                continue
            try:
                ref_imgs = [preprocess_ref(Image.open(resolve(p, json_dir)).convert("RGB"),
                                           args.ref_size)
                            for p in t["image_paths"]]
                layout = AR.layout_from_sizes([im.size for im in ref_imgs],
                                              args.width, args.height)

                def gen():
                    return pipeline(
                        prompt=t["prompt"], width=args.width, height=args.height,
                        guidance=args.guidance, num_steps=args.num_steps, seed=t["seed"],
                        ref_imgs=ref_imgs, pe="d",
                        ref_isolation=ref_iso, kv_cache=kv_cache,
                    )

                if not warmed:  # warmup 不录制,否则第一次的 cudnn 选择会混进计时
                    print("warmup ...", flush=True)
                    gen()
                    warmed = True

                rec = AR.AttnRecorder(layout, num_steps=args.num_steps,
                                      spatial_blocks=spatial_blocks,
                                      spatial_steps=spatial_steps,
                                      head_chunk=args.head_chunk)
                t0 = time.perf_counter()
                with AR.recording(rec):
                    img = gen()
                dt = time.perf_counter() - t0

                # 原子写:先 .tmp 再 replace。PIL 必须显式给 format——.tmp 后缀会让它
                # 推断失败抛 ValueError(M1 踩过)。
                img.save(stem + ".png.tmp", format="PNG")
                os.replace(stem + ".png.tmp", stem + ".png")
                np.savez_compressed(
                    stem + ".npz.tmp.npz",
                    **rec.to_npz_payload(
                        task_id=t["task_id"], variant=vname, prompt=t["prompt"],
                        seed=t["seed"], subjects=t["meta"]["subjects"],
                        ref_isolation=bool(ref_iso), kv_cache=bool(kv_cache),
                        bank=bank, num_steps=args.num_steps))
                os.replace(stem + ".npz.tmp.npz", stem + ".npz")

                times[vname].append(dt)
                records.append({"task_id": t["task_id"], "variant": vname,
                                "status": "ok", "record_s": dt,
                                "ref_lens": list(layout.ref_lens),
                                "seq_len": layout.seq_with_ref,
                                "q_len_by_step_head": rec.q_len_by_step[:3]})
                n_done += 1
                print(f"  [{vname}] {t['task_id']}  {dt:5.2f}s  "
                      f"L={layout.seq_with_ref} refs={list(layout.ref_lens)}", flush=True)
            except Exception as exc:  # noqa: BLE001 — 一个坏样本不许杀掉整轮
                fails.append({"task_id": t["task_id"], "variant": vname,
                              "error": f"{type(exc).__name__}: {exc}",
                              "tb": traceback.format_exc()})
                print(f"[{datetime.now():%H:%M:%S}] ❌ {t['task_id']}/{vname} — "
                      f"{type(exc).__name__}: {exc}", flush=True)
        peak_mem[vname] = torch.cuda.max_memory_allocated() / 1024**3
        write_results(args, records, fails, times, peak_mem)  # 边跑边落盘

    write_results(args, records, fails, times, peak_mem)
    print(f"[{datetime.now():%H:%M:%S}] === 完成 ===", flush=True)
    print(f"  录制 {n_done} | 跳过 {n_skip} | 失败 {len(fails)} | "
          f"耗时 {(time.perf_counter() - t_start) / 60:.1f}m", flush=True)
    for vname in (v[0] for v in variants):
        if times.get(vname):
            print(f"  {vname:<18} mean {statistics.mean(times[vname]):5.2f}s "
                  f"peak {peak_mem.get(vname, 0.0):.1f} GB", flush=True)
    print("\n下一步:回本地跑 `python distill/plot_attn.py`(纯 CPU,不占 GPU)。")
    if not fails:
        print("另可跑 `--verify_identical` 验证录制未改变前向(需 M4 Stage B 产物)。")


def write_results(args, records, fails, times, peak_mem):
    timing = {v: {"mean_s": statistics.mean(ts), "n": len(ts),
                  "peak_mem_gb": peak_mem.get(v, 0.0)}
              for v, ts in times.items() if ts}
    payload = {"spec": "attn-diag-v1", "config": dict(vars(args)),
               "records": records, "fails": fails, "timing": timing}
    tmp = os.path.join(args.save_path, "results.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(args.save_path, "results.json"))


# ------------------------------------------------------------------ 不变量 1 的实证

def verify_identical(args, tasks):
    """本脚本的产物 vs M4 Stage B 的同名图,应当逐比特相同(max diff = 0)。

    录制不参与前向(attn_record 不变量 1),所以任何非零差异都意味着
    monkey-patch 泄漏进了数值路径——那 D01/D02 的"eager 与线上微差"红线就得复活。
    """
    pairs, missing = [], 0
    for t in tasks:
        for vname in t["variants"]:
            mine = out_stem(args.save_path, t["task_id"], vname) + ".png"
            theirs = os.path.join(M4_DIR, f"{t['task_id']}__{vname}.png")
            if not decodable(mine):
                continue
            if not decodable(theirs):
                missing += 1
                continue
            pairs.append((t["task_id"], vname, mine, theirs))

    if not pairs:
        print(f"没有可比对的图对(本脚本产物缺失,或 M4 Stage B 尚未跑;跳过 {missing} 个)。")
        print("这不算失败——Stage B 跑完后再来比即可。")
        return

    print(f"{'task':<14}{'变体':<20}{'max':>6}{'mean':>10}  判定")
    worst = 0
    for task_id, vname, mine, theirs in pairs:
        a = np.asarray(Image.open(mine).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(theirs).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            print(f"{task_id:<14}{vname:<20}  尺寸不一致 {a.shape} vs {b.shape}")
            worst += 1
            continue
        mx, mn = int(np.abs(a - b).max()), float(np.abs(a - b).mean())
        worst += 0 if mx == 0 else 1
        print(f"{task_id:<14}{vname:<20}{mx:>6}{mn:>10.4f}  {'✓' if mx == 0 else '✗ 非零'}")
    if worst:
        print(f"\n❌ {worst} 处非零差异:录制**改变了前向**,与 attn_record 不变量 1 矛盾。"
              f"\n   停下上报——此时曲线仍可用,但报告里必须写明数值差异,不能再说'逐比特相同'。")
        raise SystemExit(1)
    print(f"\n✓ {len(pairs)} 对图逐比特相同(max = 0):录制确实没有参与前向。"
          f"{f' (另有 {missing} 个 M4 侧尚无产物,已跳过)' if missing else ''}")


def print_plan(args, tasks):
    variants = resolved_variants(args)
    # M4 冒烟实测:official_full ~5.1 s/张,kv 变体 ~2.9 s/张。
    # 录制额外算一遍注意力(只 img 行),按 +60% 估。
    cost = {"official_full": 5.1 * 1.6, "ours_kv_pre": 2.9 * 1.6,
            "ours_kv_post4000": 2.9 * 1.6, "ours_iso_nocache": 5.1 * 1.6}
    total = 0.0
    print(f"\n{len(tasks)} 个任务 × {len(variants)} 变体")
    for vname, _, iso, kv, bank in variants:
        n = sum(1 for t in tasks if vname in t["variants"])
        if not n:
            continue
        est = n * cost.get(vname, 5.0)
        total += est
        print(f"  {vname:<18} bank={bank:<10} iso={int(iso)} kv={int(kv)}  "
              f"{n:>3} 次 ≈ {est / 60:.1f} min")
    print(f"  纯推理合计 ≈ {total / 60:.1f} min(+ 模型加载 ~7 min)")
    sb = len(args.spatial_blocks.split(","))
    ss = len(args.spatial_steps.split(","))
    gh, gw = args.height // 16, args.width // 16
    # n_seg = txt + img + 每张 ref 一段,所以 2-ref 是 4、3-ref 是 5;逐任务实算,不取代表值
    sizes, tot = [], 0
    for t in tasks:
        n_seg = 2 + len(t["image_paths"])
        per = 4 * n_seg * (args.num_steps * N_BLOCKS * N_HEADS + sb * ss * gh * gw)
        sizes.append(per)
        tot += per * sum(1 for v in variants if v[0] in t["variants"])
    print(f"  每次录制的 npz {min(sizes) / 1024:.0f}–{max(sizes) / 1024:.0f} KB(压缩前),"
          f"全轮 ≈ {tot / 1024**2:.1f} MB")
    print(f"  (曲线 {args.num_steps}×{N_BLOCKS}×{N_HEADS}head×n_seg —— D04 起带 head 维度,"
          f"是之前的 {N_HEADS}×;热力图 {sb}×{ss}×{gh}×{gw}×n_seg,仍是 head 平均)")
    from collections import Counter
    print(f"  分层:{dict(sorted(Counter(t['stratum'] for t in tasks).items()))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks_json", default="datasets/eval_multiref/attn_diag_tasks.json")
    p.add_argument("--pre_lora",
                   default="log/ref_isolation/checkpoint-20000/dit_lora.safetensors")
    p.add_argument("--post4000_lora",
                   default="log/ref_distill/checkpoint-4000/dit_lora.safetensors")
    p.add_argument("--nocache_bank", default="pre", choices=["pre", "post4000"],
                   help="ours_iso_nocache 挂哪个 ckpt;默认 pre(故障现场,能同时答归因问题)")
    p.add_argument("--save_path", default="output/attn_diag")
    # ---- 录制粒度 ----
    p.add_argument("--spatial_blocks", default="4,18,37,52",
                   help="存热力图的 block(flat index:0-18 double,19-56 single)。"
                        "默认 4 层而非 D02 的 3 层——18/19 是 double→single 的架构分界,"
                        "两侧各留一层;存储代价可忽略(见 --dry_run)")
    p.add_argument("--spatial_steps", default="0,6,12,18,24")
    p.add_argument("--head_chunk", type=int, default=4,
                   help="按 head 分块算注意力以压瞬态显存;4 时 2-ref 约 2×59 MB")
    # ---- 推理超参(与 M4 一致,不要手填) ----
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--ref_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--lora_rank", type=int, default=512)
    # ---- 模式 ----
    p.add_argument("--dry_run", action="store_true", help="不碰 GPU,打印计划与成本")
    p.add_argument("--verify_identical", action="store_true",
                   help="纯 CPU:与 M4 Stage B 产物逐像素比,验证录制未改变前向")
    args = p.parse_args()

    tasks, json_dir = load_tasks(args.tasks_json)
    os.makedirs(args.save_path, exist_ok=True)

    if args.verify_identical:
        verify_identical(args, tasks)
        return
    if args.dry_run:
        print_plan(args, tasks)
        print("dry_run:未加载模型,未写任何文件。")
        return
    run(args, tasks, json_dir)


if __name__ == "__main__":
    main()
