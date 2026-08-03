"""M4 Stage B:全量评测推理 `datasets/eval_multiref/eval_set.json`(232 任务 / 711 图)。

上游规格:`distill/M4_EVAL_SPEC.md` §4。本脚本比 `smoke_eval.py` 只多四件事:
LoRA bank 从 3 个变 4 个、任务从硬编码变成读 json、变体按任务的 `variants` 字段过滤、
加 sharding 与断点续跑。变体外层循环(切 LoRA 只搬一次)、swap_lora、key 硬校验、
warmup 标志全部照抄 smoke_eval.py。

推理配置(规格 §2.2,与冒烟两轮完全一致,否则不可比;全部写成 default):
  flux-dev bf16 / 不 offload / 512×512 / ref_size 512 / 25 步 / guidance 4.0 /
  lora_rank 512 / pe="d"

用法:
  # dry_run:不碰 GPU,打印本 shard 的任务/变体计划与成本估算
  python distill/eval_multiref.py --dry_run

  # 8 卡分片(每片 ≈ 7 min 加载 + 6 min 生成;单卡全量 ~50 min 也可接受)
  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i nohup python distill/eval_multiref.py \
      --shard_idx $i --num_shards 8 > logs/m4_shard$i.log 2>&1 &
  done

  # 只跑某几层(便于分批;S0 必须先跑并通过 --check_anchor)
  python distill/eval_multiref.py --strata S0

  # S0 锚点自检:与 output/smoke_eval/case0X__*.png 逐像素比(max ≤ 2 才继续 S1–S4)
  python distill/eval_multiref.py --check_anchor

  # 全部 shard 跑完后:合并 results + 拼 JPEG 板(纯 CPU)
  python distill/eval_multiref.py --merge
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from datetime import datetime

# 与 scripts/keepalive_infer.py:44-45 一致:H800 上直连 huggingface.co 不通、
# 走日本代理会卡死,必须离线加载本地缓存(该坑已在 keepalive 踩过一次)。
os.environ.setdefault("HF_HOME", "/kaimm-distill/wuwenxuan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "multibanana_eval"))

from PIL import Image  # noqa: E402

import board  # noqa: E402  (复用 multibanana_eval/board.py 的拼图)

# 变体:(标签, 用我们的LoRA?, ref_isolation, kv_cache, LoRA bank)——规格 §2.3。
# ours_kv_post2000 只出现在 S2 任务的 variants 字段里(回答"复制是不是过训产物"),
# 脚本不自己判断,照任务单执行。
VARIANTS = [
    ("official_full",    False, False, False, "official"),
    # official_iso [新增 2026-08-03,§11.4 P-probe]:官方 LoRA **不重训**、直接开隔离+KV。
    # 回答"隔离本身值多少代价"——这个组合从没跑过,而 M4 的回退归因里它占一整项(混淆 ②)。
    # 紧跟 official_full 是有意的:两者共用 "official" bank,相邻就不会触发 swap_lora。
    # 注意元组第 2 位 use_ours 在生成循环里**从未被读取**(只有 bank / ref_iso / kv_cache
    # 真正起作用),所以 False + ref_iso=True 这个"看着矛盾"的组合是合法且正确的。
    ("official_iso",     False, True,  True,  "official"),
    ("ours_kv_pre",      True,  True,  True,  "pre"),
    ("ours_kv_post4000", True,  True,  True,  "post4000"),
    ("ours_kv_post2000", True,  True,  True,  "post2000"),
]
VARIANT_BY_NAME = {v[0]: v for v in VARIANTS}

# 拼图切块:每块 ≤12 个任务(规格 §4.3;S1 有 132 个任务,拼一张没法看也没法提交)
BOARD_CHUNK = 12

# S0 锚点自检(规格 §5.1):新产物 vs 冒烟产物。max ≤ 2 通过(bf16 正常抖动)。
ANCHOR_MAP = {  # 变体标签 → smoke_eval.py 的文件名后缀
    "official_full": "official_full",
    "ours_kv_pre": "ours_kv_pre",
    "ours_kv_post4000": "ours_kv_post",
}
ANCHOR_MAX_TOL = 2


def resolve(p: str, json_dir: str) -> str:
    """eval_set.json 的 image_paths 相对 json 所在目录(`../dreambooth/...`)。"""
    return os.path.normpath(os.path.join(json_dir, p))


def out_path(save_path: str, task_id: str, variant: str) -> str:
    return os.path.join(save_path, f"{task_id}__{variant}.png")


def decodable(path: str) -> bool:
    """图存在**且能完整解码**。**纯谓词,不动文件。**

    `Image.open` 是惰性的、只读文件头,截断到一半的图照样通过;只有 `.load()`
    才抛 `OSError: image file is truncated`。shard 被杀时写到一半的图正是这个场景。
    """
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def already_done(path: str) -> bool:
    """本脚本自己的输出图是否已完成(规格 §4.4-1);坏图就地删除以便下轮重生成。

    WHY 与 `decodable` 分家:删除是破坏性的,**只允许作用于本脚本能重新生成的产物**。
    锚点自检读的是 `output/smoke_eval/` 里已提交的基线,--merge 拼图读的是全部既有产物;
    校验/报告路径若带删除副作用,一次解码失败就会把参照物本身抹掉,自检再也没法复跑。
    所以:**破坏性清理只出现在"删掉之后紧接着就会重新生成"的地方。**
    """
    if decodable(path):
        return True
    if os.path.exists(path):
        os.remove(path)  # 坏图直接删,免得下次又被 exists 认成"已完成"
    return False


def swap_lora(model, state_dict: dict, tag: str) -> None:
    """就地替换 LoRA 权重,硬校验 key 匹配(抄 smoke_eval.py:110)。"""
    model_sd = model.state_dict()
    unexpected = [k for k in state_dict if k not in model_sd]
    if unexpected:
        raise SystemExit(
            f"❌ [{tag}] checkpoint 有 {len(unexpected)} 个 key 在模型中不存在,"
            f"加载会静默失效。例如:{unexpected[:3]}")
    dev = next(model.parameters()).device
    aligned = {k: v.to(device=dev, dtype=model_sd[k].dtype) for k, v in state_dict.items()}
    model.load_state_dict(aligned, strict=False, assign=True)


def load_tasks(eval_json: str, strata: str | None) -> tuple[list[dict], str]:
    with open(eval_json, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    tasks = payload["tasks"]
    known = set(VARIANT_BY_NAME)
    for t in tasks:
        bad = [v for v in t["variants"] if v not in known]
        if bad:
            raise SystemExit(f"❌ {t['task_id']} 的 variants 含未知标签:{bad}")
    if strata:
        keep = {s.strip() for s in strata.split(",") if s.strip()}
        unknown = keep - {t["stratum"] for t in tasks}
        if unknown:
            raise SystemExit(f"❌ --strata 含未知层:{sorted(unknown)}")
        tasks = [t for t in tasks if t["stratum"] in keep]
    json_dir = os.path.dirname(os.path.abspath(eval_json))
    return tasks, json_dir


# ------------------------------------------------------------------ 生成

def run_generate(args, tasks, json_dir):
    import torch
    from uno.flux.pipeline import UNOPipeline, preprocess_ref
    from safetensors.torch import load_file

    # ---------- LoRA bank(抄 smoke_eval.py:160-195,从 3 个变 4 个) ----------
    ckpts = {"pre": args.pre_lora, "post4000": args.post4000_lora,
             "post2000": args.post2000_lora}
    for tag, lp in ckpts.items():
        if not os.path.exists(lp):
            raise SystemExit(f"❌ LoRA 不存在({tag}):{lp}")

    t_load = time.perf_counter()
    pipeline = UNOPipeline(args.model_type, torch.device("cuda"), offload=args.offload,
                           only_lora=True, lora_rank=args.lora_rank)
    if args.offload:  # 把 t5/clip 踢回 CPU 给 DiT 腾地方(非 offload 模式无需)
        pipeline.t5.cpu()
        pipeline.clip.cpu()
        torch.cuda.empty_cache()
    print(f"[{datetime.now():%H:%M:%S}] 模型就绪,耗时 {time.perf_counter() - t_load:.1f}s",
          flush=True)

    model_sd = pipeline.model.state_dict()
    official_sd = {k: model_sd[k].detach().clone().cpu() for k in model_sd
                   if any(k.endswith(s) for s in (".lora_A.weight", ".lora_B.weight",
                                                  ".lora_A.default.weight", ".lora_B.default.weight"))}
    if not official_sd:  # only_lora 模式下官方 LoRA 已挂载,兜底备份全部 lora key
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
    print("LoRA bank:" + " / ".join(f"{tag} {len(sd)}" for tag, sd in lora_banks.items())
          + " 张量", flush=True)

    # ---------- 变体外层循环(切 LoRA 只搬一次;规格 §4.4-3) ----------
    mine = tasks[args.shard_idx::args.num_shards]
    records, fails = [], []
    times = {v[0]: [] for v in VARIANTS}
    peak_mem = {}
    current_bank = None
    warmed = False
    n_done = n_skip = 0
    t_start = time.perf_counter()

    total_runs = sum(1 for t in mine for v in t["variants"])
    print(f"[{datetime.now():%H:%M:%S}] === M4 eval 启动 | "
          f"shard {args.shard_idx}/{args.num_shards} ===", flush=True)
    print(f"  全局任务 {len(tasks)} → 本 shard {len(mine)} 任务 / {total_runs} 次生成",
          flush=True)

    for variant_name, use_ours, ref_iso, kv_cache, bank in VARIANTS:
        todo = [t for t in mine if variant_name in t["variants"]]
        if not todo:
            continue
        if current_bank != bank:
            swap_lora(pipeline.model, lora_banks[bank], variant_name)
            current_bank = bank
        torch.cuda.reset_peak_memory_stats()

        for t in todo:
            dst = out_path(args.save_path, t["task_id"], variant_name)
            if already_done(dst):
                n_skip += 1
                records.append({"task_id": t["task_id"], "variant": variant_name,
                                "status": "skipped", "path": dst})
                continue
            try:
                ref_imgs = [preprocess_ref(Image.open(resolve(p, json_dir)).convert("RGB"),
                                           args.ref_size)
                            for p in t["image_paths"]]

                def run():
                    return pipeline(
                        prompt=t["prompt"], width=args.width, height=args.height,
                        guidance=args.guidance, num_steps=args.num_steps, seed=t["seed"],
                        ref_imgs=ref_imgs, pe="d",
                        ref_isolation=ref_iso, kv_cache=kv_cache,
                    )

                if not warmed:  # warmup 一次再开始计时(规格 §4.4-4)
                    print("warmup ...", flush=True)
                    run()
                    warmed = True

                t0 = time.perf_counter()
                img = run()
                t_denoise = time.perf_counter() - t0

                # 先写临时文件再 rename(原子),且必须显式给 format——.tmp 后缀
                # 会让 PIL 推断格式抛 ValueError(M1 已踩过)
                tmp = dst + ".tmp"
                img.save(tmp, format="PNG")
                os.replace(tmp, dst)
                times[variant_name].append(t_denoise)
                records.append({"task_id": t["task_id"], "variant": variant_name,
                                "status": "ok", "denoise_s": t_denoise, "path": dst})
                n_done += 1
                print(f"  [{variant_name}] {t['task_id']}  denoise {t_denoise:5.2f}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — 一个坏样本不许杀掉整个 shard(§4.4-2)
                fails.append({"task_id": t["task_id"], "variant": variant_name,
                              "error": f"{type(exc).__name__}: {exc}",
                              "tb": traceback.format_exc()})
                print(f"[{datetime.now():%H:%M:%S}] ❌ {t['task_id']}/{variant_name} — "
                      f"{type(exc).__name__}: {exc}", flush=True)
        peak_mem[variant_name] = torch.cuda.max_memory_allocated() / 1024**3

        # 边跑边落盘:被 kill 时不能把整个 shard 的 records 丢掉
        write_shard_results(args, records, fails, times, peak_mem)

    write_shard_results(args, records, fails, times, peak_mem)
    elapsed = time.perf_counter() - t_start
    print(f"[{datetime.now():%H:%M:%S}] === shard {args.shard_idx} 完成 ===", flush=True)
    print(f"  生成 {n_done} | 跳过 {n_skip} | 失败 {len(fails)} | "
          f"耗时 {elapsed / 60:.1f}m", flush=True)
    for vname, _, _, _, _ in VARIANTS:
        if times[vname]:
            print(f"  {vname:<18} mean {statistics.mean(times[vname]):5.2f}s "
                  f"median {statistics.median(times[vname]):5.2f}s "
                  f"peak {peak_mem.get(vname, 0.0):.1f} GB", flush=True)
    if fails:
        print(f"  失败明细见 results_shard{args.shard_idx}.json 的 fails 字段", flush=True)


def write_shard_results(args, records, fails, times, peak_mem):
    timing = {}
    for vname, ts in times.items():
        if ts:
            timing[vname] = {"mean_s": statistics.mean(ts),
                             "median_s": statistics.median(ts),
                             "n": len(ts),
                             "peak_mem_gb": peak_mem.get(vname, 0.0)}
    payload = {"shard_idx": args.shard_idx, "num_shards": args.num_shards,
               "config": {k: v for k, v in vars(args).items()},
               "records": records, "fails": fails, "timing": timing}
    tmp = os.path.join(args.save_path, f"results_shard{args.shard_idx}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(args.save_path, f"results_shard{args.shard_idx}.json"))


# ------------------------------------------------------------------ 合并 + 拼图

def do_merge(args, tasks, json_dir):
    shards = sorted(f for f in os.listdir(args.save_path)
                    if re.fullmatch(r"results_shard\d+\.json", f))
    if not shards:
        raise SystemExit(f"❌ {args.save_path} 下没有 results_shard*.json")
    all_records, all_fails, per_variant_times, per_variant_peak = [], [], {}, {}
    for fn in shards:
        with open(os.path.join(args.save_path, fn), "rt", encoding="utf-8") as f:
            payload = json.load(f)
        n_ok = sum(1 for r in payload["records"] if r["status"] == "ok")
        print(f"  {fn}: {len(payload['records'])} 条记录(ok {n_ok}, "
              f"skip {len(payload['records']) - n_ok}, fail {len(payload['fails'])})")
        all_records += payload["records"]
        all_fails += payload["fails"]
        for vname, t in payload["timing"].items():
            per_variant_times.setdefault(vname, []).append(t)
            per_variant_peak[vname] = max(per_variant_peak.get(vname, 0.0),
                                          t.get("peak_mem_gb", 0.0))

    # 跨 shard 汇总:mean/median 需要从每张图的 denoise_s 重算(shard 均值不可再平均)
    denoise = {}
    for r in all_records:
        if r["status"] == "ok" and "denoise_s" in r:
            denoise.setdefault(r["variant"], []).append(r["denoise_s"])
    timing = {}
    for vname, ts in denoise.items():
        timing[vname] = {"mean_s": statistics.mean(ts), "median_s": statistics.median(ts),
                         "n": len(ts), "peak_mem_gb": per_variant_peak.get(vname, 0.0)}
    base = timing.get("official_full", {}).get("mean_s")
    for vname, t in timing.items():
        t["speedup_vs_teacher"] = (base / t["mean_s"]) if base else None

    merged = {"spec": "M4-eval-v1", "n_shards": len(shards),
              "n_records": len(all_records), "n_fails": len(all_fails),
              "timing": timing, "records": all_records, "fails": all_fails}
    out = os.path.join(args.save_path, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n合并 {len(shards)} 个 shard → {out}")

    print(f"\n{'变体':<18}{'mean':>8}{'median':>8}{'n':>6}{'peak GB':>9}{'vs teacher':>12}")
    for vname, _, _, _, _ in VARIANTS:
        t = timing.get(vname)
        if not t:
            continue
        sp = f"{t['speedup_vs_teacher']:.2f}x" if t["speedup_vs_teacher"] else "-"
        print(f"{vname:<18}{t['mean_s']:>7.2f}s{t['median_s']:>7.2f}s{t['n']:>6}"
              f"{t['peak_mem_gb']:>9.1f}{sp:>12}")
    print("(冒烟基线:KV 变体 vs teacher 1.72–1.77x,显存 37 GB;明显偏离要说明)")

    build_boards(args, tasks, json_dir, all_records)


def build_boards(args, tasks, json_dir, records):
    """按层切块拼图(每块 ≤12 任务),JPEG q90。缺图/坏图用灰底占位,不让拼图炸掉。"""
    times = {(r["task_id"], r["variant"]): r.get("denoise_s")
             for r in records if r["status"] == "ok"}
    boards_dir = os.path.join(args.save_path, "boards")
    os.makedirs(boards_dir, exist_ok=True)
    strata = []
    for t in tasks:
        if t["stratum"] not in strata:
            strata.append(t["stratum"])

    n_boards = 0
    for st in strata:
        sub = [t for t in tasks if t["stratum"] == st]
        for ci in range(0, len(sub), BOARD_CHUNK):
            chunk = sub[ci:ci + BOARD_CHUNK]
            rows = []
            for t in chunk:
                refs = [Image.open(resolve(p, json_dir)).convert("RGB")
                        for p in t["image_paths"]]
                results, tt = {}, {}
                for vname in t["variants"]:
                    pth = out_path(args.save_path, t["task_id"], vname)
                    # 同样用 decodable:--merge 是报告路径,不该改动产物集合
                    if decodable(pth):
                        results[vname] = Image.open(pth).convert("RGB")
                        dt = times.get((t["task_id"], vname))
                        if dt is not None:
                            tt[vname] = dt
                    else:
                        results[vname] = Image.new("RGB", (args.width, args.height),
                                                   (220, 220, 220))
                title = f"{t['task_id']}  ({'/'.join(t['meta']['subjects'])})"
                rows.append(board.build_row(title, t["prompt"], refs, results, tt,
                                            cell=256))
            bp = os.path.join(boards_dir, f"{st}_{ci // BOARD_CHUNK:02d}.jpg")
            board.stack_board(rows).save(bp, format="JPEG", quality=90)
            n_boards += 1
    print(f"拼图 {n_boards} 块 → {boards_dir}/(每块 ≤{BOARD_CHUNK} 任务,JPEG q90)")


# ------------------------------------------------------------------ S0 锚点自检

def check_anchor(args, tasks):
    """规格 §5.1:S0 产物必须与 output/smoke_eval/case0X__*.png 逐像素一致(max ≤ 2)。

    max 明显更大 = seed / ref 路径 / 变体配置 / 预处理有一处对不上——
    停下上报,不要继续跑 S1–S4(尺子没校准就量,711 张图全部白烧)。
    """
    import numpy as np

    s0 = sorted((t for t in tasks if t["stratum"] == "S0"), key=lambda t: t["task_id"])
    if len(s0) != 5:
        raise SystemExit(f"❌ S0 应有 5 个任务,实际 {len(s0)}(是否忘了不带 --strata 跑?)")
    print(f"{'case':<6}{'变体':<18}{'max':>6}{'mean':>10}  判定")
    worst = 0
    for i, t in enumerate(s0):
        for vname, smoke_tag in ANCHOR_MAP.items():
            new_p = out_path(args.save_path, t["task_id"], vname)
            old_p = os.path.join("output/smoke_eval", f"case{i:02d}__{smoke_tag}.png")
            # 用 decodable 而非 already_done:自检是只读校验,绝不能删掉 output/smoke_eval/
            # 里已提交的基线——那是本次比对的参照物。
            new_ok, old_ok = decodable(new_p), decodable(old_p)
            if not (new_ok and old_ok):
                print(f"case{i:<3}{vname:<18}  缺文件或无法解码:"
                      f"{new_p if not new_ok else old_p}")
                worst += 1
                continue
            a = np.asarray(Image.open(new_p).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(old_p).convert("RGB"), dtype=np.int16)
            if a.shape != b.shape:
                print(f"case{i:<3}{vname:<18}  尺寸不一致 {a.shape} vs {b.shape}")
                worst += 1
                continue
            mx, mn = int(np.abs(a - b).max()), float(np.abs(a - b).mean())
            ok = mx <= ANCHOR_MAX_TOL
            worst += 0 if ok else 1
            print(f"case{i:<3}{vname:<18}{mx:>6}{mn:>10.4f}  {'✓' if ok else '✗ 超标'}")
    if worst:
        print(f"\n❌ 锚点自检未通过({worst} 处)。停下上报,不要继续跑 S1–S4。")
        raise SystemExit(1)
    print("\n✓ 锚点自检全部通过(max ≤ 2),可以继续 S1–S4。")


# ------------------------------------------------------------------ dry_run 计划

def print_plan(args, tasks):
    mine = tasks[args.shard_idx::args.num_shards]
    # 冒烟实测成本(规格 §4.5):official_full ~5.1 s/张,ours_kv_* ~2.9 s/张
    cost = {"official_full": 5.1, "ours_kv_pre": 2.9,
            "ours_kv_post4000": 2.9, "ours_kv_post2000": 2.9}
    print(f"\nshard {args.shard_idx}/{args.num_shards}:全局 {len(tasks)} 任务 → "
          f"本 shard {len(mine)} 任务")
    total_s = 0.0
    for vname, *_ in VARIANTS:
        n = sum(1 for t in mine if vname in t["variants"])
        if not n:
            continue
        est = n * cost[vname]
        total_s += est
        print(f"  {vname:<18}{n:>4} 张 × {cost[vname]:.1f}s ≈ {est / 60:.1f} min")
    print(f"  纯 denoise 合计 ≈ {total_s / 60:.1f} min(+ 模型加载 ~7 min)")
    print("  若发现要跑几小时,说明哪里错了(LoRA 反复搬 / 没做变体外层循环)——停下上报。")
    from collections import Counter
    by_st = Counter(t["stratum"] for t in mine)
    print(f"  分层:{dict(sorted(by_st.items()))}")


# ------------------------------------------------------------------ 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", default="datasets/eval_multiref/eval_set.json")
    p.add_argument("--pre_lora", default="log/ref_isolation/checkpoint-20000/dit_lora.safetensors")
    p.add_argument("--post4000_lora", default="log/ref_distill/checkpoint-4000/dit_lora.safetensors")
    p.add_argument("--post2000_lora", default="log/ref_distill/checkpoint-2000/dit_lora.safetensors")
    p.add_argument("--save_path", default="output/eval_multiref")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--strata", default=None, help="只跑指定层,逗号分隔,如 S0,S1")
    # ---- 推理超参(规格 §2.2,默认值就是实验配置,不要手填) ----
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--ref_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--lora_rank", type=int, default=512)
    # ---- 模式 ----
    p.add_argument("--dry_run", action="store_true", help="不碰 GPU,打印计划与成本估算")
    p.add_argument("--merge", action="store_true", help="合并 results_shard*.json + 拼图(纯 CPU)")
    p.add_argument("--check_anchor", action="store_true",
                   help="S0 锚点自检:与 output/smoke_eval/case0X__*.png 逐像素比(纯 CPU)")
    args = p.parse_args()

    if not 0 <= args.shard_idx < args.num_shards:
        raise SystemExit(f"❌ shard_idx {args.shard_idx} 不在 [0, {args.num_shards}) 内")

    tasks, json_dir = load_tasks(args.eval_json, args.strata)
    os.makedirs(args.save_path, exist_ok=True)

    if args.check_anchor:
        check_anchor(args, tasks)
        return
    if args.merge:
        do_merge(args, tasks, json_dir)
        return
    if args.dry_run:
        print_plan(args, tasks)
        print("dry_run:未加载模型,未写任何文件。")
        return

    run_generate(args, tasks, json_dir)


if __name__ == "__main__":
    main()
