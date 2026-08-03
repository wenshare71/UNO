#!/usr/bin/env python3
"""生成 M5 P-probe 的任务单 `probe_iso_tasks.json`。

P-probe 要回答的问题:**把 isolated attention 直接开在未重训的官方 LoRA 上,
本身值多少代价?** 这个组合(官方权重 + ref_isolation + KV cache)在 M4 里从没跑过,
而它在 M4 的回退归因里占一整项——不知道"开隔离"这个动作本身的代价,
就没法把"官方 vs 我们的 KV 变体"之间的差距,拆成"隔离的代价"和"重训的收益"两块。

构造方式:同一条源任务生成两个变体 `official_full`(不隔离)与 `official_iso`
(隔离+KV cache),**seed 完全相同**——于是两张图之间只剩 `ref_isolation` 这一个
flag 的差别,谁给谁做减法都干净。`official_iso` 已经在 `eval_multiref.py:VARIANTS`
里登记好了(commit dfdb74f 之后的行),本脚本不碰那张表。

WHY seed 原样照抄、绝不偏移:这一点和步骤 1(`build_noise_floor.py`)正好相反。
那里要的是"除了 seed 什么都不变"来测噪声地板;这里要的是"除了 flag 什么都不变"
来测隔离本身的代价——换 seed 会把"隔离的代价"和"噪声的代价"混在一起,题面就不成立了。

WHY 选 S1 全部 132 条 + S3 全部 60 条 = 192 条,不做 CLI 参数、不抽样、不打散:
按实测 ~30% 平局率折算,192 条非平局样本数 ≈ 192 × 0.7 ≈ 134,
**刚好让 `M4_EVAL_SPEC.md` §8.2 的判据(Wilson 95% CI 下界 ≥ 0.40 且
n_nontie ≥ 94)适用**;而只取 60 条(比如单独 S3)只能折出 ≈42 条非平局样本,
连 94 都不够,判据直接"不适用",做出来的探针没法下结论。既然要凑够 94,
干脆把 S1、S3 全取——不抽样也就不用再解释"为什么抽这几条不抽那几条",
不留下第二个可以争的自由度。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# ------------------------------------------------------------------ 常数(改动 = 换一批任务,不许悄悄改)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JSON = os.path.join(REPO, "datasets/eval_multiref/eval_set.json")
OUT_JSON = os.path.join(REPO, "datasets/eval_multiref/probe_iso_tasks.json")

VARIANTS = ["official_full", "official_iso"]  # 顺序与 eval_multiref.py:VARIANTS 表同序,
                                               # 两者共用 "official" bank,相邻可省一次 LoRA swap。
STRATA = ("S1", "S3")
N_S1, N_S3 = 132, 60

# 成本估算的来源:KV cache 基准测试实测的 4.86 → 2.88 s/张(1.686× 加速)。
# **这是估算,不是本探针的实测**——P-probe 还没上过机。
# 而且 2.88 那个数是在 `ours_kv_*`(我们重训的 LoRA + 隔离 + KV)上测的;
# `official_iso` 用同样的 flag 组合,单张耗时应当相近,但**没人验证过**。
# 只用来给 --dry_run 一个数量级,不要拿它当结论。
# (对照:eval_multiref.py:450-451 的规划用的是更粗的 5.1 / 2.9。)
COST_S_PER_IMG = {"official_full": 4.86, "official_iso": 2.88}


def build(src_json: str) -> dict:
    with open(src_json, "rt", encoding="utf-8") as f:
        src = json.load(f)
    tasks = src["tasks"]

    picked = [t for t in tasks if t["stratum"] in STRATA]
    out: list[dict] = []
    for t in picked:
        meta = dict(t["meta"])           # 逐字复制,只加一个可追溯字段
        meta["probe_src_task_id"] = t["task_id"]
        out.append({
            "task_id": f"PI_{t['task_id']}",
            "stratum": t["stratum"],
            "prompt": t["prompt"],
            "image_paths": list(t["image_paths"]),
            "seed": t["seed"],           # 原样照抄,绝不偏移——见模块 docstring
            "variants": list(VARIANTS),
            "meta": meta,
        })

    n_img = sum(len(t["variants"]) for t in out)
    return {
        "meta": {
            "spec": "M5-probe-iso-v1",
            "n_tasks": len(out),
            "n_images": n_img,
            "source": "eval_set.json S1+S3",
        },
        "tasks": out,
    }


def verify(payload: dict, src_json: str, out_json: str = OUT_JSON) -> None:
    """产出自检。写完自动跑一遍,也可以单独 `--verify` 跑,免得"生成"与"校验"
    共用同一份内存里的假设(同 `build_noise_floor.py` 的纪律)。"""
    with open(src_json, "rt", encoding="utf-8") as f:
        src_by_id = {t["task_id"]: t for t in json.load(f)["tasks"]}
    tasks = payload["tasks"]
    errs: list[str] = []
    # 参考图是相对 json 所在目录解析的(eval_multiref.py:load_tasks 同规则),
    # 所以必须用**实际写出的路径**,锁死模块常量会在 --out 换目录时静默校验错地方
    json_dir = os.path.dirname(os.path.abspath(out_json))

    if len(tasks) != N_S1 + N_S3:
        errs.append(f"总数 {len(tasks)} 条,应为 {N_S1 + N_S3}")
    by_stratum = Counter(t["stratum"] for t in tasks)
    if by_stratum.get("S1", 0) != N_S1:
        errs.append(f"S1 {by_stratum.get('S1', 0)} 条,应为 {N_S1}")
    if by_stratum.get("S3", 0) != N_S3:
        errs.append(f"S3 {by_stratum.get('S3', 0)} 条,应为 {N_S3}")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        errs.append("task_id 有重复")

    n_img_actual = 0
    for t in tasks:
        if not t["task_id"].startswith("PI_"):
            errs.append(f"{t['task_id']}: 缺 PI_ 前缀")
            continue
        src_id = t["task_id"][len("PI_"):]
        src = src_by_id.get(src_id)
        if src is None:
            errs.append(f"{t['task_id']}: 剥掉前缀后的 {src_id} 在源任务单里不存在")
            continue

        if t["prompt"] != src["prompt"]:
            errs.append(f"{t['task_id']}: prompt 与源不一致")
        if t["image_paths"] != src["image_paths"]:
            errs.append(f"{t['task_id']}: image_paths 与源不一致")
        if t["stratum"] != src["stratum"]:
            errs.append(f"{t['task_id']}: stratum 与源不一致")
        if t["seed"] != src["seed"]:
            errs.append(f"{t['task_id']}: seed 与源不一致 ← 探针的前提就是只差一个 flag,这条最要命")
        if t["variants"] != VARIANTS:
            errs.append(f"{t['task_id']}: variants={t['variants']},应为 {VARIANTS}")

        for rel in t["image_paths"]:
            p = os.path.normpath(os.path.join(json_dir, rel))
            if not os.path.isfile(p):
                errs.append(f"{t['task_id']}: 参考图不存在 {p}")
        n_img_actual += len(t["variants"])

    if payload["meta"].get("n_images") != n_img_actual:
        errs.append(f"meta.n_images={payload['meta'].get('n_images')},实际 {n_img_actual}")
    if payload["meta"].get("n_tasks") != len(tasks):
        errs.append(f"meta.n_tasks={payload['meta'].get('n_tasks')},实际 {len(tasks)}")

    if errs:
        print("\n❌ 自检未通过:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("✓ 自检通过(条数 / 前缀+源对应 / prompt+refs+stratum 逐字一致 / "
          "seed 零偏移 / 变体 / 参考图存在 / n_images 一致)")


def summarize(payload: dict) -> None:
    tasks = payload["tasks"]
    print(f"\n任务 {len(tasks)} 条 / 出图 {payload['meta']['n_images']} 张")
    print(f"  分层:{dict(sorted(Counter(t['stratum'] for t in tasks).items()))}")
    n_nontie_est = round(len(tasks) * 0.7)
    print(f"  按 ~30% 平局率折合非平局样本 ≈ {n_nontie_est}"
          f"(M4_EVAL_SPEC.md §8.2 判据要求 n_nontie ≥ 94、CI 下界 ≥ 0.40)")


def print_dry_run(payload: dict, num_shards: int = 8) -> None:
    tasks = payload["tasks"]
    n_gen = sum(len(t["variants"]) for t in tasks)
    print(f"任务 {len(tasks)} 条,总生成次数 {n_gen}(= {len(tasks)} × {len(VARIANTS)})")
    total_s = 0.0
    for v in VARIANTS:
        n = len(tasks)  # 两个变体都覆盖全部任务
        s = n * COST_S_PER_IMG[v]
        total_s += s
        print(f"  {v:<16}{n:>4} 张 × {COST_S_PER_IMG[v]:.2f}s ≈ {s / 60:.1f} min")
    print(f"  单卡纯 denoise 合计 ≈ {total_s / 60:.1f} min(不含模型加载 ~7 min)")
    print(f"  分 {num_shards} shard 后单卡 ≈ {total_s / num_shards / 60:.1f} min "
          f"(+ 每 shard 各自 ~7 min 模型加载)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC_JSON)
    p.add_argument("--out", default=OUT_JSON)
    p.add_argument("--verify", action="store_true", help="只校验已有的 --out,不重新生成")
    p.add_argument("--dry_run", action="store_true", help="不写文件,只打印任务数与成本估算")
    p.add_argument("--num_shards", type=int, default=8, help="--dry_run 时用来估算分 shard 后的单卡耗时")
    args = p.parse_args()

    if args.verify:
        if not os.path.exists(args.out):
            raise SystemExit(f"❌ {args.out} 不存在")
        with open(args.out, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        verify(payload, args.src, args.out)
        summarize(payload)
        return

    payload = build(args.src)

    if args.dry_run:
        print_dry_run(payload, args.num_shards)
        return

    verify(payload, args.src, args.out)

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)  # 原子写,和 eval_multiref.py / build_noise_floor.py 一个套路
    print(f"\n写入 {os.path.relpath(args.out, REPO)}")
    summarize(payload)


if __name__ == "__main__":
    sys.exit(main())
