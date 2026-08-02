#!/usr/bin/env python3
"""生成 §11 步骤 1 的噪声地板任务单 `noise_floor_tasks.json`。

零假设对(null pair)要回答的是:**当两张图真的没有质量差异时,标注者会给出什么。**
这是 M5 全部数字的尺子刻度——不知道地板在哪里,就没法说 0.819 是"差"还是"噪声"。

构造方式(§11.0 偏离①):同 refs、同 prompt、**只换噪声 seed**。
    左 = 既有的  {src}__official_full.png        (M4 已生成,在 output/eval_multiref/)
    右 = 新生成的 NF_{src}__official_full.png     (本脚本的任务单,在 output/noise_floor/)

WHY 不用"同组合跨 seed_slot 自比":`eval_set.json` 里 seed_slot 变的时候
**参考图视角也跟着变**(S1_000_s0 用 00.jpg+01.jpg,S1_000_s1 用 01.jpg+02.jpg)。
两张图的参考物都不是同一组,再问"谁更忠于参考"就没有公共被指对象,题面不成立。

WHY 变体只用 `official_full`:这是 `eval_multiref.py:VARIANTS` 里的已知标签,
所以**不需要改那个文件**(R0:远端不改既有 .py)。加新变体才要动 VARIANTS 表。

另外附带 5 条 S0 锚点任务(逐字复制,3 个变体全带)——见 `--help` 里的 WHY。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

# ------------------------------------------------------------------ 常数(改动 = 换一批任务,不许悄悄改)

SALT = "nf-v1"          # 选样哈希盐。改它 = 选出另一批任务 = 已生成的图作废。
SEED_OFFSET = 7_000_000  # 噪声偏移。固定写死,不从命令行传——传参就意味着可以试到"好看"为止。
N_S1 = 20
N_S3 = 10

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JSON = os.path.join(REPO, "datasets/eval_multiref/eval_set.json")
OUT_JSON = os.path.join(REPO, "datasets/eval_multiref/noise_floor_tasks.json")
RESULTS = os.path.join(REPO, "output/eval_multiref/results.json")

TEACHER = "official_full"
# S0 锚点自检(`eval_multiref.py --check_anchor`)硬编码要 5 条 S0 且比 3 个变体,
# 少一个变体就判"缺文件"。所以 S0 必须逐字复制、variants 原样保留。
ANCHOR_VARIANTS = ["official_full", "ours_kv_pre", "ours_kv_post4000"]


def sort_key(task_id: str) -> str:
    """确定性选样序。md5 而非 sorted(task_id):后者会让选出来的全挤在编号前段。"""
    return hashlib.md5(f"{task_id}|{SALT}".encode()).hexdigest()


def pick(tasks: list[dict], stratum: str, group_of, n: int) -> list[dict]:
    """按 sort_key 贪心取 n 条,**同一 group 只取一条**。

    WHY 要去重:20 条噪声地板如果挤在 3 个主体组合上,测出来的是"这 3 个组合有多稳",
    不是地板。每条一个不同组合,才能让地板覆盖到与 S1/S3 判读同样的主体分布。
    """
    pool = sorted((t for t in tasks if t["stratum"] == stratum), key=lambda t: sort_key(t["task_id"]))
    if not pool:
        raise SystemExit(f"❌ 源任务单里没有 {stratum} 层")
    picked, seen = [], set()
    for t in pool:
        g = group_of(t)
        if g in seen:
            continue
        seen.add(g)
        picked.append(t)
        if len(picked) == n:
            break
    if len(picked) != n:
        raise SystemExit(
            f"❌ {stratum} 只凑出 {len(picked)}/{n} 条互不重复的组:"
            f"该层共 {len(pool)} 条、{len(seen)} 个不同组,不够去重着取")
    return picked


def load_generated_ok(results_path: str) -> set[tuple[str, str]]:
    """M4 已经产出且没失败的 (task_id, variant)。

    WHY 要查这个:左半边用的是 M4 的既有图。选到一条没图的源任务,
    这一对在配对阶段才会炸,而那时 GPU 已经白跑过了。
    这里查 results.json 而不是查文件本身——图在 H800 上,本地没有。
    """
    if not os.path.exists(results_path):
        raise SystemExit(f"❌ 找不到 {results_path};没有它就无法确认左半边的图存在")
    with open(results_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    ok = {(r["task_id"], r["variant"]) for r in payload["records"]
          if r["status"] in ("ok", "skipped")}
    failed = {(r["task_id"], r["variant"]) for r in payload.get("fails", [])}
    return ok - failed


def build(src_json: str, results_path: str) -> dict:
    with open(src_json, "rt", encoding="utf-8") as f:
        src = json.load(f)
    tasks = src["tasks"]
    have = load_generated_ok(results_path)

    # 只在"左半边已经有图"的任务里选。
    usable = [t for t in tasks if (t["task_id"], TEACHER) in have]
    dropped = len(tasks) - len(usable)
    if dropped:
        print(f"  源任务单 {len(tasks)} 条,其中 {dropped} 条没有 {TEACHER} 图(v2 新增的 S2),已排除")

    s1 = pick(usable, "S1", lambda t: tuple(t["meta"]["subjects"]), N_S1)
    s3 = pick(usable, "S3", lambda t: t["meta"]["subjects"][0], N_S3)
    s0 = sorted((t for t in tasks if t["stratum"] == "S0"), key=lambda t: t["task_id"])
    if len(s0) != 5:
        raise SystemExit(f"❌ S0 应有 5 条(--check_anchor 硬编码),实际 {len(s0)}")

    existing_seeds = {t["seed"] for t in tasks}
    out: list[dict] = []

    # ---- 零假设任务:同 refs / 同 prompt / 仅 seed 偏移 ----
    for t in s1 + s3:
        seed = t["seed"] + SEED_OFFSET
        if seed in existing_seeds:
            # 撞上既有 seed 不影响正确性(task_id 不同、文件名不同),但会让
            # "这张图的噪声和哪张图相同"变成一个要额外解释的巧合。直接拒绝。
            raise SystemExit(f"❌ {t['task_id']} 偏移后 seed={seed} 与既有任务撞了,换 SEED_OFFSET")
        meta = dict(t["meta"])            # 逐字复制,只加两个可追溯字段
        meta["nf_src_task_id"] = t["task_id"]
        meta["nf_seed_offset"] = SEED_OFFSET
        out.append({
            "task_id": f"NF_{t['task_id']}",
            "stratum": t["stratum"],
            "prompt": t["prompt"],
            "image_paths": list(t["image_paths"]),
            "seed": seed,
            "variants": [TEACHER],
            "meta": meta,
        })

    # ---- S0 锚点:逐字复制(含原 seed 3407 与 3 个变体) ----
    for t in s0:
        out.append({
            "task_id": t["task_id"],
            "stratum": "S0",
            "prompt": t["prompt"],
            "image_paths": list(t["image_paths"]),
            "seed": t["seed"],
            "variants": list(ANCHOR_VARIANTS),
            "meta": dict(t["meta"]),
        })

    n_img = sum(len(t["variants"]) for t in out)
    return {
        "meta": {
            "spec": "M5-noise-floor-v1",
            "purpose": "零假设对:同 refs / 同 prompt / 仅噪声不同的 teacher 自比",
            "salt": SALT,
            "seed_offset": SEED_OFFSET,
            "n_tasks": len(out),
            "n_images": n_img,
            "n_null_pairs": len(s1) + len(s3),
            "src_eval_set": os.path.relpath(src_json, REPO),
            "src_eval_set_meta": src["meta"],
            "left_dir": "output/eval_multiref",    # 左半边 = M4 既有图
            "right_dir": "output/noise_floor",     # 右半边 = 本任务单的产物
            "pair_rule": "left={src}__official_full.png  right=NF_{src}__official_full.png",
            "anchor_note": "S0 5 条是逐字复制的锚点,不参与配对,只供 --check_anchor",
        },
        "tasks": out,
    }


def verify(payload: dict, src_json: str) -> None:
    """产出自检。**在写盘之后跑,也可以单独 --verify 跑**,免得"生成"与"校验"共用同一份内存里的假设。"""
    with open(src_json, "rt", encoding="utf-8") as f:
        src_by_id = {t["task_id"]: t for t in json.load(f)["tasks"]}
    tasks = payload["tasks"]
    nf = [t for t in tasks if t["task_id"].startswith("NF_")]
    s0 = [t for t in tasks if t["stratum"] == "S0"]
    errs = []

    if len(nf) != N_S1 + N_S3:
        errs.append(f"零假设任务 {len(nf)} 条,应为 {N_S1 + N_S3}")
    if len(s0) != 5:
        errs.append(f"S0 锚点 {len(s0)} 条,应为 5")
    if len({t['task_id'] for t in tasks}) != len(tasks):
        errs.append("task_id 有重复")

    for t in nf:
        src = src_by_id.get(t["meta"]["nf_src_task_id"])
        if src is None:
            errs.append(f"{t['task_id']}: 源任务不存在")
            continue
        if t["task_id"] != f"NF_{src['task_id']}":
            errs.append(f"{t['task_id']}: task_id 与源不对应")
        if t["prompt"] != src["prompt"]:
            errs.append(f"{t['task_id']}: prompt 与源不一致 ← 零假设对就不成立了")
        if t["image_paths"] != src["image_paths"]:
            errs.append(f"{t['task_id']}: image_paths 与源不一致 ← 零假设对就不成立了")
        if t["seed"] != src["seed"] + SEED_OFFSET:
            errs.append(f"{t['task_id']}: seed 偏移不对")
        if t["seed"] == src["seed"]:
            errs.append(f"{t['task_id']}: seed 与源相同,两张图会一模一样")
        if t["variants"] != [TEACHER]:
            errs.append(f"{t['task_id']}: variants={t['variants']},应只有 {TEACHER}")

    for t in s0:
        src = src_by_id[t["task_id"]]
        for k in ("prompt", "image_paths", "seed"):
            if t[k] != src[k]:
                errs.append(f"{t['task_id']}: S0 锚点的 {k} 被改过,自检会误报")
        if t["variants"] != ANCHOR_VARIANTS:
            errs.append(f"{t['task_id']}: S0 变体应为 {ANCHOR_VARIANTS}")

    # 组内去重
    for st, key in (("S1", lambda t: tuple(t["meta"]["subjects"])),
                    ("S3", lambda t: t["meta"]["subjects"][0])):
        dup = [g for g, c in Counter(key(t) for t in nf if t["stratum"] == st).items() if c > 1]
        if dup:
            errs.append(f"{st} 有重复组:{dup}")

    if errs:
        print("\n❌ 自检未通过:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("✓ 自检通过(条数 / 源对应 / prompt+refs 逐字一致 / seed 偏移 / 变体 / 组去重)")


def summarize(payload: dict) -> None:
    tasks = payload["tasks"]
    nf = [t for t in tasks if t["task_id"].startswith("NF_")]
    print(f"\n任务 {len(tasks)} 条 / 出图 {payload['meta']['n_images']} 张 "
          f"(零假设对 {payload['meta']['n_null_pairs']} 组 + S0 锚点 5×3)")
    print(f"  分层:{dict(sorted(Counter(t['stratum'] for t in tasks).items()))}")
    cost = sum(5.1 if v == TEACHER else 2.9 for t in tasks for v in t["variants"])
    print(f"  纯 denoise ≈ {cost / 60:.1f} min(+ 模型加载 ~7 min)")
    print("\n  S1 选中的组合:")
    for t in nf:
        if t["stratum"] == "S1":
            print(f"    {t['task_id']:<22} {'+'.join(t['meta']['subjects'])}")
    print("  S3 选中的主体:")
    for t in nf:
        if t["stratum"] == "S3":
            print(f"    {t['task_id']:<22} {t['meta']['subjects'][0]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC_JSON)
    p.add_argument("--out", default=OUT_JSON)
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--verify", action="store_true", help="只校验已有的 --out,不重新生成")
    args = p.parse_args()

    if args.verify:
        if not os.path.exists(args.out):
            raise SystemExit(f"❌ {args.out} 不存在")
        with open(args.out, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        verify(payload, args.src)
        summarize(payload)
        return

    payload = build(args.src, args.results)
    verify(payload, args.src)

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)  # 原子写,和 eval_multiref.py 一个套路
    print(f"\n写入 {os.path.relpath(args.out, REPO)}")
    summarize(payload)


if __name__ == "__main__":
    sys.exit(main())
