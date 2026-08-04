#!/usr/bin/env python3
"""生成 M5 臂 A 读数的任务单 `arm_a_tasks.json`(`DISTILL_PLAN.md` §11.4 P1)。

臂 A 要回答的问题:**把 M3 的数据与配方原样施加在一个已经对齐的底座上,
它本身有没有代价?** 对照是 `official_full`(不交那 4000 步的税)与
`arm_a_full`(交了),两者都是**全注意力**、都用官方 LoRA 系的权重,
于是唯一差别就是"有没有跑过那 4000 步"——三边链条的第 1 条边。

WHY 只放这两个变体、不把 `ours_kv_post4000` 一起拉进来:
`arm_a_full` 与 `ours_kv_post4000` 之间同时差**底座**和**隔离**两个变量,
不是单变量边。§11.4 原文建议报 `student vs 臂 A`,那句写在
「双方交了同一笔质量税、剩下的差就是隔离的代价」这个前提上——
而 §11.6 已经证明底座也不同,前提不成立,那个数读不出隔离的代价。
把判读预算花在一个混杂对比上,不如留给臂 B。

WHY seed 原样照抄、绝不偏移(与 `build_probe_iso.py` 同纪律):
这一批要的是"除了权重什么都不变"。换 seed 会把"配方的代价"和"噪声的代价"
混在一起(噪声地板实测 45.0% [0.258, 0.658],量级足以吞掉小效应)。

WHY S1 全部 132 + S3 全部 60 = 192 条,不抽样:
按零假设对实测 33.3% 平局率折算,非平局样本数 ≈ 192 × 0.667 ≈ 128,
满足 `M4_EVAL_SPEC.md` §8.2 的判据(Wilson 95% CI 下界 ≥ 0.40 且
n_nontie ≥ 94)。**但臂 A 是所有比较里最容易平局的一组**——它近似"在自己的
分布上原地踏步"(§11.4 固有盲区),平局率超过 51% 时 n_nontie 就跌破 94,
结论变成「判据不适用」。这条风险在上机前已知,处置见 §11.7 的预登记,
本脚本不做补救(补救 = 看到平局率之后再加样本 = 事后调判据)。

WHY 不复用 M4 已有的 `official_full` 产物、要在同一次会话里重生成:
§11.3 步骤 1 已证实本栈**不可逐位复现**(同 seed 同配置两次跑 mean|Δ| 2–3.7,
个别到 13.5),且 §8.5-3 的跨批次尺子会漂(κ=0.274)。新旧混用会把
"配方的代价"和"跨会话抖动 + 尺子漂移"焊死在一起。要换就整组换。

用法(在 UNO 仓库根目录):
    python distill/build_arm_a_tasks.py --dry_run   # 只看条数与成本
    python distill/build_arm_a_tasks.py             # 写出 + 自检
    python distill/build_arm_a_tasks.py --verify    # 只校验已有产物
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

# ------------------------------------------------------------------ 常数(改动 = 换一批任务,不许悄悄改)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JSON = os.path.join(REPO, "datasets/eval_multiref/eval_set.json")
OUT_JSON = os.path.join(REPO, "datasets/eval_multiref/arm_a_tasks.json")

# 顺序与 eval_multiref.py:VARIANTS 同序。两者 bank 不同(official / arm_a),
# 整批必然发生 1 次 swap_lora,变体外层循环保证只有 1 次。
VARIANTS = ["official_full", "arm_a_full"]
STRATA = ("S1", "S3")
N_S1, N_S3 = 132, 60
PREFIX = "AA_"

# 两个变体都是全注意力、都不开 KV cache,单张耗时应当**相同**。
# 4.86 来自 §11.3 步骤 1 的 official_full 实测(P-probe 复测 4.74,同量级)。
# 只用来给 --dry_run 一个数量级,不是本批的实测。
COST_S_PER_IMG = {"official_full": 4.86, "arm_a_full": 4.86}


def build(src_json: str) -> dict:
    with open(src_json, "rt", encoding="utf-8") as f:
        src = json.load(f)
    tasks = src["tasks"]

    picked = [t for t in tasks if t["stratum"] in STRATA]
    out: list[dict] = []
    for t in picked:
        meta = dict(t["meta"])           # 逐字复制,只加一个可追溯字段
        meta["arm_a_src_task_id"] = t["task_id"]
        out.append({
            "task_id": f"{PREFIX}{t['task_id']}",
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
            "spec": "M5-arm-a-v1",
            "n_tasks": len(out),
            "n_images": n_img,
            "source": "eval_set.json S1+S3",
        },
        "tasks": out,
    }


def verify(payload: dict, src_json: str, out_json: str = OUT_JSON) -> None:
    """产出自检。写完自动跑一遍,也可以单独 `--verify` 跑,免得"生成"与"校验"
    共用同一份内存里的假设(同 `build_probe_iso.py` / `build_noise_floor.py` 的纪律)。"""
    with open(src_json, "rt", encoding="utf-8") as f:
        src_by_id = {t["task_id"]: t for t in json.load(f)["tasks"]}
    tasks = payload["tasks"]
    errs: list[str] = []
    # 参考图相对 json 所在目录解析(eval_multiref.py:load_tasks 同规则),
    # 所以必须用**实际写出的路径**,锁死模块常量会在 --out 换目录时静默校验错地方。
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
        if not t["task_id"].startswith(PREFIX):
            errs.append(f"{t['task_id']}: 缺 {PREFIX} 前缀")
            continue
        src_id = t["task_id"][len(PREFIX):]
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
            errs.append(f"{t['task_id']}: seed 与源不一致 ← 本批前提就是只差权重,这条最要命")
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
    # 用零假设对实测的 33.3% 平局率折算(SPEC §8.4),不是 build_probe_iso 里那个 30% 估计。
    n = len(tasks)
    print(f"  按零假设对实测 33.3% 平局率折合非平局样本 ≈ {round(n * 0.667)}"
          f"(§8.2 判据要求 n_nontie ≥ 94、CI 下界 ≥ 0.40)")
    breakeven = 1.0 - 94 / n
    print(f"  ⚠️ 平局率超过 {breakeven:.1%} 时 n_nontie 跌破 94 ⇒ 结论是"
          f"「判据不适用」而非「不达标」(§8.2)。臂 A 是最易平局的一组,见模块 docstring。")


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
    print("  注:两个变体都是全注意力,**没有** KV cache 的 1.67× 折扣,"
          "比 P-probe 那批贵约 1.3 倍。")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC_JSON)
    p.add_argument("--out", default=OUT_JSON)
    p.add_argument("--verify", action="store_true", help="只校验已有的 --out,不重新生成")
    p.add_argument("--dry_run", action="store_true", help="不写文件,只打印任务数与成本估算")
    p.add_argument("--num_shards", type=int, default=8,
                   help="--dry_run 时用来估算分 shard 后的单卡耗时")
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
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)  # 原子写,同 build_probe_iso / export_official_lora
    print(f"\n已写出 {args.out}({os.path.getsize(args.out) / 1024:.0f} KB)")

    # 回读复核:真正喂给 eval_multiref.py 的是**磁盘上这个文件**,不是内存里那个 dict。
    with open(args.out, "rt", encoding="utf-8") as f:
        back = json.load(f)
    print("回读复核:")
    verify(back, args.src, args.out)
    summarize(back)


if __name__ == "__main__":
    main()
