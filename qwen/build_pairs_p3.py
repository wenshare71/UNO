#!/usr/bin/env python3
"""P3 §8 偏好盲评的配对清单:`p3_iso_post` vs `p3_full` **240 对** + `run_floor` **30 对**。

WHY 不用 `distill/build_pairs.py` 的 `m6` 子命令:那条路把 320/30/350 的条数、
`M6_ISO/M6_FULL` 的变体名、`{task_id}__{variant}.png` 的文件名全写死了。P3 三样都不同
(240 条子集、变体靠**目录**分而不是文件名后缀、天花板来自 `p3_floor/a` 与 `/b`)。
R0 不许改 `distill/` 下既有的 `.py`,所以另起本文件,**逻辑照抄 `cmd_m6`**,
配对/打散/清单落盘全部 import 复用,不重写。

口径来自 `qwen/PLAN.md` §3.3 + §4(预登记,不许改):

  · 主判 = `qwen_iso_post` vs `qwen_full`,240 对;同批混入 30 条 run_floor。
  · **key_0 = iso_post(被检验方)/ key_1 = full(基线)** —— 与清单 schema 的全局约定
    一致,于是非平局胜率 `win_0/(win_0+win_1)` 直接就是「隔离腿相对全注意力的胜率」,
    `M4_EVAL_SPEC.md` §8.2 的 CI 下界 ≥ 0.40 原样套用,报告里不做方向翻转。
  · `iso_pre` **不进本批**。PLAN §3.3 把它派给 §9 客观身份留存计数,不是 §8 的偏好对。

⚠️ **本批 run_floor 的 30 对,两侧逐位相同**(`reports/20260814-p3-eval/REPORT.md` §4.3:
a/b 两个进程两张卡,30 对像素差 mean=max=0)。P3 的流水线是位级确定的,**run 噪声不存在**,
所以这 30 对量不到「会话漂移」。留着它们量的是另一件同样要紧的事:
**标注者面对两张一模一样的图,有多大比例会打平** —— 这把尺子分辨率的绝对上界。
§8.4 说「等价的东西也只有三成会被判成等价」,本批给它一个硬标定点。
结论里必须按这个含义写,不能写成「run 噪声天花板」。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from distill.build_pairs import BLIND_SEEDS, h, load_json, make_pair, twin_gaps, write_manifest
from distill.blind_eval.pairing import check_manifest
from infer_iso import load_tasks

BLIND_SEED = "p3-qwen-iso-20260815"
ORDER_SALT = "p3-order-v1"

MAIN_DIRS = {"p3_iso_post": "output/p3_iso_post", "p3_full": "output/p3_full"}
FLOOR_DIRS = {"p3_full_run_a": "output/p3_floor/a", "p3_full_run_b": "output/p3_floor/b"}
OUT_PAIRS = os.path.join(_REPO, "output/p3_eval/pairs_p3.json")

N_MAIN, N_FLOOR = 240, 30


def fresh_seed() -> str:
    """盲种守卫。`distill.build_pairs.fresh_seed` 还要求批次登记进它自己的
    `BLIND_SEEDS` 字典——那要改 `distill/` 下的 .py(R0)。这里只保留**防撞**那一半。"""
    clash = sorted(b for b, s in BLIND_SEEDS.items() if s == BLIND_SEED)
    if clash:
        raise SystemExit(f"❌ 盲种 {BLIND_SEED} 已被批次 {clash} 用过,槽位已污染,必须换一个")
    return BLIND_SEED


def img(dir_key: str, task_id: str, main: bool) -> str:
    """P3 的图名是 `{task_id}.png`,变体靠目录分(`infer_iso.py:289`),没有 `__variant` 后缀。"""
    return f"{(MAIN_DIRS if main else FLOOR_DIRS)[dir_key]}/{task_id}.png"


def spread_floor(pairs: list[dict]) -> int:
    """把 run_floor 在**它们自己的槽位之间**重排,最大化「锚点与其孪生主对」的最小间距。

    逻辑照抄 `distill.build_pairs.spread_runfloor`(二分间距 + 二分图匹配,确定性),
    只加一件那边没有的事:**本批 30 条锚点里有 8 条不在 240 子集内**(m6_floor 是从 320
    全表抽的),它们在批里只出现一次、没有孪生,原函数会 `KeyError`。这里给它们
    `tw=None` ⇒ 任何槽位都可以,不参与间距约束。
    """
    slots = [i for i, p in enumerate(pairs) if p["kind"] == "run_floor"]
    if not slots:
        return 0
    twin = {p["src_task_id"]: i for i, p in enumerate(pairs) if p["kind"] != "run_floor"}
    rfs = [pairs[i] for i in slots]
    tw = [twin.get(p["src_task_id"]) for p in rfs]

    def match(g: int):
        adj = [[j for j, s in enumerate(slots) if tw[i] is None or abs(s - tw[i]) >= g]
               for i in range(len(tw))]
        mt = [-1] * len(slots)

        def aug(i: int, vis: list[bool]) -> bool:
            for j in adj[i]:
                if vis[j]:
                    continue
                vis[j] = True
                if mt[j] == -1 or aug(mt[j], vis):
                    mt[j] = i
                    return True
            return False

        for i in range(len(tw)):
            if not aug(i, [False] * len(slots)):
                return None
        return mt

    lo, hi, best = 0, len(pairs), (0, list(range(len(slots))))
    while lo <= hi:
        mid = (lo + hi) // 2
        m = match(mid)
        if m is None:
            hi = mid - 1
        else:
            best, lo = (mid, m), mid + 1
    g_best, mt = best
    for j, i in enumerate(mt):
        pairs[slots[j]] = rfs[i]
    return g_best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true",
                    help="只核条数与打散统计,不查图、不写清单(本地没有图时用)")
    args = ap.parse_args()

    fresh_seed()
    main_tasks, _ = load_tasks("m6")                    # infer_iso 内部断言 == 240
    floor_tasks = load_json(os.path.join(_REPO, "datasets/eval_multiref/m6_floor_tasks.json"))["tasks"]

    pairs = []
    for t in main_tasks:
        pairs.append(make_pair(f"p3::main::{t['task_id']}", "p3_iso_post_vs_p3_full", t,
                               "p3_iso_post", img("p3_iso_post", t["task_id"], True),
                               "p3_full", img("p3_full", t["task_id"], True)))
    for t in floor_tasks:
        pairs.append(make_pair(f"p3::rf::{t['task_id']}", "run_floor", t,
                               "p3_full_run_a", img("p3_full_run_a", t["task_id"], False),
                               "p3_full_run_b", img("p3_full_run_b", t["task_id"], False)))

    if len(main_tasks) != N_MAIN or len(floor_tasks) != N_FLOOR or len(pairs) != N_MAIN + N_FLOOR:
        raise SystemExit(f"❌ 主 {len(main_tasks)} / 锚点 {len(floor_tasks)} / 配对 {len(pairs)},"
                         f"应为 {N_MAIN} / {N_FLOOR} / {N_MAIN + N_FLOOR}(PLAN §3.3)")

    # 打散用与槽位不同的盐,免得「排在前面」和「在左边」相关(同 cmd_m6)
    pairs.sort(key=lambda p: h(p["pair_id"], ORDER_SALT, "order"))
    before = twin_gaps(pairs)["min"]
    g_min = spread_floor(pairs)
    gaps = twin_gaps(pairs)
    gaps |= {"achieved_min_gap": g_min, "min_before_spread": before}
    if g_min < len(pairs) // 3:
        raise SystemExit(f"❌ 槽位内重排后最小间距 {g_min} < {len(pairs) // 3}(批长/3,既有纪律)。"
                         f"把 ORDER_SALT 改成 -v2 重试,不要降低要求。")

    if not args.dry_run:
        missing = [p[k] for p in pairs for k in ("img_0", "img_1")
                   if not os.path.exists(os.path.join(_REPO, p[k]))]
        if missing:
            raise SystemExit(f"❌ 缺 {len(missing)} 张图,例:{missing[:3]}。图不齐不许开评。")
        os.makedirs(os.path.dirname(OUT_PAIRS), exist_ok=True)
        errs = check_manifest({"meta": {"blind_seed": BLIND_SEED}, "pairs": pairs}, Path(_REPO))
        if errs:
            raise SystemExit("❌ 清单自检未过:\n  - " + "\n  - ".join(errs[:10]))
        write_manifest(OUT_PAIRS, {
            "batch_id": "p3",
            "blind_seed": BLIND_SEED,
            # 逐字沿用 m6 的问法,改问题 = 改尺子
            "question": "哪一张更好?(综合参考图忠实度与画面质量)",
            "eval_set_version": "m6_tasks v1 的 i%4!=3 子集(S1 165 + S3 75 = 240,"
                                "PLAN §3.3)+ m6_floor_tasks v1(30,来自 p3_floor/a 与 /b)",
            "source": "PLAN §3.3 / §4 预登记:iso_post vs full 240 对 + run_floor 30 对,不带 replay",
            "twin_gaps": gaps,
        }, pairs)

    print(f"\n条数:{len(pairs)} 对 {dict(sorted(Counter(p['kind'] for p in pairs).items()))}")
    by_st = Counter((p["kind"], p["stratum"]) for p in pairs)
    for (k, st), c in sorted(by_st.items()):
        print(f"  {k:<24}{st:<5}{c:>5}")
    print(f"\n方向约定:key_0 = p3_iso_post(被检验方)/ key_1 = p3_full(基线)")
    print("  ⇒ 非平局胜率 < 0.5 一律读作「隔离腿更差」,不翻转。")
    print(f"  §8.2 要 n_nontie ≥ 94 ⇒ 平局率 > {1 - 94 / N_MAIN:.1%} 时结论是"
          f"「判据不适用」而非「不达标」,**不许事后追加样本**")
    print(f"\n锚点孪生间距:纯 md5 最小 {gaps['min_before_spread']} → 槽位内重排后 "
          f"最小 {gaps['min']} / 中位 {gaps['median']} / 最大 {gaps['max']}(批长 {gaps['batch_len']};"
          f"{N_FLOOR - gaps['n_twin']} 条锚点不在 240 子集内,批里只出现一次)")
    tert = Counter(("头", "中", "尾")[min(2, i * 3 // len(pairs))]
                   for i, p in enumerate(pairs) if p["kind"] == "run_floor")
    print(f"run_floor 三分位分布 {dict(tert)}(槽位未动,不聚堆)")
    print(f"\n盲种 {BLIND_SEED}")
    print("⚠️ 随结论必须一起声明:")
    print("  ① 蒸馏 target 由官方**全注意力** teacher 生成 ⇒ 目标分布对基线腿有利;")
    print("  ② 本批 run_floor 两侧**逐位相同**(流水线位级确定),它量的是"
          "「标注者对一模一样的两张图的打平率」,**不是** run 噪声天花板;")
    print("  ③ 单标注者,无标注者间一致性(§8.5-1)。")


if __name__ == "__main__":
    main()
