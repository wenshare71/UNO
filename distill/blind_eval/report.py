#!/usr/bin/env python3
"""盲评统计:§11.2 的新口径。**服务器与离线报表共用这一份实现。**

WHY 单独成文件:统计口径是判据本身。M4 已经栽过一次——旧 `server.py` 把 T/S 的
含义写在两个模块常量里,错位之后 score 从 0.92 变成 0.82,而错误只有在重算时才暴露。
口径只有一份实现、且能被离线复算,才可能在下次出错时被发现。

主指标(§11.2):
  - **非平局胜率** `win_0 / (win_0 + win_1)` + **Wilson 95% CI**;
  - **平局率单列**,不并进主指标;
  - `(S+B)/(T+B)` = `(win_0+tie)/(win_1+tie)` 仍算,但只作与 M4 的可比参考。

清单约定 `key_0` = 被检验方、`key_1` = 基线,所以上面三个式子对所有 kind 通用。

用法(离线复算):
    python -m distill.blind_eval.report output/eval_multiref/pairs_m4r1.json \
                                        output/eval_multiref/blind_annotations_m4r1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """二项比例的 Wilson 得分区间。n=0 返回 None(不是 0,别把"没数据"画成"0%")。

    WHY 不用正态近似 k/n ± z·sqrt(p(1-p)/n):小 n 或 p 接近 0/1 时它会给出
    超出 [0,1] 的区间。S4 的 n=4 非平局、S2 的 n=9,正态近似在这里直接失效。
    """
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def tally(pairs: list[dict], marks: dict) -> dict:
    """一组 pair 的计数。**只读 marks[pid]["winner"](语义胜者),不碰槽位。**"""
    win = defaultdict(int)
    tie = 0
    keys = None
    for p in pairs:
        m = marks.get(p["pair_id"])
        if not m:
            continue
        if keys is None:
            keys = (p["key_0"], p["key_1"])
        elif keys != (p["key_0"], p["key_1"]):
            keys = ("mixed", "mixed")   # 混合 kind 的汇总:胜率无意义,只报计数
        w = m["winner"]
        if w == "tie":
            tie += 1
        else:
            win[w] += 1

    k0, k1 = keys or (None, None)
    w0, w1 = (win.get(k0, 0), win.get(k1, 0)) if k0 and k0 != "mixed" else (0, 0)
    n = w0 + w1 + tie
    nontie = w0 + w1
    out = {
        "key_0": k0, "key_1": k1,
        "win_0": w0, "win_1": w1, "tie": tie, "n": n,
        "tie_rate": tie / n if n else None,
        "nontie_n": nontie,
        "nontie_win_rate": w0 / nontie if nontie else None,
        "wilson95": wilson(w0, nontie),
        "legacy_score": (w0 + tie) / (w1 + tie) if (w1 + tie) else None,
    }
    if k0 == "mixed":
        out["wins"] = dict(sorted(win.items()))
    return out


def position_bias(pairs: list[dict], marks: dict) -> dict:
    """左右偏好检验:抛开语义,单看"选了左边还是右边"。

    这是 §11.1 问题 A 的一半。若显著偏离 50%,它就是一个必须从所有其它数字里
    扣掉的系统偏差项——而不是某个模型真的更好。
    """
    L = R = tie = 0
    for p in pairs:
        m = marks.get(p["pair_id"])
        if not m:
            continue
        c = m["choice"]
        if c == "tie":
            tie += 1
        elif c == "L":
            L += 1
        elif c == "R":
            R += 1
    n = L + R
    return {"L": L, "R": R, "tie": tie,
            "left_rate": L / n if n else None,
            "wilson95": wilson(L, n)}


def replay_agreement(pairs: list[dict], marks: dict, prev_marks: dict) -> dict:
    """重放一致率:同一对图两次判定是否相同(§11.1 问题 B)。

    比的是**语义胜者**,不是槽位——重放刻意重掷了槽位,比槽位只会得到噪声。
    """
    same = diff = missing = 0
    flips = []
    for p in pairs:
        if p.get("kind") != "replay":
            continue
        now = marks.get(p["pair_id"])
        before = prev_marks.get(p.get("replay_of", ""))
        if not now or not before:
            missing += 1
            continue
        if now["winner"] == before["winner"]:
            same += 1
        else:
            diff += 1
            flips.append({"src_task_id": p["src_task_id"],
                          "m4": before["winner"], "now": now["winner"]})
    n = same + diff
    return {"same": same, "diff": diff, "n": n, "unannotated": missing,
            "agreement": same / n if n else None,
            "wilson95": wilson(same, n), "flips": flips}


def full_report(manifest: dict, marks: dict, prev_marks: dict | None = None) -> dict:
    pairs = manifest["pairs"]
    kinds = sorted({p["kind"] for p in pairs})
    strata = sorted({p["stratum"] for p in pairs})

    out = {
        "batch_id": manifest["meta"].get("batch_id"),
        "n_total": len(pairs),
        "n_annotated": sum(1 for p in pairs if p["pair_id"] in marks),
        "by_kind": {k: tally([p for p in pairs if p["kind"] == k], marks) for k in kinds},
        "by_kind_stratum": {
            f"{k}/{st}": tally([p for p in pairs
                                if p["kind"] == k and p["stratum"] == st], marks)
            for k in kinds for st in strata
            if any(p["kind"] == k and p["stratum"] == st for p in pairs)},
        "position": {k: position_bias([p for p in pairs if p["kind"] == k], marks)
                     for k in kinds},
        "position_overall": position_bias(pairs, marks),
    }
    out["complete"] = out["n_annotated"] == out["n_total"]
    if prev_marks is not None:
        out["replay"] = replay_agreement(pairs, marks, prev_marks)
    return out


# ------------------------------------------------------------------ 打印

def _ci(w) -> str:
    return "—" if w is None else f"[{w[0]:.3f}, {w[1]:.3f}]"


def _rate(x) -> str:
    return "—" if x is None else f"{x:6.1%}"


def print_report(rep: dict) -> None:
    print(f"批次 {rep['batch_id']}  已标 {rep['n_annotated']}/{rep['n_total']}"
          f"{'  (完整)' if rep['complete'] else '  ⚠️ 未标完,以下为部分结果'}")

    print(f"\n{'分组':<26}{'n':>5}{'win0':>6}{'win1':>6}{'tie':>5}"
          f"{'平局率':>8}{'非平局胜率':>11}  {'Wilson 95% CI':<18}{'旧口径':>8}")
    for name, t in list(rep["by_kind"].items()) + list(rep["by_kind_stratum"].items()):
        legacy = "—" if t["legacy_score"] is None else f"{t['legacy_score']:.3f}"
        print(f"{name:<26}{t['n']:>5}{t['win_0']:>6}{t['win_1']:>6}{t['tie']:>5}"
              f"{_rate(t['tie_rate']):>8}{_rate(t['nontie_win_rate']):>11}  "
              f"{_ci(t['wilson95']):<18}{legacy:>8}")
    ks = {t["key_0"]: t["key_1"] for t in rep["by_kind"].values() if t["key_0"]}
    for k0, k1 in ks.items():
        print(f"    win0={k0}   win1={k1}")

    print(f"\n左右偏好(位置偏差检验){'':<4}{'L':>5}{'R':>5}{'tie':>5}"
          f"{'选左率':>8}  Wilson 95% CI")
    for name, p in list(rep["position"].items()) + [("总体", rep["position_overall"])]:
        print(f"  {name:<26}{p['L']:>5}{p['R']:>5}{p['tie']:>5}"
              f"{_rate(p['left_rate']):>8}  {_ci(p['wilson95'])}")
    print("  判读:CI 含 0.5 = 无位置偏好;不含 = 所有其它数字都要扣掉这个偏差项")

    if "replay" in rep:
        r = rep["replay"]
        print(f"\n重放自洽率:{r['same']}/{r['n']} = {_rate(r['agreement'])}  "
              f"{_ci(r['wilson95'])}" + (f"  (未标 {r['unannotated']})" if r['unannotated'] else ""))
        for fl in r["flips"]:
            print(f"    翻转 {fl['src_task_id']:<14} M4={fl['m4']:<20} → 本次={fl['now']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="盲评离线复算")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("marks", type=Path)
    ap.add_argument("--prev_marks", type=Path, default=None,
                    help="重放对照的旧标注(算自洽率用)")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非表格")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    marks = json.loads(args.marks.read_text(encoding="utf-8"))["marks"]
    prev = None
    if args.prev_marks:
        prev = json.loads(args.prev_marks.read_text(encoding="utf-8"))["marks"]

    rep = full_report(manifest, marks, prev)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
