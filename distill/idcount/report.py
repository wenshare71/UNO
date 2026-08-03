#!/usr/bin/env python3
"""身份留存计数——离线统计。

复用 `distill.blind_eval.report.wilson`(Wilson 95% 区间)而不重新实现,理由与 blind_eval
自己那句注释一样:口径只有一份实现,才可能在下次算错时被发现——M4 已经吃过一次
"两处口径各写一份、悄悄错位"的亏(score 0.92→0.82,靠重算才捞回来)。

两个口径都要报,不能只报一个:
  - **per-subject 留存率** = 保住的参考主体数 / 提问总数。这是最细粒度的信号。
  - **per-image 留存率** = 该图全部参考主体都保住才计 1。同一张图里的两个主体
    是相关的(同一次生成、同一套条件),per-subject 把它们当独立样本会低估方差,
    per-image 是保守口径,两个都给读者自己判断信哪个。

重放 item 是原 item 的复制品,**先从主统计里剔除**,否则同一张图的信号被计两次,
样本量看着涨了、实际信息量没涨,还会悄悄压窄置信区间。

用法:
    python -m distill.idcount.report output/probe_iso/idcount_items.json \
                                      output/probe_iso/idcount_marks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from distill.blind_eval.report import wilson  # noqa: E402  区间估计不重新实现


def subject_tally(items: list[dict], marks: dict) -> dict:
    """per-subject 留存率:回答"是"的参考主体数 / 已标注的提问总数。"""
    k = n = 0
    for it in items:
        m = marks.get(it["item_id"])
        if not m:
            continue
        answers = m["answers"]
        n += len(answers)
        k += sum(1 for a in answers if a)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def image_tally(items: list[dict], marks: dict) -> dict:
    """per-image 留存率:该图全部参考主体都保住(answers 全 True)才计 1。"""
    k = n = 0
    for it in items:
        m = marks.get(it["item_id"])
        if not m:
            continue
        n += 1
        if m["answers"] and all(m["answers"]):
            k += 1
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def replay_consistency(items: list[dict], marks: dict) -> dict:
    """重放自洽率:同一张图前后两次标的 answers **逐项相同**才算一致(不是"多数一致")。

    比较对象是 answers 整个列表,不是单个主体——一张图两个主体,其中一个翻了
    也算"这次判断不一致",这是比"平均留存率一样"更严的口径,故意的。
    """
    same = diff = missing = 0
    mismatches: list[dict] = []
    for it in items:
        rof = it.get("replay_of")
        if not rof:
            continue
        now = marks.get(it["item_id"])
        orig = marks.get(rof)
        if not now or not orig:
            missing += 1
            continue
        if now["answers"] == orig["answers"]:
            same += 1
        else:
            diff += 1
            mismatches.append({
                "item_id": it["item_id"], "replay_of": rof,
                "task_id": it["task_id"], "stratum": it["stratum"],
                "variant": it.get("variant"),
                "orig_answers": orig["answers"], "replay_answers": now["answers"],
            })
    n = same + diff
    return {"same": same, "diff": diff, "n": n, "missing": missing,
            "agreement": same / n if n else None, "wilson95": wilson(same, n),
            "mismatches": mismatches}


def full_report(manifest: dict, marks: dict) -> dict:
    items = manifest["items"]
    main_items = [it for it in items if not it.get("replay_of")]  # 剔除重放,见模块 docstring
    variants = sorted({it["variant"] for it in main_items})
    strata = sorted({it["stratum"] for it in main_items})

    def subset(variant: str | None = None, stratum: str | None = None) -> list[dict]:
        rows = main_items
        if variant is not None:
            rows = [it for it in rows if it["variant"] == variant]
        if stratum is not None:
            rows = [it for it in rows if it["stratum"] == stratum]
        return rows

    out = {
        "n_total": len(items),
        "n_main": len(main_items),
        "n_annotated": sum(1 for it in items if it["item_id"] in marks),
        "by_variant_subject": {v: subject_tally(subset(v), marks) for v in variants},
        "by_variant_image": {v: image_tally(subset(v), marks) for v in variants},
        "by_variant_stratum_subject": {
            f"{v}/{s}": subject_tally(subset(v, s), marks)
            for v in variants for s in strata if subset(v, s)},
        "by_variant_stratum_image": {
            f"{v}/{s}": image_tally(subset(v, s), marks)
            for v in variants for s in strata if subset(v, s)},
        "replay": replay_consistency(items, marks),
    }
    out["complete"] = out["n_annotated"] == out["n_total"]
    return out


# ------------------------------------------------------------------ 打印

def _ci(w) -> str:
    return "—" if w is None else f"[{w[0]:.3f}, {w[1]:.3f}]"


def _rate(x) -> str:
    return "—" if x is None else f"{x:6.1%}"


def _table(title: str, groups: dict) -> None:
    print(f"\n{title}")
    print(f"  {'分组':<20}{'k':>5}{'n':>5}{'留存率':>9}  {'Wilson 95% CI':<18}")
    for name, t in groups.items():
        print(f"  {name:<20}{t['k']:>5}{t['n']:>5}{_rate(t['rate']):>9}  {_ci(t['wilson95']):<18}")


def print_report(rep: dict) -> None:
    print(f"已标 {rep['n_annotated']}/{rep['n_total']}"
          f"{'  (完整)' if rep['complete'] else '  ⚠️ 未标完,以下为部分结果'}"
          f"(主统计已剔除 {rep['n_total'] - rep['n_main']} 个重放 item)")

    _table("per-subject 留存率(按变体)", rep["by_variant_subject"])
    _table("per-image 留存率(按变体,全部主体都保住才计 1)", rep["by_variant_image"])
    _table("per-subject 留存率(按变体 × 层)", rep["by_variant_stratum_subject"])
    _table("per-image 留存率(按变体 × 层)", rep["by_variant_stratum_image"])

    r = rep["replay"]
    print(f"\n重放自洽率:{r['same']}/{r['n']} = {_rate(r['agreement'])}  {_ci(r['wilson95'])}"
          + (f"  (未标 {r['missing']})" if r["missing"] else ""))
    for m in r["mismatches"]:
        print(f"    不一致 {m['item_id']:<10} (replay_of={m['replay_of']}) "
              f"{m['task_id']:<16} 原={m['orig_answers']}  重放={m['replay_answers']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="身份留存计数——离线统计")
    ap.add_argument("items", type=Path)
    ap.add_argument("marks", type=Path)
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非表格")
    args = ap.parse_args()

    manifest = json.loads(args.items.read_text(encoding="utf-8"))
    marks = json.loads(args.marks.read_text(encoding="utf-8"))["marks"]

    rep = full_report(manifest, marks)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
