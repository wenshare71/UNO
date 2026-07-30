"""从 `eval_set.json` 里**按固定规则挑出** 8 个注意力诊断样本(D02 定的是 5–8)。

产物:`datasets/eval_multiref/attn_diag_tasks.json`

## 为什么从 eval_set.json 挑,而不是另建一套

同 task_id、同 prompt、**同 seed**,所以诊断样本和 M4 Stage B 的产物**就是同一张图**。
注意力曲线可以直接钉到人工评测看的那张图上,不需要"另一套样本各说各话"。
也因此,Stage B 先跑后跑都不影响对齐——这把 D02 里"等评测结果再定样本"的先后依赖消掉了。

## 挑选规则(写死,不留自由度)

  - **4 × S1**:44 个合法 2-组合里等距取 k = 0 / 11 / 22 / 33,`seed_slot = 0`。
    等距是为了跨 subject、跨模板铺开,而不是挑好看的。
  - **2 × S2**:同类对(bear_plushie + grey_sloth_plushie)的前两个模板。
    S2 是主体复制的探针——**同类对上肉眼最难判、注意力最好判**,正是可视化的价值区。
    D01/D02 写于 07-29,当时只知道"丢第二主体",还不知道有"复制"这个失败模式(D03 修正 ④)。
  - **2 × S4**:3-ref 的前两个组合,压一下段数更多时的行为。

### 顺带得到的受控对照

`S1_011`(bear_plushie + candle)与 `S2_000`(bear_plushie + grey_sloth_plushie):
**同一个 subject、同一个槽位(都在 slot 0)**,只差搭档是不是同类。
若复制是类别碰撞驱动的,差异应当出现在这一对上。

**槽位必须对齐,模板对不齐。** 槽位本身就是被研究的变量(ref_2 是否被丢/被复制),
拿槽位不同的两个任务作对照会把"槽位"和"类别碰撞"混在一起,对照就废了。
而模板绑死在组合序号上(`template_id = k % 20`):bear_plushie 落在 slot 0 的组合是
k = 9..15 → 模板 9..15,S2 的组合只带模板 0..4,**eval_set 里不存在同模板的配对**。
要同模板就得离开 eval_set 另造任务,那会丢掉"与 Stage B 同一张图"的 seed 对齐——
那个性质更值钱。所以保槽位、弃模板;解读时按**对照**说,不按严格控制变量说。

## 变体

覆盖成 4 件套(D03 修正 ③):eval_set.json 里 S1/S4 只有 3 个变体,S2 有 4 个,
但那第 4 个是 `ours_kv_post2000`(过训探针),与本实验要的 `ours_iso_nocache` 不是一回事。
"""
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_data import HELD_OUT, TRAIN  # noqa: E402

EVAL_JSON = "datasets/eval_multiref/eval_set.json"
OUT_JSON = "datasets/eval_multiref/attn_diag_tasks.json"

S1_PICKS = (0, 11, 22, 33)
S2_PICKS = (0, 1)
S4_PICKS = (0, 1)

# 四件套(D03 修正 ③):加 ours_iso_nocache 让 D02 的验收产物 #3 可达,
# 同时把"隔离"与"缓存"两个变量拆开。
DIAG_VARIANTS = ["official_full", "ours_kv_pre", "ours_kv_post4000", "ours_iso_nocache"]


def pick(tasks: list[dict], stratum: str, idxs: tuple[int, ...]) -> list[dict]:
    """取该层 seed_slot==0 的任务(即每个组合的第一个 seed),再按组合序号取。"""
    pool = [t for t in tasks if t["stratum"] == stratum and t["meta"]["seed_slot"] == 0]
    pool.sort(key=lambda t: t["task_id"])
    out = []
    for k in idxs:
        if k >= len(pool):
            raise SystemExit(f"❌ {stratum} 只有 {len(pool)} 个组合,取不到 k={k}")
        out.append(pool[k])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", default=EVAL_JSON)
    p.add_argument("--out", default=OUT_JSON)
    args = p.parse_args()

    with open(args.eval_json, "rt", encoding="utf-8") as f:
        all_tasks = json.load(f)["tasks"]

    chosen = (pick(all_tasks, "S1", S1_PICKS)
              + pick(all_tasks, "S2", S2_PICKS)
              + pick(all_tasks, "S4", S4_PICKS))

    if len(chosen) != 8:
        raise SystemExit(f"❌ 应挑出 8 个样本,实际 {len(chosen)}")

    held, train = set(HELD_OUT), set(TRAIN)
    for t in chosen:
        subs = t["meta"]["subjects"]
        leaked = [s for s in subs if s in train]
        if leaked:
            raise SystemExit(
                f"❌ {t['task_id']} 含蒸馏训练用过的 subject {leaked}——"
                f"评测集泄漏,曲线会自欺(手册 R1)")
        outside = [s for s in subs if s not in held]
        if outside:
            raise SystemExit(f"❌ {t['task_id']} 含非 held-out subject {outside}")
        if len(subs) != t["meta"]["n_refs"]:
            raise SystemExit(f"❌ {t['task_id']} subjects 数与 n_refs 不符")

    s1 = chosen[:4]
    tmpl_ids = {t["meta"]["template_id"] for t in s1}
    if len(tmpl_ids) != 4:
        raise SystemExit(f"❌ 4 个 S1 样本应落在 4 个不同模板上,实际 {sorted(tmpl_ids)}——"
                         f"模板撞车会把场景因素和主体因素混在一起")

    s2 = chosen[4:6]
    for t in s2:
        cls = t["meta"]["classes"]
        if len(set(cls)) != 1:
            raise SystemExit(f"❌ S2 样本 {t['task_id']} 不是同类对({cls}),"
                             f"复制探针的前提没了")

    # 受控对照:S1_011 与 S2_000 必须是"同一 subject 在同一槽位,只换搭档类别"。
    # 只校验槽位对齐,**不校验模板**——模板绑死在组合序号上(template_id = k % 20),
    # bear_plushie 在 slot 0 的组合是 k=9..15(模板 9..15),S2 只有模板 0..4,
    # eval_set 里不存在同模板配对。详见模块 docstring。
    a, b = chosen[1], chosen[4]
    if a["meta"]["subjects"][0] != b["meta"]["subjects"][0]:
        raise SystemExit(
            f"❌ 对照失效:{a['task_id']} 的 slot0 是 {a['meta']['subjects'][0]},"
            f"{b['task_id']} 的 slot0 是 {b['meta']['subjects'][0]}——"
            f"槽位不对齐,类别碰撞与槽位效应会混在一起")
    if a["meta"]["classes"][0] == a["meta"]["classes"][1]:
        raise SystemExit(f"❌ 对照失效:{a['task_id']} 本身就是同类对,对不出差异")

    s4 = chosen[6:]
    for t in s4:
        if t["meta"]["n_refs"] != 3:
            raise SystemExit(f"❌ S4 样本 {t['task_id']} 的 n_refs 是 "
                             f"{t['meta']['n_refs']},应为 3")

    # seed 必须原样保留——这是"与 Stage B 同一张图"的唯一保证
    by_id = {t["task_id"]: t for t in all_tasks}
    out_tasks = []
    for t in chosen:
        src = by_id[t["task_id"]]
        if src["seed"] != t["seed"] or src["prompt"] != t["prompt"]:
            raise SystemExit(f"❌ {t['task_id']} 的 seed/prompt 被改动过")
        out_tasks.append({**{k: v for k, v in src.items() if k != "variants"},
                          "variants": list(DIAG_VARIANTS)})

    payload = {
        "spec": "attn-diag-v1",
        "source": os.path.relpath(args.eval_json, _REPO_ROOT),
        "note": "注意力诊断样本;task_id/prompt/seed 与 eval_set.json 逐字一致,"
                "故产出图与 M4 Stage B 同一张。",
        "variants": list(DIAG_VARIANTS),
        "picks": {"S1": list(S1_PICKS), "S2": list(S2_PICKS), "S4": list(S4_PICKS)},
        "n_tasks": len(out_tasks),
        "tasks": out_tasks,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)

    print(f"✓ {len(out_tasks)} 个诊断样本 → {args.out}")
    print(f"  变体 {len(DIAG_VARIANTS)} 个:{', '.join(DIAG_VARIANTS)}")
    print(f"  共 {len(out_tasks) * len(DIAG_VARIANTS)} 次录制推理\n")
    for t in out_tasks:
        m = t["meta"]
        print(f"  {t['task_id']:<12} {'+'.join(m['subjects']):<48} "
              f"seed {t['seed']:<8} {m['template']}")
    print(f"\n  受控对照:{chosen[1]['task_id']} vs {chosen[4]['task_id']} "
          f"({chosen[1]['meta']['subjects'][0]} 都在 slot 0,只差搭档是否同类;"
          f"模板不同,按对照解读不按控制变量解读)")


if __name__ == "__main__":
    main()
