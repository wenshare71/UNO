#!/usr/bin/env python3
"""生成盲评的**配对清单**(pair manifest),以及把 M4 的 227 条旧标注迁移到新键。

WHY 要有这个文件:旧版 `blind_eval/server.py` 把"比什么"写死成两个模块级常量
(`TEACHER` / `STUDENT`),于是每换一种比较就要改一次服务器,而服务器是**判读工具**
——改它就是改尺子。M4 已经因此吃过一次亏(T/S 错位,score 0.92→0.82)。

改成清单驱动之后,服务器只认一件事:*"这一对的两张图在哪、语义标签是什么"*。
post-vs-pre、pre-vs-teacher、teacher 自比(零假设)、重放,全都只是清单的不同行。

清单 schema(`meta` + `pairs[]`):

    meta:  batch_id, blind_seed, question, created, eval_set_version, source
    pairs[]:
      pair_id      全局唯一,**标注就按它键控**(不再按列表下标——下标会随任务单增删偏移)
      kind         post_vs_pre / null_floor / replay / teacher_vs_student
      stratum      S1/S2/S3/S4(前端可见,不泄漏身份)
      prompt       前端可见
      ref_paths[]  仓库根相对路径
      key_0,key_1  两个候选的**语义标签**(服务端专用,前端永不可见)
                   **约定:key_0 = 被检验方,key_1 = 基线。** 全部 kind 一致,
                   于是"非平局胜率" = win_0/(win_0+win_1)、旧口径
                   (S+B)/(T+B) = (win_0+tie)/(win_1+tie) 在各 kind 上同一个公式。
      img_0,img_1  两个候选的图路径(仓库根相对)
      src_task_id  溯源

**`_0/_1` 是清单顺序,不是显示顺序。** 左右由 `md5(pair_id|blind_seed) % 2` 决定,
与 0/1 无关。这两件事分开命名,是为了防止"清单里排前面的就显示在左边"这种误解。

用法:
    python distill/build_pairs.py migrate-m4    # M4 227 条 → pair_id 键控 + 复算校验
    python distill/build_pairs.py r2            # 步骤 2 的 198 条批次
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_SET = os.path.join(REPO, "datasets/eval_multiref/eval_set.json")
NOISE_FLOOR = os.path.join(REPO, "datasets/eval_multiref/noise_floor_tasks.json")
M4_RESULTS = os.path.join(REPO, "output/eval_multiref/results.json")
M4_MARKS = os.path.join(REPO, "output/eval_multiref/blind_rond1.json")
M4_STATS = os.path.join(REPO, "output/eval_multiref/blind_stats.json")

OUT_M4_PAIRS = os.path.join(REPO, "output/eval_multiref/pairs_m4r1.json")
OUT_M4_MARKS = os.path.join(REPO, "output/eval_multiref/blind_annotations_m4r1.json")
OUT_R2_PAIRS = os.path.join(REPO, "output/eval_multiref/pairs_m5r1.json")

EVAL_DIR = "output/eval_multiref"      # M4 产物目录(仓库根相对)
NF_DIR = "output/noise_floor"          # 步骤 1 产物目录

TEACHER = "official_full"
PRE = "ours_kv_pre"
POST = "ours_kv_post4000"

# M4 用过的盲种。**只用于把旧标注解码回语义胜者**,新批次一律不用它
# (你已在揭盲模式下看过全部 227 条,这个种子对应的槽位已污染)。
M4_BLIND_SEED = "m4-blind-v1"
M5_BLIND_SEED = "m5-blind-v1"

M4_STRATA = ("S1", "S2", "S3", "S4")   # S0 锚点不参与人评


# ------------------------------------------------------------------ 通用

def h(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"❌ 缺少 {os.path.relpath(path, REPO)}")
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def m4_student_on_left(task_id: str) -> bool:
    """旧版 `server.py:48-51` 的槽位函数,**逐字复制**。

    迁移的正确性完全依赖它与旧版一致——改一个字符,227 条的胜者就全反了。
    """
    return int(h(task_id, M4_BLIND_SEED), 16) % 2 == 0


def m4_slot_variant(task_id: str, slot: str) -> str:
    """旧版 `server.py:54-59`,逐字复制。slot "A"=左图,"B"=右图。"""
    sol = m4_student_on_left(task_id)
    if slot == "A":
        return POST if sol else TEACHER
    return TEACHER if sol else POST


def generated_ok(results: dict) -> set[tuple[str, str]]:
    ok = {(r["task_id"], r["variant"]) for r in results["records"]
          if r["status"] in ("ok", "skipped")}
    return ok - {(r["task_id"], r["variant"]) for r in results.get("fails", [])}


def img_rel(directory: str, task_id: str, variant: str) -> str:
    return f"{directory}/{task_id}__{variant}.png"


def make_pair(pair_id: str, kind: str, task: dict, k0: str, p0: str, k1: str, p1: str,
              src_task_id: str | None = None) -> dict:
    # ref 路径在 eval_set.json 里是相对 json 所在目录的(`../dreambooth/...`),
    # 清单里统一换成**仓库根相对**——服务器因此不必再读 eval_set.json,
    # 也就不会被那个文件的版本漂移影响。
    refs = [os.path.relpath(
        os.path.normpath(os.path.join(REPO, "datasets/eval_multiref", p)), REPO)
        for p in task["image_paths"]]
    return {
        "pair_id": pair_id,
        "kind": kind,
        "stratum": task["stratum"],
        "prompt": task["prompt"],
        "ref_paths": refs,
        "key_0": k0, "img_0": p0,
        "key_1": k1, "img_1": p1,
        "src_task_id": src_task_id or task["task_id"],
    }


def write_manifest(path: str, meta: dict, pairs: list[dict]) -> None:
    ids = [p["pair_id"] for p in pairs]
    if len(set(ids)) != len(ids):
        dup = [i for i, c in Counter(ids).items() if c > 1]
        raise SystemExit(f"❌ pair_id 重复:{dup[:5]}")
    meta = dict(meta)
    meta["n_pairs"] = len(pairs)
    meta["by_kind"] = dict(sorted(Counter(p["kind"] for p in pairs).items()))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "pairs": pairs}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"写入 {os.path.relpath(path, REPO)}:{len(pairs)} 对 {meta['by_kind']}")


# ------------------------------------------------------------------ M4 迁移

def cmd_migrate_m4(_args) -> None:
    """M4 的 227 条:下标键控 → pair_id 键控,并**复算五行验证无损**。

    存的是**语义胜者**(`winner`)而不是槽位,理由:槽位只有配上当年那个盲种才有意义,
    而盲种是会换的。存胜者之后,统计只有一条代码路径,新旧批次共用。
    """
    eval_set = load_json(EVAL_SET)
    results = load_json(M4_RESULTS)
    old = load_json(M4_MARKS)
    ref_stats = load_json(M4_STATS)

    by_id = {t["task_id"]: t for t in eval_set["tasks"]}
    have = generated_ok(results)
    marks_in = old["marks"]

    if old["meta"].get("blind_seed") != M4_BLIND_SEED:
        raise SystemExit(f"❌ 旧标注的 blind_seed 是 {old['meta'].get('blind_seed')},"
                         f"不是 {M4_BLIND_SEED};迁移会算错胜者")

    task_ids = [m["task_id"] for m in marks_in.values()]
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("❌ 旧标注里有重复 task_id,无法按 task_id 迁移")

    pairs, new_marks = [], {}
    for m in marks_in.values():
        tid = m["task_id"]
        t = by_id.get(tid)
        if t is None:
            raise SystemExit(f"❌ 标注里的 {tid} 在 eval_set.json 中不存在")
        for v in (TEACHER, POST):
            if (tid, v) not in have:
                raise SystemExit(f"❌ {tid}/{v} 没有成功产出的图,迁移无意义")
        choice = m["choice"]
        winner = "tie" if choice == "tie" else m4_slot_variant(tid, choice)
        pid = f"m4r1::tvs::{tid}"
        pairs.append(make_pair(pid, "teacher_vs_student", t,
                               POST, img_rel(EVAL_DIR, tid, POST),      # key_0 = 被检验方
                               TEACHER, img_rel(EVAL_DIR, tid, TEACHER)))
        new_marks[pid] = {
            # 旧版槽位叫 A/B(`server.py:55` 注释:"A"=左图、"B"=右图),新版叫 L/R。
            # 直接换名而不是留 A/B:留着的话位置偏差检验会在 M4 数据上静默算出 0/0,
            # 而 M4 的左右偏好恰恰是**免费就能测出来的**一个真实数字。
            "choice": {"A": "L", "B": "R", "tie": "tie"}[choice],
            "winner": winner,          # 语义胜者,统计只读这个
            "ts": m.get("ts"),
            "migrated_from": {"blind_seed": M4_BLIND_SEED, "rond": 1,
                              "slot_rename": "A→L, B→R"},
        }

    # 顺序按 eval_set 里的任务顺序,便于人工对照(不影响任何计算)
    order = {t["task_id"]: i for i, t in enumerate(eval_set["tasks"])}
    pairs.sort(key=lambda p: order[p["src_task_id"]])

    write_manifest(OUT_M4_PAIRS, {
        "batch_id": "m4r1",
        "blind_seed": M4_BLIND_SEED,
        "question": "哪一张更好?(综合参考图忠实度与画面质量)",
        "eval_set_version": "v1-232",
        "source": "由 blind_rond1.json 迁移,仅供复算与重放取样,不用于新标注",
    }, pairs)

    tmp = OUT_M4_MARKS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": {"batch_id": "m4r1", "blind_seed": M4_BLIND_SEED,
                            "migrated_from": "output/eval_multiref/blind_rond1.json",
                            "n_marks": len(new_marks)},
                   "marks": new_marks}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_M4_MARKS)
    print(f"写入 {os.path.relpath(OUT_M4_MARKS, REPO)}:{len(new_marks)} 条")

    verify_migration(pairs, new_marks, ref_stats)


def verify_migration(pairs: list[dict], marks: dict, ref: dict) -> None:
    """**验收:复算出的五行必须与 §6.3 的表逐值吻合,否则迁移作废。**"""
    st_of = {p["pair_id"]: p["stratum"] for p in pairs}

    def tally(pids) -> dict:
        T = S = B = 0
        for pid in pids:
            w = marks[pid]["winner"]
            if w == "tie":
                B += 1
            elif w == POST:
                S += 1
            elif w == TEACHER:
                T += 1
            else:
                raise SystemExit(f"❌ {pid} 的 winner 是未知标签 {w}")
        score = (S + B) / (T + B) if (T + B) else None
        return {"T": T, "S": S, "B": B, "n": T + S + B, "score": score}

    rows = {"总体": (tally(marks.keys()), ref["overall"])}
    for st in M4_STRATA:
        pids = [p for p in marks if st_of[p] == st]
        rows[st] = (tally(pids), ref["by_stratum"][st])

    print("\n迁移复算 vs blind_stats.json(§6.3 表):")
    print(f"{'层':<6}{'T':>5}{'S':>5}{'B':>5}{'n':>6}{'score':>9}   判定")
    bad = 0
    for name, (got, want) in rows.items():
        same = all(got[k] == want[k] for k in ("T", "S", "B", "n")) and \
            abs((got["score"] or 0) - (want["score"] or 0)) < 1e-12
        bad += 0 if same else 1
        print(f"{name:<6}{got['T']:>5}{got['S']:>5}{got['B']:>5}{got['n']:>6}"
              f"{got['score']:>9.4f}   {'✓' if same else '✗ 与旧值不符 ' + str(want)}")
    if bad:
        raise SystemExit(f"\n❌ {bad} 行对不上,**迁移作废**,不要基于它往下走。")
    print("\n✓ 五行逐值吻合,迁移无损。")


# ------------------------------------------------------------------ R2 批次

def pick_post_vs_pre(eval_set: dict, have: set) -> list[dict]:
    """S3 全 60 + S1 的 44 组合 × seed_slot {0,1} = 88,合计 148。

    WHY S1 只取 2 个 seed:组合内噪声要能平均掉(你在 Q2 选的),
    但三个 seed 全取会把这一批推到 242 条,判读量翻倍而边际信息很小。
    """
    out = []
    for t in eval_set["tasks"]:
        st, tid = t["stratum"], t["task_id"]
        if st == "S3":
            keep = True
        elif st == "S1":
            keep = t["meta"]["seed_slot"] in (0, 1)
        else:
            keep = False
        if not keep:
            continue
        for v in (POST, PRE):
            if (tid, v) not in have:
                raise SystemExit(f"❌ {tid}/{v} 缺图")
        out.append(make_pair(f"m5r1::pvp::{tid}", "post_vs_pre", t,
                             POST, img_rel(EVAL_DIR, tid, POST),
                             PRE, img_rel(EVAL_DIR, tid, PRE)))
    return out


def pick_null_floor(nf_tasks: dict, have: set) -> list[dict]:
    """零假设对:左 = M4 的既有 teacher 图,右 = 步骤 1 新生成的同 refs/同 prompt 图。

    两个候选的语义标签必须**能区分是哪一次跑的**——步骤 1 已查明本栈不可逐位复现,
    所以"哪次跑"是一个真实存在的、需要被统计到的因素(虽然预期它无质量含义)。
    """
    out = []
    for t in nf_tasks["tasks"]:
        if not t["task_id"].startswith("NF_"):
            continue                      # S0 锚点不参与配对
        src = t["meta"]["nf_src_task_id"]
        if (src, TEACHER) not in have:
            raise SystemExit(f"❌ 零假设对的左半边 {src}/{TEACHER} 缺图")
        out.append(make_pair(f"m5r1::nf::{src}", "null_floor", t,
                             "teacher_run_m5", img_rel(NF_DIR, t["task_id"], TEACHER),
                             "teacher_run_m4", img_rel(EVAL_DIR, src, TEACHER),
                             src_task_id=src))
    return out


REPLAY_STRATA = ("S1", "S3")


def pick_replay(m4_pairs: list[dict], m4_marks: dict, n: int = 20) -> list[dict]:
    """从 M4 已标过的条目里确定性抽 n 条重放,**按当年的判定结果按比例分层**。

    WHY 按比例而不是均衡三类:这一批要回答的是"M4 那 227 条的重测一致率是多少",
    是对既有数据集的推断,所以样本要跟既有分布一致。均衡抽样能分别看出
    T/S/B 各自的翻转率,但会让总一致率失去代表性——n=20 只够要一个数。

    WHY 只从 S1/S3 抽(`REPLAY_STRATA`):本批其余 178 条全是 S1(2 张 ref)与
    S3(1 张 ref)。混进一条 S4 就会显示 **3 张参考图**,S2 则是同一主体出现两次——
    两者都能让标注者一眼认出"这条不属于这批",重放就不再是盲的了。
    这是盲法约束,不是取样偏好。
    """
    by_winner: dict[str, list[str]] = defaultdict(list)
    st_of = {p["pair_id"]: p["stratum"] for p in m4_pairs}
    for pid, m in m4_marks.items():
        if st_of[pid] in REPLAY_STRATA:
            by_winner[m["winner"]].append(pid)

    total = sum(len(v) for v in by_winner.values())
    quota, alloc = {}, 0
    groups = sorted(by_winner)                       # 确定性顺序
    for g in groups[:-1]:
        quota[g] = round(n * len(by_winner[g]) / total)
        alloc += quota[g]
    quota[groups[-1]] = n - alloc                    # 余数给最后一组,保证总数恰为 n

    pair_by_id = {p["pair_id"]: p for p in m4_pairs}
    out = []
    for g in groups:
        pool = sorted(by_winner[g], key=lambda p: h(p, "m5r1-replay"))
        if len(pool) < quota[g]:
            raise SystemExit(f"❌ 胜者 {g} 只有 {len(pool)} 条,抽不出 {quota[g]} 条")
        for pid in pool[:quota[g]]:
            src = pair_by_id[pid]
            out.append({**src,
                        "pair_id": f"m5r1::rp::{src['src_task_id']}",
                        "kind": "replay",
                        "replay_of": pid})
    if len(out) != n:
        raise SystemExit(f"❌ 重放抽到 {len(out)} 条,应为 {n}")
    return out


def cmd_r2(args) -> None:
    eval_set = load_json(EVAL_SET)
    results = load_json(M4_RESULTS)
    nf_tasks = load_json(NOISE_FLOOR)
    have = generated_ok(results)

    if not os.path.exists(OUT_M4_PAIRS):
        raise SystemExit("❌ 先跑 `migrate-m4`:重放要从迁移后的 M4 清单里抽")
    m4_pairs = load_json(OUT_M4_PAIRS)["pairs"]
    m4_marks = load_json(OUT_M4_MARKS)["marks"]

    pairs = pick_post_vs_pre(eval_set, have) + pick_null_floor(nf_tasks, have) \
        + pick_replay(m4_pairs, m4_marks, args.n_replay)

    # 打散:与槽位用不同的盐,免得"排在前面"和"在左边"产生相关
    pairs.sort(key=lambda p: h(p["pair_id"], M5_BLIND_SEED, "order"))

    write_manifest(OUT_R2_PAIRS, {
        "batch_id": "m5r1",
        "blind_seed": M5_BLIND_SEED,
        "question": "哪一张更好?(综合参考图忠实度与画面质量)",
        "eval_set_version": "v1-232 + noise_floor_tasks v1",
        "source": "§11.3 步骤 2:post_vs_pre 148 + null_floor 30 + replay 20",
    }, pairs)

    n_st = Counter((p["kind"], p["stratum"]) for p in pairs)
    print("\n构成:")
    for (k, st), c in sorted(n_st.items()):
        print(f"  {k:<20}{st:<5}{c:>5}")
    print(f"\n盲种 {M5_BLIND_SEED}(与 M4 不同——你已在揭盲模式下看过全部 227 条,"
          f"旧槽位不可再用)")


# ------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate-m4", help="M4 227 条迁移到 pair_id 键控 + 复算校验")
    r2 = sub.add_parser("r2", help="生成步骤 2 的 198 条批次")
    r2.add_argument("--n_replay", type=int, default=20)
    args = p.parse_args()
    {"migrate-m4": cmd_migrate_m4, "r2": cmd_r2}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
