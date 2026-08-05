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
      kind         post_vs_pre / null_floor / replay / teacher_vs_student /
                   arm_a_vs_teacher
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
    python distill/build_pairs.py arm-a         # 臂 A 的 192 条批次(§11.7 第 ① 边)
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

ARM_A_TASKS = os.path.join(REPO, "datasets/eval_multiref/arm_a_tasks.json")
ARM_A_RESULTS = os.path.join(REPO, "output/eval_arm_a/results.json")

ARM_B_TASKS = os.path.join(REPO, "datasets/eval_multiref/arm_b_tasks.json")
ARM_B_RESULTS = os.path.join(REPO, "output/eval_arm_b/results.json")

OUT_M4_PAIRS = os.path.join(REPO, "output/eval_multiref/pairs_m4r1.json")
OUT_M4_MARKS = os.path.join(REPO, "output/eval_multiref/blind_annotations_m4r1.json")
OUT_R2_PAIRS = os.path.join(REPO, "output/eval_multiref/pairs_m5r1.json")
OUT_AA_PAIRS = os.path.join(REPO, "output/eval_arm_a/pairs_m5aa.json")
OUT_AA_MARKS = os.path.join(REPO, "output/eval_arm_a/blind_annotations_m5aa.json")
OUT_CTL_PAIRS = os.path.join(REPO, "output/eval_arm_a/pairs_m5aactl.json")
OUT_AB_PAIRS = os.path.join(REPO, "output/eval_arm_b/pairs_m5ab.json")
OUT_E3_PAIRS = os.path.join(REPO, "output/eval_arm_b/pairs_m5e3.json")

EVAL_DIR = "output/eval_multiref"      # M4 产物目录(仓库根相对)
NF_DIR = "output/noise_floor"          # 步骤 1 产物目录
AA_DIR = "output/eval_arm_a"           # 臂 A 读数批次产物目录
AB_DIR = "output/eval_arm_b"           # 臂 B 读数批次产物目录

TEACHER = "official_full"
PRE = "ours_kv_pre"
POST = "ours_kv_post4000"
ARM_A = "arm_a_full"
ARM_B = "arm_b_iso"

# M4 用过的盲种。**只用于把旧标注解码回语义胜者**,新批次一律不用它
# (你已在揭盲模式下看过全部 227 条,这个种子对应的槽位已污染)。
M4_BLIND_SEED = "m4-blind-v1"
M5_BLIND_SEED = "m5-blind-v1"
AA_BLIND_SEED = "m5-arm-a-v1"
CTL_BLIND_SEED = "m5-arm-a-ctl-v1"
AB_BLIND_SEED = "m5-arm-b-v1"
E3_BLIND_SEED = "m5-edge3-v1"
# 展示顺序的打散盐,与盲种分开(盲种还决定左右槽位,动它会连带改 L/R)。
# `-v1` 下 `spread_runfloor` 的最大可达最小间距只有 70 < 222//3 = 74,守卫拒绝;
# 按"**依序取第一个通过的**"这条规则换到 `-v2`(间距 90)。
# WHY 是"第一个通过"而不是"间距最大的那个"(v5 能到 99):顺序不进入任何统计量,
# 但"挑一个好看的"是个没有规则的自由度,写成规则就没有可挑的余地。
E3_ORDER_SALT = "m5e3-order-v2"

# 盲种登记表。每个批次必须有**自己**的盲种:槽位一旦在揭盲模式下被看过就永久污染,
# 而"看过"是不可撤销的。写成登记表而不是靠人记,是因为复制粘贴一个 cmd_* 函数时
# 最容易漏掉的就是改种子——漏了不会报错,只会静默复用旧槽位。
BLIND_SEEDS = {
    "m4r1":    M4_BLIND_SEED,
    "m5r1":    M5_BLIND_SEED,
    "m5aa":    AA_BLIND_SEED,
    "m5aactl": CTL_BLIND_SEED,
    "m5ab":    AB_BLIND_SEED,
    "m5e3":    E3_BLIND_SEED,
}


def fresh_seed(batch_id: str, seed: str) -> str:
    """盲种守卫:既挡"抄了别人的种子",也挡"新批次忘了登记"。"""
    clash = sorted(b for b, s in BLIND_SEEDS.items() if s == seed and b != batch_id)
    if clash:
        raise SystemExit(f"❌ 盲种 {seed} 已被批次 {clash} 用过,槽位已污染,必须换一个")
    if BLIND_SEEDS.get(batch_id) != seed:
        raise SystemExit(f"❌ 批次 {batch_id} 没在 BLIND_SEEDS 里登记 {seed}——"
                         f"先登记再建清单,否则下一个批次查不到它")
    return seed

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


# ------------------------------------------------------------------ 臂 A 批次(§11.7)

def cmd_arm_a(_args) -> None:
    """臂 A 读数批次:`arm_a_full` vs `official_full`,192 对,链条第 ① 边。

    WHY 只有这一个 kind、不掺 replay / null_floor:§11.7 的批次表把判读量写死成
    「192 条,按 §8.1 / §8.2 口径,**不新增判据**」。掺重放要占判读预算,而
    §8.5-3 的跨批次漂移在这里**不构成威胁**——判据是**批内**的非平局胜率,
    两个变体同批同会话生成、同一轮标注,尺子漂移对批内比较不起作用。

    WHY key_0 = `arm_a_full`:清单约定 key_0 = 被检验方、key_1 = 基线
    (`report.py:13`),于是非平局胜率 `win_0/(win_0+win_1)` 直接就是
    "臂 A 相对 teacher 的胜率",§8.2 的 CI 下界 ≥ 0.40 可以原样套用,
    不需要在报告里做任何方向翻转——翻转正是 M4 吃过亏的地方(score 0.92→0.82)。
    """
    tasks = load_json(ARM_A_TASKS)["tasks"]
    results = load_json(ARM_A_RESULTS)
    have = generated_ok(results)

    fresh_seed("m5aa", AA_BLIND_SEED)

    # 记录里的 path 是生成时**实际写下**的路径;img_rel 是重建出来的。
    # 两者对不上说明 out_path 的命名约定漂了,此时清单会指向不存在的图,
    # 而服务器要到上机那一刻才报错——所以在这里当场拦下。
    actual = {(r["task_id"], r["variant"]): r["path"] for r in results["records"]}

    pairs = []
    for t in tasks:
        tid = t["task_id"]
        for v in (ARM_A, TEACHER):
            if (tid, v) not in have:
                raise SystemExit(f"❌ {tid}/{v} 缺图,192 条不完整就不许开评")
            rebuilt = img_rel(AA_DIR, tid, v)
            if actual[(tid, v)] != rebuilt:
                raise SystemExit(f"❌ {tid}/{v} 记录路径 {actual[(tid, v)]} "
                                 f"≠ 重建路径 {rebuilt}")
        pairs.append(make_pair(f"m5aa::aa::{tid}", "arm_a_vs_teacher", t,
                               ARM_A, img_rel(AA_DIR, tid, ARM_A),      # key_0 = 被检验方
                               TEACHER, img_rel(AA_DIR, tid, TEACHER)))

    n_task = len(tasks)
    if len(pairs) != n_task or n_task != 192:
        raise SystemExit(f"❌ 配对 {len(pairs)} 对 / 任务 {n_task} 条,应为 192")

    # 打散:与槽位用不同的盐,免得"排在前面"和"在左边"产生相关(同 cmd_r2)
    pairs.sort(key=lambda p: h(p["pair_id"], AA_BLIND_SEED, "order"))

    write_manifest(OUT_AA_PAIRS, {
        "batch_id": "m5aa",
        "blind_seed": AA_BLIND_SEED,
        "question": "哪一张更好?(综合参考图忠实度与画面质量)",   # 逐字沿用,改问题=改尺子
        "eval_set_version": "arm_a_tasks v1(= eval_set v1-232 的 S1+S3,seed 零偏移)",
        "source": "§11.7 链条第 ① 边:配方施加在已对齐底座上的代价",
    }, pairs)

    by_st = Counter(p["stratum"] for p in pairs)
    print(f"\n分层:{dict(sorted(by_st.items()))}")
    n_nontie = round(len(pairs) * 0.667)
    print(f"  按零假设对实测 33.3% 平局率折合 n_nontie ≈ {n_nontie}"
          f"(§8.2 要求 ≥ 94、且 Wilson CI 下界 ≥ 0.40)")
    print(f"  ⚠️ 平局率 > {1 - 94 / len(pairs):.1%} ⇒ 结论是「判据不适用」而非「不达标」,"
          f"**不许事后追加样本**(§11.7)")
    print(f"\n盲种 {AA_BLIND_SEED}(登记表里与其它批次都不同)")
    print(f"⚠️ 拼图 {AA_DIR}/boards/ 是**带变体名标注**的并排图——"
          f"开评之前不要再看它,看过多少要写进 §8.5 的局限里。")


# ------------------------------------------------------------------ 臂 A 尺子标定批(§11.8)

CTL_RUNFLOOR_N = {"S1": 40, "S3": 20}     # 按臂 A 的 132:60 等比缩到 60 条
CTL_N_REPLAY = 20


def pick_run_floor(aa_tasks: list[dict], m4_have: set, aa_have: set,
                   exclude: set) -> list[dict]:
    """**同 seed、同权重、异 run** 的对照对 —— 平局率的**天花板**。

    WHY 非做不可:臂 A 的平局率 59.4% 本身读不出方向。它夹在两个基线之间——

        换 seed + 同权重(零假设对,已测)          33.3%
        同 seed  + 异权重(臂 A,本次)             59.4%
        同 seed  + 同权重(异 run,**从没测过**)   ?

    §11.7 预登记时拿 33.3% 当参照,但零假设对**换了 seed**(30/30 全换,
    `noise_floor_tasks.json` 可查),而臂 A 是共享噪声的。共享噪声本身就让两张图
    更像 ⇒ 平局率更高,**这个混杂正好朝着"配方无代价"那个结论的方向推**。
    所以 33.3% 不是单变量对照,不能直接支撑结论。

    真正的对照是"同一份权重、同一个 seed、跑两次"。它的图**已经免费存在**:
    M4 的 192 张 `official_full` 与臂 A 批的 192 张 `official_full` 同任务同 seed,
    只差生成会话(M4 ≈07-30,臂 A 08-04)。GPU 成本 0。

    **这个对照是双向的,不是为结论量身定做的**:天花板若 ≈90%,则 59.4% 明显低于它
    ⇒ 臂 A 与 teacher **可分辨**,「配方无代价」不成立;天花板若 ≈60%,则 59.4%
    已经贴到上限 ⇒ 结论成立。两种结果都可能,事先不知道是哪种。
    """
    out = []
    for st, n in CTL_RUNFLOOR_N.items():
        pool = [t for t in aa_tasks
                if t["stratum"] == st and t["task_id"] not in exclude]
        pool.sort(key=lambda t: h(t["task_id"], "m5aactl-runfloor"))
        if len(pool) < n:
            raise SystemExit(f"❌ {st} 只剩 {len(pool)} 条,抽不出 {n} 条")
        for t in pool[:n]:
            src = t["meta"]["arm_a_src_task_id"]
            if (t["task_id"], TEACHER) not in aa_have:
                raise SystemExit(f"❌ 臂 A 批缺 {t['task_id']}/{TEACHER}")
            if (src, TEACHER) not in m4_have:
                raise SystemExit(f"❌ M4 批缺 {src}/{TEACHER}")
            out.append(make_pair(f"m5aactl::rf::{src}", "run_floor", t,
                                 "teacher_run_arm_a", img_rel(AA_DIR, t["task_id"], TEACHER),
                                 "teacher_run_m4", img_rel(EVAL_DIR, src, TEACHER),
                                 src_task_id=src))
    return out


def pick_arm_a_replay(aa_pairs: list[dict], aa_marks: dict, n: int) -> list[dict]:
    """从臂 A 已标的 192 条里按**当次判定结果**等比抽 n 条重放(同 `pick_replay` 纪律)。

    WHY 要混进来、不单跑天花板批:天花板若在**另一场**判读里测,拿它和臂 A 的
    59.4% 相比又变成跨批次比较——正是 §8.5-3 禁止的那件事(而且 §8.5-3 实测的漂移
    **恰好全发生在平局边界上**,8/20)。混进同一场,才能当场确认臂 A 的平局率
    在这把尺子的当前状态下还成不成立。

    副作用是标注者会察觉"这批里有几对几乎一模一样"。这是天花板条件本身的性质,
    躲不掉;混入 20 条真臂 A 对,至少不会让整批退化成"一路平局"。
    """
    by_winner: dict[str, list[str]] = defaultdict(list)
    for pid, m in aa_marks.items():
        by_winner[m["winner"]].append(pid)

    total = sum(len(v) for v in by_winner.values())
    quota, alloc = {}, 0
    groups = sorted(by_winner)
    for g in groups[:-1]:
        quota[g] = round(n * len(by_winner[g]) / total)
        alloc += quota[g]
    quota[groups[-1]] = n - alloc

    pair_by_id = {p["pair_id"]: p for p in aa_pairs}
    out = []
    for g in groups:
        pool = sorted(by_winner[g], key=lambda p: h(p, "m5aactl-replay"))
        if len(pool) < quota[g]:
            raise SystemExit(f"❌ 胜者 {g} 只有 {len(pool)} 条,抽不出 {quota[g]} 条")
        for pid in pool[:quota[g]]:
            src = pair_by_id[pid]
            out.append({**src,
                        "pair_id": f"m5aactl::rp::{src['src_task_id']}",
                        "kind": "replay",          # 用这个名字,report.py 才会算自洽率
                        "replay_of": pid})
    if len(out) != n:
        raise SystemExit(f"❌ 重放抽到 {len(out)} 条,应为 {n}")
    return out


def cmd_arm_a_ctl(_args) -> None:
    """臂 A 的**尺子标定批**:60 条 run_floor(天花板)+ 20 条臂 A 重放。

    **臂 A 的 31/47/114 就此冻结,本批一条都不并进去。** 这不是"事后追加样本"
    (§11.7 禁的是那个)——判据早已判定「不适用」,且本批测的是**尺子**不是**样本**:
    run_floor 的两侧是同一份权重,它对"配方有没有代价"这个问题结构上无话可说。
    """
    fresh_seed("m5aactl", CTL_BLIND_SEED)

    aa_tasks = load_json(ARM_A_TASKS)["tasks"]
    aa_have = generated_ok(load_json(ARM_A_RESULTS))
    m4_have = generated_ok(load_json(M4_RESULTS))
    aa_pairs = load_json(OUT_AA_PAIRS)["pairs"]
    aa_marks = load_json(OUT_AA_MARKS)["marks"]

    if len(aa_marks) != len(aa_pairs):
        raise SystemExit(f"❌ 臂 A 只标了 {len(aa_marks)}/{len(aa_pairs)} 条,"
                         f"没标完就抽重放,比例分层是错的")

    replay = pick_arm_a_replay(aa_pairs, aa_marks, CTL_N_REPLAY)
    # 重放用掉的任务不再进 run_floor:同一个 prompt/refs 在一场里出现两次,
    # 标注者一眼就认出来,重放就不盲了。
    # 臂 A 清单的 src_task_id 就是 `AA_*` 任务号(make_pair 默认取 task_id),直接用
    used = {p["src_task_id"] for p in replay}
    pairs = pick_run_floor(aa_tasks, m4_have, aa_have, used) + replay

    pairs.sort(key=lambda p: h(p["pair_id"], CTL_BLIND_SEED, "order"))

    write_manifest(OUT_CTL_PAIRS, {
        "batch_id": "m5aactl",
        "blind_seed": CTL_BLIND_SEED,
        "question": "哪一张更好?(综合参考图忠实度与画面质量)",   # 逐字沿用
        "eval_set_version": "arm_a_tasks v1 + eval_set v1-232 的 official_full",
        "source": "§11.8 尺子标定:run_floor 60(同 seed 同权重异 run 的平局率天花板)"
                  " + 臂 A 重放 20;臂 A 的 31/47/114 冻结,不并入",
    }, pairs)

    n_st = Counter((p["kind"], p["stratum"]) for p in pairs)
    print("\n构成:")
    for (k, st), c in sorted(n_st.items()):
        print(f"  {k:<12}{st:<5}{c:>5}")
    print(f"\n重放的原判定分布:"
          f"{dict(sorted(Counter(aa_marks[p['replay_of']]['winner'] for p in replay).items()))}")
    print(f"盲种 {CTL_BLIND_SEED}(登记表里与其它批次都不同)")
    print("GPU 成本 0——两侧的图都已存在(臂 A 批 + M4 批)。")


# ------------------------------------------------------------------ 臂 B 终批(§11.11(e))

def twin_gaps(pairs: list[dict]) -> dict:
    """量一件**躲不掉**的事:那 30 条 `run_floor` 锚点的 prompt/refs 在本批里出现两次
    ——一次作 `arm_b` 对、一次作 `run_floor` 对。

    WHY 躲不掉:天花板必须跨"与被检验那一对相同的会话间隔"(§11.11 本地加固),
    这就要求两侧都是臂 A 批出过图的任务,而臂 A 批只覆盖 S1+S3 = 本批的全部 192 条。
    从 S2/S4 里另抽锚点的话,对侧只能回到 M4 的旧图,跨的又变成 07-30 那段间隔。

    纯 md5 打散下实测 **最小间距 3、30 条里 5 条 <10** —— 同一个 prompt 在十对之内
    重复出现,标注者一定会察觉。处置见 `spread_runfloor`。
    """
    pos_by_task: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        pos_by_task[p["src_task_id"]].append(i)
    gaps = sorted(max(v) - min(v) for v in pos_by_task.values() if len(v) > 1)
    if not gaps:
        return {"n_twin": 0}
    return {
        "n_twin": len(gaps),
        "min": gaps[0],
        "p25": gaps[len(gaps) // 4],
        "median": gaps[len(gaps) // 2],
        "max": gaps[-1],
        "n_under_10": sum(1 for g in gaps if g < 10),
        "batch_len": len(pairs),
    }


def spread_runfloor(pairs: list[dict]) -> int:
    """把 30 条 `run_floor` **在它们自己的槽位之间**重排,最大化"锚点与其孪生
    `arm_b` 对"的最小间距。返回达成的最小间距。

    WHY 是"槽位内重排"而不是"重新排序":md5 打散给出的 30 个 `run_floor` 位置
    **一个都不动**,动的只是"哪一条锚点坐哪个槽"。于是锚点在批里的密度分布
    与纯打散完全相同(实测三分位 10/13/7),**不会**出现"批尾全是几乎一样的对"
    ——那才是真正危险的那种结构:标注者一进尾段就发现规律,判读策略当场改变,
    连带把该段的 `arm_b` 读数一起污染。而"隔得太近的重复 prompt"只是二阶风险。
    两个风险这样就同时压住了,代价只是引入一个确定性的指派步骤。

    做法:对候选间距 G 二分,可行性用二分图匹配判(锚点 i 可放槽位 j ⟺
    |slot_j − twin_i| ≥ G),取可行的最大 G。30×30 的规模下增广路径就够。
    邻接表按槽位下标顺序生成、匹配按锚点下标顺序尝试 ⇒ **完全确定性**,
    同一份任务单永远重排出同一个清单。

    实测本批可达 G = 89(批长 222 的 40%),远超 `idcount/build_items.py` 用的
    `len//3` 那条既有纪律。
    """
    slots = [i for i, p in enumerate(pairs) if p["kind"] == "run_floor"]
    twin = {p["src_task_id"]: i for i, p in enumerate(pairs)
            if p["kind"] != "run_floor"}
    rfs = [pairs[i] for i in slots]
    tw = [twin[p["src_task_id"]] for p in rfs]
    if not slots:
        return 0

    def match(g: int):
        adj = [[j for j, s in enumerate(slots) if abs(s - tw[i]) >= g]
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


def cmd_arm_b(args) -> None:
    """臂 B 终批:`arm_b_iso` vs `arm_a_full` 192 对 + `run_floor` 30 对 = **222 对**。

    组成由 §11.11(e) 预登记,**不带 replay**(自洽率已测两次,第三次不承重)。
    这是新主命题(边 ②′)的直接单变量读数,**也是最后一个批次**。

    WHY key_0 = `arm_b_iso`:清单约定 key_0 = 被检验方、key_1 = 基线
    (`report.py:13`),于是非平局胜率 `win_0/(win_0+win_1)` 直接就是
    "臂 B 相对臂 A 的胜率",§8.2 的 CI 下界 ≥ 0.40 原样套用,报告里不做方向翻转
    ——翻转正是 M4 吃过亏的地方(score 0.92→0.82)。

    WHY `run_floor` 的两侧都是 `official_full`:天花板的定义就是"同权重、同 seed、
    异 run",与挂哪份 LoRA 无关。选官方权重是因为它由 pipeline 自带备份提供
    (`eval_multiref.py:198-207`),不依赖任何 checkpoint 文件在不在盘上。

    ⚠️ **前置条件**:§9 身份留存门必须已经判过且点估计 ≥ 60%(§11.11(d))。
    门没过就不该建这个清单——那 16 分钟判读预算按预登记要省下来。本脚本
    **不代替人去判这个门**,只在这里写明。
    """
    fresh_seed("m5ab", AB_BLIND_SEED)

    tasks = load_json(ARM_B_TASKS)["tasks"]
    aa_have = generated_ok(load_json(ARM_A_RESULTS))

    if args.dry_run:
        ab_have = {(t["task_id"], v) for t in tasks for v in t["variants"]}
        actual = None
        print("⚠️ --dry_run:假装臂 B 的图全都生成成功,只看条数与排序统计")
    else:
        ab_results = load_json(ARM_B_RESULTS)
        ab_have = generated_ok(ab_results)
        # 记录里的 path 是生成时**实际写下**的路径;img_rel 是重建出来的。对不上说明
        # out_path 的命名约定漂了,此时清单会指向不存在的图,而服务器要到上机那一刻
        # 才报错——所以在这里当场拦下(同 cmd_arm_a)。
        actual = {(r["task_id"], r["variant"]): r["path"] for r in ab_results["records"]}

    pairs = []
    n_rf = 0
    for t in tasks:
        tid = t["task_id"]
        src = t["meta"]["arm_b_src_task_id"]
        aa_id = "AA_" + src

        if (tid, ARM_B) not in ab_have:
            raise SystemExit(f"❌ {tid}/{ARM_B} 缺图,192 条不完整就不许开评")
        if (aa_id, ARM_A) not in aa_have:
            raise SystemExit(f"❌ 臂 A 批缺 {aa_id}/{ARM_A}(它是本批的基线)")
        if actual is not None:
            rebuilt = img_rel(AB_DIR, tid, ARM_B)
            if actual[(tid, ARM_B)] != rebuilt:
                raise SystemExit(f"❌ {tid}/{ARM_B} 记录路径 {actual[(tid, ARM_B)]} "
                                 f"≠ 重建路径 {rebuilt}")
        pairs.append(make_pair(f"m5ab::ab::{tid}", "arm_b_vs_arm_a", t,
                               ARM_B, img_rel(AB_DIR, tid, ARM_B),        # key_0 = 被检验方
                               ARM_A, img_rel(AA_DIR, aa_id, ARM_A),      # key_1 = 基线
                               src_task_id=tid))

        if t["meta"].get("arm_b_runfloor"):
            if (tid, TEACHER) not in ab_have:
                raise SystemExit(f"❌ {tid}/{TEACHER} 缺图,run_floor 锚点配不出对")
            if (aa_id, TEACHER) not in aa_have:
                raise SystemExit(f"❌ 臂 A 批缺 {aa_id}/{TEACHER}(天花板的另一侧)")
            if actual is not None:
                rebuilt = img_rel(AB_DIR, tid, TEACHER)
                if actual[(tid, TEACHER)] != rebuilt:
                    raise SystemExit(f"❌ {tid}/{TEACHER} 记录路径 "
                                     f"{actual[(tid, TEACHER)]} ≠ 重建路径 {rebuilt}")
            pairs.append(make_pair(f"m5ab::rf::{src}", "run_floor", t,
                                   "teacher_run_arm_b", img_rel(AB_DIR, tid, TEACHER),
                                   "teacher_run_arm_a", img_rel(AA_DIR, aa_id, TEACHER),
                                   src_task_id=tid))
            n_rf += 1

    if len(tasks) != 192 or n_rf != 30 or len(pairs) != 222:
        raise SystemExit(f"❌ 任务 {len(tasks)} 条 / 锚点 {n_rf} 条 / 配对 {len(pairs)} 对,"
                         f"应为 192 / 30 / 222(§11.11(e) 预登记的批次组成)")

    # 打散:与槽位用不同的盐,免得"排在前面"和"在左边"产生相关(同 cmd_r2 / cmd_arm_a)
    pairs.sort(key=lambda p: h(p["pair_id"], AB_BLIND_SEED, "order"))
    before = twin_gaps(pairs)
    g_min = spread_runfloor(pairs)
    gaps = twin_gaps(pairs)
    gaps["achieved_min_gap"] = g_min
    gaps["min_before_spread"] = before["min"]
    # 拿到的最小间距若还没到 idcount 那条既有纪律(len//3),说明这批任务的
    # 打散位置天生挤,得换盐重排——而不是降低要求。
    if g_min < len(pairs) // 3:
        raise SystemExit(f"❌ 槽位内重排后最小间距仅 {g_min},低于既有纪律 "
                         f"{len(pairs) // 3}(= 批长/3)。换 AB_BLIND_SEED 重来,"
                         f"不要降低要求。")

    if args.dry_run:
        print(f"\n条数:{len(pairs)} 对 {dict(sorted(Counter(p['kind'] for p in pairs).items()))}")
        print(f"锚点双出现间距:{gaps}")
        tert = Counter(("头", "中", "尾")[min(2, i * 3 // len(pairs))]
                       for i, p in enumerate(pairs) if p["kind"] == "run_floor")
        print(f"run_floor 三分位分布:{dict(tert)}(应大致均匀,聚在尾段就是危险结构)")
        return

    write_manifest(OUT_AB_PAIRS, {
        "batch_id": "m5ab",
        "blind_seed": AB_BLIND_SEED,
        "question": "哪一张更好?(综合参考图忠实度与画面质量)",   # 逐字沿用,改问题=改尺子
        "eval_set_version": "arm_b_tasks v1(= eval_set v1-232 的 S1+S3,seed 零偏移)"
                            " + 臂 A 批的 arm_a_full / official_full 作对侧",
        "source": "§11.11(e) 终批:边 ②′(隔离 vs 全注意力,同数据同配方同 init)"
                  " 192 对 + 同会话 run_floor 30 对,不带 replay",
        "twin_gaps": gaps,
    }, pairs)

    by_st = Counter((p["kind"], p["stratum"]) for p in pairs)
    print("\n构成:")
    for (k, st), c in sorted(by_st.items()):
        print(f"  {k:<16}{st:<5}{c:>5}")
    n_main = sum(1 for p in pairs if p["kind"] == "arm_b_vs_arm_a")
    print(f"\n按臂 A 实测 59.4% 平局率折合 n_nontie ≈ {round(n_main * (1 - 0.594))}"
          f"(§8.2 要求 ≥ 94、且 Wilson CI 下界 ≥ 0.40)")
    print(f"  ⚠️ 平局率 > {1 - 94 / n_main:.1%} ⇒ 结论是「判据不适用」而非「不达标」,"
          f"**不许事后追加样本**")
    print(f"\n锚点双出现间距(那 30 条 prompt 在批里出现两次,躲不掉,见 twin_gaps 的 WHY):")
    print(f"  纯 md5 打散最小 {gaps['min_before_spread']} → 槽位内重排后最小 "
          f"{gaps['min']} / 中位 {gaps['median']} / 最大 {gaps['max']}(批长 {gaps['batch_len']})")
    tert = Counter(("头", "中", "尾")[min(2, i * 3 // len(pairs))]
                   for i, p in enumerate(pairs) if p["kind"] == "run_floor")
    print(f"  run_floor 三分位分布 {dict(tert)}——槽位未动,不聚堆")
    print(f"  ⇒ 「同一 prompt 在批内出现两次」这件事本身仍要如实写进 §8.5 局限。")
    print(f"\n盲种 {AB_BLIND_SEED}(登记表里与其它批次都不同)")
    print(f"⚠️ 拼图 {AB_DIR}/boards/ 是**带变体名标注**的并排图——"
          f"开评之前不要再看它,看过多少要写进 §8.5 的局限里。")
    print("⚠️ 前置:§9 身份留存门必须已判且点估计 ≥ 60%(§11.11(d)),否则本批不该建。")


# ------------------------------------------------------------------ 边 ③ 底座(2026-08-05 补)

def cmd_edge3(args) -> None:
    """链条第 ③ 边:`ours_kv_post4000` vs `arm_b_iso`,192 对 + `run_floor` 30 对。

    **这条边此前从没被直接测过。** §11.5 的链条 08-04 就把它画出来了
    (`臂B ──③底座──> post4000`),但 §11.11(c) 的预算表把判读停在臂 B,
    于是 ③ 一直只有"两端各自的读数",没有**边本身**的读数——而 §8.5-3 禁止跨批相减,
    所以它不是"算得出来只是没算",是**真的没有数**。

    两端只差 **init** 一个变量:
      `arm_b_iso`        init = 官方 LoRA          隔离 4000 步
      `ours_kv_post4000` init = 我们的 `ckpt-20000` 隔离 4000 步
    数据(`train_mixed.json`)、步数、全部超参、推理模式(隔离 + KV)、seed 全同。
    ⇒ 读的就是「**我们自己复刻的 stage-1 底座,比官方权重差多少**」。

    WHY key_0 = `post4000`:清单约定 key_0 = 被检验方(`report.py:13`),
    而链条上**下游那个节点**是被检验方——M4(post4000 vs teacher)、
    臂 A(arm_a vs teacher)、臂 B(arm_b vs arm_a)三批都是这么排的。
    于是非平局胜率 < 0.5、旧口径 < 1 **一律读作"我们的底座更差"**,方向不翻转。

    **GPU 成本 0**:两侧的图全都已存在——`post4000` 在 M4 批(07-30),
    `arm_b_iso` 在臂 B 批(08-05)。`run_floor` 同样免费,而且它跨的正是
    **07-30 ↔ 08-05** 这段、与主对**完全同一个会话间隔**(§11.11 本地加固的要求)。

    ⚠️ **这是预登记之外新增的第四个批次。** §11.11(c) 写死了"没有第四个批次",
    本批是在**看到臂 B 结果之后**才建的 ⇒ 报告里必须声明:
    ① 这条边**早在 08-04 的链条里就定义好了**,不是看到结果才想出来的比较,
       推迟纯粹是判读预算所致;② 但"何时决定去测"确实发生在揭盲之后,
       这个**分叉自由度要如实写进 §8.5 局限**,不许说成"预登记的"。
    """
    fresh_seed("m5e3", E3_BLIND_SEED)

    tasks = load_json(ARM_B_TASKS)["tasks"]
    ab_have = generated_ok(load_json(ARM_B_RESULTS))
    m4_have = generated_ok(load_json(M4_RESULTS))

    pairs, n_rf = [], 0
    for t in tasks:
        tid = t["task_id"]
        src = t["meta"]["arm_b_src_task_id"]
        if (src, POST) not in m4_have:
            raise SystemExit(f"❌ M4 批缺 {src}/{POST}(本批的被检验方)")
        if (tid, ARM_B) not in ab_have:
            raise SystemExit(f"❌ 臂 B 批缺 {tid}/{ARM_B}(本批的基线)")
        pairs.append(make_pair(f"m5e3::e3::{tid}", "student_vs_arm_b", t,
                               POST, img_rel(EVAL_DIR, src, POST),        # key_0 = 被检验方
                               ARM_B, img_rel(AB_DIR, tid, ARM_B),        # key_1 = 基线
                               src_task_id=tid))
        if t["meta"].get("arm_b_runfloor"):
            if (src, TEACHER) not in m4_have or (tid, TEACHER) not in ab_have:
                raise SystemExit(f"❌ {src} 的 run_floor 两侧不齐")
            # key_0/key_1 的会话方向与主对一致:key_0 = 07-30 侧,key_1 = 08-05 侧
            pairs.append(make_pair(f"m5e3::rf::{src}", "run_floor", t,
                                   "teacher_run_m4", img_rel(EVAL_DIR, src, TEACHER),
                                   "teacher_run_arm_b", img_rel(AB_DIR, tid, TEACHER),
                                   src_task_id=tid))
            n_rf += 1

    if len(tasks) != 192 or n_rf != 30 or len(pairs) != 222:
        raise SystemExit(f"❌ 任务 {len(tasks)} / 锚点 {n_rf} / 配对 {len(pairs)},应为 192/30/222")

    pairs.sort(key=lambda p: h(p["pair_id"], E3_ORDER_SALT, "order"))
    before = twin_gaps(pairs)
    g_min = spread_runfloor(pairs)          # 同臂 B 批:锚点 prompt 在批内出现两次,躲不掉
    gaps = twin_gaps(pairs)
    gaps["achieved_min_gap"] = g_min
    gaps["min_before_spread"] = before["min"]
    if g_min < len(pairs) // 3:
        raise SystemExit(f"❌ 槽位内重排后最小间距仅 {g_min} < {len(pairs) // 3},换盐重来")

    if args.dry_run:
        print(f"\n条数:{len(pairs)} 对 "
              f"{dict(sorted(Counter(p['kind'] for p in pairs).items()))}")
        print(f"锚点双出现间距:{gaps}")
        print("GPU 成本 0——两侧的图都已存在(M4 批 07-30 + 臂 B 批 08-05)。")
        return

    write_manifest(OUT_E3_PAIRS, {
        "batch_id": "m5e3",
        "blind_seed": E3_BLIND_SEED,
        "question": "哪一张更好?(综合参考图忠实度与画面质量)",   # 逐字沿用,改问题=改尺子
        "eval_set_version": "arm_b_tasks v1 的 192 条 ×(M4 批 post4000 + 臂 B 批 arm_b_iso)",
        "source": "链条第 ③ 边(底座):同数据同配方同隔离,只差 init。"
                  "**预登记之外新增的第四批**,分叉自由度见 cmd_edge3 docstring",
        "twin_gaps": gaps,
    }, pairs)

    n_main = sum(1 for p in pairs if p["kind"] == "student_vs_arm_b")
    print(f"\n构成:")
    for (k, st), c in sorted(Counter((p["kind"], p["stratum"]) for p in pairs).items()):
        print(f"  {k:<18}{st:<5}{c:>5}")
    print(f"\n方向约定:key_0 = {POST}(被检验方)/ key_1 = {ARM_B}(基线)")
    print(f"  ⇒ 非平局胜率 < 0.5、旧口径 < 1 **一律读作「我们复刻的底座更差」**,不翻转。")
    print(f"  §8.2 要求 n_nontie ≥ 94;平局率 > {1 - 94 / n_main:.1%} 即跌破 ⇒ 「判据不适用」。")
    print(f"\n锚点双出现间距:最小 {gaps['min_before_spread']} → 重排后 {gaps['min']}"
          f" / 中位 {gaps['median']}(批长 {gaps['batch_len']})")
    print(f"盲种 {E3_BLIND_SEED}(登记表里与其它批次都不同)")
    print("GPU 成本 0。判读约 16 min。")
    print("⚠️ 这是**预登记之外**新增的第四个批次(§11.11(c) 原写「没有第四个批次」)。"
          "报告里要声明:边 ③ 早在 08-04 的链条里就定义好了,但**决定去测它发生在揭盲之后**"
          "——这个分叉自由度如实写进 §8.5,不许说成「预登记的」。")


# ------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate-m4", help="M4 227 条迁移到 pair_id 键控 + 复算校验")
    r2 = sub.add_parser("r2", help="生成步骤 2 的 198 条批次")
    r2.add_argument("--n_replay", type=int, default=20)
    sub.add_parser("arm-a", help="生成臂 A 的 192 条批次(§11.7 链条第 ① 边)")
    sub.add_parser("arm-a-ctl", help="臂 A 尺子标定批:run_floor 60 + 重放 20(§11.8)")
    ab = sub.add_parser("arm-b", help="臂 B 终批 222 对(§11.11(e));先过 §9 身份留存门")
    ab.add_argument("--dry_run", action="store_true",
                    help="臂 B 的图还没生成时,只核条数与打散统计,不写清单")
    e3 = sub.add_parser("edge3", help="链条第 ③ 边:post4000 vs 臂B(底座),222 对,GPU 成本 0")
    e3.add_argument("--dry_run", action="store_true", help="只核条数与打散统计,不写清单")
    args = p.parse_args()
    {"migrate-m4": cmd_migrate_m4, "r2": cmd_r2,
     "arm-a": cmd_arm_a, "arm-a-ctl": cmd_arm_a_ctl, "arm-b": cmd_arm_b,
     "edge3": cmd_edge3}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
