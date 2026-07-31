"""M4 Stage A:构建全量评测集 `datasets/eval_multiref/eval_set.json`(纯 CPU,秒级)。

上游规格:`distill/M4_EVAL_SPEC.md` §3。五层构成(层名/条数/seed 公式/视角规则全部
写死,改动 = 实验作废):

  S0 锚点(5)      逐字复刻 smoke_eval.py 的 DEFAULT_CASES,seed=3407。
                  作用不是评测,是 Stage B 的回归测试:同 prompt/ref/seed/配置下,
                  新脚本的产物必须与 output/smoke_eval/case0X__*.png 逐像素一致。
  S1 主验收(132)  held-out 44 个合法 2-组合 × 1 场景(轮转) × 3 seed。
  S2 复制探针(15) bear_plushie + grey_sloth_plushie(唯一同 class 对,训练数据规则
                  排除掉的那对,是"槽位绑定"的最强压力测试)× 5 场景 × 3 seed,
                  变体含 ours_kv_post2000(本层独有,回答"复制是不是过训产物")。
  S3 单 ref 回归(60) 10 个 held-out 各自 1-ref × 3 场景 × 2 seed。
                  验收标准是"不劣于 PRE"——混训 40% 多 ref 后单 ref 有没有退化,
                  至今没人验证过。
  S4 3-ref 诊断(20) 贪心选 10 个 3-组合 × 1 场景 × 2 seed。不进验收(teacher
                  自己在 3-ref 上系统性不行),只回答"有没有比 PRE 更差"。

用法:
  python distill/build_eval_json.py --dry_run   # 只打印统计表,不写任何文件
  python distill/build_eval_json.py             # 断言全过后写 eval_set.json + 打印统计表
"""
import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# 本文件与 gen_data.py 同目录,直接 import(规格 §3.1:import,不要重写)
from gen_data import (  # noqa: E402
    HELD_OUT, TRAIN, load_classes, load_scene_templates, load_subject_images,
    make_prompt, assert_split_clean,
)

# ------------------------------------------------------------------ 固定常量(规格 §2)
DATA_DIR = "datasets/dreambooth/dataset"
OUT_JSON = "datasets/eval_multiref/eval_set.json"

# 变体标签(规格 §2.3)。权重路径与五元组语义在 eval_multiref.py 里,这里只写标签。
V3 = ["official_full", "ours_kv_pre", "ours_kv_post4000"]
V4 = ["official_full", "ours_kv_pre", "ours_kv_post4000", "ours_kv_post2000"]

# S0 锚点:逐字复制 smoke_eval.py:43-59 的 DEFAULT_CASES(prompt 与 ref 文件名一字不改),
# 仅把路径前缀从 `./dreambooth/...`(相对 datasets/)改为 `../dreambooth/...`
# (相对 eval_set.json 所在目录 datasets/eval_multiref/)。seed=3407 与冒烟一致。
SMOKE_CASES = [
    {"prompt": "a backpack and a stuffed animal in the jungle",
     "image_paths": ["../dreambooth/dataset/backpack_dog/02.jpg",
                     "../dreambooth/dataset/bear_plushie/03.jpg"]},
    {"prompt": "a bowl and a can on the beach",
     "image_paths": ["../dreambooth/dataset/berry_bowl/02.jpg",
                     "../dreambooth/dataset/can/01.jpg"]},
    {"prompt": "a candle and a clock in the snow",
     "image_paths": ["../dreambooth/dataset/candle/02.jpg",
                     "../dreambooth/dataset/clock/03.jpg"]},
    {"prompt": "a sneaker and a toy in the jungle",
     "image_paths": ["../dreambooth/dataset/colorful_sneaker/01.jpg",
                     "../dreambooth/dataset/duck_toy/01.jpg"]},
    {"prompt": "a boot and a stuffed animal on the beach",
     "image_paths": ["../dreambooth/dataset/fancy_boot/02.jpg",
                     "../dreambooth/dataset/grey_sloth_plushie/04.jpg"]},
]
SMOKE_SEED = 3407

# seed 区间(规格 §3.2 写死):S1=3_500_000+k*10+s,S2=3_600_000+t*10+s,
# S3=3_700_000+i*100+c*10+s,S4=3_800_000+k*10+s。S0 用冒烟的 3407(锚点复现,故意为之)。
SEED_RANGE_S1_S4 = (3_500_000, 3_800_091)
M1_SEED_RANGE = (3_407_000, 3_415_999)  # M1 gen_data 用过,必须不重叠

# 规格 §3.2 表格(验收对照)
EXPECTED = {
    # 层: (任务数, 图数)
    "S0": (5, 15), "S1": (132, 396), "S2": (60, 240), "S3": (60, 180), "S4": (20, 60),
}
EXPECTED_TOTAL_TASKS = 277
EXPECTED_TOTAL_IMAGES = 891


# ------------------------------------------------------------------ 五层枚举

def _task(task_id, stratum, prompt, image_paths, seed, variants,
          subjects, classes_, template_id, template, views, seed_slot):
    return {
        "task_id": task_id,
        "stratum": stratum,
        "prompt": prompt,
        "image_paths": image_paths,
        "seed": seed,
        "variants": variants,
        "meta": {
            "subjects": subjects,
            "classes": classes_,
            "n_refs": len(subjects),
            "template_id": template_id,
            "template": template,
            "views": views,
            "seed_slot": seed_slot,
        },
    }


def build_s0(classes, object_tpl):
    """S0 锚点:复刻冒烟 5 条。template 按 golden 后缀从 prompt 反解(校验一致性)。"""
    tasks = []
    for k, case in enumerate(SMOKE_CASES):
        subjects = [p.split("/")[-2] for p in case["image_paths"]]
        cls = [classes[s] for s in subjects]
        head = make_prompt(cls, "")  # "a X and a Y " 前缀(尾部带空格)
        if not case["prompt"].startswith(head):
            raise SystemExit(f"❌ S0 case{k} prompt 与 class 表推导的前缀不符:"
                             f"\n  prompt={case['prompt']!r}\n  期望前缀={head!r}")
        template = case["prompt"][len(head):]
        if template not in object_tpl:
            raise SystemExit(f"❌ S0 case{k} 的场景后缀 {template!r} 不在 object 20 条模板里")
        tasks.append(_task(
            f"S0_{k:03d}_s0", "S0", case["prompt"], case["image_paths"], SMOKE_SEED, V3,
            subjects, cls, object_tpl.index(template), template,
            [p.split("/")[-1] for p in case["image_paths"]], 0,
        ))
    return tasks


def build_s1(classes, object_tpl, views):
    """S1 主验收:44 个合法 2-组合(字典序)× 1 场景(轮转)× 3 seed。"""
    combos = [c for c in combinations(sorted(HELD_OUT), 2)
              if len({classes[s] for s in c}) == 2]
    if len(combos) != 44:
        raise SystemExit(f"❌ S1 合法 2-组合应为 44(C(10,2)=45 减去 stuffed animal 对),"
                         f"实际 {len(combos)}")
    tasks = []
    for k, (a, b) in enumerate(combos):
        template = object_tpl[k % 20]          # 场景轮转,不与组合绑定
        for s in (0, 1, 2):
            seed = 3_500_000 + k * 10 + s
            va = views[a][(s + 0) % len(views[a])]
            vb = views[b][(s + 1) % len(views[b])]
            tasks.append(_task(
                f"S1_{k:03d}_s{s}", "S1",
                make_prompt([classes[a], classes[b]], template),
                [f"../dreambooth/dataset/{a}/{va}", f"../dreambooth/dataset/{b}/{vb}"],
                seed, V3, [a, b], [classes[a], classes[b]], k % 20, template,
                [va, vb], s,
            ))
    return tasks, combos


def build_s2(classes, object_tpl, views):
    """S2 复制探针:唯一同 class 对 × object 模板 0..19 × 3 seed,变体含 post2000。

    v3(2026-07-30):模板从 5 条扩到 20 条(object 全量场景),S2 15→60。
    既有 15 条的 task_id/seed/视角是 range(5) 子集,被 range(20) 完全覆盖,逐字不变。
    """
    a, b = "bear_plushie", "grey_sloth_plushie"   # 两者 class 都是 "stuffed animal"
    if classes[a] != classes[b]:
        raise SystemExit(f"❌ S2 前提被破坏:{a}({classes[a]}) 与 {b}({classes[b]}) 不再同类")
    tasks = []
    for t in range(20):
        template = object_tpl[t]
        for s in (0, 1, 2):
            seed = 3_600_000 + t * 10 + s
            va = views[a][(s + 0) % len(views[a])]
            vb = views[b][(s + 1) % len(views[b])]
            tasks.append(_task(
                f"S2_{t:03d}_s{s}", "S2",
                make_prompt([classes[a], classes[b]], template),
                [f"../dreambooth/dataset/{a}/{va}", f"../dreambooth/dataset/{b}/{vb}"],
                seed, V4, [a, b], [classes[a], classes[b]], t, template,
                [va, vb], s,
            ))
    return tasks


def build_s3(classes, object_tpl, views):
    """S3 单 ref 回归:10 个 held-out × 3 场景 × 2 seed。"""
    tasks = []
    for i, subj in enumerate(sorted(HELD_OUT)):
        for c in range(3):
            tpl_id = (i * 3 + c) % 20
            template = object_tpl[tpl_id]
            for s in (0, 1):
                seed = 3_700_000 + i * 100 + c * 10 + s
                v = views[subj][(c * 2 + s) % len(views[subj])]
                tasks.append(_task(
                    f"S3_{i * 3 + c:03d}_s{s}", "S3",
                    make_prompt([classes[subj]], template),
                    [f"../dreambooth/dataset/{subj}/{v}"],
                    seed, V3, [subj], [classes[subj]], tpl_id, template,
                    [v], s,
                ))
    return tasks


def build_s4(classes, object_tpl, views):
    """S4 3-ref 诊断:112 个无同类 3-组合里贪心选 10 个(最大化最小覆盖,平局取字典序)。"""
    c3 = [c for c in combinations(sorted(HELD_OUT), 3)
          if len({classes[s] for s in c}) == 3]
    if len(c3) != 112:
        raise SystemExit(f"❌ S4 合法 3-组合应为 112(C(10,3)=120 减去含 stuffed animal 对的 8 个),"
                         f"实际 {len(c3)}")

    # 贪心:每步在剩余组合里选"含当前最少覆盖 subject 最多"的,平局取字典序小者。
    # combinations(sorted(...)) 的输出本身按字典序,平判时取先出现者即字典序最小。
    coverage = {s: 0 for s in sorted(HELD_OUT)}
    remaining = list(c3)
    picked = []
    for _ in range(10):
        min_cov = min(coverage.values())
        best = max(remaining, key=lambda c: sum(1 for s in c if coverage[s] == min_cov))
        picked.append(best)
        remaining.remove(best)
        for s in best:
            coverage[s] += 1
    # 规格 §3.2:断言每个 held-out subject 覆盖 ≥2;参考实现应恰好全部 =3。
    low = {s: n for s, n in coverage.items() if n < 2}
    if low:
        raise SystemExit(f"❌ S4 覆盖断言失败,这些 subject 覆盖 <2:{low}")

    tasks = []
    for k, combo in enumerate(picked):
        template = object_tpl[k % 20]
        for s in (0, 1):
            seed = 3_800_000 + k * 10 + s
            vs = [views[sub][(s + j) % len(views[sub])] for j, sub in enumerate(combo)]
            tasks.append(_task(
                f"S4_{k:03d}_s{s}", "S4",
                make_prompt([classes[sub] for sub in combo], template),
                [f"../dreambooth/dataset/{sub}/{v}" for sub, v in zip(combo, vs)],
                seed, V3, list(combo), [classes[sub] for sub in combo], k % 20, template,
                vs, s,
            ))
    return tasks, picked, coverage


# ------------------------------------------------------------------ 启动断言(规格 §3.4)

def assert_eval_set(tasks, json_dir):
    """缺一不可;任何一条不过都 sys.exit,宁可启动炸也不带污染评测集往下跑。"""
    # 1. TRAIN subject 泄漏检查(名单 + 实际产物两遍)
    train_set = set(TRAIN)
    leaked = sorted({s for t in tasks for s in t["meta"]["subjects"] if s in train_set})
    if leaked:
        raise SystemExit(f"❌ 评测集里出现 TRAIN subject:{leaked}(泄漏,拒绝继续)")
    not_held = sorted({s for t in tasks for s in t["meta"]["subjects"]
                       if s not in set(HELD_OUT)})
    if not_held:
        raise SystemExit(f"❌ 评测集里出现非 held-out subject:{not_held}")

    # 2. task_id 全局唯一
    ids = [t["task_id"] for t in tasks]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        raise SystemExit(f"❌ task_id 重复:{dup[:5]}")

    # 3. 每个 image_paths 指向的文件实际存在
    uniq = {p for t in tasks for p in t["image_paths"]}
    missing = sorted(p for p in uniq
                     if not os.path.exists(os.path.normpath(os.path.join(json_dir, p))))
    if missing:
        raise SystemExit(f"❌ {len(missing)}/{len(uniq)} 个 ref 图不存在,例如:{missing[:3]}")

    # 4. 任务总数 / 图数(对不上说明枚举规则写错了——规格 §7:不要调规则凑数,上报)
    n_images = sum(len(t["variants"]) for t in tasks)
    if len(tasks) != EXPECTED_TOTAL_TASKS or n_images != EXPECTED_TOTAL_IMAGES:
        raise SystemExit(f"❌ 总数对不上:任务 {len(tasks)}/{EXPECTED_TOTAL_TASKS},"
                         f"图 {n_images}/{EXPECTED_TOTAL_IMAGES}")

    # 5. 分层条数与 §3.2 表格逐格对照 + S1 恰好 44 个不重复组合
    for stratum, (exp_tasks, exp_images) in EXPECTED.items():
        sub = [t for t in tasks if t["stratum"] == stratum]
        n_img = sum(len(t["variants"]) for t in sub)
        if len(sub) != exp_tasks or n_img != exp_images:
            raise SystemExit(f"❌ {stratum} 层对不上:任务 {len(sub)}/{exp_tasks},"
                             f"图 {n_img}/{exp_images}")
    s1_combos = {tuple(t["meta"]["subjects"]) for t in tasks if t["stratum"] == "S1"}
    if len(s1_combos) != 44:
        raise SystemExit(f"❌ S1 应有 44 个不重复组合,实际 {len(s1_combos)}")

    # 6. seed 唯一性(S0 除外:5 个锚点刻意共用冒烟的 3407);S1–S4 区间与 M1 不重叠
    seeds = [t["seed"] for t in tasks if t["stratum"] != "S0"]
    if len(set(seeds)) != len(seeds):
        dup = [s for s, n in Counter(seeds).items() if n > 1]
        raise SystemExit(f"❌ S1–S4 seed 有重复:{dup[:5]}")
    lo, hi = SEED_RANGE_S1_S4
    bad = [s for s in seeds if s != SMOKE_SEED and not lo <= s <= hi]
    if bad:
        raise SystemExit(f"❌ seed 越出 {lo}–{hi}:{bad[:5]}")
    m1_lo, m1_hi = M1_SEED_RANGE
    overlap = [s for s in seeds if m1_lo <= s <= m1_hi]
    if overlap:
        raise SystemExit(f"❌ seed 与 M1 区间 {m1_lo}–{m1_hi} 重叠:{overlap[:5]}")
    return len(uniq)


# ------------------------------------------------------------------ 统计表(规格 §3.5)

def print_stats(tasks, s1_combos, s4_picked, s4_coverage, object_tpl):
    print("\n" + "=" * 76)
    print("M4 评测集统计(M4_EVAL_SPEC §3.5)")
    print("=" * 76)

    print("\n[1] 分层任务数 / 图数(与 §3.2 表格逐格对照)")
    print(f"  {'层':<4}{'任务':>8}{'期望':>8}{'图数':>8}{'期望':>8}  对照")
    for stratum, (exp_t, exp_i) in EXPECTED.items():
        sub = [t for t in tasks if t["stratum"] == stratum]
        n_img = sum(len(t["variants"]) for t in sub)
        ok = "✓" if (len(sub), n_img) == (exp_t, exp_i) else "✗"
        print(f"  {stratum:<4}{len(sub):>8}{exp_t:>8}{n_img:>8}{exp_i:>8}  {ok}")
    n_images = sum(len(t["variants"]) for t in tasks)
    print(f"  {'合计':<4}{len(tasks):>8}{EXPECTED_TOTAL_TASKS:>8}"
          f"{n_images:>8}{EXPECTED_TOTAL_IMAGES:>8}  "
          f"{'✓' if (len(tasks), n_images) == (EXPECTED_TOTAL_TASKS, EXPECTED_TOTAL_IMAGES) else '✗'}")

    print("\n[2] S1 的 44 个组合 + 各自分到的场景模板(k % 20 轮转)")
    for k, (a, b) in enumerate(s1_combos):
        print(f"  k={k:02d}  {a} + {b}  →  tpl{k % 20:02d} \"{object_tpl[k % 20]}\"")

    print("\n[3] 每个 held-out subject 在各层的出现次数(按任务计,一个任务里出现一次记 1)")
    strata = list(EXPECTED)
    header = f"  {'subject':<22}" + "".join(f"{s:>5}" for s in strata) + f"{'合计':>7}"
    print(header)
    for subj in sorted(HELD_OUT):
        counts = [sum(1 for t in tasks if t["stratum"] == st and subj in t["meta"]["subjects"])
                  for st in strata]
        print(f"  {subj:<22}" + "".join(f"{c:>5}" for c in counts) + f"{sum(counts):>7}")

    print("\n[4] 场景模板使用直方图(按任务计)")
    tpl_use = Counter(t["meta"]["template_id"] for t in tasks)
    for tid in range(20):
        bar = "█" * tpl_use.get(tid, 0)
        print(f"  tpl{tid:02d} ({tpl_use.get(tid, 0):>3}) {bar} \"{object_tpl[tid]}\"")

    print("\n[5] S4 选中的 10 个 3-组合 + 覆盖分布")
    for k, combo in enumerate(s4_picked):
        print(f"  k={k}  {' + '.join(combo)}  →  tpl{k % 20:02d}")
    dist = Counter(s4_coverage.values())
    print(f"  覆盖分布(覆盖次数:subject 数):{dict(sorted(dist.items()))}"
          f"(参考实现应恰好每个 subject 3 次;断言下限 ≥2)")
    print(f"  逐 subject:{s4_coverage}")

    print("\n[6] seed 区间")
    s0_seeds = sorted({t["seed"] for t in tasks if t["stratum"] == "S0"})
    s14 = [t["seed"] for t in tasks if t["stratum"] != "S0"]
    print(f"  S0(锚点,复刻冒烟):{s0_seeds}")
    print(f"  S1–S4:{min(s14)}–{max(s14)}(规格要求 {SEED_RANGE_S1_S4[0]}–{SEED_RANGE_S1_S4[1]})")
    m1_lo, m1_hi = M1_SEED_RANGE
    print(f"  与 M1 区间 {m1_lo}–{m1_hi} 重叠检查:✓ 不重叠(断言已过)")
    print("=" * 76 + "\n")


# ------------------------------------------------------------------ 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--out_json", default=OUT_JSON)
    p.add_argument("--dry_run", action="store_true",
                   help="只枚举 + 打印统计表,不写任何文件")
    args = p.parse_args()

    # ---------- 元数据 + 名单断言(复用 gen_data,规则同源) ----------
    classes = load_classes(args.data_dir)
    assert_split_clean(classes)
    _shared, object_tpl = load_scene_templates(args.data_dir)   # 只用 object 20 条(§2.4)
    views = load_subject_images(args.data_dir, sorted(HELD_OUT))

    # ---------- 五层枚举 ----------
    tasks_s0 = build_s0(classes, object_tpl)
    tasks_s1, s1_combos = build_s1(classes, object_tpl, views)
    tasks_s2 = build_s2(classes, object_tpl, views)
    tasks_s3 = build_s3(classes, object_tpl, views)
    tasks_s4, s4_picked, s4_coverage = build_s4(classes, object_tpl, views)
    tasks = tasks_s0 + tasks_s1 + tasks_s2 + tasks_s3 + tasks_s4

    # ---------- 断言(规格 §3.4,缺一不可) ----------
    json_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(json_dir, exist_ok=True)   # ref 验存需要 json_dir 解析 `..`
    n_ref_files = assert_eval_set(tasks, json_dir)

    # ---------- 统计表(Stage A 的交付物) ----------
    print(f"ref 图验存:{n_ref_files} 个唯一文件全部可读 ✓")
    print_stats(tasks, s1_combos, s4_picked, s4_coverage, object_tpl)

    if args.dry_run:
        print("dry_run:未写任何文件。")
        return

    payload = {
        "meta": {"spec": "M4-eval-v1", "n_tasks": len(tasks),
                 "n_images": sum(len(t["variants"]) for t in tasks)},
        "tasks": tasks,
    }
    tmp = args.out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out_json)
    print(f"已写出 {args.out_json}:{len(tasks)} 个任务,"
          f"{payload['meta']['n_images']} 张图。")
    print("⛔ Stage A 到此为止。把统计表贴进 reports/M4_stageA.md 回传,"
          "等确认后再进 Stage B(distill/eval_multiref.py)。")


if __name__ == "__main__":
    main()
