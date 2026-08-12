"""Q1-B(teacher 3-ref 能力探底)的任务表。

产出 `datasets/eval_multiref/q1b_3ref_tasks.json`,122 条,覆盖 held-out 全部 112 个
合法 3-组合。**枚举规则在本地跑、产物进 git**,远程 agent 只负责跑图,不枚举任何东西。

两段:
  段 A  `build_eval_json.build_s4` 那 10 个贪心组合 × 2 seed = 20 条,
        task_id / prompt / image_paths / seed **与 eval_set.json 逐字相同**
        ⇒ 与 M4 的 UNO S4 直接可比(那批的 teacher 人工通过率只有 3.5%)。
  段 B  其余 102 个组合各 1 条,补齐 112 的覆盖。

seed:段 A 沿用 3_800_000+k*10+s;段 B 用 3_810_000+j*10(j = 组合在 112 条字典序里的
下标),与 S1–S4 既有区间 (3_500_000, 3_800_091) 和 M1 区间 (3_407_000, 3_415_999) 都不重叠。

用法:python qwen/build_q1b_3ref_tasks.py [--verify]
"""
import argparse
import json
import os
import sys
from itertools import combinations

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "distill"))

from gen_data import (  # noqa: E402
    HELD_OUT, load_classes, load_scene_templates, load_subject_images, make_prompt,
)

DATA_DIR = os.path.join(_REPO_ROOT, "datasets/dreambooth/dataset")
EVAL_SET = os.path.join(_REPO_ROOT, "datasets/eval_multiref/eval_set.json")
OUT_JSON = os.path.join(_REPO_ROOT, "datasets/eval_multiref/q1b_3ref_tasks.json")
SEED_A_BASE = 3_800_000
SEED_B_BASE = 3_810_000


def legal_combos(classes):
    """112 个无同类 3-组合,字典序。与 build_eval_json.build_s4 的第一步逐字相同。"""
    c3 = [c for c in combinations(sorted(HELD_OUT), 3) if len({classes[s] for s in c}) == 3]
    assert len(c3) == 112, f"合法 3-组合应为 112,实际 {len(c3)}"
    return c3


def greedy_ten(c3, classes):
    """复刻 build_eval_json.build_s4 的贪心选 10。逐字照抄,不许改写。"""
    coverage = {s: 0 for s in sorted(HELD_OUT)}
    remaining, picked = list(c3), []
    for _ in range(10):
        min_cov = min(coverage.values())
        best = max(remaining, key=lambda c: sum(1 for s in c if coverage[s] == min_cov))
        picked.append(best)
        remaining.remove(best)
        for s in best:
            coverage[s] += 1
    return picked


def task(task_id, prompt, combo, views_sel, seed, template_id, template):
    return {
        "task_id": task_id,
        "stratum": "S4",
        "prompt": prompt,
        "image_paths": [f"../dreambooth/dataset/{s}/{v}" for s, v in zip(combo, views_sel)],
        "seed": seed,
        "variants": ["qwen2511"],
        "meta": {"subjects": list(combo), "classes": None, "n_refs": 3,
                 "template_id": template_id, "template": template, "views": views_sel},
    }


def build():
    classes = load_classes(DATA_DIR)
    _shared, object_tpl = load_scene_templates(DATA_DIR)   # 只用 object 20 条,与 build_eval_json.py:367 同
    views = load_subject_images(DATA_DIR, sorted(HELD_OUT))
    c3 = legal_combos(classes)
    picked = greedy_ten(c3, classes)

    tasks = []
    for k, combo in enumerate(picked):                      # 段 A
        template = object_tpl[k % 20]
        for s in (0, 1):
            vs = [views[sub][(s + j) % len(views[sub])] for j, sub in enumerate(combo)]
            t = task(f"S4_{k:03d}_s{s}",
                     make_prompt([classes[sub] for sub in combo], template),
                     combo, vs, SEED_A_BASE + k * 10 + s, k % 20, template)
            t["meta"]["classes"] = [classes[sub] for sub in combo]
            t["meta"]["segment"] = "A"
            tasks.append(t)

    for j, combo in enumerate(c3):                          # 段 B
        if combo in picked:
            continue
        template = object_tpl[j % 20]
        vs = [views[sub][j % len(views[sub])] for sub in combo]
        t = task(f"Q1B_{j:03d}_s0",
                 make_prompt([classes[sub] for sub in combo], template),
                 combo, vs, SEED_B_BASE + j * 10, j % 20, template)
        t["meta"]["classes"] = [classes[sub] for sub in combo]
        t["meta"]["segment"] = "B"
        tasks.append(t)
    return tasks


def verify(tasks):
    errs = []
    if len(tasks) != 122:
        errs.append(f"条数 {len(tasks)} != 122")

    combos = {tuple(t["meta"]["subjects"]) for t in tasks}
    if len(combos) != 112:
        errs.append(f"覆盖组合 {len(combos)} != 112")

    if any(t["meta"]["n_refs"] != 3 for t in tasks):
        errs.append("有非 3-ref 条目")

    leak = {s for t in tasks for s in t["meta"]["subjects"]} - set(HELD_OUT)
    if leak:
        errs.append(f"非 held-out 主体泄漏:{leak}")

    seeds = [t["seed"] for t in tasks]
    if len(set(seeds)) != len(seeds):
        errs.append("seed 有重复")
    if any(3_407_000 <= s <= 3_415_999 for s in seeds):
        errs.append("seed 与 M1 区间重叠")

    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        errs.append("task_id 有重复")

    # 段 A 必须与 eval_set.json 的 S4 逐字一致 —— 与 M4 可比就靠这条
    ref = {t["task_id"]: t for t in json.load(open(EVAL_SET))["tasks"] if t.get("stratum") == "S4"}
    seg_a = {t["task_id"]: t for t in tasks if t["meta"]["segment"] == "A"}
    if set(ref) != set(seg_a):
        errs.append(f"段 A 的 task_id 集合与 eval_set.json 不符")
    else:
        for tid, t in seg_a.items():
            for f in ("prompt", "image_paths", "seed"):
                if t[f] != ref[tid][f]:
                    errs.append(f"段 A {tid} 的 {f} 与 eval_set.json 不一致")

    base = os.path.join(_REPO_ROOT, "datasets/eval_multiref")
    missing = [p for t in tasks for p in t["image_paths"]
               if not os.path.exists(os.path.normpath(os.path.join(base, p)))]
    if missing:
        errs.append(f"{len(missing)} 张参考图不存在,例如 {missing[:3]}")

    if errs:
        raise SystemExit("❌ 自检失败:\n  " + "\n  ".join(errs))
    print(f"✓ 自检通过:122 条 / 112 组合全覆盖 / 全 held-out / seed 唯一不重叠 / "
          f"段 A 20 条与 eval_set.json 逐字一致 / 参考图齐全")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="只自检,不写文件")
    args = p.parse_args()

    tasks = build()
    verify(tasks)
    if not args.verify:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"meta": {"spec": "Q1B-qwen-3ref-v1", "n_tasks": len(tasks),
                                "n_combos": 112, "source": "held-out 10 主体的全部合法 3-组合"},
                       "tasks": tasks}, f, indent=2, ensure_ascii=False)
        print(f"写出 {OUT_JSON}")
