#!/usr/bin/env python3
"""生成 M6 消融批的任务单:`m6_tasks.json`(320 条主池)+ `m6_floor_tasks.json`(30 条天花板)。

M6 要回答的是 `M6_ABLATION_SPEC.md` §1 的那一件事:

    除了 `--ref_isolation` 之外处处相同的两条腿,从零各训一遍,
    **隔离这个设计到底值多少代价**。

对照是 `m6_iso`(隔离 + KV cache)与 `m6_full`(全注意力,SPEC §3 硬约束:
基线腿不许开 KV,它没有隔离、cache 是有损的)。两者只差 stage-1 与蒸馏时的
一个 flag,其余(数据、步数、seed、全部超参)逐字相同。

## 任务池 192 → 320:扩张规则(**预登记,冻结前须经确认;出图后不许改**)

SPEC §5.2 登记了"主对 320 条",但**没有登记这 320 条怎么构成**。本文件补上这条,
规则是**把既有循环的取值范围拉长,不引入任何新公式**:

| 层 | 构成 | 条数 | 与 M4 任务单的关系 |
|---|---|---|---|
| S1  | 44 组合 × `object_tpl[k % 20]`      × s∈{0,1,2} | 132 | **逐字相同** |
| S1x | 同 44 组合 × `object_tpl[(k+10) % 20]` × s∈{3,4} | 88 | 新增 |
| S3  | 10 主体 × c∈{0,1,2} × s∈{0,1}                    | 60 | **逐字相同** |
| S3x | 同 10 主体 × c∈{3,4} × s∈{0,1}                   | 40 | 新增 |

- **seed 公式原样**(S1 `3_500_000+k*10+s`、S3 `3_700_000+i*100+c*10+s`),只是 s/c
  的取值范围变大 ⇒ 与既有 seed 零碰撞,且仍落在 `build_eval_json.py:SEED_RANGE_S1_S4`
  之内、与 M1 区间不重叠。
- `(k+10) % 20` 是 20 条模板轮转的**对跖点**:保证每个 k 都换到另一个场景,
  而且没有可挑的自由度(选 +1 还是 +7 就成了一次无规则的自由选择)。
- 多 ref : 单 ref = **220 : 100 = 68.75% : 31.25%**,与 192 批的 132:60 **完全同比**。
  扩张不改层配比,于是"扩了池子"与"换了题型"这两件事不会混在一起。
- **前 192 条的 prompt / refs / seed 与臂 A/B 批逐字相同**(`verify()` 拿
  `eval_set.json` 逐字段对一遍)⇒ 满足 SPEC §3「推理 seed 与臂 A/B 批逐字相同」。

`stratum` 仍然只写 `S1` / `S3`(扩出来的那 128 条不另立层名),扩张与否记在
`meta.m6_ext`。WHY:`stratum` 是 `report.py` 的分层键,也是盲评前端可见字段;
新造 `S1x` 会让本批的分层表与历史批次对不上,而扩张本身并不是一个新层。

## run_floor 30 对:为什么第二侧要另起一个 job

臂 B 那批的天花板是**跨会话**的(臂 A 批的图 ↔ 臂 B 批重生成的图),因为它的主对
两侧本来就来自两个批次。M6 不是这样:`m6_iso` 与 `m6_full` 是**同一进程、同一张卡、
变体外层循环**里先后生成的。照抄臂 B 的做法在同一个 job 内再生成一次同权重同 seed
的图,大概率**逐位相同** ⇒ 天花板退化成 100% 平局,尺子等于没有。

所以本批的天花板第二侧走**另一个 infer_hub job**(另一个进程、另一块卡),
权重取 `m6_full`(主对的基线侧)。代价必须随结论一起声明:

> **主对两侧同进程(run 噪声 ≈ 0),天花板两侧跨进程 ⇒ 本批天花板是噪声的上界,
> 偏保守。** 方向对被检验方不利,可以接受。

WHY 天花板取 `m6_full` 而不是臂 B 用的 `official_full`:臂 B 选官方权重是因为它由
pipeline 自带备份提供、不依赖任何 checkpoint 文件在不在盘上;而 M6 两腿的 checkpoint
本来就必须在盘上(它们是主对本身),那条理由在这里不成立。改选基线侧的好处是
天花板量的正是**骑在主对上的那一份噪声**,而不是另一个模型的噪声。

WHY seed 原样照抄、绝不偏移(与 `build_arm_b_tasks.py` 同纪律):本批要的是
"除了 `--ref_isolation` 什么都不变"。换 seed 会把"隔离的代价"和"噪声的代价"混在一起。

用法(在 UNO 仓库根目录):
    python distill/build_m6_tasks.py --dry_run   # 只看条数与成本,不写文件
    python distill/build_m6_tasks.py             # 写出 + 自检 + 回读复核
    python distill/build_m6_tasks.py --verify    # 只校验已有产物
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import combinations

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# 与 build_eval_json.py 同源:枚举规则只有一份实现,不重写(规格 §3.1 的既有纪律)
from gen_data import (  # noqa: E402
    HELD_OUT, assert_split_clean, load_classes, load_scene_templates,
    load_subject_images, make_prompt,
)

# ------------------------------------------------------------------ 常数(改动 = 换一批任务,不许悄悄改)

REPO = _REPO_ROOT
DATA_DIR = os.path.join(REPO, "datasets/dreambooth/dataset")
SRC_JSON = os.path.join(REPO, "datasets/eval_multiref/eval_set.json")
OUT_JSON = os.path.join(REPO, "datasets/eval_multiref/m6_tasks.json")
OUT_FLOOR_JSON = os.path.join(REPO, "datasets/eval_multiref/m6_floor_tasks.json")

M6_ISO = "m6_iso"
M6_FULL = "m6_full"

# 层配比(上面表格的四行)。对不上就是枚举规则写错了——不许调规则凑数,上报。
N_S1, N_S1X, N_S3, N_S3X = 132, 88, 60, 40
N_MAIN = N_S1 + N_S1X + N_S3 + N_S3X          # 320,SPEC §5.2 预登记
N_S1_SEEDS, N_S3_SCENES = 5, 5                # 扩张后的取值范围(核心是前 3 个)
SEED_RANGE = (3_500_000, 3_800_091)           # build_eval_json.py:SEED_RANGE_S1_S4,逐字
M1_SEED_RANGE = (3_407_000, 3_415_999)        # 同上,必须不重叠

# run_floor 锚点:20 : 10,与臂 B 批同形(那批是 132:60 上取 20:10)。
N_RUNFLOOR = {"S1": 20, "S3": 10}
# 选样哈希盐。改它 = 换一批锚点 = 已生成的图作废。与臂 A/B 的盐都不同:
# 沿用旧盐会抽到高度重叠的任务,天花板变成"重判上次那批"。
RUNFLOOR_SALT = "m6-runfloor-v1"

# 单张耗时:取自实测中位数,不是估计(与 build_arm_b_tasks.py 同源)。
#   m6_full(全注意力)  5.132s —— output/eval_arm_a/results.json 的 official_full,192 张
#   m6_iso (隔离+KV)   2.907s —— output/probe_iso/results.json 的 official_iso,192 张
# 单张耗时只取决于推理开关(ref_isolation / kv_cache),与挂哪份 LoRA 无关 ⇒ 拿它们当锚点合法。
COST_S_PER_IMG = {M6_FULL: 5.132, M6_ISO: 2.907}
LOAD_MIN = 2.0   # 模型加载:infer_hub 冒烟实测 96.1s(ceph),取 2 min


def runfloor_key(task_id: str) -> str:
    """确定性选样序。md5 而非 sorted(task_id):后者会让选出来的全挤在编号前段
    (同 `build_arm_b_tasks.py:runfloor_key` / `build_noise_floor.py:sort_key`)。"""
    return hashlib.md5(f"{task_id}|{RUNFLOOR_SALT}".encode()).hexdigest()


def pick_runfloor(tasks: list[dict]) -> set[str]:
    """挑出 30 条要在天花板 job 里再生成一次 `m6_full` 的任务 id。

    去重到**主体组合**级别:20 条锚点如果挤在 3 个组合上,测出来的是"这 3 个组合
    有多稳",不是天花板(照抄 `build_arm_b_tasks.py:pick_runfloor` 的理由)。
    S3 只有 10 个主体,10 条锚点 ⇒ 每个主体恰好一条,这是该层能给的最大分散。
    """
    chosen: set[str] = set()
    for stratum, n in N_RUNFLOOR.items():
        pool = sorted((t for t in tasks if t["stratum"] == stratum),
                      key=lambda t: runfloor_key(t["task_id"]))
        seen: set[tuple] = set()
        got: list[str] = []
        for t in pool:
            g = tuple(t["meta"]["subjects"])
            if g in seen:
                continue
            seen.add(g)
            got.append(t["task_id"])
            if len(got) == n:
                break
        if len(got) != n:
            raise SystemExit(
                f"❌ {stratum} 只凑出 {len(got)}/{n} 条互不重复的主体组合:"
                f"该层共 {len(pool)} 条、{len(seen)} 个不同组合")
        chosen.update(got)
    return chosen


# ------------------------------------------------------------------ 枚举(两层,规则见模块 docstring)

def _task(task_id, stratum, prompt, image_paths, seed, subjects, classes_,
          template_id, template, views, seed_slot, ext, m4_src):
    return {
        "task_id": task_id,
        "stratum": stratum,
        "prompt": prompt,
        "image_paths": image_paths,
        "seed": seed,
        "variants": [M6_ISO, M6_FULL],
        "meta": {
            "subjects": subjects,
            "classes": classes_,
            "n_refs": len(subjects),
            "template_id": template_id,
            "template": template,
            "views": views,
            "seed_slot": seed_slot,
            "m6_ext": ext,             # True = 192→320 扩出来的那 128 条
            "m4_src_task_id": m4_src,  # 非 None ⇒ 必须与 eval_set.json 逐字一致
            "m6_runfloor": False,      # pick_runfloor 之后回填
        },
    }


def build_s1(classes, object_tpl, views) -> list[dict]:
    """S1:44 个合法 2-组合 × 5 seed 槽;s<3 走 `object_tpl[k%20]`(与 M4 逐字),
    s∈{3,4} 走对跖模板 `object_tpl[(k+10)%20]`。"""
    combos = [c for c in combinations(sorted(HELD_OUT), 2)
              if len({classes[s] for s in c}) == 2]
    if len(combos) != 44:
        raise SystemExit(f"❌ S1 合法 2-组合应为 44(C(10,2)=45 减去 stuffed animal 对),"
                         f"实际 {len(combos)}")
    tasks = []
    for k, (a, b) in enumerate(combos):
        for s in range(N_S1_SEEDS):
            ext = s >= 3
            tpl_id = (k + 10) % 20 if ext else k % 20
            template = object_tpl[tpl_id]
            seed = 3_500_000 + k * 10 + s
            va = views[a][(s + 0) % len(views[a])]
            vb = views[b][(s + 1) % len(views[b])]
            tasks.append(_task(
                f"M6_S1_{k:03d}_s{s}", "S1",
                make_prompt([classes[a], classes[b]], template),
                [f"../dreambooth/dataset/{a}/{va}", f"../dreambooth/dataset/{b}/{vb}"],
                seed, [a, b], [classes[a], classes[b]], tpl_id, template,
                [va, vb], s, ext, None if ext else f"S1_{k:03d}_s{s}",
            ))
    return tasks


def build_s3(classes, object_tpl, views) -> list[dict]:
    """S3:10 个 held-out × 5 场景 × 2 seed;c<3 与 M4 逐字,c∈{3,4} 是扩出来的。

    `tpl_id = (i*3+c) % 20` 原样沿用 ⇒ 每个主体的 5 个场景互不重复。
    """
    tasks = []
    for i, subj in enumerate(sorted(HELD_OUT)):
        for c in range(N_S3_SCENES):
            ext = c >= 3
            tpl_id = (i * 3 + c) % 20
            template = object_tpl[tpl_id]
            for s in (0, 1):
                seed = 3_700_000 + i * 100 + c * 10 + s
                v = views[subj][(c * 2 + s) % len(views[subj])]
                tasks.append(_task(
                    f"M6_S3_{i:02d}c{c}_s{s}", "S3",
                    make_prompt([classes[subj]], template),
                    [f"../dreambooth/dataset/{subj}/{v}"],
                    seed, [subj], [classes[subj]], tpl_id, template,
                    [v], s, ext, None if ext else f"S3_{i * 3 + c:03d}_s{s}",
                ))
    return tasks


def build() -> tuple[dict, dict]:
    classes = load_classes(DATA_DIR)
    assert_split_clean(classes)
    _shared, object_tpl = load_scene_templates(DATA_DIR)
    views = load_subject_images(DATA_DIR, sorted(HELD_OUT))

    tasks = build_s1(classes, object_tpl, views) + build_s3(classes, object_tpl, views)
    runfloor = pick_runfloor(tasks)
    for t in tasks:
        t["meta"]["m6_runfloor"] = t["task_id"] in runfloor

    main = {
        "meta": {
            "spec": "M6-v1",
            "n_tasks": len(tasks),
            "n_images": sum(len(t["variants"]) for t in tasks),
            "n_runfloor": len(runfloor),
            "runfloor_salt": RUNFLOOR_SALT,
            "source": "M6_ABLATION_SPEC §5.2 的 320 主对;层配比与扩张规则见 "
                      "build_m6_tasks.py 模块 docstring 与 M6_STEP4_RUN.md",
        },
        "tasks": tasks,
    }
    # 天花板任务单:同 task_id / prompt / refs / seed,只生成 m6_full 一张。
    # 同 task_id 是**故意**的——build_pairs.py 靠它把天花板那一对与主对配起来;
    # 两批写在不同的 save_path,文件名不会撞。
    floor_tasks = [
        {**{k: v for k, v in t.items() if k != "variants"},
         "variants": [M6_FULL],
         "meta": {**t["meta"], "m6_floor_side": "b"}}
        for t in tasks if t["meta"]["m6_runfloor"]
    ]
    floor = {
        "meta": {
            "spec": "M6-floor-v1",
            "n_tasks": len(floor_tasks),
            "n_images": len(floor_tasks),
            "runfloor_salt": RUNFLOOR_SALT,
            "source": "天花板第二侧:与主批**另起一个 job**(另一进程/另一张卡)"
                      "重生成同权重同 seed 的 m6_full。理由见模块 docstring。",
        },
        "tasks": floor_tasks,
    }
    return main, floor


# ------------------------------------------------------------------ 自检

def verify(main: dict, floor: dict, out_json: str = OUT_JSON,
           src_json: str = SRC_JSON) -> None:
    """产出自检。写完自动跑一遍,也可以单独 `--verify` 跑,免得"生成"与"校验"
    共用同一份内存里的假设(同 `build_arm_b_tasks.py` 的纪律)。"""
    tasks = main["tasks"]
    errs: list[str] = []
    # 参考图相对 json 所在目录解析(eval_multiref.py:load_tasks 同规则),
    # 所以必须用**实际写出的路径**,锁死模块常量会在 --out 换目录时静默校验错地方。
    json_dir = os.path.dirname(os.path.abspath(out_json))

    # 1. 条数与层配比
    if len(tasks) != N_MAIN:
        errs.append(f"总数 {len(tasks)} 条,应为 {N_MAIN}(SPEC §5.2 预登记的 320 主对)")
    got = Counter((t["stratum"], bool(t["meta"]["m6_ext"])) for t in tasks)
    for key, exp in ((("S1", False), N_S1), (("S1", True), N_S1X),
                     (("S3", False), N_S3), (("S3", True), N_S3X)):
        if got.get(key, 0) != exp:
            errs.append(f"{key[0]}{'x' if key[1] else ''} {got.get(key, 0)} 条,应为 {exp}")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        errs.append("task_id 有重复")

    # 2. held-out 泄漏(两遍:名单 + 实际产物;同 build_eval_json:assert_eval_set)
    not_held = sorted({s for t in tasks for s in t["meta"]["subjects"]
                       if s not in set(HELD_OUT)})
    if not_held:
        errs.append(f"出现非 held-out subject:{not_held}")

    # 3. seed:唯一、落在既有区间、不与 M1 重叠
    seeds = [t["seed"] for t in tasks]
    if len(set(seeds)) != len(seeds):
        errs.append(f"seed 有重复:{[s for s, n in Counter(seeds).items() if n > 1][:5]}")
    lo, hi = SEED_RANGE
    bad = [s for s in seeds if not lo <= s <= hi]
    if bad:
        errs.append(f"seed 越出 {lo}–{hi}:{bad[:5]}")
    m1_lo, m1_hi = M1_SEED_RANGE
    if [s for s in seeds if m1_lo <= s <= m1_hi]:
        errs.append(f"seed 与 M1 区间 {m1_lo}–{m1_hi} 重叠")

    # 4. 变体与 ref 图
    for t in tasks:
        if t["variants"] != [M6_ISO, M6_FULL]:
            errs.append(f"{t['task_id']}: variants={t['variants']},应为 {[M6_ISO, M6_FULL]}")
        for rel in t["image_paths"]:
            if not os.path.isfile(os.path.normpath(os.path.join(json_dir, rel))):
                errs.append(f"{t['task_id']}: 参考图不存在 {rel}")

    # 5. 核心 192 条与 M4 任务单逐字一致 —— 本批"与臂 A/B 同 seed"这条前提就靠它
    errs.extend(_check_m4_core(tasks, src_json))

    # 6. run_floor:层配比、组合去重、与天花板任务单一致
    rf = [t for t in tasks if t["meta"]["m6_runfloor"]]
    rf_by_stratum = Counter(t["stratum"] for t in rf)
    if dict(rf_by_stratum) != N_RUNFLOOR:
        errs.append(f"run_floor 层配比 {dict(sorted(rf_by_stratum.items()))},应为 {N_RUNFLOOR}")
    for stratum in N_RUNFLOOR:
        groups = [tuple(t["meta"]["subjects"]) for t in rf if t["stratum"] == stratum]
        if len(set(groups)) != len(groups):
            errs.append(f"{stratum} 的 run_floor 锚点里主体组合有重复"
                        f"——天花板会被少数几个组合主导")
    ft = floor["tasks"]
    if len(ft) != sum(N_RUNFLOOR.values()):
        errs.append(f"天花板任务单 {len(ft)} 条,应为 {sum(N_RUNFLOOR.values())}")
    main_by_id = {t["task_id"]: t for t in tasks}
    for t in ft:
        src = main_by_id.get(t["task_id"])
        if src is None:
            errs.append(f"天花板 {t['task_id']} 不在主池里")
            continue
        if t["variants"] != [M6_FULL]:
            errs.append(f"天花板 {t['task_id']}: variants={t['variants']},应为 {[M6_FULL]}")
        for f in ("prompt", "image_paths", "seed", "stratum"):
            if t[f] != src[f]:
                errs.append(f"天花板 {t['task_id']}: {f} 与主池不一致 ← "
                            f"天花板的定义就是同权重同 seed,这条最要命")

    # 7. meta 计数
    for payload, name, n_task, n_img in (
            (main, "主池", len(tasks), sum(len(t["variants"]) for t in tasks)),
            (floor, "天花板", len(ft), len(ft))):
        if payload["meta"].get("n_tasks") != n_task:
            errs.append(f"{name} meta.n_tasks={payload['meta'].get('n_tasks')},实际 {n_task}")
        if payload["meta"].get("n_images") != n_img:
            errs.append(f"{name} meta.n_images={payload['meta'].get('n_images')},实际 {n_img}")

    if errs:
        print("\n❌ 自检未通过:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("✓ 自检通过(条数与层配比 / task_id 唯一 / 无 held-out 泄漏 / seed 唯一且在区间内 / "
          "变体固定 / 参考图存在 / 核心 192 条与 M4 逐字一致 / 锚点配比与组合去重 / "
          "天花板同 seed 同 prompt / meta 计数)")


def _check_m4_core(tasks: list[dict], src_json: str) -> list[str]:
    """核心 192 条必须与 `eval_set.json` 的 S1/S3 **逐字**相同。

    SPEC §3 那一行「推理 sampler/steps/guidance/seed/分辨率 与臂 A/B 批逐字相同」
    在任务单这一侧的落点就是这里。抄漏一行的代价是几十小时之后才看得出来。
    """
    if not os.path.isfile(src_json):
        print(f"⚠️ {os.path.relpath(src_json, REPO)} 不在本机,跳过与 M4 任务单的逐字比对"
              f"——**上机后必须补跑一次 `--verify`**")
        return []
    with open(src_json, "rt", encoding="utf-8") as f:
        src_by_id = {t["task_id"]: t for t in json.load(f)["tasks"]}
    errs: list[str] = []
    n = 0
    for t in tasks:
        sid = t["meta"]["m4_src_task_id"]
        if sid is None:
            continue
        n += 1
        src = src_by_id.get(sid)
        if src is None:
            errs.append(f"{t['task_id']}: 溯源 id {sid} 在 eval_set.json 里不存在")
            continue
        for f in ("prompt", "image_paths", "seed", "stratum"):
            if t[f] != src[f]:
                errs.append(f"{t['task_id']} ↔ {sid}: {f} 不一致 "
                            f"({t[f]!r} vs {src[f]!r})")
    if n != N_S1 + N_S3:
        errs.append(f"带 m4_src_task_id 的只有 {n} 条,应为 {N_S1 + N_S3}")
    return errs


# ------------------------------------------------------------------ 报表

def summarize(main: dict, floor: dict) -> None:
    tasks = main["tasks"]
    n = len(tasks)
    print(f"\n主池 {n} 条 / 出图 {main['meta']['n_images']} 张;"
          f"天花板另起 job {len(floor['tasks'])} 条 / {floor['meta']['n_images']} 张")
    by = Counter((t["stratum"], "扩" if t["meta"]["m6_ext"] else "核心") for t in tasks)
    for key in (("S1", "核心"), ("S1", "扩"), ("S3", "核心"), ("S3", "扩")):
        print(f"  {key[0]} {key[1]:<3} {by.get(key, 0):>4} 条")
    multi = sum(1 for t in tasks if t["meta"]["n_refs"] > 1)
    print(f"  多 ref : 单 ref = {multi} : {n - multi} = "
          f"{multi / n:.2%} : {1 - multi / n:.2%}(192 批是 132:60 = 68.75%:31.25%)")
    rf = [t for t in tasks if t["meta"]["m6_runfloor"]]
    print(f"  run_floor 锚点 {len(rf)} 条:{dict(sorted(Counter(t['stratum'] for t in rf).items()))}")
    print(f"\n终批将是 {n} 对 m6_iso vs m6_full + {len(rf)} 对 run_floor = {n + len(rf)} 对")
    print(f"  按边 ③ 实测 53.6% 平局率折合非平局样本 ≈ {round(n * 0.4635)}"
          f"(§8.2 判据要求 n_nontie ≥ 94、Wilson CI 下界 ≥ 0.40)")
    breakeven = 1.0 - 94 / n
    print(f"  ⚠️ 平局率超过 {breakeven:.1%} 时 n_nontie 跌破 94 ⇒ 结论是「判据不适用」"
          f"而非「不达标」(§8.2),**不许事后追加样本**(SPEC §5.2 / §11.7)。")
    print("  ⚠️ 天花板两侧跨进程、主对两侧同进程 ⇒ 本批天花板是噪声的**上界**,"
          "这一条随结论一起声明,不许省。")


def print_dry_run(main: dict, floor: dict) -> None:
    tasks = main["tasks"]
    print(f"主 job:{len(tasks)} 条任务,{sum(len(t['variants']) for t in tasks)} 次生成")
    total = 0.0
    for v in (M6_ISO, M6_FULL):
        k = sum(1 for t in tasks if v in t["variants"])
        s = k * COST_S_PER_IMG[v]
        total += s
        print(f"  {v:<10}{k:>4} 张 × {COST_S_PER_IMG[v]:.3f}s ≈ {s / 60:5.1f} min")
    print(f"  单卡纯 denoise ≈ {total / 60:.1f} min(+ 加载 ~{LOAD_MIN:.0f} min)"
          f" ⇒ 端到端 ≈ {total / 60 + LOAD_MIN:.0f} min ⇒ infer_hub `--gpus 1 --timeout 75`")
    fs = len(floor["tasks"]) * COST_S_PER_IMG[M6_FULL]
    print(f"\n天花板 job:{len(floor['tasks'])} 张 × {COST_S_PER_IMG[M6_FULL]:.3f}s "
          f"≈ {fs / 60:.1f} min(+ 加载 ~{LOAD_MIN:.0f} min)"
          f" ⇒ 端到端 ≈ {fs / 60 + LOAD_MIN:.0f} min ⇒ `--gpus 1 --timeout 30`")
    print(f"\n  ⇒ **两个 job 都单卡**:denoise 只有 {total / 60:.0f} min,分片省下的时间"
          f"抵不过每片各付一次加载,而且 shard 越多越容易出「某片漏跑」的对账麻烦"
          f"(臂 B 那批的既有判断,逐字沿用)。")
    print(f"  注:m6_iso 走隔离 + KV,单张 {COST_S_PER_IMG[M6_ISO]}s,比全注意力的 "
          f"{COST_S_PER_IMG[M6_FULL]}s 快 {COST_S_PER_IMG[M6_FULL] / COST_S_PER_IMG[M6_ISO]:.2f}×"
          f"——这个比值本身就是上机后判「隔离到底开没开」的信号。")


def _write(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # 原子写,同 build_arm_b_tasks / build_probe_iso
    print(f"已写出 {os.path.relpath(path, REPO)}({os.path.getsize(path) / 1024:.0f} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC_JSON, help="M4 任务单,用于核心 192 条的逐字比对")
    p.add_argument("--out", default=OUT_JSON)
    p.add_argument("--out_floor", default=OUT_FLOOR_JSON)
    p.add_argument("--verify", action="store_true", help="只校验已有产物,不重新生成")
    p.add_argument("--dry_run", action="store_true", help="不写文件,只打印条数与成本估算")
    args = p.parse_args()

    if args.verify:
        for path in (args.out, args.out_floor):
            if not os.path.exists(path):
                raise SystemExit(f"❌ {path} 不存在")
        with open(args.out, "rt", encoding="utf-8") as f:
            m = json.load(f)
        with open(args.out_floor, "rt", encoding="utf-8") as f:
            fl = json.load(f)
        verify(m, fl, args.out, args.src)
        summarize(m, fl)
        return

    m, fl = build()

    if args.dry_run:
        print_dry_run(m, fl)
        summarize(m, fl)
        return

    verify(m, fl, args.out, args.src)
    _write(args.out, m)
    _write(args.out_floor, fl)

    # 回读复核:真正喂给 eval_multiref.py 的是**磁盘上这两个文件**,不是内存里的 dict。
    with open(args.out, "rt", encoding="utf-8") as f:
        m_back = json.load(f)
    with open(args.out_floor, "rt", encoding="utf-8") as f:
        fl_back = json.load(f)
    print("\n回读复核:")
    verify(m_back, fl_back, args.out, args.src)
    summarize(m_back, fl_back)


if __name__ == "__main__":
    main()
