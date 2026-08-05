#!/usr/bin/env python3
"""生成 M5 臂 B 读数的任务单 `arm_b_tasks.json`(`M5_ARM_B_RUN.md` 末尾"之后要做的" 第 1 条)。

臂 B 要回答的是 08-04 之后的**主命题本身**——三边链条的第 ②′ 边:

    同一份数据、同一个配方、同一个 init 下,
    **隔离相对全注意力的代价是多少**,换来 1.672× 加速。

对照是 `arm_a_full`(全注意力)与 `arm_b_iso`(隔离 + KV cache),两者
**只差 `--ref_isolation`** 这一个 flag,其余(官方 LoRA init、`train_mixed.json`、
4000 步、全部超参)逐字相同。

## 本批只生成两种图,加起来 222 张

| 变体 | 任务数 | 用途 | 对侧从哪来 |
|---|---|---|---|
| `arm_b_iso` | 192(S1 132 + S3 60) | 主命题 | **复用**臂 A 批现成的 192 张 `arm_a_full` |
| `official_full` | 30(S1 20 + S3 10) | `run_floor` 同会话天花板 | **复用**臂 A 批同任务的 `official_full` |

WHY 对侧敢复用、不像臂 A 那次要求"两个变体必须同会话重生成":
臂 A 那条纪律的依据是"本栈不可逐位复现 + 尺子跨批会漂"。前半句仍然成立,
但 08-04 的标定批已经把它的**感知读数**测出来了——同 seed 同权重异 run 的
平局率 **98.3%**(59/60,CI [0.911, 0.997])⇒ 跨会话差异在人眼层面可忽略。
理由被它自己的数据拆掉了,于是省下 192 张 `arm_a_full` 的 GPU(~16 min)。

WHY 还要同会话补 30 张 `official_full`:
上面那句"可忽略"是在 **M4(07-30)→ 臂A(08-04)** 这段间隔上测的,而本批要检验的
`arm_b_iso` vs `arm_a_full` 跨的是 **臂A(08-04)→ 本批** 这段。§11.3 步骤 1 实测到
会话漂移**不是常数**(smoke→M4 那段远大于 M4→臂A 那段),所以天花板必须在
**与被检验那一对相同的会话间隔**上重测一次。30 张 ≈ 2.6 min GPU,便宜。

WHY 天花板的两侧是 `official_full` 而不是 `arm_a_full`:
`arm_a_full` 的 bank 是 `log/arm_a/checkpoint-4000`,官方权重是 pipeline 自带的
备份(`eval_multiref.py:198-207`),后者不依赖任何 checkpoint 文件在不在盘上。
两侧都是同一份权重、同一个 seed、只差生成会话——这正是天花板的定义,
换成哪一份权重都成立,选依赖最少的那个。

WHY 那 30 条从这 192 条里抽、而不是从 S2/S4 里抽:
S2/S4 在臂 A 批里**根本没出图**,拿它们做天花板就只能对 M4 的旧图,
跨的又变回 07-30 那段间隔,前一段 WHY 就白写了。代价是这 30 条的
prompt/refs 会在终批里出现两次(一次 `arm_b` 对、一次 `run_floor` 对),
**这个重复躲不掉**——处置写在 `build_pairs.py` 的 `cmd_arm_b` 里
(两条强制拉开 ≥ 1/3 批长),并要进 §8.5 局限。

WHY seed 原样照抄、绝不偏移(与 `build_arm_a_tasks.py` / `build_probe_iso.py` 同纪律):
这一批要的是"除了 `--ref_isolation` 什么都不变"。换 seed 会把"隔离的代价"和
"噪声的代价"混在一起(噪声地板实测 45.0% [0.258, 0.658],量级足以吞掉小效应)。

WHY 不掺 replay:自洽率已经在 M4→R2(跨 4 天 55.0%,κ=0.274)与臂 A 标定批
(场内 40 min 70.0%,κ=0.528)测过两次,第三次不承重,判读预算全部留给主命题。

用法(在 UNO 仓库根目录):
    python distill/build_arm_b_tasks.py --dry_run   # 只看条数与成本,不写文件
    python distill/build_arm_b_tasks.py             # 写出 + 自检 + 回读复核
    python distill/build_arm_b_tasks.py --verify    # 只校验已有产物
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

# ------------------------------------------------------------------ 常数(改动 = 换一批任务,不许悄悄改)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JSON = os.path.join(REPO, "datasets/eval_multiref/eval_set.json")
ARM_A_JSON = os.path.join(REPO, "datasets/eval_multiref/arm_a_tasks.json")
ARM_A_RESULTS = os.path.join(REPO, "output/eval_arm_a/results.json")
OUT_JSON = os.path.join(REPO, "datasets/eval_multiref/arm_b_tasks.json")

ARM_B = "arm_b_iso"
TEACHER = "official_full"

STRATA = ("S1", "S3")
N_S1, N_S3 = 132, 60
PREFIX = "AB_"

# run_floor 锚点:按臂 A 标定批的 40:20 等比缩半到 20:10(= 132:60 的层配比)。
# 缩半的理由是这次只用来"复核天花板还在不在",不是第一次测它;
# 30 条在 98.3% 附近的 Wilson 半宽约 ±11pp,够判"还在 90% 以上"。
N_RUNFLOOR = {"S1": 20, "S3": 10}
# 选样哈希盐。改它 = 换一批 run_floor 锚点 = 已生成的图作废。
# 与臂 A 标定批的 "m5aactl-runfloor" 不同盐:那 60 条已经判过一遍,
# 沿用同盐会让这次抽到高度重叠的任务,天花板变成"重判上次那批"。
RUNFLOOR_SALT = "m5ab-runfloor-v1"

# 单张耗时:取自实测中位数,不是估计。
#   official_full 5.132s —— output/eval_arm_a/results.json(192 张)
#   arm_b_iso     2.907s —— output/probe_iso/results.json 的 official_iso(192 张)
# 隔离+KV 的单张耗时只取决于推理开关,与挂哪份 LoRA 无关,所以拿 official_iso 当锚点合法。
COST_S_PER_IMG = {TEACHER: 5.132, ARM_B: 2.907}


def runfloor_key(task_id: str) -> str:
    """确定性选样序。md5 而非 sorted(task_id):后者会让选出来的全挤在编号前段
    (同 `build_noise_floor.py:sort_key` 的纪律)。"""
    return hashlib.md5(f"{task_id}|{RUNFLOOR_SALT}".encode()).hexdigest()


def pick_runfloor(picked: list[dict]) -> set[str]:
    """挑出要额外补一张 `official_full` 的 30 条源任务 id。

    去重到**主体组合**级别:20 条锚点如果挤在 3 个组合上,测出来的是"这 3 个组合
    跨会话有多稳",不是天花板(照抄 `build_noise_floor.py:pick` 的理由)。
    """
    chosen: set[str] = set()
    for stratum, n in N_RUNFLOOR.items():
        pool = sorted((t for t in picked if t["stratum"] == stratum),
                      key=lambda t: runfloor_key(t["task_id"]))
        seen: set[tuple] = set()
        got = []
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


def build(src_json: str) -> dict:
    with open(src_json, "rt", encoding="utf-8") as f:
        src = json.load(f)
    picked = [t for t in src["tasks"] if t["stratum"] in STRATA]

    runfloor = pick_runfloor(picked)

    out: list[dict] = []
    for t in picked:
        is_rf = t["task_id"] in runfloor
        meta = dict(t["meta"])            # 逐字复制,只加两个可追溯字段
        meta["arm_b_src_task_id"] = t["task_id"]
        meta["arm_b_runfloor"] = is_rf
        # 变体顺序与 eval_multiref.py:VARIANTS 表同序(official_full 在 arm_b_iso 之前)。
        # 表是外层循环,顺序本身不影响 swap 次数,但对齐了就不用在读日志时再对一次表。
        variants = ([TEACHER] if is_rf else []) + [ARM_B]
        out.append({
            "task_id": f"{PREFIX}{t['task_id']}",
            "stratum": t["stratum"],
            "prompt": t["prompt"],
            "image_paths": list(t["image_paths"]),
            "seed": t["seed"],            # 原样照抄,绝不偏移——见模块 docstring
            "variants": variants,
            "meta": meta,
        })

    n_img = sum(len(t["variants"]) for t in out)
    return {
        "meta": {
            "spec": "M5-arm-b-v1",
            "n_tasks": len(out),
            "n_images": n_img,
            "n_runfloor": len(runfloor),
            "runfloor_salt": RUNFLOOR_SALT,
            "source": "eval_set.json S1+S3;对侧复用 output/eval_arm_a 的 arm_a_full / official_full",
        },
        "tasks": out,
    }


def verify(payload: dict, src_json: str, out_json: str = OUT_JSON,
           check_arm_a: bool = True) -> None:
    """产出自检。写完自动跑一遍,也可以单独 `--verify` 跑,免得"生成"与"校验"
    共用同一份内存里的假设(同 `build_arm_a_tasks.py` / `build_probe_iso.py` 的纪律)。"""
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

    rf_by_stratum: Counter = Counter()
    rf_groups: dict[str, set] = {}
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
            errs.append(f"{t['task_id']}: seed 与源不一致 ← 本批前提就是只差一个 flag,这条最要命")
        if t["meta"].get("arm_b_src_task_id") != src_id:
            errs.append(f"{t['task_id']}: meta.arm_b_src_task_id 与前缀剥离结果不一致")

        is_rf = bool(t["meta"].get("arm_b_runfloor"))
        expect = ([TEACHER] if is_rf else []) + [ARM_B]
        if t["variants"] != expect:
            errs.append(f"{t['task_id']}: variants={t['variants']},"
                        f"按 arm_b_runfloor={is_rf} 应为 {expect}")
        if is_rf:
            rf_by_stratum[t["stratum"]] += 1
            g = tuple(t["meta"]["subjects"])
            bucket = rf_groups.setdefault(t["stratum"], set())
            if g in bucket:
                errs.append(f"{t['task_id']}: run_floor 锚点里主体组合 {g} 重复"
                            f"——天花板会被少数几个组合主导")
            bucket.add(g)

        for rel in t["image_paths"]:
            p = os.path.normpath(os.path.join(json_dir, rel))
            if not os.path.isfile(p):
                errs.append(f"{t['task_id']}: 参考图不存在 {p}")
        n_img_actual += len(t["variants"])

    if dict(rf_by_stratum) != N_RUNFLOOR:
        errs.append(f"run_floor 层配比 {dict(sorted(rf_by_stratum.items()))},应为 {N_RUNFLOOR}")
    if payload["meta"].get("n_images") != n_img_actual:
        errs.append(f"meta.n_images={payload['meta'].get('n_images')},实际 {n_img_actual}")
    if payload["meta"].get("n_tasks") != len(tasks):
        errs.append(f"meta.n_tasks={payload['meta'].get('n_tasks')},实际 {len(tasks)}")
    if payload["meta"].get("n_runfloor") != sum(rf_by_stratum.values()):
        errs.append(f"meta.n_runfloor={payload['meta'].get('n_runfloor')},"
                    f"实际 {sum(rf_by_stratum.values())}")

    # 对侧必须已经在臂 A 批里出过图,否则终批配对时才发现缺图,那时 GPU 已经跑完了。
    # 只查 results.json 的记录,不查 png 本身——本地仓库不带图片(只提交 json)。
    if check_arm_a:
        errs.extend(_check_arm_a_side(tasks))

    if errs:
        print("\n❌ 自检未通过:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("✓ 自检通过(条数 / 前缀+源对应 / prompt+refs+stratum 逐字一致 / seed 零偏移 / "
          "变体随 run_floor 标志一致 / 锚点层配比与组合去重 / 参考图存在 / "
          "对侧臂A产物齐备 / n_images 一致)")


def _check_arm_a_side(tasks: list[dict]) -> list[str]:
    """本批每条任务的**对侧**都必须已在臂 A 批里生成成功。

    - 192 条全部要有 `AA_{src}/arm_a_full`(主命题的基线);
    - 30 条锚点还要有 `AA_{src}/official_full`(天花板的另一侧)。

    臂 A 的产物不在盘上(本地仓库只提交 results.json)⇒ 查记录而不是查文件。
    记录缺失是**硬错误**:终批那时才发现就意味着 192 张图白跑。
    """
    errs: list[str] = []
    if not os.path.isfile(ARM_A_RESULTS):
        return [f"臂 A 结果不存在: {ARM_A_RESULTS}(拿它当对侧,缺了就配不出对)"]
    with open(ARM_A_RESULTS, "rt", encoding="utf-8") as f:
        recs = json.load(f)["records"]
    ok = {(r["task_id"], r["variant"]) for r in recs if r.get("status") in ("ok", "skipped")}

    miss_base, miss_rf = [], []
    for t in tasks:
        aa_id = "AA_" + t["meta"]["arm_b_src_task_id"]
        if (aa_id, "arm_a_full") not in ok:
            miss_base.append(aa_id)
        if t["meta"].get("arm_b_runfloor") and (aa_id, TEACHER) not in ok:
            miss_rf.append(aa_id)
    if miss_base:
        errs.append(f"{len(miss_base)} 条缺臂A基线 arm_a_full,例如 {miss_base[:3]}")
    if miss_rf:
        errs.append(f"{len(miss_rf)} 条缺臂A锚点 official_full,例如 {miss_rf[:3]}")
    return errs


def summarize(payload: dict) -> None:
    tasks = payload["tasks"]
    rf = [t for t in tasks if t["meta"].get("arm_b_runfloor")]
    print(f"\n任务 {len(tasks)} 条 / 本批出图 {payload['meta']['n_images']} 张")
    print(f"  分层:{dict(sorted(Counter(t['stratum'] for t in tasks).items()))}")
    print(f"  run_floor 锚点 {len(rf)} 条:"
          f"{dict(sorted(Counter(t['stratum'] for t in rf).items()))}")
    print(f"\n终批将是 {len(tasks)} 对 arm_b_iso vs arm_a_full + {len(rf)} 对 run_floor "
          f"= {len(tasks) + len(rf)} 对(不带 replay)")
    n = len(tasks)
    print(f"  按臂 A 实测 59.4% 平局率折合非平局样本 ≈ {round(n * (1 - 0.594))}"
          f"(§8.2 判据要求 n_nontie ≥ 94、CI 下界 ≥ 0.40)")
    breakeven = 1.0 - 94 / n
    print(f"  ⚠️ 平局率超过 {breakeven:.1%} 时 n_nontie 跌破 94 ⇒ 结论是"
          f"「判据不适用」而非「不达标」(§8.2)。臂 A 那批实测 59.4% 就已经越线,"
          f"本批**很可能同样判据不适用**——这在上机前已知,不做补救"
          f"(补救 = 看到平局率之后再加样本 = 事后调判据)。")
    print(f"  注:§11.11(e) 只登记了批次组成(192 + 30,无 replay),**没有**登记"
          f"「平局率是主读数」。判读完按 §8.2 原样算,落在哪一档就报哪一档;"
          f"可主张的句式见 §11.11(g)⑥。")


def print_dry_run(payload: dict, num_shards: int = 8) -> None:
    tasks = payload["tasks"]
    n_gen = sum(len(t["variants"]) for t in tasks)
    print(f"任务 {len(tasks)} 条,总生成次数 {n_gen}")
    total_s = 0.0
    for v in (TEACHER, ARM_B):
        n = sum(1 for t in tasks if v in t["variants"])
        s = n * COST_S_PER_IMG[v]
        total_s += s
        print(f"  {v:<16}{n:>4} 张 × {COST_S_PER_IMG[v]:.3f}s ≈ {s / 60:.1f} min")
    print(f"  单卡纯 denoise 合计 ≈ {total_s / 60:.1f} min(+ 模型加载 ~7 min)"
          f" ⇒ 单卡端到端 ≈ {total_s / 60 + 7:.0f} min")
    print(f"  分 {num_shards} shard 后单卡 ≈ {total_s / num_shards / 60:.1f} min "
          f"(+ 每 shard 各自 ~7 min 模型加载 ⇒ 端到端 "
          f"≈ {total_s / num_shards / 60 + 7:.0f} min)")
    print(f"  ⇒ **本批建议单卡跑**:denoise 只有 {total_s / 60:.0f} min,"
          f"分片省下的时间抵不过多付的 7 min×N 加载,而且 shard 越多"
          f"越容易出「某片漏跑」的对账麻烦。")
    print("  注:arm_b_iso 走隔离 + KV cache,单张 2.907s,比全注意力的 5.132s 快 1.77×;"
          "省下的 192 张 arm_a_full(~16 min)来自复用臂 A 批。")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC_JSON)
    p.add_argument("--out", default=OUT_JSON)
    p.add_argument("--verify", action="store_true", help="只校验已有的 --out,不重新生成")
    p.add_argument("--dry_run", action="store_true", help="不写文件,只打印任务数与成本估算")
    p.add_argument("--skip_arm_a_check", action="store_true",
                   help="跳过对侧臂A产物的存在性检查(只在臂A results.json 尚未同步时用)")
    p.add_argument("--num_shards", type=int, default=8,
                   help="--dry_run 时用来估算分 shard 后的单卡耗时")
    args = p.parse_args()

    check_aa = not args.skip_arm_a_check

    if args.verify:
        if not os.path.exists(args.out):
            raise SystemExit(f"❌ {args.out} 不存在")
        with open(args.out, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        verify(payload, args.src, args.out, check_aa)
        summarize(payload)
        return

    payload = build(args.src)

    if args.dry_run:
        print_dry_run(payload, args.num_shards)
        summarize(payload)
        return

    verify(payload, args.src, args.out, check_aa)

    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)  # 原子写,同 build_arm_a_tasks / build_probe_iso
    print(f"\n已写出 {args.out}({os.path.getsize(args.out) / 1024:.0f} KB)")

    # 回读复核:真正喂给 eval_multiref.py 的是**磁盘上这个文件**,不是内存里那个 dict。
    with open(args.out, "rt", encoding="utf-8") as f:
        back = json.load(f)
    print("回读复核:")
    verify(back, args.src, args.out, check_aa)
    summarize(back)


if __name__ == "__main__":
    main()
