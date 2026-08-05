#!/usr/bin/env python3
"""生成臂 B 的 **§9 身份留存门** item 清单 `output/eval_arm_b/gate_items.json`。

对应 `DISTILL_PLAN.md` §11.11(d)(**出图之前写死的预登记**):

> 在任何 §8 偏好判读之前先做 §9 客观计数。逐字复用 P-probe 的抽样
> (同 30 条任务、同 `random.Random(20260803)`),只数 `arm_b` 一个变体的 51 问。
> 锚点 **0/51**(`official_iso`)与 **45/51 = 88.2%**(`official_full`)现成可比。

| per-subject 留存(**点估计**) | 判定 |
|---|---|
| **< 60%** | 偏好批取消,项目在此收口,判读总支出 5 分钟 |
| **≥ 60%** | 进 §11.11(c) 步 5,222 对终批 |

**阈值 60% 是预登记的,出图之后不许改。** 用点估计不用 CI:51 问在 60% 处
Wilson 半宽约 ±13pp,用 CI 判两边都不确定,门就失去决断力。
**这是决策规则,不是测量值。**

> **诚实声明(引用这个门时必须一起引用)**:这把 §9 尺子只在**极端对比**
> (0% vs 88%)上验证过,§11.6「不许外推到细微差别」明写在案。
> 落在 50–70% 这个带里时它最不可靠。

## WHY 是新文件而不是就地改 `distill/idcount/build_items.py`

冻结纪律:`build_items.py` 生成的那份清单已经标注完、结果(0/51 与 45/51)已进报告。
就地改它 = 改掉一份已被引用的产物的**生成器**,以后没人能重放出当时那份清单。
新文件、新输出路径,旧的一个字节不动。

## WHY 抽样要"再推导一遍"而不是只从旧清单里读

§11.11(d) 要求「同 30 条任务」。有两条独立路径能得到这 30 条:

1. **读**`output/probe_iso/idcount_items.json` 里实际出现过的 task_id;
2. **推**:用 `random.Random(20260803)` 在本批任务池上重跑一遍同样的分层抽样。

本脚本**两条都走,并断言结果集合相等**。单走 (1) 的话,"同种子"就成了一句
无法验证的注释;单走 (2) 的话,一旦 `AB_` 前缀让 `sorted()` 的顺序与 `PI_` 时
不同(比如日后有人改了前缀长度),就会静默抽到另外 30 条,而**错的抽样看不出来**。
两条对不上就当场退出——这正是我们想让它炸的地方。

## WHY 不带 replay(P-probe 那次带了 6 条)

自洽率已经测过两次(M4→R2 跨 4 天 κ=0.274;臂 A 标定批场内 40 min κ=0.528),
§11.11(e) 已就此把终批的 replay 砍掉,理由是"第三次不承重"。门这里同理,
而且门的判读预算按预登记只有 **5 分钟**,塞 3 条重放会挤掉它。
需要时用 `--replay_frac` 打开,但那就偏离了预登记的判读预算,要写进局限。

用法:
    python distill/build_arm_b_gate.py --dry_run   # 只打印抽到哪 30 条、共几问
    python distill/build_arm_b_gate.py            # 写出 + 自检
    python distill/build_arm_b_gate.py --verify   # 只校验已有产物(需要图在盘上)

判读(出图之后,在 H800 上):
    python -m distill.idcount.server \
        --items output/eval_arm_b/gate_items.json \
        --marks output/eval_arm_b/gate_marks.json --port 8011
    python -m distill.idcount.report \
        output/eval_arm_b/gate_items.json output/eval_arm_b/gate_marks.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 路径解析规则的唯一来源,同 `distill/idcount/build_items.py` 的理由:
# 拼接一旦和推理脚本差一个字符,--verify 会稳定报"文件不存在",而真正的错因是我们拼错了。
from distill.eval_multiref import out_path, resolve  # noqa: E402

# ---- 预登记常数。改任何一个 = 换一个门,不许悄悄改 ----
SEED = 20260803                  # 与 P-probe 的 idcount 逐字相同(§11.11(d) 明写)
N_S1, N_S3 = 21, 9               # 与 idcount/build_items.py 相同的层配比,合计 30 条
GATE_THRESHOLD = 0.60            # per-subject 留存点估计的分叉阈值
EXPECT_N_QUESTIONS = 51          # 21×2 + 9×1;§11.11(d) 写死的"51 问",对不上就是抽错了
VARIANT = "arm_b_iso"

TASKS_JSON = REPO_ROOT / "datasets/eval_multiref/arm_b_tasks.json"
TASKS_JSON_DIR = "datasets/eval_multiref"
SAVE_PATH = "output/eval_arm_b"          # 与出图时 `--save_path` 必须一致
PROBE_ITEMS = REPO_ROOT / "output/probe_iso/idcount_items.json"
OUT_JSON = REPO_ROOT / "output/eval_arm_b/gate_items.json"

SPEC_NAME = "M5-arm-b-gate-v1"
ORDER_SALT = "m5abgate-order-v1"         # 展示顺序的打散盐,与抽样种子分开

# 现成锚点(P-probe,同一把尺子、同 30 条任务):报告里门的读数必须与它们并排引用。
ANCHORS = {"official_iso": (0, 51), "official_full": (45, 51)}


def _decodable(path: Path) -> bool:
    """图存在且能完整解码。`Image.open` 惰性只读文件头,截断图只有 `.load()` 才炸
    (照抄 `distill.blind_eval.pairing.decodable`)。"""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def derive_task_ids(tasks: list[dict]) -> list[tuple[str, str]]:
    """路径 (2):用 `random.Random(SEED)` 在本批任务池上重跑 P-probe 的分层抽样。

    抽样调用序列必须与 `distill/idcount/build_items.py:105-112` **逐字相同**
    (一个 Random 实例、先 S1 后 S3、`rng.sample(sorted(pool), n)`),
    否则"同种子"给不出同一批任务。
    """
    ids_by_stratum: dict[str, list[str]] = {}
    for t in tasks:
        ids_by_stratum.setdefault(t["stratum"], []).append(t["task_id"])

    rng = random.Random(SEED)
    picked: list[tuple[str, str]] = []
    for stratum, n in (("S1", N_S1), ("S3", N_S3)):
        pool = sorted(ids_by_stratum.get(stratum, []))
        if len(pool) < n:
            raise SystemExit(f"❌ 层 {stratum} 只有 {len(pool)} 条任务,不够抽 {n} 条")
        picked.extend((stratum, tid) for tid in rng.sample(pool, n))
    return picked


def read_probe_task_ids(probe_items: Path = PROBE_ITEMS) -> set[str]:
    """路径 (1):从 P-probe 已标注过的清单里读出那 30 条源任务号(剥掉 `PI_` 前缀)。"""
    payload = json.loads(Path(probe_items).read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("seed") != SEED:
        raise SystemExit(f"❌ {probe_items} 的 meta.seed = "
                         f"{payload.get('meta', {}).get('seed')!r},本门要求 {SEED}")
    out = set()
    for it in payload["items"]:
        tid = it["task_id"]
        if not tid.startswith("PI_"):
            raise SystemExit(f"❌ {probe_items} 里出现非 PI_ 前缀的 task_id: {tid}")
        out.add(tid[len("PI_"):])
    return out


def build_manifest(tasks_json: Path = TASKS_JSON,
                   probe_items: Path = PROBE_ITEMS) -> dict:
    """纯函数:同种子 + 同任务单 → 逐字节相同的清单。不碰磁盘输出,便于测试直接调用。"""
    payload = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    by_id = {t["task_id"]: t for t in tasks}

    picked = derive_task_ids(tasks)

    # ---- 两条路径必须给出同一批 30 条源任务(见模块 docstring) ----
    derived_src = {tid[len("AB_"):] if tid.startswith("AB_") else tid for _, tid in picked}
    probe_src = read_probe_task_ids(probe_items)
    if derived_src != probe_src:
        only_d = sorted(derived_src - probe_src)[:5]
        only_p = sorted(probe_src - derived_src)[:5]
        raise SystemExit(
            "❌ 「同 30 条任务」不成立——重推的抽样与 P-probe 旧清单对不上。\n"
            f"   只在重推里: {only_d}\n"
            f"   只在旧清单里: {only_p}\n"
            "   §11.11(d) 要求逐字复用同一批任务,不许换。检查 task_id 前缀/排序是否变过。")

    items: list[dict] = []
    for seq, (stratum, tid) in enumerate(picked):
        task = by_id[tid]
        if VARIANT not in task["variants"]:
            raise SystemExit(f"❌ {tid} 的 variants={task['variants']} 里没有 {VARIANT}")
        ref_names = task["meta"]["subjects"]
        image_paths = [resolve(p, TASKS_JSON_DIR) for p in task["image_paths"]]
        if len(image_paths) != len(ref_names):
            raise SystemExit(f"❌ {tid}: image_paths 与 meta.subjects 长度不一致")
        items.append({
            "item_id": f"BG_{seq:04d}",   # 不含变体名——只有一个变体,但保持与 idcount 同形
            "task_id": tid,
            "variant": VARIANT,
            "stratum": stratum,
            "prompt": task["prompt"],
            "image_paths": image_paths,
            "ref_names": list(ref_names),
            "img_path": out_path(SAVE_PATH, tid, VARIANT),
            "replay_of": None,
        })

    n_q = sum(len(it["ref_names"]) for it in items)
    if n_q != EXPECT_N_QUESTIONS:
        raise SystemExit(f"❌ 共 {n_q} 问,§11.11(d) 写死的是 {EXPECT_N_QUESTIONS} 问"
                         f"——对不上说明抽到的不是同一批任务")

    # 展示顺序打散:不用抽样那个 rng 的后续状态(它在 P-probe 那边被 60 个 item
    # 的 shuffle 消耗过,这里只有 30 个,状态天然对不齐,沿用会给人"同一次抽样"的错觉)。
    # 改用与种子无关的 md5 排序,同 `build_pairs.py` 的打散纪律。
    items.sort(key=lambda it: hashlib.md5(
        f"{it['task_id']}|{ORDER_SALT}".encode()).hexdigest())

    return {
        "meta": {
            "spec": SPEC_NAME,
            "seed": SEED,
            "order_salt": ORDER_SALT,
            "variant": VARIANT,
            "n_tasks": len(items),
            "n_items": len(items),
            "n_questions": n_q,
            "n_replay": 0,
            "n_tasks_by_stratum": dict(sorted(Counter(s for s, _ in picked).items())),
            "gate_threshold": GATE_THRESHOLD,
            "anchors": {k: {"k": v[0], "n": v[1], "rate": v[0] / v[1]}
                        for k, v in ANCHORS.items()},
            "source": "§11.11(d) 臂 B 身份留存门;抽样逐字复用 P-probe 的 30 条任务",
        },
        "items": items,
    }


def check_items(manifest: dict, root: Path, check_images: bool = True) -> list[str]:
    """返回问题清单(空 = 通过)。**纯谓词,不写文件**,供 --verify 与测试共用。"""
    errs: list[str] = []
    meta = manifest.get("meta", {})
    items = manifest.get("items", [])

    if meta.get("spec") != SPEC_NAME:
        errs.append(f"meta.spec 应为 {SPEC_NAME!r},实为 {meta.get('spec')!r}")
    if meta.get("seed") != SEED:
        errs.append(f"meta.seed 应为 {SEED},实为 {meta.get('seed')!r}")
    if meta.get("gate_threshold") != GATE_THRESHOLD:
        errs.append(f"meta.gate_threshold 应为 {GATE_THRESHOLD},"
                    f"实为 {meta.get('gate_threshold')!r} ← 阈值是预登记的,不许改")
    if meta.get("n_tasks_by_stratum") != {"S1": N_S1, "S3": N_S3}:
        errs.append(f"层配比应为 S1={N_S1}/S3={N_S3},实为 {meta.get('n_tasks_by_stratum')!r}")

    if len(items) != N_S1 + N_S3:
        errs.append(f"item 总数应为 {N_S1 + N_S3},实为 {len(items)}")
    ids = [it.get("item_id") for it in items]
    if len(set(ids)) != len(ids):
        errs.append(f"item_id 有重复:{[i for i, c in Counter(ids).items() if c > 1][:5]}")
    if len({it.get('task_id') for it in items}) != len(items):
        errs.append("task_id 有重复——一条任务只该出一个 item")
    if any(it.get("variant") != VARIANT for it in items):
        errs.append(f"存在非 {VARIANT} 的 item ← 本门只数 arm_b 一个变体")
    if any(it.get("replay_of") for it in items):
        errs.append("存在 replay item ← 默认不带重放,见模块 docstring")

    n_q = sum(len(it.get("ref_names", [])) for it in items)
    if n_q != EXPECT_N_QUESTIONS:
        errs.append(f"共 {n_q} 问,应为 {EXPECT_N_QUESTIONS}")
    if meta.get("n_questions") != n_q or meta.get("n_items") != len(items):
        errs.append("meta 里的计数与 items 列表实际长度不一致")

    for it in items:
        if len(it.get("image_paths", [])) != len(it.get("ref_names", [])):
            errs.append(f"{it.get('item_id')}: image_paths 与 ref_names 长度不一致")

    if check_images:
        missing: list[str] = []
        for it in items:
            for rel in [it.get("img_path"), *it.get("image_paths", [])]:
                if not rel:
                    errs.append(f"{it.get('item_id')} 缺路径字段")
                    continue
                target = (root / rel).resolve()
                try:  # 清单里的路径当不可信输入,防路径逃逸(抄 blind_eval.pairing)
                    target.relative_to(root.resolve())
                except ValueError:
                    errs.append(f"{it.get('item_id')} 的路径逃出仓库:{rel}")
                    continue
                if not _decodable(target):
                    missing.append(f"{it.get('item_id')}: {rel}")
        if missing:
            errs.append(f"{len(missing)} 个图片缺失或不可解码,例如:\n    " +
                        "\n    ".join(missing[:5]))
    return errs


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # 原子替换:半写的 json 不能被 server 读到


def print_summary(manifest: dict) -> None:
    m = manifest["meta"]
    print(f"\n{m['n_items']} 个 item(任务 {m['n_tasks']} 条 {m['n_tasks_by_stratum']}),"
          f"共 **{m['n_questions']} 问**,变体只有 {m['variant']}")
    print(f"生成图目录:{SAVE_PATH}/  ← 出图时 `--save_path` 必须写这个")
    print("\n判定(**预登记,出图后不许改**):")
    a = m["anchors"]
    print(f"  锚点 official_iso  {a['official_iso']['k']}/{a['official_iso']['n']} "
          f"= {a['official_iso']['rate']:.1%}   (隔离未适配的地板)")
    print(f"  锚点 official_full {a['official_full']['k']}/{a['official_full']['n']} "
          f"= {a['official_full']['rate']:.1%}  (全注意力的天花板)")
    print(f"  per-subject 留存**点估计** < {GATE_THRESHOLD:.0%} ⇒ 偏好批取消,项目收口")
    print(f"                            ≥ {GATE_THRESHOLD:.0%} ⇒ 进 222 对终批")
    print("  ⚠️ 声明随门一起引用:这把尺子只在 0% vs 88% 的极端对比上验证过,"
          "落在 50–70% 带里时最不可靠;它是决策规则,不是测量值。")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成/校验臂 B §9 身份留存门的 item 清单",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="只校验已生成的 gate_items.json,不重新生成")
    ap.add_argument("--dry_run", action="store_true",
                    help="不写文件,只打印抽样与判定口径")
    ap.add_argument("--skip_image_check", action="store_true",
                    help="跳过图片存在性检查(本地仓库不带 png 时用)")
    args = ap.parse_args()

    check_img = not args.skip_image_check

    if args.verify:
        if not OUT_JSON.exists():
            print(f"❌ {OUT_JSON} 不存在,请先不带 --verify 跑一次", file=sys.stderr)
            return 1
        manifest = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        errs = check_items(manifest, REPO_ROOT, check_img)
        if errs:
            print("❌ gate_items.json 自检未通过:")
            for e in errs:
                print(f"  - {e}")
            raise SystemExit(1)
        print("✓ 校验通过")
        print_summary(manifest)
        return 0

    manifest = build_manifest()
    print("✓ 抽样两路一致(重推 random.Random(20260803) == P-probe 旧清单的 30 条)")

    if args.dry_run:
        by_st = Counter(it["stratum"] for it in manifest["items"])
        print(f"抽到 {dict(sorted(by_st.items()))}:"
              f"{sorted(it['task_id'] for it in manifest['items'])[:4]} …")
        print_summary(manifest)
        return 0

    errs = check_items(manifest, REPO_ROOT, check_img)
    if errs:
        print("❌ 自检未通过:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    _write_atomic(OUT_JSON, manifest)
    print(f"[INFO] 写入 {OUT_JSON}")
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
