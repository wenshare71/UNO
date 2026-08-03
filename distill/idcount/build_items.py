#!/usr/bin/env python3
"""生成 M5 身份留存计数标注的 item 清单 `output/probe_iso/idcount_items.json`。

背景:`output/probe_iso/` 下已有 P-probe 的产物——`datasets/eval_multiref/probe_iso_tasks.json`
里 192 条任务 × `official_full` / `official_iso` 两个变体 = 384 张生成图。目视已确认
`official_iso` 几乎完全丢失参考主体身份、退化成纯文生图。这份清单是把这个观察变成
**客观计数**的第一步:从 192 条里按层配比抽 30 条,展开成单图是/否问题,交给人工标注。

这不是偏好比较(见 `distill/blind_eval/`),是对**单张图**问"这个参考主体的身份保住了吗?"——
所以每个 item 只对应一个变体的一张生成图,item_id 里必须**不含变体名**,否则标注者
一看 ID 就能猜出哪张是 official_iso,整个客观计数就废了。

## 判定口径

**是(身份保住)**:生成图里存在一个物体,能与该参考主体认定为**同一个具体个体**——其特有的花纹 / 配色 / 形状细节 / 文字或图标标识,**至少一项明确对应**。

**否**:只有**类别**一致(都是背包 / 都是毛绒玩具 / 都是碗),或该主体压根没出现。

边界规则:
1. 只对上颜色、对不上任何结构或标识 → **否**
2. 主体出现但被严重遮挡 / 过小无法辨认 → **否**(不设第三档)
3. 同一主体在图中出现多次,任一实例对上 → **是**

用法:
    # 第一次:生成清单
    python distill/idcount/build_items.py

    # 上机 / 换机器后:校验清单与磁盘上的图片、参考图是否对得上
    python distill/idcount/build_items.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
# 直接 `python distill/idcount/build_items.py` 跑时,sys.path[0] 是 idcount/ 目录、
# 不含仓库根,`import distill.eval_multiref` 会找不到包——必须在 import 之前手动补上。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# eval_multiref.out_path / resolve 是生成图路径、参考图路径的**唯一**解析规则来源。
# 这里不重新实现,是因为路径拼接一旦和实际推理脚本的写法有一个字符不一致,
# --verify 就会稳定地报"文件不存在",而错误原因却是我们自己拼错了路径。
from distill.eval_multiref import out_path, resolve  # noqa: E402

# 种子硬编码为模块常量而不是 CLI 参数:这份清单只抽一次,历史上任何一次重新生成
# 都必须产出与第一次逐字节相同的结果,否则已经标注过的 item_id 就对不上新清单了。
SEED = 20260803

# 按 132:60(S1:S3 任务总数)的层配比抽 30 条,硬编码不做 CLI 参数——同样是"只抽一次"。
N_S1 = 21
N_S3 = 9

# 60 个 item 里抽 10% 做重放(自洽率检验),写成分数而不是硬编码 6,
# 是为了让"10%"这句话在代码里可以直接核对,而不是要靠注释去保证两处数字一致。
REPLAY_FRAC = 0.10

TASKS_JSON = REPO_ROOT / "datasets/eval_multiref/probe_iso_tasks.json"
# probe_iso_tasks.json 里的 image_paths 是相对**清单自身所在目录**写的相对路径
# (抄自 eval_multiref.resolve 的用法),所以这里也必须用同一个目录做 base。
TASKS_JSON_DIR = "datasets/eval_multiref"
# 生成图目录:与当时跑 P-probe 时 `--save_path output/probe_iso` 一致,
# 见 output/probe_iso/results.json 里记录的 path 字段(已核对过一致)。
SAVE_PATH = "output/probe_iso"

OUT_JSON = REPO_ROOT / "output/probe_iso/idcount_items.json"

SPEC_NAME = "M5-idcount-v1"


def _decodable(path: Path) -> bool:
    """图存在且能完整解码。抄 `distill.blind_eval.pairing.decodable` 的理由同源:
    `Image.open` 惰性只读文件头,只有 `.load()` 才会在截断图上炸。"""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def build_manifest(tasks_json: Path = TASKS_JSON) -> dict:
    """纯函数:同种子 + 同任务单 → 逐字节相同的清单。不碰磁盘输出,便于测试直接调用。"""
    payload = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    by_id = {t["task_id"]: t for t in tasks}

    ids_by_stratum: dict[str, list[str]] = {}
    for t in tasks:
        ids_by_stratum.setdefault(t["stratum"], []).append(t["task_id"])

    # 一个 Random 实例贯穿全程(两层抽样 → 打乱 item 顺序 → 抽重放 → 打乱重放顺序)。
    # 这样"可复现"只需要记住一个种子,不必分别确认五个调用点是否各自用对了同一个种子。
    rng = random.Random(SEED)

    picked: list[tuple[str, str]] = []  # (stratum, task_id),按 (S1, S3) 顺序抽
    for stratum, n in (("S1", N_S1), ("S3", N_S3)):
        pool = sorted(ids_by_stratum.get(stratum, []))
        if len(pool) < n:
            raise SystemExit(f"❌ 层 {stratum} 只有 {len(pool)} 条任务,不够抽 {n} 条")
        picked.extend((stratum, tid) for tid in rng.sample(pool, n))

    items: list[dict] = []
    seq = 0
    for stratum, tid in picked:
        task = by_id[tid]
        ref_names = task["meta"]["subjects"]
        image_paths = [resolve(p, TASKS_JSON_DIR) for p in task["image_paths"]]
        if len(image_paths) != len(ref_names):
            raise SystemExit(f"❌ {tid}: image_paths 与 meta.subjects 长度不一致")
        for variant in task["variants"]:
            items.append({
                "item_id": f"IC_{seq:04d}",
                "task_id": tid,
                "variant": variant,
                "stratum": stratum,
                "prompt": task["prompt"],
                "image_paths": image_paths,
                "ref_names": ref_names,
                "img_path": out_path(SAVE_PATH, tid, variant),
                "replay_of": None,
            })
            seq += 1

    # 用同一个 rng 打乱 60 个 item 的展示顺序,标注者才看不出"同任务两变体相邻出现"的规律
    # ——那本身就是能反推变体身份的信号,和 blind_eval 里"kind 不能下发"是同一类问题。
    rng.shuffle(items)

    # ---- 重放的摆放:源取自前半段,重放插进后半段 ----
    # WHY 不能像最初写的那样"洗牌后 append 到列表尾部":那样最后 6 条**全部**是重放,
    # 标注者一进入尾段就会发现"这些我都见过",于是转去回忆上次答案而不是重新判读。
    # 自洽率会被抬高成一个假数——而单标注者条件下,自洽率是我们唯一的质量控制手段,
    # 抬高它等于把这个手段废掉。`build_pairs.py:366` 的既有纪律是**全部条目一起打散**,
    # 这里对齐同一条纪律,并再加一条它没有的约束:
    # 源必须落在前半段、重放必须落在后半段 ⇒ 间隔恒 ≥ len/2,不会出现"刚看过就重问"。
    # (旧写法实测出过间隔仅 2 的一对,见 2026-08-03 的清单。)
    n_replay = round(len(items) * REPLAY_FRAC)
    half = len(items) // 2
    replay_src = rng.sample(items[:half], n_replay)
    replay_items = []
    for src in replay_src:
        # 显式复制可变字段:`{**src}` 是浅拷贝,list 是共享引用,
        # 日后谁就地改一处会静默连坐另一处。
        replay_items.append({**src,
                             "item_id": f"IC_{seq:04d}",
                             "replay_of": src["item_id"],
                             "image_paths": list(src["image_paths"]),
                             "ref_names": list(src["ref_names"])})
        seq += 1

    tail = items[half:]
    slots = sorted(rng.sample(range(len(tail) + 1), n_replay))
    merged = items[:half]
    prev = 0
    for slot, rep in zip(slots, replay_items):
        merged.extend(tail[prev:slot])
        merged.append(rep)
        prev = slot
    merged.extend(tail[prev:])
    items = merged

    meta = {
        "spec": SPEC_NAME,
        "seed": SEED,
        "n_tasks": len(picked),
        "n_items": len(items),
        "n_replay": n_replay,
        "n_tasks_by_stratum": dict(sorted(Counter(s for s, _ in picked).items())),
    }
    return {"meta": meta, "items": items}


def check_items(manifest: dict, root: Path) -> list[str]:
    """返回问题清单(空 = 通过)。**纯谓词,不写文件**,供 --verify 与测试共用。"""
    errs: list[str] = []
    meta = manifest.get("meta", {})
    items = manifest.get("items", [])

    if meta.get("spec") != SPEC_NAME:
        errs.append(f"meta.spec 应为 {SPEC_NAME!r},实为 {meta.get('spec')!r}")
    if meta.get("seed") != SEED:
        errs.append(f"meta.seed 应为 {SEED},实为 {meta.get('seed')!r}")
    if meta.get("n_tasks") != N_S1 + N_S3:
        errs.append(f"meta.n_tasks 应为 {N_S1 + N_S3},实为 {meta.get('n_tasks')!r}")
    if meta.get("n_tasks_by_stratum") != {"S1": N_S1, "S3": N_S3}:
        errs.append(f"层配比应为 S1={N_S1}/S3={N_S3},实为 {meta.get('n_tasks_by_stratum')!r}")

    ids = [it.get("item_id") for it in items]
    if len(set(ids)) != len(ids):
        dup = [i for i, c in Counter(ids).items() if c > 1]
        errs.append(f"item_id 有重复:{dup[:5]}")

    main_items = [it for it in items if not it.get("replay_of")]
    replay_items = [it for it in items if it.get("replay_of")]
    if len(items) != 66:
        errs.append(f"item 总数应为 66,实为 {len(items)}")
    if len(main_items) != 60:
        errs.append(f"非重放 item 应为 60 个,实为 {len(main_items)}")
    if len(replay_items) != 6:
        errs.append(f"重放 item 应为 6 个,实为 {len(replay_items)}")
    if meta.get("n_items") != len(items) or meta.get("n_replay") != len(replay_items):
        errs.append("meta 里的计数与 items 列表实际长度不一致")

    by_id = {it["item_id"]: it for it in items if it.get("item_id")}
    pos = {it["item_id"]: i for i, it in enumerate(items) if it.get("item_id")}
    # 摆放不变量。**这条是判据的一部分,不是代码整洁度问题**:重放离源太近,
    # 标注者是在回忆而不是重判,自洽率就成了假数(见 build_manifest 里那段 WHY)。
    min_gap = len(items) // 3
    if replay_items and min(pos[it["item_id"]] for it in replay_items) < len(items) // 2:
        errs.append("有重放 item 落在清单前半段——源与重放必须分居前后半")
    for it in replay_items:
        src = by_id.get(it.get("replay_of"))
        if src is None:
            errs.append(f"{it.get('item_id')} 的 replay_of={it.get('replay_of')!r} 找不到对应原 item")
            continue
        gap = pos[it["item_id"]] - pos[src["item_id"]]
        if gap < min_gap:
            errs.append(f"{it['item_id']} 与其源 {src['item_id']} 只隔 {gap} 条"
                        f"(要求 ≥ {min_gap})——标注者会靠回忆作答,自洽率失真")
        for key in ("task_id", "variant", "stratum", "prompt", "image_paths",
                    "ref_names", "img_path"):
            if it.get(key) != src.get(key):
                errs.append(f"{it.get('item_id')} 与其 replay_of={src['item_id']} 的 {key} 不一致")

    missing: list[str] = []
    for it in items:
        rels = [it.get("img_path"), *it.get("image_paths", [])]
        for rel in rels:
            if not rel:
                errs.append(f"{it.get('item_id')} 缺路径字段")
                continue
            target = (root / rel).resolve()
            try:  # 清单里的路径当成不可信输入,防路径逃逸(抄 blind_eval.pairing 的纪律)
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


def verify_output() -> int:
    if not OUT_JSON.exists():
        print(f"❌ {OUT_JSON} 不存在,请先不带 --verify 跑一次以生成它", file=sys.stderr)
        return 1
    manifest = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    errs = check_items(manifest, REPO_ROOT)
    if errs:
        print("❌ idcount_items.json 自检未通过:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    m = manifest["meta"]
    print(f"✓ 校验通过:{m['n_items']} 个 item(任务 {m['n_tasks']} 条 "
          f"{m['n_tasks_by_stratum']},重放 {m['n_replay']} 条)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="生成/校验身份留存计数标注用的 item 清单")
    ap.add_argument("--verify", action="store_true",
                    help="只校验已生成的 idcount_items.json,不重新生成")
    args = ap.parse_args()

    if args.verify:
        return verify_output()

    manifest = build_manifest()
    _write_atomic(OUT_JSON, manifest)
    m = manifest["meta"]
    print(f"[INFO] 写入 {OUT_JSON}")
    print(f"[INFO] {m['n_tasks']} 条任务 {m['n_tasks_by_stratum']} → "
          f"{m['n_items']} 个 item(含重放 {m['n_replay']} 个)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
