#!/usr/bin/env python3
"""按**官方口径**重建 stage-1 训练集(`score_final >= 4.0` 全库过滤)。

WHY 要有这个文件 —— 现有底座 `log/ref_isolation/checkpoint-20000` 的三处偏离:

| | 官方 stage-1 | 我们 4090 上那次 |
|---|---|---|
| 过滤 | `uno/utils/filter_uno_1m_dataset.py` 阈值 **4** | `scripts/convert_uno_labels.py`,**连这个参数都没有** |
| 数据 | UNO-1M 全库 ∩ `score≥4` = **404,259** 对 | 磁盘只解压 split1-5,**~5 万**对未过滤(够官方标准的仅 **16,966**) |
| 步数 | `train.py:208` 默认 **100000** | 20000 |

`template/uno_instructions.py` 的 CoT 第 3 轮定义了 0–4 的 per-part 标尺,
**4 = "virtually indistinguishable"** ⇒ `≥4.0` 字面意思是每个 part 都满分。
有效满分样本 **≈24×** 的差距,就是从"没跑那一行过滤"来的。

━━ 与官方脚本的关系:判据逐字相同,只加**环境必需**的两件事 ━━
官方 `filter_uno_1m_dataset.py:45` 的判据是 `item['vlm_filter_cot'].get('score_final', 0) >= t`,
输出 schema 是 `{prompt, image_tgt_path, image_paths}`、路径加 `images/` 前缀、单向
`img_path1 → img_path2`。**这四件本文件一个字不改**。加的两件都是我们这台机器的硬约束:

1. **按文件存在性过滤** —— 官方假设约 2.0 TB 全解压。我们分批解压,漏一张就是
   dataloader 训到一半 `FileNotFoundError`(4090 那次踩过)。**这不是配方改动,
   是磁盘现实**;而"到底覆盖了多少"恰恰是判断"这次算不算官方复刻"的关键读数,
   所以按 split 逐个报覆盖率,并用 `--strict` 在覆盖不足时直接拒绝出文件。
2. **剔除异常高分** —— 实测有 1 条 `score_final = 131184.67`(标注管线的脏数据)。
   官方 `>= 4` 会把它留下。`--anomaly_max` 默认 1000 剔掉它,**这是对官方的一处
   有意偏离**,影响 1/404259 条,脚本会显式打印出来,不许在报告里省掉。

━━ 用法 ━━
    # 1) 先看现状:score 分布 + 每个 split 的磁盘覆盖率,不写文件
    python distill/build_stage1_official.py --dry_run

    # 2) 确认全量解压后正式出文件(覆盖率不到 --min_coverage 直接拒绝)
    python distill/build_stage1_official.py --strict

    # 3) 磁盘还没全,但想先拿现有数据跑通管线(**产物不算官方复刻**,文件名会带 partial)
    python distill/build_stage1_official.py --allow_partial
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABELS = os.path.join(REPO, "datasets/UNO-1M/uno_1m_total_labels.json")

# 官方 README.md:138 的命令行里写死的就是这个 4,不是我们挑的。
OFFICIAL_THRESHOLD = 4.0
# 官方脚本 :47-51 的输出 schema 与 images/ 前缀,逐字沿用。
IMG_PREFIX = "images/"
# 已核实的官方满分池规模。达不到它就说明磁盘没全,拿来当覆盖率的分母。
OFFICIAL_POOL = 404_259


def _chunk_exists(paths: list[str], workers: int, timeout: float,
                  retries: int) -> list[bool]:
    """单块:daemon 线程池跑原生 os.path.exists,超时判不存在并重试。

    用 daemon 线程而不是 ThreadPoolExecutor:worker 若在 ceph 上 D 状态永久挂死,
    daemon 线程在进程退出时不被 join,挂死只损失它自己;ThreadPoolExecutor 的
    线程非 daemon,会在进程退出时被 join 卡死(2026-08-08 实测的挂起就撞这个)。

    worker 用阻塞 get() + 哨兵退出:若用 get_nowait(),主线程「先启线程再塞任务」
    的间隙会让 worker 撞上空队列直接退出,并发度塌掉(实测 300s 跑不完,后来发现
    是这个竞态)。
    """
    n = len(paths)
    if n == 0:
        return []
    results = [False] * n
    pending = list(range(n))
    _STOP = object()
    for _ in range(retries + 1):
        if not pending:
            break
        futs = {idx: Future() for idx in pending}
        work: queue.Queue = queue.Queue()

        def worker() -> None:
            while True:
                idx = work.get()
                if idx is _STOP:
                    return
                try:
                    ok = os.path.exists(paths[idx])
                except BaseException:
                    ok = False
                futs[idx].set_result(ok)

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(workers)]
        for t in threads:
            t.start()
        for idx in pending:
            work.put(idx)
        for _ in threads:
            work.put(_STOP)
        still = []
        for idx in pending:
            try:
                results[idx] = futs[idx].result(timeout=timeout)
            except FutureTimeoutError:
                still.append(idx)
        pending = still
    return results


def robust_exists_many(paths: list[str], workers: int = 64,
                       timeout: float = 30.0, retries: int = 1,
                       chunk: int = 50_000) -> list[bool]:
    """批量文件存在性检查:线程池 + 原生 os.path.exists + 超时保护。

    os.path.exists 在 ceph 上偶发永久挂起(2026-08-08 实测,404k 次 stat 撞上就
    D 状态卡死)。子进程 `test -f` 能防挂起,但每次 fork ~7ms,808k 次 ~100 分钟;
    本方案实测 9.2k/s,808k 次 ~90 秒。

    机制:stat() 系统调用释放 GIL → 线程真正并行;某线程挂死只损失它自己,其余
    线程照常出结果。每个结果设超时,超时判为不存在并重试一次(挂起多为瞬时抖动,
    重试大概率恢复)。分块处理,控制内存。
    """
    n = len(paths)
    if n == 0:
        return []
    out: list[bool] = []
    for start in range(0, n, chunk):
        out.extend(_chunk_exists(paths[start:start + chunk], workers,
                                 timeout, retries))
    return out


def split_of(raw_path: str) -> str:
    """`split1/object365_xxx.png` → `split1`。标签里引用了 102 个 split。"""
    head = raw_path.split("/", 1)[0]
    return head if head.startswith("split") else "(其它)"


def load_labels(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"❌ 缺少 {os.path.relpath(path, REPO)}\n"
                 f"   UNO-1M 要先从 HuggingFace 下载并解压到 datasets/UNO-1M/,\n"
                 f"   见 scripts/fetch_uno1m.py")
    with open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        sys.exit(f"❌ {path} 不是非空列表")
    probe = raw[0]
    if "vlm_filter_cot" not in probe:
        sys.exit(f"❌ 标签里没有 `vlm_filter_cot` 字段——官方过滤判据依赖它。\n"
                 f"   第一条的键:{sorted(probe)}\n"
                 f"   大概率下错了版本(2025-08-18 那版才带 CoT 打分)")
    return raw


def build(raw: list[dict], image_root: str, threshold: float,
          anomaly_max: float) -> tuple[list[dict], dict]:
    """返回 (样本列表, 统计)。判据与官方 `filter_uno_1m_dataset.py:45` 逐字相同。"""
    out: list[dict] = []
    stat = Counter()
    scores: list[float] = []
    # 每个 split 的「过了 score 关」与「图也在磁盘上」两个计数,用来算覆盖率
    per_split = defaultdict(lambda: [0, 0])

    # 先收集候选,再一次批量 stat(见 robust_exists_many:808k 次 stat 从子进程
    # 逐条 fork(~100 分钟)压成线程池并发(~90 秒))
    cand: list[tuple[str, str, str, str, str, str]] = []
    for d in raw:
        vlc = d.get("vlm_filter_cot") or {}
        score = vlc.get("score_final", 0)
        if not isinstance(score, (int, float)):
            stat["score 字段非数值"] += 1
            continue
        scores.append(float(score))
        if score > anomaly_max:
            stat["异常高分剔除(对官方的偏离)"] += 1
            continue
        if score < threshold:
            stat[f"score < {threshold}"] += 1
            continue

        ref_raw, tgt_raw = d.get("img_path1"), d.get("img_path2")
        prompt = (d.get("caption") or {}).get("img_path2", "")
        if not (ref_raw and tgt_raw and prompt):
            stat["键缺失"] += 1
            continue

        sp = split_of(tgt_raw)
        per_split[sp][0] += 1

        ref_rel, tgt_rel = IMG_PREFIX + ref_raw, IMG_PREFIX + tgt_raw
        cand.append((sp, ref_rel, tgt_rel, prompt,
                     os.path.join(image_root, ref_rel),
                     os.path.join(image_root, tgt_rel)))

    ref_ok = robust_exists_many([c[4] for c in cand])
    tgt_ok = robust_exists_many([c[5] for c in cand])

    for (sp, ref_rel, tgt_rel, prompt, _, _), rok, tok in zip(cand, ref_ok, tgt_ok):
        if rok and tok:
            per_split[sp][1] += 1
            out.append({
                "prompt": prompt,
                "image_tgt_path": tgt_rel,
                "image_paths": [ref_rel],
            })
        else:
            stat["图片不在磁盘"] += 1

    return out, {"skip": stat, "scores": scores, "per_split": dict(per_split)}


def report(n_out: int, meta: dict, threshold: float) -> float:
    """打印 score 分布与逐 split 覆盖率,返回总覆盖率(相对官方满分池)。"""
    scores = meta["scores"]
    print(f"\n[score 分布] 共 {len(scores)} 条有数值的记录")
    edges = [0, 1, 2, 3, 3.5, 4, 4.5, float("inf")]
    for lo, hi in zip(edges, edges[1:]):
        c = sum(1 for s in scores if lo <= s < hi)
        bar = "█" * min(40, round(c / max(1, len(scores)) * 100))
        mark = "  ← 官方阈值在这里" if lo == threshold else ""
        print(f"  [{lo:>4}, {hi if hi != float('inf') else '∞':>4})  {c:>7}  {bar}{mark}")

    print(f"\n[跳过原因]")
    for k, v in meta["skip"].most_common():
        print(f"  {k:<28}{v:>8}")

    per = meta["per_split"]
    passed = sum(p for p, _ in per.values())
    on_disk = sum(d for _, d in per.values())
    missing = {s: (p, d) for s, (p, d) in per.items() if d < p}
    print(f"\n[磁盘覆盖率] 引用 {len(per)} 个 split;过 score 关 {passed} 条,"
          f"其中磁盘上有 {on_disk} 条")
    if missing:
        print(f"  ⚠️ {len(missing)} 个 split 不全(按缺口从大到小,只列前 12):")
        for s, (p, d) in sorted(missing.items(), key=lambda kv: kv[1][1] - kv[1][0])[:12]:
            print(f"    {s:<12} 磁盘 {d:>6} / 需要 {p:>6}  ({d / p if p else 0:.1%})")

    cov = n_out / OFFICIAL_POOL
    print(f"\n[对官方满分池] {n_out} / {OFFICIAL_POOL} = **{cov:.1%}**"
          f"   (4090 那次的可用满分样本是 16,966 = {16966 / OFFICIAL_POOL:.1%})")
    return cov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--out", default=None,
                    help="默认 datasets/UNO-1M/stage1_official_score4.json;"
                         "覆盖率不足且 --allow_partial 时自动加 _partial 后缀")
    ap.add_argument("--threshold", type=float, default=OFFICIAL_THRESHOLD,
                    help="官方 README 写死是 4,**别动它**——动了就不是复刻")
    ap.add_argument("--anomaly_max", type=float, default=1000.0,
                    help="剔除 score 高于此值的脏数据(实测 1 条 131184.67)。"
                         "这是对官方的一处有意偏离,设为 inf 可关掉")
    ap.add_argument("--min_coverage", type=float, default=0.95,
                    help="--strict 下要求达到的、相对官方 404259 满分池的覆盖率")
    ap.add_argument("--strict", action="store_true",
                    help="覆盖率不足 --min_coverage 直接退出,不出文件")
    ap.add_argument("--allow_partial", action="store_true",
                    help="磁盘没全也出文件,文件名带 _partial —— **产物不算官方复刻**")
    ap.add_argument("--dry_run", action="store_true", help="只统计,不写文件")
    args = ap.parse_args()

    if args.threshold != OFFICIAL_THRESHOLD:
        print(f"⚠️ 阈值被改成 {args.threshold}(官方是 {OFFICIAL_THRESHOLD})——"
              f"这样出来的底座**不能叫「按官方流程复刻」**", file=sys.stderr)

    image_root = os.path.dirname(os.path.abspath(args.labels))
    print(f"[build] 读取 {os.path.relpath(args.labels, REPO)} ...", flush=True)
    raw = load_labels(args.labels)
    print(f"[build] 原始 {len(raw)} 条", flush=True)

    out, meta = build(raw, image_root, args.threshold, args.anomaly_max)
    cov = report(len(out), meta, args.threshold)

    if not out:
        sys.exit("❌ 0 条样本——检查图片是否解压到 datasets/UNO-1M/images/split*/")

    full = cov >= args.min_coverage
    if args.strict and not full:
        sys.exit(f"\n❌ 覆盖率 {cov:.1%} < {args.min_coverage:.0%},--strict 拒绝出文件。\n"
                 f"   先补齐磁盘:python scripts/fetch_uno1m.py\n"
                 f"   只想跑通管线的话用 --allow_partial(产物不算官方复刻)")
    if not full and not args.allow_partial and not args.dry_run:
        sys.exit(f"\n❌ 覆盖率只有 {cov:.1%},拒绝静默出一个"
                 f"「看起来像官方复刻、其实不是」的文件。\n"
                 f"   补齐磁盘后重跑,或显式加 --allow_partial")

    if args.dry_run:
        print("\n[--dry_run] 不写文件")
        return

    out_path = args.out or os.path.join(
        image_root, f"stage1_official_score4{'' if full else '_partial'}.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    print(f"\n写入 {os.path.relpath(out_path, REPO)}:{len(out)} 条")
    if not full:
        print("⚠️ 文件名带 _partial:磁盘没全,**这个底座不能叫官方复刻**,"
              "报告里要写实际条数与覆盖率")


if __name__ == "__main__":
    sys.exit(main())
