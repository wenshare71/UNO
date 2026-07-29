#!/usr/bin/env python3
"""M3 前置:构建混合训练 JSON。

把两路数据合并成 ``FluxPairedDatasetV2`` 期望的 schema,输出到
``datasets/distill_multiref/train_mixed.json``:

1. 单 ref 侧(任务类型 = 1 张 ref,按 ref 数分类不按来源):
   - UNO-1M 原始标签 (``datasets/UNO-1M/uno_1m_total_labels.json``)
     按 ``vlm_filter_cot.score_final >= 4.0`` 过滤(对齐论文只用满分数据的配方),
     剔除异常高分(>1000),按文件存在性过滤(避免训练途中 ``FileNotFoundError``),
     每条产出 ``{image_paths:[ref], prompt:tgt_caption, image_tgt_path:tgt}``。
     [已验证] score_final 不在顶层,在 ``vlm_filter_cot.score_final`` 里;
              ≥4.0 共 404,259 条,1 条异常值 131184.67 需剔除。
   - 蒸馏 1-ref(M2 manifest_filtered 里 n_refs=1 的 900 条)**归入单 ref 池**,
     不参与多 ref 权重 oversample。理由:它只有 1 张 ref,任务类型与 UNO-1M 单 ref
     相同(都不教"多主体不丢"),但用的是 dreambooth 主体(与 2-ref/3-ref 同域),
     有 in-domain 桥接价值——让模型先学会"单个 dreambooth 主体入画",再学
     "两个主体同框"。900 条对 16966 条只占 5%,不 oversample 不会被淹没。

2. 多 ref 侧(任务类型 = ≥2 张 ref,真正教"多主体不丢"的数据):
   - 蒸馏 2-ref(M2 manifest_filtered 里 n_refs=2 的 2041 条)
   - 蒸馏 3-ref(M2 manifest_filtered 里 n_refs=3 的 138 条)
   - 只有这部分真正教"多主体不丢",oversample 到总流的 40%。

3. 混合比例:有效样本流里**真·多 ref(≥2 张)占 40%**(``--multi_ratio`` 控制,默认 0.4)。
   多 ref 内部按 ``--mix``(默认 ``"90:10"`` 对应 2-ref:3-ref)分配权重。
   2-ref 为主,3-ref 降权保留(§4.2 决策:138 条 3-ref pass 率仅 3.5%,
   不追加生成,但保留降权以防 2-ref 淹没)。
   单 ref 池(UNO-1M + 蒸馏 1-ref)不重复,占 60%;多 ref 靠重复 N 遍补量。

4. 路径重算:输出 json 落在 ``datasets/distill_multiref/``,
   ``FluxPairedDatasetV2`` 的 ``image_root = dirname(json_file)`` 即此目录。
   单 ref 原始路径 ``images/split1/xxx.png`` → ``../UNO-1M/images/split1/xxx.png``;
   多 ref 路径(``../dreambooth/...``、``images/xxx.jpg``)保持不变
   (已是相对此目录,经 ``relpath`` 归一化后等价)。

5. 确定性:固定 seed,``random.Random(seed)`` 全流程可复现。

用法(在 UNO 仓库根目录,激活 ``.venv-uno`` 后):

    python distill/build_train_json.py --dry_run    # 只统计不写盘
    python distill/build_train_json.py              # 正式写出

    # 自定义混比与输出:
    python distill/build_train_json.py --multi_ratio 0.4 --mix 90:10 \
        --out datasets/distill_multiref/train_mixed.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter


# ---------- 路径处理 ----------

def find_rel_path(src_dir: str, raw_path: str, dst_dir: str) -> str | None:
    """在 ``src_dir`` 下找 ``raw_path``(尝试原值与 ``images/<raw>`` 两种布局),
    命中后返回相对 ``dst_dir`` 的路径;找不到返回 ``None``。

    与 ``scripts/convert_uno_labels.py:29`` 的 ``find_rel_path`` 同逻辑,
    只是输出改成相对 ``dst_dir``(输出 json 所在目录)。
    """
    for rel in (raw_path, os.path.join("images", raw_path)):
        abs_cand = os.path.normpath(os.path.join(src_dir, rel))
        if os.path.isfile(abs_cand):
            return os.path.relpath(abs_cand, dst_dir)
    return None


def rebase_rel_path(path: str, src_dir: str, dst_dir: str) -> str:
    """把一个相对 ``src_dir`` 的路径改写成相对 ``dst_dir``。

    用于多 ref 侧:manifest 里的路径相对 ``datasets/distill_multiref/``,
    若输出也在同目录则等价不变;若不同则自动重算。
    """
    abs_path = os.path.normpath(os.path.join(src_dir, path))
    return os.path.relpath(abs_path, dst_dir)


# ---------- 单 ref 转换(UNO-1M) ----------

def build_single_ref_pool(labels_path: str, dst_dir: str,
                          score_threshold: float,
                          anomaly_max: float = 1000.0) -> list[dict]:
    """从 UNO-1M 原始标签生成单 ref 样本池。

    - 按 ``vlm_filter_cot.score_final >= score_threshold`` 过滤;
    - 剔除 ``score_final > anomaly_max`` 的异常高分(实测 1 条 131184.67);
    - 按文件存在性过滤(ref 与 target 都要在磁盘上);
    - 单向(``img_path1 → img_path2``),与论文配方一致,不双向扩倍;
    - 路径重算为相对 ``dst_dir``。
    """
    src_dir = os.path.dirname(os.path.abspath(labels_path))
    with open(labels_path, "rt") as f:
        raw = json.load(f)

    out: list[dict] = []
    n_skip_key = n_skip_score = n_skip_anomaly = n_skip_file = 0
    for d in raw:
        vlc = d.get("vlm_filter_cot") or {}
        score = vlc.get("score_final")
        if score is None:
            n_skip_key += 1
            continue
        if not isinstance(score, (int, float)):
            n_skip_key += 1
            continue
        if score > anomaly_max:
            n_skip_anomaly += 1
            continue
        if score < score_threshold:
            n_skip_score += 1
            continue

        ref_raw, tgt_raw = d.get("img_path1"), d.get("img_path2")
        cap = d.get("caption") or {}
        prompt = cap.get("img_path2") or cap.get("img_path1")
        if not (ref_raw and tgt_raw and prompt):
            n_skip_key += 1
            continue

        ref_rel = find_rel_path(src_dir, ref_raw, dst_dir)
        tgt_rel = find_rel_path(src_dir, tgt_raw, dst_dir)
        if ref_rel is None or tgt_rel is None:
            n_skip_file += 1
            continue

        out.append({
            "image_paths": [ref_rel],
            "prompt": prompt,
            "image_tgt_path": tgt_rel,
            "meta": {
                "source": "uno_1m",
                "score_final": score,
                "n_refs": 1,
            },
        })

    print(f"[build] UNO-1M 单 ref 池:{len(out)} 条 "
          f"(score<{score_threshold} 跳过 {n_skip_score}, "
          f"异常高分跳过 {n_skip_anomaly}, 键缺失跳过 {n_skip_key}, "
          f"图片不在磁盘跳过 {n_skip_file})", flush=True)
    return out


# ---------- 多 ref 加载(M2 manifest) ----------

def load_multi_ref(manifest_path: str, dst_dir: str) -> dict[int, list[dict]]:
    """加载 M2 过滤后的多 ref manifest,按 ``n_refs`` 分组。

    路径重算为相对 ``dst_dir``(默认与 manifest 同目录,等价不变)。
    返回 ``{1: [...], 2: [...], 3: [...]}``。
    注意:返回的 1-ref 组会被 main 归入单 ref 池(任务类型相同),不参与多 ref 权重。
    """
    src_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "rt") as f:
        raw = json.load(f)

    groups: dict[int, list[dict]] = {}
    n_missing = 0
    for d in raw:
        paths = d.get("image_paths") or [d["image_path"]]
        n = len(paths)
        tgt = d.get("image_tgt_path")
        # 文件存在性兜底检查
        ok = True
        for p in paths:
            if not os.path.isfile(os.path.normpath(os.path.join(src_dir, p))):
                n_missing += 1
                ok = False
                break
        if tgt and not os.path.isfile(os.path.normpath(os.path.join(src_dir, tgt))):
            n_missing += 1
            ok = False
        if not ok:
            continue
        # 路径重算(同目录则等价不变,跨目录则自动重算)
        rec = {
            "image_paths": [rebase_rel_path(p, src_dir, dst_dir) for p in paths],
            "prompt": d["prompt"],
            "image_tgt_path": rebase_rel_path(tgt, src_dir, dst_dir) if tgt else None,
            "meta": dict(d.get("meta") or {}),
        }
        rec["meta"].setdefault("source", "distill_multiref")
        rec["meta"].setdefault("n_refs", n)
        groups.setdefault(n, []).append(rec)

    print(f"[build] 蒸馏数据池(按 ref 数分组):"
          f"{ {k: len(v) for k, v in sorted(groups.items())} }, "
          f"缺失文件跳过 {n_missing}", flush=True)
    return groups


# ---------- oversample ----------

def oversample(records: list[dict], target: int, rng: random.Random,
               label: str) -> list[dict]:
    """把 ``records`` 重复到 ``target`` 条:整除部分纯重复,余数无放回抽补。

    这样每条至少出现 ``target // len(records)`` 次,余下随机补齐,
    既保证覆盖度又有轻微随机性;确定性由 ``rng`` 固定 seed 保证。
    """
    n = len(records)
    if n == 0:
        print(f"[build]   ⚠️ {label}: 0 条 pass,target={target} 无法 oversample,跳过",
              flush=True)
        return []
    if target <= n:
        out = rng.sample(records, target)
        print(f"[build]   {label}: pool={n} ≥ target={target},直接抽样 {target} 条(不重复)",
              flush=True)
        return out
    full_reps, remainder = divmod(target, n)
    out = records * full_reps
    if remainder > 0:
        out += rng.sample(records, remainder)
    factor = target / n
    print(f"[build]   {label}: pool={n}, target={target}, "
          f"整重复 {full_reps}× + 余补 {remainder}, "
          f"平均 {factor:.1f}x", flush=True)
    return out


# ---------- 主流程 ----------

def parse_mix(s: str) -> tuple[int, int]:
    """解析 ``"90:10"`` → ``(90, 10)``(2-ref:3-ref 权重)。

    蒸馏 1-ref 不参与 mix——它归入单 ref 池(任务类型相同,不教多主体)。
    """
    parts = [int(x) for x in s.split(":")]
    if len(parts) != 2:
        raise ValueError(f"--mix 需要 2 个冒号分隔的整数(2-ref:3-ref),得到 {s!r}")
    if sum(parts) == 0:
        raise ValueError(f"--mix 之和不能为 0,得到 {s!r}")
    return parts[0], parts[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 混合训练 JSON 构建器")
    ap.add_argument("--uno_1m_labels",
                    default="datasets/UNO-1M/uno_1m_total_labels.json",
                    help="UNO-1M 原始标签路径")
    ap.add_argument("--manifest_filtered",
                    default="datasets/distill_multiref/manifest_filtered.json",
                    help="M2 过滤后的多 ref manifest")
    ap.add_argument("--out",
                    default="datasets/distill_multiref/train_mixed.json",
                    help="输出 json 路径(决定 image_root)")
    ap.add_argument("--score_threshold", type=float, default=4.0,
                    help="单 ref 最低 score_final(论文配方 4.0)")
    ap.add_argument("--multi_ratio", type=float, default=0.4,
                    help="有效样本流里真·多 ref(≥2 张)占比(默认 0.4)")
    ap.add_argument("--mix", default="90:10",
                    help="多 ref 内部 2-ref:3-ref 权重(默认 90:10,2-ref 为主)")
    ap.add_argument("--seed", type=int, default=42,
                    help="打乱与余补采样 seed")
    ap.add_argument("--dry_run", action="store_true",
                    help="只统计不写盘")
    args = ap.parse_args()

    if not (0.0 < args.multi_ratio < 1.0):
        sys.exit(f"❌ --multi_ratio 必须在 (0, 1) 开区间,得到 {args.multi_ratio}")
    w2, w3 = parse_mix(args.mix)

    out_abs = os.path.abspath(args.out)
    dst_dir = os.path.dirname(out_abs)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"[build] === build_train_json ===", flush=True)
    print(f"[build] uno_1m_labels = {args.uno_1m_labels}", flush=True)
    print(f"[build] manifest_filtered = {args.manifest_filtered}", flush=True)
    print(f"[build] out = {args.out}  (image_root = {dst_dir})", flush=True)
    print(f"[build] score_threshold = {args.score_threshold}, "
          f"multi_ratio(真·多ref≥2) = {args.multi_ratio}, "
          f"mix = {args.mix} (2r:3r), "
          f"seed = {args.seed}, dry_run = {args.dry_run}", flush=True)

    # 1) UNO-1M 单 ref 池
    uno_single = build_single_ref_pool(args.uno_1m_labels, dst_dir,
                                        args.score_threshold)
    if len(uno_single) == 0:
        sys.exit("❌ UNO-1M 单 ref 池为空,检查 uno_1m_labels 路径与图片解压情况")

    # 2) 蒸馏数据池(按 ref 数分组)
    groups = load_multi_ref(args.manifest_filtered, dst_dir)
    distill_1 = groups.get(1, [])
    pool_2 = groups.get(2, [])
    pool_3 = groups.get(3, [])

    # 3) 单 ref 池 = UNO-1M 单 ref + 蒸馏 1-ref(in-domain 桥接,不 oversample)
    #    任务类型相同(都只有 1 张 ref),蒸馏 1-ref 用 dreambooth 主体,
    #    让模型先学会"单个 dreambooth 主体入画"再学"两个主体同框"
    single = uno_single + distill_1
    S = len(single)
    if S == 0:
        sys.exit("❌ 单 ref 池为空")

    total_multi_pass = len(pool_2) + len(pool_3)
    if total_multi_pass == 0:
        sys.exit("❌ 真·多 ref 池(2-ref+3-ref)为空,检查 manifest_filtered")

    print(f"[build] --- 单 ref 池(不 oversample)---", flush=True)
    print(f"[build]   UNO-1M 单 ref = {len(uno_single)}", flush=True)
    print(f"[build]   蒸馏 1-ref   = {len(distill_1)} (in-domain 桥接,不重复)", flush=True)
    print(f"[build]   单 ref 池 S  = {S}", flush=True)
    print(f"[build] --- 真·多 ref 池(oversample)---", flush=True)
    print(f"[build]   2-ref pool  = {len(pool_2)}", flush=True)
    print(f"[build]   3-ref pool  = {len(pool_3)}", flush=True)

    # 4) 计算目标条数:M = S * r / (1 - r);按 mix 权重分到 2/3 ref
    M = int(round(S * args.multi_ratio / (1.0 - args.multi_ratio)))
    wsum = w2 + w3
    T_2 = int(round(M * w2 / wsum))
    T_3 = int(round(M * w3 / wsum))
    print(f"[build] --- 目标条数 ---", flush=True)
    print(f"[build]   单 ref 池 S = {S}", flush=True)
    print(f"[build]   真·多 ref 目标总量 M = S * {args.multi_ratio:.2f} / "
          f"{1 - args.multi_ratio:.2f} = {M}", flush=True)
    print(f"[build]     2-ref target = {T_2} (权重 {w2}/{wsum})", flush=True)
    print(f"[build]     3-ref target = {T_3} (权重 {w3}/{wsum})", flush=True)

    # 5) oversample(顺序固定 2->3,保证 rng 状态可复现)
    rng = random.Random(args.seed)
    multi_2 = oversample(pool_2, T_2, rng, "2-ref")
    multi_3 = oversample(pool_3, T_3, rng, "3-ref")

    # 6) 合并 + 确定性打乱
    mixed = single + multi_2 + multi_3
    rng.shuffle(mixed)
    total = len(mixed)
    n_multi = len(multi_2) + len(multi_3)
    actual_multi_ratio = n_multi / total if total else 0.0

    # 7) 统计
    ref_dist = Counter(len(r["image_paths"]) for r in mixed)
    src_dist = Counter(r["meta"].get("source", "?") for r in mixed)
    print(f"[build] --- 混合结果 ---", flush=True)
    print(f"[build]   总条数 = {total}", flush=True)
    print(f"[build]   单 ref(UNO-1M + 蒸馏1ref, 不重复)= {S}", flush=True)
    print(f"[build]   真·多 ref(2-ref+3-ref, 含 oversample)= {n_multi}", flush=True)
    print(f"[build]   真·多 ref 实际占比 = {actual_multi_ratio:.4f}", flush=True)
    print(f"[build]   ref 数分布(含重复)= {dict(ref_dist)}", flush=True)
    print(f"[build]   来源分布(含重复)= {dict(src_dist)}", flush=True)
    print(f"[build]   独立 pass 样本:蒸馏1ref={len(distill_1)}, "
          f"2-ref={len(pool_2)}, 3-ref={len(pool_3)}", flush=True)

    # 8) 4000 步训练里每条样本被看到次数估算(辅助判断是否过拟合/欠拟合)
    eff_batch = 8 * 1 * 2  # 8 卡 × batch_size 1 × grad_accum 2
    total_samples_seen = 4000 * eff_batch
    epochs = total_samples_seen / total if total else 0
    print(f"[build] --- 4000 步训练覆盖估算 ---", flush=True)
    print(f"[build]   有效 batch = 8卡×1×2 = {eff_batch}", flush=True)
    print(f"[build]   4000 步消耗样本 = {total_samples_seen}", flush=True)
    print(f"[build]   数据集遍历轮数 ≈ {epochs:.2f}", flush=True)
    for label, pool_size, stream_size in [
        ("UNO-1M 单 ref", len(uno_single), len(uno_single)),
        ("蒸馏 1-ref", len(distill_1), len(distill_1)),
        ("蒸馏 2-ref", len(pool_2), len(multi_2)),
        ("蒸馏 3-ref", len(pool_3), len(multi_3)),
    ]:
        if pool_size == 0:
            continue
        seen = total_samples_seen * (stream_size / total) / pool_size
        print(f"[build]   {label}: 每条被看 ≈ {seen:.1f} 次 "
              f"(pool={pool_size}, stream={stream_size})", flush=True)

    # 9) 路径正确性抽检(前 2 条 UNO-1M + 前 2 条蒸馏多 ref)
    print(f"[build] --- 路径抽检(image_root = {dst_dir})---", flush=True)
    shown_uno = shown_distill = 0
    for r in mixed:
        src = r["meta"].get("source", "?")
        if shown_uno < 2 and src == "uno_1m":
            ref = r["image_paths"][0]
            ok = os.path.isfile(os.path.normpath(os.path.join(dst_dir, ref)))
            print(f"[build]   [UNO-1M 单 ref] ref={ref}\n"
                  f"           tgt={r['image_tgt_path']}  exists={ok}",
                  flush=True)
            shown_uno += 1
        elif shown_distill < 2 and src == "distill_multiref":
            refs = r["image_paths"]
            oks = [os.path.isfile(os.path.normpath(os.path.join(dst_dir, p)))
                   for p in refs]
            print(f"[build]   [蒸馏 {len(refs)}-ref] refs={refs}\n"
                  f"           tgt={r['image_tgt_path']}  exists={oks}",
                  flush=True)
            shown_distill += 1
        if shown_uno >= 2 and shown_distill >= 2:
            break

    # 10) 写盘或 dry_run 退出
    if args.dry_run:
        print(f"[build] --dry_run 模式,不写盘 ✅", flush=True)
        return

    # 写盘前最终一次路径全量校验,避免训练途中 FileNotFoundError
    print(f"[build] --- 写盘前全量路径校验 ---", flush=True)
    n_miss = 0
    miss_examples: list[str] = []
    for r in mixed:
        for p in r["image_paths"]:
            if not os.path.isfile(os.path.normpath(os.path.join(dst_dir, p))):
                n_miss += 1
                if len(miss_examples) < 3:
                    miss_examples.append(p)
        tgt = r.get("image_tgt_path")
        if tgt and not os.path.isfile(os.path.normpath(os.path.join(dst_dir, tgt))):
            n_miss += 1
            if len(miss_examples) < 3:
                miss_examples.append(tgt)
    if n_miss > 0:
        sys.exit(f"❌ 全量校验发现 {n_miss} 条路径文件不存在,示例: {miss_examples}\n"
                 f"   检查 image_root={dst_dir} 与路径重算逻辑")
    print(f"[build] 全量校验通过(0 条缺失)", flush=True)

    with open(out_abs, "wt") as f:
        json.dump(mixed, f, ensure_ascii=False)
    size_mb = os.path.getsize(out_abs) / (1024 * 1024)
    print(f"[build] 已写出 {args.out} ({total} 条, {size_mb:.2f} MB) ✅", flush=True)

    # 11) 回读校验
    with open(out_abs, "rt") as f:
        check = json.load(f)
    assert len(check) == total, f"回读条数不符: {len(check)} vs {total}"
    print(f"[build] 回读校验通过({len(check)} 条)", flush=True)
    print(f"[build] 训练:bash scripts/train_distill.sh", flush=True)


if __name__ == "__main__":
    main()
