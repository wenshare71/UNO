#!/usr/bin/env python3
"""把 UNO-1M 原始标签转换成 FluxPairedDatasetV2 期望的 schema。

原始 (uno_1m_total_labels.json) 每条:
    {"img_path1": "split1/xxx.png",
     "img_path2": "split1/yyy.png",
     "caption": {"img_path1": "...", "img_path2": "...", ...},
     ...}

目标 (uno_1m_total_labels_convert.json) 每条:
    {"image_paths": ["<ref 相对路径>"],
     "prompt": "<target 图的 caption>",
     "image_tgt_path": "<target 相对路径>"}

关键: 标签共 ~101 万条, 但磁盘可能只解压了部分 split(如 split1-5 共 ~10 万图),
默认按文件存在性过滤, 否则 dataloader 训练途中会 FileNotFoundError。

用法(在 UNO 仓库根目录):
    python scripts/convert_uno_labels.py \
        --labels datasets/UNO-1M/uno_1m_total_labels.json \
        --out datasets/UNO-1M/uno_1m_total_labels_convert.json
"""
import argparse
import json
import os
import sys


def find_rel_path(json_dir: str, raw_path: str) -> str | None:
    """返回相对 json 所在目录可解析的路径(dataset 的 image_root = dirname(json))。"""
    for rel in (raw_path, os.path.join("images", raw_path)):
        if os.path.exists(os.path.join(json_dir, rel)):
            return rel
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="datasets/UNO-1M/uno_1m_total_labels.json")
    ap.add_argument("--out", default="datasets/UNO-1M/uno_1m_total_labels_convert.json")
    ap.add_argument("--no-check-exists", action="store_true",
                    help="不按文件存在性过滤(仅当确认全部 split 已解压时使用)")
    ap.add_argument("--bidirectional", action="store_true",
                    help="每条样本双向各出一条(img1->img2 和 img2->img1), 数据量翻倍")
    args = ap.parse_args()

    json_dir = os.path.dirname(os.path.abspath(args.labels))
    print(f"[convert] 读取 {args.labels} ...", flush=True)
    with open(args.labels, "rt") as f:
        raw = json.load(f)
    print(f"[convert] 原始 {len(raw)} 条", flush=True)

    out, n_missing_key, n_missing_file = [], 0, 0
    directions = [("img_path1", "img_path2")]
    if args.bidirectional:
        directions.append(("img_path2", "img_path1"))

    for i, d in enumerate(raw):
        cap = d.get("caption") or {}
        for ref_key, tgt_key in directions:
            ref_raw, tgt_raw = d.get(ref_key), d.get(tgt_key)
            prompt = cap.get(tgt_key) or cap.get(ref_key)
            if not (ref_raw and tgt_raw and prompt):
                n_missing_key += 1
                continue
            if args.no_check_exists:
                ref_rel, tgt_rel = ref_raw, tgt_raw
            else:
                ref_rel = find_rel_path(json_dir, ref_raw)
                tgt_rel = find_rel_path(json_dir, tgt_raw)
                if ref_rel is None or tgt_rel is None:
                    n_missing_file += 1
                    continue
            out.append({
                "image_paths": [ref_rel],
                "prompt": prompt,
                "image_tgt_path": tgt_rel,
            })
        if (i + 1) % 100_000 == 0:
            print(f"[convert] 已处理 {i + 1}/{len(raw)}, 有效 {len(out)}", flush=True)

    if not out:
        print("[convert] ERROR: 0 条有效样本 —— 检查图片解压路径是否与标签里的相对路径一致\n"
              f"          (脚本会尝试 <json 目录>/<path> 和 <json 目录>/images/<path> 两种布局)",
              file=sys.stderr)
        sys.exit(1)

    print(f"[convert] 写出 {args.out}: {len(out)} 条 "
          f"(键缺失跳过 {n_missing_key}, 图片不在磁盘跳过 {n_missing_file})", flush=True)
    with open(args.out, "wt") as f:
        json.dump(out, f, ensure_ascii=False)
    print("[convert] 完成 ✅")


if __name__ == "__main__":
    main()
