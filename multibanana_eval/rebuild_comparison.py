"""不重新推理,只用已生成的单变体图 + results.json 重拼 ALL_COMPARISON.png。

修复 prompt 过长在总览图里被截断的问题:infer 旧版把 prompt 截到 70 字符且单行
渲染,长 prompt 被裁。本脚本读 output 目录里每个 <task>__<variant>.png 重拼,标题
用 board.build_row 完整多行显示。不需要 GPU,在 output/ 与参考图都在的机器上跑即可。

用法(远程,推理跑过的同一台机器):
  python multibanana_eval/rebuild_comparison.py
  python multibanana_eval/rebuild_comparison.py --result_dir output/multibanana_eval --out output/ALL_fixed.png
"""
import argparse
import json
import os
import sys

# 让 `import board` 无论从哪运行都能找到(与 board.py 同目录)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import board


def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="output/multibanana_eval",
                   help="infer_multibanana.py 的输出目录(含 results.json 与 <task>__<variant>.png)")
    p.add_argument("--out", default=None,
                   help="输出路径;默认覆盖 result_dir/ALL_COMPARISON.png")
    p.add_argument("--cell", type=int, default=256, help="每个小图的边长像素")
    args = p.parse_args()

    rj = os.path.join(args.result_dir, "results.json")
    if not os.path.exists(rj):
        raise SystemExit(f"❌ 没找到 {rj}\n   请在跑过 infer_multibanana.py 的机器上、指向它的 --save_path")
    with open(rj, "rt", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    timing = data.get("timing", {})
    variants = list(timing)  # dict 保序,即 infer 时的变体顺序
    if not tasks or not variants:
        raise SystemExit("❌ results.json 里没有 tasks 或 timing,无法重拼")

    # 每个 task 各变体的 denoise 耗时,用于小图标签(和原图一致)
    times_by_task: dict[str, dict] = {}
    for v, t in timing.items():
        for pt in t.get("per_task", []):
            times_by_task.setdefault(pt["task"], {})[v] = pt.get("denoise_s")

    rows, missing_refs = [], 0
    for task in tasks:
        name = task["name"]
        # 参考图:远程有 data/ 时能读到;缺失就用灰占位(不因缺图整张拼不出来)
        refs = []
        for pth in task["image_paths"]:
            if os.path.exists(pth):
                refs.append(_open_rgb(pth))
            else:
                refs.append(Image.new("RGB", (args.cell, args.cell), (230, 230, 230)))
                missing_refs += 1
        # 各变体生成图
        results = {}
        for v in variants:
            fp = os.path.join(args.result_dir, f"{name}__{v}.png")
            if os.path.exists(fp):
                results[v] = _open_rgb(fp)
        if not results:
            print(f"⚠️  {name}: 没找到任何 <task>__<variant>.png,跳过")
            continue
        rows.append(board.build_row(name, task["prompt"], refs, results,
                                    times_by_task.get(name), cell=args.cell))

    if not rows:
        raise SystemExit("❌ 一行都没拼出来(单变体图缺失?检查 --result_dir)")

    out = args.out or os.path.join(args.result_dir, "ALL_COMPARISON.png")
    board.stack_board(rows).save(out)
    print(f"✅ 已重拼(prompt 完整换行):{out}  共 {len(rows)} 行")
    if missing_refs:
        print(f"⚠️  {missing_refs} 张参考图本地缺失,用灰占位(在有 data/ 的机器上跑可完整显示)")


if __name__ == "__main__":
    main()
