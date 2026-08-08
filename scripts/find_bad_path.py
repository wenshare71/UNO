#!/usr/bin/env python3
"""定位「stat 会卡死」的坏路径。

WHY:build_stage1_official.py 在 os.path.exists 上确定性卡死(三次都在前 20000 条内),
手动 stat 任何文件都是 0ms —— 说明 labels 里某条引用指向一个 ceph 上 stat 会挂起的文件。

方法:逐条 score>=4,每条用 `bash -c 'test -f PATH'` 子进程 + 3s 超时跑。
- 返回快 = 正常;
- 超时 = 嫌疑坏路径,打印出来停下;
- 也统计总数,可选 --all 跑完。
"""
import argparse, json, os, subprocess, sys, time

def stat_with_timeout(path: str, timeout: float = 3.0) -> bool | None:
    """None = 超时(卡死), True/False = 正常。"""
    try:
        r = subprocess.run(["test", "-f", path], timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  异常 {type(e).__name__}: {e}", file=sys.stderr)
        return None

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="datasets/UNO-1M/uno_1m_total_labels.json")
    ap.add_argument("--image-root", default="datasets/UNO-1M")
    ap.add_argument("--limit", type=int, default=None, help="只测前 N 条 score>=4(定位用)")
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args()

    with open(args.labels) as f:
        raw = json.load(f)
    print(f"[probe] 原始 {len(raw)} 条", flush=True)

    t0 = time.time()
    checked = stuck = 0
    for d in raw:
        vlc = d.get("vlm_filter_cot") or {}
        s = vlc.get("score_final", 0)
        if not isinstance(s, (int, float)) or s < 4:
            continue
        if args.limit and checked >= args.limit:
            break
        ref, tgt = d.get("img_path1"), d.get("img_path2")
        if not (ref and tgt):
            continue
        rp = os.path.join(args.image_root, "images", ref)
        tp = os.path.join(args.image_root, "images", tgt)
        for label, p in (("ref", rp), ("tgt", tp)):
            st = stat_with_timeout(p, args.timeout)
            checked += 1
            if st is None:
                stuck += 1
                print(f"❌ 卡死路径 [{label}] {p}", flush=True)
                print(f"   原始记录 img_path1={ref!r} img_path2={tgt!r}", flush=True)
                sys.exit(1)
        if checked % 5000 == 0:
            print(f"[probe] checked {checked} {time.time()-t0:.0f}s", flush=True)

    print(f"[probe] 完成: checked {checked} 卡死 {stuck} {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
