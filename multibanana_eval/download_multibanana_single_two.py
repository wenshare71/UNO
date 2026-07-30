"""从 HuggingFace 下载 MultiBanana 的 single-ref + 2-ref 全量数据。

不依赖 huggingface_hub(当前环境没装),直接用 urllib + HF web API 列目录、
并发下载单个文件。结果落在 data/multibanana/<task_dir>/ 下,和
infer_multibanana.py 的 --data_dir 默认约定保持一致。

目标 task 目录(共 12 个,2691 文件,~2.5 GB):
  single  (1 ref, 264 tasks)
  add / replace / background / color / material / pose / hair /
  makeup / tone / style / text  (2 refs, 717 tasks total)

每个目录会拿到:
  {NNN}_{i}.jpg    参考图
  {NNN}_prompt.txt 文本指令
  types.json       难度标签(domain/scale/rare/ling)
根目录额外下 from_where.csv(ref 图来源 real/generated 元数据)。

幂等:已存在且大小匹配的文件会跳过,可中断重跑。
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ID = "kohsei/MultiBanana-Benchmark"
API_BASE = "https://huggingface.co/api/datasets"
DL_BASE = "https://huggingface.co/datasets/kohsei/MultiBanana-Benchmark/resolve/main"
UA = "multibanana-dl/1.0"

# 1 single-ref + 11 two-ref task dirs (paper §3.1)
DEFAULT_DIRS = [
    "single",
    "add", "replace", "background", "color", "material",
    "pose", "hair", "makeup", "tone", "style", "text",
]


def list_dir(d):
    """通过 HF API 列出 dataset 子目录下的文件(含 size)。带重试。"""
    url = f"{API_BASE}/{REPO_ID}/tree/main/{d}?recursive=false"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    return []


def download_one(repo_path, local_root, expected_size):
    """下单个文件。已存在且大小匹配则跳过。3 次重试 + 退避。"""
    local_path = os.path.join(local_root, repo_path)
    if os.path.exists(local_path):
        cur = os.path.getsize(local_path)
        if expected_size is None or cur == expected_size:
            return ("skip", repo_path, cur)
        # 大小不匹配:删掉重下
        os.remove(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"{DL_BASE}/{repo_path}"
    tmp = local_path + ".part"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, local_path)
            return ("ok", repo_path, os.path.getsize(local_path))
        except Exception as e:
            if attempt == 3:
                return ("err", repo_path, f"{type(e).__name__}: {e}")
            time.sleep(1 + 2 * attempt)
    return ("err", repo_path, "exhausted")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/multibanana",
                   help="本地保存根目录(推理脚本默认从这里读)")
    p.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                   help="要下载的 task 目录(默认 single + 11 个 2-ref 目录)")
    p.add_argument("--workers", type=int, default=16,
                   help="并发下载数")
    p.add_argument("--no_from_where", action="store_true",
                   help="不下根目录 from_where.csv")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) 列文件清单
    all_files = []  # (repo_path, size)
    per_dir = {}
    for d in args.dirs:
        print(f"Listing {d}/ ...", flush=True)
        items = list_dir(d)
        cnt = 0
        for it in items:
            if it.get("type") == "file":
                all_files.append((it["path"], it.get("size")))
                cnt += 1
        per_dir[d] = cnt
        print(f"  {d}/: {cnt} files", flush=True)

    if not args.no_from_where:
        all_files.append(("from_where.csv", None))

    total = len(all_files)
    print(f"\nTotal files to fetch: {total}")
    print(f"Output root: {args.out}/")
    print(f"Workers: {args.workers}\n", flush=True)

    # 2) 并发下载
    ok = skip = err = 0
    err_files = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, rp, args.out, sz): rp
                   for rp, sz in all_files}
        n = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            status, rp, info = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                err_files.append((rp, info))
            if i % 100 == 0 or i == n:
                dt = time.time() - t0
                print(f"  [{i}/{n}] ok={ok} skip={skip} err={err} "
                      f"({dt:.0f}s, {i/dt:.1f} f/s)", flush=True)

    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s: ok={ok} skip={skip} err={err}")
    print(f"Per-dir counts: {per_dir}")
    if err_files:
        print("\nFailed files (first 30):")
        for rp, info in err_files[:30]:
            print(f"  {rp}: {info}")
        sys.exit(1)


if __name__ == "__main__":
    main()
