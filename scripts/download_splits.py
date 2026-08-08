#!/usr/bin/env python3
"""多线程 range 分块下载 UNO-1M split,走指定国内代理 + hf-mirror。

WHY:主下载器 fetch_uno1m.py 用 hf_hub_download(hf_transfer)。实测:
  - squid(oversea-squid1)带宽被共享租户吃光,只有 ~0-7 MB/s;
  - hf_transfer 走国内代理会卡 D 状态(uninterruptible sleep,分块协议不兼容);
  - curl 走国内代理 + hf-mirror 稳定,4 代理 x 4 连接聚合 ~28 MB/s。
本脚本用「N 个 curl range 分块并发下载 + 拼接」复刻 curl 聚合机制,
每进程走一个代理,多进程(不同代理)并行即可叠加带宽。

用法(每个进程一个代理一个分片区间):
  export HF_ENDPOINT=https://hf-mirror.com
  setsid python scripts/download_splits.py --splits split12 split13 ... \
    --proxy 10.66.29.113:11080 --threads 8 > logs/dl_p1.log 2>&1 < /dev/null &

断点续传:
  - 已完成解压的 split 自动跳过(already_done)
  - 下载一半被杀:tar.gz.partN 残留会被下次启动清理后重下;
    已拼好的 tar.gz 且大小正确则跳过下载直接解压
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fetch_uno1m import already_done, safe_extract, human  # noqa: E402

DEFAULT_DIR = os.path.join(REPO, "datasets/UNO-1M")
# 默认走 hf-mirror(国内可达);可被 HF_ENDPOINT 覆盖回 HF 官方
BASE = (os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        + "/datasets/bytedance-research/UNO-1M/resolve/main/images/")


def _signed_url(url: str, proxy: str) -> tuple[str, int]:
    """GET hf-mirror,拿 302 的签名 URL + X-Linked-Size(真实大小)。

    2026-08-07 晚:hf-mirror 缓存过期后会把 UNO-1M 统一 302 到 Xet bridge
    (us.aws.cdn.hf.co),签名 URL 有效期 1 小时。curl 取 302 用 --http1.1
    (hf-mirror 侧),但**签名 URL 本体必须用 HTTP/2**(--http1.1 会 0 字节断)。
    返回 (签名URL, size)。若未 302(旧缓存路径直接给数据)size 取 Content-Range。
    """
    r = subprocess.run(
        ["curl", "-s", "-D", "-", "-o", "/dev/null", "--http1.1",
         "--proxy", f"http://{proxy}", url],
        capture_output=True, text=True, timeout=60)
    loc = re.search(r"(?im)^Location:\s*(\S+)", r.stdout)
    if loc:
        size_m = re.search(r"(?im)^X-Linked-Size:\s*(\d+)", r.stdout)
        if size_m:
            return loc.group(1).strip(), int(size_m.group(1))
        # 有 Location 但没 X-Linked-Size:罕见,回退 HEAD
        return loc.group(1).strip(), get_size(url, proxy)
    # 没 302:缓存直出。Content-Range: bytes 0-1023/848845406
    cr = re.search(r"(?im)^Content-Range:\s*bytes\s+\d+-\d+/(\d+)", r.stdout)
    if cr:
        return url, int(cr.group(1))
    raise RuntimeError(f"拿不到签名 URL/大小:{r.stdout[-300:]}")


def get_size(url: str, proxy: str) -> int:
    proxy_url = f"http://{proxy}"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "curl/8"})
    with opener.open(req, timeout=30) as r:
        n = int(r.headers.get("Content-Length", 0))
    if n <= 0:
        raise RuntimeError(f"HEAD {url} 拿不到 Content-Length")
    return n


def curl_range(url: str, proxy: str, start: int, end: int, out: str) -> None:
    # 签名 URL(us.aws.cdn.hf.co)必须 HTTP/2:--http1.1 会 206/0 字节断(exit 18)。
    # 去掉 --http1.1 让 curl 走默认的 HTTP/2。hf-mirror 直出路径(未 302)走 HTTP/1.1 也兼容。
    subprocess.run(
        ["curl", "-sL", "--proxy", f"http://{proxy}",
         "-r", f"{start}-{end}", "-o", out, url],
        check=True, timeout=1800)


def download_split(name: str, proxy: str, images_dir: str,
                   threads: int, rm_tar: bool) -> tuple[int, float, int]:
    """下载 + 解压一个 split。返回 (文件数, 下载耗时s, 字节数)。"""
    url = BASE + f"{name}.tar.gz"
    tar_path = os.path.join(images_dir, name + ".tar.gz")

    # 每个 split 取一次签名 URL(1 小时有效,够整片);未 302 时直接返回原 URL
    signed, size = _signed_url(url, proxy)
    sig_lock = threading.Lock()

    def refresh_sig() -> str:
        nonlocal signed
        new_sig, new_size = _signed_url(url, proxy)
        if new_size != size:
            raise RuntimeError(f"签名刷新后大小 {new_size} != {size}")
        signed = new_sig
        return signed

    # 已拼好的 tar 且大小对 → 跳过下载直接解压
    if os.path.exists(tar_path) and os.path.getsize(tar_path) == size:
        t0 = time.time()
        n = safe_extract(tar_path, images_dir, name)
        if rm_tar:
            os.remove(tar_path)
        return n, time.time() - t0, 0  # 0 字节下载

    # 代理会把 >1GB 的 range 截断(实测 2.3GB range 只回 1GB),用 256MB 小块规避
    CHUNK = 256 * 1024 * 1024
    nblocks = (size + CHUNK - 1) // CHUNK
    # 清理残留 part
    for i in range(nblocks):
        p = f"{tar_path}.part{i}"
        if os.path.exists(p):
            os.remove(p)

    t0 = time.time()
    print(f"[{name}] {human(size)} 分 {nblocks} 块(各 {CHUNK//1048576}MB,并发 {threads})",
          flush=True)

    def one(i: int) -> None:
        s = i * CHUNK
        e = min((i + 1) * CHUNK - 1, size - 1)
        expect = e - s + 1
        out = f"{tar_path}.part{i}"
        for attempt in range(2):  # 签名过期(403)→ 刷新重试一次
            try:
                with sig_lock:
                    cur = signed
                curl_range(cur, proxy, s, e, out)
                break
            except subprocess.CalledProcessError:
                if attempt == 0:
                    with sig_lock:
                        refresh_sig()
                    continue
                raise
        got = os.path.getsize(out)
        if got != expect:
            raise RuntimeError(
                f"块 {i} 期望 {expect}B 实际 {got}B(range 被截断,删 {tar_path} 重跑)")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, range(nblocks)))

    # 拼接并校验
    got = 0
    with open(tar_path, "wb") as out:
        for i in range(nblocks):
            p = f"{tar_path}.part{i}"
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            got += os.path.getsize(p)
            os.remove(p)
    if got != size:
        raise RuntimeError(f"{name} 拼完 {got} != 预期 {size},删掉重下")
    t_dl = time.time() - t0

    n = safe_extract(tar_path, images_dir, name)
    if rm_tar:
        os.remove(tar_path)
    return n, t_dl, size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="*", required=True)
    ap.add_argument("--proxy", required=True,
                    help="如 10.66.29.113:11080(不写 http://)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--no-rm-tar", action="store_true")
    args = ap.parse_args()

    images_dir = os.path.join(args.dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"[download] 数据目录: {images_dir}\n[download] 代理: {args.proxy} "
          f"线程: {args.threads}\n[download] 源: {BASE}", flush=True)

    dl_bytes = dl_sec = ok = skip = fail = 0
    for name in args.splits:
        if already_done(images_dir, name):
            print(f"[{name}] 已解压,跳过", flush=True)
            skip += 1
            continue
        try:
            n, t_dl, size = download_split(name, args.proxy, images_dir,
                                           args.threads, not args.no_rm_tar)
            ok += 1
            dl_bytes += size
            dl_sec += t_dl
            print(f"[{name}] ✅ {n} 张,下载 {t_dl:.0f}s"
                  f"({size / max(t_dl, 1e-9) / 1048576:.2f} MB/s),完成",
                  flush=True)
        except Exception as e:
            fail += 1
            print(f"[{name}] ❌ {type(e).__name__}: {e}", file=sys.stderr,
                  flush=True)

    print(f"\n[download] 本轮: 成功 {ok} / 跳过 {skip} / 失败 {fail}", flush=True)
    if dl_sec > 0 and dl_bytes > 0:
        print(f"[download] 实测 {dl_bytes / dl_sec / 1048576:.2f} MB/s",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
