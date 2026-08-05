#!/usr/bin/env python3
"""补齐 UNO-1M 的 split 分片(下载 → 解压 → 校验),让 stage-1 能按官方口径重训。

WHY:4090 那次的底座只解压了 split1-5,`score≥4` 池因此只有 **16,966** 条,
而官方是全库 **404,259** 条 —— 差 24×。`distill/build_stage1_official.py` 会把
覆盖率算出来,但补齐磁盘这件事本身得有个能断点续传的东西来干:118 GB、102 个分片、
HF 只能走海外代理(实测单连接 0.33 MB/s,必须开 `HF_HUB_ENABLE_HF_TRANSFER`),
一次拉不完是常态。

**幂等**:每次运行只处理"还没解压好"的分片,已完成的直接跳过。中途断了重跑即可。

━━ 用法(H800,仓库根目录) ━━
    export http_proxy=http://oversea-squid1.jp.txyun:11080
    export https_proxy=http://oversea-squid1.jp.txyun:11080
    export HF_HUB_ENABLE_HF_TRANSFER=1
    unset HF_HUB_OFFLINE          # 平时训练是 offline=1,下载时必须解掉

    python scripts/fetch_uno1m.py --dry_run     # 先看要拉哪些、多大、盘够不够
    python scripts/fetch_uno1m.py --rm_tar      # 正式拉;解压成功后删 tar 省盘

后台跑(会跑很久,别用 nohup —— 见 DISTILL_PLAN §11.12(a) 的 SIGHUP 教训):
    setsid python scripts/fetch_uno1m.py --rm_tar > logs/fetch_uno1m.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ID = "bytedance-research/UNO-1M"
DEFAULT_DIR = os.path.join(REPO, "datasets/UNO-1M")
LABELS = "uno_1m_total_labels.json"
# 解压后至少要有这么多张图才认为这个 split 是好的。分片实测都在万张量级,
# 用一个很松的下界只为挡住"解压到一半被 Ctrl-C"留下的半截目录。
MIN_FILES_PER_SPLIT = 100


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def split_name(remote: str) -> str | None:
    """`images/split7.tar.gz` → `split7`;非分片文件返回 None。"""
    base = os.path.basename(remote)
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return None


def already_done(images_dir: str, name: str) -> bool:
    d = os.path.join(images_dir, name)
    if not os.path.isdir(d):
        return False
    # 只数到阈值就停,102 个 split 全量 listdir 太慢
    n = 0
    with os.scandir(d) as it:
        for _ in it:
            n += 1
            if n >= MIN_FILES_PER_SPLIT:
                return True
    return False


def safe_extract(tar_path: str, images_dir: str, name: str) -> int:
    """解压到 `<images_dir>/<name>.part` 再原子改名 —— 中途断了不会留下"看着完整
    其实缺图"的目录,而那种目录会让下一次运行误判为已完成、最终训练时才 FileNotFound。
    """
    final = os.path.join(images_dir, name)
    staging = final + ".part"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    n = 0
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            # 归档里的路径不可信:`../` 和绝对路径能写到目录外面去
            target = os.path.realpath(os.path.join(staging, m.name))
            if not target.startswith(os.path.realpath(staging) + os.sep):
                raise RuntimeError(f"{tar_path} 里有越界路径 {m.name},拒绝解压")
            tf.extract(m, staging)
            if m.isfile():
                n += 1

    # tar 里通常自带一层 `split7/`,有就把它提上来,保证最终是 images/split7/*.png
    inner = os.path.join(staging, name)
    if os.path.isdir(inner):
        os.replace(inner, final + ".tmp")
        shutil.rmtree(staging)
        os.replace(final + ".tmp", final)
    else:
        os.replace(staging, final)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--only", nargs="*", default=None,
                    help="只处理这些 split(如 split6 split7);默认处理全部缺的")
    ap.add_argument("--limit", type=int, default=None, help="本次最多处理几个分片")
    ap.add_argument("--rm_tar", action="store_true",
                    help="解压成功后删掉 tar(118 GB 的盘省一半)")
    ap.add_argument("--min_free_gb", type=float, default=40.0,
                    help="剩余空间低于此值就停下,不要把盘写满")
    ap.add_argument("--dry_run", action="store_true", help="只列计划,不下载")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("❌ 缺 huggingface_hub:pip install 'huggingface_hub[cli]' hf_transfer")

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        sys.exit("❌ HF_HUB_OFFLINE=1,下载不可能成功。先 `unset HF_HUB_OFFLINE`")

    images_dir = os.path.join(args.dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[fetch] 列出 {DATASET_ID} 的文件 ...", flush=True)
    api = HfApi()
    infos = {f.rfilename: f for f in
             api.repo_info(DATASET_ID, repo_type="dataset", files_metadata=True).siblings}

    if LABELS not in {os.path.basename(k) for k in infos}:
        print(f"⚠️ 仓库里没找到 {LABELS},确认 repo 版本", file=sys.stderr)
    label_remote = next((k for k in infos if os.path.basename(k) == LABELS), None)
    if label_remote and not os.path.exists(os.path.join(args.dir, LABELS)):
        print(f"[fetch] 先拿标签 {label_remote}", flush=True)
        if not args.dry_run:
            hf_hub_download(DATASET_ID, label_remote, repo_type="dataset",
                            local_dir=args.dir)

    shards = {}
    for remote in infos:
        name = split_name(remote)
        if name and name.startswith("split"):
            shards[name] = remote
    if not shards:
        sys.exit("❌ 仓库里一个 split 分片都没列到——文件命名可能变了,先手动 ls 一下")

    todo = sorted((n for n in shards if not already_done(images_dir, n)),
                  key=lambda s: int(s[len("split"):]) if s[len("split"):].isdigit() else 0)
    if args.only:
        todo = [n for n in todo if n in set(args.only)]
    done_n = len(shards) - len([n for n in shards if not already_done(images_dir, n)])
    total_bytes = sum(infos[shards[n]].size or 0 for n in todo)

    print(f"\n[fetch] 分片 {len(shards)} 个,已解压 {done_n} 个,待处理 {len(todo)} 个"
          f"(下载量约 {human(total_bytes)})")
    if args.limit:
        todo = todo[: args.limit]
        print(f"        --limit {args.limit} ⇒ 本次只处理 {todo}")
    if not todo:
        print("✅ 全部分片都已就位。下一步:python distill/build_stage1_official.py --strict")
        return

    free = shutil.disk_usage(args.dir).free
    print(f"[fetch] 目标盘剩余 {human(free)};解压后总占用约 "
          f"{human(total_bytes * (1 if args.rm_tar else 2))}")
    if args.dry_run:
        print("\n[--dry_run] 计划:", " ".join(todo[:20]), "..." if len(todo) > 20 else "")
        return

    ok = fail = 0
    for i, name in enumerate(todo, 1):
        free = shutil.disk_usage(args.dir).free
        if free < args.min_free_gb * 1024 ** 3:
            print(f"\n⛔ 剩余空间 {human(free)} < {args.min_free_gb}GB,停止。"
                  f"腾出空间后重跑即可(已完成的会跳过)")
            break
        remote = shards[name]
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {name}  ({human(infos[remote].size or 0)})", flush=True)
        try:
            tar_path = hf_hub_download(DATASET_ID, remote, repo_type="dataset",
                                       local_dir=args.dir)
            n_files = safe_extract(tar_path, images_dir, name)
            if args.rm_tar:
                os.remove(tar_path)
            ok += 1
            print(f"      ✅ {n_files} 张,用时 {time.time() - t0:.0f}s", flush=True)
        except KeyboardInterrupt:
            print("\n中断——已完成的分片不受影响,重跑会接着来")
            break
        except Exception as e:
            fail += 1
            # 网络/磁盘出错是常态,不要因为一个分片挂掉就丢掉整轮进度
            print(f"      ❌ {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    print(f"\n[fetch] 本轮成功 {ok} / 失败 {fail}")
    print("下一步:python distill/build_stage1_official.py --dry_run   # 看覆盖率")


if __name__ == "__main__":
    sys.exit(main())
