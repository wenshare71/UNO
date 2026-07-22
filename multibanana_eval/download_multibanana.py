"""从 HuggingFace 只下载 MultiBanana 的一小批任务(不 git clone 全量 3769)。

MultiBanana 数据托管在 HF dataset `kohsei/MultiBanana-Benchmark`,官方给的
`git clone .../MultiBanana-Benchmark ./data` 会把整个 benchmark(3769 个任务、
上万张图)全拉下来。冒烟验证只要几十个任务,所以这里用 huggingface_hub 按文件
精确下载:先列出 repo 全部文件清单,筛出目标任务目录里的前 N 个任务,只下它们
的参考图 + prompt(+ 该目录的 types.json 难度标签)。

数据结构(HF repo 内,任务目录名 = <参考图数>_<子类型>):
  3_back/006_0.jpg 006_1.jpg 006_2.jpg 006_prompt.txt 014_...
  3_global/  3_local/  4_*/  ...  一直到 8_*
每个任务 = 若干张参考图 <num>_<i> + 一个 <num>_prompt.txt。
**没有 ground-truth 目标图**:`_generated` 后缀是留给推理脚本写模型输出的,
benchmark 靠 VLM-as-judge(judge.py)评分,不是和真值比像素。

用法(远程,联网即可,不需要 GPU):
  python multibanana_eval/download_multibanana.py
  python multibanana_eval/download_multibanana.py --task_dirs 3_back 3_local --max_per_dir 8
"""
import argparse
import os
import re
from collections import defaultdict

REPO_ID = "kohsei/MultiBanana-Benchmark"

# 任务目录名形如 "3_back" / "4_global":数字=参考图数量,后缀=子类型(back/global/local…)。
# 用 (?:^|/) 兼容两种布局:repo 根目录直接是任务目录,或外面套了一层 data/。
_TASK_DIR_RE = re.compile(r"(?:^|/)(\d+_[a-zA-Z]+)/")
# 任务编号:文件名开头的数字段,如 "006_0.jpg" / "006_prompt.txt" → "006"
_NUMBER_RE = re.compile(r"^(\d+)_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_dirs", nargs="+", default=["3_back", "3_global", "3_local"],
                   help="要下载的任务目录名(不含 data/ 前缀);默认三个 3-参考子类型,"
                        "正好匹配我们训过的多主体上限(像 dress_bag_flowers 那种 3 主体)")
    p.add_argument("--max_per_dir", type=int, default=5,
                   help="每个任务目录最多下几个任务(按编号排序取前 N),控制'只下一小批'")
    p.add_argument("--out", default="data/multibanana",
                   help="本地保存根目录;推理脚本默认从这里递归读取")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("❌ 需要 huggingface_hub:pip install huggingface_hub")

    api = HfApi()
    print(f"列出 {REPO_ID} 的文件清单 …")
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    # 分组:dirname("3_back") -> { number("006") -> [该任务的完整 repo 文件路径, …] }
    # 同时记录每个目录的 types.json(难度标签,可选但很小,顺手下)。
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    types_json: dict[str, str] = {}
    wanted = set(args.task_dirs)
    for f in all_files:
        m = _TASK_DIR_RE.search(f)
        if not m:
            continue
        dirname = m.group(1)                 # "3_back"
        if dirname not in wanted:
            continue
        base = os.path.basename(f)
        if base == "types.json":
            types_json[dirname] = f
            continue
        nm = _NUMBER_RE.match(base)
        if not nm:
            continue
        groups[dirname][nm.group(1)].append(f)

    # 逐目录取前 max_per_dir 个任务下载。hf_hub_download 会在 out 下按 repo 内相对
    # 路径落盘(保留 3_back/… 或 data/3_back/… 结构),推理脚本用 rglob 递归发现,
    # 不在乎中间是否多一层 data/。
    picked_total = 0
    for dirname in args.task_dirs:
        numbers = sorted(groups.get(dirname, {}))
        if not numbers:
            print(f"⚠️  {dirname}:repo 里没找到,跳过(检查目录名是否拼对)")
            continue
        pick = numbers[:args.max_per_dir]
        print(f"\n{dirname}:共 {len(numbers)} 个任务,取前 {len(pick)} 个 → {pick}")
        # 该目录的难度标签
        if dirname in types_json:
            hf_hub_download(REPO_ID, types_json[dirname], repo_type="dataset", local_dir=args.out)
        for number in pick:
            files = sorted(groups[dirname][number])
            for repo_path in files:
                hf_hub_download(REPO_ID, repo_path, repo_type="dataset", local_dir=args.out)
            picked_total += 1
            print(f"  ✓ {number}  ({len(files)} 个文件)")

    if picked_total == 0:
        raise SystemExit("❌ 一个任务都没下到,请检查 --task_dirs 是否是真实存在的目录名")

    print(f"\n完成:共下载 {picked_total} 个任务到 {args.out}/")
    print("接下来跑推理:")
    print(f"  python multibanana_eval/infer_multibanana.py --data_dir {args.out} \\")
    print("      --lora_path log/ref_isolation/checkpoint-7000/dit_lora.safetensors --model_type flux-dev-fp8")


if __name__ == "__main__":
    main()
