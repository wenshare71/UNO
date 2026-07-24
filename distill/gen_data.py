"""M1:用官方 UNO(full attention)当 teacher,在 dreambooth TRAIN 集上合成多主体蒸馏数据。

背景:UNO 的多参考能力来自论文里 stage-2 的合成多主体数据,而那批数据**没有开源**
(`train.py:211` 的 TODO)。我们的 ref_isolation LoRA 只喂过单 ref 数据,
所以从没拿到过"把注意力分配到多个 ref 段"的梯度信号 —— 这就是掉第二主体的根因。
这个脚本就是自己把 stage-2 补出来:官方 full-attention 权重当 teacher 生成多主体图,
再混进继续训练。完整推导见 `distill/DISTILL_PLAN.md`。

模型加载 / LoRA / 计时路径抄自 `multibanana_eval/infer_multibanana.py`(远程真跑通过的),
但 teacher 用的就是官方权重本身,所以**不做 swap_lora**——不加载任何我们的 checkpoint,
UNOPipeline(only_lora=True) 挂上的就是 HF 上的官方 dit_lora。

用法:
    # 先看任务枚举对不对(纯 CPU,不加载模型,不写任何文件)
    python distill/gen_data.py --dry_run

    # 8 卡各跑 1/8(每个进程一张卡)
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i nohup python distill/gen_data.py \
        --shard_idx $i --num_shards 8 > logs/m1_shard$i.log 2>&1 &
    done

    # 全部跑完后合并
    python distill/gen_data.py --merge
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations, product

# 必须在 import torch 之前设置才生效(与 smoke / infer_multibanana 一致)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

# ------------------------------------------------------------------ 固定切分
# DISTILL_PLAN.md §2。**这两个名单不可改**:held-out 泄漏会让 M4 的评测整个作废,
# 而且泄漏之后从结果里根本看不出来(模型见过的主体当然生成得好)。
# 所以下面有启动断言,宁可 sys.exit 也不要带着污染的数据往下跑。
HELD_OUT = [
    "backpack_dog", "bear_plushie", "berry_bowl", "can", "candle",
    "clock", "colorful_sneaker", "duck_toy", "fancy_boot", "grey_sloth_plushie",
]
TRAIN = [
    "backpack", "cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8",
    "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
    "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie",
]

# DreamBench 的 live subject 只有猫狗;plushie 归物体(它们是 "stuffed animal" 类)。
# TRAIN 里因此有 9 个动物 / 11 个物体,这决定了 §3 表格里 113/49 的组合分布。
ANIMAL_CLASSES = {"cat", "dog"}

# DISTILL_PLAN.md §3 的目标配比。动物 60% 是 D-2 的决策:
# 自然分布是 70%,压到 50% 会把仅有的 49 个非动物组合复用得太狠。
TARGETS = {
    # (n_refs, has_animal): 目标条数
    (2, True): 3180, (2, False): 2120,   # 合计 5300
    (3, True): 1620, (3, False): 1080,   # 合计 2700
}
DATA_DIR = "datasets/dreambooth/dataset"
OUT_DIR = "datasets/distill_multiref"
_IMG_EXTS = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------ 元数据加载

def load_classes(data_dir: str) -> dict[str, str]:
    """读 prompts_and_classes.txt 的 subject_name,class 表。"""
    path = os.path.join(data_dir, "prompts_and_classes.txt")
    classes = {}
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 只取 "Classes" 段的 a,b 行;后面 prompt_list 里的行都以 ' 开头
            if not line or line.startswith(("'", "prompt", "]", "Classes", "Prompts")):
                continue
            if line == "subject_name,class" or "," not in line:
                continue
            name, cls = line.split(",", 1)
            classes[name.strip()] = cls.strip()
    return classes


def load_scene_templates(data_dir: str) -> tuple[list[str], list[str]]:
    """抽出**场景**模板的后缀,丢掉单主体属性/换装模板(评审 B3)。

    模板原文形如 `'a {0} {1} in the jungle'.format(...)`,我们只要 "in the jungle"
    这个后缀,再自己拼多主体句式。

    弃用的部分:object 21–25 是 `a red {}` / `a cube shaped {}` 这类**属性词**,
    live 11–20 是 `in a chef outfit` 这类**换装词**——套到双主体句式上语义就废了
    ("a cat and a teapot in a chef outfit" 谁穿?)。所以:
      - 共享场景模板 = object/live 的前 10 条(两者逐字相同,下面有断言),含动物的组合用;
      - object 场景模板 = object 的前 20 条,纯物体组合用。
    """
    path = os.path.join(data_dir, "prompts_and_classes.txt")
    text = open(path, "rt", encoding="utf-8").read()
    blocks = re.findall(r"prompt_list\s*=\s*\[(.*?)\]", text, re.S)
    if len(blocks) != 2:
        raise SystemExit(f"❌ {path} 里应有 2 个 prompt_list(object / live),实际 {len(blocks)}")

    def suffixes(block: str) -> list[str]:
        out = []
        for raw in re.findall(r"'([^']*)'\s*\.format", block):
            if not raw.startswith("a {0} {1} "):
                continue  # 属性词模板(a red {0} {1})没有场景后缀,天然被这里过滤掉
            out.append(raw[len("a {0} {1} "):].strip())
        return out

    object_all, live_all = suffixes(blocks[0]), suffixes(blocks[1])
    shared = object_all[:10]
    if shared != live_all[:10]:
        raise SystemExit("❌ object 与 live 的前 10 条场景模板不再一致,模板规则需要重新确认")
    object_scene = object_all[:20]
    if len(object_scene) != 20:
        raise SystemExit(f"❌ object 场景模板应有 20 条,实际 {len(object_scene)}")
    return shared, object_scene


def load_subject_images(data_dir: str, subjects: list[str]) -> dict[str, list[str]]:
    """每个 subject 的视角图列表(排序固定,保证可复现)。"""
    out = {}
    for s in subjects:
        d = os.path.join(data_dir, s)
        if not os.path.isdir(d):
            raise SystemExit(f"❌ 缺 subject 目录:{d}")
        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS))
        if not imgs:
            raise SystemExit(f"❌ {d} 下没有图片")
        out[s] = imgs
    return out


# ------------------------------------------------------------------ 启动断言

def assert_split_clean(classes: dict[str, str]) -> None:
    """泄漏检查。这几条不通过就 sys.exit——带着污染数据跑完 8000 张再发现,代价是整个实验。"""
    overlap = set(TRAIN) & set(HELD_OUT)
    if overlap:
        raise SystemExit(f"❌ TRAIN 与 HELD-OUT 相交:{sorted(overlap)}")
    if len(TRAIN) != 20 or len(HELD_OUT) != 10:
        raise SystemExit(f"❌ 切分大小应为 20/10,实际 {len(TRAIN)}/{len(HELD_OUT)}")
    known = set(classes)
    missing = (set(TRAIN) | set(HELD_OUT)) - known
    if missing:
        raise SystemExit(f"❌ 这些 subject 不在 class 表里:{sorted(missing)}")
    if set(TRAIN) | set(HELD_OUT) != known:
        raise SystemExit(f"❌ 切分并集 != class 表全集,差集:{sorted(known ^ (set(TRAIN) | set(HELD_OUT)))}")


def assert_no_leak(tasks: list[dict]) -> None:
    """再查一遍任务列表本身——上面查的是名单,这里查的是实际产物。"""
    bad = sorted({s for t in tasks for s in t["subjects"] if s in set(HELD_OUT)})
    if bad:
        raise SystemExit(f"❌ 任务列表里出现 held-out subject:{bad}(泄漏,拒绝继续)")


def assert_refs_exist(tasks: list[dict], out_dir: str) -> int:
    """跑之前把全部 ref 文件验存一遍。

    只有约 105 个唯一文件,代价可忽略;但能把"数据没同步全"这类问题挡在启动阶段,
    而不是跑到第 3000 张才炸。
    """
    uniq = {p for t in tasks for p in t["image_paths"]}
    missing = sorted(p for p in uniq
                     if not os.path.exists(os.path.normpath(os.path.join(out_dir, p))))
    if missing:
        raise SystemExit(f"❌ {len(missing)}/{len(uniq)} 个 ref 图不存在,例如:{missing[:3]}")
    return len(uniq)


# ------------------------------------------------------------------ 任务枚举

def _stable_seed(*parts) -> int:
    """跨进程/跨次运行稳定的 seed。

    不能用内置 hash():Python 对 str 的 hash 默认带 PYTHONHASHSEED 随机化,
    8 个 shard 各自算出来的会不一样,采样就不可复现了。
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.md5(key).digest()[:8], "big")


def build_tasks(classes, shared_tpl, object_tpl, subject_images, base_seed: int) -> list[dict]:
    """枚举全部 8000 条任务。纯函数、完全确定性:同样的输入永远得到同样的列表。

    唯一性策略(DISTILL_PLAN.md §3):
      槽位 = (组合, 模板)。目标条数 > 槽位数时允许**同一槽位出现多条**,
      唯一性下沉到 (组合, 模板, 视角元组) 三元组——每条样本仍然全局唯一。
      每个 subject 有 4–6 张视角图,所以 2-ref 组合至少 16 种视角元组,
      而单槽位复用最多 3 次,取不重复的视角元组绰绰有余。
    """
    tasks = []
    for n_refs in (2, 3):
        # 合法组合:同 class 的不能凑一组("a dog and a dog"没有意义,也无法区分)
        valid = [c for c in combinations(TRAIN, n_refs)
                 if len({classes[s] for s in c}) == n_refs]
        for has_animal in (True, False):
            group = [c for c in valid
                     if any(classes[s] in ANIMAL_CLASSES for s in c) == has_animal]
            tpl = shared_tpl if has_animal else object_tpl
            slots = [(c, t) for c in group for t in range(len(tpl))]
            target = TARGETS[(n_refs, has_animal)]
            if not slots:
                raise SystemExit(f"❌ ({n_refs}-ref, animal={has_animal}) 没有可用槽位")

            # 分层取样:先保证每个槽位都被用满 floor(target/slots) 次(全覆盖),
            # 余数再随机撒。这样复用次数最多相差 1,不会出现某些组合被反复刷、
            # 另一些一次没出现的偏斜。
            reps_full, remainder = divmod(target, len(slots))
            picked = []
            for r in range(reps_full):
                picked += [(s, r) for s in slots]
            if remainder:
                rng = random.Random(_stable_seed("remainder", n_refs, has_animal, base_seed))
                extra = list(slots)
                rng.shuffle(extra)
                picked += [(s, reps_full) for s in extra[:remainder]]

            for (combo, tpl_id), rep in picked:
                # 该 (组合, 模板) 槽位下,第 rep 次复用用第 rep 个视角元组。
                # 每个槽位独立采样,所以不同模板可以撞上同样的视角元组——
                # 这没关系,三元组仍然不同。
                views = list(product(*[range(len(subject_images[s])) for s in combo]))
                need = reps_full + (1 if remainder else 0)
                if need > len(views):
                    raise SystemExit(
                        f"❌ {combo} 需要 {need} 个不重复视角元组,但只有 {len(views)} 个")
                rng = random.Random(_stable_seed("view", combo, tpl_id, base_seed))
                chosen = rng.sample(views, need)[rep]
                tasks.append({
                    "subjects": list(combo),
                    "view_idx": list(chosen),
                    "template_id": tpl_id,
                    "template": tpl[tpl_id],
                    "n_refs": n_refs,
                    "has_animal": has_animal,
                })

    # 排序 + 编号:必须在分片之前定死,否则 8 个 shard 对 idx 的理解会不一致,
    # 输出文件名和 seed 全乱。按内容排序而不是按生成顺序,保证与枚举实现解耦。
    tasks.sort(key=lambda t: (t["n_refs"], t["subjects"], t["template_id"], t["view_idx"]))
    for i, t in enumerate(tasks):
        t["idx"] = i
        t["seed"] = base_seed + i
        t["prompt"] = make_prompt([classes[s] for s in t["subjects"]], t["template"])
        t["image_paths"] = [
            f"../dreambooth/dataset/{s}/{subject_images[s][v]}"
            for s, v in zip(t["subjects"], t["view_idx"])
        ]
        t["image_tgt_path"] = f"images/{i:06d}.jpg"
    return tasks


def make_prompt(class_tokens: list[str], suffix: str) -> str:
    """`a cat and a teapot in the jungle` / `a cat, a teapot and a vase in the jungle`。

    句式与 `multibanana_eval/dreambench_multiip.json` 保持一致。
    """
    if len(class_tokens) == 1:
        head = f"a {class_tokens[0]}"
    else:
        head = ", ".join(f"a {c}" for c in class_tokens[:-1]) + f" and a {class_tokens[-1]}"
    return f"{head} {suffix}"


# ------------------------------------------------------------------ 统计(--dry_run 的产出)

def print_stats(tasks: list[dict], classes: dict[str, str]) -> None:
    """烧 GPU 之前唯一能验证配比对不对的地方,所以打全一点。"""
    print("\n" + "=" * 72)
    print(f"任务总数:{len(tasks)}")
    by_n = Counter(t["n_refs"] for t in tasks)
    for n in sorted(by_n):
        sub = [t for t in tasks if t["n_refs"] == n]
        animal = sum(t["has_animal"] for t in sub)
        print(f"  {n}-ref:{len(sub):>5}  含动物 {animal:>5} ({animal / len(sub):.1%})  "
              f"纯物体 {len(sub) - animal:>5}")
    total_animal = sum(t["has_animal"] for t in tasks)
    print(f"  全体含动物比例:{total_animal / len(tasks):.1%}(目标 60%)")

    print("-" * 72)
    print("槽位复用((组合,模板) 被用了几次):")
    for n in sorted(by_n):
        for ha in (True, False):
            sub = [t for t in tasks if t["n_refs"] == n and t["has_animal"] == ha]
            if not sub:
                continue
            reuse = Counter((tuple(t["subjects"]), t["template_id"]) for t in sub)
            dist = Counter(reuse.values())
            print(f"  {n}-ref/{'动物' if ha else '物体'}:槽位 {len(reuse)},"
                  f"复用分布 {dict(sorted(dist.items()))},最大 {max(reuse.values())}")

    print("-" * 72)
    triples = {(tuple(t["subjects"]), t["template_id"], tuple(t["view_idx"])) for t in tasks}
    ok = "✓" if len(triples) == len(tasks) else "✗"
    print(f"(组合,模板,视角元组) 唯一性:{len(triples)}/{len(tasks)} {ok}")
    seeds = {t["seed"] for t in tasks}
    print(f"seed 唯一性:{len(seeds)}/{len(tasks)} {'✓' if len(seeds) == len(tasks) else '✗'}")

    print("-" * 72)
    subj = Counter(s for t in tasks for s in t["subjects"])
    print(f"涉及 subject {len(subj)} 个(应为 20),出现次数 "
          f"min {min(subj.values())} / max {max(subj.values())}")
    leaked = [s for s in subj if s in set(HELD_OUT)]
    print(f"held-out 泄漏检查:{'✓ 无' if not leaked else '✗ ' + str(leaked)}")
    print("最少出现的 5 个:", ", ".join(f"{s}={c}" for s, c in subj.most_common()[-5:]))

    print("-" * 72)
    print("样例 prompt:")
    for t in (tasks[0], tasks[len(tasks) // 2], tasks[-1]):
        print(f"  [{t['idx']:>5}] {t['prompt']}")
        print(f"          refs={t['image_paths']}")
    print("=" * 72 + "\n")


# ------------------------------------------------------------------ 断点续跑

def already_done(path: str) -> bool:
    """输出图存在**且能完整解码**才算跳过。

    不能只 `Image.open`:它是惰性的,只读文件头。[已验证] 一张截断到一半的 JPEG,
    `Image.open` 照样通过并报出正确尺寸 (512,512),只有 `.load()` 才抛
    `OSError: image file is truncated`。shard 被 kill 时写到一半的图正是这个场景,
    而断点续跑存在的意义就是应对被 kill。
    """
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        os.remove(path)  # 坏图直接删掉,免得下次又被 os.path.exists 认成"已完成"
        return False


# ------------------------------------------------------------------ 合并

def do_merge(out_dir: str) -> None:
    shards = sorted(f for f in os.listdir(out_dir)
                    if re.fullmatch(r"manifest_shard\d+\.json", f))
    if not shards:
        raise SystemExit(f"❌ {out_dir} 下没有 manifest_shard*.json")
    merged, seen = [], set()
    for fn in shards:
        with open(os.path.join(out_dir, fn), "rt", encoding="utf-8") as f:
            items = json.load(f)
        print(f"  {fn}: {len(items)} 条")
        for it in items:
            key = it["image_tgt_path"]
            if key in seen:  # shard 划分应当互斥,撞了说明分片参数用错了
                raise SystemExit(f"❌ {key} 在多个 shard 里出现,分片参数不一致,拒绝合并")
            seen.add(key)
            merged.append(it)
    merged.sort(key=lambda x: x["image_tgt_path"])
    out = os.path.join(out_dir, "manifest_raw.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    n_animal = sum(1 for m in merged if m["meta"].get("has_animal"))
    by_n = Counter(m["meta"]["n_refs"] for m in merged)
    print(f"\n合并 {len(shards)} 个 shard → {out}")
    print(f"  总计 {len(merged)} 条;{dict(sorted(by_n.items()))};含动物 {n_animal / len(merged):.1%}")


# ------------------------------------------------------------------ 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--base_seed", type=int, default=3407000)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条,用于标定吞吐(0=全部)")
    # teacher 配置,DISTILL_PLAN.md D-1:H800 上用 bf16 不 offload,没理由省精度。
    # 换 fp8 会改变蒸馏数据质量上限 → 红灯,别自己改。
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ref_size", type=int, default=512,
                   help="必须与训练侧对齐:train.py 的 resolution_ref=None 会回落到 resolution=512")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--lora_rank", type=int, default=512)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--log_interval", type=float, default=30.0)
    p.add_argument("--dry_run", action="store_true",
                   help="不加载模型、不写任何文件,只枚举任务并打印统计表")
    p.add_argument("--merge", action="store_true", help="合并各 shard 的 manifest")
    args = p.parse_args()

    if args.merge:
        do_merge(args.out_dir)
        return

    # ---------- 元数据 + 断言 ----------
    classes = load_classes(args.data_dir)
    assert_split_clean(classes)
    shared_tpl, object_tpl = load_scene_templates(args.data_dir)
    subject_images = load_subject_images(args.data_dir, TRAIN)
    tasks = build_tasks(classes, shared_tpl, object_tpl, subject_images, args.base_seed)
    assert_no_leak(tasks)

    # ref 验存要在 out_dir 建好之后做(路径含 `..`),dry_run 时目录可能还不存在,
    # 所以这里先建目录——dry_run 除此之外不写任何文件
    os.makedirs(args.out_dir, exist_ok=True)
    n_ref_files = assert_refs_exist(tasks, args.out_dir)

    if args.dry_run:
        print(f"ref 图验存:{n_ref_files} 个唯一文件全部可读 ✓")
        print(f"模板:共享场景 {len(shared_tpl)} 条(含动物组合用)、"
              f"object 场景 {len(object_tpl)} 条(纯物体组合用)")
        print(f"视角图:每 subject "
              f"{min(len(v) for v in subject_images.values())}–"
              f"{max(len(v) for v in subject_images.values())} 张")
        print_stats(tasks, classes)
        print("dry_run:未加载模型,未写任何文件。")
        return

    # ---------- 分片 ----------
    if not 0 <= args.shard_idx < args.num_shards:
        raise SystemExit(f"❌ shard_idx {args.shard_idx} 不在 [0, {args.num_shards}) 内")
    mine = tasks[args.shard_idx::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    done_before = sum(1 for t in mine
                      if already_done(os.path.join(args.out_dir, t["image_tgt_path"])))

    # 启动自检摘要:让"任务数不对"这种致命错误在第一秒暴露,而不是等跑完
    tag = f"shard {args.shard_idx}/{args.num_shards}"
    print(f"[{datetime.now():%H:%M:%S}] === M1 gen_data 启动 ===", flush=True)
    print(f"  {tag} | 全局任务 {len(tasks)} → 本 shard {len(mine)} 条 "
          f"(断点续跑已完成 {done_before},待跑 {len(mine) - done_before})", flush=True)
    print(f"  teacher: {args.model_type} offload={args.offload} ref_size={args.ref_size} "
          f"steps={args.num_steps} guidance={args.guidance} {args.width}x{args.height}", flush=True)
    print(f"  输出: {img_dir} | manifest: "
          f"{args.out_dir}/manifest_shard{args.shard_idx}.json", flush=True)

    # ---------- 加载 teacher ----------
    # teacher = 官方 UNO dit_lora + full attention。UNOPipeline(only_lora=True) 挂上的
    # 就是官方权重,所以这里**不做任何 swap_lora**——我们要的正是"官方 vs 官方"。
    import torch
    from uno.flux.pipeline import UNOPipeline, preprocess_ref

    t_load = time.perf_counter()
    pipeline = UNOPipeline(args.model_type, torch.device("cuda"), offload=args.offload,
                           only_lora=True, lora_rank=args.lora_rank)
    if args.offload:
        # __init__ 把 t5/clip 放 GPU 且只在 forward 里 offload,首个 forward 搬 DiT 上卡时
        # 它们还占着约 10G。先踢回 CPU 腾地方(抄自 infer_multibanana.py)。
        pipeline.t5.cpu()
        pipeline.clip.cpu()
        torch.cuda.empty_cache()
    print(f"[{datetime.now():%H:%M:%S}] teacher 就绪,耗时 {time.perf_counter() - t_load:.1f}s",
          flush=True)

    # ---------- 逐条生成 ----------
    manifest, fails = [], []
    t_start = time.perf_counter()
    t_last_log = t_start
    n_done = n_skip = 0

    for i, task in enumerate(mine):
        tgt = os.path.join(args.out_dir, task["image_tgt_path"])
        if already_done(tgt):
            n_skip += 1
            manifest.append(to_record(task))
            continue
        try:
            refs = []
            for rel in task["image_paths"]:
                # image_paths 是相对 manifest 所在目录的(FluxPairedDatasetV2 的约定:
                # image_root = os.path.dirname(json_file)),这里要还原成真实路径。
                # normpath 是必须的:路径里的 `..` 在中间目录尚未创建时会让 os.path.exists
                # 假阴性,归一化之后才是稳的
                src = os.path.normpath(os.path.join(args.out_dir, rel))
                refs.append(preprocess_ref(Image.open(src).convert("RGB"), args.ref_size))
            img = pipeline(
                prompt=task["prompt"], width=args.width, height=args.height,
                guidance=args.guidance, num_steps=args.num_steps, seed=task["seed"],
                ref_imgs=refs, pe="d",
                ref_isolation=False, kv_cache=False,   # teacher 就是 full attention
            )
            # 先写临时文件再 rename:rename 在同一文件系统上是原子的,
            # 这样被 kill 时不会留下半张图(虽然 already_done 也能兜住,但少一次重跑)
            tmp = tgt + ".tmp"
            img.save(tmp, quality=args.jpeg_quality)
            os.replace(tmp, tgt)
            manifest.append(to_record(task))
            n_done += 1
        except Exception as exc:  # noqa: BLE001 — 一个坏样本不许杀掉整个 shard
            fails.append({"idx": task["idx"], "error": f"{type(exc).__name__}: {exc}"})
            # 失败当场打印,不攒到最后:攒起来的后果是跑了两小时才发现前十分钟就全在失败
            print(f"[{datetime.now():%H:%M:%S}] ❌ idx={task['idx']} "
                  f"{'+'.join(task['subjects'])} — {type(exc).__name__}: {exc}", flush=True)

        now = time.perf_counter()
        if (i + 1) % args.log_every == 0 or now - t_last_log >= args.log_interval \
                or i == len(mine) - 1:
            rate = (now - t_start) / max(n_done, 1)
            left = len(mine) - i - 1
            print(f"[{datetime.now():%H:%M:%S}] {tag} | {i + 1}/{len(mine)} "
                  f"({(i + 1) / len(mine):.1%}) | {rate:.2f} s/img | "
                  f"ETA {left * rate / 60:.0f}m | skip {n_skip} | fail {len(fails)}", flush=True)
            t_last_log = now
            # 边跑边落盘:2–3 小时无人值守,被 kill 时不能把整个 manifest 丢掉
            write_manifest(args.out_dir, args.shard_idx, manifest, fails)

    write_manifest(args.out_dir, args.shard_idx, manifest, fails)
    elapsed = time.perf_counter() - t_start
    print(f"[{datetime.now():%H:%M:%S}] === {tag} 完成 ===", flush=True)
    print(f"  生成 {n_done} | 跳过 {n_skip} | 失败 {len(fails)} | "
          f"耗时 {elapsed / 60:.1f}m | 平均 {elapsed / max(n_done, 1):.2f} s/img", flush=True)
    if fails:
        print(f"  失败明细见 {args.out_dir}/failures_shard{args.shard_idx}.json", flush=True)


def to_record(task: dict) -> dict:
    """输出 schema 兼容 FluxPairedDatasetV2([已验证] __getitem__ 忽略多余键,meta 安全)。"""
    return {
        "image_paths": task["image_paths"],
        "prompt": task["prompt"],
        "image_tgt_path": task["image_tgt_path"],
        "meta": {
            "subjects": task["subjects"],
            "view_idx": task["view_idx"],
            "seed": task["seed"],
            "template_id": task["template_id"],
            "n_refs": task["n_refs"],
            "has_animal": task["has_animal"],
        },
    }


def write_manifest(out_dir: str, shard_idx: int, manifest: list, fails: list) -> None:
    for name, payload in ((f"manifest_shard{shard_idx}.json", manifest),
                          (f"failures_shard{shard_idx}.json", fails)):
        if not payload and name.startswith("failures"):
            continue
        tmp = os.path.join(out_dir, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
