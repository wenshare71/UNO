"""M2:用 DINO(dino_vits16)给 M1 蒸馏数据打分并按质量过滤。

M1 的 teacher 也会掉主体、也会画糊——不是每张都能进 student 的续训集。M2 的作用是把
teacher 确实"把每个被要求的主体都还原出来了"的样本挑出来,其余丢掉。完整推导见
`distill/DISTILL_PLAN.md` §4。

━━ 为什么用 min-over-refs,而不是官方的 mean(这是对官方实现的**有意偏离**)━━
官方 `eval/evaluate_clip_dino_score_multi_subject.py:260-268` 对一条多主体样本的每个
subject 各算一个 DINO 相似度,然后 `np.mean` 取平均。问题正出在这:如果 teacher 把
第一个主体画得很好(sim 0.8)、第二个主体整个丢了(sim 0.2),平均是 0.5——看着"中等",
掉主体这件事被高分主体**掩盖**了。而我们要解决的根因恰恰就是掉第二主体,用 mean 过滤
等于把要抓的坏样本放走。所以改成 **min over refs**:取一条样本里最差的那个主体的相似度。
掉主体 → min 直接掉到低位,一抓一个准。

━━ 为什么把特征提取代码复制过来,而不是 import 官方脚本 ━━
[已验证] `eval/evaluate_clip_dino_score_multi_subject.py` 有两个 import 期阻塞点:
  (a) 第 8 行 `import clip`,而 OpenAI CLIP 包没装(ModuleNotFoundError);
  (b) 第 199-215 行是**模块级**的 `parser.parse_args()`(required=True)和
      `clip.load(..., device='cuda')`——一 import 就 SystemExit 并往 GPU 加载 CLIP。
所以下面把 `DINOImageDataset`(原 :70)和 `extract_all_images`(原 :109)**逐字抄进来**,
只去掉用不上的 clip 依赖。backbone 仍是 `dino_vits16`(不是 DINOv2),与官方评测脚本
和 DreamBench/UNO 论文一致,换 backbone 会失去与论文数字的可比性(评审 B1)。

━━ 用法 ━━
    # 1) 标定:打分 + 分位数表 + 按 min_ref_sim 排序抽 40 张拼图,然后**人工看图定阈值**
    python distill/filter_data.py --calibrate

    # 2) 定好阈值后过滤,产出 manifest_filtered.json + 通过率
    python distill/filter_data.py --threshold 0.45

分数会写进 manifest_scored.json(meta.dino_sims / meta.min_ref_sim)缓存;第 2 步默认
复用它,不再重跑 GPU 打分(除非 --force_rescore)。
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 与 gen_data.py 的默认输出目录保持一致
DEFAULT_OUT = "datasets/distill_multiref"
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUT, "manifest_raw.json")
DEFAULT_SCORED = os.path.join(DEFAULT_OUT, "manifest_scored.json")


# ══════════════════════════════════════════════════════════ 复制区(勿 import)
# 下面两段抄自 eval/evaluate_clip_dino_score_multi_subject.py(:70 / :109),
# 原因见文件头 docstring。除去掉 clip 依赖外,预处理与提特征逻辑逐字保持一致,
# 以确保和官方评测的 DINO 数值口径相同。

def _make_dino_preprocess():
    """DINOImageDataset._transform_test(224) 的原样搬运(原 :76-83)。"""
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

    def _convert(image):
        return image.convert("RGB")

    return Compose([
        Resize(256, interpolation=Image.BICUBIC),
        CenterCrop(224),
        _convert,
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    """抄自原 :109。DINO 没有 `encode_image`,走 else 分支 `model(b)`,保持 fp32。

    返回 [N, 384] 的 numpy 特征,顺序与输入 `images` 一致。
    """
    import torch

    class _DS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            self.preprocess = _make_dino_preprocess()

        def __getitem__(self, idx):
            return {"image": self.preprocess(Image.open(self.data[idx]))}

        def __len__(self):
            return len(self.data)

    loader = torch.utils.data.DataLoader(
        _DS(images), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats = []
    with torch.no_grad():
        for b in loader:
            b = b["image"].to(device)
            feats.append(model(b).cpu().numpy())
    return np.vstack(feats)


# ══════════════════════════════════════════════════════════ 路径与打分

def resolve(out_dir: str, rel: str) -> str:
    """manifest 里的路径都相对 out_dir。normpath 处理 image_paths 里的 `..`。"""
    return os.path.normpath(os.path.join(out_dir, rel))


def load_dino(device: str):
    """dino_vits16。假定 M0 已把 repo + 权重预取进 $TORCH_HOME(见 DISTILL_PLAN §3.0)。"""
    import torch
    # trust_repo=True:无人值守时不能卡在 hub 的 y/n 确认提示上
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                           pretrained=True, trust_repo=True)
    return model.to(device).eval()


def build_feature_table(paths, device, batch_size, num_workers):
    """对一批**去重后**的图路径提 DINO 特征,L2 归一化后返回 {path: unit_vector}。

    生成图 8000 张 + ref 图约 105 张,一次性提完再查表,避免同一张 ref 被重复提特征。
    """
    model = load_dino(device)
    feats = extract_all_images(paths, model, device, batch_size, num_workers)
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    return {p: feats[i] for i, p in enumerate(paths)}


def score_records(records, out_dir, device, batch_size, num_workers):
    """给每条样本算 dino_sims(每个 ref 与生成图的余弦)与 min_ref_sim,写回 meta。

    返回 (scored, skipped):生成图缺失/损坏的样本进 skipped(不参与后续过滤)。
    """
    # 先分出哪些样本的图齐全。生成图可能因 M1 失败而缺失;ref 图理论上都在,仍做检查。
    usable, skipped = [], []
    need_paths = set()
    for r in records:
        gen = resolve(out_dir, r["image_tgt_path"])
        refs = [resolve(out_dir, p) for p in r["image_paths"]]
        missing = [p for p in [gen, *refs] if not os.path.exists(p)]
        if missing:
            skipped.append({"image_tgt_path": r["image_tgt_path"], "missing": missing})
            continue
        r["_gen"], r["_refs"] = gen, refs
        need_paths.update([gen, *refs])
        usable.append(r)

    if not usable:
        raise SystemExit("❌ 没有一条样本的图是齐全的——manifest 与图目录对不上?检查 --out_dir")

    print(f"打分:{len(usable)} 条可用,{len(skipped)} 条因缺图跳过;"
          f"去重后 {len(need_paths)} 张图待提特征", flush=True)
    table = build_feature_table(sorted(need_paths), device, batch_size, num_workers)

    for r in usable:
        sims = [float(table[r["_gen"]] @ table[p]) for p in r["_refs"]]
        r["meta"]["dino_sims"] = sims
        r["meta"]["min_ref_sim"] = min(sims)
        del r["_gen"], r["_refs"]  # 临时字段不落盘
    return usable, skipped


# ══════════════════════════════════════════════════════════ 标定:分位表 + 拼图

def _strata(records):
    """(标签, 子集) 列表:总体 + 按 n_refs + 按 has_animal,便于发现系统性偏差。"""
    out = [("全体", records)]
    for n in (2, 3):
        out.append((f"{n}-ref", [r for r in records if r["meta"]["n_refs"] == n]))
    out.append(("含动物", [r for r in records if r["meta"]["has_animal"]]))
    out.append(("纯物体", [r for r in records if not r["meta"]["has_animal"]]))
    return out


def print_quantiles(records):
    """min_ref_sim 的分位数表(附 mean 分位作对照,凸显 min 抓掉主体的价值)。"""
    ps = [1, 5, 10, 25, 50, 75, 90]
    print("\n" + "=" * 72)
    print("min_ref_sim 分位数(阈值就切在这张表上;越低越可能掉主体)")
    print(f"{'子集':<8}{'n':>6} " + " ".join(f"p{p:<4}" for p in ps))
    print("-" * 72)
    for label, sub in _strata(records):
        if not sub:
            continue
        arr = np.array([r["meta"]["min_ref_sim"] for r in sub])
        cells = " ".join(f"{np.percentile(arr, p):.3f}" for p in ps)
        print(f"{label:<8}{len(sub):>6} {cells}")
    # mean 对照:同样本用平均聚合会高出多少——高出越多,说明 min 拦下的掉主体越多
    print("-" * 72)
    print("对照:全体在 mean 聚合下的同分位(官方口径,会掩盖掉主体)")
    mean_arr = np.array([float(np.mean(r["meta"]["dino_sims"])) for r in records])
    print(f"{'mean':<8}{len(records):>6} " + " ".join(f"{np.percentile(mean_arr, p):.3f}" for p in ps))
    print("=" * 72)


def make_board(records, out_dir, n, save_path):
    """按 min_ref_sim 排序、等间隔抽 n 条拼成一张总览图,人工据此定阈值。

    等间隔(而非只取最低 n 张)是为了让整条分数谱都呈现出来——既看得到"多低算掉主体",
    也看得到"多高算干净",阈值切在哪一目了然。复用 multibanana_eval/board.py。
    """
    from multibanana_eval.board import build_row, stack_board

    ordered = sorted(records, key=lambda r: r["meta"]["min_ref_sim"])
    picks = np.linspace(0, len(ordered) - 1, min(n, len(ordered))).round().astype(int)
    rows = []
    for i in sorted(set(picks.tolist())):
        r = ordered[i]
        refs = [Image.open(resolve(out_dir, p)).convert("RGB") for p in r["image_paths"]]
        gen = Image.open(resolve(out_dir, r["image_tgt_path"])).convert("RGB")
        sims = " ".join(f"{s:.2f}" for s in r["meta"]["dino_sims"])
        title = (f"{os.path.basename(r['image_tgt_path'])}  "
                 f"min={r['meta']['min_ref_sim']:.3f}  sims=[{sims}]")
        rows.append(build_row(title, r["prompt"], refs, {"teacher": gen}))
    stack_board(rows).save(save_path)
    print(f"\n拼图已存:{save_path}({len(rows)} 行,按 min_ref_sim 从低到高)", flush=True)


# ══════════════════════════════════════════════════════════ 过滤

def apply_threshold(records, thr):
    """保留 min_ref_sim >= thr 的样本;返回 (kept, passrate_by_stratum)。"""
    kept = [r for r in records if r["meta"]["min_ref_sim"] >= thr]
    report = []
    for label, sub in _strata(records):
        if not sub:
            continue
        n_pass = sum(1 for r in sub if r["meta"]["min_ref_sim"] >= thr)
        report.append((label, n_pass, len(sub)))
    return kept, report


# ══════════════════════════════════════════════════════════ 主流程

def _clean(records):
    """落盘前移除临时字段(以 _ 开头),保证 manifest 干净。"""
    for r in records:
        for k in [k for k in r if k.startswith("_")]:
            del r[k]
    return records


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f"❌ 找不到 manifest:{path}(M1 是否已 --merge?)")
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def have_scores(records):
    return bool(records) and all("min_ref_sim" in r.get("meta", {}) for r in records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--scored", default=DEFAULT_SCORED, help="打分结果缓存(读/写)")
    p.add_argument("--out_dir", default=DEFAULT_OUT, help="图路径的基准目录")
    p.add_argument("--calibrate", action="store_true",
                   help="出分位数表 + 拼图(不给 --threshold 时的默认行为)")
    p.add_argument("--threshold", type=float, default=None,
                   help="保留 min_ref_sim >= 阈值的样本,产出 manifest_filtered.json")
    p.add_argument("--board_n", type=int, default=40)
    p.add_argument("--board_out", default=None)
    p.add_argument("--filtered_out", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force_rescore", action="store_true",
                   help="即使 scored 缓存已存在也重新打分")
    args = p.parse_args()

    # ---------- 拿到带分数的记录:优先复用缓存,否则跑 GPU 打分 ----------
    if not args.force_rescore and os.path.exists(args.scored):
        records = load_manifest(args.scored)
        if have_scores(records):
            print(f"复用打分缓存 {args.scored}({len(records)} 条),跳过 GPU 打分。"
                  f"要重打分加 --force_rescore", flush=True)
        else:
            records = None  # 缓存不完整,重打
    else:
        records = None

    if records is None:
        raw = load_manifest(args.manifest)
        records, skipped = score_records(
            raw, args.out_dir, args.device, args.batch_size, args.num_workers)
        write_json(args.scored, _clean(records))
        print(f"打分完成 → {args.scored}(跳过 {len(skipped)} 条缺图样本)", flush=True)

    # ---------- 标定(默认或显式 --calibrate)----------
    if args.calibrate or args.threshold is None:
        print_quantiles(records)
        board_out = args.board_out or os.path.join(args.out_dir, "board_calibrate.png")
        make_board(records, args.out_dir, args.board_n, board_out)

    # ---------- 过滤 ----------
    if args.threshold is not None:
        kept, report = apply_threshold(records, args.threshold)
        filtered_out = args.filtered_out or os.path.join(args.out_dir, "manifest_filtered.json")
        write_json(filtered_out, _clean(kept))
        print("\n" + "=" * 72)
        print(f"阈值 {args.threshold}:通过率")
        for label, n_pass, n_tot in report:
            print(f"  {label:<8} {n_pass:>5}/{n_tot:<5} ({n_pass / n_tot:.1%})")
        print("=" * 72)
        print(f"→ {filtered_out}({len(kept)} 条)", flush=True)
        overall = report[0]
        if overall[1] / overall[2] < 0.5:
            print("⚠️ 总体通过率 <50%。先看 board 找系统性原因(某类模板差?3-ref 差?),"
                  "不要盲目降阈值(DISTILL_PLAN §4)。", flush=True)


if __name__ == "__main__":
    main()
