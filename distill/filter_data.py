"""M2:逐主体「定位 + 双侧裁剪 + DINO 身份比对」给 M1 蒸馏数据打分并过滤(度量 v2)。

M1 的 teacher 也会掉主体、也会画糊——不是每张都能进 student 的续训集。M2 的作用是把
teacher 确实"把每个被要求的主体都还原出来了"的样本挑出来,其余丢掉。完整推导见
`distill/DISTILL_PLAN.md` §4(含 v2 修订记录)。

━━ 为什么是 v2:整图 DINO(v1)已被 smoke + 地板线实验证伪(2026-07-24)━━
v1 直接拿整张生成图与整张 ref 图算 DINO 余弦、取 min-over-refs。16 张 smoke 样本上:
  * 与 text-only 地板线(同 prompt/seed、不喂 ref 的纯文本生成)分布几乎重叠:
    中位差仅 +0.028,16 组里 6 组**反转**(泛型物体比忠实复刻分还高),0/16 越过地板线;
  * 排序与人眼判读接近相反:视觉满分的 005500/006500 排倒数一二,
    主体融合失败的 005000 反而排第 14/16;
  * 根因:整图 CLS 特征测的是**全局场景构图**,多主体 + 丰富背景把主体信号稀释光了
    (绝对值 0.13–0.36,远低于单主体 DreamBench 的 0.6–0.8);而且 **ref 侧同样被污染**
    ——dreambooth 的 backpack ref 是"人背着包",人和天空占画面主导。
官方 eval(:181-196)同样用整图,但那是几百样本 × 全部视角求 mean 的 benchmark 级聚合,
系统性差异能从噪声里浮出来;逐样本判定要的信噪比完全不同,不能沿用整图口径。

━━ v2 度量:presence + identity 两级 ━━
对样本里每个被要求的主体:
  1. **presence** —— GroundingDino(tiny)用类别词(如 "a teapot.")在生成图上做
     开放词表检测。一个框都没有 → 该主体判掉,sim 记 0.0。掉主体是要抓的头号失败,
     检测器直接给出二值信号,不再指望相似度阈值去猜"多低算没有"。
  2. **identity** —— 每个候选框 crop(带 12% 上下文 padding),与该主体**全部参考视角**
     的主体 crop(ref 侧同样用检测器裁,去掉"背包上的人"这类污染)算 DINO 余弦,
     取 max over (框 × 视角):不确定哪个框是它 → 让最像的框代表它;生成姿态与单张 ref
     不一致 → 让最像的视角代表它(官方靠"比全部视角求均值"抵消姿态噪声,逐样本场景
     用 max 达到同样目的)。
样本分 `min_ref_sim` = **min over subjects**(理由与 v1 相同,是对官方 mean 的有意偏离:
teacher 把主体 A 画到 0.8、把 B 整个丢了记 0.0,mean=0.4 看着"中等",掉主体被高分主体
掩盖;min 直接掉到 0,一抓一个准)。

已知盲区(标定时人工盯):同类双主体(如 cat+cat2)共用一个类别词,若 teacher 只画了
一只,两个 ref 可能同时匹配到它,identity 分数是否拉得开取决于两实例长相差异;
主体融合(两身份糊成一体)会以"两边都中等分"出现,未必垫底。

━━ 为什么把官方特征提取代码复制过来,而不是 import ━━
[已验证] `eval/evaluate_clip_dino_score_multi_subject.py` 有两个 import 期阻塞点:
  (a) 第 8 行 `import clip`,而 OpenAI CLIP 包没装(ModuleNotFoundError);
  (b) 第 199-215 行是**模块级**的 `parser.parse_args()`(required=True)和
      `clip.load(..., device='cuda')`——一 import 就 SystemExit 并往 GPU 加载 CLIP。
所以把 `DINOImageDataset` 的预处理与 `extract_all_images`(原 :70/:109)搬进来,唯一改动
是条目允许 PIL.Image(v2 喂的是内存里的 crop),数值口径不变。backbone 仍是 `dino_vits16`
(不是 DINOv2),与官方评测脚本、DreamBench/UNO 论文一致(评审 B1)。

━━ 用法 ━━
    # 0)(远程一次性)预取检测器权重,~0.7GB,走 HF 代理约 20 分钟
    python -c "from transformers import AutoProcessor, GroundingDinoForObjectDetection as G; \
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
G.from_pretrained('IDEA-Research/grounding-dino-tiny'); print('grounder ok')"

    # 1) 标定:打分 + 分位数表 + 按 min_ref_sim 排序抽样拼图,然后**人工看图定阈值**
    CUDA_VISIBLE_DEVICES=0 python distill/filter_data.py --calibrate

    # 2) 定好阈值后过滤,产出 manifest_filtered.json + 通过率
    python distill/filter_data.py --threshold 0.45

分数写进 manifest_scored.json(meta.dino_sims / meta.min_ref_sim / meta.det),并带
meta.metric 版本标记——旧的 v1 整图分数缓存会因标记不符自动作废重打,无需 --force_rescore。
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 与 gen_data.py 的默认目录保持一致(DATA_DIR 直接复用它的常量,避免两处漂移)
DEFAULT_OUT = "datasets/distill_multiref"
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUT, "manifest_raw.json")
DEFAULT_SCORED = os.path.join(DEFAULT_OUT, "manifest_scored.json")

GROUNDER_ID = "IDEA-Research/grounding-dino-tiny"
METRIC_TAG = "v2-grounded-crop"   # 写进 meta.metric;改度量必须换标记,让旧缓存自动失效


# ══════════════════════════════════════════════════════════ 复制区(勿 import)

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


def extract_all_images(images, model, device, batch_size=64, num_workers=0):
    """抄自原 :109;唯一改动:条目允许 PIL.Image(v2 的 crop 在内存里,不落盘)。

    DINO 没有 `encode_image`,走 else 分支 `model(b)`,保持 fp32。
    返回 [N, 384] 的 numpy 特征,顺序与输入一致。in-memory 条目用 num_workers=0,
    避免 DataLoader 把整批 PIL 图 pickle 给 worker。
    """
    import torch

    class _DS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            self.preprocess = _make_dino_preprocess()

        def __getitem__(self, idx):
            d = self.data[idx]
            img = d if isinstance(d, Image.Image) else Image.open(d)
            return {"image": self.preprocess(img)}

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


# ══════════════════════════════════════════════════════════ 模型与检测

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


def load_grounder(device: str):
    """GroundingDino tiny(transformers 4.43.3 自带该架构,4.40+ 均可)。"""
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    proc = AutoProcessor.from_pretrained(GROUNDER_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(GROUNDER_ID).to(device).eval()
    return proc, model


def detect_boxes(ctx, image, phrase, top_k=6):
    """开放词表检测:返回 [(box_xyxy, score)],按置信度降序,最多 top_k 个。

    查询按 GroundingDino 契约:小写、以句点结尾。top_k 只是护栏——类别词通常命中 1~3 框,
    偶发的碎框全收进来白费 DINO 前向。
    """
    import torch
    proc, model = ctx["grounder"]
    text = phrase.lower().strip().rstrip(".") + "."
    inputs = proc(images=image, text=text, return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, box_threshold=ctx["box_thr"], text_threshold=0.25,
        target_sizes=[image.size[::-1]])[0]
    pairs = sorted(zip(res["boxes"].tolist(), res["scores"].tolist()), key=lambda t: -t[1])
    return [(tuple(b), float(s)) for b, s in pairs[:top_k]]


def crop_pad(img, box, pad=0.12):
    """带上下文 padding 的裁剪:紧框会把物体轮廓贴边裁死,DINO 特征需要一点边界上下文。"""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((max(0, int(x0 - px)), max(0, int(y0 - py)),
                     min(img.width, int(x1 + px)), min(img.height, int(y1 + py))))


def extract_unit_feats(images, ctx):
    """一批图(路径或 PIL)→ L2 归一化 DINO 特征 [N, 384]。"""
    feats = extract_all_images(images, ctx["dino"], ctx["device"], ctx["batch_size"])
    return feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)


def make_ctx(device, data_dir, box_thr, batch_size):
    """一次加载两个模型 + 类别词表 + 懒建的 ref 特征库。

    filter_data 与 floor_line 共用同一个 ctx 打分,保证 teacher / text-only 口径严格一致。
    """
    from distill.gen_data import load_classes
    return {
        "device": device, "data_dir": data_dir, "box_thr": box_thr,
        "batch_size": batch_size, "dino": load_dino(device),
        "grounder": load_grounder(device), "classes": load_classes(data_dir),
        "ref_bank": {},
    }


def _class_word(ctx, subject):
    cls = ctx["classes"].get(subject)
    if not cls:
        raise SystemExit(f"❌ subject {subject!r} 不在 prompts_and_classes.txt 的类别表里")
    return cls


def _ref_vecs(ctx, subject):
    """subject → 全部参考视角的主体 crop 单位特征 [n_views, 384]。懒建 + 缓存。

    ref 侧也必须过检测器裁剪([已验证] backpack 的 ref 大半画面是背着它的人和天空,
    整图特征被污染);某视角检不到类别词就整图兜底并告警——宁可退化也不丢视角。
    """
    if subject in ctx["ref_bank"]:
        return ctx["ref_bank"][subject]
    from distill.gen_data import load_subject_images
    fnames = load_subject_images(ctx["data_dir"], [subject])[subject]
    phrase = _class_word(ctx, subject)
    crops, n_fallback = [], 0
    for fname in fnames:
        img = Image.open(os.path.join(ctx["data_dir"], subject, fname)).convert("RGB")
        boxes = detect_boxes(ctx, img, phrase)
        if boxes:
            crops.append(crop_pad(img, boxes[0][0]))
        else:
            crops.append(img)
            n_fallback += 1
    if n_fallback:
        print(f"⚠️ ref {subject}:{n_fallback}/{len(fnames)} 个视角检不到 '{phrase}',"
              "该视角整图兜底(特征可能被背景稀释)", flush=True)
    ctx["ref_bank"][subject] = extract_unit_feats(crops, ctx)
    return ctx["ref_bank"][subject]


# ══════════════════════════════════════════════════════════ 打分

# 每块先攒完检测框再一次性提 DINO 特征。128 条 × 每主体 ≤6 框的 crop 常驻内存,
# 上限约几百 MB;整跑 8000 条也不会把 crop 全堆在内存里。
_CHUNK = 128


def score_records(records, out_dir, ctx):
    """给每条样本算逐主体分(presence+identity)与 min_ref_sim,写回 meta。

    meta 写入:
      dino_sims[j]  第 j 个主体的分(与 meta.subjects / image_paths 同序);检不到框 = 0.0
      min_ref_sim   min(dino_sims)
      det[j]        {"n_boxes", "det_score", "best_box"}:best_box 是 identity 胜出框,
                    画进 board 供人工核对"框对了吗"
      metric        METRIC_TAG,版本标记
    返回 (scored, skipped):生成图缺失的样本进 skipped(不参与后续过滤)。
    """
    usable, skipped = [], []
    for r in records:
        gen = resolve(out_dir, r["image_tgt_path"])
        if not os.path.exists(gen):
            skipped.append({"image_tgt_path": r["image_tgt_path"], "missing": [gen]})
            continue
        r["_gen"] = gen
        usable.append(r)
    if not usable:
        raise SystemExit("❌ 没有一条样本有生成图——manifest 与图目录对不上?检查 --out_dir")

    print(f"打分({METRIC_TAG}):{len(usable)} 条可用,{len(skipped)} 条缺生成图跳过", flush=True)
    for lo in range(0, len(usable), _CHUNK):
        chunk = usable[lo:lo + _CHUNK]
        crops, owners = [], []   # owners[i] = (record, subj_idx, box)
        for r in chunk:
            img = Image.open(r["_gen"]).convert("RGB")
            subs = r["meta"]["subjects"]
            r["meta"]["dino_sims"] = [0.0] * len(subs)
            r["meta"]["det"] = []
            for j, s in enumerate(subs):
                boxes = detect_boxes(ctx, img, _class_word(ctx, s))
                r["meta"]["det"].append({
                    "n_boxes": len(boxes),
                    "det_score": round(boxes[0][1], 3) if boxes else 0.0,
                    "best_box": None,
                })
                for box, _score in boxes:
                    crops.append(crop_pad(img, box))
                    owners.append((r, j, box))
        if crops:
            vecs = extract_unit_feats(crops, ctx)
            for v, (r, j, box) in zip(vecs, owners):
                sim = float(np.max(_ref_vecs(ctx, r["meta"]["subjects"][j]) @ v))
                if sim > r["meta"]["dino_sims"][j] or r["meta"]["det"][j]["best_box"] is None:
                    r["meta"]["dino_sims"][j] = sim
                    r["meta"]["det"][j]["best_box"] = [round(x, 1) for x in box]
        for r in chunk:
            r["meta"]["min_ref_sim"] = min(r["meta"]["dino_sims"])
            r["meta"]["metric"] = METRIC_TAG
            del r["_gen"]
        print(f"  … {min(lo + _CHUNK, len(usable))}/{len(usable)}", flush=True)
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
    """min_ref_sim 的分位数表(附 mean 聚合对照,凸显 min 抓掉主体的价值)。"""
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
    print("-" * 72)
    print("对照:全体在 mean 聚合下的同分位(官方口径,会掩盖掉主体)")
    mean_arr = np.array([float(np.mean(r["meta"]["dino_sims"])) for r in records])
    print(f"{'mean':<8}{len(records):>6} " + " ".join(f"{np.percentile(mean_arr, p):.3f}" for p in ps))
    # 掉主体率:presence 级信号单独报一行,这是 v2 才有的直接读数
    n_drop = sum(1 for r in records if any(d["n_boxes"] == 0 for d in r["meta"].get("det", [])))
    print("-" * 72)
    print(f"presence:至少一个主体检不到框(直接判掉)的样本 {n_drop}/{len(records)}"
          f" ({n_drop / len(records):.1%})")
    print("=" * 72)


def annotate_det(img, det):
    """把 identity 胜出框画到生成图副本上(红框 + 主体序号),board 里人工核对框位。"""
    if not det:
        return img
    out = img.copy()
    d = ImageDraw.Draw(out)
    for j, item in enumerate(det):
        box = item.get("best_box")
        if box:
            d.rectangle(box, outline=(255, 0, 0), width=4)
            d.text((box[0] + 5, box[1] + 3), str(j + 1), fill=(255, 0, 0))
    return out


def _sims_label(meta):
    """'0.00! 0.71 0.63' —— ! 标记该主体一个框都没检到(presence 判掉)。"""
    return " ".join(
        f"{s:.2f}" + ("!" if d["n_boxes"] == 0 else "")
        for s, d in zip(meta["dino_sims"], meta.get("det", [{}] * len(meta["dino_sims"]))))


def make_board(records, out_dir, n, save_path):
    """按 min_ref_sim 排序、等间隔抽 n 条拼成总览图,人工据此定阈值。

    等间隔(而非只取最低 n 张)是为了让整条分数谱都呈现出来——既看得到"多低算掉主体",
    也看得到"多高算干净",阈值切在哪一目了然。复用 multibanana_eval/board.py。
    生成图上叠了 identity 胜出框,顺带核对检测器有没有框错东西。
    """
    from multibanana_eval.board import build_row, stack_board

    ordered = sorted(records, key=lambda r: r["meta"]["min_ref_sim"])
    picks = np.linspace(0, len(ordered) - 1, min(n, len(ordered))).round().astype(int)
    rows = []
    for i in sorted(set(picks.tolist())):
        r = ordered[i]
        refs = [Image.open(resolve(out_dir, p)).convert("RGB") for p in r["image_paths"]]
        gen = annotate_det(
            Image.open(resolve(out_dir, r["image_tgt_path"])).convert("RGB"),
            r["meta"].get("det"))
        title = (f"{os.path.basename(r['image_tgt_path'])}  "
                 f"min={r['meta']['min_ref_sim']:.3f}  sims=[{_sims_label(r['meta'])}]")
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
    """分数在且是当前口径。v1 整图分数(无 metric 标记)在这里自动判为过期。"""
    return bool(records) and all(
        r.get("meta", {}).get("metric") == METRIC_TAG and "min_ref_sim" in r["meta"]
        for r in records)


def main():
    from distill.gen_data import DATA_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--scored", default=DEFAULT_SCORED, help="打分结果缓存(读/写)")
    p.add_argument("--out_dir", default=DEFAULT_OUT, help="图路径的基准目录")
    p.add_argument("--data_dir", default=DATA_DIR, help="dreambooth 参考图目录")
    p.add_argument("--calibrate", action="store_true",
                   help="出分位数表 + 拼图(不给 --threshold 时的默认行为)")
    p.add_argument("--threshold", type=float, default=None,
                   help="保留 min_ref_sim >= 阈值的样本,产出 manifest_filtered.json")
    p.add_argument("--board_n", type=int, default=40)
    p.add_argument("--board_out", default=None)
    p.add_argument("--filtered_out", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--box_thr", type=float, default=0.22,
                   help="GroundingDino 框置信度阈值;调高漏检更多(误杀好样本),调低碎框更多")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force_rescore", action="store_true",
                   help="即使 scored 缓存已存在且口径一致也重新打分")
    args = p.parse_args()

    # ---------- 拿到带分数的记录:优先复用缓存,否则跑 GPU 打分 ----------
    records = None
    if not args.force_rescore and os.path.exists(args.scored):
        cached = load_manifest(args.scored)
        if have_scores(cached):
            print(f"复用打分缓存 {args.scored}({len(cached)} 条,口径 {METRIC_TAG})。"
                  f"要重打分加 --force_rescore", flush=True)
            records = cached
        else:
            print(f"缓存 {args.scored} 缺分或为旧口径(v1 整图),自动重新打分", flush=True)

    if records is None:
        raw = load_manifest(args.manifest)
        ctx = make_ctx(args.device, args.data_dir, args.box_thr, args.batch_size)
        records, skipped = score_records(raw, args.out_dir, ctx)
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
