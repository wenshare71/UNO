"""用**合成 npz** 端到端跑 plot_attn.py,不需要 GPU / torch / 真实推理(只要 numpy + pillow)。

    python distill/test_plot_attn.py

合成数据故意植入三种可识别形态,报告若画对了就应该一眼看出来:
  official_full     两个 ref 份额均衡           → 失衡比 ≈ 1
  ours_kv_pre       ref2 份额在 late 段塌掉      → 失衡比 << 1,且曲线后段下坠
  ours_iso_nocache  与 pre 几乎一致(缓存无罪的形态)
  ours_kv_post4000  ref2 恢复                   → 失衡比回到 ≈ 1
再给 S2 样本植入**复制签名**:ref2 全暗,而 ref1 热力图有两处热区。

D04 起 curve 多了 head 维度,于是**另起一个样本** S1_022 植入第四种形态:
**只有 head 5/17 塌,其余 22 个头纹丝不动**。这正是 D04 要验证的那个假设的合成版——
全 head 平均后失衡比看着仍然"均衡"(>0.8,总览表会判它没问题),
而那两个头自己已经塌到 0.2 以下。报告的 per-head 节若写对了,这两个头必须浮到
Δ失衡比 榜首;若浮不上来,就是"拆了 head 但没读出来",和没拆一样。

(S1_000 / S2 / S4 维持 D04 之前的**全 head 一起塌**形态,旧断言原样跑在它们上面。)

同一份数据还会另存一份 **D04 之前格式**(curve 先在 head 维度平均掉)。
两份跑出来的 head 平均视图必须逐值相同——这是"D04 只增不改"的实证:
既有报告的所有数字不能因为多存了 head 维度而漂移。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
OUT = tempfile.mkdtemp(prefix="fake_attn_diag_")
OUT_OLD = tempfile.mkdtemp(prefix="fake_attn_diag_v3_")  # D04 之前的 3 维 curve

STEPS, BLOCKS = 25, 57
N_HEAD = 24
ID_HEADS = (5, 17)  # 植入的"身份头":ref2 份额更高、塌得更狠
SPATIAL_BLOCKS, SPATIAL_STEPS = (4, 18, 37, 52), (0, 6, 12, 18, 24)
GRID = 32

# (task_id, subjects, prompt, seed, ref2 塌陷模式)
#   all_head —— 24 个头一起塌,D04 之前就看得见
#   id_head  —— 只有 ID_HEADS 塌,head 平均后基本看不出来
TASKS = [
    ("S1_000_s0", ["backpack_dog", "bear_plushie"], "a backpack and a stuffed animal in the jungle", 3500000, "all_head"),
    ("S1_022_s0", ["berry_bowl", "grey_sloth_plushie"], "a bowl and a stuffed animal in the jungle", 3520000, "id_head"),
    ("S2_000_s0", ["bear_plushie", "grey_sloth_plushie"], "a stuffed animal and a stuffed animal in the jungle", 3600000, "all_head"),
    ("S4_000_s0", ["backpack_dog", "bear_plushie", "berry_bowl"], "a backpack, a stuffed animal and a bowl in the jungle", 3800000, "all_head"),
]
VARIANTS = ["official_full", "ours_kv_pre", "ours_iso_nocache", "ours_kv_post4000"]
# 每个变体给 ref2 一个 late 段衰减因子
DECAY = {"official_full": 1.0, "ours_kv_pre": 0.12,
         "ours_iso_nocache": 0.14, "ours_kv_post4000": 0.85}

rng = np.random.default_rng(0)


def blob(cx, cy, s=5.0, amp=1.0):
    y, x = np.mgrid[0:GRID, 0:GRID]
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))


IS_ID = np.zeros(N_HEAD, dtype=bool)
IS_ID[list(ID_HEADS)] = True

for tid, subs, prompt, seed, mode in TASKS:
    n_ref = len(subs)
    seg_names = ["txt", "img"] + [f"ref{i + 1}" for i in range(n_ref)]
    n_seg = len(seg_names)
    # id_head 模式:身份头 ref2 基线略高(条带上看得见),且**只有它们**会塌
    gain = np.where(IS_ID, 1.6, 1.0) if mode == "id_head" else np.ones(N_HEAD)
    for v in VARIANTS:
        curve = np.zeros((STEPS, BLOCKS, N_HEAD, n_seg), dtype=np.float32)
        for s in range(STEPS):
            for b in range(BLOCKS):
                # ref2 的衰减**只由变体决定**:DECAY=1.0 的 teacher 应当全程平坦。
                # (最初把 late 衰减写成了与变体无关,结果 teacher 的失衡比也掉到 0.6,
                #  被下面的断言抓出来——fixture 自己的 bug,不是 plot_attn 的。)
                shape = (1.0 if b < 38
                         else 1.0 - (1.0 - DECAY[v]) * (s / (STEPS - 1)))
                raw = np.empty((N_HEAD, n_seg), dtype=np.float64)
                raw[:, 0] = 0.30 + 0.02 * rng.standard_normal(N_HEAD)   # txt
                raw[:, 1] = 0.45 + 0.02 * rng.standard_normal(N_HEAD)   # img
                raw[:, 2] = 0.14 + 0.01 * rng.standard_normal(N_HEAD)   # ref1
                # ref2 会塌。all_head:24 个头同步塌(与 D04 之前的 fixture 等价,
                # 旧断言继续有效);id_head:只有身份头塌,其余头恒定不动。
                hs = np.where(IS_ID, shape, 1.0) if mode == "id_head" else shape
                raw[:, 3] = 0.13 * gain * hs + 0.005 * rng.standard_normal(N_HEAD)
                for j in range(4, n_seg):
                    raw[:, j] = 0.10 + 0.01 * rng.standard_normal(N_HEAD)
                raw = np.clip(raw, 1e-5, None)
                curve[s, b] = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

        spatial = {}
        dup = (tid.startswith("S2") and v in ("ours_kv_pre", "ours_iso_nocache"))
        for st in SPATIAL_STEPS:
            for bl in SPATIAL_BLOCKS:
                m = np.zeros((GRID, GRID, n_seg), dtype=np.float32)
                m[:, :, 0] = 0.30 / (GRID * GRID) * GRID * GRID / (GRID * GRID)
                m[:, :, 1] = 0.45 / (GRID * GRID) * GRID * GRID / (GRID * GRID)
                # ref1:左侧热区;复制形态下右侧再来一处
                r1 = blob(10, 16, amp=0.30)
                if dup:
                    r1 = r1 + blob(22, 16, amp=0.28)
                m[:, :, 2] = r1
                # ref2:正常时右侧热区;丢失/复制时整体压暗
                amp2 = 0.30 * (DECAY[v] if bl >= 38 else 1.0)
                m[:, :, 3] = blob(22, 16, amp=amp2) if not dup else np.full((GRID, GRID), 0.004)
                for j in range(4, n_seg):
                    m[:, :, j] = blob(16, 24, amp=0.22)
                spatial[(st, bl)] = m

        payload = {
            "curve": curve,
            "seg_names": np.array(seg_names),
            "spatial_keys": np.array(sorted(spatial), dtype=np.int32).reshape(-1, 2),
            "layout_txt_len": np.int32(512),
            "layout_img_len": np.int32(1024),
            "layout_ref_lens": np.array([1024] * n_ref, dtype=np.int32),
            "layout_img_grid": np.array([32, 32], dtype=np.int32),
            "n_heads": np.int32(24),
            "n_double": np.int32(19), "n_single": np.int32(38),
            "q_len_by_step": np.array(
                [512 + 1024 + 1024 * n_ref] + [512 + 1024] * (STEPS - 1), dtype=np.int32)
            if "kv" in v else np.array([512 + 1024 + 1024 * n_ref] * STEPS, dtype=np.int32),
            "meta_task_id": np.array(tid), "meta_variant": np.array(v),
            "meta_prompt": np.array(prompt), "meta_seed": np.array(seed),
            "meta_subjects": np.array(subs),
        }
        for (st, bl), arr in sorted(spatial.items()):
            payload[f"spatial_{st:03d}_{bl:03d}"] = arr
        img = Image.new("RGB", (512, 512), (40 + 60 * VARIANTS.index(v), 90, 140))
        for d, cv in ((OUT, curve), (OUT_OLD, curve.mean(axis=2))):
            stem = os.path.join(d, f"{tid}__{v}")
            np.savez_compressed(stem + ".npz", **{**payload, "curve": cv})
            img.save(stem + ".png")  # 假出图:纯色块,只为验证内联缩略图这条路通

n_npz = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
print(f"合成 {n_npz} 个 npz → {OUT}(4 维 curve)")
print(f"          另一份 → {OUT_OLD}(3 维 curve,D04 之前格式)")


def render(d):
    r = subprocess.run([sys.executable, os.path.join(REPO, "distill", "plot_attn.py"),
                        "--save_path", d], capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="")
    if r.returncode != 0:
        sys.exit(r.returncode)


render(OUT)
render(OUT_OLD)

# ---- 校验产物 ----
import json  # noqa: E402

fails = []
rep = os.path.join(OUT, "report.html")
doc = open(rep, encoding="utf-8").read()
for name, cond in [
    ("HTML 非空且有 doctype", doc.startswith("<!doctype html>") and len(doc) > 50_000),
    ("内联了热力图 PNG", doc.count("data:image/png;base64,") >= 12),
    ("内联了出图 JPEG", doc.count("data:image/jpeg;base64,") >= 12),
    ("画了 SVG 曲线板", doc.count("<polyline") >= 3 * 2 * 4),
    ("印了解读红线", "≠" in doc and "K/V" in doc),
    ("每个样本都有小节", all(t[0] in doc for t in TASKS)),
    ("标了 write/read 切换", "step0 query 长" in doc),
    ("无未转义的 numpy repr 泄漏", "array(" not in doc and "dtype=" not in doc),
    ("有 per-head 节(D04)", "per-head 分辨" in doc and "Δ失衡比" in doc),
    ("画了 per-head 条带", doc.count("per-head share") >= len(VARIANTS)),
    ("per-head 节没退化成'无数据'", "本节无数据" not in doc),
]:
    print(f"  {'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)

summ = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
by = {(s["task_id"], s["variant"]): s for s in summ}
imb = {v: by[("S1_000_s0", v)]["imbalance_late_steps"] for v in VARIANTS}
imb_e = {v: by[("S1_000_s0", v)]["imbalance_early_steps"] for v in VARIANTS}
print(f"  失衡比(晚 1/3):{ {k: round(v, 3) for k, v in imb.items()} }")
print(f"  失衡比(早 1/3):{ {k: round(v, 3) for k, v in imb_e.items()} }")
for name, cond in [
    (f"summary 覆盖全部 {len(TASKS) * len(VARIANTS)} 条",
     len(summ) == len(TASKS) * len(VARIANTS)),
    ("teacher 失衡比 ≈ 1", imb["official_full"] > 0.85),
    ("pre 失衡比明显偏低", imb["ours_kv_pre"] < 0.5),
    ("nocache 与 pre 接近(缓存无罪形态)",
     abs(imb["ours_iso_nocache"] - imb["ours_kv_pre"]) < 0.12),
    ("post 失衡比回升", imb["ours_kv_post4000"] > imb["ours_kv_pre"] * 2),
    ("3-ref 样本记了 3 个 ref 份额",
     len(by[("S4_000_s0", "official_full")]["late_block_ref_share_late_steps"]) == 3),
    ("早段全部均衡(合成数据的衰减只加在后段)", all(v > 0.8 for v in imb_e.values())),
    ("报告标出了形态判定", "细化阶段衰减" in doc),
]:
    print(f"  {'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)

# ---- D04:S1_022 上,全 head 平均看不见的塌陷,per-head 必须看得见 ----
ID_TASK = "S1_022_s0"
ph = {k: np.asarray(s["per_head_ref_share_late_steps"]) for k, s in by.items()
      if "per_head_ref_share_late_steps" in s}
imb_h = {k: np.asarray(s["per_head_imbalance_late_steps"]) for k, s in by.items()
         if "per_head_imbalance_late_steps" in s}
teach = ph.get((ID_TASK, "official_full"))
d_imb = imb_h[(ID_TASK, "ours_kv_post4000")] - imb_h[(ID_TASK, "ours_kv_pre")]
top2 = np.argsort(-np.abs(d_imb))[:2].tolist()
avg_view = by[(ID_TASK, "ours_kv_pre")]["imbalance_late_steps"]
print(f"  {ID_TASK} pre:全 head 平均失衡比 {avg_view:.3f}(看着均衡),"
      f"身份头 {list(ID_HEADS)} 自己 {imb_h[(ID_TASK, 'ours_kv_pre')][list(ID_HEADS)].round(3)}")
print(f"  Δ失衡比 top2 head:{top2}(植入的是 {list(ID_HEADS)}),"
      f"最大 {np.abs(d_imb).max():.3f} vs 全 head 平均 {abs(float(d_imb.mean())):.3f}")
for name, cond in [
    ("summary 每条都带 per-head 字段", len(ph) == len(summ) and len(imb_h) == len(summ)),
    ("per-head 形状 = (24, n_ref)", teach is not None and teach.shape == (N_HEAD, 2)),
    ("身份头的 ref2 份额最高", teach is not None and int(np.argmax(teach[:, 1])) in ID_HEADS),
    # 前提:这个样本上全 head 平均**确实**看不出问题(否则下一条就是白证的)
    ("全 head 平均把塌陷藏住了(失衡比仍 >0.8)", avg_view > 0.8),
    ("身份头自己已经塌到 <0.4",
     float(imb_h[(ID_TASK, "ours_kv_pre")][list(ID_HEADS)].max()) < 0.4),
    ("Δ失衡比 榜首就是植入的身份头", set(top2) == set(ID_HEADS)),
    # 单头的 Δ 必须显著大于全 head 平均——这正是 D04 想验证的"被摊薄"效应本身
    ("单头 Δ 明显大于全 head 平均 Δ",
     np.abs(d_imb).max() > 4.0 * abs(float(d_imb.mean()))),
]:
    print(f"  {'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)

# ---- D04 只增不改:head 平均视图与老格式逐值相同 ----
old = json.load(open(os.path.join(OUT_OLD, "summary.json"), encoding="utf-8"))
by_old = {(s["task_id"], s["variant"]): s for s in old}
shared = ["imbalance_early_steps", "imbalance_late_steps",
          "late_block_ref_share_early_steps", "late_block_ref_share_late_steps"]
drift = max(float(np.abs(np.asarray(by[k][f]) - np.asarray(by_old[k][f])).max())
            for k in by for f in shared)
doc_old = open(os.path.join(OUT_OLD, "report.html"), encoding="utf-8").read()
for name, cond in [
    ("老格式(3 维 curve)仍能出报告", len(doc_old) > 50_000),
    ("老格式下 per-head 节自报无数据", "本节无数据" in doc_old),
    ("head 平均视图与老格式逐值相同", drift < 1e-9),
]:
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  max drift {drift:.1e}"
                                                if "逐值" in name else ""))
    if not cond:
        fails.append(name)

print("=" * 58)
if fails:
    print(f"❌ {len(fails)} 项失败:{fails}")
    sys.exit(1)
print(f"✓ 全部通过  报告 {os.path.getsize(rep) / 1024:.0f} KB")
