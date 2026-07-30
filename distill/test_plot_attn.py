"""用**合成 npz** 端到端跑 plot_attn.py,不需要 GPU / torch / 真实推理(只要 numpy + pillow)。

    python distill/test_plot_attn.py

合成数据故意植入三种可识别形态,报告若画对了就应该一眼看出来:
  official_full     两个 ref 份额均衡           → 失衡比 ≈ 1
  ours_kv_pre       ref2 份额在 late 段塌掉      → 失衡比 << 1,且曲线后段下坠
  ours_iso_nocache  与 pre 几乎一致(缓存无罪的形态)
  ours_kv_post4000  ref2 恢复                   → 失衡比回到 ≈ 1
再给 S2 样本植入**复制签名**:ref2 全暗,而 ref1 热力图有两处热区。
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

STEPS, BLOCKS = 25, 57
SPATIAL_BLOCKS, SPATIAL_STEPS = (4, 18, 37, 52), (0, 6, 12, 18, 24)
GRID = 32

TASKS = [
    ("S1_000_s0", ["backpack_dog", "bear_plushie"], "a backpack and a stuffed animal in the jungle", 3500000),
    ("S2_000_s0", ["bear_plushie", "grey_sloth_plushie"], "a stuffed animal and a stuffed animal in the jungle", 3600000),
    ("S4_000_s0", ["backpack_dog", "bear_plushie", "berry_bowl"], "a backpack, a stuffed animal and a bowl in the jungle", 3800000),
]
VARIANTS = ["official_full", "ours_kv_pre", "ours_iso_nocache", "ours_kv_post4000"]
# 每个变体给 ref2 一个 late 段衰减因子
DECAY = {"official_full": 1.0, "ours_kv_pre": 0.12,
         "ours_iso_nocache": 0.14, "ours_kv_post4000": 0.85}

rng = np.random.default_rng(0)


def blob(cx, cy, s=5.0, amp=1.0):
    y, x = np.mgrid[0:GRID, 0:GRID]
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))


for tid, subs, prompt, seed in TASKS:
    n_ref = len(subs)
    seg_names = ["txt", "img"] + [f"ref{i + 1}" for i in range(n_ref)]
    n_seg = len(seg_names)
    for v in VARIANTS:
        curve = np.zeros((STEPS, BLOCKS, n_seg), dtype=np.float32)
        for s in range(STEPS):
            for b in range(BLOCKS):
                # ref2 的衰减**只由变体决定**:DECAY=1.0 的 teacher 应当全程平坦。
                # (最初把 late 衰减写成了与变体无关,结果 teacher 的失衡比也掉到 0.6,
                #  被下面的断言抓出来——fixture 自己的 bug,不是 plot_attn 的。)
                shape = (1.0 if b < 38
                         else 1.0 - (1.0 - DECAY[v]) * (s / (STEPS - 1)))
                raw = np.empty(n_seg, dtype=np.float64)
                raw[0] = 0.30 + 0.02 * rng.standard_normal()          # txt
                raw[1] = 0.45 + 0.02 * rng.standard_normal()          # img
                raw[2] = 0.14 + 0.01 * rng.standard_normal()          # ref1
                raw[3] = 0.13 * shape + 0.005 * rng.standard_normal()  # ref2 会塌
                for j in range(4, n_seg):
                    raw[j] = 0.10 + 0.01 * rng.standard_normal()
                raw = np.clip(raw, 1e-5, None)
                curve[s, b] = (raw / raw.sum()).astype(np.float32)

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
        stem = os.path.join(OUT, f"{tid}__{v}")
        np.savez_compressed(stem + ".npz", **payload)
        # 假出图:纯色块,只为验证内联缩略图这条路通
        Image.new("RGB", (512, 512), (40 + 60 * VARIANTS.index(v), 90, 140)).save(stem + ".png")

n_npz = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
print(f"合成 {n_npz} 个 npz → {OUT}")

r = subprocess.run([sys.executable, os.path.join(REPO, "distill", "plot_attn.py"),
                    "--save_path", OUT], capture_output=True, text=True)
print(r.stdout, r.stderr, sep="")
if r.returncode != 0:
    sys.exit(r.returncode)

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
    ("三个样本都有小节", all(t in doc for t, _, _, _ in TASKS)),
    ("标了 write/read 切换", "step0 query 长" in doc),
    ("无未转义的 numpy repr 泄漏", "array(" not in doc and "dtype=" not in doc),
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
    ("summary 覆盖全部 12 条", len(summ) == 12),
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

print("=" * 58)
if fails:
    print(f"❌ {len(fails)} 项失败:{fails}")
    sys.exit(1)
print(f"✓ 全部通过  报告 {os.path.getsize(rep) / 1024:.0f} KB")
