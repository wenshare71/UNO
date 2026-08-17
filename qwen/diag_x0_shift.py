"""诊断:训练时对齐的那些点,和部署时真正经过的点,差多远。

**不训练、不出图、不进盲评。** 只量残差,产物是一张表。

── 为什么做这个 ────────────────────────────────────────────────────────

§9 判读给出的 student 特有退化集中在**小尺度高频细节**:罐头上的汉字更模糊、
闹钟只保住颜色保不住数字、蓝莓碗主体丢失更重。低频(颜色、轮廓)基本保得住。
蜡烛不算 —— 它在 `GOAL.md` §5.1 的豁免集里,teacher 自己 3/6 就坏。

被检验的机制链:

    manifest 的 x₀ 是 UNO 的 512² 产物,`train_iso.encode_sample` 把它上采样到 1024²
      ⇒ 它的高频段是插值伪影,不是真实纹理
      ⇒ 低 σ 时 x_t ≈ x₀,student 几乎没在「带真实高频细节的图像」上对齐过
      ⇒ 汉字/数字这类靠高频承载的身份特征丢失

注意速度匹配里 **x₀ 不是拟合目标**(见 `train_iso` 模块头),它只决定在哪些点上
对齐 teacher。所以这里量的不是「图好不好」,是「点选得偏不偏」。

── 单变量设计 ──────────────────────────────────────────────────────────

A/B 两组用**同一条 (prompt, refs)**、同一份 image_latents、同一份 embeds,
只差 x_t 怎么来:

    A `train_dist`  x_t = (1-σ)·x₀_UNO + σ·ε      ← 训练时喂的
    B `on_policy`   x_t = teacher 40 步采样(Q1 口径)在 σ 处的实际 latent  ← 部署时经过的

ε 取 B 组轨迹的起点(σ₀=1.0 处 latent 就是初始噪声)⇒ **σ₀ 那一行两组逐位相同**。
这是本脚本的机制自检,角色同 `diag_kv.py` 的探针 A:σ₀ 不等就是装置坏了
(轨迹取错位、encode_sample 与 pipeline 的 image_latents 对不上……),后面的读数全部作废。

── 读数(全部无量纲,不引外部阈值)────────────────────────────────────

    rel_pre (σ) = ‖v_iso_untrained − v_full‖ / ‖v_full‖   ← mask 造成的原始损伤
    rel_post(σ) = ‖v_iso_trained   − v_full‖ / ‖v_full‖
    修复率  (σ) = 1 − rel_post / rel_pre                   ← 训练在这个 σ 上补回了多少

`rel_pre` 是**批内标尺**:它就是 iso_pre 臂,同一批、同一条样本上量的,
所以「修复率」不需要任何外部阈值就能读。残差再按块内高/低频拆一次,
直接对上 §9 那批失效模式。

判读(A/B 相互比较,脚本不自动下结论):
  · 两组修复率曲线基本重合         ⇒ 对齐点没选偏,换 x₀ 重训不会有收益,假说否掉;
  · B 组在低 σ 端修复率显著低于 A  ⇒ 假说立住,并指出该补哪一段 σ;
  · 残差的高频相对误差 B ≫ A       ⇒ 直接对上罐头汉字/闹钟数字。

顺带量一条**前提**:x₀_UNO 与 on-policy 末端 latent 的高频能量占比。
两者差不多的话,上面整条链从第一步就断了,后面的读数不用解释。

── 用法 ────────────────────────────────────────────────────────────────

    # 1. 单卡,载 VL,把选中样本的 正/负 prompt_embeds 落盘(负样本 Q1 口径是 " ")
    QWEN_WEIGHTS=... python qwen/diag_x0_shift.py embeds --n 24

    # 2. 8 卡各一片(每片 ~3 条样本,约 15 min)
    QWEN_WEIGHTS=... python qwen/diag_x0_shift.py run --lora <ckpt> \
        --n 24 --shard_idx $i --num_shards 8

    # 3. 合并 + 判读表
    python qwen/diag_x0_shift.py merge

**只新建本文件,既有 `.py` 一个字不动(R0)。**
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE, os.path.join(_REPO_ROOT, "distill")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ------------------------------------------------------------------ Q1 口径
# 与 `infer_iso.py` / `train_iso.py` 逐字相同。改任何一个,这批读数就与评测对不上。
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = " "
RESOLUTION = 1024

OUT_DIR = os.path.join(_REPO_ROOT, "output/diag_x0_shift")
EMBED_DIR = os.path.join(OUT_DIR, "embeds")

GROUPS = ("train_dist", "on_policy")

# σ 分段只为了把 40 行压成能读的 4 行,**不是判据**。判读靠 A/B 相互比。
SIGMA_BANDS = [
    (0.75, 1.01, "高 σ [0.75,1.0]"),
    (0.50, 0.75, "中高 [0.50,0.75)"),
    (0.25, 0.50, "中低 [0.25,0.50)"),
    (0.00, 0.25, "低 σ  [0.00,0.25)"),
]


# ==================================================================== 纯函数
# 这一段不 import torch —— 只用张量的方法,好在本地无 GPU 环境干跑。

def band_split(v):
    """把 packed latent 的最后一维拆成 块内均值(低频)/ 偏差(高频),形状不变。

    `_pack_latents`(pipeline L397-401):

        view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C*4)

    permute 之后维序是 (B, H//2, W//2, C, 2, 2),所以最后那一维 `C*4` 的排布是
    **(C, 2, 2)**。reshape 成 (..., C, 4) 之后,末轴那个 4 就是一个 2×2 空间块 ——
    块内均值 = 最低频,减掉它 = 最高频(Nyquist 附近)。不用 unpack、不用 FFT。
    """
    b, ln, c4 = v.shape
    if c4 % 4:
        raise ValueError(f"packed latent 末维应是 4 的倍数(C×4),实际 {c4}")
    x = v.reshape(b, ln, c4 // 4, 4)
    lo = x.mean(dim=-1, keepdim=True)
    return lo.expand_as(x).reshape(b, ln, c4), (x - lo).reshape(b, ln, c4)


def _norm(t) -> float:
    return float(t.float().norm().item())


def rel(a, b) -> float:
    """‖a−b‖ / ‖b‖。无量纲,可跨 σ、跨样本比。"""
    d = _norm(a - b)
    n = _norm(b)
    return d / n if n > 0 else float("nan")


def hf_ratio(v) -> float:
    """高频能量占比 ‖高频‖ / ‖全量‖。前提检查用(x₀ 到底缺不缺高频)。"""
    _, hi = band_split(v)
    n = _norm(v)
    return _norm(hi) / n if n > 0 else float("nan")


def residual_stats(v_hat, v_full) -> dict:
    """一个 (x_t, σ) 点上的全部读数。"""
    lo_f, hi_f = band_split(v_full)
    lo_h, hi_h = band_split(v_hat)
    d_lo, d_hi = _norm(lo_h - lo_f), _norm(hi_h - hi_f)
    n_all, n_lo, n_hi = _norm(v_full), _norm(lo_f), _norm(hi_f)
    d_all = _norm(v_hat - v_full)
    return {
        "rel": d_all / n_all if n_all > 0 else float("nan"),
        # 低/高频**各自**的相对误差 —— 这两个数分开看才对得上「颜色保住了、数字没保住」
        "rel_lo": d_lo / n_lo if n_lo > 0 else float("nan"),
        "rel_hi": d_hi / n_hi if n_hi > 0 else float("nan"),
        # 残差里高频占多少(与 teacher 自己的高频占比对照才有意义)
        "hf_share": d_hi / d_all if d_all > 0 else float("nan"),
    }


def pick_samples(items: list[dict], n: int, seed: int) -> list[dict]:
    """按 n_refs 分层抽样。

    manifest 是**按 n_refs 排序**的(1-ref 1000 / 2-ref 4000 / 3-ref 4000),
    直接取前 n 条会全是 1-ref,而 mask 的扰动强度恰好随 ref 段长度涨 —— 那样
    等于专挑扰动最小的样本量。
    """
    by_n: dict[int, list[dict]] = {}
    for it in items:
        by_n.setdefault(it["meta"]["n_refs"], []).append(it)
    rng = random.Random(seed)
    per = max(1, n // len(by_n))
    picked: list[dict] = []
    for k in sorted(by_n):
        picked += rng.sample(by_n[k], min(per, len(by_n[k])))
    picked.sort(key=lambda it: it["_idx"])       # 分片要稳定,不能跟着 rng 走
    return picked[:n]


def refs_hist(items: list[dict]) -> str:
    """`{1: 8, 2: 8, 3: 8}` 这样的一行摘要。分层抽样有没有生效,看这个。"""
    h: dict[int, int] = {}
    for it in items:
        n = it["meta"]["n_refs"]
        h[n] = h.get(n, 0) + 1
    return "{" + ", ".join(f"{k}-ref: {v}" for k, v in sorted(h.items())) + "}"


def shard_of(picked: list[dict], shard_idx: int, num_shards: int) -> list[dict]:
    if not (0 <= shard_idx < num_shards):
        raise SystemExit(f"❌ shard_idx {shard_idx} 超出 [0, {num_shards})")
    return [it for i, it in enumerate(picked) if i % num_shards == shard_idx]


# ==================================================================== 判读表

def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]                # 丢 nan
    return sum(xs) / len(xs) if xs else float("nan")


def selfcheck_sigma0(rows: list[dict]) -> list[str]:
    """σ₀ 两组用的是同一个 x_t(ε 取自轨迹起点)⇒ 读数必须逐位相同。

    返回不一致的样本说明。空列表 = 装置成立。**这是本脚本唯一的硬门禁**,
    而且它是构造出来的恒等式,不是猜的阈值。
    """
    bad = []
    by_sample: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["step"] != 0:
            continue
        by_sample.setdefault(str(r["idx"]), {})[r["group"]] = r
    for idx, g in sorted(by_sample.items()):
        if set(g) != set(GROUPS):
            bad.append(f"样本 {idx}:σ₀ 只有 {sorted(g)},两组不齐")
            continue
        a, b = g["train_dist"], g["on_policy"]
        for f in ("rel_pre", "rel_post"):
            if a[f] != b[f]:
                bad.append(f"样本 {idx}:σ₀ 的 {f} 两组不等 "
                           f"({a[f]:.6e} vs {b[f]:.6e})")
    return bad


def summarize(rows: list[dict]) -> list[dict]:
    """按 σ 分段 × 分组聚合。"""
    out = []
    for lo, hi, label in SIGMA_BANDS:
        for grp in GROUPS:
            sel = [r for r in rows if r["group"] == grp and lo <= r["sigma"] < hi]
            if not sel:
                continue
            m_pre, m_post = _mean([r["rel_pre"] for r in sel]), _mean([r["rel_post"] for r in sel])
            out.append({
                "band": label, "group": grp, "n": len(sel),
                "rel_pre": m_pre, "rel_post": m_post,
                "repair": 1 - m_post / m_pre if m_pre and m_pre == m_pre else float("nan"),
                "rel_lo": _mean([r["rel_lo_post"] for r in sel]),
                "rel_hi": _mean([r["rel_hi_post"] for r in sel]),
                "hf_share": _mean([r["hf_share_post"] for r in sel]),
            })
    return out


def print_report(rows: list[dict], premise: list[dict]) -> None:
    print("\n" + "=" * 86)
    print("前提:x₀ 到底缺不缺高频(高频能量占比 ‖hi‖/‖all‖)")
    print("-" * 86)
    print(f"{'样本':>8} {'n_refs':>7} {'x₀_UNO':>10} {'on-policy 末端':>16} {'比值':>8}")
    for p in premise:
        ratio = p["hf_x0"] / p["hf_traj"] if p["hf_traj"] else float("nan")
        print(f"{p['idx']:>8} {p['n_refs']:>7} {p['hf_x0']:>10.4f} "
              f"{p['hf_traj']:>16.4f} {ratio:>8.3f}")
    if premise:
        r = _mean([p["hf_x0"] / p["hf_traj"] for p in premise if p["hf_traj"]])
        print(f"{'平均比值':>8} {'':>7} {'':>10} {'':>16} {r:>8.3f}")
        print("  比值 ≈ 1 ⇒ x₀ 的高频并不比部署末端少,机制链第一步就不成立,下表不用解释。")

    print("\n" + "=" * 86)
    print("主表:同一条 (prompt, refs),只差 x_t 从哪来")
    print("-" * 86)
    print(f"{'σ 段':>18} {'组':>11} {'n':>5} {'rel_pre':>9} {'rel_post':>9} "
          f"{'修复率':>8} {'rel_lo':>8} {'rel_hi':>8} {'残差高频占比':>12}")
    for s in summarize(rows):
        print(f"{s['band']:>18} {s['group']:>11} {s['n']:>5} {s['rel_pre']:>9.4f} "
              f"{s['rel_post']:>9.4f} {s['repair']:>8.3f} {s['rel_lo']:>8.4f} "
              f"{s['rel_hi']:>8.4f} {s['hf_share']:>12.3f}")

    print("\n  rel_pre  = 未训练的隔离腿(=iso_pre)在这一段的相对残差,批内标尺")
    print("  修复率   = 1 − rel_post/rel_pre,训练在这一段补回了多少")
    print("  rel_lo/hi= 低频/高频**各自**的相对误差;闹钟「保住颜色保不住数字」对应 rel_hi ≫ rel_lo")
    print("  两组修复率重合 ⇒ 对齐点没偏,换 x₀ 重训无收益;B 在低 σ 显著更低 ⇒ 假说立住")
    print("  **判据由作者定,本脚本不下结论。**")
    print("=" * 86)

    bad = selfcheck_sigma0(rows)
    if bad:
        print("\n❌ σ₀ 机制自检未过 —— 两组在 σ₀ 用的是同一个 x_t,读数本应逐位相同:")
        for b in bad[:10]:
            print(f"   · {b}")
        print("   装置有问题,上面的表不要解读。")
    else:
        print(f"\n✅ σ₀ 机制自检:{len({r['idx'] for r in rows})} 条样本两组逐位相同。")


# ==================================================================== 预算 embeds

def cmd_embeds(args):
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image
    from train_iso import CONDITION_IMAGE_SIZE, calculate_dimensions, load_manifest, require_weights, resolve

    weights = require_weights()
    picked = pick_samples(load_manifest(), args.n, args.seed)
    os.makedirs(EMBED_DIR, exist_ok=True)
    print(f"[自检] 选中 {len(picked)} 条 | n_refs 分布 {refs_hist(picked)}", flush=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda")      # 只要 VL;transformer / vae 不上卡

    n_done = n_skip = 0
    for it in picked:
        pos_path = os.path.join(EMBED_DIR, f"{it['_idx']:06d}.pt")
        neg_path = os.path.join(EMBED_DIR, f"{it['_idx']:06d}.neg.pt")
        if os.path.exists(pos_path) and os.path.exists(neg_path):
            n_skip += 1
            continue
        refs = [Image.open(resolve(p)).convert("RGB") for p in it["image_paths"]]
        cond = []
        for img in refs:
            w, h = img.size
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
            cond.append(pipe.image_processor.resize(img, ch, cw))
        with torch.no_grad():
            pos, m_pos = pipe.encode_prompt(image=cond, prompt=it["prompt"], device="cuda")
            neg, m_neg = pipe.encode_prompt(image=cond, prompt=NEGATIVE_PROMPT, device="cuda")
        # batch=1 且无 padding 时 encode_prompt 把 mask 置 None(pipeline L326-327)。
        # 不是 None 说明这条有 padding,而训练侧全程按无 padding 处理 —— 停下来看。
        if m_pos is not None or m_neg is not None:
            raise SystemExit(f"❌ 第 {it['_idx']} 条出现 padding mask,与训练/推理路径不一致,停。")
        # 正样本按 `train_iso.encode_sample` 认的格式存(裸张量),这样它能原样复用
        torch.save(pos[0].to(torch.bfloat16).cpu(), pos_path)
        torch.save(neg[0].to(torch.bfloat16).cpu(), neg_path)
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} 条", flush=True)

    print(f"完成 {n_done} 条,跳过 {n_skip} 条 → {EMBED_DIR}")


# ==================================================================== 主体

def record_trajectory(pipe, torch, item, noise_len, device):
    """跑一遍 Q1 口径的 teacher 采样,记下每一步**进 transformer 的** x_t。

    为什么用 stock `pipe(...)` 而不是自己写 Euler 循环:去噪循环里除了 transformer
    还有 sigma 网格、true_cfg 之后的范数重标定、`scheduler.step` 三样决定轨迹落点,
    自己抄一遍抄错一个字,这条就不是部署轨迹了(理由同 `IsoPipelineHook` 的 docstring)。

    ⚠️ **必须在装 iso processor / 加 LoRA 之前调用。** `IsoAttnProcessor` 依赖
    `ctx.prepare()` 填的 mask,而 stock forward 不会调它 —— 装了之后再走 stock 路径
    会读到上一次的 mask,静默错。
    """
    from PIL import Image
    from train_iso import resolve

    refs = [Image.open(resolve(p)).convert("RGB") for p in item["image_paths"]]
    pos = torch.load(os.path.join(EMBED_DIR, f"{item['_idx']:06d}.pt"),
                     map_location=device).to(torch.bfloat16)[None]
    neg = torch.load(os.path.join(EMBED_DIR, f"{item['_idx']:06d}.neg.pt"),
                     map_location=device).to(torch.bfloat16)[None]

    xs: list = []
    orig = pipe.transformer.forward

    def rec(hidden_states, **kw):
        xs.append(hidden_states[:, :noise_len].detach().to("cpu").clone())
        return orig(hidden_states=hidden_states, **kw)

    pipe.transformer.forward = rec
    try:
        pipe(image=refs, prompt=None,
             prompt_embeds=pos, negative_prompt_embeds=neg,
             true_cfg_scale=TRUE_CFG_SCALE,
             num_inference_steps=NUM_INFERENCE_STEPS,
             height=RESOLUTION, width=RESOLUTION,
             generator=torch.Generator(device="cuda").manual_seed(item["meta"]["seed"]),
             output_type="latent")
    finally:
        pipe.transformer.forward = orig

    # cond 在前、uncond 在后(pipeline L816 / L830),每步两次 ⇒ 取偶数位就是 40 个 x_t
    if len(xs) != 2 * NUM_INFERENCE_STEPS:
        raise SystemExit(
            f"❌ 轨迹记到 {len(xs)} 次前向,期望 {2 * NUM_INFERENCE_STEPS}(40 步 × cond/uncond)。\n"
            f"   true_cfg 或步数被改了,或者 pipeline 版本不同 —— 这条轨迹不是 Q1 口径,停。")
    return xs[0::2]


def measure_sample(pipe, torch, runner, transformer, item, traj, sigmas, device) -> tuple:
    """一条样本上跑满 40 σ × 2 组 × 3 次前向,返回 (rows, premise_row)。"""
    from train_iso import encode_sample

    x0, image_latents, embeds, img_shapes = encode_sample(
        pipe, item, EMBED_DIR, item["_idx"], device)
    if x0.shape[1] != traj[0].shape[1]:
        raise SystemExit(
            f"❌ encode_sample 的噪声段 {x0.shape[1]} token 与轨迹的 {traj[0].shape[1]} 对不上。\n"
            f"   两边的 img_shapes 不一致,A/B 就不是单变量了,停。")

    # ε 取轨迹起点(σ₀=1.0 处 latent 就是初始噪声)⇒ σ₀ 两组逐位相同,成为机制自检
    eps = traj[0].to(device=device, dtype=x0.dtype)

    rows = []
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        xa = (1.0 - sigma) * x0 + sigma * eps
        xb = traj[i].to(device=device, dtype=x0.dtype)
        # 推理侧传的是 bf16(timesteps)/1000,即**舍入两次**;照抄那条路径(同 train_iso)
        ts = (sigma[None] * 1000).to(x0.dtype) / 1000

        for grp, x_t in (("train_dist", xa), ("on_policy", xb)):
            with torch.no_grad():
                transformer.disable_adapters()
                v_full = runner.teacher_forward(x_t, image_latents, embeds, None, ts, img_shapes)
                v_pre = runner.student_forward(x_t, image_latents, embeds, None, ts, img_shapes)
                transformer.enable_adapters()
                v_post = runner.student_forward(x_t, image_latents, embeds, None, ts, img_shapes)
            s_pre = residual_stats(v_pre, v_full)
            s_post = residual_stats(v_post, v_full)
            rows.append({
                "idx": item["_idx"], "n_refs": item["meta"]["n_refs"],
                "step": i, "sigma": float(sigma), "group": grp,
                "rel_pre": s_pre["rel"], "rel_post": s_post["rel"],
                "rel_lo_post": s_post["rel_lo"], "rel_hi_post": s_post["rel_hi"],
                "hf_share_post": s_post["hf_share"],
                "rel_lo_pre": s_pre["rel_lo"], "rel_hi_pre": s_pre["rel_hi"],
            })

    premise = {"idx": item["_idx"], "n_refs": item["meta"]["n_refs"],
               "hf_x0": hf_ratio(x0),
               # 末端 σ 已经 ~0.02,latent 基本干净,拿它代表「真实图的高频」
               "hf_traj": hf_ratio(traj[-1].to(device))}
    return rows, premise


def cmd_run(args):
    import time

    if args.dry_run:
        return _dry_run(args)

    import torch
    from diffusers import QwenImageEditPlusPipeline
    from infer_iso import apply_lora_ckpt
    from pipeline_iso import IsoRunner
    from train_iso import build_sigma_grid, load_manifest, require_weights

    weights = require_weights()
    picked = pick_samples(load_manifest(), args.n, args.seed)
    shard = shard_of(picked, args.shard_idx, args.num_shards)

    missing = [it["_idx"] for it in shard
               if not (os.path.exists(os.path.join(EMBED_DIR, f"{it['_idx']:06d}.pt"))
                       and os.path.exists(os.path.join(EMBED_DIR, f"{it['_idx']:06d}.neg.pt")))]
    if missing:
        raise SystemExit(f"❌ {len(missing)} 条样本的 embeds 不在 {EMBED_DIR},例如 {missing[:3]}。\n"
                         f"   先跑:python qwen/diag_x0_shift.py embeds --n {args.n}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[自检] 全量 {len(picked)} 条 | shard {args.shard_idx}/{args.num_shards} "
          f"共 {len(shard)} 条 | ckpt {args.lora}", flush=True)

    device = torch.device("cuda", 0)
    pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    pipe.text_encoder = None                  # embeds 已离线预算,7B VL 不上卡
    pipe.vae.to(device).eval()
    transformer = pipe.transformer.to(device)
    if not transformer.zero_cond_t:
        raise SystemExit("❌ zero_cond_t 未生效,隔离的地基不成立,停。")

    noise_len = (RESOLUTION // pipe.vae_scale_factor // 2) ** 2
    sigmas = build_sigma_grid(pipe, noise_len).to(device)

    # ── 第一趟:pristine pipeline,采 teacher 轨迹 ─────────────────────
    # 顺序不能反,见 record_trajectory 的 ⚠️
    t0 = time.perf_counter()
    trajs = {}
    for k, it in enumerate(shard):
        trajs[it["_idx"]] = record_trajectory(pipe, torch, it, noise_len, device)
        print(f"[轨迹] {k + 1}/{len(shard)} idx {it['_idx']} "
              f"({it['meta']['n_refs']}-ref) | {time.perf_counter() - t0:.0f}s", flush=True)

    # ── 第二趟:装 LoRA + iso processor,量残差 ───────────────────────
    apply_lora_ckpt(transformer, args.lora)
    runner = IsoRunner(transformer, block_diag=False, store=False)

    rows, premise = [], []
    for k, it in enumerate(shard):
        r, p = measure_sample(pipe, torch, runner, transformer, it,
                              trajs[it["_idx"]], sigmas, device)
        rows += r
        premise.append(p)
        print(f"[残差] {k + 1}/{len(shard)} idx {it['_idx']} | "
              f"{time.perf_counter() - t0:.0f}s | 峰值 "
              f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB", flush=True)

    out = os.path.join(OUT_DIR, f"rows_shard{args.shard_idx}.json")
    with open(out, "wt", encoding="utf-8") as f:
        json.dump({"meta": {"lora": args.lora, "n": args.n, "seed": args.seed,
                            "shard_idx": args.shard_idx, "num_shards": args.num_shards,
                            "steps": NUM_INFERENCE_STEPS, "true_cfg": TRUE_CFG_SCALE,
                            "resolution": RESOLUTION},
                   "rows": rows, "premise": premise}, f, indent=1)
    print(f"\n✓ 写 {out}({len(rows)} 行)")
    print_report(rows, premise)


def _dry_run(args):
    """不加载模型,用假读数把分片 / 落盘 / 合并 / 判读表 / σ₀ 自检全走一遍。"""
    picked = [{"_idx": i, "meta": {"n_refs": 1 + i % 3, "seed": 3407000 + i}}
              for i in range(args.n)]
    shard = shard_of(picked, args.shard_idx, args.num_shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    rows, premise = [], []
    for it in shard:
        for i in range(NUM_INFERENCE_STEPS):
            sigma = 1.0 - i / NUM_INFERENCE_STEPS
            base_pre = 0.30 + 0.05 * rng.random()
            for grp in GROUPS:
                # σ₀ 两组必须相同 —— 假数据也照这个构造,好验自检真的会过
                jitter = 0.0 if i == 0 else (0.04 * rng.random() if grp == "on_policy" else 0.0)
                rows.append({
                    "idx": it["_idx"], "n_refs": it["meta"]["n_refs"],
                    "step": i, "sigma": sigma, "group": grp,
                    "rel_pre": base_pre, "rel_post": base_pre * (0.35 + jitter),
                    "rel_lo_post": 0.05, "rel_hi_post": 0.18,
                    "hf_share_post": 0.6, "rel_lo_pre": 0.09, "rel_hi_pre": 0.31,
                })
        premise.append({"idx": it["_idx"], "n_refs": it["meta"]["n_refs"],
                        "hf_x0": 0.21, "hf_traj": 0.34})
    out = os.path.join(OUT_DIR, f"rows_shard{args.shard_idx}.json")
    with open(out, "wt", encoding="utf-8") as f:
        json.dump({"meta": {"dry_run": True}, "rows": rows, "premise": premise}, f, indent=1)
    print(f"[dry_run] 写 {out}({len(rows)} 行,{len(shard)} 条样本)")
    print_report(rows, premise)


def cmd_merge(args):
    import glob

    files = sorted(glob.glob(os.path.join(OUT_DIR, "rows_shard*.json")))
    if not files:
        raise SystemExit(f"❌ {OUT_DIR} 下没有 rows_shard*.json。先跑 run。")
    rows, premise, metas = [], [], []
    for p in files:
        with open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        rows += d["rows"]
        premise += d["premise"]
        metas.append(d["meta"])

    loras = {m.get("lora") for m in metas}
    if len(loras) > 1:
        raise SystemExit(f"❌ 分片之间 ckpt 不一致 {loras}。混判无意义,停。")
    dup = len({r["idx"] for r in rows})
    if dup != len(premise):
        raise SystemExit(f"❌ {len(premise)} 条 premise 但只有 {dup} 个不同 idx —— "
                         f"分片重叠了,同一条样本被算了多次,停。")

    out = os.path.join(OUT_DIR, "rows.json")
    with open(out, "wt", encoding="utf-8") as f:
        json.dump({"meta": {"shards": len(files), "lora": metas[0].get("lora")},
                   "rows": rows, "premise": premise}, f, indent=1)
    print(f"合并 {len(files)} 个分片 | {dup} 条样本 | {len(rows)} 行 → {out}")
    print_report(rows, premise)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embeds", help="单卡,载 VL,预算 正/负 prompt_embeds")
    pe.add_argument("--n", type=int, default=24)
    pe.add_argument("--seed", type=int, default=1234)

    pr = sub.add_parser("run", help="采 teacher 轨迹 + 量残差")
    pr.add_argument("--lora", required=True, help="train_iso.save() 落的 ckpt,如 step002000.pt")
    pr.add_argument("--n", type=int, default=24)
    pr.add_argument("--seed", type=int, default=1234)
    pr.add_argument("--shard_idx", type=int, default=0)
    pr.add_argument("--num_shards", type=int, default=1)
    pr.add_argument("--dry_run", action="store_true", help="不加载模型,用假读数验分片/落盘/判读表")

    sub.add_parser("merge", help="合并分片 + 打判读表")

    args = p.parse_args()
    if args.cmd == "embeds":
        cmd_embeds(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
