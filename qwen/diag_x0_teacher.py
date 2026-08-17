"""诊断续篇:把 x₀ 换成 teacher 自己的输出,低 σ 的 A/B 差距会不会塌掉。

**不训练、不出图、不进盲评。** 只量残差,产物是三张表。

── 为什么有这一单 ──────────────────────────────────────────────────────

`reports/20260817-diag-x0/REPORT.md`(commit 28009f2)的读数:

  · σ₀ 自检 24/24 过,装置成立;
  · **前提表平均比值 0.973** —— x₀_UNO 的高频占比并不低于 on-policy 末端。
    原假说「512² 上采样 ⇒ 高频是插值伪影」在第一环就断了;
  · 但主表命中了另一条预登记分支:on_policy 的修复率
    0.562 → 0.472 → 0.330 → 0.288 随 σ→0 单调下降,train_dist 基本平;
    低 σ 段 rel_post 0.0396 vs 0.1299(3.3×),rel_hi 0.0473 vs 0.1618(3.4×)。

于是有了一条**事后**的重新归因,本脚本就是来验它的:

    问题不是 x₀ 糊,是 x₀ **不是 teacher 自己的不动点**。
    x_t = (1-σ)x₀ + σε 是条直线。σ→1 时噪声主导,直线与 ODE 轨迹的边缘分布重合
    ⇒ 高 σ 两组不分;σ→0 时直线塌到你挑的那张 x₀ 上,而真轨迹塌到 teacher 自己
    会生成的那张图上 —— 这两张是不同的图。σ 越小,训练点离部署点越远。

flow matching **训练**一个模型时这条直线没问题(只要边缘分布对);但**蒸馏**一个
已经训好的模型时,相关的点只有 solver 真正走过的那些,逐样本并不相等。

⚠️ 这个归因是看完数据才提出来的,不是预登记的。所以它只能用这一单来验,
不能拿它直接去申请重训。**预登记的读法写在文件末尾 `READINGS`,跑之前就定死。**

── 单变量设计:三臂,同一批 ────────────────────────────────────────────

同一条 (prompt, refs)、同一份 image_latents / embeds / ε,只差 x_t 从哪来:

    A `train_dist`  x_t = (1-σ)·x₀_UNO     + σ·ε   ← 现在训练喂的
    C `teacher_x0`  x_t = (1-σ)·x₀_teacher + σ·ε   ← 换成 teacher 自己的输出
    B `on_policy`   x_t = teacher 40 步采样在 σ 处的实际 latent  ← 部署真正经过的

A 和 B **必须在这一批重测**,不能引 28009f2 那张表来并排比 —— 跨批次读数不得
并排引用(`distill/M4_EVAL_SPEC.md` §8.5-3)。这也是为什么本单不是「只加一臂」。

x₀_teacher 是白拿的:`record_trajectory` 里那次 `pipe(...)` 本来就跑满 40 步,
`output_type="latent"` 的返回值就是 σ=0 处的 latent(pipeline L873-874,
与 `encode_sample` 的 x₀ 同为 pack + 归一化后的空间)。上一版把它丢了,这版捡起来。
**换 x₀ 这个干预不产生任何额外生成成本。**

ε 仍取轨迹起点 ⇒ σ₀=1.0 处三臂逐位相同,机制自检从 2 组扩到 3 组。

── 三张表 ──────────────────────────────────────────────────────────────

1. 前提表(每条样本):x₀_UNO / x₀_teacher / 轨迹末端 的高频占比,
   外加 ‖x₀_UNO − x₀_teacher‖/‖x₀_teacher‖ —— 两张目标图差多远。
2. 机制表(0 次前向,免费):`dx` = ‖x_t − x_t^{on_policy}‖/‖x_t^{on_policy}‖。
   **干预有没有真的把训练点拉到部署点上**,看这一列。它不通过模型,不会被
   「模型碰巧对某类输入更鲁棒」污染。
3. 主表:与上一单同口径的 rel_pre / rel_post / 修复率 / rel_lo / rel_hi。

── 用法 ────────────────────────────────────────────────────────────────

embeds 复用上一单的产物(同 n / 同 seed ⇒ 同一批 24 条),**不用重算**:

    # 8 卡各一片,约 35 min/片
    QWEN_WEIGHTS=... python qwen/diag_x0_teacher.py run --lora <ckpt> \
        --n 24 --shard_idx $i --num_shards 8

    # 合并 + 判读表(不加载模型,不需要 GPU)
    python qwen/diag_x0_teacher.py merge

**只新建本文件。`diag_x0_shift.py` 一个字不动** —— 28009f2 那批读数得能原样复现。
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

from diag_x0_shift import (  # noqa: E402  路径要先补好
    EMBED_DIR,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    RESOLUTION,
    SIGMA_BANDS,
    TRUE_CFG_SCALE,
    _mean,
    hf_ratio,
    pick_samples,
    refs_hist,
    rel,
    residual_stats,
    shard_of,
)

SHIFT_EMBED_DIR = EMBED_DIR              # 上一单的 embeds,本单原样复用
OUT_DIR = os.path.join(_REPO_ROOT, "output/diag_x0_teacher")

# 顺序即打印顺序:两个「直线」臂挨着,on_policy 垫底当参照
GROUPS = ("train_dist", "teacher_x0", "on_policy")

assert NEGATIVE_PROMPT == " " and TRUE_CFG_SCALE == 4.0     # Q1 口径没被动过


# ==================================================================== 预登记读法
# 跑之前定死。写下就不许改,只能加带日期的订正注记(`PLAN.md` §4)。
READINGS = """
  【机制检查,先看这个】机制表的 dx 列:
    σ→0 时 dx(teacher_x0) 明显小于 dx(train_dist)  ⇒ 干预确实把训练点拉到了部署点上,
                                                     主表可以读;
    两者差不多                                      ⇒ 干预根本没生效(x₀_teacher 与
                                                     x₀_UNO 差不多,或轨迹末端不收敛),
                                                     主表不用解释,回报原始数据即可。

  【主判据】低 σ 段,teacher_x0 与 on_policy 的差距,对比 train_dist 与 on_policy 的差距:
    teacher_x0 的差距明显收窄(rel_post 降、修复率升)⇒ 归因立住:x₀ 必须是 teacher
                                                     自己的不动点,重训换 x₀ 源有据;
    teacher_x0 与 train_dist 基本一样               ⇒ 归因否掉:低 σ 的差距不是
                                                     「x₀ 选谁」的问题,换数据白干,
                                                     矛头回到 mask 本身;
    teacher_x0 反而更差                             ⇒ 原样记录,不要解释。

  【必须挂上的限制】样本 24 条;同一条样本的 40 步高度相关,**有效 n ≈ 24,不是表里的行数**。
  单个 ckpt、无置信区间。量的是对 teacher 的残差,不是画质 —— 残差小 ≠ 身份保得住。

  **判据由作者读,本脚本不下结论。**
"""


# ==================================================================== 判读表

def selfcheck_sigma0(rows: list[dict]) -> list[str]:
    """σ₀ 三臂用的是同一个 x_t(σ₀=1.0,ε 取自轨迹起点)⇒ 读数必须逐位相同。

    空列表 = 装置成立。**唯一的硬门禁**,而且是构造出来的恒等式,不是猜的阈值。
    dx 一并查:σ₀ 处三臂都等于 on_policy 自己,dx 必须是 0。
    """
    bad = []
    by_sample: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["step"] != 0:
            continue
        by_sample.setdefault(str(r["idx"]), {})[r["group"]] = r
    for idx, g in sorted(by_sample.items()):
        if set(g) != set(GROUPS):
            bad.append(f"样本 {idx}:σ₀ 只有 {sorted(g)},三臂不齐")
            continue
        ref = g["on_policy"]
        for grp in GROUPS:
            for f in ("rel_pre", "rel_post"):
                if g[grp][f] != ref[f]:
                    bad.append(f"样本 {idx}:σ₀ 的 {f} {grp} 与 on_policy 不等 "
                               f"({g[grp][f]:.6e} vs {ref[f]:.6e})")
            if g[grp]["dx"] != 0.0:
                bad.append(f"样本 {idx}:σ₀ 的 dx[{grp}] = {g[grp]['dx']:.3e},应为 0")
    return bad


def summarize(rows: list[dict]) -> list[dict]:
    """按 σ 分段 × 分臂聚合。分段只为把 40 行压成能读的 4 行,不是判据。"""
    out = []
    for lo, hi, label in SIGMA_BANDS:
        for grp in GROUPS:
            sel = [r for r in rows if r["group"] == grp and lo <= r["sigma"] < hi]
            if not sel:
                continue
            m_pre = _mean([r["rel_pre"] for r in sel])
            m_post = _mean([r["rel_post"] for r in sel])
            out.append({
                "band": label, "group": grp, "n": len(sel),
                "dx": _mean([r["dx"] for r in sel]),
                "rel_pre": m_pre, "rel_post": m_post,
                "repair": 1 - m_post / m_pre if m_pre and m_pre == m_pre else float("nan"),
                "rel_lo": _mean([r["rel_lo_post"] for r in sel]),
                "rel_hi": _mean([r["rel_hi_post"] for r in sel]),
                "hf_share": _mean([r["hf_share_post"] for r in sel]),
            })
    return out


def print_report(rows: list[dict], premise: list[dict]) -> None:
    agg = summarize(rows)

    print("\n" + "=" * 92)
    print("表 1 · 前提:两张目标图差多远,各自缺不缺高频")
    print("-" * 92)
    print(f"{'样本':>8} {'n_refs':>7} {'hf(x₀_UNO)':>12} {'hf(x₀_teacher)':>15} "
          f"{'hf(轨迹末端)':>14} {'‖Δx₀‖/‖x₀_t‖':>14}")
    for p in premise:
        print(f"{p['idx']:>8} {p['n_refs']:>7} {p['hf_x0_uno']:>12.4f} "
              f"{p['hf_x0_teacher']:>15.4f} {p['hf_traj_end']:>14.4f} {p['rel_x0']:>14.3f}")
    if premise:
        print(f"{'平均':>8} {'':>7} {_mean([p['hf_x0_uno'] for p in premise]):>12.4f} "
              f"{_mean([p['hf_x0_teacher'] for p in premise]):>15.4f} "
              f"{_mean([p['hf_traj_end'] for p in premise]):>14.4f} "
              f"{_mean([p['rel_x0'] for p in premise]):>14.3f}")
        print("  hf(x₀_teacher) ≈ hf(轨迹末端) 是应该的(末端 σ ~0.02);两者差很多说明轨迹没收敛。")
        print("  ‖Δx₀‖ 是「两张目标图有多不一样」。它接近 0 ⇒ 这个干预本身没换掉什么,下面两张表不用读。")

    print("\n" + "=" * 92)
    print("表 2 · 机制(不过模型,0 次前向):dx = ‖x_t − x_t^on_policy‖ / ‖x_t^on_policy‖")
    print("-" * 92)
    print(f"{'σ 段':>18} {'臂':>12} {'n':>5} {'dx':>10}")
    for s in agg:
        print(f"{s['band']:>18} {s['group']:>12} {s['n']:>5} {s['dx']:>10.4f}")
    print("  这一列量的是「训练点离部署点多远」,与模型无关。σ→0 时它是不是塌下来,决定干预有没有生效。")

    print("\n" + "=" * 92)
    print("表 3 · 主表:同一条 (prompt, refs),只差 x_t 从哪来")
    print("-" * 92)
    print(f"{'σ 段':>18} {'臂':>12} {'n':>5} {'rel_pre':>9} {'rel_post':>9} "
          f"{'修复率':>8} {'rel_lo':>8} {'rel_hi':>8} {'残差高频占比':>12}")
    for s in agg:
        print(f"{s['band']:>18} {s['group']:>12} {s['n']:>5} {s['rel_pre']:>9.4f} "
              f"{s['rel_post']:>9.4f} {s['repair']:>8.3f} {s['rel_lo']:>8.4f} "
              f"{s['rel_hi']:>8.4f} {s['hf_share']:>12.3f}")
    print("\n  rel_pre  = 未训练的隔离腿(=iso_pre)在这一段的相对残差,批内标尺")
    print("  修复率   = 1 − rel_post/rel_pre,训练在这一段补回了多少")
    print("  rel_lo/hi= 低频/高频**各自**的相对误差;「保住颜色保不住数字」对应 rel_hi ≫ rel_lo")
    print(READINGS)
    print("=" * 92)

    bad = selfcheck_sigma0(rows)
    if bad:
        print("\n❌ σ₀ 机制自检未过 —— 三臂在 σ₀ 用的是同一个 x_t,读数本应逐位相同:")
        for b in bad[:10]:
            print(f"   · {b}")
        print(f"   (共 {len(bad)} 条)装置有问题,上面的表不要解读。")
    else:
        print(f"\n✅ σ₀ 机制自检:{len({r['idx'] for r in rows})} 条样本三臂逐位相同,dx 全 0。")


# ==================================================================== 主体

def record_trajectory(pipe, torch, item, noise_len, device, embed_dir):
    """跑一遍 Q1 口径的 teacher 采样,返回 (每步进 transformer 的 x_t, σ=0 处的 latent)。

    与 `diag_x0_shift.record_trajectory` 的唯一差别:**保留 `pipe(...)` 的返回值**。
    `output_type="latent"` 时 pipeline 直接把去噪循环结束后的 packed latent 交出来
    (L873-874,不 unpack、不反归一化),正好与 `encode_sample` 的 x₀ 同空间。

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
    pos = torch.load(os.path.join(embed_dir, f"{item['_idx']:06d}.pt"),
                     map_location=device).to(torch.bfloat16)[None]
    neg = torch.load(os.path.join(embed_dir, f"{item['_idx']:06d}.neg.pt"),
                     map_location=device).to(torch.bfloat16)[None]

    xs: list = []
    orig = pipe.transformer.forward

    def rec(hidden_states, **kw):
        xs.append(hidden_states[:, :noise_len].detach().to("cpu").clone())
        return orig(hidden_states=hidden_states, **kw)

    pipe.transformer.forward = rec
    try:
        out = pipe(image=refs, prompt=None,
                   prompt_embeds=pos, negative_prompt_embeds=neg,
                   true_cfg_scale=TRUE_CFG_SCALE,
                   num_inference_steps=NUM_INFERENCE_STEPS,
                   height=RESOLUTION, width=RESOLUTION,
                   generator=torch.Generator(device="cuda").manual_seed(item["meta"]["seed"]),
                   output_type="latent", return_dict=False)
    finally:
        pipe.transformer.forward = orig

    # cond 在前、uncond 在后(pipeline L816 / L830),每步两次 ⇒ 取偶数位就是 40 个 x_t
    if len(xs) != 2 * NUM_INFERENCE_STEPS:
        raise SystemExit(
            f"❌ 轨迹记到 {len(xs)} 次前向,期望 {2 * NUM_INFERENCE_STEPS}(40 步 × cond/uncond)。\n"
            f"   true_cfg 或步数被改了,或者 pipeline 版本不同 —— 这条轨迹不是 Q1 口径,停。")
    x0_teacher = out[0].detach()
    if x0_teacher.shape != xs[0].shape:
        raise SystemExit(
            f"❌ pipeline 交出来的 latent {tuple(x0_teacher.shape)} 与轨迹 x_t "
            f"{tuple(xs[0].shape)} 形状不同。output_type='latent' 的返回口径变了,停。")
    return xs[0::2], x0_teacher


def measure_sample(pipe, torch, runner, transformer, item, traj, x0_teacher,
                   sigmas, device, embed_dir) -> tuple:
    """一条样本上跑满 40 σ × 3 臂 × 3 次前向,返回 (rows, premise_row)。"""
    from train_iso import encode_sample

    x0_uno, image_latents, embeds, img_shapes = encode_sample(
        pipe, item, embed_dir, item["_idx"], device)
    if x0_uno.shape[1] != traj[0].shape[1]:
        raise SystemExit(
            f"❌ encode_sample 的噪声段 {x0_uno.shape[1]} token 与轨迹的 {traj[0].shape[1]} 对不上。\n"
            f"   两边的 img_shapes 不一致,三臂就不是单变量了,停。")
    x0_tea = x0_teacher.to(device=device, dtype=x0_uno.dtype)

    # ε 取轨迹起点(σ₀=1.0 处 latent 就是初始噪声)⇒ σ₀ 三臂逐位相同,成为机制自检
    eps = traj[0].to(device=device, dtype=x0_uno.dtype)

    rows = []
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        xb = traj[i].to(device=device, dtype=x0_uno.dtype)
        x_by_group = {
            "train_dist": (1.0 - sigma) * x0_uno + sigma * eps,
            "teacher_x0": (1.0 - sigma) * x0_tea + sigma * eps,
            "on_policy": xb,
        }
        # 推理侧传的是 bf16(timesteps)/1000,即**舍入两次**;照抄那条路径(同 train_iso)
        ts = (sigma[None] * 1000).to(x0_uno.dtype) / 1000

        for grp in GROUPS:
            x_t = x_by_group[grp]
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
                "dx": rel(x_t, xb),               # 不过模型,免费
                "rel_pre": s_pre["rel"], "rel_post": s_post["rel"],
                "rel_lo_post": s_post["rel_lo"], "rel_hi_post": s_post["rel_hi"],
                "hf_share_post": s_post["hf_share"],
                "rel_lo_pre": s_pre["rel_lo"], "rel_hi_pre": s_pre["rel_hi"],
            })

    premise = {
        "idx": item["_idx"], "n_refs": item["meta"]["n_refs"],
        "hf_x0_uno": hf_ratio(x0_uno),
        "hf_x0_teacher": hf_ratio(x0_tea),
        # 末端 σ 已经 ~0.02,与 σ=0 的输出应当很接近;差很多说明轨迹没收敛
        "hf_traj_end": hf_ratio(traj[-1].to(device)),
        "rel_x0": rel(x0_uno, x0_tea),
    }
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

    # embeds 复用上一单的产物:同 n / 同 seed ⇒ pick_samples 选出同一批 24 条
    missing = [it["_idx"] for it in shard
               if not (os.path.exists(os.path.join(args.embed_dir, f"{it['_idx']:06d}.pt"))
                       and os.path.exists(os.path.join(args.embed_dir, f"{it['_idx']:06d}.neg.pt")))]
    if missing:
        raise SystemExit(f"❌ {len(missing)} 条样本的 embeds 不在 {args.embed_dir},例如 {missing[:3]}。\n"
                         f"   本单不重算 embeds。先补:python qwen/diag_x0_shift.py embeds --n {args.n}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[自检] 全量 {len(picked)} 条 {refs_hist(picked)} | shard {args.shard_idx}/{args.num_shards} "
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

    # ── 第一趟:pristine pipeline,采 teacher 轨迹 + 拿 teacher 的 x₀ ──
    # 顺序不能反,见 record_trajectory 的 ⚠️
    t0 = time.perf_counter()
    trajs, x0s = {}, {}
    for k, it in enumerate(shard):
        trajs[it["_idx"]], x0s[it["_idx"]] = record_trajectory(
            pipe, torch, it, noise_len, device, args.embed_dir)
        print(f"[轨迹] {k + 1}/{len(shard)} idx {it['_idx']} "
              f"({it['meta']['n_refs']}-ref) | {time.perf_counter() - t0:.0f}s", flush=True)

    # ── 第二趟:装 LoRA + iso processor,量残差 ───────────────────────
    # ⚠️ 这一段每条样本 360 次纯 no_grad 前向、无进度条,`[残差]` 要等一条跑完才打印。
    #    静默期约 10 分钟是正常的(上一单 §4-5 踩过)。
    apply_lora_ckpt(transformer, args.lora)
    runner = IsoRunner(transformer, block_diag=False, store=False)

    rows, premise = [], []
    for k, it in enumerate(shard):
        r, p = measure_sample(pipe, torch, runner, transformer, it, trajs[it["_idx"]],
                              x0s[it["_idx"]], sigmas, device, args.embed_dir)
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
                            "resolution": RESOLUTION, "groups": list(GROUPS)},
                   "rows": rows, "premise": premise}, f, indent=1)
    print(f"\n✓ 写 {out}({len(rows)} 行)")
    print_report(rows, premise)


def _dry_run(args):
    """不加载模型,用假读数把分片 / 落盘 / 三张表 / σ₀ 自检全走一遍。"""
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
                # σ₀ 三臂必须相同、dx 必须为 0 —— 假数据也照这个构造,好验自检真的会过
                jitter = 0.0 if i == 0 or grp == "on_policy" else 0.04 * rng.random()
                rows.append({
                    "idx": it["_idx"], "n_refs": it["meta"]["n_refs"],
                    "step": i, "sigma": sigma, "group": grp,
                    "dx": 0.0 if i == 0 or grp == "on_policy" else (1.0 - sigma) * 0.9,
                    "rel_pre": base_pre, "rel_post": base_pre * (0.35 + jitter),
                    "rel_lo_post": 0.05, "rel_hi_post": 0.18,
                    "hf_share_post": 0.6, "rel_lo_pre": 0.09, "rel_hi_pre": 0.31,
                })
        premise.append({"idx": it["_idx"], "n_refs": it["meta"]["n_refs"],
                        "hf_x0_uno": 0.21, "hf_x0_teacher": 0.26,
                        "hf_traj_end": 0.25, "rel_x0": 0.88})
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
    # 逐样本查,不查全体并集 —— 只要有一片是旧的两臂脚本跑的,并集照样凑齐三臂,
    # 但那条样本的 teacher_x0 就来自另一批。三臂不同批不能并排比(§8.5-3)。
    short = {r["idx"] for r in rows}
    short = {i for i in short
             if {r["group"] for r in rows if r["idx"] == i} != set(GROUPS)}
    if short:
        raise SystemExit(f"❌ {len(short)} 条样本的臂不齐(例如 {sorted(short)[:3]}),"
                         f"期望每条都有 {sorted(GROUPS)}。\n"
                         f"   有分片是旧版脚本跑的、或中途挂了。重跑那一片,停。")

    out = os.path.join(OUT_DIR, "rows.json")
    with open(out, "wt", encoding="utf-8") as f:
        json.dump({"meta": {"shards": len(files), "lora": metas[0].get("lora")},
                   "rows": rows, "premise": premise}, f, indent=1)
    print(f"合并 {len(files)} 个分片 | {dup} 条样本 | {len(rows)} 行 → {out}")
    print_report(rows, premise)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="采 teacher 轨迹 + x₀,量三臂残差")
    pr.add_argument("--lora", required=True, help="train_iso.save() 落的 ckpt,如 step002000.pt")
    pr.add_argument("--n", type=int, default=24)
    pr.add_argument("--seed", type=int, default=1234)
    pr.add_argument("--shard_idx", type=int, default=0)
    pr.add_argument("--num_shards", type=int, default=1)
    # 上一单的 embeds 在 output/diag_x0_shift/embeds;worker 上 output/ 是软链,给个逃生口
    pr.add_argument("--embed_dir", default=SHIFT_EMBED_DIR)
    pr.add_argument("--dry_run", action="store_true", help="不加载模型,用假读数验分片/落盘/判读表")

    sub.add_parser("merge", help="合并分片 + 打三张表")

    args = p.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
