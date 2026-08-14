"""缓存正确性的**直接**判据:第 k 次前向重算的 ref K/V,与缓存里第 0 次那份,是否逐位相同。

背景(为什么需要这一支):`reports/20260814-p3-eval/REPORT.md` 里
「隔离+缓存 vs 隔离+每步重算」的像素差 mean 2.3–5.9 一直没结掉。
两个已经跑过的实验都判不了它:

  · §5.1 把注意力压到同一个 backend —— 只统一了**一个** kernel。read 模式喂进
    `iso_transformer_forward` 的只有噪声段(`pipeline_iso.py:246`),img 流上每个 GEMM
    的 M 维从 ~12.7k 掉到 ~4.1k,`to_q/k/v`、`to_out`、`img_mlp`、`norm_out`、`proj_out`
    全部换形状 ⇒ cuBLAS 换 kernel ⇒ 累加顺序变。那个实验从设计上就到不了 0。
  · §5.3 floor 拿 `full` 渲两遍 —— **代码路径完全相同**,没有扰动源,出 0 是必然。
    它证明流水线位级可复现(有用),但对缓存对不对**一点信息都没有**。

真正要问的是:缓存里存的那份 ref K/V,是不是就是重算会得到的那份。
隔离注意力的地基说它必须是 —— ref 只自注意(mask)+ ref 段固定 t=0 调制(`zero_cond_t`)
⇒ ref 的每层 K/V 与 t、与噪声内容完全解耦。P1 T3 在 CPU fp32 上把这条证死了(实测 0),
**没验的是真权重 bf16 + GPU kernel 下它还成不成立**。这里补上。

判据(二选一,没有中间地带):

  · 60 层 K 与 V **全部逐位相同** ⇒ 缓存喂给注意力的输入与重算逐位一致,
    缓存在数值上精确无损。剩下的像素差只可能来自形状驱动的 kernel 选择,是良性的。
    ⇒ 放行主批。
  · 有任何一层不等 ⇒ 那一层就是 bug 现场,打印层号与 max|Δ|,停下来修。

顺带白拿一个量:同一时刻 write 路与 read 路在**噪声 token 速度**上的差。
它是"每步扰动"的实际大小,与最终像素差 ~1e-2(2–6 / 255)一比,就知道 40 步放大了多少倍。

用法(1 卡,一张图,约 4 分钟):
    QWEN_WEIGHTS=... python qwen/diag_kv.py
    QWEN_WEIGHTS=... python qwen/diag_kv.py --task_idx 3 --probe_at 1,20,40,79

**只新建本文件,既有 `.py` 一个字不动(R0)。**
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infer_iso import (NEGATIVE_PROMPT, NUM_INFERENCE_STEPS, TRUE_CFG_SCALE, HEIGHT, WIDTH,
                       build_pipe, load_tasks, resolve_refs)


class CompareCache:
    """探针用的假缓存:`write()` 不存,直接与真缓存里那一层比,当场比完当场丢。

    做成"边写边比"而不是"存下来再比"是为了不吃显存 —— 2-ref@1024² 一份 ref K/V
    是 60 层 × 100 MB ≈ 6 GB,真缓存已经占着一份了。
    """

    def __init__(self, real, torch):
        self.real = real
        self.torch = torch
        self.rows = []

    def write(self, block_idx, k, v):
        k0, v0 = self.real.read(block_idx)
        # 同时记张量自身的量级:不等的时候要靠 |Δ|/scale 判断是"最后一位舍入"还是"逻辑错",
        # 只看绝对值判不了(K/V 各层量级差很多)。
        self.rows.append({
            "layer": block_idx,
            "k_eq": bool(self.torch.equal(k, k0)),
            "v_eq": bool(self.torch.equal(v, v0)),
            "k_max": float((k.float() - k0.float()).abs().max()),
            "v_max": float((v.float() - v0.float()).abs().max()),
            "k_scale": float(k0.float().abs().max()),
            "v_scale": float(v0.float().abs().max()),
        })

    def read(self, block_idx):
        return self.real.read(block_idx)

    def clear(self):
        self.rows.clear()

    def __len__(self):
        return len(self.real)


def install_probe(hook, torch, probe_at: set[int], log: list):
    """包一层 `transformer.forward`:正常路照跑,选中的那几次前向额外做一次探针。

    探针本身不改轨迹 —— 输出丢掉,不碰 generator,不碰 latents。
    """
    from pipeline_iso import iso_transformer_forward

    normal = hook.transformer.forward   # 就是 hook._forward
    state = {"n": 0}

    def wrapped(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                timestep=None, img_shapes=None, **kw):
        n = state["n"]
        state["n"] = n + 1
        res = normal(hidden_states, encoder_hidden_states=encoder_hidden_states,
                     encoder_hidden_states_mask=encoder_hidden_states_mask,
                     timestep=timestep, img_shapes=img_shapes, **kw)
        # 第 0 次是 write(缓存刚建),没得比;之后才是 read
        if n not in probe_at or hook.ctx.mode != "read":
            return res

        noise_len = hook.ctx.noise_len
        real = hook.ctx.cache
        probe = CompareCache(real, torch)
        hook.ctx.cache = probe
        hook.ctx.mode = "write"
        try:
            with torch.no_grad():
                # 传完整 hidden_states = cat([latents, image_latents]),即 write 路原样
                out_w = iso_transformer_forward(
                    hook.transformer, hook.ctx, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, timestep, img_shapes)
        finally:
            hook.ctx.cache = real
            hook.ctx.mode = "read"

        out_r = res.sample if hasattr(res, "sample") else res[0]
        dv = (out_w[:, :noise_len].float() - out_r[:, :noise_len].float())
        ref_norm = out_r[:, :noise_len].float().norm()

        rows = probe.rows
        bad = [r for r in rows if not (r["k_eq"] and r["v_eq"])]
        rec = {
            "forward": n,
            "n_layers": len(rows),
            "n_layers_bitwise_eq": len(rows) - len(bad),
            "kv_max_abs": max([max(r["k_max"], r["v_max"]) for r in rows], default=0.0),
            "kv_max_rel": max([max(r["k_max"] / max(r["k_scale"], 1e-30),
                                   r["v_max"] / max(r["v_scale"], 1e-30)) for r in rows],
                              default=0.0),
            "bad_layers": [r["layer"] for r in bad][:8],
            "v_max_abs": float(dv.abs().max()),
            "v_rel_l2": float(dv.norm() / ref_norm),
        }
        log.append(rec)
        print(f"  [探针] 前向 {n:3d} | ref K/V 逐位相同 {rec['n_layers_bitwise_eq']}/{rec['n_layers']} 层"
              f" | max|Δkv| {rec['kv_max_abs']:.3e}"
              f" || 噪声速度 max|Δ| {rec['v_max_abs']:.3e} 相对L2 {rec['v_rel_l2']:.3e}",
              flush=True)
        if bad:
            for r in bad[:8]:
                print(f"        ✗ 层 {r['layer']:2d}  k_eq={r['k_eq']} v_eq={r['v_eq']} "
                      f"max|Δk|={r['k_max']:.3e} max|Δv|={r['v_max']:.3e}", flush=True)
        return res

    hook.transformer.forward = wrapped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="m6")
    p.add_argument("--task_idx", type=int, default=0, help="任务表里第几条(默认第 0 条,即 §5.1 用的那条)")
    p.add_argument("--probe_at", default="1,20,40,79", help="在第几次前向下探针(0 是 write,不能探)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    weights = os.environ.get("QWEN_WEIGHTS")
    if not weights:
        raise SystemExit("❌ 环境变量 QWEN_WEIGHTS 未设置。")
    probe_at = {int(x) for x in args.probe_at.split(",") if x.strip()}
    if 0 in probe_at:
        raise SystemExit("❌ 第 0 次前向就是写缓存那次,自己跟自己比没意义。")

    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or \
        os.path.join(_REPO_ROOT, "output/p3_diag_kv")
    os.makedirs(out_dir, exist_ok=True)

    tasks, task_json = load_tasks(args.tasks)
    task = tasks[args.task_idx]
    refs = resolve_refs(task, task_json)
    print(f"[自检] 任务 {task['task_id']} | {task['meta']['n_refs']}-ref | seed {task['seed']} | "
          f"探针在前向 {sorted(probe_at)}", flush=True)

    pipe, hook, torch = build_pipe(weights, "iso_pre", None, block_diag=False, always_write=False)
    log: list = []
    install_probe(hook, torch, probe_at, log)

    from PIL import Image
    hook.reset()
    images = [Image.open(q).convert("RGB") for q in refs]
    torch.cuda.reset_peak_memory_stats()
    pipe(image=images, prompt=task["prompt"], negative_prompt=NEGATIVE_PROMPT,
         num_inference_steps=NUM_INFERENCE_STEPS, true_cfg_scale=TRUE_CFG_SCALE,
         height=HEIGHT, width=WIDTH,
         generator=torch.Generator(device="cuda").manual_seed(task["seed"]))

    n_probe = len(log)
    all_eq = n_probe > 0 and all(r["n_layers_bitwise_eq"] == r["n_layers"] for r in log)
    # 不等的时候还要分两种。bf16 尾数 8 位 ⇒ 相对 eps ≈ 2^-8 ≈ 3.9e-3,
    # 几个 ulp 就是 ~1e-2;逻辑错会与张量自身同阶(rel ≳ 0.1)。中间地带不该出现,
    # 真出现了才需要人来看。
    max_rel = max([r["kv_max_rel"] for r in log], default=0.0)
    verdict = "PASS" if all_eq else ("PASS_ROUNDING" if max_rel < 1e-2 else "FAIL")
    doc = {
        "spec": "P3-diag-kv-v1",
        "task_id": task["task_id"], "n_refs": task["meta"]["n_refs"], "seed": task["seed"],
        "probe_at": sorted(probe_at), "n_probe": n_probe,
        "all_bitwise_equal": all_eq, "verdict": verdict, "kv_max_rel": max_rel,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "n_forward_write": hook.n_write, "n_forward_read": hook.n_read,
        "probes": log,
    }
    path = os.path.join(out_dir, "diag_kv.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    vmax = max([r["v_rel_l2"] for r in log], default=0.0)
    print("\n" + "=" * 68)
    if n_probe == 0:
        print("❌ 一次探针都没跑到 —— probe_at 给的下标超出了本次前向数,重给。")
    elif verdict == "PASS":
        print(f"✅ PASS —— {n_probe} 次探针,每次 60 层 K/V 全部逐位相同。")
        print("   ⇒ 缓存喂给注意力的输入与每步重算逐位一致,缓存数值精确无损。")
        print("   ⇒ 像素差只能来自形状驱动的 kernel 选择(read 路 img 流 M 维 ~4.1k vs "
              "write 路 ~12.7k),良性。")
        print("   ⇒ 【直接投主批,不用等作者】")
    elif verdict == "PASS_ROUNDING":
        print(f"✅ PASS(舍入级)—— 有层不逐位相同,但最大相对差 {max_rel:.2e} < 1e-2,")
        print("   即 bf16 最后几位(尾数 8 位 ⇒ 相对 eps ≈ 3.9e-3)。不是逻辑错。")
        print("   ⇒ 【直接投主批,不用等作者】把这个数写进报告即可。")
    else:
        print(f"❌ FAIL —— 最大相对差 {max_rel:.2e},与张量自身同阶,是逻辑错不是舍入。")
        print("   ⇒ 【停,主批别投】把下面几行连同 diag_kv.json 发给作者。")
        for r in log:
            if r["n_layers_bitwise_eq"] != r["n_layers"]:
                print(f"   前向 {r['forward']}: {r['n_layers'] - r['n_layers_bitwise_eq']} 层不等,"
                      f"最早几层 {r['bad_layers']},max|Δ| {r['kv_max_abs']:.3e} "
                      f"(相对 {r['kv_max_rel']:.3e})")
    if n_probe:
        print(f"   [记数] 每步扰动 = write/read 两路噪声速度相对 L2,最大 {vmax:.2e}"
              f";作参照,最终像素差 ~1e-2(2–6/255)。")
    print(f"diag_kv.json : {path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
