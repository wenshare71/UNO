"""缓存正确性的直接判据:重算的 ref K/V 与缓存里那份是否逐位相同。

第一轮(2026-08-14)跑出来的结果把问题定位到了一个更窄的地方,所以这一支现在问两件事:

  **① 步不变性** —— ref K/V 会不会随去噪步变。这是缓存的地基。
     第一轮已答:**不会**,cond 支路第 20/40 次前向重算的 ref K/V 与第 0 步缓存的
     60/60 层逐位相同。bf16 + GPU kernel 下成立,不用再验(本文件仍会顺带复验)。

  **② 文本无关性** —— ref K/V 会不会随文本段变。第一轮发现:**会**。
     uncond 支路(negative_prompt=" ")重算出来的 ref K/V 与 cond 支路缓存的
     59/60 层不等,max|Δ| 1216,相对 0.246;只有第 0 层(不过注意力那层)相等。

  ②有两种完全不同的成因,信号长得一模一样,但一个是 bug 一个是良性:

    (a) **真漏**:ref 行实际看见了 txt。若如此,`iso_attn.build_isolated_attn_mask`
        没起到作用,整条隔离路线的地基就塌了 —— ref K/V 依赖 prompt,缓存跨 cond/uncond
        共享就是错的,P1 T6 的结论在 bf16 下不成立。
    (b) **长度效应**:cond 和 uncond 的 txt 段长度不同 ⇒ 联合序列总长不同 ⇒
        ref 段相对注意力 kernel 分块边界的对齐不同 ⇒ 累加顺序不同 ⇒ bf16 舍入不同。
        数学上 ref 仍然只看 ref,只是算得不一样。良性。

  分辨方法**不需要看任何阈值**,只要两个逐位判断:

    A(机制自检)  在第 1 次前向用**存下来的 cond 文本**重跑 write 路。
                  第 1 次前向与第 0 次是同一步、同一个 latent,所以这等于把第 0 次
                  原样重算一遍 —— **必须逐位等于缓存**。不等就是探针自己有问题,
                  别去怪缓存。
    C(判据)      同样在第 1 次前向,用 **cond 文本沿 token 维翻转**重跑 write 路。
                  长度一模一样、内容完全不同。
                    · 仍逐位相等 ⇒ ref 对 txt **内容**是盲的 ⇒ 成因是 (b),良性。
                    · 不等       ⇒ ref 看见了 txt 内容 ⇒ 成因是 (a),**真 bug**。
    B(留档)      用当次真实的 uncond 文本重跑,即第一轮测到的那个数,继续记录。

顺带记 mean|Δ| 与不等元素占比 —— 第一轮只有 max,而 max=1216 却几乎不影响噪声速度
(相对 L2 3.6e-3,与 K/V 逐位相同时的 4.3e-3 同量级),说明差异可能只集中在极少数
离群通道上。mean 能把这件事说清楚。

用法(1 卡,一张图,约 6 分钟):
    QWEN_WEIGHTS=... python qwen/diag_kv.py

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

    def _cmp(self, x, x0):
        d = (x.float() - x0.float()).abs()
        return {
            "eq": bool(self.torch.equal(x, x0)),
            "max": float(d.max()),
            "mean": float(d.mean()),
            "scale_max": float(x0.float().abs().max()),
            "scale_mean": float(x0.float().abs().mean()),
            "frac_ne": float((x != x0).float().mean()),
        }

    def write(self, block_idx, k, v):
        k0, v0 = self.real.read(block_idx)
        row = {"layer": block_idx}
        for name, cur, ref in (("k", k, k0), ("v", v, v0)):
            for key, val in self._cmp(cur, ref).items():
                row[f"{name}_{key}"] = val
        self.rows.append(row)

    def read(self, block_idx):
        return self.real.read(block_idx)

    def clear(self):
        self.rows.clear()

    def __len__(self):
        return len(self.real)


def summarize(rows: list) -> dict:
    """把 60 层压成一条。相对量用「层内 max|Δ| / 层内 max|参考|」和 mean 的对应比值。"""
    bad = [r for r in rows if not (r["k_eq"] and r["v_eq"])]

    def top(field, scale):
        return max((r[f"{p}_{field}"] / max(r[f"{p}_{scale}"], 1e-30)
                    for r in rows for p in ("k", "v")), default=0.0)

    return {
        "n_layers": len(rows),
        "n_bitwise_eq": len(rows) - len(bad),
        "bad_layers": [r["layer"] for r in bad][:10],
        "max_abs": max((max(r["k_max"], r["v_max"]) for r in rows), default=0.0),
        "max_rel": top("max", "scale_max"),
        "mean_rel": top("mean", "scale_mean"),
        "frac_ne": max((max(r["k_frac_ne"], r["v_frac_ne"]) for r in rows), default=0.0),
    }


def install_probe(hook, torch, probe_at: set[int], log: list, txt_test_at: int):
    """包一层 `transformer.forward`:正常路照跑,选中的那几次前向额外做探针。

    探针不改轨迹 —— 输出全丢掉,不碰 generator,不碰 latents,不写真缓存。
    """
    from pipeline_iso import iso_transformer_forward

    normal = hook.transformer.forward   # 就是 hook._forward
    state = {"n": 0, "cond_ehs": None, "cond_mask": None}

    def write_probe(hidden_states, ehs, ehs_mask, timestep, img_shapes):
        """跑一次 write 路前向,逐层与缓存比。返回 (rows, out_w)。"""
        real = hook.ctx.cache
        probe = CompareCache(real, torch)
        hook.ctx.cache = probe
        hook.ctx.mode = "write"
        try:
            with torch.no_grad():
                out_w = iso_transformer_forward(hook.transformer, hook.ctx, hidden_states,
                                                ehs, ehs_mask, timestep, img_shapes)
        finally:
            hook.ctx.cache = real
            hook.ctx.mode = "read"
        return probe.rows, out_w

    def wrapped(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                timestep=None, img_shapes=None, **kw):
        n = state["n"]
        state["n"] = n + 1
        res = normal(hidden_states, encoder_hidden_states=encoder_hidden_states,
                     encoder_hidden_states_mask=encoder_hidden_states_mask,
                     timestep=timestep, img_shapes=img_shapes, **kw)
        if n == 0:
            # 写缓存那一次的文本,后面 A/C 两个变体要用
            state["cond_ehs"] = encoder_hidden_states
            state["cond_mask"] = encoder_hidden_states_mask
        if n not in probe_at or hook.ctx.mode != "read":
            return res

        noise_len = hook.ctx.noise_len
        out_r = res.sample if hasattr(res, "sample") else res[0]

        rows, out_w = write_probe(hidden_states, encoder_hidden_states,
                                  encoder_hidden_states_mask, timestep, img_shapes)
        dv = out_w[:, :noise_len].float() - out_r[:, :noise_len].float()
        rec = {"forward": n, "txt_len": int(encoder_hidden_states.shape[1]),
               "B_uncond_or_current": summarize(rows),
               "v_max_abs": float(dv.abs().max()),
               "v_rel_l2": float(dv.norm() / out_r[:, :noise_len].float().norm())}

        s = rec["B_uncond_or_current"]
        print(f"  [探针] 前向 {n:3d} txt_len={rec['txt_len']:4d} | 逐位相同 "
              f"{s['n_bitwise_eq']}/{s['n_layers']} 层 | max|Δ| {s['max_abs']:.3e} "
              f"(相对 {s['max_rel']:.2e}) | mean 相对 {s['mean_rel']:.2e} | "
              f"不等元素占比 {s['frac_ne']:.2e} || 噪声速度相对L2 {rec['v_rel_l2']:.3e}",
              flush=True)

        # ── 只在指定的那一次前向做 A / C 两个变体 ────────────────────────────
        if n == txt_test_at and state["cond_ehs"] is not None:
            cond_ehs, cond_mask = state["cond_ehs"], state["cond_mask"]
            rows_a, _ = write_probe(hidden_states, cond_ehs, cond_mask, timestep, img_shapes)
            rec["A_cond_replay"] = summarize(rows_a)
            # 同长度、异内容:沿 token 维翻转。ref 若真对 txt 内容盲,这必须逐位不变。
            rows_c, _ = write_probe(hidden_states, cond_ehs.flip(dims=[1]), cond_mask,
                                    timestep, img_shapes)
            rec["C_txt_flipped"] = summarize(rows_c)
            rec["cond_txt_len"] = int(cond_ehs.shape[1])
            for tag, key in (("A 复算 cond", "A_cond_replay"), ("C 翻转 txt", "C_txt_flipped")):
                t = rec[key]
                print(f"    [{tag}] 逐位相同 {t['n_bitwise_eq']}/{t['n_layers']} 层 | "
                      f"max|Δ| {t['max_abs']:.3e} | mean 相对 {t['mean_rel']:.2e}", flush=True)

        log.append(rec)
        return res

    hook.transformer.forward = wrapped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="m6")
    p.add_argument("--task_idx", type=int, default=0, help="任务表里第几条(默认第 0 条)")
    p.add_argument("--probe_at", default="1,20,40,79", help="在第几次前向下探针(0 是 write,不能探)")
    p.add_argument("--txt_test_at", type=int, default=1,
                   help="在第几次前向做 A/C 变体。必须是与前向 0 同一步的那次(即 1),"
                        "否则 A 复算不成立")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    weights = os.environ.get("QWEN_WEIGHTS")
    if not weights:
        raise SystemExit("❌ 环境变量 QWEN_WEIGHTS 未设置。")
    probe_at = {int(x) for x in args.probe_at.split(",") if x.strip()}
    if 0 in probe_at:
        raise SystemExit("❌ 第 0 次前向就是写缓存那次,自己跟自己比没意义。")
    if args.txt_test_at not in probe_at:
        raise SystemExit(f"❌ --txt_test_at {args.txt_test_at} 不在 --probe_at {sorted(probe_at)} 里。")

    out_dir = args.out or os.environ.get("INFER_OUTPUT_DIR") or \
        os.path.join(_REPO_ROOT, "output/p3_diag_kv")
    os.makedirs(out_dir, exist_ok=True)

    tasks, task_json = load_tasks(args.tasks)
    task = tasks[args.task_idx]
    refs = resolve_refs(task, task_json)
    print(f"[自检] 任务 {task['task_id']} | {task['meta']['n_refs']}-ref | seed {task['seed']} | "
          f"探针在前向 {sorted(probe_at)} | A/C 变体在前向 {args.txt_test_at}", flush=True)

    pipe, hook, torch = build_pipe(weights, "iso_pre", None, block_diag=False, always_write=False)
    log: list = []
    install_probe(hook, torch, probe_at, log, args.txt_test_at)

    from PIL import Image
    hook.reset()
    images = [Image.open(q).convert("RGB") for q in refs]
    torch.cuda.reset_peak_memory_stats()
    pipe(image=images, prompt=task["prompt"], negative_prompt=NEGATIVE_PROMPT,
         num_inference_steps=NUM_INFERENCE_STEPS, true_cfg_scale=TRUE_CFG_SCALE,
         height=HEIGHT, width=WIDTH,
         generator=torch.Generator(device="cuda").manual_seed(task["seed"]))

    # ── 判读。全部是逐位判断,没有阈值。──────────────────────────────────────
    txt_rec = next((r for r in log if "A_cond_replay" in r), None)
    cond_probes = [r for r in log if r["forward"] % 2 == 0]          # 偶数 = cond 支路
    step_inv = bool(cond_probes) and all(
        r["B_uncond_or_current"]["n_bitwise_eq"] == r["B_uncond_or_current"]["n_layers"]
        for r in cond_probes)

    if txt_rec is None:
        verdict, why = "INCONCLUSIVE", "A/C 变体没跑到(--txt_test_at 不在 probe_at 里?)"
    elif txt_rec["A_cond_replay"]["n_bitwise_eq"] != txt_rec["A_cond_replay"]["n_layers"]:
        verdict, why = "INCONCLUSIVE", ("A 复算 cond 都没能逐位重现缓存 —— 探针机制自身有问题,"
                                        "先别怪缓存")
    elif not step_inv:
        verdict, why = "FAIL_STEP", "cond 支路的 ref K/V 随去噪步变 —— 缓存的地基不成立"
    elif txt_rec["C_txt_flipped"]["n_bitwise_eq"] == txt_rec["C_txt_flipped"]["n_layers"]:
        verdict, why = "PASS", ("同长度换掉 txt 内容,ref K/V 逐位不变 ⇒ ref 对 txt 内容是盲的。"
                                "uncond 那边的差异只能来自 txt 段长度不同导致的 kernel 分块差异,良性")
    else:
        verdict, why = "FAIL_LEAK", ("同长度只换 txt 内容,ref K/V 就变了 ⇒ ref 看见了 txt。"
                                     "隔离 mask 没起作用,缓存跨 cond/uncond 共享不成立")

    doc = {
        "spec": "P3-diag-kv-v2",
        "task_id": task["task_id"], "n_refs": task["meta"]["n_refs"], "seed": task["seed"],
        "probe_at": sorted(probe_at), "txt_test_at": args.txt_test_at,
        "verdict": verdict, "why": why, "step_invariant_cond": step_inv,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "n_forward_write": hook.n_write, "n_forward_read": hook.n_read,
        "probes": log,
    }
    path = os.path.join(out_dir, "diag_kv.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    mark = {"PASS": "✅", "FAIL_LEAK": "❌", "FAIL_STEP": "❌", "INCONCLUSIVE": "⚠️"}[verdict]
    print("\n" + "=" * 74)
    print(f"{mark} {verdict} —— {why}")
    print(f"   步不变性(cond 支路,前向 {[r['forward'] for r in cond_probes]}):"
          f"{'逐位成立' if step_inv else '不成立'}")
    if txt_rec:
        print(f"   A 复算 cond:{txt_rec['A_cond_replay']['n_bitwise_eq']}/60 层逐位相同")
        print(f"   C 翻转 txt :{txt_rec['C_txt_flipped']['n_bitwise_eq']}/60 层逐位相同")
        print(f"   B 真 uncond:{txt_rec['B_uncond_or_current']['n_bitwise_eq']}/60 层逐位相同"
              f"(cond txt_len={txt_rec.get('cond_txt_len')}, "
              f"uncond txt_len={txt_rec['txt_len']})")
    if verdict == "PASS":
        print("   ⇒ 【直接投主批,不用等作者】")
    else:
        print("   ⇒ 【停,主批别投】把这段输出连同 diag_kv.json 发给作者")
    print(f"diag_kv.json : {path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
