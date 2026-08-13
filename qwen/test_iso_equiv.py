"""P1 硬门禁:隔离注意力 + ref KV 缓存的等价性自检。

规格:`qwen/PLAN.md` §3.1「等价性自检 —— 这是唯一的硬门禁」。
**这个自检不过,后面所有数字都没有意义。**

## 为什么用一个 2 层随机初始化的小模型,而不是 20B 真权重

等价性是**结构性**的:`PLAN.md` §1 的归纳(ref 段对注意力封闭 ⇒ 输出步不变 ⇒
逐层归纳 ⇒ ref K/V 步不变)整条推理里没有一处用到"权重是训练出来的"。
所以同一份代码路径,拿随机权重跑,该相等的照样相等、该不等的照样不等。

换来的是:**几秒钟、纯 CPU、fp32 下可以要求近乎逐位相同**。
20B bf16 真权重那一遍是**确认**不是**开发回路**——bf16 归约顺序不同会引入噪声,
只能用 `scripts/bench_kv_cache.py` 那种"像素差 mean < 0.5"的弱判据,
抓不出这里 T4 那种平移几个位置的静默错。

用法:
    python qwen/test_iso_equiv.py          # 全部 6 项
    python qwen/test_iso_equiv.py -v       # 打印每项的实测差值
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from iso_attn import IsoContext, RefKVCache, install_iso_processors
from pipeline_iso import iso_transformer_forward

# ---------------------------------------------------------------- 小模型规格
# 与 2511 的 transformer_config.json 同构,只把每个维度缩到最小可跑。
# axes_dims_rope 必须逐项为偶数且求和 == attention_head_dim(rope_params 有 assert,
# 且 _compute_video_freqs 按 axes_dim 切分)。2511 是 (16,56,56)=128;这里 (4,6,6)=16。
TINY = dict(
    patch_size=2, in_channels=64, out_channels=16,
    num_layers=2, attention_head_dim=16, num_attention_heads=2,
    joint_attention_dim=32, guidance_embeds=False,
    axes_dims_rope=(4, 6, 6), zero_cond_t=True,
)

# 第三张 ref 故意做成非方形:它让 max_vid_index 在"带不带 ref"之间发生变化,
# 于是 T4 那条 RoPE 平移陷阱才测得出来。真实 dreambooth 全是方形,测不出——
# 但那是数据的巧合,不是实现的正确性(见 pipeline_iso.py 文件头)。
IMG_SHAPES = [[(1, 4, 4), (1, 4, 4), (1, 6, 4)]]
TXT_LEN = 7
N_PAD = 2          # 末尾 2 个 txt token 是 padding,用来测 padding mask 有没有被弄丢
SEED = 20260813


def token_counts():
    noise = IMG_SHAPES[0][0]
    refs = IMG_SHAPES[0][1:]
    n_noise = noise[0] * noise[1] * noise[2]
    n_refs = [f * h * w for f, h, w in refs]
    return n_noise, n_refs


def build():
    torch.manual_seed(SEED)
    model = QwenImageTransformer2DModel(**TINY).to(torch.float32).eval()

    n_noise, n_refs = token_counts()
    g = torch.Generator().manual_seed(SEED)
    latents = torch.randn(1, n_noise, TINY["in_channels"], generator=g)
    image_latents = torch.randn(1, sum(n_refs), TINY["in_channels"], generator=g)
    ehs = torch.randn(1, TXT_LEN, TINY["joint_attention_dim"], generator=g)
    mask = torch.ones(1, TXT_LEN, dtype=torch.long)
    mask[:, TXT_LEN - N_PAD:] = 0
    return model, latents, image_latents, ehs, mask


def diff(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    model, latents, image_latents, ehs, mask = build()
    n_noise, n_refs = token_counts()
    full = torch.cat([latents, image_latents], dim=1)
    t1 = torch.tensor([0.9])
    t2 = torch.tensor([0.1])

    print(f"[自检] 小模型 {TINY['num_layers']} 层 / hidden {TINY['num_attention_heads'] * TINY['attention_head_dim']}"
          f" | txt {TXT_LEN}(pad {N_PAD}) | noise {n_noise} | refs {n_refs} | fp32 CPU")

    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  {'✅' if ok else '❌'} {name}  {detail}")

    # ── T4 先跑:它不需要改 processor,而且如果这条不成立,后面的推理前提就错了 ──────
    with torch.no_grad():
        _, txt_freqs_full = model.pos_embed(IMG_SHAPES, max_txt_seq_len=TXT_LEN, device=latents.device)
        _, txt_freqs_noise = model.pos_embed([[IMG_SHAPES[0][0]]], max_txt_seq_len=TXT_LEN,
                                             device=latents.device)
    shifted = not torch.equal(txt_freqs_full, txt_freqs_noise)
    check("T4 RoPE 陷阱:摘掉 ref 会平移 txt 频率(所以 read 必须传完整 img_shapes)",
          shifted, "两份 txt_freqs 不同 ⇒ 陷阱确实存在" if shifted
          else "两份相同 ⇒ 本测试用例没触发陷阱,IMG_SHAPES 需要一张非方形 ref")

    # ── T1:stock forward 与 off 模式必须逐位相同 ────────────────────────────────
    with torch.no_grad():
        stock = model(hidden_states=full, encoder_hidden_states=ehs,
                      encoder_hidden_states_mask=mask, timestep=t1,
                      img_shapes=IMG_SHAPES, return_dict=False)[0][:, :n_noise]

    ctx = IsoContext(mode="off", cache=RefKVCache())
    install_iso_processors(model, ctx)

    def run(mode, hs, t, txt=None):
        ctx.mode = mode
        with torch.no_grad():
            out = iso_transformer_forward(model, ctx, hs, ehs if txt is None else txt,
                                          mask, t, IMG_SHAPES)
        return out[:, :n_noise]

    off = run("off", full, t1)
    mx, mn = diff(stock, off)
    check("T1 改写没改语义:stock processor == off 模式(含 txt padding mask)",
          mx == 0.0, f"max={mx:.3e} mean={mn:.3e}")

    # ── T2:隔离确实生效(否则 T3/T5 是空的)─────────────────────────────────────
    wrote = run("write", full, t1)
    mx, mn = diff(off, wrote)
    check("T2 mask 不是摆设:全注意力 != 隔离注意力",
          mx > 1e-4, f"max={mx:.3e} mean={mn:.3e}")

    # ── T3:唯一的硬门禁 —— 隔离-无缓存 == 隔离-有缓存 ───────────────────────────
    read = run("read", latents, t1)
    mx, mn = diff(wrote, read)
    check("T3 【硬门禁】隔离-无缓存 == 隔离-有缓存",
          mx < 1e-5, f"max={mx:.3e} mean={mn:.3e}  (fp32 判据 max<1e-5)")

    # ── T5:ref K/V 步不变(PLAN §1 归纳的实验对应物)────────────────────────────
    kv_t1 = {i: (k.clone(), v.clone()) for i, (k, v) in ctx.cache.storage.items()}
    ctx.cache.clear()
    run("write", full, t2)
    worst = max(max(diff(kv_t1[i][0], k)[0], diff(kv_t1[i][1], v)[0])
                for i, (k, v) in ctx.cache.storage.items())
    check(f"T5 ref K/V 与去噪步无关(t={t1.item()} vs t={t2.item()},{len(ctx.cache)} 层全比)",
          worst == 0.0, f"max={worst:.3e}")

    # ── T6:ref K/V 与 prompt 无关 ⇒ cond / uncond 共享一份缓存(PLAN §1-2)──────
    ctx.cache.clear()
    torch.manual_seed(SEED + 1)
    neg_ehs = torch.randn_like(ehs)
    run("write", full, t1, txt=neg_ehs)
    worst = max(max(diff(kv_t1[i][0], k)[0], diff(kv_t1[i][1], v)[0])
                for i, (k, v) in ctx.cache.storage.items())
    check("T6 ref K/V 与 prompt 无关 ⇒ cond/uncond 共享缓存(80 次前向 1 写 79 读)",
          worst == 0.0, f"max={worst:.3e}")

    n_bad = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 68)
    if n_bad:
        print(f"❌ {n_bad}/{len(results)} 项不通过。P1 门禁未过,不要往下走。")
    else:
        print(f"✅ {len(results)}/{len(results)} 项通过。P1 结构门禁过。")
        print("   下一步:全权重 bf16 那一遍走 infer_hub 确认(弱判据,像素差 mean<0.5)。")
    print("=" * 68)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
