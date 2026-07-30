"""attn_record.py 的行为测试。小张量 + CPU,不需要 GPU 也不需要 FLUX 权重。

**上机前先跑这个**:H800 的 torch 版本与本地不同,而录制器依赖
scaled_dot_product_attention 的 scale 约定、softmax 数值行为、apply_rope 的广播形状。
本地全绿不等于那边全绿,而真跑一轮要 11 分钟,预检只要几秒。

    python distill/test_attn_record.py && python distill/run_attn_diag.py

要证明的是四件事,每一件都不是"读代码看着对",而是跑出来对:
  1. 段归约算得对        —— 与独立写的 float64 参考实现比
  2. 掩码确实不影响 img 行 —— 用真 build_isolated_attn_mask 实测,不是引注释
  3. wrapper 不改前向     —— patch 后输出与原函数逐比特相同
  4. 断言真的会炸        —— 布局错、调用次数错、嵌套,全部必须抛
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import attn_record as AR  # noqa: E402
from uno.flux.math import apply_rope  # noqa: E402
from uno.flux.math import attention as orig_attention  # noqa: E402
from uno.flux.ref_attention import RefKVCache, build_isolated_attn_mask  # noqa: E402

torch.manual_seed(0)

TXT, IMG_H, IMG_W = 6, 2, 2
IMG = IMG_H * IMG_W
REF_LENS = (3, 2)
H, D = 3, 8
SEQ_REF = TXT + IMG + sum(REF_LENS)   # 15
SEQ_NO = TXT + IMG                    # 10

LAYOUT = AR.Layout(txt_len=TXT, img_len=IMG, ref_lens=REF_LENS, img_h=IMG_H, img_w=IMG_W)

_fail = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        _fail.append(name)


def mk(L):
    return torch.randn(1, H, L, D, dtype=torch.float32)


def mk_pe(L):
    """apply_rope 要 freqs_cis 形如 (1,1,L,D/2,2,2);随便给个非平凡值即可——
    参考实现也走同一个 apply_rope,测的是切段不是 rope 本身。"""
    return torch.randn(1, 1, L, D // 2, 2, 2, dtype=torch.float32)


def reference(q, k, pe, ref_kv=None, cache_key=None):
    """独立参考实现:float64 全量 softmax,再切段。故意不复用 AR 的任何代码。"""
    qr, kr = apply_rope(q, k, pe)
    if ref_kv is not None and ref_kv.mode == "read":
        kr = torch.cat((kr, ref_kv.read(cache_key)[0]), dim=2)
    qi = qr[:, :, TXT:TXT + IMG, :].double()
    scores = torch.matmul(qi, kr.double().transpose(-1, -2)) * (D ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    segs = [probs[0, :, :, s:e].sum(dim=-1) for _, s, e in LAYOUT.segments()]
    return torch.stack(segs, dim=-1).mean(dim=0)  # (IMG, n_seg)


def run_blocks(rec, mode, n_steps=1, mask=None, cache=None):
    """跑 n_steps × 57 次 attention 调用,返回每次的参考值。"""
    refs = []
    for step in range(n_steps):
        for blk in range(AR.N_BLOCKS):
            ck = f"double_{blk}" if blk < AR.N_DOUBLE else f"single_{blk - AR.N_DOUBLE}"
            if mode == "read":
                q, k, v, pe = mk(SEQ_NO), mk(SEQ_NO), mk(SEQ_NO), mk_pe(SEQ_NO)
                cache.mode = "read"
            else:
                q, k, v, pe = mk(SEQ_REF), mk(SEQ_REF), mk(SEQ_REF), mk_pe(SEQ_REF)
                if cache is not None:
                    cache.mode = "write"
            refs.append(reference(q, k, pe, cache, ck))
            AR._layers.attention(q, k, v, pe, attn_mask=mask, ref_kv=cache,
                                 cache_key=ck, ref_len=sum(REF_LENS) if mode != "read" else 0)
    return refs


# ------------------------------------------------------------------ 1 布局
print("\n[1] 布局推导")
L512 = AR.layout_from_sizes([(512, 512), (512, 512)], 512, 512)
check("2-ref 512² → L=3584", L512.seq_with_ref == 3584,
      f"txt {L512.txt_len} + img {L512.img_len} + ref {L512.ref_lens}")
L3 = AR.layout_from_sizes([(512, 512)] * 3, 512, 512)
check("3-ref 512² → L=4608", L3.seq_with_ref == 4608)
check("段名顺序 [txt,img,ref1,ref2]", L512.seg_names == ["txt", "img", "ref1", "ref2"])
check("非方形 ref 也算对", AR.layout_from_sizes([(384, 512)], 512, 512).ref_lens == (32 * 24,),
      f"384x512 → {AR.layout_from_sizes([(384, 512)], 512, 512).ref_lens}")
try:
    AR.Layout(txt_len=512, img_len=1000, ref_lens=(4,), img_h=32, img_w=32)
    check("img 网格与 img_len 不符应抛", False)
except ValueError:
    check("img 网格与 img_len 不符应抛", True)
check("flat_block_index", (AR.flat_block_index("double_3"), AR.flat_block_index("single_7"))
      == (3, 26))

# ------------------------------------------------------------------ 2 full 模式
print("\n[2] full 模式(ref 在序列内,无掩码无缓存)—— official_full 形态")
rec = AR.AttnRecorder(LAYOUT, num_steps=2, spatial_blocks=(0, 5), spatial_steps=(0, 1))
with AR.recording(rec):
    refs = run_blocks(rec, "full", n_steps=2)
check("调用计数 = 2×57", rec.n_calls == 2 * AR.N_BLOCKS, f"{rec.n_calls}")
err = max(float((torch.tensor(rec.curve[s, b]).double() - refs[s * AR.N_BLOCKS + b].mean(0))
                .abs().max())
          for s in range(2) for b in range(AR.N_BLOCKS))
check("曲线 vs float64 参考实现", err < 2e-6, f"max abs err {err:.2e}")
check("各段份额之和 = 1", abs(float(rec.curve.sum(axis=-1).min()) - 1.0) < 1e-5
      and abs(float(rec.curve.sum(axis=-1).max()) - 1.0) < 1e-5,
      f"[{rec.curve.sum(axis=-1).min():.6f}, {rec.curve.sum(axis=-1).max():.6f}]")
sp_err = max(float((torch.tensor(rec.spatial[(s, b)]).double()
                    - refs[s * AR.N_BLOCKS + b].reshape(IMG_H, IMG_W, -1)).abs().max())
             for s in (0, 1) for b in (0, 5))
check("热力图 vs 参考实现", sp_err < 2e-6, f"max abs err {sp_err:.2e}")
check("热力图只存了指定 (step,block)", set(rec.spatial) == {(0, 0), (0, 5), (1, 0), (1, 5)})

# ------------------------------------------------------------------ 3 掩码不影响 img 行
print("\n[3] 隔离掩码对 img 行无效 —— ours_iso_nocache 形态(不变量 2 的实证)")
mask = build_isolated_attn_mask(TXT, IMG, list(REF_LENS), device=torch.device("cpu"))
check("掩码形状 (1,1,L,L)", tuple(mask.shape) == (1, 1, SEQ_REF, SEQ_REF), str(tuple(mask.shape)))
check("txt/img 行全 True", bool(mask[0, 0, :SEQ_NO].all()))
check("ref 行不全 True", not bool(mask[0, 0, SEQ_NO:].all()))
q, k, pe = mk(SEQ_REF), mk(SEQ_REF), mk_pe(SEQ_REF)
qr, kr = apply_rope(q, k, pe)
sc = torch.matmul(qr[:, :, TXT:TXT + IMG].double(), kr.double().transpose(-1, -2)) * (D ** -0.5)
p_free = torch.softmax(sc, dim=-1)
p_mask = torch.softmax(sc.masked_fill(~mask[:, :, TXT:TXT + IMG, :], float("-inf")), dim=-1)
check("img 行:带掩码 softmax == 不带掩码 softmax",
      float((p_free - p_mask).abs().max()) == 0.0,
      f"max diff {float((p_free - p_mask).abs().max()):.1e}")
rec = AR.AttnRecorder(LAYOUT, num_steps=1)
with AR.recording(rec):
    refs = run_blocks(rec, "full", n_steps=1, mask=mask)
err = max(float((torch.tensor(rec.curve[0, b]).double() - refs[b].mean(0)).abs().max())
          for b in range(AR.N_BLOCKS))
check("有掩码时曲线仍与无掩码参考一致", err < 2e-6, f"max abs err {err:.2e}")

# ------------------------------------------------------------------ 4 kv write / read
print("\n[4] kv_cache 的 write(step 0)与 read(step 1+)")
cache = RefKVCache()
rec = AR.AttnRecorder(LAYOUT, num_steps=1)
with AR.recording(rec):
    refs_w = run_blocks(rec, "full", n_steps=1, mask=mask, cache=cache)
check("write 后缓存了 57 个 block", len(cache.storage) == AR.N_BLOCKS, f"{len(cache.storage)}")
check("缓存的 ref 段长度 = 5", cache.storage["double_0"][0].shape[2] == sum(REF_LENS))
err = max(float((torch.tensor(rec.curve[0, b]).double() - refs_w[b].mean(0)).abs().max())
          for b in range(AR.N_BLOCKS))
check("write 步曲线正确", err < 2e-6, f"max abs err {err:.2e}")

rec = AR.AttnRecorder(LAYOUT, num_steps=1)
with AR.recording(rec):
    refs_r = run_blocks(rec, "read", n_steps=1, cache=cache)
err = max(float((torch.tensor(rec.curve[0, b]).double() - refs_r[b].mean(0)).abs().max())
          for b in range(AR.N_BLOCKS))
check("read 步曲线正确(q 无 ref、k 拼回缓存)", err < 2e-6, f"max abs err {err:.2e}")
check("read 步 q 长度记录为 seq_no_ref", rec.q_len_by_step == [SEQ_NO], str(rec.q_len_by_step))
check("read 二次读缓存无副作用", len(cache.storage) == AR.N_BLOCKS)

# ------------------------------------------------------------------ 5 不改前向
print("\n[5] wrapper 不改前向(不变量 1)")
AR.install()
q, k, v, pe = mk(SEQ_REF), mk(SEQ_REF), mk(SEQ_REF), mk_pe(SEQ_REF)
a = AR._layers.attention(q, k, v, pe, attn_mask=mask)          # 未激活 → 纯转发
b = orig_attention(q, k, v, pe, attn_mask=mask)
check("未激活时逐比特相同", torch.equal(a, b))
rec = AR.AttnRecorder(LAYOUT, num_steps=1)
outs = []
with AR.recording(rec):
    for blk in range(AR.N_BLOCKS):
        ck = f"double_{blk}" if blk < AR.N_DOUBLE else f"single_{blk - AR.N_DOUBLE}"
        outs.append(AR._layers.attention(q, k, v, pe, attn_mask=mask, cache_key=ck))
check("激活时输出仍与原函数逐比特相同", all(torch.equal(o, b) for o in outs))
AR.uninstall()
check("uninstall 复原", AR._layers.attention is orig_attention)

# ------------------------------------------------------------------ 6 断言
print("\n[6] 断言必须真的抛")


def expect(name, fn, exc=RuntimeError):
    try:
        fn()
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:58]}")
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"抛了 {type(e).__name__} 而非 {exc.__name__}")
        return
    check(name, False, "没抛")


def bad_layout():
    wrong = AR.Layout(txt_len=TXT, img_len=IMG, ref_lens=(3, 3), img_h=IMG_H, img_w=IMG_W)
    r = AR.AttnRecorder(wrong, num_steps=1)
    with AR.recording(r):
        run_blocks(r, "full", n_steps=1)


def too_many():
    r = AR.AttnRecorder(LAYOUT, num_steps=1)
    with AR.recording(r):
        run_blocks(r, "full", n_steps=2)


def too_few():
    r = AR.AttnRecorder(LAYOUT, num_steps=2)
    with AR.recording(r):
        run_blocks(r, "full", n_steps=1)


def wrong_block_order():
    r = AR.AttnRecorder(LAYOUT, num_steps=1)
    with AR.recording(r):
        q_, k_, v_, pe_ = mk(SEQ_REF), mk(SEQ_REF), mk(SEQ_REF), mk_pe(SEQ_REF)
        AR._layers.attention(q_, k_, v_, pe_, cache_key="double_5")  # 计数说 0


def nested():
    r1, r2 = AR.AttnRecorder(LAYOUT, 1), AR.AttnRecorder(LAYOUT, 1)
    with AR.recording(r1), AR.recording(r2):
        pass


expect("ref_lens 写错 → 抛(key 长度对不上)", bad_layout)
expect("调用次数超出申报步数 → 抛", too_many)
expect("调用次数不足(finalize)→ 抛", too_few)
expect("block 顺序与 cache_key 不符 → 抛", wrong_block_order)
expect("嵌套录制 → 抛", nested)
expect("空 ref_lens → 抛",
       lambda: AR.Layout(txt_len=1, img_len=1, ref_lens=(), img_h=1, img_w=1), ValueError)
check("异常后全局状态已清理", AR._ACTIVE is None)

# ------------------------------------------------------------------ 7 head_chunk 无关性
print("\n[7] head_chunk 不影响结果")
vals = []
for hc in (1, 2, 3, 5):
    torch.manual_seed(7)
    r = AR.AttnRecorder(LAYOUT, num_steps=1, head_chunk=hc)
    with AR.recording(r):
        run_blocks(r, "full", n_steps=1)
    vals.append(r.curve.copy())
spread = max(float(abs(v - vals[0]).max()) for v in vals[1:])
check("head_chunk 1/2/3/5 结果一致", spread < 1e-6, f"max spread {spread:.2e}")

# ------------------------------------------------------------------ 8 inference_mode
print("\n[8] torch.inference_mode 下可用")
# WHY 单独测:pipeline.forward 带 @torch.inference_mode(pipeline.py:243),
# 里面新建的张量是 inference tensor。若 .cpu().numpy() 或 np.savez 在这种张量上
# 受限,真跑时才炸,而那时已经加载了 7 分钟的模型。


@torch.inference_mode()
def under_inference_mode():
    r = AR.AttnRecorder(LAYOUT, num_steps=1, spatial_blocks=(0,), spatial_steps=(0,))
    with AR.recording(r):
        run_blocks(r, "full", n_steps=1)
    return r


try:
    r = under_inference_mode()
    check("inference_mode 下录制不报错", True)
    check("curve 已填充且份额之和 = 1", bool(r.curve.any())
          and abs(float(r.curve.sum(axis=-1).mean()) - 1.0) < 1e-5)
    import io  # noqa: E402

    import numpy as np  # noqa: E402
    buf = io.BytesIO()
    np.savez_compressed(buf, **r.to_npz_payload(task_id="t", variant="v"))
    check("npz 可序列化", len(buf.getvalue()) > 0, f"{len(buf.getvalue())} bytes")
except Exception as e:  # noqa: BLE001
    check("inference_mode 下录制不报错", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 62)
if _fail:
    print(f"❌ {len(_fail)} 项失败:{_fail}")
    sys.exit(1)
print("✓ 全部通过")
