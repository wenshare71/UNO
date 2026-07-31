"""推理时 ref→主图 注意力录制器。**零侵入:不修改任何既有 `.py`。**

上游讨论:`.codeflicker/discuss/2026-07-29/attention-visualization/`(D01 方向 / D02 首轮配置)。
本模块实现 D01 选定的"`math.py:attention()` 统一汇聚点 + 在线降维"方案,并落实
D03 对 D01/D02 的四处修正。

## 为什么 monkey-patch 而不是改 `math.py`

`uno/flux/modules/layers.py:23` 是 `from ..math import attention`——模块级名字绑定,
所以运行时替换 `layers.attention` 就能拦下全部 **4 个活调用点**
(layers.py:258/306/397/420,double/single 各自的 Lora 与非 Lora processor)。
另外 2 个调用点(:163/:180)所在的 `FLuxSelfAttnProcessor` / `LoraFluxAttnProcessor`
没有被任何 block 装配(`DoubleStreamBlock.__init__:350` / `SingleStreamBlock.__init__:460`
装的是另外两个),是死代码。

代价是既有文件 diff 为 0:M1–M4 全部结果都是这份代码产出的,改它等于改历史。

## 三条不变量

1. **出图与不录制时逐比特相同。** wrapper 先原样调用真 `attention()` 拿前向输出,
   录制用的概率矩阵是**另算**的、不参与前向。因此 D01/D02 那条"eager 记录状态与线上
   flash 行为理论等价但数值微差,措辞需注明"的红线**在本实现下不存在**——不用注明,
   因为没有差异。代价是注意力算了两遍,诊断场景可接受(D01 已预期 1.5–2x)。

2. **只算 img query 行,因而完全不需要处理掩码。** `build_isolated_attn_mask`
   (ref_attention.py:86) 对 txt/img 行全 True——img 行从不被遮;official_full 更是
   压根没有掩码。少一个出错点,顺带把 2-ref 瞬态从 ~0.59 GB 压到 ~0.12 GB(见下)。

3. **布局外部注册 + 每次调用硬断言。** wrapper 只拿得到 `ref_len`(总和),拿不到
   per-ref 切分——`_ref_attn_kwargs`(layers.py:28)只传了 `sum(ref_lens)`。所以布局
   由 driver 预先注册,wrapper 每次校验 q/k 长度,对不上立刻抛。
   **静默的错误归因比崩溃危险得多**:段边界错一位,曲线照样画得出来,而且看着挺合理。

## 序列布局:全部 57 个 block 统一为 `[txt, img, ref_1..ref_N]`

  - double block:`q = cat((txt_q, img_q))`(layers.py:254),而 `img` 在 model.py:197
    早已是 `[img, ref]`,故整体是 [txt, img, ref]
  - single block:model.py:249 `cat((txt, img), 1)`,同一布局
  - kv read 步:q=`[txt, img]`,k=`[txt, img]` + cached_ref(math.py:47)——**段结构一致**,
    只是 query 少了 ref 行,而我们本来就只取 img 行
  - official_full:无掩码,但 ref 照样在序列末尾(model.py:197 与隔离无关)

所以一套切法通吃四个变体、通吃 write/read/full 三种模式。

## 显存

D01 按 L≈2300 估(那对应 ref_size=320);本项目全线 ref_size=512 且 T5 max_length=512
(pipeline.py:114),实际 L = 512 + 1024 + 1024·N,即 2-ref 3584 / 3-ref 4608。
全头一次算的话 fp32 概率矩阵 2-ref 就要 352 MB。所以按 head 分块累加,
`head_chunk=4` 时瞬态 ≈ 2×59 MB。

## D04:curve 改为按 head 分辨(不再对 24 头取平均)

首轮 8 样本×4 变体跑完后发现:全 head 平均的 late-block 份额**测不出主体是否真的
出现**——S1_022 里 `ours_kv_pre` 完全丢了 grey_sloth_plushie(肉眼确认),但它的份额
比正确渲染的 `ours_kv_post4000` 还高;S1_000 里 `ours_kv_pre` 整个生成崩坏,份额同样
"均衡"。24 个头一起平均,可能把某几个真正做身份绑定的头的信号冲淡了。

所以 `curve` 从 `(step, block, seg)` 改为 `(step, block, head, seg)`——每个 head 的
份额单独存,不在 head 维度归约,归约留给下游分析自己选怎么切。`spatial` 维持 head 平均
不变(热力图按 head 拆开是 24× 的存储,先看 curve 层面有没有值得深挖的头再决定)。

两种约简共用同一次 softmax:per-head curve 对 img 位置取均值,spatial 对 head 求和后
再统一除以 n_heads,同一批 `probs` 张量算两次归约,不重复算注意力分数。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

import uno.flux.modules.layers as _layers
from uno.flux.math import apply_rope
from uno.flux.math import attention as _orig_attention

N_DOUBLE = 19
N_SINGLE = 38
N_BLOCKS = N_DOUBLE + N_SINGLE  # 57;flat index 0..18 = double_i,19..56 = single_(i-19)
N_HEADS = 24  # util.py:128/161/194,double/single 全部 block 统一 24 头

_ACTIVE: "AttnRecorder | None" = None
_INSTALLED = False


def flat_block_index(cache_key: str) -> int:
    """`double_3` → 3;`single_7` → 26。用于与调用计数交叉校验。"""
    kind, _, idx = cache_key.partition("_")
    if kind == "double":
        return int(idx)
    if kind == "single":
        return N_DOUBLE + int(idx)
    raise ValueError(f"无法解析 cache_key: {cache_key!r}")


@dataclass(frozen=True)
class Layout:
    """一次推理的序列布局。由 driver 从预处理后的图尺寸算出并注册。

    ref token 数 = (h // 16) * (w // 16):AE 下采样 8 倍,再按 ph=pw=2 打包
    (sampling.py:141)。img 同理。txt 恒为 512(T5 max_length,pipeline.py:114)。
    """

    txt_len: int
    img_len: int
    ref_lens: tuple[int, ...]
    img_h: int  # img query 还原成 2D 网格用,img_h * img_w 必须等于 img_len
    img_w: int

    def __post_init__(self) -> None:
        if self.img_h * self.img_w != self.img_len:
            raise ValueError(
                f"img 网格 {self.img_h}x{self.img_w} 与 img_len {self.img_len} 不符")
        if not self.ref_lens:
            raise ValueError("ref_lens 为空:本录制器只用于多 ref 诊断")

    @property
    def ref_total(self) -> int:
        return sum(self.ref_lens)

    @property
    def seq_with_ref(self) -> int:
        return self.txt_len + self.img_len + self.ref_total

    @property
    def seq_no_ref(self) -> int:
        return self.txt_len + self.img_len

    def segments(self) -> list[tuple[str, int, int]]:
        """key 维度的分段 `[(名字, start, end)]`;顺序即存储顺序。"""
        segs = [("txt", 0, self.txt_len),
                ("img", self.txt_len, self.seq_no_ref)]
        off = self.seq_no_ref
        for i, n in enumerate(self.ref_lens):
            segs.append((f"ref{i + 1}", off, off + n))
            off += n
        return segs

    @property
    def seg_names(self) -> list[str]:
        return [s[0] for s in self.segments()]


class AttnRecorder:
    """累计 img query 对各段的注意力质量。

    存储(D03 修正 ② + D04 按 head 拆分):
      - `curve`:每 (step, block, head, seg) 一个标量(只对 img token 取均值,
        **不对 head 取均值**)→ num_steps × 57 × 24 × n_seg,25 步 2-ref 时约 547 KB
      - `spatial`:只在指定 (step, block) 上留 img 网格,head 仍取均值
        → n_step_sel × n_blk_sel × img_h × img_w × n_seg,默认约 245 KB

    D01 设想的"全 57 块 × 全步 × 全分辨率"是 17 MB/样本/变体 → 全轮 400 MB,进不了 git;
    D04 之前 curve 也对 head 取了均值(约 17 KB),但这掩盖了"是否有少数头专门做身份
    绑定"这个问题——24 个头一起平均,单个头的强信号会被摊薄到 1/24。curve 拆开 head
    后每样本每变体约 547 KB,全轮(8 样本×4 变体)约 17.5 MB,仍然进得了 git。
    """

    def __init__(self, layout: Layout, num_steps: int,
                 spatial_blocks: tuple[int, ...] = (),
                 spatial_steps: tuple[int, ...] = (),
                 head_chunk: int = 4) -> None:
        self.layout = layout
        self.num_steps = num_steps
        self.spatial_blocks = set(spatial_blocks)
        self.spatial_steps = set(spatial_steps)
        self.head_chunk = head_chunk

        n_seg = len(layout.segments())
        self.curve = np.zeros((num_steps, N_BLOCKS, N_HEADS, n_seg), dtype=np.float32)
        self.spatial: dict[tuple[int, int], np.ndarray] = {}
        self.n_calls = 0
        self.n_heads: int | None = None
        # 每步 query 长度,用来事后确认 kv 变体确实只有 step 0 带 ref(sampling.py:239)
        self.q_len_by_step: list[int] = []

    # -------------------------------------------------------------- 录制

    def observe(self, q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor,
                ref_kv, cache_key: str | None) -> None:
        """由 wrapper 调用。q/k 是 **rope 之前**的,本函数自己补 rope 与 read 拼接。

        WHY 自己补:真 `attention()` 在内部做 rope(math.py:38)和 read 拼接(math.py:47),
        但不把中间结果返回。重算比改函数签名便宜——rope 是逐元素的,拼接是纯 view/cat,
        而 `RefKVCache.read` 是纯字典查找(ref_attention.py:48),二次读没有副作用。
        """
        step, blk = divmod(self.n_calls, N_BLOCKS)
        self.n_calls += 1
        if step >= self.num_steps:
            raise RuntimeError(
                f"attention 调用次数超出预期:第 {self.n_calls} 次落到 step {step},"
                f"但只申报了 {self.num_steps} 步。每步应恰好 {N_BLOCKS} 次调用。")
        if cache_key is not None and flat_block_index(cache_key) != blk:
            raise RuntimeError(
                f"block 计数与 cache_key 对不上:计数说 {blk},cache_key 说 "
                f"{cache_key}({flat_block_index(cache_key)})。block 顺序假设被打破了。")

        L = self.layout
        q, k = apply_rope(q, k, pe)
        if ref_kv is not None and ref_kv.mode == "read":
            cached_k, _ = ref_kv.read(cache_key)
            k = torch.cat((k, cached_k), dim=2)

        if q.shape[0] != 1:
            raise RuntimeError(f"录制器只支持 batch=1,实到 {q.shape[0]}")
        if k.shape[2] != L.seq_with_ref:
            raise RuntimeError(
                f"key 长度 {k.shape[2]} != 注册布局的 {L.seq_with_ref} "
                f"(txt {L.txt_len} + img {L.img_len} + ref {L.ref_lens})。"
                f"布局注册错了,继续录只会得到错位但看着合理的曲线。")
        if q.shape[2] not in (L.seq_with_ref, L.seq_no_ref):
            raise RuntimeError(
                f"query 长度 {q.shape[2]} 既不是含 ref 的 {L.seq_with_ref} "
                f"也不是不含 ref 的 {L.seq_no_ref}")
        if blk == 0:
            self.q_len_by_step.append(int(q.shape[2]))

        n_heads = q.shape[1]
        if n_heads != N_HEADS:
            raise RuntimeError(f"head 数 {n_heads} != 预期的 {N_HEADS}(util.py 硬编码值),"
                               f"curve 的 head 维度分配错了")
        self.n_heads = n_heads
        segs = L.segments()
        # img query 行:两种模式下都在 [txt_len, txt_len + img_len)
        q_img = q[:, :, L.txt_len:L.seq_no_ref, :]
        scale = q.shape[-1] ** -0.5
        want_spatial = blk in self.spatial_blocks and step in self.spatial_steps

        # per_head:(head, seg) —— 对 img 位置取均值,head 维度**保留**(D04)。
        # spatial_acc:(img_len, seg) —— head 维度求和(稍后除以 n_heads),img 位置**保留**。
        # 两者是同一批 probs 的两种不同归约,只算一次 softmax。
        per_head = torch.zeros(n_heads, len(segs), dtype=torch.float32, device=q.device)
        spatial_acc = (torch.zeros(L.img_len, len(segs), dtype=torch.float32, device=q.device)
                      if want_spatial else None)
        for h0 in range(0, n_heads, self.head_chunk):
            h1 = min(h0 + self.head_chunk, n_heads)
            # fp32 做 softmax:bf16 在 3584 长度上求和会有可见的累积误差,
            # 而这里算的是要拿去比大小的份额,不是前向数值。
            scores = torch.matmul(q_img[:, h0:h1].float(),
                                  k[:, h0:h1].float().transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)  # (1, hc, img_len, L)
            for si, (_, s, e) in enumerate(segs):
                seg_prob = probs[0, :, :, s:e].sum(dim=-1)  # (hc, img_len)
                per_head[h0:h1, si] = seg_prob.mean(dim=1)
                if want_spatial:
                    spatial_acc[:, si] += seg_prob.sum(dim=0)
            del scores, probs

        self.curve[step, blk] = per_head.float().cpu().numpy()
        if want_spatial:
            spatial_acc /= n_heads
            self.spatial[(step, blk)] = (
                spatial_acc.reshape(L.img_h, L.img_w, len(segs)).float().cpu().numpy())

    # -------------------------------------------------------------- 收尾

    def finalize(self) -> None:
        """跑完后校验调用次数,不对就抛——半截的曲线比没有曲线更害人。"""
        expected = self.num_steps * N_BLOCKS
        if self.n_calls != expected:
            raise RuntimeError(
                f"attention 调用 {self.n_calls} 次,预期 {expected} "
                f"({self.num_steps} 步 × {N_BLOCKS} block)。录到的曲线不完整。")

    def to_npz_payload(self, **meta) -> dict:
        L = self.layout
        payload = {
            "curve": self.curve,
            "seg_names": np.array(L.seg_names),
            "spatial_keys": np.array(sorted(self.spatial), dtype=np.int32).reshape(-1, 2),
            "layout_txt_len": np.int32(L.txt_len),
            "layout_img_len": np.int32(L.img_len),
            "layout_ref_lens": np.array(L.ref_lens, dtype=np.int32),
            "layout_img_grid": np.array([L.img_h, L.img_w], dtype=np.int32),
            "n_heads": np.int32(self.n_heads or 0),
            "n_double": np.int32(N_DOUBLE),
            "n_single": np.int32(N_SINGLE),
            "q_len_by_step": np.array(self.q_len_by_step, dtype=np.int32),
        }
        for (step, blk), arr in sorted(self.spatial.items()):
            payload[f"spatial_{step:03d}_{blk:03d}"] = arr
        for key, val in meta.items():
            payload[f"meta_{key}"] = np.array(val)
        return payload


# ------------------------------------------------------------------ patch 安装

def _wrapped_attention(q, k, v, pe, attn_mask=None, ref_kv=None,
                       cache_key=None, ref_len=0):
    """与 `uno.flux.math.attention` 同签名。未激活时是纯转发,零开销。"""
    out = _orig_attention(q, k, v, pe, attn_mask=attn_mask, ref_kv=ref_kv,
                          cache_key=cache_key, ref_len=ref_len)
    rec = _ACTIVE
    if rec is not None:
        # 放在真前向**之后**:write 模式下缓存此时已由真 attention 写好,
        # 我们的二次读拿到的就是它;顺序反过来则 read(cache_key) 会 KeyError。
        rec.observe(q, k, pe, ref_kv, cache_key)
    return out


def install() -> None:
    """替换 `layers.attention`。幂等;进程内只需调一次。"""
    global _INSTALLED
    if _INSTALLED:
        return
    if _layers.attention is not _orig_attention:
        raise RuntimeError(
            f"layers.attention 已经不是原函数({_layers.attention!r}),"
            f"有别的东西也在打补丁,拒绝叠加。")
    _layers.attention = _wrapped_attention
    _INSTALLED = True


def uninstall() -> None:
    global _INSTALLED
    _layers.attention = _orig_attention
    _INSTALLED = False


@contextlib.contextmanager
def recording(recorder: AttnRecorder):
    """在 with 块内激活录制;退出时校验完整性。

    嵌套会静默丢数据,所以直接禁掉。
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("录制不可嵌套")
    install()
    _ACTIVE = recorder
    try:
        yield recorder
    finally:
        _ACTIVE = None
    recorder.finalize()


def layout_from_sizes(ref_sizes: list[tuple[int, int]], width: int, height: int,
                      txt_len: int = 512) -> Layout:
    """从**预处理后**的 ref 图尺寸 `[(w, h), ...]` 和出图尺寸算布局。

    token 数 = (h // 16) * (w // 16)——AE 下采样 8 倍 + ph=pw=2 打包(sampling.py:141)。
    `preprocess_ref` 已把边长对齐到 16 的倍数(pipeline.py:84),所以整除是精确的。
    """
    return Layout(
        txt_len=txt_len,
        img_len=(height // 16) * (width // 16),
        ref_lens=tuple((h // 16) * (w // 16) for (w, h) in ref_sizes),
        img_h=height // 16,
        img_w=width // 16,
    )
