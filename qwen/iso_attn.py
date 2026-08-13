"""隔离注意力 + ref K/V 缓存 —— Qwen-Image-Edit-2511 版。

规格:`qwen/PLAN.md` §1(地基)、§3.1(实现)。**只新建本文件,既有 `.py` 一个字不动(R0)。**
`qwen/_vendor/diffusers_0.40.0.dev0/` 是只读证据快照,不 import、不修改;
运行期 import 的是 env 里装的 diffusers(同一个 commit `90c0ffdc`)。

序列布局(逐行核对过 vendored 源码,不是推测):

    joint = [ txt │ noise │ ref₁ │ ref₂ … ]
              └── QwenDoubleStreamAttnProcessor2_0 里 cat([txt_*, img_*], dim=1)
                  而 img = cat([latents, image_latents], dim=1)  ← pipeline L811

隔离靠两条,缺一条 ref 的 K/V 就仍随去噪步变:

  (a) ref 只自注意 —— 本文件的 mask;
  (b) ref 段用固定 t=0 调制 —— **2511 原生**,`zero_cond_t: true`,
      transformer L898-904 把 timestep 拼成 [t, 0],`modulate_index` 给 img_shapes[1:] 标 1。

两条同时成立 ⇒ ref 的每层 K/V 与 t、与噪声图内容完全解耦 ⇒ 第 0 步算一次、后续步只读。

显存备注(已知,本轮不优化):write 模式的 mask 是 (B,1,L,L) bool。2-ref@1024² 下
L ≈ 12.7k ⇒ 约 161 MB,3-ref 约 285 MB。read 模式只用 key 侧 (B,1,1,L) 的 padding mask,
不吃这份内存——而 79/80 次前向都是 read。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

# env 里的 diffusers(0.40.0.dev0 @ 90c0ffdc)。_vendor/ 只作证据,不参与运行。
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

MODES = ("off", "write", "read")


class RefKVCache:
    """按 block 缓存 ref 段的 K/V(RoPE 与 QK-norm **之后**)。

    存 rope 后的值是必须的:read 模式下序列里没有 ref token,拿不到 ref 的 rope 频率
    (`img_freqs` 虽然算了 ref 那段,但 query 侧不需要,key 侧直接用缓存更省)。
    """

    def __init__(self):
        self.storage: dict[int, tuple[Tensor, Tensor]] = {}

    def write(self, block_idx: int, k: Tensor, v: Tensor) -> None:
        self.storage[block_idx] = (k, v)

    def read(self, block_idx: int) -> tuple[Tensor, Tensor]:
        if block_idx not in self.storage:
            raise KeyError(
                f"block {block_idx} 的 ref K/V 不在缓存里。read 模式前必须先跑一次 write 模式,"
                f"当前已缓存 {len(self.storage)} 层。")
        return self.storage[block_idx]

    def clear(self) -> None:
        self.storage.clear()

    def __len__(self) -> int:
        return len(self.storage)


def build_isolated_attn_mask(
    txt_len: int,
    noise_len: int,
    ref_lens: list[int],
    device: torch.device,
    block_diag: bool = False,
) -> Tensor:
    """(1,1,L,L) bool,True = 允许注意。布局 [txt, noise, ref₁..ref_N]。

    - txt / noise 行全 True:它们能看见一切,包括每张 ref;
    - ref 行只在 ref 段上 True:看不见 txt、看不见噪声图。

    `block_diag=False`(默认,`PLAN.md` §3.1):整段自注意,refs 互相可见,与 BFL
    `FLUX.2-klein-9b-kv` 同构。`block_diag=True`:ref_i 只看自己,与上一轮 UNO 的
    `build_isolated_attn_mask` 同语义。**两种写法都可无损缓存**(ref 段对注意力封闭即可),
    差别是纯质量取舍。本轮只训 `False` 那条臂,开关照写不训。
    """
    base = txt_len + noise_len
    total = base + sum(ref_lens)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:base, :] = True
    if block_diag:
        off = base
        for n in ref_lens:
            mask[off:off + n, off:off + n] = True
            off += n
    else:
        mask[base:, base:] = True
    return mask[None, None]


@dataclass
class IsoContext:
    """随一次 forward 传给所有 block 的共享状态。

    mask 只建一次(60 层共用),`prepare()` 在 fork 出来的 transformer forward 里调用。
    """

    mode: str = "off"
    block_diag: bool = False
    cache: RefKVCache | None = None

    # prepare() 填的
    txt_len: int = 0
    noise_len: int = 0
    ref_lens: list[int] = field(default_factory=list)
    attn_mask: Tensor | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode 只能是 {MODES},收到 {self.mode!r}")
        if self.mode in ("write", "read") and self.cache is None:
            self.cache = RefKVCache()

    @property
    def ref_len(self) -> int:
        return sum(self.ref_lens)

    def prepare(self, txt_len: int, img_shapes: list, pad_mask: Tensor | None,
                device: torch.device) -> None:
        """从**完整** img_shapes(含 ref)算段长并建 mask。

        `img_shapes` 必须始终是完整的那份——read 模式下序列里虽然没有 ref,
        但 RoPE 的 `max_vid_index` 是对 img_shapes 里所有图取的(见 `pipeline_iso` 的注释)。
        """
        shapes = img_shapes[0]
        self.txt_len = txt_len
        self.noise_len = shapes[0][0] * shapes[0][1] * shapes[0][2]
        self.ref_lens = [f * h * w for f, h, w in shapes[1:]]

        if self.mode == "off":
            # 与 stock processor 逐字同义:只有 key 侧的 txt padding mask
            self.attn_mask = self._pad_only_mask(pad_mask, self.noise_len + self.ref_len, device)
            return

        if self.mode == "read":
            # query 只有 [txt, noise],它们本来就能看见一切 ⇒ 行方向无约束,
            # 只留 key 侧 padding。省掉 (L,L) 那份内存,而 read 占 79/80 次前向。
            self.attn_mask = self._pad_only_mask(pad_mask, self.noise_len + self.ref_len, device)
            return

        mask = build_isolated_attn_mask(
            self.txt_len, self.noise_len, self.ref_lens, device, self.block_diag)
        if pad_mask is not None:
            # 别把 stock 那条 padding mask 弄丢了(PLAN §3.1 坑 1)
            pad_cols = torch.ones(pad_mask.shape[0], mask.shape[-1], dtype=torch.bool, device=device)
            pad_cols[:, :self.txt_len] = pad_mask.to(torch.bool)
            mask = mask & pad_cols[:, None, None, :]
        self.attn_mask = mask

    def _pad_only_mask(self, pad_mask: Tensor | None, img_len: int,
                       device: torch.device) -> Tensor | None:
        if pad_mask is None:
            return None
        img_ones = torch.ones(pad_mask.shape[0], img_len, dtype=torch.bool, device=device)
        return torch.cat([pad_mask.to(torch.bool), img_ones], dim=1)[:, None, None, :]


class IsoAttnProcessor:
    """替换 `QwenDoubleStreamAttnProcessor2_0`。

    改了三处,其余逐行照抄 vendored 源码 L488-583:

    1. stock 那句 `raise ValueError("... does not accept an external attention_mask ...")`
       换成从 ctx 取 mask(PLAN §3.1 坑 1);
    2. write 模式把 ref 段的 joint K/V 存进 cache;
    3. read 模式下序列没有 ref,把缓存的 ref K/V 拼回 key/value,
       并把 `img_freqs` 切到噪声段(它是按完整 img_shapes 算的,含 ref 那截)。
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, ctx: IsoContext, block_idx: int):
        self.ctx = ctx
        self.block_idx = block_idx

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        encoder_hidden_states_mask: Tensor = None,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("IsoAttnProcessor 需要 encoder_hidden_states(text stream)")
        ctx = self.ctx
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            if ctx.mode == "read":
                # img_freqs 覆盖 [noise, ref…](按完整 img_shapes 算的),这里只有 noise
                img_freqs = img_freqs[:ctx.noise_len]
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        if ctx.mode == "write":
            base = seq_txt + ctx.noise_len
            ctx.cache.write(self.block_idx, joint_key[:, base:], joint_value[:, base:])
        elif ctx.mode == "read":
            ref_k, ref_v = ctx.cache.read(self.block_idx)
            joint_key = torch.cat([joint_key, ref_k], dim=1)
            joint_value = torch.cat([joint_value, ref_v], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=ctx.attn_mask if attention_mask is None else attention_mask,
            dropout_p=0.0, is_causal=False,
            backend=self._attention_backend, parallel_config=self._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


def install_iso_processors(transformer, ctx: IsoContext) -> None:
    """把每个 block 的 attn processor 换成 `IsoAttnProcessor`,block_idx 即缓存 key。"""
    for i, block in enumerate(transformer.transformer_blocks):
        block.attn.set_processor(IsoAttnProcessor(ctx, i))
