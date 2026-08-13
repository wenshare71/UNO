"""隔离注意力下的 transformer forward:write / read 两种模式。

规格:`qwen/PLAN.md` §3.1。**只新建本文件,既有 `.py` 一个字不动(R0)。**

这里 fork 了 `QwenImageTransformer2DModel.forward`(vendored 源码 L847-966)。
为什么必须 fork 而不是包一层:read 模式下序列里没有 ref token,而 stock forward 里
有两处按 `img_shapes` 算的东西**长度对不上**——

  1. `modulate_index`(L900):`[0]*prod(shapes[0]) + [1]*Σprod(shapes[1:])`,
     长度 = noise+ref;read 模式 hidden_states 只有 noise ⇒ `_modulate` 里
     `torch.where(index[:,:,None]==0, ...)` 直接广播失败。
  2. `image_rotary_emb`(L925):`img_freqs` 覆盖 noise+ref,同样长了。

而这两处**不能靠"把 ref 从 img_shapes 摘掉"来解决**——见下面 RoPE 那段注释。
所以:img_shapes 永远传完整的,长度对不上的地方在这里各切一刀。

── RoPE 平移(PLAN §3.1 坑 2,已在 vendored 源码上逐行核实)───────────────────

`QwenEmbedRope.forward` L285-293:

    if self.scale_rope:                                    # 2511 是 True(L815 写死)
        max_vid_index = max(height // 2, width // 2, max_vid_index)   # 对**所有**图取
    ...
    txt_freqs = pos_freqs_device[max_vid_index : max_vid_index + max_txt_seq_len_int]

`max_vid_index` 是对 `img_shapes` 里每一张图取的 max,**txt 的 RoPE 起点跟着它走**。
read 模式若顺手把 ref 从 img_shapes 摘掉,max_vid_index 变小 ⇒ txt 频率整体平移 ⇒
写/读两侧的 txt 表示对不上。这是静默 bug,只体现成质量下降。

⚠️ 在**我们这批数据上它恰好不发作**:dreambooth 158 张全是正方形(本地实测),
`calculate_dimensions(1024², 1) = (1024, 1024)` ⇒ 每张图都是 (1,64,64) ⇒
带不带 ref,max_vid_index 都是 32。**但这是数据的巧合,不是实现的正确性。**
换一批非方形参考图就炸,所以照样按完整 img_shapes 算。`test_iso_equiv.py` 里
用一张非方形 ref 把这条钉成回归测试。
"""

from __future__ import annotations

import os
import sys
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask

from iso_attn import IsoContext, RefKVCache, install_iso_processors


def iso_transformer_forward(
    transformer,
    ctx: IsoContext,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    encoder_hidden_states_mask: Tensor | None,
    timestep: Tensor,
    img_shapes: list,
) -> Tensor:
    """fork 自 `QwenImageTransformer2DModel.forward`。

    `hidden_states`:write/off 模式是 `cat([latents, image_latents], dim=1)`,
    read 模式只有 `latents`。**`img_shapes` 两种模式都传完整的那份。**

    砍掉的分支(2511 用不到,留着只会让人以为它们在起作用):
    `guidance`(config `guidance_embeds: false`)、`controlnet_block_samples`、
    `additional_t_cond`(config 无 `use_additional_t_cond`)、梯度检查点
    (训练时再加,`train_iso.py` 的事)。
    """
    if not transformer.zero_cond_t:
        raise SystemExit("❌ zero_cond_t 未生效。隔离注意力的地基是 ref 段 t=0 调制,"
                         "没有它缓存从无损降级成近似,整条路线不成立。")

    hs = transformer.img_in(hidden_states)
    timestep = timestep.to(hs.dtype)
    timestep = torch.cat([timestep, timestep * 0], dim=0)

    ehs = transformer.txt_norm(encoder_hidden_states)
    ehs = transformer.txt_in(ehs)
    text_seq_len, _, ehs_mask = compute_text_seq_len_from_mask(ehs, encoder_hidden_states_mask)

    ctx.prepare(text_seq_len, img_shapes, ehs_mask, hs.device)

    if ctx.mode == "read":
        # 剩下的全是噪声图 token ⇒ 调制 index 全 0(走真实 t 那一半)
        modulate_index = torch.zeros((hs.shape[0], ctx.noise_len),
                                     device=hs.device, dtype=torch.int)
        if hs.shape[1] != ctx.noise_len:
            raise ValueError(f"read 模式下 hidden_states 应只含噪声图 {ctx.noise_len} 个 token,"
                             f"实际 {hs.shape[1]}。ref 不该出现在序列里。")
    else:
        modulate_index = torch.tensor(
            [[0] * prod(s[0]) + [1] * sum(prod(x) for x in s[1:]) for s in img_shapes],
            device=hs.device, dtype=torch.int)

    temb = transformer.time_text_embed(timestep, hs)
    # ← img_shapes 完整传入,max_vid_index 才不会漂(见文件头注释)
    image_rotary_emb = transformer.pos_embed(img_shapes, max_txt_seq_len=text_seq_len,
                                             device=hs.device)

    for block in transformer.transformer_blocks:
        ehs, hs = block(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            encoder_hidden_states_mask=ehs_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=None,
            modulate_index=modulate_index,
        )

    temb = temb.chunk(2, dim=0)[0]
    hs = transformer.norm_out(hs, temb)
    return transformer.proj_out(hs)


class IsoRunner:
    """把 write / read 的调度收在一处:第 0 次前向写缓存,之后只读。

    `PLAN.md` §1-2:ref 段看不见 txt ⇒ cond / uncond 两次前向的 ref K/V 逐位相同,
    **共享同一份缓存**。Q1 口径 40 步 × true_cfg ⇒ 80 次前向,1 次写、79 次读。
    """

    def __init__(self, transformer, block_diag: bool = False):
        self.transformer = transformer
        self.ctx = IsoContext(mode="off", block_diag=block_diag, cache=RefKVCache())
        install_iso_processors(transformer, self.ctx)

    def reset(self) -> None:
        self.ctx.cache.clear()

    def __call__(self, latents: Tensor, image_latents: Tensor | None, encoder_hidden_states: Tensor,
                 encoder_hidden_states_mask: Tensor | None, timestep: Tensor, img_shapes: list,
                 mode: str) -> Tensor:
        """返回**噪声图 token 位置**上的 velocity,形状 (B, noise_len, patch²·out_ch)。

        ref 位置的输出 stock pipeline 本来就用 `noise_pred[:, :latents.size(1)]` 丢掉
        (pipeline L826),这里直接不返回,免得下游忘了切。
        """
        self.ctx.mode = mode
        if mode == "read":
            hs = latents
        else:
            hs = latents if image_latents is None else torch.cat([latents, image_latents], dim=1)
        out = iso_transformer_forward(self.transformer, self.ctx, hs, encoder_hidden_states,
                                      encoder_hidden_states_mask, timestep, img_shapes)
        return out[:, :latents.shape[1]]
