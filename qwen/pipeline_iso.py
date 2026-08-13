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
    `additional_t_cond`(config 无 `use_additional_t_cond`)。梯度检查点保留了。
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

    # 12.7k token × 60 层的激活装不下,训练必开(PLAN §3.2 起手配置)。
    # 参数顺序照抄 stock L929-938,位置参数,错一个就静默传错。
    use_ckpt = torch.is_grad_enabled() and transformer.gradient_checkpointing
    for block in transformer.transformer_blocks:
        if use_ckpt:
            ehs, hs = transformer._gradient_checkpointing_func(
                block, hs, ehs, ehs_mask, temb, image_rotary_emb, None, modulate_index)
        else:
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

    def __init__(self, transformer, block_diag: bool = False, store: bool = True):
        """`store=False`:write 模式下不往 cache 里存。训练侧用这个,理由见
        `student_forward`。推理侧走 `IsoPipelineHook`,它必须存。"""
        self.transformer = transformer
        self.ctx = IsoContext(mode="off", block_diag=block_diag, cache=RefKVCache(), store=store)
        install_iso_processors(transformer, self.ctx)

    def reset(self) -> None:
        self.ctx.cache.clear()

    # ------------------------------------------------------------------ 训练侧
    def teacher_forward(self, latents: Tensor, image_latents: Tensor, encoder_hidden_states: Tensor,
                        encoder_hidden_states_mask: Tensor | None, timestep: Tensor,
                        img_shapes: list) -> Tensor:
        """全注意力、无缓存 —— `off` 模式。T1 已证它与 stock forward 逐位相同。"""
        return self(latents, image_latents, encoder_hidden_states, encoder_hidden_states_mask,
                    timestep, img_shapes, mode="off")

    def student_forward(self, latents: Tensor, image_latents: Tensor, encoder_hidden_states: Tensor,
                        encoder_hidden_states_mask: Tensor | None, timestep: Tensor,
                        img_shapes: list) -> Tensor:
        """隔离注意力、不走缓存 —— `write` 模式。

        训练时**故意不用缓存**:一次前向只有一个 t,缓存省不下任何东西,
        而 write 模式与 read 模式的输出已由 T3 证明相同(判据 max<1e-5,实测 0)。
        走 write 还能让 ref 段拿到梯度——它们的 K/V 是 LoRA 的输出,要参与反传。

        缓存靠构造时的 `store=False` 关掉,**不是"forward 之后 clear 一下"**:
        梯度检查点反传时会重跑每个 block,重跑的还是 write 分支、还会往 cache 里写,
        写在 clear 之后 ⇒ 一步下来 60 层 ref K/V 照样挂着(2-ref 约 6 GB,3-ref 约 9 GB)。
        同理也不能在 forward 里临时翻转再复原——复原发生在反传之前。
        """
        return self(latents, image_latents, encoder_hidden_states, encoder_hidden_states_mask,
                    timestep, img_shapes, mode="write")

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


class IsoPipelineHook:
    """把 write / read 调度挂进 stock `QwenImageEditPlusPipeline`,**不重写 `__call__`**。

    为什么是挂钩而不是继承重写:P3 的判据是"隔离臂 vs Q1 口径",而 Q1 口径就是
    `pipe(...)` 那条代码路径本身。去噪循环里除了 transformer 还有三样会影响结果——

      · sigma 网格(`calculate_shift` 按 `latents.shape[1]` 动态 shift,pipeline L765);
      · true_cfg 之后的**范数重标定** `comb_pred * (cond_norm / noise_norm)`(L843-845);
      · `scheduler.step` 的 Euler 更新。

    重写 `__call__` 就等于把这三样也抄一遍,抄错一个字基线就废了(§5:改了基线就废了)。
    换 `transformer.forward` 则让这三样逐字沿用 stock 代码,差异**只**在 transformer 内部。

    调度:每张图第一次前向写缓存,其余 79 次读。cond / uncond 共享同一份——
    T6 已证 ref K/V 与 prompt 无关。**每张图之间必须 `reset()`**,否则会读到上一张的 ref。
    """

    def __init__(self, pipe, block_diag: bool = False, always_write: bool = False):
        """`always_write=True`:隔离但**不缓存**,每一步都重算 ref K/V。

        它存在只为一个用途——真权重 bf16 下的「隔离-无缓存 vs 隔离-有缓存」对照,
        即 `PLAN.md` §3.1 门禁引的那条先例(`../scripts/bench_kv_cache.py`,像素差判据)。
        CPU fp32 那一遍(T3)已经把结构证死了,这一遍是确认 bf16 数值和 60 层累积。
        """
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.always_write = always_write
        self.ctx = IsoContext(mode="write", block_diag=block_diag, cache=RefKVCache())
        install_iso_processors(self.transformer, self.ctx)
        self._orig_forward = self.transformer.forward
        self.transformer.forward = self._forward
        self.n_write = 0
        self.n_read = 0

    def reset(self) -> None:
        """每张图开跑前调。不调 = 第 2 张图读到第 1 张的 ref K/V,而且**不会报错**。"""
        self.ctx.cache.clear()

    def detach(self) -> None:
        """还原 forward。同一进程里跑完隔离臂再跑 full 臂时用。"""
        self.transformer.forward = self._orig_forward

    def _forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                 timestep=None, img_shapes=None, guidance=None, attention_kwargs=None,
                 controlnet_block_samples=None, additional_t_cond=None, return_dict=True, **kw):
        if guidance is not None:
            raise ValueError("2511 的 config 是 guidance_embeds: false,不该收到 guidance。"
                             "收到了说明换了权重,停下来看。")
        if controlnet_block_samples is not None or additional_t_cond is not None:
            raise ValueError("iso forward 砍掉了 controlnet / additional_t_cond 分支,2511 用不到。")

        first = self.always_write or len(self.ctx.cache) == 0
        self.ctx.mode = "write" if first else "read"
        if first:
            self.n_write += 1
            hs = hidden_states
        else:
            self.n_read += 1
            # pipeline 每步都传 cat([latents, image_latents]);读模式只要噪声段那截
            hs = hidden_states[:, :prod(img_shapes[0][0])]

        out = iso_transformer_forward(self.transformer, self.ctx, hs, encoder_hidden_states,
                                      encoder_hidden_states_mask, timestep, img_shapes)
        if self.always_write:
            self.ctx.cache.clear()
        if not return_dict:
            return (out,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=out)
