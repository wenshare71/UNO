# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from einops import rearrange
from torch import Tensor


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor,
    attn_mask: Tensor | None = None,
    ref_kv=None,
    cache_key: str | None = None,
    ref_len: int = 0,
) -> Tensor:
    """FLUX 注意力，支持隔离掩码与 ref KV-Cache。

    默认参数下与原实现逐字节一致。ref_kv 为 RefKVCache 时：
      - write 模式：把序列末尾 ref_len 个 token 的 rope 后 K/V 存入缓存；
      - read 模式：当前序列不含 ref token，把缓存 K/V 拼到末尾供 txt/img 查询。
    缓存的 K 已带 ref 自己的位置编码，read 步的 pe 只覆盖 [txt, img]，两者正交。
    """
    q, k = apply_rope(q, k, pe)

    if ref_kv is not None:
        if ref_kv.mode == "write" and ref_len > 0:
            # 必须 clone：切片是 view，会连带把整个 k/v（含 txt+img 段）钉在显存里活满
            # 整轮去噪。57 个 block 累积下来白占 1.5 GB+，而真正要缓存的只有末尾 ref 段。
            ref_kv.write(cache_key, k[:, :, -ref_len:].clone(), v[:, :, -ref_len:].clone())
        elif ref_kv.mode == "read":
            cached_k, cached_v = ref_kv.read(cache_key)
            k = torch.cat((k, cached_k), dim=2)
            v = torch.cat((v, cached_v), dim=2)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
