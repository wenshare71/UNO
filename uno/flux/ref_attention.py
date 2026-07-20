# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""隔离注意力 + 参考图 KV-Cache（移植自 OminiControl 的 independent_condition / feature reuse 方案）。

UNO 的 baseline 把 [txt, img, ref_1..ref_N] 拼成单序列做全双向注意力，且 ref token
与主图共享同一 timestep 调制，导致 ref 的 K/V 随去噪步变化，无法缓存。

本模块提供三件事，把 OminiControl 的多分支 group_mask 语义翻译到 UNO 的单序列布局：
1. RefContext    —— 随 forward 传递的上下文（段边界、ref 专用调制向量、注意力掩码、缓存）。
2. RefKVCache    —— 按 block 存 ref 段 rope 后的 K/V；step 0 写入，后续步只读。
3. build_isolated_attn_mask —— bool 掩码：ref 段只能看自己（不看 txt/img/其它 ref），
   txt/img 仍能看到一切。由此 ref 的每层 K/V 与 timestep、主图内容完全解耦，
   缓存后续步复用才是数学上无损的。
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


class RefKVCache:
    """按 block 缓存 ref 段的 K/V（rope 之后）。

    mode:
      - "write": 本次 forward 中每个 attention 把 ref 段 K/V 存进来（step 0）。
      - "read" : 本次 forward 序列里没有 ref token，attention 把缓存的 K/V 拼回。
    """

    def __init__(self):
        self.storage: dict[str, tuple[Tensor, Tensor]] = {}
        self.mode: str | None = None

    def write(self, key: str, k: Tensor, v: Tensor) -> None:
        self.storage[key] = (k, v)

    def read(self, key: str) -> tuple[Tensor, Tensor]:
        return self.storage[key]

    def clear(self) -> None:
        self.storage.clear()


@dataclass
class RefContext:
    """随 Flux.forward 一路传到各 block processor 的上下文。

    ref_len 是当前序列中 ref token 的总数；read 模式下序列不含 ref，ref_len=0。
    processor 用「序列末尾 ref_len 个 token」定位 ref 段做分段调制与缓存写入。
    """

    ref_len: int = 0
    vec_ref: Tensor | None = None          # ref 段专用调制向量（t=0，与去噪步无关）
    attn_mask: Tensor | None = None        # (1,1,L,L) bool，True=允许注意
    kv: RefKVCache | None = None
    ref_lens: list[int] = field(default_factory=list)  # 每张 ref 的 token 数


def build_isolated_attn_mask(
    txt_len: int,
    img_len: int,
    ref_lens: list[int],
    device: torch.device,
) -> Tensor:
    """构建隔离掩码，序列布局 [txt, img, ref_1, ..., ref_N]。

    规则与 OminiControl 的 group_mask 一致：
      - txt/img 行全 True（可以看所有 token，包括每张 ref）；
      - ref_i 行只在自己的段上为 True（不看 txt/img，不看其它 ref）。
    """
    total = txt_len + img_len + sum(ref_lens)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    base = txt_len + img_len
    mask[:base, :] = True
    offset = base
    for ref_len in ref_lens:
        mask[offset:offset + ref_len, offset:offset + ref_len] = True
        offset += ref_len
    return mask[None, None]  # 广播到 (B, heads)
