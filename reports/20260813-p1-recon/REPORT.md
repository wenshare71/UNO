# P1 侦察 · 执行报告

> 对应 `qwen/P1_RECON_RUN.md`。本轮零 GPU、零风险:只取东西、只清点。
> 一节对一节(§1 A / §2 B / §3 C / §4 D),命令原样 + 输出原样,不转述。
> 执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),执行时间 2026-08-13。

---

## 1. A · diffusers 源码快照 → `qwen/_vendor/`

### 1.1 五个文件拷贝

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
SP=$E/lib/python3.11/site-packages
D=$R/qwen/_vendor/diffusers_0.40.0.dev0
mkdir -p $D/models/transformers $D/models $D/pipelines/qwenimage $D/schedulers
# 逐个 cp
```

输出:

```text
OK  models/transformers/transformer_qwenimage.py
OK  pipelines/qwenimage/pipeline_qwenimage_edit_plus.py
OK  schedulers/scheduling_flow_match_euler_discrete.py
OK  models/normalization.py
OK  models/attention_dispatch.py
```

**五个文件全部取到**(`models/attention_dispatch.py` 存在,147586 B)。落盘清单:

```text
qwen/_vendor/diffusers_0.40.0.dev0/
├── models/
│   ├── attention_dispatch.py          147586 B
│   ├── normalization.py                24518 B
│   └── transformers/
│       └── transformer_qwenimage.py    44061 B
├── pipelines/
│   └── qwenimage/
│       └── pipeline_qwenimage_edit_plus.py  43960 B
└── schedulers/
    └── scheduling_flow_match_euler_discrete.py  26832 B
```

### 1.2 前两个文件顶部 import 行原样

#### `models/transformers/transformer_qwenimage.py`(`sed -n '1,60p'`)

```python
# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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

import math
from math import prod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from ...utils.torch_utils import lru_cache_unless_export, maybe_allow_in_graph
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
```

#### `pipelines/qwenimage/pipeline_qwenimage_edit_plus.py`(`sed -n '1,60p'`)

```python
# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

import inspect
import math
from typing import Any, Callable

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import QwenImageLoraLoaderMixin
from ...models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import QwenImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import QwenImageEditPlusPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... ).convert("RGB")
        >>> prompt = (
        ...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
        ... )
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
```

### 1.3 两份 `ls -1` 目录清单

`$SP/diffusers/pipelines/qwenimage/`:

```text
__init__.py
pipeline_output.py
pipeline_qwenimage_controlnet_inpaint.py
pipeline_qwenimage_controlnet.py
pipeline_qwenimage_edit_inpaint.py
pipeline_qwenimage_edit_plus.py
pipeline_qwenimage_edit.py
pipeline_qwenimage_img2img.py
pipeline_qwenimage_inpaint.py
pipeline_qwenimage_layered.py
pipeline_qwenimage.py
__pycache__
```

`$SP/diffusers/models/transformers/`:

```text
ace_step_transformer.py
auraflow_transformer_2d.py
cogvideox_transformer_3d.py
consisid_transformer_3d.py
dit_transformer_2d.py
dual_transformer_2d.py
hunyuan_transformer_2d.py
__init__.py
latte_transformer_3d.py
lumina_nextdit2d.py
pixart_transformer_2d.py
prior_transformer.py
__pycache__
sana_transformer.py
stable_audio_transformer.py
t5_film_transformer.py
transformer_2d_dreamlite.py
transformer_2d.py
transformer_allegro.py
transformer_anyflow_far.py
transformer_anyflow.py
transformer_bria_fibo.py
transformer_bria.py
transformer_chroma.py
transformer_chronoedit.py
transformer_cogview3plus.py
transformer_cogview4.py
transformer_cosmos3.py
transformer_cosmos.py
transformer_easyanimate.py
transformer_ernie_image.py
transformer_flux2.py
transformer_flux.py
transformer_glm_image.py
transformer_helios.py
transformer_hidream_image.py
transformer_hunyuanimage.py
transformer_hunyuan_video15.py
transformer_hunyuan_video_framepack.py
transformer_hunyuan_video.py
transformer_ideogram4.py
transformer_joyimage_edit_plus.py
transformer_joyimage.py
transformer_kandinsky.py
transformer_krea2.py
transformer_longcat_audio_dit.py
transformer_longcat_image.py
transformer_ltx2.py
transformer_ltx.py
transformer_lumina2.py
transformer_minimax_h3.py
transformer_mochi.py
transformer_motif_video.py
transformer_nucleusmoe_image.py
transformer_omnigen.py
transformer_ovis_image.py
transformer_prx.py
transformer_qwenimage.py
transformer_sana_video.py
transformer_sd3.py
transformer_skyreels_v2.py
transformer_temporal.py
transformer_wan_animate.py
transformer_wan.py
transformer_wan_vace.py
transformer_z_image.py
```

### 1.4 diffusers 确切来源

```bash
ls -d $SP/diffusers*.dist-info && cat $SP/diffusers*.dist-info/METADATA | head -20
```

```text
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/diffusers-0.40.0.dev0.dist-info
Metadata-Version: 2.4
Name: diffusers
Version: 0.40.0.dev0
Summary: State-of-the-art diffusion in PyTorch.
Home-page: https://github.com/huggingface/diffusers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/diffusers/graphs/contributors)
Author-email: diffusers@huggingface.co
License: Apache 2.0 License
Keywords: deep learning diffusion pytorch stable diffusion audioldm
Classifier: Development Status :: 5 - Production/Stable
Classifier: Intended Audience :: Developers
Classifier: Intended Audience :: Education
Classifier: Intended Audience :: Science/Research
Classifier: License :: OSI Approved :: Apache Software License
Classifier: Operating System :: OS Independent
Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence
Classifier: Programming Language :: Python
Classifier: Programming Language :: Python :: 3
Classifier: Programming Language :: Python :: 3.8
Classifier: Programming Language :: Python :: 3.9
Classifier: Programming Language :: Python :: 3.10
```

`direct_url.json`:

```json
{"url": "https://github.com/huggingface/diffusers", "vcs_info": {"commit_id": "90c0ffdc045902a3667d473d2fbfc03e8716dba9", "vcs": "git"}}
```

`_version.py`:**没有**。

`git -C $SP/diffusers log -1`:**(不是 git checkout)**。

**来源结论(原样事实,非推断):pip 直接从 GitHub 安装,commit `90c0ffdc045902a3667d473d2fbfc03e8716dba9`。**

---

## 2. B · 权重侧 config → `qwen/_vendor/qwen2511_config/`

```bash
W=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
C=$R/qwen/_vendor/qwen2511_config
```

输出:

```text
OK  scheduler/scheduler_config.json -> scheduler_config.json
OK  text_encoder/config.json -> text_encoder_config.json
OK  vae/config.json -> vae_config.json
OK  transformer/config.json -> transformer_config.json
OK  model_index.json -> model_index.json
```

**五个 JSON 全部取到,无缺失。**

`find $W -maxdepth 2 -not -name "*.safetensors" | sort`:

```text
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/.gitattributes
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/model_index.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/added_tokens.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/chat_template.jinja
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/merges.txt
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/preprocessor_config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/special_tokens_map.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/tokenizer_config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/tokenizer.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/video_preprocessor_config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor/vocab.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/README.md
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/scheduler
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/scheduler/scheduler_config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/text_encoder
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/text_encoder/config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/text_encoder/generation_config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/text_encoder/model.safetensors.index.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer/added_tokens.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer/chat_template.jinja
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer/merges.txt
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer/special_tokens_map.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer/tokenizer_config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer/vocab.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/transformer
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/transformer/config.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model.safetensors.index.json
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/vae
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/vae/config.json
```

`du -sh $W/*`:

```text
1.0K	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/model_index.json
16M	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/processor
7.5K	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/README.md
512	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/scheduler
16G	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/text_encoder
4.9M	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/tokenizer
39G	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/transformer
243M	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511/vae
```

---

## 3. C · 环境与机器清点

落盘文件:

- `reports/20260813-p1-recon/pip_freeze.txt`(231 行,全量未过滤)
- `reports/20260813-p1-recon/nvidia-smi.txt`(48 行)

### 3.1 关键包版本

命令:

```bash
$E/bin/python - <<'PY'
import importlib.metadata as m
for pkg in ["torch","diffusers","transformers","accelerate","peft","deepspeed","bitsandbytes","safetensors","xformers","flash-attn","numpy","Pillow"]:
    try:
        print(f"{pkg}: {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"{pkg}: 未安装")
PY
```

输出原样:

```text
torch: 2.5.1+cu124
diffusers: 0.40.0.dev0
transformers: 5.14.1
accelerate: 1.14.0
peft: 0.20.0
deepspeed: 0.16.4
bitsandbytes: 未安装
safetensors: 0.8.0
xformers: 未安装
flash-attn: 2.7.4.post1
numpy: 1.26.4
Pillow: 10.2.0
```

汇总表:

| 包 | 版本 |
|---|---|
| torch | 2.5.1+cu124 |
| diffusers | 0.40.0.dev0 |
| transformers | 5.14.1 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| deepspeed | 0.16.4 |
| bitsandbytes | **未安装** |
| safetensors | 0.8.0 |
| xformers | **未安装** |
| flash-attn | 2.7.4.post1 |
| numpy | 1.26.4 |
| Pillow | 10.2.0 |

### 3.2 python / glibc / 内核

```text
$E/bin/python -V        → Python 3.11.15
ldd --version | head -1 → ldd (Ubuntu GLIBC 2.31-0ubuntu9.14) 2.31
uname -a                → Linux aiplatform-bjy-ge47-391.idchb1az2.hb1.kwaidc.com 4.18.0-2.4.3.3.kwai.x86_64 #1 SMP Wed Apr 6 06:31:51 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
```

### 3.3 当前 4090 开发机实况

`nvidia-smi --query-gpu=index,name,memory.total --format=csv`:

```text
index, name, memory.total [MiB]
0, NVIDIA GeForce RTX 4090, 24564 MiB
1, NVIDIA GeForce RTX 4090, 24564 MiB
2, NVIDIA GeForce RTX 4090, 24564 MiB
3, NVIDIA GeForce RTX 4090, 24564 MiB
4, NVIDIA GeForce RTX 4090, 24564 MiB
5, NVIDIA GeForce RTX 4090, 24564 MiB
6, NVIDIA GeForce RTX 4090, 24564 MiB
7, NVIDIA GeForce RTX 4090, 24564 MiB
```

`df -h`(三行):

```text
Filesystem                                               Size  Used Avail Use% Mounted on
10.80.201.108,10.84.154.28,10.84.170.20:/kaimm-distill/  3.0P  2.7P  351T  89% /kaimm-distill
overlay                                                  7.0T  268G  6.8T   4% /
tmpfs                                                    504G     0  504G   0% /dev/shm
```

`free -g`(一行):

```text
              total        used        free      shared  buff/cache   available
Mem:           1007          19         918           0          69         982
```

`import torch` 实测(4090 上):

```text
2.5.1+cu124 True 8
```

(另有一条 `FutureWarning: The pynvml package is deprecated...`,stderr,不影响。)

**`qwen-edit` env 在 4090 上 import torch 实测通过。** nvidia-smi 原样全文见 `nvidia-smi.txt`(8 卡全部 idle,0 进程)。

---

## 4. D · 训练数据清点

### 4.1 定位与判据

```bash
find /kaimm-distill/wuwenxuan -name "manifest_raw.json" -o -name "manifest_filtered.json" 2>/dev/null
```

```text
/kaimm-distill/wuwenxuan/UNO/datasets/distill_multiref/manifest_raw.json
/kaimm-distill/wuwenxuan/UNO/datasets/distill_multiref/manifest_filtered.json
```

```text
-rw-rw-r-- 1 wuwenxuan03 wuwenxuan03 4452044 7月  29 11:44 manifest_raw.json
-rw-rw-r-- 1 wuwenxuan03 wuwenxuan03 1355665 7月  29 13:23 manifest_filtered.json
```

**判据:`manifest_raw.json` = 4,452,044 B ≈ 4.25 MB ≤ 20 MB ⇒ 判据为真,放行入 git。**
`.gitignore` 末尾已追加规格 §6 给定的放行块(逐字),`git status` 实测可见(`??`)。
`manifest_filtered.json` 未拷、未放行,条数见 §4.2 末尾。

### 4.2 统计(`$E/bin/python` 现场算,输出原样)

```text
===== 1. 顶层结构 =====
type: list
总条数: 9000
条目 type: dict | keys: ['image_paths', 'prompt', 'image_tgt_path', 'meta']

===== 2. 前 3 条原样 =====
{
  "image_paths": [
    "../dreambooth/dataset/backpack/02.jpg",
    "../dreambooth/dataset/cat/03.jpg"
  ],
  "prompt": "a backpack and a cat in the jungle",
  "image_tgt_path": "images/000000.jpg",
  "meta": {
    "subjects": [
      "backpack",
      "cat"
    ],
    "view_idx": [
      2,
      3
    ],
    "seed": 3407000,
    "template_id": 0,
    "n_refs": 2,
    "has_animal": true
  }
}
{
  "image_paths": [
    "../dreambooth/dataset/backpack/02.jpg",
    "../dreambooth/dataset/cat/04.jpg"
  ],
  "prompt": "a backpack and a cat in the jungle",
  "image_tgt_path": "images/000001.jpg",
  "meta": {
    "subjects": [
      "backpack",
      "cat"
    ],
    "view_idx": [
      2,
      4
    ],
    "seed": 3407001,
    "template_id": 0,
    "n_refs": 2,
    "has_animal": true
  }
}
{
  "image_paths": [
    "../dreambooth/dataset/backpack/01.jpg",
    "../dreambooth/dataset/cat/04.jpg"
  ],
  "prompt": "a backpack and a cat in the snow",
  "image_tgt_path": "images/000002.jpg",
  "meta": {
    "subjects": [
      "backpack",
      "cat"
    ],
    "view_idx": [
      1,
      4
    ],
    "seed": 3407002,
    "template_id": 1,
    "n_refs": 2,
    "has_animal": true
  }
}

===== 3. meta.n_refs 分布 =====
  n_refs=1: 1000
  n_refs=2: 4000
  n_refs=3: 4000

===== 4. meta.subjects 主体频次 =====
  shiny_sneaker: 1536
  backpack: 1531
  teapot: 1529
  vase: 1501
  red_cartoon: 1487
  pink_sunglasses: 1481
  wolf_plushie: 1481
  rc_car: 1080
  poop_emoji: 1074
  monster_toy: 1055
  robot_toy: 1054
  cat2: 951
  cat: 931
  dog2: 627
  dog7: 618
  dog8: 617
  dog5: 614
  dog: 613
  dog3: 611
  dog6: 609
  主体总数: 20

===== 5. held-out 泄漏断言 =====
落在 HELD_OUT 里的条数(期望 0): 0
不在 TRAIN∪HELD_OUT 里的主体名: (无)

===== manifest_filtered.json 条数 =====
type: list | 条数: 3079
```

**原样事实(非结论)**:
- n_refs 分布与期望逐字一致(1-ref 1000 / 2-ref 4000 / 3-ref 4000)。
- held-out 泄漏条数 = **0**;无 TRAIN∪HELD_OUT 之外的主体名。
- 主体 20 个,与 TRAIN 名单 20 个逐一对应。

### 4.3 x₀ 目标图

绝对路径:`/kaimm-distill/wuwenxuan/UNO/datasets/distill_multiref/images`

```text
张数:     9000
总大小:   573M
扩展名:   9000 jpg(无其他)
挂载点:   10.80.201.108,10.84.154.28,10.84.170.20:/kaimm-distill/  3.0P  2.7P  351T  89% /kaimm-distill
```

抽 5 张(排序后第 1 / 2250 / 4500 / 6750 / 9000 张,PIL 实测):

```text
排序后总数: 9000
第    1 张: images/000000.jpg   size=(512, 512) mode=RGB
第 2250 张: images/002249.jpg   size=(512, 512) mode=RGB
第 4500 张: images/004499.jpg   size=(512, 512) mode=RGB
第 6750 张: images/006749.jpg   size=(512, 512) mode=RGB
第 9000 张: images/008999.jpg   size=(512, 512) mode=RGB
```

5 张抽样全部 512×512 RGB。**图一张都没拷回仓库。**

---

## 未取到

无。§1–§4 全部项目均取到,含规格预判可能不存在的 `models/attention_dispatch.py`(实际存在,已拷)。
