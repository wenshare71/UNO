# UNO 冒烟测试报告

- **日期**：2026-07-20
- **作者**：Claude (协助 wuwenxuan03)
- **目标**：在 8×RTX 4090 节点上完成 UNO 推理链路冒烟测试，验证模型权重 + 数据集 + 推理管道是否端到端可用
- **状态**：✅ 全部通过

---

## 1. 环境

### 1.1 硬件

| 项目 | 规格 |
|---|---|
| GPU | 8× NVIDIA GeForce RTX 4090（24 GB GDDR6X each = 192 GB 总量） |
| CPU | 256 cores |
| 磁盘 | `/home/wuwenxuan03` 挂载在 NFS `10.80.201.41:/mmu_ssd/wuwenxuan03`（1.0 TB, 已用 249 G, 剩 776 G） |
| `/root` | 7.0 TB 容器 overlay（HF 缓存所在） |
| 内核 | Linux 4.18.0-2.4.3.3.kwai.x86_64（**低于 PyTorch 推荐的 5.5.0**，运行时会有 warning，但未造成 hang） |

### 1.2 软件（`.venv-uno`）

| 包 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.13 | — |
| PyTorch | 2.4.0+cu124 | CUDA 12.4，cuDNN 90100 |
| transformers | **4.43.3** | 已降回 res.txt 钉死的版本（自检时装到 5.14.1，太新） |
| diffusers | **0.30.1** | 已降回（自检时装到 0.39.0，太新） |
| accelerate | 1.1.1 | — |
| safetensors | 0.8.0 | — |

### 1.3 关键环境变量

```bash
CUDA_VISIBLE_DEVICES=0                # 单卡冒烟，多卡训练时去掉
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 # RTX 4000 系列必须，否则 accelerate 抛 NotImplementedError
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 2. 模型权重下载

缓存位置：`~/.cache/huggingface/hub/`

| 仓库 | 大小 | 关键文件 | 状态 |
|---|---|---|---|
| `black-forest-labs/FLUX.1-dev` | **54 GB** | `flux1-dev.safetensors`、`ae.safetensors`、`text_encoder/`、`text_encoder_2/`、`transformer/`、`vae/`、`tokenizer/`、`tokenizer_2/`、`scheduler/`、`model_index.json` | ✅ |
| `bytedance-research/UNO` | 1.8 GB | `dit_lora.safetensors` (1.91 G) + `assets/` | ✅ |
| `xlabs-ai/xflux_text_encoders` | 8.9 GB | `model-00001-of-00002.safetensors` + `…00002…` + `spiece.model` (T5-XXL) | ✅ |
| `openai/clip-vit-large-patch14` | 6.4 GB | `model.safetensors` + `pytorch_model.bin` + `tf_model.h5` + `flax_model.msgpack`（全格式，占空间） | ✅ |

> **关于 FLUX.1-dev 偏大（54 GB vs 预估 24 GB）**：`text_encoder_2/` 里其实把 T5-XXL (~9.5 GB) 也一起下了一份。diffusers 加载流程优先用 FLUX.1-dev 自带的 `t5xxl_fp16` / `t5xxl_fp8_e4m3fn`，XLabs 那份是备选。训练时若 VRAM 紧张可改 `T5` 环境变量指 XLabs，或反之。

下载方式：海外代理 + `hf_transfer`，4 个仓库累计约 3-4 小时。

---

## 3. 数据集下载

位置：`/home/wuwenxuan03/UNO/datasets/UNO-1M/`

| 文件 | 大小 | 状态 |
|---|---|---|
| `uno_1m_total_labels.json` | 810 MB | ✅ |
| `images/split1.tar.gz` | 23.9 GB | ✅ |
| `images/split2.tar.gz` | 23.9 GB | ✅ |
| `images/split3.tar.gz` | 25.4 GB | ✅ |
| `images/split4.tar.gz` | 24.9 GB | ✅ |
| `images/split5.tar.gz` | 25.2 GB | ✅ |
| **合计** | **~124 GB（压缩态）** | |

> ⚠️ **训练前需先解包**：`cd datasets/UNO-1M/images && for f in split*.tar.gz; do tar -xzf "$f"; done`，解压后预计再加 ~125 GB，确认 `df` 空间后再执行。
>
> ⚠️ **labels 是 810 MB 单文件 JSON**：`json.load()` 慢且吃内存（>8 GB），训练 dataloader 应改用 `ijson` 流式或 `json.loads(line)` per-line 风格。

---

## 4. 冒烟测试

### 4.1 命令

```bash
cd /home/wuwenxuan03/UNO
source .venv-uno/bin/activate

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_ENABLE_HF_TRANSFER=1 \
python -u inference.py \
  --prompt "a cat sitting in a cozy cafe" \
  --image_paths assets/cat_cafe.png \
  --width 512 --height 512 \
  --num_steps 25 \
  --offload \
  --model_type flux-dev-fp8 \
  --save_path output/inference/smoke
```

### 4.2 参数说明

| 参数 | 值 | 备注 |
|---|---|---|
| `--model_type` | `flux-dev-fp8` | **关键**：bf16 版 24 GB 装不进 24 GB 4090，fp8 边加载边转约 12 GB |
| `--offload` | True | CPU 卸载，进一步省 VRAM |
| `--width/--height` | 512/512 | 冒烟够用，训练时按需放大（704/704 是单参考图常用） |
| `--num_steps` | 25 | 与官方推理一致 |
| `--only_lora` | True（默认） | **名字有迷惑性**：仅指 LoRA 可训练，base FLUX 仍要全量加载到 VRAM |

### 4.3 执行结果

| 阶段 | 结果 |
|---|---|
| T5 shards 加载 | ✅ 2/2 shards, 11.26 it/s |
| AE / model init | ✅ |
| UNO LoRA 加载 | ✅ (`dit_lora.safetensors`) |
| FLUX.1-dev 主权重 → fp8 转换 | ✅ on-the-fly |
| 25 步去噪 | ✅ 25/25 用时 **~11s**（~2.3 it/s） |
| 保存 PNG + JSON | ✅ `output/inference/smoke/0_0.png` (270 KB) |
| GPU 0 峰值显存 | **~10 GB**（fp8 + offload） |

### 4.4 产物

- **图片**：`output/inference/smoke/0_0.png`（512×512）
  - 一只虎斑白猫坐在咖啡馆木桌上，背景暖色调，氛围与参考图 `assets/cat_cafe.png` 一致
  - 猫的毛色、姿态明显受 ref image 条件通路影响 → ref 条件工作正常
- **配置快照**：`output/inference/smoke/0_0.json`（494 B，含全部推理参数）
- **完整日志**：`output/inference/smoke.log`（2.3 KB）

---

## 5. 踩到的坑（要记下来）

### 5.1 RTX 4090 + accelerate 必须禁 P2P/IB

第一次裸跑直接挂在 `Accelerator()` 构造：

```
NotImplementedError: Using RTX 4000 series doesn't support faster communication
broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1"`
or use `accelerate launch` which will do this automatically.
```

`accelerate >= 0.27` 在 4000 系列上**单卡**也会触发。后续所有脚本（`inference.py`、`train.py`、评测脚本）都要在环境变量里钉死。

### 5.2 `--only_lora` 不省 VRAM

`UNOPipeline(..., only_lora=True)` 名字误导——它只是把 LoRA 设为可训练参数，**base FLUX.1-dev (24 GB bf16) 仍要完整装进 VRAM**，然后 `sd.update(lora_sd)` 在加载时把 LoRA 权重合并进去。24 GB 4090 上：

| 方案 | 峰值 VRAM | 备注 |
|---|---|---|
| `flux-dev` + `--offload` | ~20 GB | 推理慢（CPU↔GPU 来回搬） |
| `flux-dev-fp8` + `--offload` | **~10 GB** ✅ | 边加载边转，推荐 |
| `flux-dev` 裸跑 | OOM | 24 GB 装不下 |

> 多卡训练不受影响：8×4090 (192 GB) 装 FLUX-dev 全精度富余。

### 5.3 内核 4.18 < PyTorch 推荐 5.5

启动时 warning：
```
Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0;
this can cause the process to hang.
```
**未实际造成 hang**，但 dev/SA 同学后续踩坑的话要记得这是已知环境噪声，不是真 hang。

---

## 6. 端到端验证清单

| 通路 | 验证状态 |
|---|---|
| FLUX.1-dev 权重加载 | ✅ |
| T5-XXL 文本编码器（XLabs 版） | ✅ |
| CLIP-L 文本编码器 | ✅ |
| UNO 官方 LoRA 权重 | ✅ |
| 参考图条件通路 | ✅（成片明显受 ref 影响） |
| fp8 量化推理路径 | ✅ |
| 单卡 512px × 25 步推理耗时 | ✅ ~11s |

---

## 7. 后续建议

环境、模型、数据三件套已齐。下一步两条主线，可并行：

1. **改造训练管道**：把 OminiControl 的隔离注意力 + KV-Cache LoRA 移植到 UNO
   - 改 `uno/flux/modules/attention.py` 加 attention mask
   - 改 `uno/flux/sampling.py` 在 denoise 循环里注入 ref KV 缓存
   - 改 `train.py` 配 8 卡 + LoRA + gradient checkpointing
   - **不依赖数据解压**和后续步骤，可立刻开干

2. **解压 UNO-1M 数据集 + dataloader 验证**
   - 挂在另一个 tmux 慢慢解包（~124 GB → ~125 GB）
   - 写一个精简 dataloader，先 `for batch in dataloader: print(batch.shape, batch.dtype)` 验证 shape/dtype
   - 端到端数据通路也确认后，再上正式训练

任一主线需要我开始动手，告诉我哪条先。
