#!/usr/bin/env bash
# 8×4090 上训练「隔离注意力」LoRA（推理时可开 --kv_cache 复用 ref K/V）
# 用法: bash scripts/train_ref_isolation.sh   （在 UNO 仓库根目录、激活 .venv-uno 后）
set -euo pipefail
cd "$(dirname "$0")/.."

# RTX 4000 系列必须禁 P2P/IB（train.py 里也有兜底 setdefault）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_processes 8 --mixed_precision bf16 train.py \
    --ref_isolation True \
    --lora_rank 512 \
    --gradient_checkpoint True \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --resolution 512 \
    --learning_rate 8e-5 \
    --max_train_steps 20000 \
    --checkpointing_steps 1000 \
    --train_data_json datasets/UNO-1M/uno_1m_total_labels_convert.json \
    --project_dir log/ref_isolation \
    "$@"
