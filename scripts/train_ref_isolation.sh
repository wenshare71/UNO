#!/usr/bin/env bash
# 8×4090 上训练「隔离注意力」LoRA（推理时可开 --kv_cache 复用 ref K/V）
# 用法: bash scripts/train_ref_isolation.sh   （在 UNO 仓库根目录、激活 .venv-uno 后）
set -euo pipefail
cd "$(dirname "$0")/.."

# RTX 4000 系列必须禁 P2P/IB（train.py 里也有兜底 setdefault）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# train.py 顶层 import wandb，包必须已安装（pip install wandb）；disabled 表示不上报
export WANDB_MODE="${WANDB_MODE:-disabled}"

# 启动前自检：wandb 可导入 + 转换后的标签存在 + eval 用的 dreambooth submodule 已初始化
python - <<'EOF'
import importlib.util, os, sys
if importlib.util.find_spec("wandb") is None:
    sys.exit("❌ wandb 未安装（train.py 顶层 import wandb，WANDB_MODE=disabled 挡不住 ImportError）\n   先执行: pip install wandb")
labels = "datasets/UNO-1M/uno_1m_total_labels_convert.json"
if not os.path.exists(labels):
    sys.exit(f"❌ {labels} 不存在\n   先执行: python scripts/convert_uno_labels.py")
# eval_data_json 默认指向 datasets/dreambench_toy.json，图片在 datasets/dreambooth/ 这个 git submodule 里；
# 没 init 的话 checkpointing_steps 之前训练正常，一到 eval_dataloader 就 FileNotFoundError（README.md 99-102 行提过）
dreambooth = "datasets/dreambooth/dataset"
if not os.path.isdir(dreambooth) or not os.listdir(dreambooth):
    sys.exit(f"❌ {dreambooth} 为空（dreambench submodule 未初始化，训练到第一次 checkpoint 才会炸）\n   先执行: git submodule update --init datasets/dreambooth")
EOF

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
    --resume_from_checkpoint latest \
    "$@"
