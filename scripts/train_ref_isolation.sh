#!/usr/bin/env bash
# 8×4090 上训练「隔离注意力」LoRA（推理时可开 --kv_cache 复用 ref K/V）
# 用法: bash scripts/train_ref_isolation.sh   （在 UNO 仓库根目录、激活 .venv-uno 后）
set -euo pipefail
cd "$(dirname "$0")/.."

# RTX 4000 系列必须禁 P2P/IB（train.py 里也有兜底 setdefault），故默认为 1。
# 但 H800/A100 这类有 NVLink+IB 的机器禁掉等于自废武功（实测 NV18 全互联约
# 478 GB/s，退回 PCIe 会拖慢 all-gather），所以允许外部 export 覆盖：
#   NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 bash scripts/train_ref_isolation.sh
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# train.py 顶层 import wandb，包必须已安装（pip install wandb）；disabled 表示不上报
export WANDB_MODE="${WANDB_MODE:-disabled}"
# 默认续训最新 checkpoint；不想续训则 RESUME_FROM_CHECKPOINT= bash scripts/... （置空）
# 注意: 不能靠 "$@" 传 --resume_from_checkpoint None 覆盖——HfArgumentParser 拿到的是
# 字符串 "None" 不是 Python None，会走 resume_from_checkpoint() 里的 else 分支去
# load_file("None") 直接崩溃；必须整个不传这个 flag 才会用 train.py 里的默认值 None。
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT-latest}"

# 启动前自检：wandb 可导入 + 转换后的标签存在 + eval 用的 dreambooth submodule 已初始化
# + 如果要续训，最新 checkpoint 的 dit_lora.safetensors 必须能正常加载
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

# 保存(dit_lora.safetensors/optimizer.bin 落盘)和 eval 之间原来没有同步屏障，
# 如果上次是 eval 先炸把主进程写到一半的进程也 SIGTERM 掉，checkpoint 文件可能是残档；
# --resume_from_checkpoint latest 会不加分辨地加载它，得在启动前验证，别把训练带进错误状态。
if os.environ.get("RESUME_FROM_CHECKPOINT") == "latest":
    project_dir = "log/ref_isolation"
    if os.path.isdir(project_dir):
        ckpts = [d for d in os.listdir(project_dir) if d.startswith("checkpoint")]
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
        if ckpts:
            latest_dir = os.path.join(project_dir, ckpts[-1])
            lora_path = os.path.join(latest_dir, "dit_lora.safetensors")
            try:
                from safetensors.torch import load_file
                state = load_file(lora_path)
                if len(state) == 0:
                    raise ValueError("state_dict 是空的")
            except Exception as e:
                sys.exit(
                    f"❌ {lora_path} 加载失败（{e}）\n"
                    f"   大概率是上次训练被杀时写到一半的残档，不能用来续训。\n"
                    f"   先执行: rm -rf {latest_dir}  （会从上一个更早的 checkpoint 续训，都没有就从头训）"
                )
EOF

resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

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
    ${resume_args[@]+"${resume_args[@]}"} \
    "$@"
