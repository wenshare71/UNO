#!/usr/bin/env bash
# M3 蒸馏训练:在 ref_isolation checkpoint-20000 基础上,用混合数据续训 4000 步。
#
# 与 train_ref_isolation.sh 的关键差异(评审 C1/C2 + DISTILL_PLAN §5):
#   1. NCCL P2P/IB 默认开启(H800 有 NVLink+IB,4090 必须禁,正好相反);
#   2. --resume_from_checkpoint 指向 checkpoint-20000 的显式 safetensors 路径
#      (train.py:145 传具体路径时 global_step=0,不恢复 optimizer,符合预期);
#   3. --max_train_steps 4000(唯一变量是数据,其余超参一律不动);
#   4. --project_dir log/ref_distill(不覆盖 log/ref_isolation);
#   5. --train_data_json datasets/distill_multiref/train_mixed.json(M3 混合集);
#   6. 自检 heredoc 里硬编码的 project_dir / labels 路径同步改了(评审 C2)。
#
# 标定用法(100 步,不污染正式目录):
#   MAX_TRAIN_STEPS=100 PROJECT_DIR=log/ref_distill_calibration \
#   CHECKPOINTING_STEPS=50 bash scripts/train_distill.sh
#
# 用法(正式 4000 步,在 UNO 仓库根目录、激活 .venv-uno 后):
#   bash scripts/train_distill.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# --- NCCL: H800 有 NV18 全互联 + IB,禁掉等于自废武功(commit 7023e70 已改为可外部覆盖) ---
# 4090 必须禁;H800/A100 这类保持开启。默认 0(开)。
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# train.py 顶层 import wandb,包必须已安装(pip install wandb);disabled 表示不上报
export WANDB_MODE="${WANDB_MODE:-disabled}"

# --- 可由环境变量覆盖的关键参数(方便 100 步标定) ---
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-log/ref_isolation/checkpoint-20000/dit_lora.safetensors}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
export PROJECT_DIR="${PROJECT_DIR:-log/ref_distill}"
export TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-datasets/distill_multiref/train_mixed.json}"

# --- 启动前自检:wandb + 混合 json + dreambooth submodule + resume ckpt + project_dir 可写 ---
# 注意:heredoc 里的 project_dir / labels 路径要与上面 export 一致(评审 C2)
python - <<'EOF'
import importlib.util, os, sys

# 1) wandb 必须可 import(train.py 顶层 import wandb,WANDB_MODE=disabled 挡不住 ImportError)
if importlib.util.find_spec("wandb") is None:
    sys.exit("❌ wandb 未安装(train.py 顶层 import wandb,WANDB_MODE=disabled 挡不住 ImportError)\n"
             "   先执行: pip install wandb")

# 2) 混合训练 json 必须存在且非空
train_json = os.environ["TRAIN_DATA_JSON"]
if not os.path.exists(train_json):
    sys.exit(f"❌ {train_json} 不存在\n"
             f"   先执行: python distill/build_train_json.py")
import json
with open(train_json, "rt") as f:
    records = json.load(f)
if len(records) == 0:
    sys.exit(f"❌ {train_json} 是空文件,检查 build_train_json.py 输出")
print(f"[preflight] 训练 json: {train_json} ({len(records)} 条)", flush=True)

# 3) eval 用的 dreambooth submodule 必须初始化
# (eval_data_json 默认 datasets/dreambench_toy.json,图片在 datasets/dreambooth/dataset/ 这个 git submodule 里;
#  没 init 的话 checkpointing_steps 之前训练正常,一到 eval_dataloader 就 FileNotFoundError)
dreambooth = "datasets/dreambooth/dataset"
if not os.path.isdir(dreambooth) or not os.listdir(dreambooth):
    sys.exit(f"❌ {dreambooth} 为空(dreambench submodule 未初始化,训练到第一次 checkpoint 会炸)\n"
             f"   先执行: git submodule update --init datasets/dreambooth")
print(f"[preflight] dreambooth submodule: OK", flush=True)

# 4) resume checkpoint 必须存在且能正常加载(DISTILL_PLAN §5:M3 前要先 chown,这里只检查可读)
resume_path = os.environ["RESUME_FROM_CHECKPOINT"]
if not os.path.isfile(resume_path):
    sys.exit(f"❌ resume checkpoint 不存在: {resume_path}\n"
             f"   检查 checkpoint-20000 是否已从旧机器同步到 H800")
try:
    from safetensors.torch import load_file
    state = load_file(resume_path)
    if len(state) == 0:
        raise ValueError("state_dict 是空的")
except Exception as e:
    sys.exit(f"❌ {resume_path} 加载失败({e})\n"
             f"   大概率是文件残档或权限问题;检查文件完整性与属主")
print(f"[preflight] resume checkpoint: {resume_path} ({len(state)} 个张量)", flush=True)

# 5) project_dir 必须可写(DISTILL_PLAN §5:log/ 下 rsync 来的文件可能 root 属主,保存 checkpoint 会崩)
project_dir = os.environ["PROJECT_DIR"]
os.makedirs(project_dir, exist_ok=True)
test_file = os.path.join(project_dir, ".preflight_write_test")
try:
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
except PermissionError as e:
    sys.exit(f"❌ {project_dir} 不可写({e})\n"
            f"   DISTILL_PLAN §5 提到 log/ 下文件可能 root 属主,先执行:\n"
            f"   sudo chown -R $(whoami) {os.path.dirname(project_dir)}/ref_isolation {project_dir}")
print(f"[preflight] project_dir 可写: {project_dir}", flush=True)

# 6) 8 卡可见(加速训练需要 8 卡,少了要么 OOM 要么慢)
import torch
n = torch.cuda.device_count()
if n < 8:
    print(f"⚠️  GPU 可见数 = {n}(预期 8),训练会继续但吞吐受影响", file=sys.stderr)
else:
    print(f"[preflight] GPU: {n} 卡可见", flush=True)

print("[preflight] === 所有自检通过,开始训练 ===", flush=True)
EOF

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# --- 训练:唯一变量是数据,其余超参与 train_ref_isolation.sh 完全一致 ---
# DeepSpeed ZeRO-3 切分(bf16 FLUX ~25.8GB,H800 143GB 其实放得下完整模型,
# 但改回 non-ZeRO-3 要动 train.py 的 deepspeed.zero.Init 与 plugin 注册,不在本实验范围;
# 保持 ZeRO-3 配置不变,只换数据——这是「唯一变量是数据」的字面要求)。
accelerate launch --num_processes 8 --mixed_precision bf16 train.py \
    --ref_isolation True \
    --lora_rank 512 \
    --gradient_checkpoint True \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --resolution 512 \
    --learning_rate 8e-5 \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --train_data_json "${TRAIN_DATA_JSON}" \
    --project_dir "${PROJECT_DIR}" \
    ${resume_args[@]+"${resume_args[@]}"} \
    "$@"
