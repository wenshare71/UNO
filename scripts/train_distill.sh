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
# 2026-08-06:`--ref_isolation` 从写死的 True 提成环境变量 `REF_ISOLATION`。
#   动机是 M6 隔离消融要一条全注意力的 baseline 腿(distill/M6_ABLATION_SPEC.md),
#   两腿必须共用同一个脚本、同一份数据、同样的超参,**只差这一个 flag** ——
#   为 baseline 另写一个脚本就等于给自己开一个"抄漏一行"的口子。
#   默认仍是 True:既有 log/ref_distill 与所有已发布读数都是 True 训出来的。
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
# M6 消融的 baseline 腿要用 False(见 distill/M6_ABLATION_SPEC.md)。默认 True 保持
# 既有 log/ref_distill 的语义不变——过去所有读数都是 True 训出来的。
export REF_ISOLATION="${REF_ISOLATION:-True}"

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
#
# "latest" 是 train.py:129-143 的特殊值:从 PROJECT_DIR 里挑编号最大的 checkpoint-N,
# 并把 global_step 直接置为 N —— 断点续训要的就是这个语义(显式文件路径会让
# global_step 归 0,续 4000 步变成总共 6000 步)。它不是文件,得先放行再解析成真实路径,
# 否则下面的 isfile 检查会把合法的续训调用挡掉。[2026-08-05 臂B 2593 步被杀后新增]
resume_path = os.environ["RESUME_FROM_CHECKPOINT"]
if resume_path == "latest":
    project_dir_for_resume = os.environ["PROJECT_DIR"]
    ckpts = sorted(
        (d for d in os.listdir(project_dir_for_resume) if d.startswith("checkpoint-")),
        key=lambda x: int(x.split("-")[1]),
    ) if os.path.isdir(project_dir_for_resume) else []
    if not ckpts:
        sys.exit(f"❌ RESUME_FROM_CHECKPOINT=latest 但 {project_dir_for_resume} 下没有 checkpoint-*\n"
                 f"   要么这是首次训练(改用显式 safetensors 路径),要么 PROJECT_DIR 写错了")
    resume_path = os.path.join(project_dir_for_resume, ckpts[-1], "dit_lora.safetensors")
    print(f"[preflight] latest → {ckpts[-1]}(global_step 将从 {ckpts[-1].split('-')[1]} 起算)", flush=True)
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

# 7) 把这一腿是谁打进日志。M6 两腿共用本脚本,只差 REF_ISOLATION;
#    事后翻 log 时唯一能确认"这个 checkpoint 是哪条腿"的地方就是这一行。
print(f"[preflight] ref_isolation={os.environ['REF_ISOLATION']} / "
      f"data={train_json} / out={project_dir}", flush=True)

print("[preflight] === 所有自检通过,开始训练 ===", flush=True)
EOF

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# --- 训练:相对 train_ref_isolation.sh,变的只有数据(与 M6 baseline 腿的 REF_ISOLATION)---
# DeepSpeed ZeRO-3 切分(bf16 FLUX ~25.8GB,H800 143GB 其实放得下完整模型,
# 但改回 non-ZeRO-3 要动 train.py 的 deepspeed.zero.Init 与 plugin 注册,不在本实验范围;
# 保持 ZeRO-3 配置不变,只换数据——这是「唯一变量是数据」的字面要求)。
accelerate launch --num_processes 8 --mixed_precision bf16 train.py \
    --ref_isolation "${REF_ISOLATION}" \
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
