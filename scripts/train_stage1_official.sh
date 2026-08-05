#!/usr/bin/env bash
# 按**官方配方**重训 stage-1 底座(替换 4090 上那个 `log/ref_isolation/checkpoint-20000`)。
#
# 为什么要重训:现有底座对官方 stage-1 有三处偏离(已核实,DISTILL_PLAN §11.10):
#   ① 没跑 score 过滤   —— 用的是 scripts/convert_uno_labels.py,连阈值参数都没有;
#   ② 数据只有 split1-5 —— 够官方标准的样本 16,966 : 404,259 ≈ 24×;
#   ③ 步数 20000        —— train.py:208 的默认值是 100000。
# 本脚本把三条全部对齐官方。
#
# ━━ 与 train_ref_isolation.sh 的逐条差异 ━━
#   1. 数据换成 distill/build_stage1_official.py 的产物(score_final ≥ 4 全库);
#   2. gradient_accumulation_steps **1**(官方默认;4090 那次用的是 2);
#   3. max_train_steps **100000**(官方默认;4090 那次是 20000);
#   4. ref_isolation **False**(官方默认)—— 见下面那段,这是唯一需要你拍板的开关;
#   5. NCCL P2P/IB 默认开(H800 有 NVLink+IB;4090 必须禁,正好相反);
#   6. project_dir=log/stage1_official,不碰任何既有目录。
# 其余(lora_rank 512 / lr 8e-5 / batch 1 / res 512 / bf16 / ckpt 每 1000 步)
# 本来就与官方默认一致,**一个都不要手填** —— 手填就有填错的机会。
#
# ━━ REF_ISOLATION:这个开关决定新底座是什么 ━━
#   False(默认,= 官方):新底座是**官方 stage-1 的忠实复刻**。它与官方发布的 LoRA
#     构成"同配方、不同训练者"的对照 ⇒ 边 ③ 变成干净的「我们复刻得像不像」,
#     而且与臂 B(官方 LoRA 作 init + 4000 步隔离)完全平行。
#   True(= 4090 那次的做法):新底座仍是隔离训练的,与现有 ckpt-20000 同型,
#     可以直接顶替它、保持旧链条的语义不变。
#   ⚠️ 两者不是一回事,选错了整条链的读数含义就变了。默认走官方。
#
# ━━ 用法(H800,仓库根目录,已激活 .venv-uno)━━
#   # 0) 先补磁盘 + 出数据(见各自脚本的 --help)
#   python scripts/fetch_uno1m.py --rm_tar
#   python distill/build_stage1_official.py --strict
#
#   # 1) 100 步标定(不污染正式目录)
#   MAX_TRAIN_STEPS=100 CHECKPOINTING_STEPS=50 \
#   PROJECT_DIR=log/stage1_official_calib bash scripts/train_stage1_official.sh
#
#   # 2) 正式训练。**别用 nohup** —— torch.distributed.elastic 会装自己的 SIGHUP
#   #    handler 覆盖掉 nohup 的 SIG_IGN(臂 B 被这个杀过两次,DISTILL_PLAN §11.12(a))
#   setsid bash scripts/train_stage1_official.sh > logs/stage1_official.log 2>&1 < /dev/null &
#
#   # 3) 断了续训(latest 会让 global_step 从 checkpoint 编号起算,别传显式路径)
#   RESUME_FROM_CHECKPOINT=latest bash scripts/train_stage1_official.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"

export TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-datasets/UNO-1M/stage1_official_score4.json}"
export PROJECT_DIR="${PROJECT_DIR:-log/stage1_official}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export REF_ISOLATION="${REF_ISOLATION:-False}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
# 首次训练必须为空(传 "latest" 时 project_dir 下没 checkpoint 会被自检拦下)
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

python - <<'EOF'
import importlib.util, json, os, sys

# 1) wandb:train.py 顶层 import,WANDB_MODE=disabled 挡不住 ImportError
if importlib.util.find_spec("wandb") is None:
    sys.exit("❌ wandb 未安装,先: pip install wandb")

# 2) 训练 json:存在 + 非空 + schema 对得上 FluxPairedDatasetV2
train_json = os.environ["TRAIN_DATA_JSON"]
if not os.path.exists(train_json):
    sys.exit(f"❌ {train_json} 不存在\n"
             f"   先执行: python distill/build_stage1_official.py --strict")
with open(train_json, "rt") as f:
    records = json.load(f)
if not records:
    sys.exit(f"❌ {train_json} 是空的")
need = {"prompt", "image_tgt_path", "image_paths"}
if not need <= set(records[0]):
    sys.exit(f"❌ {train_json} schema 不对,缺 {need - set(records[0])}")

# 3) 抽查 200 条图片是否真在磁盘上。全查太慢(40 万条),但完全不查就等于把
#    "训到第 3 万步 FileNotFoundError" 留给几十小时之后 —— 4090 那次就是这么炸的。
root = os.path.dirname(os.path.abspath(train_json))
import random
miss = []
for r in random.Random(0).sample(records, min(200, len(records))):
    for p in list(r["image_paths"]) + [r["image_tgt_path"]]:
        if not os.path.exists(os.path.join(root, p)):
            miss.append(p)
if miss:
    sys.exit(f"❌ 抽查 200 条里有 {len(miss)} 个路径不在磁盘,例:{miss[:3]}\n"
             f"   先补齐: python scripts/fetch_uno1m.py --rm_tar")

# 4) 这份数据到底算不算"官方口径"——不让它含糊过去
n = len(records)
OFFICIAL_POOL = 404_259
tag = "⚠️ 部分数据" if "_partial" in os.path.basename(train_json) else "官方口径"
print(f"[preflight] 训练集: {train_json}", flush=True)
print(f"[preflight] {n} 条 = 官方满分池的 {n / OFFICIAL_POOL:.1%}  [{tag}]", flush=True)
if "_partial" in os.path.basename(train_json):
    print("            这个底座**不能叫「按官方流程复刻」**,报告里要写实际条数",
          file=sys.stderr, flush=True)

# 5) eval 用的 dreambooth submodule(第一次 checkpoint 才会用到,炸得很晚)
if not os.path.isdir("datasets/dreambooth/dataset") or not os.listdir("datasets/dreambooth/dataset"):
    sys.exit("❌ datasets/dreambooth/dataset 为空\n"
             "   先执行: git submodule update --init datasets/dreambooth")

# 6) 续训 checkpoint(传了才查)
resume = os.environ.get("RESUME_FROM_CHECKPOINT", "")
project_dir = os.environ["PROJECT_DIR"]
if resume:
    path = resume
    if resume == "latest":
        ckpts = sorted((d for d in os.listdir(project_dir) if d.startswith("checkpoint-")),
                       key=lambda x: int(x.split("-")[1])) if os.path.isdir(project_dir) else []
        if not ckpts:
            sys.exit(f"❌ RESUME_FROM_CHECKPOINT=latest 但 {project_dir} 下没有 checkpoint-*\n"
                     f"   首次训练请置空这个变量,不要传 latest")
        path = os.path.join(project_dir, ckpts[-1], "dit_lora.safetensors")
        print(f"[preflight] latest → {ckpts[-1]}(global_step 从 {ckpts[-1].split('-')[1]} 起算)",
              flush=True)
    from safetensors.torch import load_file
    try:
        state = load_file(path)
        assert state
    except Exception as e:
        sys.exit(f"❌ {path} 加载失败({e})——大概率是被杀时写到一半的残档")
    print(f"[preflight] resume: {path}({len(state)} 个张量)", flush=True)
elif os.path.isdir(project_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(project_dir)):
    sys.exit(f"❌ {project_dir} 下已有 checkpoint 但没传 RESUME_FROM_CHECKPOINT。\n"
             f"   续训: RESUME_FROM_CHECKPOINT=latest bash scripts/train_stage1_official.sh\n"
             f"   重来: 换个 PROJECT_DIR(**不要**覆盖已有的训练)")

# 7) project_dir 可写(log/ 下 rsync 来的文件可能是 root 属主)
os.makedirs(project_dir, exist_ok=True)
t = os.path.join(project_dir, ".preflight_write_test")
try:
    open(t, "w").write("ok"); os.remove(t)
except PermissionError as e:
    sys.exit(f"❌ {project_dir} 不可写({e}),先 sudo chown -R $(whoami) {project_dir}")

# 8) 卡数 + 时间预算。100000 步不是个能顺手一跑的数,得在启动前就摆在眼前。
import torch
ndev = torch.cuda.device_count()
if ndev < int(os.environ["NUM_PROCESSES"]):
    print(f"⚠️ 可见 {ndev} 卡 < NUM_PROCESSES={os.environ['NUM_PROCESSES']}", file=sys.stderr)
steps = int(os.environ["MAX_TRAIN_STEPS"])
# 实测:H800 8 卡 grad_accum=2 全注意力 4.86 s/it、隔离 5.3-5.9 s/it(DISTILL_PLAN §11.12)。
# grad_accum=1 少一次前后向,按 0.55× 粗估,只用来排数量级,不是承诺。
iso = os.environ["REF_ISOLATION"].lower() in ("true", "1", "yes")
sec = steps * (5.6 if iso else 4.86) * 0.55 * (2 if int(os.environ["GRAD_ACCUM"]) > 1 else 1)
print(f"[preflight] GPU {ndev} 卡 / ref_isolation={os.environ['REF_ISOLATION']} / "
      f"grad_accum={os.environ['GRAD_ACCUM']}", flush=True)
print(f"[preflight] {steps} 步 ≈ **{sec / 3600:.0f} 小时**({sec / 86400:.1f} 天)"
      f" —— 粗估,只看数量级", flush=True)
print("[preflight] === 自检通过,开始训练 ===", flush=True)
EOF

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# lora_rank / learning_rate / batch_size / resolution / lr_scheduler / warmup 全部
# 走 train.py 的默认值 —— 那就是官方配方,显式重写一遍只会引入抄错的机会。
accelerate launch --num_processes "${NUM_PROCESSES}" --mixed_precision bf16 train.py \
    --ref_isolation "${REF_ISOLATION}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --train_data_json "${TRAIN_DATA_JSON}" \
    --project_dir "${PROJECT_DIR}" \
    ${resume_args[@]+"${resume_args[@]}"} \
    "$@"
