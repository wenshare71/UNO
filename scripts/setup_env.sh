#!/usr/bin/env bash
# UNO 训练/推理环境一键安装脚本（在远程机器上、UNO 仓库根目录执行）
# 用法: bash scripts/setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-uno}"
PYTHON_VERSION="3.11"   # pyproject 要求 >=3.10, <=3.12

cd "$(dirname "$0")/.."
echo "[1/5] 仓库目录: $(pwd)"

# ---- 创建虚拟环境：优先 conda，否则退回 venv ----
if command -v conda >/dev/null 2>&1; then
    echo "[2/5] 使用 conda 创建环境 ${ENV_NAME} (python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
else
    echo "[2/5] 未找到 conda，使用 python venv 创建 .venv-${ENV_NAME}"
    PY_BIN="$(command -v python${PYTHON_VERSION} || command -v python3.10 || command -v python3)"
    "${PY_BIN}" -m venv ".venv-${ENV_NAME}"
    source ".venv-${ENV_NAME}/bin/activate"
fi
python -V

# ---- PyTorch（cu124，对应 4090）----
echo "[3/5] 安装 PyTorch 2.4.0 + cu124"
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# ---- UNO 本体 + 训练依赖（accelerate/deepspeed）----
echo "[4/5] pip install -e '.[train]'"
pip install -e ".[train]"

# ---- 下载提速工具 ----
pip install hf_transfer "huggingface_hub[cli]"

# ---- 自检 ----
echo "[5/5] 环境自检"
python - <<'EOF'
import torch, transformers, diffusers, einops
print(f"torch        : {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu          : {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
print(f"transformers : {transformers.__version__}")
print(f"diffusers    : {diffusers.__version__}")
import uno
print("uno package  : OK")
EOF

cat <<'EOF'

✅ 环境安装完成。后续常用命令：
  # 下载数据时提速（写进 ~/.bashrc 更方便）
  export HF_HUB_ENABLE_HF_TRANSFER=1
  # 国内机器可加镜像
  export HF_ENDPOINT=https://hf-mirror.com

  # 先下标签 + 5 个分片试跑
  huggingface-cli download bytedance-research/UNO-1M --repo-type dataset \
    --include "uno_1m_total_labels.json" --local-dir ./datasets/UNO-1M
  huggingface-cli download bytedance-research/UNO-1M --repo-type dataset \
    --include "images/split1.tar.gz" "images/split2.tar.gz" "images/split3.tar.gz" \
              "images/split4.tar.gz" "images/split5.tar.gz" \
    --local-dir ./datasets/UNO-1M
EOF
