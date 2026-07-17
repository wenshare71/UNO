#!/usr/bin/env bash
# UNO 训练/推理环境一键安装脚本（在远程机器上、UNO 仓库根目录执行）
# 用法: bash scripts/setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-uno}"
PYTHON_VERSION="3.11"   # pyproject 要求 >=3.10, <=3.12

# ---- 镜像配置（均可用环境变量覆盖；普通 pip 包走机器默认源，不在此覆盖）----
# PyTorch 轮子：南京大学镜像（已验证有 torch-2.4.0+cu124 全套轮子）
# 备选: https://mirror.sjtu.edu.cn/pytorch-wheels/cu124   官方: https://download.pytorch.org/whl/cu124
TORCH_INDEX="${TORCH_INDEX:-https://mirror.nju.edu.cn/pytorch/whl/cu124}"

cd "$(dirname "$0")/.."
echo "[1/5] 仓库目录: $(pwd)"

# ---- 找一个 >=3.10 的 Python（UNO 要求 >=3.10,<=3.12；3.8 会因旧 setuptools 报 license 配置错）----
find_python() {
    for cand in python3.11 python3.12 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
                command -v "$cand"
                return 0
            fi
        fi
    done
    return 1
}

# ---- 创建虚拟环境：优先 conda；无 conda 则要求系统有 3.10~3.12 的 python ----
if command -v conda >/dev/null 2>&1; then
    echo "[2/5] 使用 conda 创建环境 ${ENV_NAME} (python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
elif PY_BIN="$(find_python)"; then
    echo "[2/5] 未找到 conda，使用 ${PY_BIN} 创建 .venv-${ENV_NAME}"
    rm -rf ".venv-${ENV_NAME}"
    "${PY_BIN}" -m venv ".venv-${ENV_NAME}"
    source ".venv-${ENV_NAME}/bin/activate"
else
    cat >&2 <<'EOF'
❌ 没有找到 conda，也没有找到 Python 3.10~3.12（系统 python3 版本太旧，UNO 无法运行）。

请先装一个 Miniconda（不需要 root，装到用户目录；用清华 TUNA 镜像下载）：
  wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"

（可选）conda 包源也换成 TUNA，建环境更快：
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  conda config --set show_channel_urls yes

然后重新执行本脚本即可（脚本会自动检测到 conda）。
EOF
    exit 1
fi
python -V
python -c 'import sys; assert sys.version_info >= (3,10), "Python 版本仍低于 3.10，环境激活异常"'

# ---- PyTorch（cu124，对应 4090）----
echo "[3/5] 安装 PyTorch 2.4.0 + cu124（源: ${TORCH_INDEX}）"
pip install --upgrade pip
if ! pip install torch==2.4.0 torchvision==0.19.0 --index-url "${TORCH_INDEX}"; then
    echo "⚠️ 镜像 ${TORCH_INDEX} 安装失败，回退官方源 download.pytorch.org"
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
fi

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
