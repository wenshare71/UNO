#!/usr/bin/env bash
# UNO 训练/推理环境一键安装脚本（在远程机器上、UNO 仓库根目录执行）
# 用法: bash scripts/setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-uno}"
PYTHON_VERSION="3.11"   # pyproject 要求 >=3.10, <=3.12

# ---- 镜像/代理配置（均可用环境变量覆盖；普通 pip 包走机器默认源+no_proxy 内网直达）----
# 2026-07 矩阵测速：机器无直连出口；国内镜像走内网代理最快（阿里云 5.97 MB/s），
# 官方 download.pytorch.org 两条代理均不通；HF 只有海外代理能通。
TORCH_FINDLINKS="${TORCH_FINDLINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"
INTERNAL_PROXY="${INTERNAL_PROXY:-http://10.68.24.160:11080}"

cd "$(dirname "$0")/.."
echo "[1/5] 仓库目录: $(pwd)"

# ---- 找一个 >=3.10 的 Python（UNO 要求 >=3.10,<=3.12；3.8 会因旧 setuptools 报 license 配置错）----
find_python() {
    for cand in python3.11 python3.12 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            local p
            p="$(command -v "$cand")"
            # 不能选目标 venv 自己的解释器（重建时会把它删掉）
            case "$p" in *"/.venv-${ENV_NAME}/"*) continue ;; esac
            if "$p" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
                echo "$p"
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
elif [ -x ".venv-${ENV_NAME}/bin/python" ] && \
     ".venv-${ENV_NAME}/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    # 已有可用 venv：直接复用（幂等，重跑不会毁掉已装好的 torch）
    echo "[2/5] 复用已有虚拟环境 .venv-${ENV_NAME}"
    source ".venv-${ENV_NAME}/bin/activate"
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
# 版本号带 +cu124 强制 pip 从 find-links 里选 cu124 轮子，而不是默认源里的普通版。
# 优先用 fetch_torch_wheels.sh 并行下好的本地轮子；否则经内网代理从阿里云单连接下载。
echo "[3/5] 安装 PyTorch 2.4.0 + cu124"
pip install --upgrade pip
if ls wheels/torch-2.4.0+cu124-*.whl >/dev/null 2>&1; then
    echo "   使用本地 wheels/ 下的轮子"
    pip install wheels/torch-2.4.0+cu124-*.whl wheels/torchvision-0.19.0+cu124-*.whl
else
    echo "   本地无轮子，经内网代理从 ${TORCH_FINDLINKS} 下载（建议先跑 scripts/fetch_torch_wheels.sh 并行下载更快）"
    env http_proxy="${INTERNAL_PROXY}" https_proxy="${INTERNAL_PROXY}" \
        pip install "torch==2.4.0+cu124" "torchvision==0.19.0+cu124" -f "${TORCH_FINDLINKS}"
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
  # HF 只有海外代理能通(实测 0.33 MB/s 单连接)，必须开 hf_transfer 多连接并行拉:
  export http_proxy=http://oversea-squid1.jp.txyun:11080
  export https_proxy=http://oversea-squid1.jp.txyun:11080
  export HF_HUB_ENABLE_HF_TRANSFER=1
  # 不要设 HF_ENDPOINT（hf-mirror 实测更慢且同样要走海外代理）

  # 先下标签 + 5 个分片试跑
  huggingface-cli download bytedance-research/UNO-1M --repo-type dataset \
    --include "uno_1m_total_labels.json" --local-dir ./datasets/UNO-1M
  huggingface-cli download bytedance-research/UNO-1M --repo-type dataset \
    --include "images/split1.tar.gz" "images/split2.tar.gz" "images/split3.tar.gz" \
              "images/split4.tar.gz" "images/split5.tar.gz" \
    --local-dir ./datasets/UNO-1M
EOF
