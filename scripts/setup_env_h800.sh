#!/usr/bin/env bash
# UNO 环境重建 —— 8×H800 机器专用（aiplatform-wlf3-ge90-19）
#
# 与 scripts/setup_env.sh 的区别，以及为什么必须另写一份：
#   1. 旧脚本假设有 conda 或能 `python3 -m venv`；这台机器的 python3.10 缺
#      ensurepip，venv 建不出 pip，而 apt 又够不到 archive.ubuntu.com（实测
#      Network is unreachable）。改用 `venv --without-pip` + pip wheel 自举。
#   2. 旧脚本从阿里云/官方拉 torch；这台机器所有出网都经日本代理，实测
#      0.05~0.66 MB/s，装完 torch 栈要近 2 小时。内网源 pypi.corp.kuaishou.com
#      命中 no_proxy 直连，实测 241.91 MB/s，快 672 倍——必须走内网源。
#   3. 旧脚本装 cu124 是为了 4090；H800 是 sm90，PyPI 上默认的 torch 2.4.0
#      （cu121 构建）官方 arch 列表里含 sm_90，直接可用，不必绕 cu124 索引。
#   4. `pip install -e ".[train]"` 会因 pyproject 的开区间 transformers>=4.43.3
#      / diffusers>=0.30.1 拉到 5.x / 0.39（冒烟测试踩过），故先钉版本再装本体：
#      已满足的 >= 约束 pip 不会再升级。
#
# 用法（UNO 仓库根目录）：
#   bash scripts/setup_env_h800.sh                 # 只建环境
#   COPY_TO_LOCAL=1 bash scripts/setup_env_h800.sh # 顺带把 76G 权重搬到本地 NVMe
#
# 幂等：重复执行会复用已有 venv，不会重装已装好的包。

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${VENV:-$REPO/.venv-uno}"
INTERNAL_HOST="pypi.corp.kuaishou.com"
INTERNAL_INDEX="${INTERNAL_INDEX:-https://${INTERNAL_HOST}/kuaishou/prod/+simple/}"
PIP_VERSION="${PIP_VERSION:-24.3.1}"
# 内网镜像已改用内容寻址（+f/<hash>/...），写死 prod/pip/<ver>/ 这种路径会 404，
# 故留空默认、运行时从 simple 索引现解析；仍允许用 PIP_WHL_URL 手动覆盖。
PIP_WHL_URL="${PIP_WHL_URL:-}"
CEPH_HF_CACHE="${CEPH_HF_CACHE:-/kaimm-distill/wuwenxuan/hf_cache}"
LOCAL_ROOT="${LOCAL_ROOT:-/code/uno}"

step() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$1" "$2"; }

# ---------------------------------------------------------------- 0. 前置检查
step 0/7 "前置检查"
command -v python3 >/dev/null || { echo "❌ 没有 python3"; exit 1; }
python3 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' \
    || { echo "❌ python3 版本不在 3.10~3.12：$(python3 -V)"; exit 1; }
echo "  python3      : $(python3 -V 2>&1)"
# 内网源必须直连（no_proxy 含 corp.kuaishou.com）。若这里慢，说明 no_proxy 没生效。
python3 - <<PY
import time, urllib.request
t0 = time.time()
urllib.request.urlopen("${INTERNAL_INDEX}", timeout=15).read(4096)
dt = time.time() - t0
print(f"  内网 PyPI    : 可达，{dt:.2f}s" + ("" if dt < 2 else "  ⚠️ 偏慢，检查 no_proxy 是否含 corp.kuaishou.com"))
PY

# ------------------------------------------------- 1. 修 rsync 带过来的 root 权限
step 1/7 "修复 checkpoint 权限"
# 权重是从源机器 /root 下用 rsync -a 拷的，dit_lora.safetensors 保留了 root:root 0600，
# 当前用户（uid $(id -u)）读不了；resume 时会在 load_file 处抛 PermissionError。
BAD=$(find log -type f ! -readable 2>/dev/null | wc -l | tr -d ' ')
if [ "$BAD" -gt 0 ]; then
    echo "  发现 $BAD 个当前用户读不了的文件，用 sudo 修正属主"
    sudo chown -R "$(id -u):$(id -g)" log
    echo "  修正后仍不可读: $(find log -type f ! -readable 2>/dev/null | wc -l | tr -d ' ')"
else
    echo "  log/ 全部可读，跳过"
fi

# ------------------------------------------------------------ 2. venv + pip 自举
step 2/7 "创建虚拟环境并自举 pip"
if [ -x "$VENV/bin/pip" ]; then
    echo "  复用已有环境 $VENV（$("$VENV/bin/python" -V 2>&1)）"
else
    rm -rf "$VENV"
    # --without-pip 跳过 ensurepip 这一步；这正是这台机器 venv 失败的唯一原因
    python3 -m venv --without-pip "$VENV"
    echo "  venv 已建（无 pip），从内网源自举 pip"
    # pip 的 wheel 本身是可执行 zipapp：把 whl 当目录传给 python 即可运行里面的 pip。
    # 地址优先用 PIP_WHL_URL；否则从 simple 索引现解析（镜像用内容寻址，不能写死路径）。
    whl_url="$PIP_WHL_URL"
    if [ -z "$whl_url" ]; then
        href=$(wget -qO- "${INTERNAL_INDEX%/}/pip/" \
            | grep -oE "href=\"[^\"]*pip-${PIP_VERSION//./\\.}-py3-none-any\.whl" \
            | head -1 | sed -E 's/^href="//')
        [ -n "$href" ] || { echo "❌ 内网 simple 索引里找不到 pip-${PIP_VERSION}（换 PIP_VERSION 或设 PIP_WHL_URL）"; exit 1; }
        # href 形如 ../../../../root/pypi/+f/xxx/pip-...whl，逐级 ../ 恰好回到镜像根
        whl_url="https://${INTERNAL_HOST}/$(echo "$href" | sed -E 's#^(\.\./)+##')"
    fi
    echo "  pip whl      : $whl_url"
    # 必须用合规 wheel 文件名落盘：pip 24.3+ 会按 PEP 427 校验，_pip.whl 会被拒
    whl_path="/tmp/$(basename "${whl_url%%#*}")"
    wget -q -O "$whl_path" "$whl_url" \
        || { echo "❌ 下载 pip whl 失败：$whl_url"; exit 1; }
    [ -s "$whl_path" ] || { echo "❌ pip whl 下到 0 字节（源地址可能已失效）：$whl_url"; exit 1; }
    "$VENV/bin/python" "$whl_path/pip" install --no-index "$whl_path"
    rm -f "$whl_path"
    echo "  $("$VENV/bin/pip" --version)"
fi

# venv 级 pip 配置：只用内网源，避免任何请求走日本代理
cat > "$VENV/pip.conf" <<EOF
[global]
index-url = ${INTERNAL_INDEX}
trusted-host = ${INTERNAL_HOST}
timeout = 60
EOF
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade setuptools wheel

# ------------------------------------------------------------------ 3. PyTorch
step 3/7 "安装 PyTorch 2.4.0（内网源，cu121 构建，含 sm_90）"
if python -c 'import torch, sys; sys.exit(0 if torch.__version__.startswith("2.4.0") else 1)' 2>/dev/null; then
    echo "  已装 torch $(python -c 'import torch; print(torch.__version__)')，跳过"
else
    pip install torch==2.4.0 torchvision==0.19.0
fi

# --------------------------------------------------- 4. 先钉死版本，再装 UNO 本体
step 4/7 "钉死 transformers / diffusers / accelerate"
# 顺序很重要：先装精确版本，后面 `-e .[train]` 的 >= 约束已被满足，pip 不会再升级。
# 反过来先装本体的话会拿到 transformers 5.x + diffusers 0.39，UNO 跑不起来。
pip install transformers==4.43.3 diffusers==0.30.1 accelerate==1.1.1

step 5/7 "安装 deepspeed 0.14.4"
# deepspeed 在 PyPI 上只有 sdist，setup.py 会 import torch；pip 默认的构建隔离
# 会在干净环境里重新下载一份 torch，故优先关掉隔离（venv 里已有 torch/setuptools）。
if python -c 'import deepspeed' 2>/dev/null; then
    echo "  已装 deepspeed $(python -c 'import deepspeed; print(deepspeed.__version__)')，跳过"
else
    pip install --no-build-isolation deepspeed==0.14.4 \
        || { echo "  --no-build-isolation 失败，回退到标准安装"; pip install deepspeed==0.14.4; }
fi

step 6/7 "安装 UNO 本体"
pip install -e ".[train]"

# -------------------------------------------- 7.（可选）权重搬到本地 NVMe
if [ "${COPY_TO_LOCAL:-0}" = "1" ]; then
    step 7/7 "把 HF 权重搬到本地 NVMe（ceph 读 136 MB/s vs NVMe 3.4 GB/s）"
    mkdir -p "$LOCAL_ROOT"
    if [ -d "$LOCAL_ROOT/hf_cache" ]; then
        echo "  $LOCAL_ROOT/hf_cache 已存在，跳过（要重拷请先手动删除）"
    else
        echo "  拷贝 76 GB，受限于 ceph 读速，预计约 10 分钟…"
        cp -a "$CEPH_HF_CACHE" "$LOCAL_ROOT/hf_cache"
        echo "  完成: $(du -sh "$LOCAL_ROOT/hf_cache" | cut -f1)"
    fi
    HF_HOME_FINAL="$LOCAL_ROOT/hf_cache"
else
    step 7/7 "跳过权重本地化（要做请加 COPY_TO_LOCAL=1）"
    HF_HOME_FINAL="$CEPH_HF_CACHE"
fi

# ------------------------------------------------------------------ 自检
step ✓ "环境自检"
HF_HOME="$HF_HOME_FINAL" python - <<'PY'
import os, torch, transformers, diffusers, accelerate
print(f"  torch        : {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu          : {n} x {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.0f} GiB")
    assert (p.major, p.minor) >= (9, 0), "算力低于 sm_90，与本脚本假设不符"
print(f"  transformers : {transformers.__version__}" + ("  ✅" if transformers.__version__ == "4.43.3" else "  ⚠️ 期望 4.43.3"))
print(f"  diffusers    : {diffusers.__version__}" + ("  ✅" if diffusers.__version__ == "0.30.1" else "  ⚠️ 期望 0.30.1"))
print(f"  accelerate   : {accelerate.__version__}")
try:
    import deepspeed; print(f"  deepspeed    : {deepspeed.__version__}")
except Exception as e:
    print(f"  deepspeed    : ❌ {type(e).__name__}: {e}（只影响训练，不影响 M1 数据生成）")
import uno; print("  uno package  : OK")
hub = os.path.join(os.environ["HF_HOME"], "hub")
repos = sorted(d for d in os.listdir(hub) if d.startswith("models--")) if os.path.isdir(hub) else []
print(f"  HF_HOME      : {os.environ['HF_HOME']}  ({len(repos)} 个模型仓库)")
for r in repos: print(f"                 {r}")
PY

cat <<EOF

✅ 环境就绪。每次开工前 source 这段（也可以写进 ~/.bashrc）：

  cd $REPO
  source $VENV/bin/activate
  export HF_HOME=$HF_HOME_FINAL
  export HF_HUB_OFFLINE=1                      # 权重已全在本地，禁掉联网探测（否则每次加载都卡日本代理）
  export NCCL_P2P_DISABLE=0                    # 这台是 NV18 全互联，禁 P2P 等于自废武功
  export NCCL_IB_DISABLE=0                     # 12 张 mlx5 网卡
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

⚠️ scripts/train_ref_isolation.sh 第 8-9 行硬编码了 NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1
   （那是 4090 的补丁），在这台机器上跑训练前必须改掉。详见 docs/H800_REBUILD.md。
EOF
