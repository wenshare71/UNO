#!/usr/bin/env bash
# 新机器环境探测：一次性收集重建 UNO 训练环境所需的全部信息。
#
# 用法（在 UNO 仓库根目录执行，不需要任何 Python 环境）：
#   bash scripts/probe_env.sh 2>&1 | tee /tmp/probe_report.txt
# 然后把 /tmp/probe_report.txt 的内容整个贴回来。
#
# 只读脚本：不装任何东西、不改任何配置、不下载大文件（网速测试最多各 12 秒）。
# 故意不用 set -e —— 缺命令、探测失败都要继续往下跑，报告的完整性比中途退出重要。

sec() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
# 跑命令；命令不存在就说明缺了什么，而不是留一段空白让人猜
run() {
    local cmd="$1"
    if have "${cmd%% *}"; then
        eval "$cmd" 2>&1 | head -n "${2:-40}"
    else
        echo "[缺失] ${cmd%% *} 未安装"
    fi
}

sec "0. 基本信息"
echo "时间     : $(date '+%F %T %Z')"
echo "主机名   : $(hostname 2>/dev/null)"
echo "当前用户 : $(whoami 2>/dev/null)  (uid=$(id -u 2>/dev/null))"
echo "是否 root: $([ "$(id -u)" = 0 ] && echo yes || echo no)"
echo "当前目录 : $(pwd)"
echo "内核     : $(uname -srm)"
echo "发行版   : $(grep -E '^PRETTY_NAME' /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '\"')"
echo "glibc    : $(ldd --version 2>/dev/null | head -1)"
echo "容器内?  : $([ -f /.dockerenv ] && echo 'yes (/.dockerenv 存在)' || echo 'probably no')"

sec "1. GPU 与互联拓扑"
# 型号/显存/驱动决定 torch 版本；拓扑决定 NCCL 要不要禁 P2P（4090 必须禁，H100 禁了反而慢）
run "nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv" 20
echo "--- CUDA 驱动/运行时 ---"
run "nvidia-smi --query-gpu=driver_version --format=csv,noheader" 2
have nvcc && nvcc --version 2>&1 | tail -2 || echo "[缺失] nvcc（只用预编译 torch 的话不阻塞）"
echo "--- NVLink 拓扑（看是否有 NV# / NVSwitch）---"
run "nvidia-smi topo -m" 25
echo "--- NVLink 状态 ---"
nvidia-smi nvlink -s 2>&1 | head -20 || echo "无 nvlink 子命令输出"
echo "--- fabricmanager（HGX/NVSwitch 机型必须在跑，否则 NVLink 不通）---"
systemctl is-active nvidia-fabricmanager 2>/dev/null || \
  (pgrep -a fabricmanager 2>/dev/null | head -3 || echo "未检测到 fabricmanager 进程")
echo "--- InfiniBand ---"
if [ -d /sys/class/infiniband ]; then
    echo "IB 设备: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')"
    have ibstat && ibstat -l 2>/dev/null | head -5
else
    echo "无 /sys/class/infiniband（无 IB，NCCL_IB_DISABLE 保持 1 即可）"
fi

sec "2. CPU / 内存"
echo "逻辑核数 : $(nproc 2>/dev/null)"
run "free -g" 5
echo "（注意：uno_1m_total_labels.json 是 810MB 单文件 JSON，json.load 峰值 >8GB 内存）"

sec "3. 磁盘空间"
# 重建至少需要：权重 ~70G + UNO-1M 压缩 124G + 解压 ~125G ≈ 320G 起
echo "--- 关键路径可用空间 ---"
for p in . "$HOME" /root /tmp /dev/shm; do
    [ -d "$p" ] && printf '%-24s %s\n' "$p" "$(df -h "$p" 2>/dev/null | awk 'NR==2{print $2" 总 / "$4" 可用  ("$1" -> "$6")"}')"
done
echo "--- 全部挂载点 ---"
run "df -hT" 30
echo "--- HF 缓存位置与占用 ---"
echo "HF_HOME=${HF_HOME:-<未设置，默认 ~/.cache/huggingface>}"
for d in "${HF_HOME:-$HOME/.cache/huggingface}" /root/.cache/huggingface; do
    [ -d "$d" ] && echo "$d : $(du -sh "$d" 2>/dev/null | cut -f1)"
done

sec "4. Python / 包管理器"
for c in python3 python3.10 python3.11 python3.12 python conda mamba uv pip pip3; do
    if have "$c"; then printf '%-10s %s  (%s)\n' "$c" "$($c --version 2>&1 | head -1)" "$(command -v $c)"; fi
done
echo "--- conda 环境列表 ---"
have conda && conda env list 2>&1 | head -10 || echo "[无 conda]"

sec "5. 已有的 UNO 环境（如果之前装过）"
for v in .venv-uno .venv venv; do
    if [ -x "$v/bin/python" ]; then
        echo "发现 $v : $("$v/bin/python" --version 2>&1)"
        "$v/bin/python" - <<'PY' 2>&1 | head -12
try:
    import torch
    print(f"  torch        {torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu          {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  torch 未装或不可用: {e}")
for m in ("transformers", "diffusers", "accelerate", "deepspeed", "wandb"):
    try:
        print(f"  {m:<13}{__import__(m).__version__}")
    except Exception as e:
        print(f"  {m:<13}未装 ({type(e).__name__})")
PY
    fi
done
[ -x .venv-uno/bin/python ] || echo "（未发现已有 venv，属正常：新机器只 clone 了代码）"

sec "6. 网络出口（决定 torch/权重/数据从哪拉）"
echo "--- 代理相关环境变量 ---"
env | grep -iE '^(http_proxy|https_proxy|no_proxy|all_proxy)=' || echo "（未设置任何代理变量）"
echo "--- pip 配置（是否有内网源）---"
have pip && pip config list 2>&1 | head -10 || echo "[pip 未装]"
[ -f /etc/pip.conf ] && { echo "/etc/pip.conf:"; cat /etc/pip.conf; }
[ -f "$HOME/.pip/pip.conf" ] && { echo "~/.pip/pip.conf:"; cat "$HOME/.pip/pip.conf"; }
[ -f "$HOME/.config/pip/pip.conf" ] && { echo "~/.config/pip/pip.conf:"; cat "$HOME/.config/pip/pip.conf"; }

echo "--- 连通性（HEAD 请求，8 秒超时）---"
if have curl; then
    for url in \
        https://pypi.org/simple/ \
        https://files.pythonhosted.org/ \
        https://download.pytorch.org/whl/cu124/ \
        https://mirrors.aliyun.com/pytorch-wheels/cu124/ \
        https://pypi.tuna.tsinghua.edu.cn/simple/ \
        https://huggingface.co/ \
        https://hf-mirror.com/ \
        https://github.com/ ; do
        code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 8 -I "$url" 2>/dev/null)
        printf '  %-52s HTTP %s %s\n' "$url" "${code:-000}" \
            "$([ "${code:-000}" = 000 ] && echo '❌不通' || echo '✅')"
    done
else
    echo "[缺失] curl —— 请先装 curl 再重跑本节"
fi

echo "--- 下载测速（各最多 12 秒，只测通的那些）---"
speed_test() {  # $1=标签 $2=URL
    have curl || return
    local out
    out=$(curl -sS -o /dev/null --max-time 12 -w '%{speed_download} %{size_download}' "$2" 2>/dev/null)
    local bps=${out%% *}
    if [ -z "$bps" ] || [ "${bps%%.*}" = "0" ]; then
        printf '  %-22s ❌ 不通或超时\n' "$1"
    else
        printf '  %-22s %.2f MB/s\n' "$1" "$(echo "${bps%%.*}/1048576" | bc -l 2>/dev/null || echo 0)"
    fi
}
speed_test "PyPI(files.python)"  "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz"
speed_test "阿里云 torch 轮子"   "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.4.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
speed_test "官方 pytorch.org"    "https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
speed_test "HuggingFace(公开)"   "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors"
speed_test "hf-mirror"           "https://hf-mirror.com/openai/clip-vit-large-patch14/resolve/main/model.safetensors"

sec "7. HuggingFace 凭证与已有权重"
# FLUX.1-dev 是 gated repo：必须有 token 且在网页上同意过协议，否则 403
echo "--- token 是否存在（只看有无，不打印内容）---"
for f in "$HOME/.cache/huggingface/token" "$HF_TOKEN_PATH"; do
    [ -f "$f" ] && echo "  ✅ 存在: $f  (长度 $(wc -c < "$f" | tr -d ' ') 字节)"
done
[ -n "${HF_TOKEN:-}" ] && echo "  ✅ 环境变量 HF_TOKEN 已设置（长度 ${#HF_TOKEN}）"
[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && echo "  ✅ 环境变量 HUGGING_FACE_HUB_TOKEN 已设置"
echo "  （以上都没有 = 需要先 huggingface-cli login，且 FLUX.1-dev 要在网页同意协议）"
echo "--- 已缓存的模型仓库 ---"
HFHUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -d "$HFHUB" ]; then
    # 注意：不能写成 `du ... || echo`——du 在管道里失败时 head 仍返回 0，兜底永远不触发
    if ls -d "$HFHUB"/models--* >/dev/null 2>&1; then
        du -sh "$HFHUB"/models--* 2>/dev/null | head -20
    else
        echo "  （hub 目录存在但没有任何已下载的模型）"
    fi
else
    echo "  无 $HFHUB —— 权重需要全新下载（约 70 GB）"
fi

sec "8. 仓库状态"
echo "--- git ---"
run "git rev-parse --show-toplevel" 2
run "git remote -v" 6
echo "分支: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)  HEAD: $(git log --oneline -1 2>/dev/null)"
echo "--- submodule（datasets/dreambooth 不 init，训练到第一次 checkpoint 才崩）---"
run "git submodule status" 5
echo "dreambooth 目录文件数: $(ls datasets/dreambooth/dataset 2>/dev/null | wc -l | tr -d ' ')"
echo "--- 数据集 / 权重 / checkpoint 现状 ---"
for p in datasets/UNO-1M datasets/UNO-1M/images log data/multibanana output wheels; do
    if [ -e "$p" ]; then printf '  %-26s 存在  %s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
    else printf '  %-26s 不存在\n' "$p"; fi
done

sec "9. 常用工具"
for c in git curl wget aria2c tar unzip pigz tmux screen rsync jq bc make gcc g++; do
    printf '%-8s %s\n' "$c" "$(have "$c" && command -v "$c" || echo '[缺失]')"
done

sec "10. 探测完成"
echo "把本文件完整内容贴回给上游即可：/tmp/probe_report.txt"
