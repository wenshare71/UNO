#!/usr/bin/env bash
# 镜像 x 代理方式 矩阵测速：每个组合实测下载 torch 轮子的前 16MB，输出 MB/s
# 用法: bash scripts/speedtest_mirrors.sh
set -uo pipefail

RANGE_BYTES=$((16 * 1024 * 1024))   # 每个组合测 16MB（组合多，单次减小）
WHL="torch-2.4.0%2Bcu124-cp312-cp312-linux_x86_64.whl"

OVERSEA_PROXY="http://oversea-squid1.jp.txyun:11080"
INTERNAL_PROXY="http://10.68.24.160:11080"

# 名字|URL
MIRRORS=(
  "阿里云|https://mirrors.aliyun.com/pytorch-wheels/cu124/${WHL}"
  "南京大学|https://mirror.nju.edu.cn/pytorch/whl/cu124/${WHL}"
  "上海交大|https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/${WHL}"
  "官方pytorch.org|https://download.pytorch.org/whl/cu124/${WHL}"
  "HF官方|https://huggingface.co/datasets/bytedance-research/UNO-1M/resolve/main/uno_1m_total_labels.json"
  "hf-mirror|https://hf-mirror.com/datasets/bytedance-research/UNO-1M/resolve/main/uno_1m_total_labels.json"
)

# 代理方式: 名字|proxy值（direct = 清空代理直连）
PROXY_MODES=(
  "海外代理|${OVERSEA_PROXY}"
  "内网代理|${INTERNAL_PROXY}"
  "直连|direct"
)

measure() {  # measure <url> <proxy>  -> "MB/s HTTP码"
    local url="$1" proxy="$2" result
    if [ "$proxy" = "direct" ]; then
        result=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
                 curl -sL -r 0-$((RANGE_BYTES - 1)) -o /dev/null -m 60 \
                      -w "%{speed_download} %{http_code}" "$url" 2>/dev/null) || result="0 000"
    else
        result=$(env http_proxy="$proxy" https_proxy="$proxy" \
                 curl -sL -r 0-$((RANGE_BYTES - 1)) -o /dev/null -m 60 \
                      -w "%{speed_download} %{http_code}" "$url" 2>/dev/null) || result="0 000"
    fi
    local mbps
    mbps=$(awk -v s="${result%% *}" 'BEGIN { printf "%.2f", s / 1048576 }')
    echo "${mbps} ${result##* }"
}

echo "== 镜像 x 代理 矩阵测速（每格 16MB，HTTP 应为 206/200，000=连不上）=="
printf "%-16s" "源\\代理"
for pm in "${PROXY_MODES[@]}"; do printf " %16s" "${pm%%|*}"; done
echo
for entry in "${MIRRORS[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    printf "%-16s" "$name"
    for pm in "${PROXY_MODES[@]}"; do
        proxy="${pm##*|}"
        read -r mbps code <<< "$(measure "$url" "$proxy")"
        printf " %10s(%s)" "$mbps" "$code"
    done
    echo
done

echo
echo "完成。把整段输出发回。速度单位 MB/s，括号内为 HTTP 状态码。"
