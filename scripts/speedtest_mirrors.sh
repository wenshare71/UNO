#!/usr/bin/env bash
# 镜像测速脚本：每个源实测下载同一个 torch 轮子的前 32MB，输出 MB/s
# 用法: bash scripts/speedtest_mirrors.sh
set -uo pipefail

RANGE_BYTES=$((32 * 1024 * 1024))   # 每个源测 32MB
WHL="torch-2.4.0%2Bcu124-cp312-cp312-linux_x86_64.whl"

# 候选源: 名字|测速 URL
CANDIDATES=(
  "官方 download.pytorch.org|https://download.pytorch.org/whl/cu124/${WHL}"
  "南京大学 NJU|https://mirror.nju.edu.cn/pytorch/whl/cu124/${WHL}"
  "上海交大 SJTU|https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/${WHL}"
  "阿里云 Aliyun|https://mirrors.aliyun.com/pytorch-wheels/cu124/${WHL}"
)

echo "== PyTorch 轮子源测速（各拉 32MB，约几十秒/源）=="
printf "%-28s %10s %10s\n" "源" "MB/s" "HTTP"
for entry in "${CANDIDATES[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    # -r 只取前 32MB；-m 90 单源最多 90 秒防卡死
    result=$(curl -sL -r 0-$((RANGE_BYTES - 1)) -o /dev/null -m 90 \
                  -w "%{speed_download} %{http_code}" "$url" 2>/dev/null) || result="0 000"
    speed_bps="${result%% *}"
    http_code="${result##* }"
    mbps=$(awk -v s="$speed_bps" 'BEGIN { printf "%.2f", s / 1048576 }')
    printf "%-28s %10s %10s\n" "$name" "$mbps" "$http_code"
done

echo
echo "== 快手内网源是否有 cu124 轮子（有则内网源最优）=="
corp_index="https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/torch/"
hits=$(curl -sL -m 30 "$corp_index" | grep -oi 'torch-2\.4\.0[^"<]*cu124[^"<]*cp312[^"<]*' | sort -u)
if [ -n "$hits" ]; then
    echo "✅ 内网源有 cu124 轮子:"
    echo "$hits"
    # 顺带测速
    whl_url=$(curl -sL -m 30 "$corp_index" | grep -o 'href="[^"]*torch-2\.4\.0[^"]*cu124[^"]*cp312[^"]*whl[^"]*"' | head -1 | sed 's/href="//;s/"$//')
    if [ -n "$whl_url" ]; then
        case "$whl_url" in
            http*) : ;;
            /*)  whl_url="https://pypi.corp.kuaishou.com${whl_url}" ;;
            *)   whl_url="${corp_index}${whl_url}" ;;
        esac
        result=$(curl -sL -r 0-$((RANGE_BYTES - 1)) -o /dev/null -m 90 \
                      -w "%{speed_download} %{http_code}" "$whl_url" 2>/dev/null) || result="0 000"
        mbps=$(awk -v s="${result%% *}" 'BEGIN { printf "%.2f", s / 1048576 }')
        printf "%-28s %10s %10s\n" "快手内网源" "$mbps" "${result##* }"
    fi
else
    echo "❌ 内网源 simple 索引里没找到 torch cu124 轮子（或索引页无法访问）"
fi

echo
echo "== HuggingFace 端点测速（拉 UNO-1M 标签文件前 32MB）=="
HF_FILE="datasets/bytedance-research/UNO-1M/resolve/main/uno_1m_total_labels.json"
for entry in "官方 huggingface.co|https://huggingface.co/${HF_FILE}" \
             "国内镜像 hf-mirror.com|https://hf-mirror.com/${HF_FILE}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    result=$(curl -sL -r 0-$((RANGE_BYTES - 1)) -o /dev/null -m 90 \
                  -w "%{speed_download} %{http_code}" "$url" 2>/dev/null) || result="0 000"
    mbps=$(awk -v s="${result%% *}" 'BEGIN { printf "%.2f", s / 1048576 }')
    printf "%-28s %10s %10s\n" "$name" "$mbps" "${result##* }"
done

echo
echo "完成。把整段输出发回即可。HTTP 应为 206/200；000 表示连不上。"
