#!/usr/bin/env bash
# 多镜像并行下载 torch/torchvision cu124 轮子，绕过单连接限速
# 用法: bash scripts/fetch_torch_wheels.sh
# 完成后: pip install wheels/torch-*.whl wheels/torchvision-*.whl
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p wheels
JOBS=16   # 并行连接数

# 名字(本地文件名)|URL路径(编码后)|sha256
WHEELS=(
  "torch-2.4.0+cu124-cp312-cp312-linux_x86_64.whl|torch-2.4.0%2Bcu124-cp312-cp312-linux_x86_64.whl|f6c94ca3a403e79fd25d4bc2ea7325de7c6682a372c5d525b65b06367b1cc618"
  "torchvision-0.19.0+cu124-cp312-cp312-linux_x86_64.whl|torchvision-0.19.0%2Bcu124-cp312-cp312-linux_x86_64.whl|2aa511f7073928cc3cab03d1e5524d840a7a17da7f9bde7e771116c233d011e8"
)

# 各镜像的目录前缀（文件在这些目录下同名、字节一致）
MIRRORS=(
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/"
  "https://mirror.nju.edu.cn/pytorch/whl/cu124/"
  "https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/"
  "https://download.pytorch.org/whl/cu124/"
)

sha_check() {  # sha_check <file> <expected>
    local got
    got=$(sha256sum "$1" | awk '{print $1}')
    [ "$got" = "$2" ]
}

fetch_one() {  # fetch_one <本地名> <URL文件名> <sha256>
    local out="wheels/$1" urlname="$2" sha="$3"

    if [ -f "$out" ] && sha_check "$out" "$sha"; then
        echo "✅ $1 已存在且校验通过，跳过"
        return 0
    fi

    # ---- 方案 A: aria2c 多镜像多连接 ----
    if command -v aria2c >/dev/null 2>&1; then
        echo "== aria2c 下载 $1 (${JOBS} 连接 x ${#MIRRORS[@]} 镜像) =="
        local uris=()
        for m in "${MIRRORS[@]}"; do uris+=("${m}${urlname}"); done
        aria2c -x "$JOBS" -s $((JOBS * 2)) -k 1M --min-split-size=1M \
               --checksum=sha-256="$sha" --auto-file-renaming=false --allow-overwrite=true \
               -d wheels -o "$1" "${uris[@]}"
        return 0
    fi

    # ---- 方案 B: curl 并行分片（无需安装任何东西）----
    echo "== 未找到 aria2c，用 curl ${JOBS} 路并行分片下载 $1 =="
    local url="${MIRRORS[0]}${urlname}"
    local size
    size=$(curl -sIL "$url" | grep -i '^content-length:' | tail -1 | tr -dc '0-9')
    [ -n "$size" ] || { echo "❌ 拿不到文件大小: $url"; return 1; }
    echo "   文件大小: $((size / 1048576)) MB"

    local chunk=$(( (size + JOBS - 1) / JOBS ))
    local tmpdir="wheels/.parts_$1"
    mkdir -p "$tmpdir"
    local pids=()
    for i in $(seq 0 $((JOBS - 1))); do
        local start=$((i * chunk))
        local end=$((start + chunk - 1))
        [ "$end" -ge "$size" ] && end=$((size - 1))
        [ "$start" -gt "$end" ] && break
        # 分片轮流分配给不同镜像，绕开单镜像单连接限速
        local mirror="${MIRRORS[$((i % ${#MIRRORS[@]}))]}"
        ( curl -sL --retry 5 --retry-delay 2 -r "${start}-${end}" \
               -o "$tmpdir/part_$(printf '%02d' "$i")" "${mirror}${urlname}" ) &
        pids+=($!)
    done
    local fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ "$fail" -eq 0 ] || { echo "❌ 有分片下载失败，重跑本脚本会续传"; return 1; }

    cat "$tmpdir"/part_* > "$out"
    if sha_check "$out" "$sha"; then
        rm -rf "$tmpdir"
        echo "✅ $1 下载完成，sha256 校验通过"
    else
        echo "❌ $1 sha256 校验失败，已删除，请重跑"
        rm -f "$out"
        return 1
    fi
}

for w in "${WHEELS[@]}"; do
    IFS='|' read -r localname urlname sha <<< "$w"
    fetch_one "$localname" "$urlname" "$sha"
done

cat <<'EOF'

✅ 全部轮子就绪。接下来在激活的 venv 里执行：
  pip install wheels/torch-2.4.0+cu124-cp312-cp312-linux_x86_64.whl \
              wheels/torchvision-0.19.0+cu124-cp312-cp312-linux_x86_64.whl
  pip install -e ".[train]"
  pip install hf_transfer "huggingface_hub[cli]"
EOF
