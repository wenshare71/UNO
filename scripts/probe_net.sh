#!/usr/bin/env bash
# 第二轮探测：补齐 probe_env.sh 没测到的部分。
#
# 第一轮机器上 curl 和 bc 都没装，导致「网络出口」整节为空——而这一节决定
# torch 从哪装、权重从哪拉。本脚本改用 wget / python3（都已确认存在）重测，
# 另外补上首轮没覆盖但会卡住重建的三件事：pip/venv 能否自举、各挂载点可写性、
# ceph 与本地 NVMe 的实际读写速度。
#
# 用法（UNO 仓库根目录）：
#   bash scripts/probe_net.sh 2>&1 | tee /tmp/probe_net.txt
#
# 只读为主：只在 /tmp 下建临时文件和临时 venv，结束时自己删掉。

sec() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

sec "1. 网络连通性 + 测速（python3 urllib，自动遵守 http_proxy / no_proxy）"
python3 - <<'PY' 2>&1
import os, socket, ssl, time, urllib.request, urllib.error

print("生效的代理设置:")
for k in ("http_proxy", "https_proxy", "no_proxy"):
    print(f"  {k:<12}= {os.environ.get(k, '<未设置>')}")
print()

# (标签, URL, 说明)  —— internal 的走 no_proxy 直连，其余走海外代理
TARGETS = [
    ("PyPI 官方",        "https://pypi.org/simple/"),
    ("PyPI 文件源",      "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz"),
    ("快手内网 PyPI",    "https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/"),
    ("清华 PyPI",        "https://pypi.tuna.tsinghua.edu.cn/simple/"),
    ("PyTorch 官方轮子", "https://download.pytorch.org/whl/cu124/"),
    ("阿里云 torch 轮子","https://mirrors.aliyun.com/pytorch-wheels/cu124/"),
    ("HuggingFace",      "https://huggingface.co/"),
    ("hf-mirror",        "https://hf-mirror.com/"),
    ("GitHub",           "https://github.com/"),
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE   # 只测连通性，不做证书校验（内网源常用自签）

def probe(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "probe/1.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            r.read(2048)
            return f"✅ HTTP {r.status}", time.time() - t0
    except urllib.error.HTTPError as e:
        # 403/404 也算「网络通」——能拿到 HTTP 状态说明链路没问题
        return f"⚠️  HTTP {e.code}（链路通，仅状态码非 200）", time.time() - t0
    except Exception as e:
        return f"❌ {type(e).__name__}: {str(e)[:60]}", time.time() - t0

print("连通性:")
reachable = []
for label, url in TARGETS:
    status, dt = probe(url)
    print(f"  {label:<18} {status}   ({dt:.1f}s)")
    if status.startswith("✅") or status.startswith("⚠️"):
        reachable.append((label, url))

# ---- 测速：对能通的大文件源，限时 10 秒看实际吞吐 ----
SPEED = [
    ("PyPI 文件源",       "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz"),
    ("阿里云 torch 轮子", "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.4.0%2Bcu124-cp310-cp310-linux_x86_64.whl"),
    ("PyTorch 官方轮子",  "https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp310-cp310-linux_x86_64.whl"),
    ("HuggingFace 公开档","https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors"),
    ("hf-mirror 公开档",  "https://hf-mirror.com/openai/clip-vit-large-patch14/resolve/main/model.safetensors"),
]
print("\n测速（每项最多 10 秒 / 200 MB）:")
for label, url in SPEED:
    req = urllib.request.Request(url, headers={"User-Agent": "probe/1.0"})
    try:
        t0 = time.time(); got = 0
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            while time.time() - t0 < 10 and got < 200 * 1024 * 1024:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
        dt = time.time() - t0
        if got == 0:
            print(f"  {label:<20} ❌ 未取到数据")
        else:
            print(f"  {label:<20} {got/1048576/dt:6.2f} MB/s   (取了 {got/1048576:.0f} MB)")
    except Exception as e:
        print(f"  {label:<20} ❌ {type(e).__name__}: {str(e)[:50]}")
PY

sec "2. pip / venv 能否自举（非 root、无 conda，这是重建的第一个卡点）"
echo "python3        : $(python3 --version 2>&1)  ($(command -v python3))"
echo "--- ensurepip 是否可用 ---"
python3 -m ensurepip --version 2>&1 | head -3
echo "--- venv 模块是否可用 ---"
python3 -c "import venv; print('  venv 模块可导入 ✅')" 2>&1 | head -3
echo "--- 实际建一个临时 venv 试试（最能说明问题）---"
rm -rf /tmp/_probe_venv
if python3 -m venv /tmp/_probe_venv 2>&1 | head -8; then
    if [ -x /tmp/_probe_venv/bin/pip ]; then
        echo "  ✅ venv + pip 创建成功: $(/tmp/_probe_venv/bin/pip --version 2>&1)"
    elif [ -x /tmp/_probe_venv/bin/python ]; then
        echo "  ⚠️  venv 建出来了但没有 pip（缺 python3.10-venv 的 ensurepip 部分）"
    fi
else
    echo "  ❌ venv 创建失败——需要走 get-pip.py 或用户级 miniconda/uv"
fi
rm -rf /tmp/_probe_venv
echo "--- 是否有 sudo / apt 权限 ---"
have sudo && (sudo -n true 2>/dev/null && echo "  sudo 免密可用 ✅" || echo "  有 sudo 但需要密码/不可用") || echo "  [无 sudo]"
echo "--- 系统里是否已有 torch ---"
python3 -c "import torch; print('  系统 torch:', torch.__version__)" 2>&1 | head -2

sec "3. 各挂载点可写性（决定 HF_HOME 和数据集放哪）"
for d in /code /root /tmp /dev/shm "$HOME" "$(pwd)" /kaimm-distill/wuwenxuan; do
    if [ ! -d "$d" ]; then printf '  %-28s 不存在\n' "$d"; continue; fi
    if touch "$d/.probe_write_test" 2>/dev/null; then
        rm -f "$d/.probe_write_test"
        printf '  %-28s ✅ 可写   (可用 %s)\n' "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')"
    else
        printf '  %-28s ❌ 只读/无权限\n' "$d"
    fi
done

sec "4. 磁盘吞吐：ceph vs 本地 NVMe（决定 12 万张训练图放哪）"
bench_io() {  # $1=标签 $2=目录
    local d="$2" f
    [ -d "$d" ] || { printf '  %-26s 目录不存在\n' "$1"; return; }
    f="$d/.probe_io_$$"
    touch "$f" 2>/dev/null || { printf '  %-26s 不可写，跳过\n' "$1"; return; }
    # 写 2GB，绕过页缓存看真实带宽
    local w r
    w=$(dd if=/dev/zero of="$f" bs=1M count=2048 oflag=direct 2>&1 | tail -1 | grep -oE '[0-9.]+ [MG]B/s' | tail -1)
    [ -z "$w" ] && w=$(dd if=/dev/zero of="$f" bs=1M count=2048 conv=fsync 2>&1 | tail -1 | grep -oE '[0-9.]+ [MG]B/s' | tail -1)
    r=$(dd if="$f" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1 | grep -oE '[0-9.]+ [MG]B/s' | tail -1)
    [ -z "$r" ] && r=$(dd if="$f" of=/dev/null bs=1M 2>&1 | tail -1 | grep -oE '[0-9.]+ [MG]B/s' | tail -1)
    rm -f "$f"
    printf '  %-26s 写 %-12s 读 %-12s\n' "$1" "${w:-N/A}" "${r:-N/A}"
}
bench_io "ceph (仓库所在)"    "$(pwd)"
bench_io "本地 NVMe (/code)"  "/code"
bench_io "overlay (/tmp)"     "/tmp"

sec "5. 已有资产盘点（哪些还需要从旧机器拷）"
echo "--- log/ 下的 checkpoint ---"
ls -d log/*/ 2>/dev/null | head -5
for d in log/*/; do
    [ -d "$d" ] && echo "  $d : $(ls -d "$d"checkpoint-* 2>/dev/null | wc -l | tr -d ' ') 个 checkpoint, 最新 $(ls -d "$d"checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)"
done
echo "--- datasets/UNO-1M 现状（完整应为 labels 810MB + images 解压后约 125GB）---"
ls -la datasets/UNO-1M/ 2>/dev/null | head -10
echo "  images 下条目数: $(ls datasets/UNO-1M/images 2>/dev/null | wc -l | tr -d ' ')"
echo "  labels json    : $(ls -lh datasets/UNO-1M/*.json 2>/dev/null | awk '{print $5, $9}' | tr '\n' ' ')"
echo "--- dreambooth（submodule 状态是 - 但文件已拷入，不要再 git submodule update）---"
echo "  subject 目录数: $(ls -d datasets/dreambooth/dataset/*/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  抽查图片数    : $(find datasets/dreambooth/dataset -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
echo "--- HF 缓存（若旧机器权重已拷入会显示在这里）---"
echo "  HF_HOME=${HF_HOME:-<未设置，默认 ~/.cache/huggingface>}"
for d in "$HOME/.cache/huggingface/hub" /root/.cache/huggingface/hub "${HF_HOME:-}/hub"; do
    [ -d "$d" ] && { echo "  $d:"; du -sh "$d"/models--* 2>/dev/null | head -10; }
done

sec "5b. 权重全盘定位（拷到非默认路径时，上面那节会假装什么都没有）"
# 只按目录名找，不遍历文件内容，避免在 P 级 ceph 上跑飞。
# maxdepth 6 足够覆盖 <root>/.cache/huggingface/hub/models--org--name。
for root in "$HOME" /code /root /tmp; do
    [ -d "$root" ] || continue
    echo "--- 扫描 $root ---"
    timeout 120 find "$root" -maxdepth 6 -type d \
        \( -name 'models--*' -o -iname '*FLUX.1-dev*' -o -iname 'xflux_text_encoders' \) \
        -not -path '*/.git/*' 2>/dev/null | head -20
done
echo "--- 散装权重文件（没走 HF 缓存、直接拷 safetensors 的情况）---"
for root in "$HOME" /code; do
    [ -d "$root" ] || continue
    timeout 120 find "$root" -maxdepth 5 -type f -size +1G \
        \( -name '*.safetensors' -o -name '*.sft' -o -name '*.ckpt' \) \
        -not -path '*/log/*' 2>/dev/null | head -20
done

sec "5c. UNO-1M 自洽性：labels 里引用的图片是否真的在盘上"
# 只有 6.1G 图片而 labels 是全量的话，dataloader 会在随机取到缺失样本时才崩，
# 训练跑几百步之后炸最难查——所以现在先抽 200 条对一遍。
python3 - <<'PY' 2>&1
import json, os, glob, random
cands = sorted(glob.glob("datasets/UNO-1M/*.json"))
if not cands:
    print("  ❌ datasets/UNO-1M 下没有任何 json —— labels 还没拷过来")
    raise SystemExit
for p in cands:
    print(f"  发现 {p}  ({os.path.getsize(p)/1e6:.0f} MB)")
# 优先检查转换后的（训练真正吃的那份）
pick = next((p for p in cands if "convert" in p), cands[0])
print(f"  抽查: {pick}")
try:
    with open(pick) as f:
        data = json.load(f)
except Exception as e:
    print(f"  ❌ 读不动: {type(e).__name__}: {e}")
    raise SystemExit
print(f"  条目总数: {len(data)}")
root = os.path.dirname(pick)
random.seed(0)
sample = random.sample(data, min(200, len(data)))
miss_tgt = miss_ref = 0
for it in sample:
    t = it.get("image_tgt_path")
    if t and not os.path.exists(os.path.join(root, t)):
        miss_tgt += 1
    for r in it.get("image_paths", []) or ([it["image_path"]] if "image_path" in it else []):
        if not os.path.exists(os.path.join(root, r)):
            miss_ref += 1
            break
n = len(sample)
print(f"  抽样 {n} 条: 目标图缺失 {miss_tgt} ({miss_tgt/n:.0%})，参考图缺失 {miss_ref} ({miss_ref/n:.0%})")
if miss_tgt or miss_ref:
    print("  ⚠️  labels 是全量但图片只拷了一部分 —— 训练前必须按实际存在的图片重新过滤 labels")
else:
    print("  ✅ 抽样全部命中")
PY

sec "6. 探测完成"
echo "把 /tmp/probe_net.txt 完整贴回即可。"
