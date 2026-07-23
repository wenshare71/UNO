#!/usr/bin/env bash
# 第三轮探测：只补两个决定性未知数 + 一项数据自洽性抽查。
#
# 背景（来自 probe_net.sh 结果）：
#   机器所有出网都走 http://oversea-squid1.jp.txyun:11080（日本代理），实测
#   0.05~0.66 MB/s——装 torch 全家桶（约 2.5 GB）要 1~2 小时。唯一命中 no_proxy
#   的是 pypi.corp.kuaishou.com（0.2s 握手，直连），但上一轮没测它的吞吐，
#   也没确认它有没有 torch 轮子。这两件事决定装依赖走哪条路。
#   同时 python3 缺 ensurepip 导致 venv 建不出 pip，但 sudo 免密可用，
#   所以要确认 apt 能否装上 python3.10-venv。
#
# 用法（UNO 仓库根目录）：
#   bash scripts/probe_pypi.sh 2>&1 | tee /tmp/probe_pypi.txt
#
# 只读：apt 用 --dry-run，不实际安装；下载只取前 10 秒到 /dev/null。

sec() { printf '\n===== %s =====\n' "$1"; }

sec "1. 内网 PyPI：有没有 torch、实测多快（唯一不走日本代理的路径）"
python3 - <<'PY' 2>&1
import re, time, urllib.request, urllib.error, urllib.parse

INDEX = "https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/"

def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "pip/24.0"})
    return urllib.request.urlopen(req, timeout=timeout)

for pkg in ("torch", "deepspeed", "transformers"):
    url = urllib.parse.urljoin(INDEX, f"{pkg}/")
    try:
        t0 = time.time()
        html = fetch(url).read().decode("utf-8", "ignore")
        print(f"\n[{pkg}] 索引 {len(html)/1024:.0f} KB, {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n[{pkg}] ❌ {type(e).__name__}: {str(e)[:80]}")
        continue

    hrefs = re.findall(r'href="([^"]+)"', html)
    # 只看能装在这台机器上的：cp310 + linux x86_64（或 py3-none-any）
    ok = [h for h in hrefs
          if ("cp310" in h or "py3-none-any" in h)
          and ("linux_x86_64" in h or "py3-none-any" in h)]
    print(f"  cp310/linux 候选轮子: {len(ok)} 个（总链接 {len(hrefs)}）")
    if pkg == "torch":
        v240 = [h for h in ok if "torch-2.4.0" in h]
        print(f"  其中 torch-2.4.0: {len(v240)} 个")
        for h in v240[:6]:
            print("    ", h.split("#")[0].split("/")[-1])
        if not v240:
            print("    ⚠️ 内网源没有 torch 2.4.0 的 cp310 轮子——最近几个版本:")
            for h in ok[-6:]:
                print("    ", h.split("#")[0].split("/")[-1])

    # 拿最后一个候选测吞吐（通常是最新版，体积有代表性）
    if ok:
        target = urllib.parse.urljoin(url, ok[-1].split("#")[0])
        try:
            t0, got = time.time(), 0
            with fetch(target, timeout=20) as r:
                while time.time() - t0 < 10 and got < 300 * 1024 * 1024:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    got += len(chunk)
            dt = time.time() - t0
            name = target.split("/")[-1][:50]
            if got:
                print(f"  测速 {name}: {got/1048576/dt:7.2f} MB/s  (取了 {got/1048576:.0f} MB)")
            else:
                print(f"  测速 {name}: ❌ 未取到数据")
        except Exception as e:
            print(f"  测速失败: {type(e).__name__}: {str(e)[:80]}")
PY

sec "2. apt 能否装上 python3.10-venv（venv 缺 ensurepip 的正规解法）"
echo "--- apt 源 ---"
grep -rhE '^\s*deb ' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | head -8
echo "--- apt 代理配置 ---"
grep -rh -i proxy /etc/apt/apt.conf.d/ /etc/apt/apt.conf 2>/dev/null | head -5 || echo "  （apt 无独立代理配置，将继承 http_proxy 环境变量）"
echo "--- update + dry-run 安装（不实际改系统）---"
sudo -n apt-get update -qq 2>&1 | tail -5
sudo -n apt-get install -y --dry-run python3.10-venv python3-pip curl 2>&1 | tail -15

sec "3. UNO-1M labels 与图片是否对得上（118G 已整份拷入，此处是确认而非排查）"
python3 - <<'PY' 2>&1
import json, os, glob, random
cands = sorted(glob.glob("datasets/UNO-1M/*.json"))
if not cands:
    print("  ❌ 没有任何 labels json")
    raise SystemExit
pick = next((p for p in cands if "convert" in p), cands[0])
print(f"  抽查: {pick}  ({os.path.getsize(pick)/1e6:.0f} MB)")
with open(pick) as f:
    data = json.load(f)
print(f"  条目总数: {len(data)}")
root = os.path.dirname(pick)
random.seed(0)
sample = random.sample(data, min(300, len(data)))
miss_tgt = miss_ref = 0
for it in sample:
    t = it.get("image_tgt_path")
    if t and not os.path.exists(os.path.join(root, t)):
        miss_tgt += 1
    refs = it.get("image_paths") or ([it["image_path"]] if "image_path" in it else [])
    if any(not os.path.exists(os.path.join(root, r)) for r in refs):
        miss_ref += 1
n = len(sample)
print(f"  抽样 {n} 条: 目标图缺失 {miss_tgt} ({miss_tgt/n:.0%})，参考图缺失 {miss_ref} ({miss_ref/n:.0%})")
print("  ✅ 抽样全部命中" if not (miss_tgt or miss_ref)
      else "  ⚠️ 有缺失——需在本机重跑 scripts/convert_uno_labels.py 重新生成 convert json")
# 顺带确认 ref 数量分布（蒸馏前的基线事实：应 100% 单 ref）
from collections import Counter
c = Counter(len(it.get("image_paths", [])) for it in data)
print(f"  ref 数量分布: {dict(sorted(c.items()))}")
PY

sec "4. checkpoint 实际到哪一步（清单写 13000，probe 显示 9000，需对齐）"
ls -d log/ref_isolation/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tr '\n' ' '; echo
du -sh log/ref_isolation/checkpoint-* 2>/dev/null | tail -3
echo "--- 每份 ckpt 里有什么 ---"
ls -la "$(ls -d log/ref_isolation/checkpoint-* 2>/dev/null | tail -1)" 2>/dev/null

sec "5. 探测完成"
