# P2 前置 · 执行报告

> 对应 `qwen/P2_PREFLIGHT_RUN.md`。两件事:A 探内网源 cp310(4090,纯 CPU),
> B 投 infer_hub bf16 确认(未开始,待 push 后进行)。
> 执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),执行时间 2026-08-13。

---

# A · 内网源 cp310 探测

## A0 · 两个环境问题(原命令跑不通,已用等价替代)

手册原命令有两个假设在本机不成立,探测命令因此做了等价调整,**探的问题不变**:

| # | 原假设 | 实测 | 调整 |
|---|---|---|---|
| 1 | pip 能直连 `pypi.corp.kuaishou.com` | env 的 `http_proxy/https_proxy=oversea-squid1...` 把内网源也塞进海外代理,`pip index versions torch` **502 Bad Gateway**;该源解析到内网 `10.20.248.16`,应直连 | 命令前加 `no_proxy=pypi.corp.kuaishou.com NO_PROXY=...`(curl 探测用 `--noproxy '*'`) |
| 2 | `pip download --dry-run` 可用 | 本环境 **pip 26.2**,`pip download` 的 `--dry-run` 选项已被移除(报 `no such option: --dry-run`) | 改用 `pip install --dry-run --only-binary=:all: --no-deps`(pip ≥23 保留),外加直接 grep simple 索引页 |

调整后结论见各节。

## A1 · 源上 torch 有哪些版本

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
IDX=https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/
$E/bin/pip index versions torch --index-url $IDX      # 加 no_proxy=pypi.corp.kuaishou.com
```

输出(直连后):

```text
torch (2.13.0)
Available versions: 2.13.0, 2.12.1, 2.12.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.1, 2.7.0, 2.6.0, 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.1, 2.3.0, 2.2.2, 2.2.1, 2.2.0, 2.1.2, 2.1.1, 2.1.0, 2.0.1, 2.0.0, 1.13.1, 1.13.0
  INSTALLED: 2.5.1+cu124
  LATEST:    2.13.0
```

**2.5.1 在可用版本列表里。**

## A2 · 关键一问:cp310 的 2.5.1 能不能下

### A2.1 simple 索引页直接证据(零下载)

```bash
curl --noproxy '*' -s "$IDX/torch/" | grep -oE 'torch-2\.5\.1[^"#<>]*\.whl'
```

命中(该页 2.5.1 全部轮子,按标签分类):

```text
torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl        ← 本机目标(cp310 / manylinux1)
torch-2.5.1-cp310-cp310-manylinux2014_aarch64.whl
torch-2.5.1-cp310-cp310-win_amd64.whl
torch-2.5.1-cp310-none-macosx_11_0_arm64.whl
torch-2.5.1-cp311-... / cp312-... / cp313-... / cp39-... (同上四平台)
```

### A2.2 pip 解析器实测(等价替代 dry-run)

```bash
$E/bin/pip install --dry-run --only-binary=:all: --no-deps \
    --python-version 3.10 --implementation cp --abi cp310 \
    --platform manylinux1_x86_64 --target "$D/x" \
    torch==2.5.1 --index-url $IDX      # 加 no_proxy;--dry-run 不安装
```

输出:

```text
Looking in indexes: https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/
Collecting torch==2.5.1
  Downloading torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl (906.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 906.4/906.4 MB 375.2 MB/s  0:00:02
Would install torch-2.5.1
```

退出码 0。注:`--dry-run` 仍会把轮子真下载进 pip 缓存(906 MB,不安装、不污染 env),这条恰好证明整包在源上完好可拉。

### 结论

> **torch 2.5.1 cp310:能**

## A3 · 其余几个包(cp310 逐个解析)

原命令逐包 `pip download --dry-run`(因 A0-2 改用 `pip install --dry-run`,其余参数同):

```bash
for pkg in "transformers==5.14.1" "accelerate==1.14.0" "peft==0.20.0" \
           "safetensors" "sentencepiece" "einops"; do
  $E/bin/pip install --dry-run --only-binary=:all: --no-deps \
      --python-version 3.10 --implementation cp --abi cp310 \
      --platform manylinux1_x86_64 --target "$D/x" "$pkg" --index-url $IDX
done
```

逐包结果(`Would install` = 可解析):

```text
=== transformers==5.14.1 ===   Would install transformers-5.14.1
=== accelerate==1.14.0 ===     Would install accelerate-1.14.0
=== peft==0.20.0 ===           Would install peft-0.20.0
=== safetensors ===            ERROR: Could not find a version that satisfies the requirement safetensors (from versions: none)
=== sentencepiece ===          ERROR: Could not find a version that satisfies the requirement sentencepiece (from versions: none)
=== einops ===                 Would install einops-0.8.2
```

### A3.1 后两个失败的两层原因(已逐层验证)

**第一层:`kuaishou/prod` 聚合索引有白名单,不含这俩。**

```bash
curl -s -o /dev/null -w "%{http_code}" https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/safetensors/
# 404(devpi "Not Found")
curl -s -o /dev/null -w "%{http_code}" https://pypi.corp.kuaishou.com/root/pypi/+simple/safetensors/
# 200(root/pypi 上游镜像有)
```

sentencepiece 同理(prod 404 / root/pypi 200)。torch 页里的文件实际也挂在 `root/pypi` 源下,`kuaishou/prod` 是聚合上游的索引。

**第二层:就算换 root/pypi,这俩的 cp310 轮子是 `manylinux_2_17`/`manylinux2014` 标签,手册的 `--platform manylinux1` 太严(编译轮子不吃 manylinux1)。** 用正确标签实测:

```bash
$E/bin/pip install --dry-run --only-binary=:all: --no-deps \
    --python-version 3.10 --implementation cp --abi cp310 \
    --platform manylinux2014_x86_64 --target "$D/x" \
    safetensors --index-url https://pypi.corp.kuaishou.com/root/pypi/+simple/
```

输出:

```text
=== safetensors @ root/pypi, platform=manylinux2014_x86_64 ===
  Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (507 kB)
Would install safetensors-0.7.0
=== sentencepiece @ root/pypi, platform=manylinux2014_x86_64 ===
  Downloading sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
Would install sentencepiece-0.2.0
```

- safetensors:root/pypi 有 cp310 轮子(0.4.0–0.7.0),最新 0.7.0(经 `cp38-abi3` 稳定 ABI,cp310 可用)。**注意镜像里没有 0.8.0**(qwen-edit 里装的是 0.8.0,来源非此镜像)。
- sentencepiece:root/pypi 有 cp310 轮子,0.2.0。

### 附注:devpi 冷缓存漂移

同一条 simple 页会间歇返回全量页(~200KB+)或 5.4KB 小页(零轮子),疑似多后端/刷新中。判据以 pip 解析器为准,curl grep 要多取几次。已在 A3.1 用 pip 实测兜底,结论不受影响。

## A4 · 两件顺带确认

```bash
$E/bin/python -V
```
```text
Python 3.11.15
```

```bash
$E/bin/pip show diffusers | grep -i "location\|version\|editable"
```
```text
Version: 0.40.0.dev0
Location: /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages
```
(无 "Editable project location" 行 → **非 editable**,site-packages 常规安装;H800 复用源码走 `qwen/_vendor/diffusers_0.40.0.dev0/` P1 快照)

```bash
ls -d /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 && du -sh $_
```
```text
/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
54G	/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
```

---

# B · bf16 确认 + 加速比(infer_hub)

**未开始。** 前置:`qwen/P2_PREFLIGHT_RUN.md` §B 要求目标 HEAD(含本报告,即 B 前先把 A 的 commit push)被 infer_hub 认可。待 push 后按 §B 的 `infer_submit` 模板投出,再回来补本节(命令、job 状态、完整 stdout、两份 `results_shard0.json` 的 `meta`)。
