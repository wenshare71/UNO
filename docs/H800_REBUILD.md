# 8×H800 机器环境重建指南

面向 `aiplatform-wlf3-ge90-19.idchb2az3.hb2.kwaidc.com`。
所有数字都是 `scripts/probe_env.sh` / `probe_net.sh` / `probe_pypi.sh` 在这台机器上的实测值，不是推测。

---

## 1. 机器画像

| 项目 | 实测值 | 影响 |
|---|---|---|
| GPU | 8 × H800，143771 MiB / 卡，sm_90 | 140 GB 显存，不需要 fp8 / CPU offload |
| 互联 | NV18 全互联（18 条 × 26.562 GB/s ≈ 478 GB/s）+ 12 张 mlx5 IB 网卡 | **必须开 P2P/IB**，见 §2 |
| 驱动 | 560.35.03（CUDA 12.6），**无 nvcc** | 只能用预编译轮子；不能 JIT 编译 CUDA 算子 |
| CPU / 内存 | 224 逻辑核 / 3023 GB | `uno_1m_total_labels.json` 810 MB 单文件 JSON 峰值 >8 GB，内存充裕 |
| OS / Python | Ubuntu 22.04.4，**仅 python3.10.12** | 旧机器是 3.12，cp312 轮子全部作废 |
| 包管理器 | **无 pip、无 conda**，`ensurepip` 缺失 | `python3 -m venv` 直接失败，见 §3 |
| 权限 | 非 root（`wuwenxuan03`, uid 1000），但 **sudo 免密可用** | 能 chown，但 apt 装不了东西（源不可达） |

### 存储

| 挂载点 | 类型 | 写 | 读 | 可用 |
|---|---|---|---|---|
| `/kaimm-distill/wuwenxuan`（仓库所在） | ceph | 67 MB/s | 136 MB/s | 215 T |
| `/code` | 本地 NVMe (xfs) | **3.1 GB/s** | **3.4 GB/s** | 2.6 T |
| `/tmp`, `/` | overlay（同一块 NVMe） | 3.1 GB/s | 2.7 GB/s | 2.6 T |
| `/dev/shm` | tmpfs | — | — | 1.5 T |

**本地 NVMe 比 ceph 快 25 倍。** 权重和数据放本地，checkpoint 放 ceph——权重丢了能重拷，checkpoint 丢了没了，而 `/code` 是本机盘、容器重建可能不保留。

### 网络

出网全部经 `http://oversea-squid1.jp.txyun:11080`（日本代理），九个目标全通但**全慢**：

| 源 | 实测吞吐 |
|---|---|
| `pypi.corp.kuaishou.com`（命中 `no_proxy`，**直连**） | **241.91 MB/s** |
| HuggingFace（经代理） | 0.66 MB/s |
| download.pytorch.org（经代理） | 0.36 MB/s |
| 阿里云 pytorch-wheels（经代理） | 0.20 MB/s |
| PyPI 官方（经代理） | 0.05 MB/s |
| `archive.ubuntu.com` / `security.ubuntu.com` | **不可达**（apt 无代理配置，直连解析到 Cloudflare IPv6 后 `Network is unreachable`） |

结论：**所有 Python 依赖走内网源，一条都不要走代理。** torch 栈约 2.5 GB，内网源约 10 秒，走代理要 1.9 小时。

---

## 2. 必须反转的旧机器（4090）假设

仓库里有几处是为 4090 打的补丁，在这台机器上是负优化甚至错误：

| 位置 | 4090 上的写法 | 这台机器 | 为什么 |
|---|---|---|---|
| `train.py:23-24` | `os.environ.setdefault("NCCL_P2P_DISABLE", "1")` | 外部 `export NCCL_P2P_DISABLE=0` 覆盖即可 | 用的是 `setdefault`，已导出的环境变量优先 |
| `scripts/train_ref_isolation.sh:8-9` | `export NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` | **必须改脚本**，硬 export 覆盖不掉 | NV18 全互联 + 12 张 IB 网卡，禁掉走 PCIe/共享内存 |
| `requirements.txt` | `--extra-index-url .../whl/cu124` | 不用，装 PyPI 默认的 `torch==2.4.0`（cu121） | cu121 官方构建的 arch 列表含 `sm_90`；且 cu124 索引在这里 0.36 MB/s |
| `scripts/setup_env.sh` | conda 优先、阿里云拉 torch、走 `INTERNAL_PROXY` | 用 `scripts/setup_env_h800.sh` | 无 conda、无 ensurepip、代理慢 672 倍 |

**尚未处理、留到 M3 训练时再决定的一项**：`train.py:229-233` 给 dit / t5 / clip 各挂了 DeepSpeed ZeRO-3。
`config/deepspeed/zero3_dit_config.json` 里 `offload_optimizer` 与 `offload_param` 都是 `"device": "none"`，
所以不需要 JIT 编译 `cpu_adam`（这台没 nvcc，否则会炸）。但 140 GB 显存下 ZeRO-3 的参数聚合纯属开销，
仓库里已有现成的 `config/deepspeed/zero2_config.json` 可换。
注意 `train.py:358` 的 `deepspeed.zero.Init(..., enabled=True)` 是 ZeRO-3 专属构造，换 stage 2 时要一并 gate 掉。
**M1 数据生成是纯推理，完全不碰这块**（全仓库只有 `train.py:31` 一处 `import deepspeed`），可以先不动。

---

## 3. 执行步骤

```bash
cd /kaimm-distill/wuwenxuan/UNO
git pull

# 一次搞定：建环境 + 修权限 + 权重搬本地盘（约 15 分钟，大头是拷 76 GB）
COPY_TO_LOCAL=1 bash scripts/setup_env_h800.sh
```

脚本做的七件事，以及每件事踩的坑：

1. **前置检查** —— 确认 python3.10 和内网源直连（若内网源握手 >2s，说明 `no_proxy` 没生效）。
2. **修 checkpoint 权限** —— `rsync -a` 从源机器 `/root` 拷过来时保留了 `root:root 0600`，
   实测 `log/` 下有 9 个文件当前用户读不了（就是 9 份 `dit_lora.safetensors`）。
   不修的话 resume 时在 `load_file()` 处抛 `PermissionError`。`hf_cache` 那 76 GB 实测 0 个不可读，无需处理。
3. **venv + pip 自举** —— `python3 -m venv --without-pip` 跳过缺失的 ensurepip；
   pip 的 wheel 本身是可执行 zipapp，`python pip-24.3.1.whl/pip install pip-24.3.1.whl` 即可自举，
   全程走内网直连，不碰 apt、不碰代理。
4. **装 PyTorch 2.4.0** —— 内网源的 `torch-2.4.0-cp310-cp310-manylinux1_x86_64.whl`，241.91 MB/s。
5. **先钉死 transformers==4.43.3 / diffusers==0.30.1** —— 顺序关键。
   `pyproject.toml` 写的是开区间 `transformers>=4.43.3` / `diffusers>=0.30.1`，
   先装 `-e ".[train]"` 会拉到 5.x / 0.39（冒烟测试踩过）。先装精确版本，
   后续 `>=` 约束已满足，pip 不会再升级。
6. **装 deepspeed 0.14.4** —— PyPI 上只有 sdist，`setup.py` 会 `import torch`，
   pip 默认的构建隔离会在干净环境里重新下载一份 torch，故用 `--no-build-isolation`。
7. **权重搬到 `/code/uno/hf_cache`** —— 加载 FLUX 54 GB：ceph 约 6.8 分钟，本地 NVMe 约 16 秒。

### 每次开工前

```bash
cd /kaimm-distill/wuwenxuan/UNO
source .venv-uno/bin/activate
export HF_HOME=/code/uno/hf_cache
export HF_HUB_OFFLINE=1                    # 权重已全在本地，禁掉联网探测
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`HF_HUB_OFFLINE=1` 很重要：不设的话 `from_pretrained` 每次都会去 HF 探测 etag，
而 HF 在这里只有 0.66 MB/s 且要过日本代理，每次加载白等十几秒。

---

## 4. 已有资产（已从旧机器拷入，实测确认）

| 资产 | 位置 | 状态 |
|---|---|---|
| HF 权重缓存 76 GB | `/kaimm-distill/wuwenxuan/hf_cache` | ✅ 完整，全部可读。含 FLUX.1-dev 54 G、xflux_text_encoders 8.9 G、clip-vit-large-patch14 6.4 G、bytedance-research/UNO 1.8 G |
| UNO-1M | `datasets/UNO-1M/` | ✅ 118 GB / 100000 图。`uno_1m_total_labels_convert.json` 50000 条，抽样 300 条零缺失，**ref 数量分布 `{1: 50000}`——100% 单 ref** |
| dreambooth | `datasets/dreambooth/dataset/` | ✅ 30 个 subject / 158 张图。**不要跑 `git submodule update --init`**，目录非空会报错；文件已在，直接用 |
| ref_isolation checkpoint | `log/ref_isolation/checkpoint-{1000..9000}` | ⚠️ 9 份，最新 **9000**（交付清单里写的 13000 有误）。源机器仍在训，目标 20000 |

关于那 9 份 checkpoint：每份 4.9 GB = `dit_lora.safetensors` 3.8 G + `optimizer.bin` 1.4 G。
**`optimizer.bin` 永远不会被读取**——`train.py:118-165` 的 `resume_from_checkpoint()` 只
`load_file()` 了 `dit_lora.safetensors`，返回值里根本没有 optimizer。
所以恢复训练会丢掉 Adam 动量，前几百步有 loss 抖动（这是原有行为，非新引入）。
那 12.6 GB 的 `optimizer.bin` 可以删。

---

## 5. 现在该做什么

`distill/DISTILL_PLAN.md` 的 M1（蒸馏数据生成）**不依赖 `log/ref_isolation/` 的任何 checkpoint**——
`DISTILL_PLAN.md:72-73` 写明 teacher 是官方 `bytedance-research/UNO` 的 dit_lora + full attention，
那份 1.8 GB 权重已在 HF 缓存里。ref_isolation 的 checkpoint 只有 M3（student 续训）才需要。

所以两台机器可以并行：

| | 旧机器（4090，已到 13000 步） | 这台（8×H800） |
|---|---|---|
| 现在 | 继续训到 20000 | **M1：生成 ~8000 条多参考蒸馏数据** |
| 之后 | 训完，拉最终 ckpt 过来 | M2 过滤 → M3 用最终 ckpt 续训 |

`DISTILL_PLAN.md:129` 估的「8000 张 ≈ 60 GPU 小时」是 4090 上的数。
H800 单卡快 3~4 倍且 140 GB 显存不必 offload，8 卡并行预计 **2~3 小时**。

`checkpoint-9000` 留着有用：M3 的训练链路（`train_distill.sh`、混合采样、`log/ref_distill` 落盘）
可以先用它跑 50 步冒烟验证，等旧机器的 20000 到位再正式开跑。
