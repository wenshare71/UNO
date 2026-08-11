# Q1 环境底座因 glibc 不兼容由 aio_n26 切换到 kling-mini

状态: 黄灯 — 已自行处理,需复核
阶段: Q1(Qwen 裸基线上机)
时间: 2026-08-11 09:51 UTC
commit: 74fd933

## 1. 我在做什么

按 `distill/Q1_QWEN_BASELINE_RUN.md` 执行:在 4090(`aiplatform-bjy-ge47-391`)上为
Qwen-Image-Edit-2511 裸基线做 A1 下权重 / A2 建环境 / A3 验脚本。执行到 A2 门禁
(import diffusers)时,指定底座 `aio_n26` 的 torch 在这台 4090 上跑不起来。

## 2. 实际发生了什么

- `aio_n26`(执行单 §3 指定底座,py3.12, torch 2.7.0a0+cu 的 NVIDIA 容器 build)在
  4090 上 `import torch` 失败:
  `ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.33' not found
  (required by .../envs/qwen-edit/lib/libucs.so.0)`。
  4090 是 Ubuntu 20.04(glibc 2.31);`libucs.so.0`(UCX,torch 分布式/NCCL 依赖)要求
  GLIBC_2.33,系统 glibc 无法替换。
- 按执行单 §3 兜底链测试 `kling-mini`(py3.11, torch 2.5.1+cu124):
  `torch.cuda.is_available()=True`, **8 卡全识别**,并成功 import `QwenImageEditPlusPipeline`。

## 3. 我已经试过什么

| 尝试 | 依据 | 结果 |
|---|---|---|
| aio_n26 import torch(补 LD_LIBRARY_PATH: env/lib + site-packages/nvidia/*/lib + torch/lib) | 执行单 §3 首选底座 | ❌ libucs.so.0 需 GLIBC_2.33 |
| aio_n26 前序:libcusparseLt.so.0 / libucs.so.0 缺失 | 同为 LD_LIBRARY_PATH 问题,已定位并补 | 补了仍卡在 glibc,判定为机器系统层不兼容 |
| kling-mini(py3.11, torch 2.5.1) | 执行单 §3 兜底链 | ✅ torch 2.5.1+cu124, 8 卡, diffusers main import OK |
| v4moe | 兜底链第三位 | 未单独试:torch 同为 2.7.0a0+cu build,预期与 aio_n26 同问题 |

## 4. 我的判断

根因: `aio_n26` / `v4moe` 的 torch 是为 infer_hub H 机(glibc ≥ 2.34)构建的容器版
PyTorch,其 UCX 库要求 GLIBC_2.33;4090 是 Ubuntu 20.04(glibc 2.31),系统 glibc 不可
降级替换。`kling-mini` 的 torch 2.5.1 无此问题,在 4090 上验证可用。
置信度: 高

不确定点:
- kling-mini 在 H 机(执行单 C 阶段)上跑 torch 2.5.1 的最终推理未实测——它是
  infer_hub 共享 env,理论上在 H 机可正常加载(C 阶段 job 会直接验证);
- torch 2.5.1 与执行单 §6.1 的口径(diffusers main 默认参数)无冲突,但若有
  数值/行为差异,只影响基线数值本身,不影响"零点"口径的成立。

## 5. 我做了什么改动

- **env**: `qwen-edit` 底座由 aio_n26 改为 kling-mini(tar 并行拷贝,49242/49242 文件对齐);
  原 aio_n26 副本 `mv` 到 `qwen-edit-aio` 保留(未删,随时可切回)。
- **env 路径修正**: 拷贝后 `bin/*` console script 的 shebang 仍指向原 kling-mini,
  经 `infer_hub/envs/fix_env_paths.sh` 重写到 `qwen-edit`(83 个文件),`python -m pip` 验证落位正确。
- **pip 依赖**(内网 index + git 国内代理): transformers 5.14.1 / accelerate 1.14.0 /
  safetensors 0.8.0 / peft 0.20.0(diffusers main 要求 >=0.17.0)/ diffusers 0.40.0.dev0(main)。
- **权重**: 下到 `/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511`,
  35 文件 / 57.72 GB,与 hf-mirror manifest 逐字节一致。
- **未改动任何 `.py` / `.sh`**(R0 遵守)。

## 6. 我需要你判断什么

C 阶段(infer_hub, H 机)用哪个 env:

- **A. qwen-edit(kling-mini, torch 2.5.1)**——单 env,4090 冒烟与 H 正式跑噪声同源、
  最简单;代价是 H 侧 torch 2.5.1 而非执行单原定的 2.7.0a0。**我倾向 A**,已按 A 继续。
- **B. qwen-edit-aio(aio_n26, torch 2.7.0a0)**——保持执行单原计划,需在 aio env
  再装一遍 diffusers main + peft;冒烟与正式跑跨 torch 版本。

用户回复"将现在情况上报提交",按 A 继续推进;要切 B 随时可切(env 副本都在)。

## 7. 现场数据

- 权重: 35 文件 / 57.72 GB,manifest 全对齐;含 20B transformer(5 shard)+ Qwen2.5-VL
  text_encoder(4 shard)+ VAE + 全量 config/tokenizer。
- A2 门禁: `diffusers 0.40.0.dev0` / `torch 2.5.1+cu124` / `transformers 5.14.1` / `OK`。
- A3 dry_run: 40/40 通过,40 个 task_id 与执行单 §4 逐条一致,拼图 5 part,
  `results.json: /tmp/q1dry/results.json`。
- 4090: 8× NVIDIA RTX 4090(24GB),Ubuntu 20.04.6, glibc 2.31, 内核 4.18.0-2.4.3.3.kwai.x86_64。
- 下载中的插曲(已修): 代理 10.66.37.111 全死→换 22.211(慢)→P1 完事后 29.113 接手;
  hf-mirror 对 Qwen 仓库走 Xet bridge(302 签名单 URL,HTTP/2),大文件 OK;小配置文件
  直出 200 无 X-Linked-Size → 下载器补 Content-Length/直连逻辑后重下,全部对齐。

## 8. B 冒烟更新(2026-08-11 追加)

- 4090 单例冒烟(`--limit 1 --offload`,qwen-edit/kling-mini 底座)在**生成阶段 OOM**:
  `OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB.
   GPU 0 23.65 GiB,仅 84 MiB 空闲,进程占用 23.56 GiB`。
- **模型加载成功(123.9s)**,权重 + diffusers API + pipeline 构造全部验通;OOM 只发生在
  生成调用(`pipe(...)` 40 步 × 1024²)。
- **根因**: Qwen 20B MMDiT transformer bf16 ≈ 40GB,`enable_model_cpu_offload()` 是
  **按组件整体搬 GPU** 的粗粒度 offload,单组件 40GB > 24GB,必 OOM;换
  `enable_sequential_cpu_offload` / `device_map="auto"` 需要非仓库的一次性命令(R0 冲突)。
- **处置**: 未降分辨率/steps(执行单 §6.6 红档,降了就不是真实水平);4090 无法完成生成。
- **建议**: 跳过 4090 生成,直接投 C —— H 机单卡 80GB 装得下 40GB,正式跑本身就是全量
  验证;4090 已验的加载环节消除了 C 的主要通路风险。备选:在有 80GB 显存的机器上
  先手动冒烟一张。
