# M0 pre-flight 全部通过;但机器正被另一 8 卡训练任务占用,M1 开跑时机需用户拍板

状态: 黄灯 — 已自行处理(M0 部分),需复核;GPU 排期已阻塞,等待指示
阶段: M0
时间: 2026-07-23 09:13 UTC
commit: cfff8e8

## 1. 我在做什么

按 DISTILL_PLAN.md §3.0 执行 M0 pre-flight:dino_vits16 预取 + 8 卡/数据/权重自检,
为 M1 数据生成扫清依赖。

## 2. 实际发生了什么

### M0 检查项(全部通过)

| 检查项 | 结果 |
|---|---|
| 8 卡可见 | ✅ 8×H800 143771 MiB/卡 |
| dreambooth 30 subjects | ✅ 齐全 |
| venv | ✅ Python 3.10.12 + torch 2.4.0+cu121,CUDA 可用 |
| 官方 UNO dit_lora | ✅ /code/uno/hf_cache 可读,304 keys |
| dino_vits16 | ✅ 已预取并验证加载,21665664 参数 |

### dino 预取过程(计划 [假设] 的实测结果)

- **github.com 经代理:基本不可用**。zip 下载两次超时(60s/300s),但其中一次
  torch.hub 侥幸把仓库代码拉进了缓存(时通时不通,不可依赖);
- **raw.githubusercontent.com:可达**;
- **dl.fbaipublicfiles.com:可达但仅 ~0.13 MB/s**。85 MB 权重用 `wget -c` 断点
  续传约 10 分钟下完,校验尺寸 86710517 B 与官方一致,torch.hub 从缓存加载成功;
- **M2 注意**:`torch.hub.load` 必须带 `skip_validation=True`,否则每次加载都会
  请求 github API 校验 repo,在这台机器上会挂死。缓存位置
  `~/.cache/torch/hub/`(facebookresearch_dino_main + checkpoints/dino_deitsmall16_pretrain.pth)。

### ⚠️ 计划外发现:机器非空闲(这是黄灯的主体)

用户 `mayuezhou` 的 8 卡训练任务(root 属主,PID 138878-138885,
zimage_tdm_laion 蒸馏训练,已运行 4h+)正占用全部 8 卡:

- 每卡显存 ~97 GB / 143 GB → **剩余 ~46 GB/卡**
- GPU 利用率 96–97%,功耗 690W/700W(打满)

## 3. 我已经试过什么

| 尝试 | 依据 | 结果 |
|---|---|---|
| torch.hub.load 直连预取 | 计划 §3.0 原始命令 | 300s 超时(卡在 github zip) |
| wget 测 github zip / codeload | 定位卡点 | 均超时 → github 经代理不可用 |
| wget 测 fbaipublicfiles Range 1MB | 定位权重站可达性 | 成功 → 只有速度问题 |
| wget -c 后台断点续传权重 | G1/G6 绿灯 | 成功,已验证加载 |

## 4. 我的判断

根因猜测: 代理 oversea-squid1 对 github.com 的 CONNECT 不稳定/被限,与本机备忘
(代理吞 POST、github 直连被防火墙封)一致;权重站走静态 CDN 所以能通。
置信度: 高(对 dino 部分)。

GPU 占用的影响判断(置信度 中):
- **显存**:bf16 FLUX DiT ~22 GB + T5/CLIP/VAE + 激活,46 GB 余量**大概率够**
  (不 offload 单进程峰值预计 35–40 GB),但没实测,且对方占用可能波动;
- **算力**:SM 已被占满,我们的推理吞吐会严重缩水,M1 "2–3 h" 的 [假设] 在共享
  状态下不成立,可能变成 6–10 h 甚至更差(纯猜,须以 50 张标定实测为准)。

## 5. 我做了什么改动(黄灯必填)

- 未改任何仓库代码(工作区里 setup_env_h800.sh 的未提交改动是接手前就有的,
  内容是 pip 自举地址改从 simple 索引解析——保持原样,未动);
- 系统侧:下载了 dino 权重到 ~/.cache/torch/hub(缓存,不进 git)。

## 6. 我需要你判断什么

GPU 排期,三条路我倾向 A:
- **A. 先并行做不占卡的事,等对方任务结束再跑 M1 全量**:我现在就写 gen_data.py,
  跑 --dry_run 核对枚举(纯 CPU),再用 1 张卡试跑 50 张标定(46 GB 应该放得下,
  即使慢也能验证正确性);对方结束后再 8 卡全量。风险:不知道对方还要跑多久。
- B. 立即在共享状态下 8 卡慢跑全量:吞吐差、还可能互相 OOM,不推荐。
- C. 协调对方让卡:这是你和 mayuezhou 之间的事,我无权也无渠道。

我按 A 先做(写码 + dry_run 不占卡,50 张标定占 1 张卡且随时可停),
**全量开跑前等你确认**。

## 7. 现场数据

- 8 卡各占 ~96.5–97.0 GB(对方),利用率 96–97%;
- dino 权重 86710517 B,下载耗时 ~10 min(0.13 MB/s);
- 蒸馏数据 0 条(尚未开始生成);
- env 快照见同目录 env.txt。
