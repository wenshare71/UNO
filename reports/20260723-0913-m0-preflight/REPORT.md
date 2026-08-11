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

## 3. 已试过什么(根因)

torch.hub 直连预取 300s 超时;定位到 github zip/codeload 经代理均超时,
**代理 oversea-squid1 对 github.com 的 CONNECT 不稳定/被限**;
`fbaipublicfiles` 走静态 CDN 可达,`wget -c` 断点续传 ~10 min 下完并验证加载成功。

GPU 占用的影响(置信度 中):46 GB 显存余量大概率够单进程推理,但算力被占满,
M1 "2–3 h" 的 [假设] 在共享状态下可能变成 6–10 h(须以 50 张标定实测为准)。

## 4. 处理

未改任何仓库代码。系统侧下载 dino 权重到 `~/.cache/torch/hub`(缓存,不进 git)。
先做不占卡的事(写 `gen_data.py`、`--dry_run` 核对、50 张标定占 1 卡),
**全量 8 卡开跑前等用户确认**。

## 5. 现场数据

- 8 卡各占 ~96.5–97.0 GB(对方),利用率 96–97%;
- dino 权重 86710517 B,下载耗时 ~10 min(0.13 MB/s);
- 蒸馏数据 0 条(尚未开始生成);
- env 快照见同目录 env.txt。
