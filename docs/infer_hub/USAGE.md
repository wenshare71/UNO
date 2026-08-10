# infer_hub 使用方式

## 0. 前置

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH   # 建议写进 ~/.bashrc
```

- **owner 必填且须在名单里**（`wuwenxuan` 已在）。它是公平调度与归属的主键，不会自动填
  whoami。不在名单：到控制台登录页填用户名自助注册，或管理员 `infer_admin --add <user>`。
- **weights / output / env 路径必须在 `/kaimm-distill/` 下**。`/home/<user>` 是各人独立的
  mmu_ssd 挂载，推理机上看不到，提交端直接拒绝。
- **公共环境** `/kaimm-distill/infer_hub/envs/`（`kling-mini`、`v4moe`、`aio_n26`）填路径
  直接用。公共环境是只读共识，别往里 pip install；要改就 `cp -a` 一份到自己目录。

## 1. 投一个任务

只有 git 提交式一种：commit 链接 + 权重路径。

```bash
infer_submit --owner wuwenxuan --project m2v-aio \
  --commit-url https://github.com/<org>/<repo>/commit/<40位sha> \
  --weights /kaimm-distill/.../checkpoints/checkpoint-1000 \
  --uv-env /kaimm-distill/infer_hub/envs/aio_n26 \
  --label exp1_Iter1000
```

等价写法（`--repo` + `--commit`，commit 必须已 push）：

```bash
infer_submit --owner wuwenxuan --project m2v-aio \
  --repo https://git.corp.kuaishou.com/.../xxx \
  --commit 6b7c20e1...（40 位，必须已 push） \
  --weights .../checkpoints/checkpoint-400 \
  --output-dir .../outputs/ts999_nor/ckpt400_nfe5 \
  --label exp1_Iter400 --gpus 1 \
  --uv-env /kaimm-distill/infer_hub/envs/aio_n26 \
  --cmd 'TS=999 NO_R=1 NFE=5 CKPT=$INFER_WEIGHTS_DIR OUTPUT=$INFER_OUTPUT_DIR bash scripts/infer_xxx.sh'
```

### 参数速查

| 参数 | 含义 |
|---|---|
| `--owner` | 任务归属账号（必填，须在名单里） |
| `--project` | 项目/模型名，只用于队列归类与网页分组展示，**不决定环境** |
| `--cluster` | 可选，硬绑定卡型：`h`/`default`（H 卡）或 `5kpro`；不填走项目默认去向 |
| `--commit-url` / `--repo`+`--commit` | 代码来源，二选一；commit 40 位且已 push |
| `--weights` | 权重路径（必填，须在共享盘） |
| `--output-dir` | 输出目录，默认 `<weights>/infer_results`（worker 预创建） |
| `--uv-env`（`--env`/`--conda-env` 旧名） | 环境路径（**必填**，uv/venv 或 conda 都认） |
| `--cmd` | 覆盖默认 `bash infer_entry.sh`，在 checkout 出的代码根执行 |
| `--label` | 任务名，约定 `<实验名>_Iter<步数>` |
| `--gpus` | 卡数，默认 8，上限 8；纯 CPU 写 0 |
| `--timeout` | 推理超时分钟，默认 30（看 GPU 利用率决定杀/延长，最多 3 倍） |
| `--prep-cmd` / `--prep-marker` / `--prep-timeout` | 两阶段流水线（切权重与推理并行，见 §2） |
| `--dry-run` | 只打印将要投的 job json，不真写入 |
| `--force` | 同 job_id 已存在也强制重投（job_id 加时间戳后缀） |

### worker 注入的三个环境变量

| 变量 | 含义 | 权限 |
|---|---|---|
| `INFER_CODE_DIR` | 本次 checkout 的代码根（即 cwd，推理机本地缓存盘） | 读写 |
| `INFER_WEIGHTS_DIR` | `--weights` 给的权重路径 | 约定只读 |
| `INFER_OUTPUT_DIR` | 输出目录 | 读写 |

## 2. 两阶段流水线（切权重与推理并行）

推理入口若是「先 CPU 切权重/转格式，再上卡采样」，强烈建议拆成两阶段：
切分阶段不占卡，与同机其他任务的推理并行，推理超时从发卡起算。

```bash
infer_submit ...（其余参数同上）... \
  --prep-cmd 'CKPT=$INFER_WEIGHTS_DIR bash scripts/split_weights.sh && touch $INFER_WEIGHTS_DIR/infer_prep.done' \
  --prep-marker .../checkpoints/checkpoint-400/infer_prep.done \
  --prep-timeout 45
```

- marker 存在就跳过切分直接领卡（同一份权重多个任务共用一次切分）；
- 切完等卡超 30 分钟放生回队，不浪费切分成果；
- 失败归因分开（`prep_failed` / `prep_timeout`）；
- **别把切分藏进 `--cmd`**：发卡 15 分钟后 GPU 利用率过低会被判 `infer_gpu_idle` 提前杀。

判断方法：看 `ENTRY` 入口脚本里是否先跑 convert/split/save_pretrained 再采样。
拆分需要改 repo 代码时，改完必须 commit + push 并更新 `--commit`（推理机只认 push 过的 commit）。

## 3. watchdog 常驻（盯 ckpt 产出自动投递）

**推荐用法：开训练时顺手把 watchdog 一起拉起，让它常驻。** checkpoint 落盘即自动进推理
队列，训练全程不用再碰 git。只有真改了推理代码才需要 push 新 commit 并更新配置。

```bash
# 抄模板到自己目录，只改头部配置块（不要改模板本体）
cp /kaimm-distill/infer_hub/bin/watchdog_submit.sh \
   /kaimm-distill/<你>/watchdog_<项目>.sh
vim /kaimm-distill/<你>/watchdog_<项目>.sh
# 先 dry-run 验证一条命令，再常驻
tmux new -d -s watchdog "bash /kaimm-distill/<你>/watchdog_<项目>.sh"
```

配置块变量（`======== 配置块 ========` 到 `SUB_DIR` 之间的区域）：

| 变量 | 改成什么 |
|---|---|
| `OWNER` | 你的账号 |
| `PROJECT` | 项目名（只用于分组展示） |
| `CLUSTER` | 可选，硬绑定卡型：`h`/`default` 或 `5kpro` |
| `REPO` | 仓库 https 地址（不要 `git@` ssh 地址） |
| `COMMIT` | 手写完整 40 位 commit id，**必须已 push** |
| `EXP_ROOT` / `OUT_ROOT` | checkpoint 扫描根 / 推理输出根（须在共享盘） |
| `EXPS` | 实验清单：实验名 → `"环境变量前缀\|输出子目录"` |
| `ENTRY` | 在 checkout 出的代码根执行的入口命令 |
| `UV_ENV` | **必填**，环境路径（uv/venv 或 conda） |
| `GPUS` / `TIMEOUT_MIN` / `INTERVAL` | 卡数 / 超时分钟 / 扫描间隔秒 |
| `PREP_ENTRY` / `PREP_MARKER_NAME` / `PREP_TIMEOUT_MIN` | 可选，两阶段（留空=单阶段） |

## 4. 看状态

```bash
infer_status                    # 全局：推理机 + 各队列 + 最近结果
infer_status --owner wuwenxuan  # 只看自己的
infer_status --env-tag m2v-aio  # 只看某个队列
tail -f /kaimm-distill/infer_hub/queues/<project>/logs/<job_id>.log   # 看任务输出
```

控制台（办公网）：<https://inferhub.test.gifshow.com/infer/login>
- 成员：用户名免密（首登自动注册），看全部机器与队列（有项目筛选）+ 自己的任务总览；
- 管理员：用户名+密码，另可看报警、操作流水；
- 任务树按 `<实验名>` 分组，日志有实时链接。

## 5. 常见操作

| 想做什么 | 怎么做 |
|---|---|
| 取消排队中的任务 | 控制台「取消」（自己/管理员）；或 `rm .../queues/<project>/pending/<job_id>.json` |
| 杀掉在跑的任务 | 控制台「杀掉」（需 worker ≥ v2.2.2 响应杀信号） |
| 误取消/误杀掉恢复 | 管理员页「回收站」/ 普通用户首页「我的回收站」重新拉起 |
| 重新排队失联任务 | `infer_janitor`（开发机提交脚本可顺手每轮调一次） |
| 重跑某任务 | 同参数加 `--force` |

## 6. 硬规矩（违反会被提交端拒绝或运行时翻车）

1. **`--cmd` / `--prep-cmd` 用单引号**：`CKPT=$INFER_WEIGHTS_DIR OUTPUT=$INFER_OUTPUT_DIR`
   等注入变量要到推理机上才展开。worker 注入 `INFER_CODE_DIR` / `INFER_WEIGHTS_DIR` /
   `INFER_OUTPUT_DIR`。
2. **GPU 千万不要写死卡号**：用 `GPU=${CUDA_VISIBLE_DEVICES:-0}`。写死 0 会把一台机器上
   所有并行任务压到同一张卡 OOM。
3. **`--cmd` 里不许出现未声明的共享盘绝对路径**：`cd` 到个人代码目录、`PYTHONPATH` 指向
   个人仓库都会被提交端拒绝——代码必须全部来自 commit，保证可复现。
4. **参数不进 commit**：同一 commit 配不同参数就是不同任务；只有改了代码才需要新 commit
   + push + 更新 watchdog 里的 `COMMIT`。
5. **label 按 `<实验名>_Iter<步数>`** 命名，控制台任务树靠它分组。
6. **幂等**：`job_id = <owner>__<label>__<sha12>`，同任务重复投跳过，换 commit 是新任务。
7. **命令会自动加 `set -e -o pipefail`**：`... 2>&1 | tee log` 的失败不会被吞掉。

## 7. 常见坑

- **commit 没 push / 写了分支名**：推理机报 `commit_not_found`。分支名提交时会被解析成当时
  的 sha 入队，但 watchdog 里必须手写 40 位 sha。
- **超时被杀 vs 延长**：到点看 GPU 平均利用率，<30% 杀、在算则延长（最多 3 倍）；
  `--gpus 0` 到点即杀，超时要给足。
- **两阶段任务从「运行中」回到「等待+已切分」**：切完等卡超 30 分钟放生回队，不是故障。
- **两阶段任务 `infer_gpu_idle` 提前杀**：检查是不是把切权重塞进了 `--cmd`（应放 `--prep-cmd`）。
- **公共环境别装包**：要改包 `cp -a` 一份到自己目录再改。
