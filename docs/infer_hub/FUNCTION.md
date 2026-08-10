# infer_hub 功能描述

## 1. 架构总览

- **无中心角色**：没有服务端、数据库、常驻调度器。整个系统 = 共享盘上的一个目录
  `/kaimm-distill/infer_hub`。
- **任务状态 = 文件所在的文件夹**：

  ```
  queues/<project>/
      tmp/          写入中（对 worker 不可见，原子写保护）
      pending/      排队中
      claimed/<机器名>/  某台机器正在跑
      done/         成功
      failed/       失败（含超时 / 准备失败 / killed / cancelled）
      logs/         每个任务的完整输出
  ```

- **互斥靠 `mv` 原子性**：同一挂载点内 `os.rename` 在 CephFS 上是原子的，两台机器抢
  同一个任务一定只有一台成功，不需要锁文件。
- **机器零登记**：机器上线跑 setup 脚本即自动上报，下线/失联静默 24h 后状态自动消失；
  派活不读机器名单，机器停了自动退出竞争。
- **两套交互通道**：开发机（投任务 / 看状态 / 看日志，纯 CPU）与推理机（抢活 / 执行），
  只通过共享盘队列交互，无 SSH。

## 2. 任务生命周期

1. 开发机 `infer_submit` 提交 job json 到 `pending/`（提交时做**轻量** git 校验，见 §4）。
2. 某台 worker 按公平规则**抢单**：`mv` 到 `claimed/<本机>/`，成功即抢到。
3. 推理机准备代码：本地 bare mirror 增量 fetch + 每 commit 独立 worktree 物化（秒级缓存命中）。
4. 沙盒里执行命令，后台线程每分钟写心跳、每半分钟采 GPU 利用率。
5. 按退出码把 job 挪进 `done/` 或 `failed/`，写回耗时 / 日志路径；
   git 任务另在输出目录留 `infer_result_<job_id>.json`。

## 3. 关键机制

### 3.1 公平调度（不是先来先服务）

取活顺序（`order_pending()`，控制台等待队列显示的就是这个顺序）：

1. **管理员置顶**的任务整段排最前；
2. **已切分**的两阶段任务（marker 就绪只差上卡）次之；
3. 其余按 **owner 交错**：谁当前在跑的任务少，谁的下一个任务先上；平手才看投递时间，
   同一个人内部严格 FIFO。

> 你一次投 20 个 ckpt 不会堵死别人——别人投 1 个，那 1 个会插到你的第 2 位。

### 3.2 装箱路由（多机之间）

单卡等小任务自动集中到「空闲卡最少但装得下」的机器，整机空闲留给 8 卡任务，
不会 5 台机器各占一张卡。同机内：队头任务卡不够时后面的小任务不许插队。

### 3.3 超时规则

- git 准备阶段（fetch + checkout）有独立 10 分钟超时，**不占推理超时额度**。
- 推理默认 30 分钟超时，但**到点不无条件杀**：看这段时间平均 GPU 利用率——
  - 低于 30%：判卡死，kill 整个进程组（`mpirun` 的 8 个 rank 必须一起杀，否则残留
    进程占显存，下个任务 OOM）；
  - 还在正常算：再放一个周期，最多放到 3 倍。
- 两个例外走「到点即杀」：`--gpus 0` 的纯 CPU 任务；`nvidia-smi` 取不到数。
- 结果里记 `avg_gpu_util` 和延长次数，能看出哪些任务估时不准。

### 3.4 心跳与「主动撒手」（最坏失败防护）

- 心跳线程和任务各跑各的。共享盘卡住时心跳写不进去但任务还在跑，别的机器会判本机失联、
  把任务捞走重跑——同一个 eval 在两台机器上各占 8 卡并发跑、往同一目录写，是这套设计
  里**最坏的失败**。
- 所以：连续写不进心跳超过半个超时窗口（默认 7.5 分钟，早于别人判失联的 15 分钟）时，
  worker 会 kill 掉自己的任务并留在 `claimed/`，等盘恢复后正常回收重派。**续不上租约的
  一方必须停手**，两边同时跑成为不可能。

### 3.5 两阶段流水线（v3）：切权重与推理并行

很多推理入口实际是两步：先在 CPU 上切权重/转格式（不用卡），再上卡采样。声明
`--prep-cmd` 后任务走两阶段：

- **切分阶段不占卡**（worker 强制 `CUDA_VISIBLE_DEVICES=` 空），单机同时只切一个，
  与同机其他任务的推理并行；**推理超时从发卡时刻起算**，排队和切分不占额度。
- **marker 判已切分**：`--prep-marker` 指向切分产物标志路径，存在就跳过切分直接领卡；
  同一份权重多个任务共用一次切分（worker 有锁互斥）。
- 切完写 `.ready` 举手等发卡；主 worker 写入 `.gpus`（物理卡号）后 subworker 开始推理。
- 切完等卡超 30 分钟**放生回队**（带已切分标记退回排队，不浪费切分成果）。
- 失败归因分开：切分失败 `prep_failed`、切分超时 `prep_timeout`、推理失败照旧。
- 发卡 15 分钟后 GPU 平均利用率过低（默认 <3%）判 `infer_gpu_idle` 提前杀——防止把
  切分藏进推理命令占着卡跑 CPU。

### 3.6 沙盒强度（第一档）

**防误操作互相污染，不防恶意**（组内互信）。独立进程组 + cgroup 限内存 + 任务私有
`HOME` + 白名单环境变量注入 + 提交时危险命令静态拦截（`rm -rf /`、`sudo` 等）。
**不做容器级隔离**。

### 3.7 集群 / 卡型路由

- 机器声明所属物理集群（`default`=H 卡主集群，或 `5kpro`）。
- 任务可用 `--cluster h|5kpro` **硬绑定**卡型，永不回落。
- 不填则按项目默认：`v4` / `moe_v4` / `m2v-aio` → H 卡；其余优先 5kpro，满员超时
  （默认 5 分钟）回落 H 卡，不会饿死。

### 3.8 git 代码缓存

- 每仓库一个 bare mirror 增量 fetch，每 commit 一个独立 worktree（不存在「一个目录来回
  切分支」）；缓存命中秒级。
- 缓存放推理机本地盘 `/var/infer_cache/`（`mirrors/` `worktrees/` `homes/`），LRU 淘汰，
  机器回收丢了就丢，第一个任务重新 clone 即可。
- 带子模块的仓库自动 `submodule update --init --recursive`，并把 `git@host:` 地址重写成
  https（推理机无 ssh 私钥，凭据只走 https + token）。
- 机房直连 github 不通，git 走 `git_proxy` 代理，worker 跑 git 时**强制注入**，不依赖
  shell 环境恰好配好。

### 3.9 报警

`alerts/<日期>/`：git 权限、空转、堆积等，管理员页渲染与确认。同对象同规则一天只写一条
（文件存在性判断），写失败静默（报警是锦上添花，不能把 worker 搞挂）。

## 4. 提交时的轻量校验

`infer_submit` 提交时做 KB 级校验，不拖慢网速；重量级 fetch 只发生在推理机上：

1. 链接格式解析（github/gitlab commit 链接）；分支/短 sha 用 `ls-remote` 解析成完整 sha；
2. `git ls-remote` 判仓库可达与凭据有效；
3. 对 commit 网页发一次 HTTP 探测判 sha 是否存在（404 只有仓库主页可匿名访问时才判 missing，
   私有仓库探测不可用则放行，由推理机 `commit_not_found` 兜底）。

## 5. 目录布局（简化）

```
/kaimm-distill/infer_hub/
    config.json           全局配置（INFER_HUB_<大写键名> 环境变量可临时覆盖）
    lib/                  CLI：infer_submit / infer_status / infer_admin /
                          infer_janitor / hubcore.py（公共库，纯标准库）
    worker/               推理机端 worker.py + gitprep.py（worker.sh 复制到 /tmp 再跑）
    web/                  控制台（login / home / admin），寄生在可视化服务 /infer 路径
    bin/setup_infer_machine.sh  推理机一键启动
    bin/watchdog_submit.sh      开发机 watchdog 模板（抄走改配置块）
    queues/<project>/     任务队列，状态即目录
    workers/<机器名>.json worker 上报状态；.drain=停机开关；.format=接单格式开关(v2/v3)
    registry/users/       成员名单（登录页自助注册 / infer_admin 维护）
    registry/audit.log    操作流水
    alerts/<日期>/         报警
    envs/                 公共环境（kling-mini / v4moe / aio_n26 等）
    docs/DESIGN.md        权威设计文档
```

## 6. 当前环境快照（2026-08-10）

- **机器**：5 台在线，均 v2.4.0（8 卡）——`aiplatform-wlf3-ge90-10`（format=v3，蓝绿灰度
  机）、`ge90-26`、`ge90-70`（正跑 liucongyi 的 aio 任务）、`klingai-wlf2-ge124-node194`；
  `ge90-82` 处于 drain 停机中。
- **队列**（`queues/` 下有目录的）：`default` `kling-mini` `m2v-aio` `moe_v4`
  `moe_v4_x2va` `playground` `sandbox_fmt` `v4` `v4lite-kd`。
- **UNO 相关**：`m2v-aio` 队列承接 aio 体系任务；`/kaimm-distill/infer_hub/envs/aio_n26`
  是 venv 推理环境（worker 自动激活）；`wuwenxuan` 在成员名单中，可直接投任务。

> 快照信息会随运行漂移，以 `infer_status` 实时输出为准。
