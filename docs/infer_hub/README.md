# infer_hub —— 多人共享推理机

一句话：把「每个实验各自起一个 watchdog 占一台机器空转」换成「所有人往一个共享队列投任务，
空闲的机器自己来拉活」。任务用 **github commit 链接 + 权重路径** 描述，推理机自己拉代码、
在沙盒里执行，结果写回权重目录旁。

> 本目录是给 UNO 项目整理的 **功能描述 + 使用手册**（精简版）。
> 系统本体与权威文档在 `/kaimm-distill/infer_hub/`（README.md、SKILL.md、docs/DESIGN.md）。
> 完整细节以那边为准，这里是挑 UNO 团队日常会用到的部分。

## 它是什么

- **多人共享推理机**：无服务端、无数据库、无常驻调度器。整个系统就是共享盘上一个目录，
  **任务的状态就是它所在的文件夹**。
- **任务 = git commit + 权重路径 + 项目名**：代码只能来自钉死的 commit（保证可复现），
  推理机在沙盒里跑，结果写到权重旁的 `infer_results/`。
- **开发机和推理机完全解耦**：提交脚本挂了、ssh 断了不影响已入队的任务；推理机猝死了，
  它手上的任务会被别的机器自动接走。
- 用户和推理机之间**只通过共享盘队列交互**，不需要也不能 SSH 登上推理机。

## 快速上手（UNO 团队视角）

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH   # 建议写进 ~/.bashrc

# 投一个任务（最小命令，默认 30min 超时 / 8 卡 / 默认输出到 <weights>/infer_results）
infer_submit --owner wuwenxuan --project m2v-aio \
  --commit-url https://github.com/<org>/<repo>/commit/<40位sha> \
  --weights /kaimm-distill/.../checkpoints/checkpoint-1000 \
  --uv-env /kaimm-distill/infer_hub/envs/aio_n26 \
  --label exp1_Iter1000

# 看状态
infer_status                  # 全部机器 + 队列
infer_status --owner wuwenxuan
```

`wuwenxuan` 已在成员名单里，`/kaimm-distill/infer_hub/envs/aio_n26` 是现成的推理环境
（venv，worker 自动激活），m2v-aio 队列当前有 UNO 同体系的 aio 任务在跑。

## 目录导航

| 文件 | 内容 |
|---|---|
| [`FUNCTION.md`](FUNCTION.md) | **功能描述**：架构、任务生命周期、关键机制（公平调度 / 超时 / 心跳 / 两阶段流水线 / 沙盒 / 集群路由） |
| [`USAGE.md`](USAGE.md) | **使用方式**：注册、`infer_submit` 全参数、watchdog 常驻、看状态、常见操作、硬规矩与坑 |

## 三个最重要的心智模型

1. **状态即目录**：`queues/<project>/` 下 `pending/`（排队）→ `claimed/<机器>/`（在跑）
   → `done/` / `failed/`（终态）。互斥靠同一挂载点内 `mv` 的原子性，不需要锁。
2. **派活是拉不是推**：没有中心调度器挑机器，是每台活着的 worker 自己从队列抢。
   机器挂了它就不再抢，等于自动退出竞争，无需任何注销。
3. **代码只来自 commit，参数只走环境变量**：一个 commit 只对应代码版本；TS/NFE 等推理
   设置走 `--cmd` 的环境变量前缀，不进 commit。改了代码才需要新 commit + push。
