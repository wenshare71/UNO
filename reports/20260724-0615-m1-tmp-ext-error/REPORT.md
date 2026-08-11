# M1 gen_data 100% 失败:ValueError: unknown file extension: .tmp

状态: 红灯 — 已阻塞,等待指示
阶段: M1(数据生成)
时间: 2026-07-24 UTC 06:15(本地 ~14:23)
commit: 5587490

## 1. 我在做什么

按计划 §8 跑 M1 第一块标定:`CUDA_VISIBLE_DEVICES=0 python distill/gen_data.py
--num_shards 500 --shard_idx 0`,预期 8000 条任务里这一 shard 拿 16 条,
每条 25 step FLUX teacher 出图,落 `datasets/distill_multiref/images/`。

## 2. 实际发生了什么

- 第一轮启动 → HF token 缺失,`ae.safetensors` 抛 GatedRepoError 401(已自修,见 §5)。
- 重启后 teacher 加载 OK:ae.safetensors(335M)+ flux1-dev(23.8G)+ dit_lora(1.91G)
  全部下载完成,耗时 738.9s 进 teacher 就绪状态。
- 第一个 shard 16/16 **全部失败**,每条同一错误:
  `ValueError: unknown file extension: .tmp`
- 失败率 100%,耗时 1.5m,平均 92.50 s/img(实际单步推理仅 5s,其余时间在 except 链路)。
- `datasets/distill_multiref/images/` **0 个产物**;`failures_shard0.json` 16 条全是这条错。

### 2.1 终端原始输出(orca term_3b43b012,最近 125 行)

```
[14:09:28] === M1 gen_data 启动 ===
  shard 0/500 | 全局任务 8000 → 本 shard 16 条 (断点续跑已完成 0,待跑 16)
  teacher: flux-dev offload=False ref_size=512 steps=25 guidance=4.0 512x512
... 401 GatedRepoError 拦截在 load_ae / ae.safetensors(已自修,见 §5) ...
[14:21:49] teacher 就绪,耗时 738.9s
[14:21:55] ❌ idx=0    backpack+cat            — ValueError: unknown file extension: .tmp
[14:22:00] ❌ idx=500  backpack+robot_toy      — ValueError: unknown file extension: .tmp
... (其余 14 条,idx 1000–7500 递增,均同一错误) ...
[14:23:21] === shard 0/500 完成 ===
  生成 0 | 跳过 0 | 失败 16 | 耗时 1.5m | 平均 92.50 s/img
```

## 3. 已试过什么

HF token 已修好(见 §5)。逐一排查过:`gen_data.py` 里 `.tmp` 只出现在两处
写文件(`os.replace` rename,不是读)、全盘 `find -name '*.tmp'` 为 0、
ref image 完整可 `Image.open` 解码、PIL 源码定位到 `Image.py:2668` 仅在
`os.path.splitext(path)[1]` 不在 EXTENSION 表时抛此错、`uno/flux/` 全仓
0 处 `Image.open` / `.tmp` 调用。

## 4. 判断

PIL 这条 `ValueError` **只在**路径后缀是 `.tmp` 时抛,但已穷尽
`gen_data.py` + `uno/flux/` + `uno/dataset/`,没有任何代码路径会把 `.tmp`
拼进 `Image.open` 的 path,全盘也没有 `.tmp` 文件。矛盾的根源:
`gen_data.py:549` 的 `except Exception as exc` 只打印了类型+消息,
**完整 traceback 被吞了**,看不到 `Image.open` 真正被调用的行号和参数。
四个根因猜测(未 grep 到的调用 / `load_ae` 残留 .tmp / teacher 加载副作用 /
路径拼接边界 case)**均缺直接证据,置信度低**。

## 5. 改动

绿灯:HF token 写入 `~/.cache/huggingface/token`,chmod 600(§R6 G3)。
红灯:未做任何代码改动——按手册 §2.0 代码改动由用户做,没 patch except、
没临时改 `_IMG_EXTS` 接受 .tmp(那会改实验数据)。

## 6. 需要判断什么

最小诊断:`gen_data.py:549` 的 `except Exception as exc` 只打印类型+消息,
完整 traceback 被吞了,请求加 `traceback.format_exc()` 打印后重跑一次,
拿到 stack frame 才能确定 `.tmp` 路径从哪个调用帧传入(§R11)。
附加信号:Pillow 12.3.0 是新主线,transformers 4.43.3 可能没测过,与本错误的
关系不能确定,记一笔。

## 7. 现场数据

- 已完成条数: **0 / 16**(失败率 100%)
- 失败条数: 16
- 跳过条数: 0
- 单 shard 耗时: 1.5m(纯 wall time,几乎全是 except 链路 + 5s/步推理)
- teacher 加载耗时: 738.9s(13 min,首次下载权重)
- 数据集状态:
  - `datasets/dreambooth/dataset/`: 32 个 subject 目录,ref 视角图齐
  - `datasets/distill_multiref/images/`: 0 个文件
  - 全 datasets 下 `.tmp` 文件: **0**
- 硬件: 8×H800(143771 MiB each),util 0%(teacher 加载后未在跑——M1 已结束)
- 关键版本: torch 2.4.0+cu121 / transformers 4.43.3 / huggingface_hub 0.36.2 /
  Pillow **12.3.0** / safetensors 0.8.0 / numpy 2.2.6 / python 3.10.12
- 代理: `https_proxy=http://oversea-squid1.jp.txyun:11080` 已设
- 报告目录: `/kaimm-distill/wuwenxuan/UNO/reports/20260724-0615-m1-tmp-ext-error/`
- 失败清单: `extra/failures_shard0.json`(16 条,均 `ValueError: unknown file extension: .tmp`)

---

附录 A:本报告**未推送**(手册 §3.5 修订:这台机器 push 走不通,POST 被代理吃)。
按 §3.5 步骤 2:本报告已**完整打印**到 stdout,用户转达给 Opus。本地仍 commit。