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
  输出: datasets/distill_multiref/images | manifest: datasets/distill_multiref/manifest_shard0.json
... 401 GatedRepoError 拦截在 load_ae / ae.safetensors(已自修,见 §5) ...
[14:21:49] teacher 就绪,耗时 738.9s
100%|████| 25/25 [00:05<00:00, 4.99it/s]
[14:21:55] ❌ idx=0    backpack+cat            — ValueError: unknown file extension: .tmp
100%|████| 25/25 [00:04<00:00, 5.03it/s]
[14:22:00] ❌ idx=500  backpack+robot_toy      — ValueError: unknown file extension: .tmp
[14:22:05] ❌ idx=1000 cat+rc_car             — ValueError: unknown file extension: .tmp
[14:22:10] ❌ idx=1500 cat2+red_cartoon       — ValueError: unknown file extension: .tmp
[14:22:16] ❌ idx=2000 dog2+poop_emoji        — ValueError: unknown file extension: .tmp
[14:22:21] ❌ idx=2500 dog5+monster_toy       — ValueError: unknown file extension: .tmp
[14:22:26] ❌ idx=3000 dog6+teapot            — ValueError: unknown file extension: .tmp
[14:22:31] ❌ idx=3500 dog8+robot_toy         — ValueError: unknown file extension: .tmp
[14:22:37] ❌ idx=4000 pink_sunglasses+red_cartoon — ValueError: unknown file extension: .tmp
[14:22:42] ❌ idx=4500 rc_car+shiny_sneaker   — ValueError: unknown file extension: .tmp
[14:22:47] ❌ idx=5000 robot_toy+wolf_plushie — ValueError: unknown file extension: .tmp
[14:22:54] ❌ idx=5500 backpack+dog5+wolf_plushie — ValueError: unknown file extension: .tmp
[14:23:01] ❌ idx=6000 cat+dog3+poop_emoji    — ValueError: unknown file extension: .tmp
[14:23:08] ❌ idx=6500 cat2+red_cartoon+vase  — ValueError: unknown file extension: .tmp
[14:23:15] ❌ idx=7000 dog6+rc_car+shiny_sneaker — ValueError: unknown file extension: .tmp
[14:23:21] ❌ idx=7500 pink_sunglasses+rc_car+vase — ValueError: unknown file extension: .tmp
[14:23:21] === shard 0/500 完成 ===
  生成 0 | 跳过 0 | 失败 16 | 耗时 1.5m | 平均 92.50 s/img
```

## 3. 我已经试过什么

| # | 尝试 | 依据 | 结果 |
|---|------|------|------|
| 1 | HF token 写入 `~/.cache/huggingface/token`,whoami + curl gated 验证 | §R6 G3(环境配置) | OK,FLUX.1-dev gated 权限通过,token 已就位 |
| 2 | grep `distill/gen_data.py` 找 `.tmp` 来源 | 定位根因 | 仅 544 行(save 用)+ 598 行(manifest write 用),都是写不是读,`os.replace` rename 是对的 |
| 3 | `find datasets -name '*.tmp'` 全盘扫描 | 是否残留临时文件 | **0 个** |
| 4 | `ls datasets/dreambooth/dataset/backpack/` | ref image 是否完整 | 6 个 .jpg,无 .tmp |
| 5 | dry_run 枚举 task[0..2] 拼出 src 路径 | 验证 src 是 .jpg | `src=datasets/dreambooth/dataset/backpack/02.jpg exists=True` |
| 6 | 直接 `Image.open('datasets/dreambooth/dataset/backpack/00.jpg').convert('RGB')` | 验证 ref 能解码 | OK,`(767, 767)` |
| 7 | 在 PIL 12.3.0 源码搜 "unknown file extension" | 找抛出位置 | `Image.py:2668`,仅在 `os.path.splitext(path)[1]` 不在 EXTENSION 表时抛 |
| 8 | 模拟 .tmp 文件触发 PIL | 验证错形态 | 抛 `OSError: Truncated File Read`,**不是 ValueError** |
| 9 | 扫 `uno/flux/` 全部 `Image.open` / `.tmp` 调用 | teacher 内部是否偷偷读 .tmp | **0 处 Image.open,0 处 .tmp** |

## 4. 我的判断

### 4.1 关键观察(高置信度)

- PIL `Image.py:2668` 的 `ValueError: unknown file extension: {ext}` **只在且仅在**
  `os.path.splitext(path)[1].lower() == '.tmp'` 时抛。
- 我已穷尽 `distill/gen_data.py` + `uno/flux/` + `uno/dataset/`:没有任何代码路径
  会把字符串 `.tmp` 拼进传给 `Image.open` 的 path。
- 全 datasets 下 `find -name '*.tmp'` 返回 **0**。

### 4.2 矛盾(中等置信度)

唯一能让 `Image.open` 拿到 `.tmp` 后缀的途径是**真实存在一个扩展名是 .tmp 的文件**,
且某行 `Image.open(<那个文件>)` 被调用。我看不到这样的文件,也没看到这样的调用。

但脚本仍然在 16/16 的样本上稳定报这条错。意味着 **我缺一段诊断**:脚本 `gen_data.py:549`
的 `except Exception as exc` 只 print 了 `f"{type(exc).__name__}: {exc}"`,
**完整 traceback 被吞了**——我看不到 Image.open 真正被调用的行号和参数。

### 4.3 根因猜测(置信度:低)

按可能性排序:

1. **scripts 某处有未 grep 到的 Image.open 调用**(eg 第三方 lib、动态生成的代码)。
2. **`uno/flux/util.py:400` load_ae 下载 ae.safetensors 时残留了一个 .tmp 文件**,
   而某个 retry / cleanup 路径不小心让 `Image.open(<ae>.tmp)` 被触发——
   但 hf_hub_download 不会产生 .tmp 残留文件,且 Image.open 跟 hf_hub_download 没关联。
3. **teacher 加载时的某个副作用**:pipeline init 后,某个 watcher / callback 读了一个
   临时产物并触发 Image.open。但 uno/flux/ 里没有这类代码。
4. **路径拼接在某种边界 case 下产生 .tmp**:可能是 `os.path.join(out_dir, "../tmp/xxx")`?
   但 rel 都是 `../dreambooth/dataset/<s>/<v>.jpg`,我已 dry_run 验证过输出是 .jpg。

**置信度:低**——三种猜测都缺直接证据,而事实(0 个 .tmp + 0 个能产生 .tmp 的调用)又堵死了
我可见的所有根因路径。

## 5. 我做了什么改动

### 5.1 绿灯:HF token(已自行处理)

- 写入 `~/.cache/huggingface/token`,chmod 600(§R6 G3 环境变量/凭据配置)。
- 未改任何代码、未触碰 TRAIN/HELD-OUT、未触碰 task 列表、未触碰 teacher 配置。

### 5.2 红灯:未做任何代码改动

按手册 §2.0,**代码改动由你做**。我**没有**改 `gen_data.py`、没 patch except 打印 traceback、
没临时改 `_IMG_EXTS` 接受 .tmp(那会改实验数据)。一切停在 16/16 fail + 报告。

## 6. 我需要你判断什么

最小诊断:请**加一行让 `gen_data.py:549` 的 except 把完整 traceback 打印出来**,
比如:
```python
except Exception as exc:
    import traceback
    fails.append({"idx": task["idx"], "error": f"{type(exc).__name__}: {exc}",
                  "tb": traceback.format_exc()})
    print(f"[{datetime.now():%H:%M:%S}] ❌ idx={task['idx']} "
          f"{'+'.join(task['subjects'])} — {type(exc).__name__}: {exc}", flush=True)
    print(traceback.format_exc(), flush=True)
```
用户 push 后我 `git pull` 重跑一次,拿到 stack frame 就知道 .tmp 路径从哪个调用帧传入。
这是 100% 必要的最小诊断动作——没有 stack frame 我没法继续按 §R11 自己解释。

更长远的两个问题(等堆栈后再说,别先决策):

- **A. 修完后要不要先做 50 张标定吞吐再上全量?** 计划 §8 M1 标定阶段,先 50 张估时再 8000。
  当前 16 张全 fail 已经说明脚本流程没绿,标定意义不大,先排错。
- **B. PIL 12.3.0 是否要 pin 旧版本?** Pillow 12.3.0 是 2025 才发布的新主线,transformers 4.43.3
  (2024-07)可能没测过。当前错和 PIL 版本的关系**不能确定**,但作为附加信号记一笔。

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