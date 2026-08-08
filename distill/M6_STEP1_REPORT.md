# M6 步骤 1 完成报告 — 数据补齐 + ZeRO-2 标定

> 对应 `distill/M6_STEP1_RUN.md` 的 P1。**2026-08-08,闸门 A / 闸门 B 均已通过。**
> 本报告取代旧 `M6_STEP1_CALIB_REPORT.md`(那份只记录了闸门 A 的失败,
> 没有闸门 B 的数据;那份的根因分析与修复决策已并入 §2,不丢内容)。

## 结论先行

| 闸门 | 判据 | 结果 |
|---|---|---|
| **A** — ZeRO-2 标定 | 两腿 checkpoint 都是 `304 张量,空/全零 0`,训练无 OOM | ✅ **通过** |
| **B** — 训练集 | `stage1_official_score4.json` 落盘、不带 `_partial`、覆盖 ≥95% | ✅ **通过**(404258/404259 = **100.0%**) |

两个闸门都过,可以进 P2。

---

## 1. 这一步要交付什么(回顾)

P1 交付**两件**东西,缺一件都不能进 P2(那是 7 天 8 卡):

1. **官方口径的 stage-1 训练集** —— `score_final ≥ 4` 的全库过滤。开始前磁盘只有
   10/102 个分片,可用满分样本 16,966 条 = 4.2%;现在 **404,258 条 = 100.0%**。
2. **ZeRO-2 能在这台机器上正常存 checkpoint 的证据** —— DiT 刚从 ZeRO-3 切回官方
   ZeRO-2(commit `3172b67`),ZeRO-2 下 `accelerator.get_state_dict(dit)` 走
   `clone_tensors_for_torch_save`,8 rank 同时来一次 25.8 GB 整模型量级的内存峰值,
   "8 × 25.8 GB ≈ 206 GB host RAM"是全程唯一的真风险,必须实证。

---

## 2. 闸门 A:ZeRO-2 标定

### 2.1 第一次跑:失败(2026-08-07,已修)

第一版标定 2a(iso 腿)在第 50 步第一次存 checkpoint 时**全部 8 rank 崩溃**:

```
train.py:468  unwrapped_model_state = accelerator.get_state_dict(dit)
accelerate/accelerator.py:3365
ValueError: Cannot get 16bit model weights because
`stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False.
```

**根因**(已对照 `accelerate==1.1.1` wheel 逐行核实):

1. `train.py:235-237` 混档:`dit` → `zero2_config.json`(ZeRO-2),`t5`/`clip` →
   `zero3_config.json`(ZeRO-3,官方原样)。
2. `accelerator.prepare()` 每次把全局 `self.deepspeed_config` 覆盖成当前插件的配置
   (`accelerator.py:1822`)。prepare 顺序 dit → t5 → clip(`train.py:362/364/366`),
   全局 config 最后停在 **stage 3**。
3. 存 checkpoint 时 `get_state_dict(dit)` 只读全局 config(`accelerator.py:3361`)判断
   `stage == 3` → 走 ZeRO-3 分支;但 dit 引擎实际是 stage-2,
   `zero_gather_16bit_weights_on_model_save()` 返回 False → 直接抛错。
4. **为什么以前没事**:ZeRO-3 时代三插件全是 zero3,`zero3_config.json` 里
   `stage3_gather_16bit_weights_on_model_save: true`;切回 dit=zero2 后没处理这个残留。

**修复决策**:方案 A(调换 prepare 顺序让 dit 最后)被否决 —— `deepspeed_engine_wrapped`
绑的是**第一个** prepare 的引擎(`accelerator.py:1868-1870`),`accelerator.backward()`
只认它;改成 t5→clip→dit 后 backward 会打到纯推理的 t5 引擎上,
`engine.backward()` 里 `self.optimizer.backward(...)` 会 AttributeError。
**dit 必须第一个 prepare,这是 accelerate 的硬约束。**

采用**方案 B**(症状修复,等价于 `get_state_dict` 在 `stage != 3` 时的原样分支
`accelerator.py:3371-3373`),commit `596931c`:

```python
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
unwrapped_model_state = clone_tensors_for_torch_save(unwrapped_model.state_dict())
```

### 2.2 重跑:通过(2026-08-07,fix `596931c` 之后)

两条命令与 RUN.md §2 完全一致(`MAX_TRAIN_STEPS=100 CHECKPOINTING_STEPS=50`),
两腿都跑满 100 步,无 OOM,`checkpoint-100/dit_lora.safetensors` 正常落盘:

| 腿 | preflight | checkpoint 张量 | 结果 |
|---|---|---|---|
| 2a iso(`REF_ISOLATION=True`) | 50000 条 / 8 卡 / grad_accum=1 | **304 张量,空 0,全零 0** | ✅ |
| 2b baseline(`REF_ISOLATION=False`) | 同上 | **304 张量,空 0,全零 0** | ✅ |

> 校验方式:本报告用字节级解析 safetensors(不依赖 torch),判断 `shape 含 0` 为空分片、
> 数据段全 0x00 为全零张量。等价于 RUN.md §3 的 torch 判据。

**checkpoint 大小**:两腿各 ~1.9 GB。**host RAM 峰值 ~247 GB**(8 rank × 25.8 GB 克隆),
机器 3023 GB,无压力 —— 闸门 A 的"真风险"已实证无碍。

### 2.3 吞吐(重估 P2 时间账的依据)

取后 20 步稳态 s/it:

| 腿 | 稳态 s/it | 100k 步 | 备注 |
|---|---|---|---|
| iso(student) | **1.09** | ~30.3 h | 旧账 5.3–5.9 s/it 是 ZeRO-3 + grad_accum=2 |
| baseline(full) | **1.00** | ~27.8 h | 旧账 4.86 s/it 同上 |

两腿合计 **~58 h**(GPU 8 卡独占串行),已回填 SPEC §8(commit `4a942ad`,172 h → 70 h)。

---

## 3. 闸门 B:数据补齐 + 训练集

### 3.1 下载:102/102 全部分片就位

开始前 10/102 分片,要补齐 92 片约 1.9 TB。最终:

```
[fetch] 分片 102 个,已解压 102 个,待处理 0 个(下载量约 0.0B)
✅ 全部分片都已就位。下一步:python distill/build_stage1_official.py --strict
```

- 数据总量 **2.1 TB**,102 个 split 目录(最早的 split1 2026-07-20,最晚的
  split99 2026-08-08 06:11)。
- 下载过程遇到的坑与对策都记录在 `scripts/DOWNLOAD_RUNBOOK.md`:
  hf-mirror 把文件 302 到 Xet bridge 的签名 URL(必须 HTTP/2,`--http1.1` 会拿 0 字节)、
  代理截断 >1GB 的 range 请求(256MB 分块安全)、429 限速按代理出口 IP(换代理)、
  `os.path.exists` 在 ceph 上偶发永久挂起(见 §4)。

### 3.2 训练集:build --strict 通过

```bash
python distill/build_stage1_official.py --strict
```

输出(`logs/stage1_strict.log`):

```
[score 分布] 共 1011093 条有数值的记录
  [   4,  4.5)   404258  ████...  ← 官方阈值在这里
  [ 4.5,    ∞)        1

[跳过原因]
  score < 4.0                   606834
  异常高分剔除(对官方的偏离)                     1

[磁盘覆盖率] 引用 102 个 split;过 score 关 404258 条,其中磁盘上有 404258 条
[对官方满分池] 404258 / 404259 = **100.0%**
写入 datasets/UNO-1M/stage1_official_score4.json:404258 条
```

- **文件名不带 `_partial`**,覆盖 **100.0%** ≥ 95%。
- **异常高分剔除 1 条**:`score_final = 131184.67`(标注管线脏数据),`--anomaly_max=1000`
  剔掉,这是对官方判据的**唯一有意偏离**,已显式打印。
- 产物 `datasets/UNO-1M/stage1_official_score4.json`(145.6 MB,404258 条),schema
  `{prompt, image_tgt_path, image_paths}`,0 条缺字段。

---

## 4. 本步的工程优化:build 提速 ~50×

`build_stage1_official.py` 对 404,258 条记录做 ~808k 次文件存在性检查。
**第一版用子进程 `test -f`**(为了防 ceph 上 `os.path.exists` 偶发永久挂起),
但每次 fork 一个进程 ~7ms,808k 次要 **~100 分钟**。实测三种方案:

| 方案 | 单次 stat | 808k 次 |
|---|---|---|
| 子进程 `test -f`(原) | ~7 ms(fork 开销) | ~100 min |
| 原生 `os.path.exists` 串行 | ~1 ms | ~14 min |
| **daemon 线程池 64 + 原生 stat**(现) | — | **~90 s** |

**机制**:`stat()` 系统调用释放 GIL → 64 线程真正并行;某线程若在 ceph 上 D 状态
永久挂死,只损失它自己(daemon 线程,进程退出不被 join 卡住),其余线程照常出结果。
每个结果设超时 + 重试一次,仍超时才判不存在。

**修掉一个隐蔽竞态**:第一版 worker 用 `get_nowait()`,`先启线程再塞任务` 的间隙让
worker 撞上空队列提前退出,并发度塌掉(实测 300s 跑不完)。改成阻塞 `get()` + 哨兵退出后,
200k 条 10.9s(18.4k/s)。

dry_run 与 strict 输出与子进程版**逐字一致**(404258/404259 = 100.0%),正确性不受影响。

---

## 5. 带回来的证据(RUN.md §5 核对)

1. ✅ git log(四个关键 commit):
   `3172b67`(dit 切 ZeRO-2)→ `596931c`(fix checkpoint 保存)→
   `e304e6a`(旧标定报告)→ `4a942ad`(SPEC §8 重估 172 h → 70 h)
2. ✅ 两次标定 preflight 全段 + 张量检查两段完整输出(见 §2.2)
3. ✅ 两次标定 s/it 后 20 行(见 §2.3)
4. ✅ 下载收尾 `--dry_run` 输出(§3.1,`✅ 全部分片都已就位`)
5. ✅ `build_stage1_official.py --strict` 的 `[跳过原因]` 起到结尾完整输出(§3.2)

---

## 6. 下一步

P1 完成,P2 可开:两腿 stage-1 训练,串行 student ~30 h + baseline ~28 h。
具体命令与验收见 SPEC §8。
