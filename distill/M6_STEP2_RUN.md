# M6 步骤 2 执行单 — 两腿 stage-1,双机并行

> 对应 `distill/M6_ABLATION_SPEC.md` §8 的 P2。**档位:🟢 绿档**——不改任何既有代码,
> 只是在两台机器上各起一条已经标定过的训练。
> 墙钟 ~30 h(并行),GPU 总账 ~58 h 不变。P1 两个闸门已过,见 `M6_STEP1_REPORT.md`。

## 这一步在干什么

训出消融的两个底座。两腿**只差 `REF_ISOLATION` 这一个 flag**:

```
机器 A  REF_ISOLATION=True   log/stage1_official        30.3 h   ← student 腿
机器 B  REF_ISOLATION=False  log/stage1_official_full   27.8 h   ← baseline 腿 = 官方 stage-1 复刻
```

同一份 `stage1_official_score4.json`(404,258 条)、同样 100000 步、
同样 `grad_accum=1`、同样 8 卡、同一个脚本、同一个 commit。

## 为什么可以拆到两台机器上

SPEC §3 的同一性条件里钉死的是 **world size = 8**——
`train.py` 的 `set_seed(args.seed, device_specific=True)` 让每个 rank 的种子是
`seed + process_index`,数据顺序因此依赖**卡数**,不依赖机器。
两台各 8 卡,两腿的数据顺序就是同一条,与串行跑在同一台上等价。

§3 没有钉"同一台机器",所以双机并行**不需要修正预登记**。
但下面这条是硬的:

> ⚠️ **两台都必须正好 8 卡。** 任意一台不是 8,按 §3 是"本组作废"——
> 不是慢一点,是这一组数据不能用。`scripts/train_stage1_official.sh` 的 preflight
> 对卡数只**警告**不退出(`⚠️ 可见 N 卡 < NUM_PROCESSES=8`),所以这条得人来把。

> § SPEC §8 写的是"串行 student 30 h + baseline 28 h"。改成并行只动墙钟,
> 不动 GPU 总账,也不动任何同一性条件。跑完按房规补一条带日期的说明即可。

---

## 阶段 0 · 4090 上先做完(等 H800 期间)

### ❌ 先说不能做的:别在 4090 上跑训练

三条,任意一条都足以否掉:

1. **跑不起来。** dit 现在走 ZeRO-2(`3172b67`),每张卡都要放下完整的
   25.8 GB bf16 FLUX;4090 是 23.65 GB,加载模型就 OOM。
   ZeRO-3 那条求生路是 4090 时代的产物,已经按 SPEC §3.1 撤了。
2. **改回 ZeRO-3 测的是另一条路径,而且会得到假阴性。**
   `596931c` 的 checkpoint 修复(`clone_tensors_for_torch_save(unwrapped_model.state_dict())`)
   **只在 ZeRO-2 下正确**:ZeRO-3 下参数是切开的,`state_dict()` 拿到的是空分片,
   会静静存出一个全空的 LoRA。到时候你看到的"bug"是自己造的。
3. **冒烟测试已经做过了。** 闸门 A(2026-08-07,H800,fix `596931c` 之后)
   两腿各跑满 100 步、存了两次 checkpoint、`304 张量 / 空 0 / 全零 0`、
   host RAM 峰值 247 GB / 3023 GB。那就是真硬件真并行配置下的冒烟测试。

> **推论:如果你在 4090 上改了 `train.py` 或 `scripts/train_*.sh`,闸门 A 作废。**
> 改动 push 之后,H800 上必须先重跑一遍 100 步标定(`M6_STEP1_RUN.md` 步骤 2/3),
> 才能开长跑。60 GPU-小时不值得省这 20 分钟。

### 0.1 代码状态

```bash
cd ~/UNO && git pull
git log --oneline -1
git status --short
```

记下 commit 短哈希,阶段 1 要跟 H800 对。工作区应该是干净的
(`log/` 下的东西不算,那不进 git)。

### 0.2 共享盘与数据

```bash
ls -d /kaimm-distill/wuwenxuan/UNO/datasets/UNO-1M/images/split1
ls -l  datasets/UNO-1M/stage1_official_score4.json
sha256sum datasets/UNO-1M/stage1_official_score4.json
```

**把这个 sha256 抄下来。** 两台 H800 要对同一个值——SPEC §3 第一条要求
两腿用"同一次 build 的产物",不是"两份看起来一样的文件"。

### 0.3 preflight 空跑(不碰 GPU)

`train_stage1_official.sh` 在 `accelerate launch` 之前有一段纯 Python 自检:
json 存在/非空/schema、抽查 200 条图片在不在盘上、dreambooth submodule、
PROJECT_DIR 可写。这段在 4090 上跑得动,而且值钱——
它防的是"训到第 3 万步 FileNotFoundError",4090 那次就是这么炸的。

想只跑自检不进训练,把步数设成 0 会被 accelerate 拒;直接看输出就行:

```bash
MAX_TRAIN_STEPS=100 PROJECT_DIR=log/_preflight_only \
bash scripts/train_stage1_official.sh 2>&1 | head -20
```

看到这几行就够了,然后 **Ctrl-C**(它接着会去加载模型,在 4090 上必 OOM):

```
[preflight] 训练集: datasets/UNO-1M/stage1_official_score4.json
[preflight] 404258 条 = 官方满分池的 100.0%  [官方口径]
[preflight] GPU N 卡 / ref_isolation=True / grad_accum=1
[preflight] === 自检通过,开始训练 ===
```

`[官方口径]` 不能是 `⚠️ 部分数据`;404258 不能是别的数。
跑完删掉 `log/_preflight_only`。

### 0.4 把两条启动命令写死

临场手打是最容易出事的一步(打错 `PROJECT_DIR` 要 30 小时后才看得出来)。
在 4090 上先把两个文件写好,H800 到手直接 `bash`:

`run_A_iso.sh`:

```bash
#!/usr/bin/env bash
cd ~/UNO && mkdir -p logs
REF_ISOLATION=True PROJECT_DIR=log/stage1_official \
setsid bash scripts/train_stage1_official.sh \
  > logs/p2_iso.log 2>&1 < /dev/null &
echo "pid=$!  ← 记下来"
```

`run_B_full.sh`:

```bash
#!/usr/bin/env bash
cd ~/UNO && mkdir -p logs
REF_ISOLATION=False PROJECT_DIR=log/stage1_official_full \
setsid bash scripts/train_stage1_official.sh \
  > logs/p2_full.log 2>&1 < /dev/null &
echo "pid=$!  ← 记下来"
```

> **别用 `nohup`。** `torch.distributed.elastic` 会装自己的 SIGHUP handler
> 覆盖掉 nohup 的 `SIG_IGN`,臂 B 被这个杀过两次(DISTILL_PLAN §11.12(a))。
> `setsid` + `< /dev/null` 才是对的。

其余参数**一个都不要填**——`lora_rank 512 / lr 8e-5 / batch 1 / res 512 /
100000 步 / grad_accum 1 / checkpointing 1000` 全走 `train.py` 与脚本的默认值,
那就是官方配方,显式重写一遍只会引入抄错的机会。

---

## 阶段 1 · H800 到手,开跑前的五分钟闸门

**两台上都跑一遍,逐条对齐。任何一行对不上就停下来。**

```bash
cd ~/UNO && git pull
echo "=== 卡数 ===";   nvidia-smi -L | wc -l
echo "=== commit ==="; git rev-parse --short HEAD
echo "=== 工作区 ==="; git status --short
echo "=== 数据 ===";   sha256sum datasets/UNO-1M/stage1_official_score4.json
echo "=== 依赖 ===";   python -c "
import torch, deepspeed, accelerate
print(torch.__version__, deepspeed.__version__, accelerate.__version__)"
echo "=== 盘 ===";     df -h . | tail -1
echo "=== 残留 ===";   ls -d log/stage1_official log/stage1_official_full 2>/dev/null
```

| 项 | 必须 | 对不上的后果 |
|---|---|---|
| 卡数 | **两台都是 8** | 有效 batch 变了,§3 判本组作废 |
| commit | 两台相同,且 ≥ `596931c` | 存 checkpoint 会崩(闸门 A 那个 bug) |
| 工作区 | 干净 | 跑的不是仓库里那份代码,事后说不清 |
| 数据 sha256 | 两台相同 | §3 第一条:必须是同一次 build 的产物 |
| torch / deepspeed / accelerate | 两台相同 | 不是 §3 的硬条件,但不同就得在报告里写明 |
| `log/stage1_official*` | **不存在** | 已有 checkpoint 时脚本会拒绝启动(要求传 RESUME) |

先激活环境:`source .venv-uno/bin/activate`(不激活的话脚本第一行 python 都找不到)。

另外确认**这两台上没有别的活**——尤其是数据下载。纯 CPU 长任务会因为
"GPU 空转"被利用率考核强杀(2026-08-06 实测),而训练本身满载,不受影响。

---

## 阶段 2 · 开跑

机器 A:

```bash
bash run_A_iso.sh
```

机器 B:

```bash
bash run_B_full.sh
```

**起来后第一件事,各自确认自己是哪条腿**(这是事后唯一能确认
"这个 checkpoint 出自哪条腿"的地方):

```bash
grep -m1 "ref_isolation" logs/p2_iso.log      # 机器 A:ref_isolation=True  / grad_accum=1
grep -m1 "ref_isolation" logs/p2_full.log     # 机器 B:ref_isolation=False / grad_accum=1
```

机器 B 还会多打一行 `ℹ️ ref_isolation=False:这是官方 stage-1 的复刻`,那是对的。

脚本里有一条硬 guard:`REF_ISOLATION=False` 却往 `log/stage1_official` 写会直接
退出。**反方向没有 guard**——`True` 写进 `log/stage1_official_full` 不会被拦,
所以上面那两行 grep 必须自己看一眼。

---

## 阶段 3 · 盯什么

### 前 10 分钟

模型加载 ~7 min 期间日志是静的,正常。之后:

```bash
tail -f logs/p2_iso.log
```

- 稳态 s/it 应该在 **1.09(iso)/ 1.00(full)** 附近。差一倍以上就停下来查
  (最可能是卡数或 `grad_accum` 不对)。
- 第一个 checkpoint 在 **第 1000 步**(iso 约 18 min)。它落盘之前,
  ZeRO-2 那个整模型量级的内存峰值还没被这台机器验过——闸门 A 是在另一次开机上验的。

### 第一个 checkpoint 落盘后(必看)

```bash
du -sh log/stage1_official/checkpoint-1000
free -g | head -2
```

100000 步 / 1000 = **每腿 100 个 checkpoint**。拿上面量到的单个大小 ×100×2 腿
估总量,对着 `df -h` 看一眼。P1 时盘上有 142 TB 可用,大概率不是约束,
但量一次比事后清理便宜。

### 之后

每天看一次就够:

```bash
tail -3 logs/p2_iso.log; ls log/stage1_official | tail -3
```

**不许中途看结果挑 checkpoint**(SPEC §7.1)。两腿都跑满 100000 步,
评测只用最后一个。

### 断了怎么续

```bash
RESUME_FROM_CHECKPOINT=latest REF_ISOLATION=True PROJECT_DIR=log/stage1_official \
setsid bash scripts/train_stage1_official.sh > logs/p2_iso_resume.log 2>&1 < /dev/null &
```

`latest` 会让 `global_step` 从 checkpoint 编号起算(传显式路径会归 0,
续训变成多跑一遍)。**别传显式路径。**
`REF_ISOLATION` 和 `PROJECT_DIR` 续训时同样要带上,默认值不是你要的那条腿。

---

## 跑完直接接 P3(同一台机器,不用换)

每台机器跑完自己的 stage-1,立刻在**同一台**上跑自己的蒸馏腿,继续并行:

```bash
# 机器 A
REF_ISOLATION=True \
RESUME_FROM_CHECKPOINT=log/stage1_official/checkpoint-100000/dit_lora.safetensors \
PROJECT_DIR=log/ref_distill_iso \
setsid bash scripts/train_distill.sh > logs/p3_iso.log 2>&1 < /dev/null &

# 机器 B
REF_ISOLATION=False \
RESUME_FROM_CHECKPOINT=log/stage1_official_full/checkpoint-100000/dit_lora.safetensors \
PROJECT_DIR=log/ref_distill_full \
setsid bash scripts/train_distill.sh > logs/p3_full.log 2>&1 < /dev/null &
```

**三个变量一个都不能漏**,`train_distill.sh` 的默认值全是旧的:

| 变量 | 默认值 | 漏了会怎样 |
|---|---|---|
| `PROJECT_DIR` | `log/ref_distill` | **覆盖掉 M3 的既有结果**。这个脚本没有"目录里已有 checkpoint 就拒绝启动"的 guard |
| `RESUME_FROM_CHECKPOINT` | `log/ref_isolation/checkpoint-20000/...` | 从 4090 那个旧底座续训,消融的两腿就不成对了 |
| `REF_ISOLATION` | `True` | 机器 B 会变成第二条 iso 腿 |

蒸馏用的 `train_mixed.json` **不许重新生成**:它当初的单 ref 池是被
"盘上只有 split1-5"卡出来的 16,966 条,现在 102 个 split 全在盘上,
重跑 `build_train_json.py` 会变成 404,258 条,60/40 的混比彻底改掉。
SPEC §3 要求两腿用**同一份**,而那一份是冻结的。

`--gradient_accumulation_steps 2` 写死在 `train_distill.sh` 里,与 SPEC §3
的"蒸馏 4000 步 / accum 2"一致,**不要动**。

> 时间账:SPEC §8 记的"合计 ~12 h"是 ZeRO-3 时代的数。ZeRO-2 下大概率快很多,
> 但**没标定过**,跑完拿实测回填,别先改文档。

---

## 带回来

1. 两台阶段 1 闸门表的完整输出(卡数 / commit / sha256 / 版本);
2. 两条日志各自的 `[preflight] ref_isolation=...` 那一行;
3. 两条日志的**末尾 20 行**(总步数 / 耗时 / 有无报错);
4. 两腿最后一个 checkpoint 的张量检查(判据同 `M6_STEP1_RUN.md` 步骤 3:
   `304 张量,空分片 0,全零 0`);
5. 两腿的稳态 s/it(取最后 20 步),用来回填 SPEC §8。

## ✅ 确认点(用户来判)

- 两腿都跑满 **100000** 步,没有中途挑 checkpoint。
- 两腿的 `ref_isolation` 分别是 True / False,`PROJECT_DIR` 没写反。
- 两腿 checkpoint 的张量检查都过。

三条都过,P2 完成,进 P4(扩任务池 192→320 → 两腿生图 → 盲评)。
