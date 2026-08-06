# M6 隔离消融 —— 预登记

> **版本** v1(2026-08-06) · **上游** `distill/DISTILL_PLAN.md` §8 / `distill/OVERVIEW.md` §5
>
> 本文档是**预登记**,不是实现规格。它回答"看结果之前我们承诺了什么",
> 不回答"代码写成什么样"。实现规格(如果需要)另开文件。
>
> **本文档在任何一腿训练启动之前落盘。** 落盘后按 §7 冻结。

---

## 1. 为什么要有这一组

现有的三段链条 `official_full ─①─ 臂A ─②′─ 臂B ─③─ post4000` 是补出来的:
每条边两侧的模型来自不同的 init、不同的训练历史,单条边能读的东西有限,
而 §8.5-3 又禁止把三条边的读数相加。三条边至今**全部**落在"判据不适用"上。

本组换一个做法:**从零起两条腿,除了 `--ref_isolation` 之外处处相同。**
这是隔离这个设计唯一一次被单变量地测量。

顺带一个不是副产品的副产品:**baseline 腿就是官方 stage-1 的忠实复刻**
(官方 `train.py` 里 `ref_isolation` 默认 `False`,这个 flag 是本 fork 加的)。
我们至今没有过这个参照物。

---

## 2. 设计

```
                    stage-1 (UNO-1M, score≥4)        蒸馏 (train_mixed.json)
  student  腿  ──►  REF_ISOLATION=True        ──►    REF_ISOLATION=True     ──► M6_ISO
  baseline 腿  ──►  REF_ISOLATION=False       ──►    REF_ISOLATION=False    ──► M6_FULL
```

**对角线设计**,不是 2×2 全因子。因此本组**不能**区分隔离的效应来自 stage-1
还是来自蒸馏。要区分得再补两条腿(各 4000 步,算力几乎白送),但多两批判读
——判读时间是瓶颈资源。**先跑对角线,看到效应再决定补不补。**

顺带一提:全因子里 "full stage-1 + iso 蒸馏" 那一格约等于现有的**臂 B**,
已有近似读数(§11.11)。

### 2.1 两腿的命令

```bash
# stage-1
REF_ISOLATION=True  PROJECT_DIR=log/stage1_official      bash scripts/train_stage1_official.sh
REF_ISOLATION=False PROJECT_DIR=log/stage1_official_full bash scripts/train_stage1_official.sh

# 蒸馏
REF_ISOLATION=True  PROJECT_DIR=log/m6_iso  \
  RESUME_FROM_CHECKPOINT=log/stage1_official/checkpoint-<N>/dit_lora.safetensors \
  bash scripts/train_distill.sh
REF_ISOLATION=False PROJECT_DIR=log/m6_full \
  RESUME_FROM_CHECKPOINT=log/stage1_official_full/checkpoint-<N>/dit_lora.safetensors \
  bash scripts/train_distill.sh
```

两腿共用同一个脚本。**不许为 baseline 另写脚本** —— 那等于给自己开一个
"抄漏一行"的口子,而抄漏的那一行几十小时后才看得出来。

---

## 3. 同一性条件(不可改清单)

以下任意一条两腿不同,**本组作废**:

| 项 | 值 |
|---|---|
| stage-1 训练集 | 同一个 `stage1_official_score4*.json` 文件(同一次 build 的产物) |
| 蒸馏训练集 | 同一份 `datasets/distill_multiref/train_mixed.json` |
| world size | **8**(`train.py` 的 `set_seed(..., device_specific=True)` 让数据顺序依赖卡数) |
| seed / lr / lora_rank / batch_size / resolution / lr_scheduler / warmup | `train.py` 默认,两腿都不手填 |
| stage-1 步数 / grad_accum | 见 §4(待定,但**两腿必须相同**) |
| 蒸馏步数 / grad_accum | 4000 / 2 |
| DeepSpeed 配置 | 现状 ZeRO-3,两腿相同。**不改回 ZeRO-2**——它对两腿一样,不是变量 |
| 推理:sampler / steps / guidance / seed / 分辨率 | 与臂 A/B 批逐字相同 |

**推理侧一条硬约束:baseline 腿不许开 KV cache。** 它没有隔离,cache 是有损的。
速度对比是另一件事,不许混进质量批。

---

## 4. 两个待定档位 —— 由 P1 的实测下载速率决定

以下两项**现在不定**,因为定它们要先知道补齐 118 GB 数据要几小时还是几天:

| 档位 | 选项 A(官方复刻) | 选项 B(只要消融) |
|---|---|---|
| stage-1 步数 | 100000(`train.py` 默认) | 40000 |
| 数据 | 全库 `--strict`(覆盖率 ≥ 95%) | 现有 split1-5,`--allow_partial` |
| 算力 | ~160 h | ~64 h |
| 得到 | 消融结论 **+** 官方 stage-1 参照物 | 消融结论 |

**现在就预登记的不变量:无论选哪个,两腿取值必须相同。** 消融的效力只依赖
"两腿相同",不依赖"等于官方"。选 B 时,产出的底座**不得**被称作官方复刻,
报告里要写实际步数与实际覆盖率。

**选定后回填本节并注明日期,此后冻结。** 回填必须发生在任何一腿训练启动之前。

### 4.1 选 B 的已知风险

40000 步可能欠训到两腿都很差,此时"两腿无差别"是被地板压出来的,不是隔离无代价。
**缓解**:两腿都留 loss 曲线(§6),末端 loss 若仍在明显下降,报告里必须标注
本组是"欠训区间内的比较"。

---

## 5. 判据与样本量

### 5.1 判据

沿用 `M4_EVAL_SPEC.md` §8.2,一字不改:

```
非平局胜率 p̂ = S / (S + T) 的 Wilson 95% CI 下界 ≥ 0.40
且  n_nontie ≥ 94
```

`key_0` = M6_ISO(被检验方),`key_1` = M6_FULL(基线)。
样本不足时结论是**「判据不适用」**,不是「没达标」。

### 5.2 样本量 —— 本组唯一一个事后补不回来的东西

三条边全部卡在同一处:`n_nontie` 不够。边 ③ 是 192 主对 → **n_nontie = 89 < 94**,
CI 下界 0.404 其实已经过线,**只有样本量没过**。

§11.7 的记录显示,当初按零假设对的 33.3% 平局率折算 `n_nontie ≈ 128`,
而实测平局率是 53.6%(非平局率 **0.4635**)。折算用错了平局率,于是连撞三次。

按 Wilson 反解(z=1.96),要让判据**能过**所需的样本:

| 真实 p̂ | 需要 n_nontie | 折合主对(÷0.4635) |
|---|---|---|
| 0.50(隔离完全无损) | 94 | 203 |
| 0.48 | 145 | 313 |
| 0.46 | 255 | 550 |

**预登记:主对 320 条 + in-batch `run_floor` 30 对 = 350 对。**
按 0.4635 折算 `n_nontie ≈ 148`,覆盖到 p̂ ≥ 0.48 那一档。
判读约 25 分钟(按臂 B 批 222 对 ≈ 16 分钟的实测速率)。

再往上性价比迅速下降:覆盖 0.46 要 550 主对,判读接近一小时。

> **这不是"事后追加样本"。** §11.7 禁的是"看到 `n_nontie` 不够、回头补样本救判据"。
> 本组的 320 是在**任何一张图生成之前**定的,并且写在这里。
> 一旦本组跑完,若 `n_nontie` 仍不足 94,结论就是「判据不适用」,
> **不许再补**。

### 5.3 平局率的假设

0.4635 来自边 ③(post4000 vs 臂B,同族比较)。M6 两腿是各自从零训的,
差异可能更大 ⇒ 平局更少 ⇒ `n_nontie` 更高,这个方向对我们有利。
但反向也可能。**320 这个数按 0.4635 定死,不因实际平局率而调整。**

### 5.4 尺子

本批自带 30 对 `run_floor`,**只在批内使用**。
§8.5-3:跨批非平局胜率不得并列引用,跨批相减禁止(κ=0.274)。

---

## 6. 方向性偏置声明(必须在看结果前写下)

`train_mixed.json` 的 target 图**是官方全注意力 teacher 生成的**。
蒸馏目标因此天然带全注意力的归纳偏置,**这对 baseline 腿有利**。

所以本设计对隔离方是**保守**的:

- 隔离腿若打平或胜出 ⇒ 结论更硬(顶着不利的目标分布还打平);
- 隔离腿若落败 ⇒ **不能**直接归因于隔离本身,必须在报告里写明这条偏置。

**这条现在写下来才叫预登记,结果出来再说就是找补。**

## 6.1 两腿都要留的东西

- 完整 loss 曲线(每腿 stage-1 + 蒸馏各一条)
- 稳态 s/it(顺便补上一直缺的"隔离在训练侧确实更慢"的实证)
- 每 1000 步 checkpoint

---

## 7. 停止规则与冻结

1. **两腿都跑满 §4 定下的步数。** 不许中途看结果挑 checkpoint。
2. 评测只做一次,用最后一个 checkpoint。
3. 本文档 §2/§3/§5/§6 自落盘起冻结。§4 允许**一次**回填(选 A 或 B),
   回填后同样冻结。
4. 之后任何修正按房规:`~~原文~~` + `**[日期 修正]**`,不许原地改写。
5. 盲法:本批 `boards/` **不放行进 git**,判读完成前不得生成带变体名的拼图。

---

## 8. 执行顺序

```
P0 ✅ 预登记(本文档)+ 代码改动
P1    实测下载速率 ──► 回填 §4 ──► 补数据 ──► build_stage1_official.py
P2    两腿 stage-1(串行)
P3    两腿蒸馏
P4    扩任务池 192→320 ──► 两腿生图 ──► build_pairs.py m6 ──► 盲评 ──► report
P5    报告
```
