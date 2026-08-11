# M5 臂 B 执行单 —— 官方 init + **隔离** 4000 步

> 对应 `DISTILL_PLAN.md` §11.9(设计)/ §11.11(执行顺序)。**档位:🟡 黄档**
> ——不改任何既有 `.py`/`.sh`(手册 R0),但要跑一次 ~6.2 小时的 8 卡训练。
> 总耗时:顺手件 ~2 min + 标定 ~15 min + 正式训练 **~6.2 h**。
>
> **导师已确认:这条消融就是他原本要的做法。** §11.5 的口径沟通条款不再阻塞本单。

## 这一步在干什么

臂 B 与臂 A **只差一个 flag**。这一差就是三边链条的第 ②′ 边,
也是 2026-08-04 之后的**主命题本身**:

| | init | 数据 | 配方 | 注意力 | 推理 |
|---|---|---|---|---|---|
| 臂 A(已完成) | 官方 LoRA | `train_mixed.json` | 同 | **全注意力** | 1.00× |
| **臂 B(本次)** | 官方 LoRA | `train_mixed.json` | 同 | **隔离** | 1.672× |

> **主命题**:同一份数据、同一个配方、同一个 init 下,
> **隔离相对全注意力的代价是多少**,换来 1.672× 加速。

**命题已按 §11.9 候选 ③ 事先弱化**为「官方 init + 4000 步能把隔离恢复到什么程度」,
**不是**「隔离有没有代价」。这条弱化是**预登记的**,不许读到结果后再调。
无论臂 B 落在哪一端都定得下东西:

| 臂 B 落点 | 边 ②′ | 边 ③(底座) | 意味着 |
|---|---|---|---|
| 身份 ≈ 0(像 `official_iso`) | 极大 | 小 | `ckpt-20000` 那 20000 步的隔离适配是**承重结构**,4000 步补不完 |
| 接近臂 A | 小 | 大 | 隔离适配便宜,`post4000` 的差距主要来自**底座** |

---

## 步骤 0:顺手两件事(纯只读,~2 min,**先做**)

两件都只作报告脚注,**不阻塞训练**,但趁 GPU 还空着先跑掉。

### 0a. `run_floor` 像素差确认(~30 s)

**在确认什么**:§11.10 的**附带产出二**——"跨会话重生成在感知上可忽略(59/60 判平局)"
——前提是那 60 对的两侧图**确实不同**。若逐位相同,这条要撤回。

> **⚠️ 这条翻不动边 ①。** §11.10 原先写的"逐位相同则边 ① 要撤回重做"
> 已在 08-04 划掉并订正:`run_floor` 即使逐位相同,它作为
> **"标注者在无差异对上的读数"**依然成立(而且那是这个读数最纯的形态),
> 59.4% ≪ 98.3% 的推理一个字不用改。

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull

python - <<'PY'
import json
import numpy as np
from PIL import Image

pairs = json.load(open("output/eval_arm_a/pairs_m5aactl.json"))["pairs"]
rf = [p for p in pairs if p["kind"] == "run_floor"]
same, diffs = 0, []
for p in rf:
    a = np.asarray(Image.open(p["img_0"]), dtype=np.int16)
    b = np.asarray(Image.open(p["img_1"]), dtype=np.int16)
    d = np.abs(a - b)
    if not d.any():
        same += 1
    diffs.append((float(d.mean()), float(d.max()), p["src_task_id"]))
diffs.sort()
print(f"共 {len(rf)} 对 | 逐位相同 {same} 对")
print(f"mean|Δ|  最小 {diffs[0][0]:.3f} ({diffs[0][2]}) / "
      f"中位 {diffs[len(diffs)//2][0]:.3f} / 最大 {diffs[-1][0]:.3f} ({diffs[-1][2]})")
print("最像的 5 对:", [(f"{m:.3f}", t) for m, _, t in diffs[:5]])
PY
```

**判读**:
- `逐位相同 0 对` ⇒ 附带产出二成立,照写;
- `逐位相同 > 0 对` ⇒ **把数字原样带回来**,附带产出二按相同的比例削弱或撤回。
  **不要自己改结论**,边 ① 不受影响。

### 0b. 查 97 个未解压 split 在不在盘上(~1 min)

**在确认什么**:标签引用 102 个 split,我们只解压了 split1-5(§11.7(a))。

```bash
du -sh datasets/UNO-1M/ 2>/dev/null
ls datasets/UNO-1M/ | head -30
ls datasets/UNO-1M/ | wc -l
ls datasets/UNO-1M/ | grep -c '^split' || true
```

**⚠️ 无论结果如何,都不走「重跑 stage-1」修复路线**(§11.11 已关死):
30 h GPU,且加上 `score_final ≥ 4.0` 过滤后盘子从 ~5 万掉到 1.7 万,
**样本量反而少 3 倍**,结果可能更差。这一条**只作报告脚注**。

---

## 步骤 1:前置检查(只读,~1 min)

臂 A 的门禁①(LoRA 键集合严格相等)**已经过了**,产物就在盘上,**不需要重跑**。
只确认它还在、没被动过:

```bash
ls -la log/official_init/dit_lora.safetensors
```

**没有这个文件就停下上报**——重新导出要 ~7 min 加载,而且要先跑
`python distill/export_official_lora.py --compare_only` 重过门禁,不许直接导。

### HF 离线环境变量(与臂 A 同,必须 shell 层补)

`train.py` 和训练 shell 都没设,H800 上走代理会**卡死**(无报错、无超时)。

```bash
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
```

### ⚠️ 与臂 A 最大的操作差异:**这次什么 flag 都不要追加**

`scripts/train_distill.sh:118` 的 accelerate 行里**本来就写死** `--ref_isolation True`,
那正是臂 B 要的值。臂 A 当初要追加 `--ref_isolation False` 去覆盖它;
**臂 B 直接用脚本默认,命令行末尾不加任何东西**——少一个可以敲错的地方。

**最容易犯的错就是照抄臂 A 的命令**,那样会跑出第二个臂 A,6 小时全废
且事后从产物分辨不出来(两者 init 相同、数据相同)。

---

## 步骤 2:100 步标定(~15 min,不污染正式目录)

```bash
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
MAX_TRAIN_STEPS=100 \
CHECKPOINTING_STEPS=50 \
PROJECT_DIR=log/arm_b_calibration \
RESUME_FROM_CHECKPOINT=log/official_init/dit_lora.safetensors \
bash scripts/train_distill.sh
```

### 要看的三件事

**1. `[preflight]` 六行全过**,特别是
`resume checkpoint: log/official_init/... (304 个张量)` —— 304 要与臂 A 那次一致。

**2. s/it 落在隔离档,不是全注意力档。** 这是本单唯一能直接抓住"flag 跑错"的信号:

```
M3 post4000  (隔离)     5.61 s/it
臂 A         (全注意力)  4.86 s/it
臂 B         (隔离)      预期 5.3 – 5.9 s/it   ← 应显著慢于臂 A
```

**方向解释同 `M5_ARM_A_RUN.md` 步骤 2**(§11.4 已逐行核过,别记反):训练里
**隔离更贵**(稠密掩码挡掉 FlashAttention、多一套 t=0 调制、训练从不用 KV cache),
**1.672× 加速全部来自推理侧,与掩码无关**。

⇒ **稳态 s/it ≤ 5.0(贴着臂 A 的 4.86)就停下上报**,那说明隔离没生效。

**3. 直接证据:确认进程命令行里没有混进 `False`。** 训练起来之后:

```bash
ps -ef | grep -o -- '--ref_isolation [A-Za-z]*' | sort | uniq -c
```

只应打出 `--ref_isolation True`。出现任何 `False` 就 kill 掉重来。

> **第二个佐证(非铁证)**:预览推理传 `kv_cache=args.ref_isolation`,臂 B 的预览
> 应**快于**臂 A 的 1.93–1.94 it/s,但耗时被 ZeRO-3 的 all-gather 主导,只能当佐证。

标定完 `log/arm_b_calibration` 留着别删,它是"配置确实生效过"的证据。

---

## 步骤 3:正式 4000 步(~6.2 h,后台 + 日志)

```bash
mkdir -p logs
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
nohup env \
  PROJECT_DIR=log/arm_b \
  RESUME_FROM_CHECKPOINT=log/official_init/dit_lora.safetensors \
  MAX_TRAIN_STEPS=4000 \
  CHECKPOINTING_STEPS=1000 \
  bash scripts/train_distill.sh \
  > logs/m5_arm_b.log 2>&1 &
echo "pid=$!"
```

- `PROJECT_DIR=log/arm_b` —— **不要用 `log/ref_distill`(M3,冻结)也不要用 `log/arm_a`**。
  同名会把 checkpoint 覆盖掉,而臂 A 的产物是唯一的现成基线,覆盖了就得重训 5.4 h。
- 每 1000 步一个 checkpoint,共 4 个。
- 按手册惯例每 ~30 min 报一行心跳(step / loss / s-per-it)。

**中途异常**:训练崩了不要自己重启。把最后 50 行日志带回来——
6 小时的任务重启一次就是半天,得先判断是配置问题还是偶发。

---

## 步骤 4:带回来

1. `logs/m5_arm_b.log` 的**首 40 行**(preflight 六项 + 加载 + 前几步 loss)
   与**末 20 行**(总步数 / 总耗时 / 最终 loss);
2. `ls -la log/arm_b/` —— 4 个 checkpoint 目录都在;
3. 稳态 s/it,与臂 A 的 **4.86** 和 M3 的 **5.61** 两个锚点对照;
4. 中途有无 NaN / OOM / NCCL 警告;
5. 步骤 0a / 0b 两段的输出。

### ✅ 确认点(用户来判)

- `ps` 只打出 `--ref_isolation True`,且稳态 s/it 明显慢于 4.86
  ⇒ 跑的确实是臂 B 而不是第二个臂 A;
- preflight resume 是 `log/official_init/...` 的 304 个张量
  ⇒ init 与臂 A 完全对齐,边 ②′ 是单变量;
- 4000 步零异常、4 个 checkpoint 落盘 ⇒ 可以进入出图环节。

---

## 之后要做的(**本执行单不含**,顺序已在 §11.11 锁死)

### 1. 出图:**只生成 `arm_b` 一个变体**(~23 min GPU)

对侧**直接复用臂 A 批现成的 192 张 `arm_a_full`**。§11.7 当时"两个变体必须同会话
重生成"的理由已被 08-04 的标定批自己拆掉:跨会话差异的**感知读数就是 98.3% 平局**。

**外加同会话补 30 张 `official_full`** 作 `run_floor` 对照(+2.6 min GPU)。
WHY 不复用 M4↔臂A 的旧 `run_floor`:对照就该跨**与被检验那一对相同的会话间隔**,
而 §11.3 步骤 1 实测到会话漂移**不是常数**(smoke→M4 那段远大于 M4→08-04 那段)。

> **本地已就绪 [2026-08-04]**:`eval_multiref.py` 的 `VARIANTS` 已加
> `("arm_b_iso", False, True, True, "arm_b")`,并加了 `--arm_b_lora`
> (默认 `log/arm_b/checkpoint-4000/dit_lora.safetensors`)。
> bank 存在性检查仍是**只查本批用到的**,已回归验过:现有五个任务单
> 没有一个会因此要求 `arm_b` bank。任务单生成脚本待写。

### 2. §9 身份留存门(判读 **5 min**,**在任何偏好盲评之前**)

逐字复用 P-probe 的抽样(同 30 条任务、同 `random.Random(20260803)`),
只数 `arm_b` 一个变体的 51 问。锚点 **0/51**(`official_iso`)与
**45/51 = 88.2%**(`official_full`)现成可比。

| per-subject 留存(**点估计**) | 判定 |
|---|---|
| **< 60%** | **偏好批取消。** 臂 B 与臂 A「一眼可辨」,§8 偏好盲评名不副实。结论直接写:**4000 步补不完隔离适配,边 ②′ 承重、边 ③ 小,留存率就是读数。** 项目在此收口 |
| **≥ 60%** | 进第 3 步 |

**这个阈值是预登记的(§11.11d),出图之后不许改。**
声明随门一起引用:这把 §9 尺子只在极端对比(0% vs 88%)上验证过,
落在 50–70% 带里时最不可靠;**它是决策规则,不是测量值**。

### 3. 终批:222 对(判读 ~16 min)

`arm_b_iso`(`key_0` = 被检验方)vs `arm_a_full`(`key_1` = 基线)192 对
+ `run_floor` 30 对,**不带 replay**(自洽率已测两次,第三次不承重)。
新盲种,问题逐字沿用。**这是新主命题本身的直接单变量读数,也是最后一个批次。**

⚠️ 走 §8 盲评的批次,交付物里**不放拼图**(拼图带变体名列头,是已揭盲的)。
