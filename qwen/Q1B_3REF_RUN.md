# Q1-B:teacher 3-ref 能力探底 + P0 交付 · 上机执行单

> 给远程 agent。**先读 `../distill/REMOTE_AGENT_HANDBOOK.md` §2.0(红/黄/绿三档)。**
> 本文件是**黄档规格**:点名的新建文件你可以写,常量和规则写死到没有自由度,
> 产物我在本地按规格逐条复算。**既有 `.py`/`.sh` 一个字不许改(R0)。**
>
> 上下文:`qwen/PLAN.md`(本轮计划)、`qwen/GOAL.md`(目标)。本单只覆盖 P0 与 Q1-B。

---

## 0. 你要做三件事,顺序不能换

| | 事 | 在哪做 | 卡 |
|---|---|---|---|
| **①** | 必查:diffusers 认不认 `zero_cond_t` | 任意机器 | 0 |
| **②** | P0 交付:把 Q1 的产物取回仓库并提交 | 4090(共享盘已挂) | 0 |
| **③** | Q1-B:122 条 3-ref,**8 卡分片**推理 | infer_hub `--cluster h --gpus 8` | 8 |

① 不过 **全部停下来回报**——它可能推翻已经跑完的 Q1。

---

## 1. ① 必查:`zero_cond_t`

**为什么这条排第一。** Qwen-Image-Edit-2511 的 transformer config 里有
`"zero_cond_t": true`,它让**参考图段用固定 t=0 的调制**、不跟去噪步走
(`transformer_qwenimage.py` 里 `timestep = torch.cat([timestep, timestep*0])` +
`modulate_index` 给 `img_shapes[1:]` 标 1)。这是整个 Qwen 路线的地基。

但 env 里的 diffusers 是 `0.40.0.dev0` 的某个快照,而 `zero_cond_t` 是较晚合入的。
**若这个快照的 `QwenImageTransformer2DModel.__init__` 不接受这个参数,
`from_pretrained` 会静默把它丢掉**——不报错、不警告到你会注意的程度,
但 ref 就不再是 t=0 调制。那样:

- Q1 那 40 张**是在错的语义下跑出来的**,基线要重跑;
- `qwen/PLAN.md` §1 的地基在实际运行环境里不成立。

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
$E/bin/python - <<'PY'
import inspect, diffusers
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as M
sig = inspect.signature(M.__init__).parameters
print("diffusers:", diffusers.__version__)
print("zero_cond_t in __init__:", "zero_cond_t" in sig)
import json, os
cfg = json.load(open(os.path.join(os.environ["QWEN_WEIGHTS"], "transformer/config.json")))
print("config zero_cond_t:", cfg.get("zero_cond_t"))
PY
```

**两个都是 `True` ⇒ 继续。任何一个不是 ⇒ 停,把输出原样贴进报告,等我裁决。**

实例层面的第二道确认已经写进 §3.3 的脚本规格(加载后断言),不用你单独跑。

---

## 2. ② P0 交付:把 Q1 的产物取回仓库

Q1 的产物还在 H 机 `/kaimm-distill/wuwenxuan/output/qwen_baseline`,**仓库里零凭据**。
而"Qwen 比 UNO 好不少"是后面**所有** student 数字的锚点,`qwen/GOAL.md` §5.1 的
7 条豁免集也出自它。按这个项目的规矩(`OVERVIEW.md` 开头:「所有承重数字都在」),
它必须有出处。

**做什么**:

1. 把 `results.json` 与 `ALL_COMPARISON*.png` 拷进仓库 `output/qwen_baseline/`。
   `.gitignore` 已经放行这两类,**全分辨率单图不要拷**(40 张 1024² 约 60 MB,
   留在共享盘)。
2. 新建 `reports/20260812-q1b-3ref/REPORT.md`(绿档,你可以写),里面写清楚:
   - Q1 实际跑了什么口径(从 `results.json` 的 `meta` 原样抄,不要转述);
   - 40 条是哪 40 条(`i % 8 == 0`,给出 task_id 列表);
   - 实测 `total_s` / `n_run` / `s/img` / `peak_mem_gb` / `n_fail`;
   - §1 那条必查的原始输出。
3. `git add` 上面这些 + 本轮新增的 `qwen/`、`datasets/eval_multiref/q1b_3ref_tasks.json`,
   commit + push。commit message:`docs(qwen): Q1 产物回仓 + Q1-B 3-ref 任务表`。

**不要**做的:不要判读那 40 张图、不要写"效果好/不好"的结论、不要改 §5.1 那 7 条。
判读是我的事。

---

## 3. ③ Q1-B:122 条 3-ref

### 3.1 问的是什么

上一轮在 UNO 上,3-ref 是**系统性失效**的:M1 生成的 4000 条 3-ref 数据人工通过率只有
**3.5%**(`DISTILL_PLAN.md` §4.2),M4 的 S4 层 20 条里 75% 被判平局、
明写"这些平局大概率是一样烂"。所以「3-ref 到底能不能做」这件事,**在 UNO 上一直没有答案,
只有一个坏答案**。

Q1-B 就是拿新 teacher 把这个问题重问一遍。**这是探底,不是验收**——
没有判据、不进 §8.2、不做盲评,产出是 122 张图和一份可看的拼图。

### 3.2 任务表(已在 git 里,**不许你枚举任何东西**)

`datasets/eval_multiref/q1b_3ref_tasks.json`,**122 条,全跑,不取子集**。

| | |
|---|---|
| 来源 | held-out 10 主体的**全部 112 个合法 3-组合**(C(10,3)=120 减去含两个 stuffed animal 的 8 个) |
| 段 A | 20 条,`task_id` 形如 `S4_000_s0`。与 `eval_set.json` 的 S4 层 **prompt / image_paths / seed 逐字相同** ⇒ 与 M4 的 UNO 结果直接可比 |
| 段 B | 102 条,`task_id` 形如 `Q1B_001_s0`,补齐 112 组合的覆盖 |
| 生成器 | `qwen/build_q1b_3ref_tasks.py`,已在本地跑过自检并提交。**你不要重跑它,也不要改它** |

自检已过:122 条 / 112 组合全覆盖 / 全部 held-out / seed 唯一且不与 M1 区间重叠 /
段 A 与 `eval_set.json` 逐字一致 / 参考图齐全。

### 3.3 你要新建的脚本:`qwen/infer_qwen_3ref.py`

结构照 `scripts/infer_qwen_edit.py`(那是既有文件,**只许读、只许照着写新的,不许改**)。
差别只有两处:任务表换了、加了 8 卡分片。

**CONSTANTS 段(写死,不做 CLI 可调。改任何一个都是红档)**:

```python
TASK_JSON           = "datasets/eval_multiref/q1b_3ref_tasks.json"   # 全部 122 条,无 stride
NUM_INFERENCE_STEPS = 40        # 与 Q1 逐字相同
TRUE_CFG_SCALE      = 4.0       # 与 Q1 逐字相同
NEGATIVE_PROMPT     = " "       # 一个空格,与 Q1 逐字相同
HEIGHT = WIDTH      = 1024      # 与 Q1 逐字相同
DEFAULT_OUT         = "output/qwen_3ref"
BOARD_MAX_BYTES     = 2 * 1024 * 1024
BOARD_ROWS_PER_PART = 8
```

prompt **原样**,不加前缀不改写;seed 取任务自带的 `task["seed"]`;
`image` 顺序**原样**,不重排。以上四条与 Q1 相同,不许动。

**CLI(只有这六个,都不改变实验语义)**:

| 开关 | 语义 |
|---|---|
| `--dry_run` | 不加载模型、不用 GPU,用纯色占位图验证分片 / 落盘 / 合并 / 拼图 |
| `--shard_idx N` | 分片下标,默认 0 |
| `--num_shards N` | 分片总数,默认 1 |
| `--merge` | 只合并 + 出拼图,不推理 |
| `--out PATH` | 覆盖输出目录 |
| `--limit N` | 只跑本分片的前 N 条(冒烟用,不改变分片规则) |

**分片规则(唯一一条)**:按任务数组**原始顺序**取
`[t for i, t in enumerate(tasks) if i % num_shards == shard_idx]`。
不排序、不打乱、不按耗时均衡。122 条切 8 份 ⇒ **shard 0/1 各 16 条,shard 2–7 各 15 条**,
这两个数你在 `--dry_run` 里必须打印出来,我本地按规格复算比对。

**加载后的硬断言(这是 §1 的实例层确认)**:

```python
assert pipe.transformer.config.zero_cond_t is True, "zero_cond_t 没生效,停"
assert pipe.transformer.zero_cond_t is True,        "zero_cond_t 没传到实例,停"
```

不满足直接 `SystemExit` 并打印两个值。**不许加开关绕过它**——绕过去跑出来的 122 张图
不是 2511 的能力,是另一个模型的能力。

**其余硬要求(都是既有纪律,不是新加的)**:

- **断点续跑**:产物已存在**且 `Image.open(p).load()` 不抛异常**才跳过。
  只 `Image.open` 不行——它是惰性的,一张截断到一半的 PNG 照样通过并报出正确尺寸
  (`DISTILL_PLAN.md` §3 有实测)。而被 kill 时写到一半的图正是这个场景。
- **逐样本 try/except**:失败**当场打印**(不要攒到最后),条目**保留**在 records 里
  只记 `error`,**不许从任务表里删掉**。一个坏样本不许杀掉整个 shard。
- **每 shard 写 `results_shard{idx}.json`**,记录字段与 Q1 相同
  (`task_id / n_refs / seed / prompt / image_paths / elapsed_s / peak_mem_gb / error`),
  另加 `shard_idx`。
- **`--merge`**:读全部 `results_shard*.json`,按 task_id 在任务表里的**原顺序**排序,
  合并成 `results.json`;`meta` 里除 Q1 那些字段外另记 `n_shards`、各 shard 的 `total_s`、
  `n_fail` 汇总。
- **拼图只在 `--merge` 里出**,shard 进程不出图(8 个进程同时写同一张图是灾难)。
  复用 `multibanana_eval/board.py`:`board.build_row(task_id, prompt, refs, {"qwen2511": gen})`
  + `board.stack_board`,单文件超 2 MB 就按 8 行分批成 `ALL_COMPARISON_partNN.png`。
  **只 import,不改 board.py。**
- **启动自检行**:第一秒就打印 `全量 122 | 本 shard N 条 | 待跑 M | 已跳过 K | 输出目录 | 权重路径`。
  目的是"任务数不对"这种致命错误在第一秒暴露,而不是跑完才发现。

### 3.4 提交命令

先在本地(或任意 CPU 机器)过 `--dry_run` 门禁,再投这一条。

```bash
infer_submit --owner wuwenxuan --project default --cluster h --gpus 8 --timeout 60 \
  --repo https://github.com/wenshare71/UNO.git \
  --commit <你 push 后的 40 位 commit> \
  --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --output-dir /kaimm-distill/wuwenxuan/output/qwen_3ref \
  --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
  --label qwen2511_3ref_122 \
  --prep-cmd 'true' \
  --prep-marker /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --cmd 'E=/kaimm-distill/wuwenxuan/envs/qwen-edit; SP=$E/lib/python3.11/site-packages; export LD_LIBRARY_PATH=$E/lib:$SP/torch/lib:$(echo $SP/nvidia/*/lib | tr " " :):$LD_LIBRARY_PATH; for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i QWEN_WEIGHTS=$INFER_WEIGHTS_DIR $E/bin/python qwen/infer_qwen_3ref.py --shard_idx $i --num_shards 8 --out $INFER_OUTPUT_DIR > $INFER_OUTPUT_DIR/shard$i.log 2>&1 & sleep 20; done; wait; QWEN_WEIGHTS=$INFER_WEIGHTS_DIR $E/bin/python qwen/infer_qwen_3ref.py --merge --out $INFER_OUTPUT_DIR'
```

四处照抄 Q1 的教训(`reports/20260811-0951-q1-env-glitch/REPORT.md`),不要改:

- `--cmd` 用**单引号**,`$INFER_*` 要留给远程展开;
- `LD_LIBRARY_PATH` 自己导出,别赌 worker 注入;
- `--output-dir` 必须显式给,默认值会往权重目录里写;
- `--prep-cmd 'true'` + `--prep-marker` 指向已存在的权重目录 ⇒ 跳过准备阶段。

**`sleep 20` 是有意的**:8 个进程同时读 57.7 GB 权重会把宿主内存打爆。
错开 20 秒启动不影响总时长(瓶颈在采样不在加载),但能避开这个坑。
显存上单卡 20B bf16 ≈ 40 GB + VL 7B ≈ 14 GB + VAE,80 GB 的 H 卡放得下一个进程。

**timeout 60 的算法**:3-ref 序列比 2-ref 长(img token 4096+3×4096 = 16384,
2-ref 是 12288),注意力是平方项,估 1.6–1.8× 于 Q1 的单张耗时。
每 shard 16 条 ⇒ 估 15–20 分钟 + 加载。给 60 分钟有余量,不够时队列会按 GPU 利用率
自动延长(最多 3×)。**Q1 的 `results.json` 取回后,用它的实测 `total_s / n_run` 重算一遍
再投**,别用我这个估计。

---

## 4. 三道门禁

**每道停下来回报,等我确认再往下。**

| 门 | 做什么 | 我要看什么 |
|---|---|---|
| **G1** | §1 的必查 | 三行输出原样 |
| **G2** | `--dry_run` 跑 8 次(`--shard_idx 0..7 --num_shards 8`) | 8 个分片的条数(应为 16,16,15,15,15,15,15,15)、合并后 122 条、拼图分了几张 |
| **G3** | 正式 job 跑完 | `results.json` 的 `meta`、`n_fail`、s/img、peak_mem、拼图 |

G2 我会在本地按 `i % 8 == shard_idx` 独立复算并逐条 diff `task_id` 列表——
**这是黄档放权的依据,不是形式主义。**

---

## 5. 交付物

```
output/qwen_baseline/results.json                     ← ② 取回
output/qwen_baseline/ALL_COMPARISON*.png              ← ② 取回
output/qwen_3ref/results.json                         ← ③ 产出
output/qwen_3ref/results_shard*.json                  ← ③ 产出
output/qwen_3ref/ALL_COMPARISON*.png                  ← ③ 产出
qwen/infer_qwen_3ref.py                               ← ③ 你新写的
reports/20260812-q1b-3ref/REPORT.md                   ← ②③ 你写的
```

`.gitignore` 已经为上面这些路径开好白名单(仓库根 `.gitignore` 末尾的 Qwen 段),
**不要改 `.gitignore`**。全分辨率单图不放行,留在共享盘。

拼图可以进 git 的理由:这两批**只有一个变体**(`qwen2511`),不存在 A/B,
不受"判读完成前不得生成带变体名的拼图"那条盲评纪律的约束。

---

## 6. 红线

- **既有 `.py` / `.sh` 一个字不许改**(R0)。`scripts/infer_qwen_edit.py`、
  `multibanana_eval/board.py`、`distill/**` 全部只读。要复用就 import。
- **不许改任何常量**:steps 40 / true_cfg 4.0 / 1024² / negative_prompt `" "` /
  prompt 原样 / seed 取任务自带。这些与 Q1 逐字相同,改了两批就不可比。
- **不许绕过 `zero_cond_t` 断言。**
- **不许自己枚举任务**。任务表在 git 里,你只跑它。
- **不许写结论。** 你交数字、日志、图和现象;"3-ref 行不行"是我判。
- **规格自相矛盾时:报告 + 按优先级执行,禁止沉默修复**(手册 §2.0 黄档义务)。
- 规格没写到的语义决定一律红档——**规格留白 ≠ 授权你填空**,那说明我漏了,该问我。
