# M4 评测集构建与推理 —— 实现规格(交给 H800 agent)

> **版本** v1(2026-07-30) · **上游** `distill/DISTILL_PLAN.md` §6 · **执行者** H800 上的 agent
>
> 本文档是**实现规格**,不是研究计划。计划回答"为什么这么测",本文档回答"写成什么样"。

---

## 0. 授权:本文档是手册 §2.0 **黄档**的规格文档

> **修订 2026-07-30**:本节原先写的是"凌驾于手册 §2.0 的一次性解除"——那是因为当时
> 手册 §2.0 还是"任何 `.py` 都不许你写"的一刀切。手册已改为**三档**,本文档因此
> 不再是例外,而是**黄档的标准入口**。

`distill/REMOTE_AGENT_HANDBOOK.md` §2.0 黄档的三条准入,本文档逐条满足:

| 准入条件 | 本文档如何满足 |
|---|---|
| ① 规格把实验语义写死到没有自由度 | §2 不可改清单 + §3.2 的组合枚举规则 / seed 公式 / 视角选取规则 / 变体配置 |
| ② 产物本地可机械复核 | 评测集完全确定性,本地可按 §3.2 重算一遍逐条 diff |
| ③ 规格里有阶段门禁 | §3.5 之后的 Stage A 门禁 |

**授权范围仅限以下两个新建文件**:

- `distill/build_eval_json.py`
- `distill/eval_multiref.py`

**边界不变,继续按手册执行**:

- 除上述两个文件外,**任何既有 `.py` / `.sh` 一律不改**(R0)。需要改 → 按手册 §3 出诊断包上报;
- 手册 §0 铁律不变:**你可以修"让代码跑起来"的问题,不可以修"改变实验含义"的问题**。
  本文档 §2 是"实验含义"的完整清单,清单内任何一项与你的判断冲突时,**停下上报,不要自己定夺**;
- **规格留白 ≠ 授权你填空**(R13):本文档没写到的语义决定,说明我漏了,问我;
- 本文档自相矛盾时:**报告 + 按优先级执行,禁止沉默修复**(黄档义务)。
  [已发生] Stage A 就命中过一次——§3.3 的 JSON 示例与 §3.2 伪代码枚举出的 k=0 组合对不上,
  正确处理是按 §3.2(它在 §2.5 不可改清单的保护范围内)执行并上报;
- 绿灯 G1–G6 照旧。

---

## 1. 背景:为什么评测集要这么切

M3 蒸馏已完成(4000 步),M4 冒烟两轮已跑完。**本地已逐张读过那 39 张图**,结论修正如下,
本文档的评测集设计直接由这些观察推出:

1. **in-distribution(dreambench held-out 2-ref)蒸馏有效但不均匀**:5 个 case 里 POST 4 胜 1 败,
   不是冒烟报告说的"5 个全胜"。
2. **发现新失败模式:主体复制**。case02 生成了**两个闹钟、蜡烛消失**;multibanana add_143 生成了
   **两个茶壶**。推测机制:模型学到了"画面里要有 N 个物体"的**数量先验**,但没学到
   "第 j 个槽位绑定第 j 张 ref"的**绑定关系**;隔离注意力下 ref 互不可见,结构上没有任何东西
   强制两个槽位产出不同的东西,于是复制强主体来凑数。2-ref 被 oversample 到每条看 ~11.3 次,
   会强化数量先验。**这是可证伪的:若 ckpt-2000 复制率低于 ckpt-4000,就是过训产物。**
3. **PRE 会画面级崩坏**(add_092 图裂成上下两半、add_143 变成火场、case02 泡沫雪堆),
   **POST 39 张里一张都没有**。这是比"丢不丢主体"更基础的稳定性提升,值得单独守住。
4. **OOD(multibanana add)pre/post 差异很大**,不是冒烟报告说的"几乎无差别":POST 4 胜 2 败 1 平 1 混。

由此确定五层分工(§3.2)。核心取舍:**人判是稀缺资源,每一条样本必须回答一个明确问题**。

---

## 2. 不可改清单(改动 = 实验作废)

以下任何一项与你的判断冲突,**停下上报,不要自行调整**。

### 2.1 数据划分

- **HELD-OUT 10 个**(评测唯一来源):
  `backpack_dog, bear_plushie, berry_bowl, can, candle, clock, colorful_sneaker, duck_toy, fancy_boot, grey_sloth_plushie`
- **TRAIN 20 个**(蒸馏数据来源,**严禁出现在评测集**):
  `backpack, cat, cat2, dog, dog2, dog3, dog5, dog6, dog7, dog8, monster_toy, pink_sunglasses, poop_emoji, rc_car, red_cartoon, robot_toy, shiny_sneaker, teapot, vase, wolf_plushie`
- 两个名单写成**模块级常量 + 启动断言**;评测集里出现任一 TRAIN subject → `sys.exit`。

### 2.2 推理配置(与冒烟两轮完全一致,否则不可比)

```
model_type = flux-dev        # bf16
offload    = False
width = height = 512
ref_size   = 512
num_steps  = 25
guidance   = 4.0
lora_rank  = 512
pe         = "d"
```

### 2.3 变体定义

沿用 `distill/smoke_eval.py:64` 的五元组格式 `(标签, 用我们的LoRA?, ref_isolation, kv_cache, LoRA bank)`:

| 标签 | use_ours | ref_isolation | kv_cache | bank | 权重 |
|---|---|---|---|---|---|
| `official_full` | False | False | False | `official` | 官方 UNO dit_lora(模型加载时自带,备份下来) |
| `ours_kv_pre` | True | True | True | `pre` | `log/ref_isolation/checkpoint-20000/dit_lora.safetensors` |
| `ours_kv_post4000` | True | True | True | `post4000` | `log/ref_distill/checkpoint-4000/dit_lora.safetensors` |
| `ours_kv_post2000` | True | True | True | `post2000` | `log/ref_distill/checkpoint-2000/dit_lora.safetensors` |

**`ours_kv_post2000` 只用于 S2 分层**,不进其他分层——它存在的唯一目的是回答
"复制是不是过训产物",全量跑它会让人工判读量凭空涨 1/3。

### 2.4 场景模板

用 `distill/gen_data.py:132 load_scene_templates()` 读出的 **object 20 条**。
**held-out 10 个 subject 全部是物体(无活体动物)**,所以 20 条全部可用,不受"含动物只能用 10 条"限制。
**严禁使用属性/换装模板**(object 21–25、live 11–25)——隔离注意力结构上学不了属性迁移,
测它是在测已知的结构边界,不是测蒸馏。

### 2.5 评测集构成(条数与规则)

见 §3.2 表格。**分层名称、任务数、seed 公式、视角选取规则全部写死,不得调整。**

---

## 3. Stage A —— `distill/build_eval_json.py`

**纯 CPU,不碰 GPU,秒级完成。** 这是烧 GPU 之前唯一能验证评测集对不对的地方。

### 3.1 输入

- `datasets/dreambooth/dataset/` —— subject 图与 `prompts_and_classes.txt`
- 可复用(**import,不要重写**):
  - `distill.gen_data.load_scene_templates(data_dir)` → `(shared_10, object_20)`
  - `distill.gen_data.load_subject_images(data_dir, subjects)` → `{subject: [排序后的文件名]}`
  - `distill.gen_data.make_prompt(class_tokens, suffix)` → `a X and a Y in the jungle`(已支持 n=1)
  - 类别表解析:抄 `gen_data.py:92` 那段(只取 `Classes` 段的 `a,b` 行)

### 3.2 五层构成

| 层 | 内容 | 组合 × 场景 × seed | 任务 | 变体数 | 图数 |
|---|---|---|---|---|---|
| **S0** 锚点 | 复刻冒烟的 5 个 case | 5 × 固定 × 1 | 5 | 3 | 15 |
| **S1** 主验收 | held-out 全部 44 个合法 2-组合 | 44 × 1(轮转) × 3 | 132 | 3 | 396 |
| **S2** 复制探针 | `bear_plushie + grey_sloth_plushie` | 1 × 5 × 3 | 15 | **4** | 60 |
| **S3** 单 ref 回归 | held-out 10 个 subject 各自 1-ref | 10 × 3 × 2 | 60 | 3 | 180 |
| **S4** 3-ref 诊断 | held-out 10 个 3-组合 | 10 × 1 × 2 | 20 | 3 | 60 |
| | | | **232** | | **711** |

**S5(multibanana OOD)已有产物**(`output/multibanana_eval_distill/`,24 张),
**本次不重跑**,分析时直接沿用。

#### S0 锚点(5 条)—— 这是 Stage B 的回归测试

**逐字复制 `distill/smoke_eval.py:43-59` 的 `DEFAULT_CASES`**(prompt 与 ref 文件名一字不改),
`seed = 3407`。

它的作用不是评测,是**验证新脚本没写错**:同 prompt、同 ref、同 seed、同配置,
`eval_multiref.py` 生成的图必须与 `output/smoke_eval/case0X__*.png` 一致(判据见 §5.1)。

#### S1 主验收(132 条)

```
combos = 按字典序排列的 44 个合法 2-组合
         # C(10,2)=45,减去唯一的同 class 对 (bear_plushie, grey_sloth_plushie)
for k, (a, b) in enumerate(combos):          # k = 0..43
    template = object_templates[k % 20]      # 场景轮转,不与组合绑定
    for s in (0, 1, 2):                      # seed slot
        seed     = 3_500_000 + k * 10 + s
        view[a]  = views[a][(s + 0) % len(views[a])]
        view[b]  = views[b][(s + 1) % len(views[b])]
        prompt   = make_prompt([class[a], class[b]], template)
```

**为什么用满 44 个组合而不是计划 §6 的"20 组合 × 2 场景"**:同样成本下主体对覆盖翻倍,
而读图显示失败与**具体组合**强相关(candle+clock 复制、bowl+can 丢罐),与场景基本无关。

**为什么场景轮转**:原案"每组合固定 2 个场景"会让全部任务挤在 2 种场景里,
且场景与组合绑死——某个场景恰好难就会污染整体结论。

**为什么 `view = (s + j) % n`**:seed 顺带白嫖视角覆盖,零额外成本;`+j` 避免两个主体永远同号视角。
[已验证] held-out 每个 subject 有 4–6 个视角,3 个 seed 取到 3 个不同视角处处成立。

#### S2 复制探针(15 条)

```
pair = ("bear_plushie", "grey_sloth_plushie")   # 唯一的同 class 对,两者 class 都是 "stuffed animal"
for t in range(5):                               # object_templates[0..4]
    for s in (0, 1, 2):
        seed = 3_600_000 + t * 10 + s
        视角规则同 S1
prompt 形如 "a stuffed animal and a stuffed animal in the jungle"
variants 含 ours_kv_post2000(本层独有)
```

**为什么专门捞这一对**:它是训练数据规则 §2「同 class 组合跳过」**排除掉**的那一对,
正因如此是"槽位绑定"的最强压力测试——两个 ref 同类,模型若只学了数量先验就会画两只一样的。

**这一层不进 S1 主指标**,单列报告。它违反训练数据的构造规则,拿它算主验收对 student 不公平。

#### S3 单 ref 回归(60 条)

```
for i, subj in enumerate(sorted(HELD_OUT)):      # i = 0..9
    for c in range(3):                            # 3 个场景
        template = object_templates[(i * 3 + c) % 20]
        for s in (0, 1):
            seed = 3_700_000 + i * 100 + c * 10 + s
            view = views[subj][(c * 2 + s) % len(views[subj])]
            prompt = make_prompt([class[subj]], template)
```

**为什么必须有这层**(计划 §6 没有,是本次补上的):训练混了 40% 多 ref、跑了 4000 步,
**至今没有任何人验证过单 ref 有没有退化**。60% 单 ref 的存在意义就是防遗忘,
但"防住了"目前是**假设不是事实**。若 2-ref 变好而 1-ref 变差,模型整体是退步的。
本层验收标准是**不劣于 PRE**,不是"要变好"。

#### S4 3-ref 诊断(20 条)

```
c3   = 112 个无同类 3-组合(C(10,3)=120 减去含 stuffed animal 对的 8 个)
选法 = 字典序遍历 + 贪心:每次选"能最大提升当前最小 subject 覆盖数"的组合,
       平局取字典序小者,直到选满 10 个
断言 = 每个 held-out subject 覆盖 ≥2,并打印实际覆盖分布
       # [已验证] 正确的贪心实现会得到**每个 subject 恰好覆盖 3 次**(10 组 × 3 = 30 = 10 × 3)。
       # 断言留 ≥2 是给实现差异留余量;但若你算出来 <3,大概率你的贪心与参考实现不同,值得先核对。
for k, combo in enumerate(picked):                # k = 0..9
    template = object_templates[k % 20]
    for s in (0, 1):
        seed = 3_800_000 + k * 10 + s
        view[combo[j]] = views[...][(s + j) % n]
```

**本层不进验收,只做诊断**:训练里 3-ref 只通过 138 条(3.5%),
且根因是 **teacher 自己在 3-ref 上就系统性不行**——拿一把 teacher 都不及格的尺子量 student
量不出东西。它唯一的价值是回答"3-ref 有没有**比 PRE 更差**"
(138 条被 oversample 8.6×、每条看 ~18.5 次,过拟合风险最高)。

### 3.3 输出

写到 **`datasets/eval_multiref/eval_set.json`**。
路径约定与 M1 manifest 一致:`image_paths` 相对于 **json 所在目录**,即 `../dreambooth/dataset/...`
(这样 `image_root = dirname(json_file)` 可直接解析)。

```json
{
  "meta": {"spec": "M4-eval-v1", "n_tasks": 232, "n_images": 711},
  "tasks": [
    {
      "task_id": "S1_000_s0",
      "stratum": "S1",
      "prompt": "a backpack and a bowl in the jungle",
      "image_paths": ["../dreambooth/dataset/backpack_dog/00.jpg",
                      "../dreambooth/dataset/berry_bowl/01.jpg"],
      "seed": 3500000,
      "variants": ["official_full", "ours_kv_pre", "ours_kv_post4000"],
      "meta": {"subjects": ["backpack_dog", "berry_bowl"],
               "classes": ["backpack", "bowl"],
               "n_refs": 2, "template_id": 0, "template": "in the jungle",
               "views": ["00.jpg", "01.jpg"], "seed_slot": 0}
    }
  ]
}
```

`task_id` 命名 `{层}_{组合序号:03d}_s{seed槽}`,**全局唯一**,直接用作输出文件名前缀。

`variants` **写进每个任务**——组合逻辑集中在 build 脚本一处,`eval_multiref.py` 只管照单执行,
不再自己判断哪层跑几个变体。

### 3.4 启动断言(缺一不可)

1. 评测集里出现任一 TRAIN subject → `sys.exit`
2. `task_id` 有重复 → `sys.exit`
3. 每个 `image_paths` 指向的文件**实际存在** → 否则 `sys.exit`
4. 任务总数 == 232、图数 == 711 → 否则 `sys.exit`(数字对不上说明枚举规则写错了)
5. S1 恰好 44 个不重复组合、S4 覆盖断言(§3.2)

### 3.5 `--dry_run` 统计表(**Stage A 的交付物**)

参考 `gen_data.py:413 print_stats()` 的风格,打印:

- 每层任务数 / 图数,与 §3.2 表格逐格对照
- S1 的 44 个组合列表 + 各自分到的场景模板
- 每个 held-out subject 在各层的出现次数
- 场景模板使用直方图
- S4 选中的 10 个 3-组合 + 覆盖分布
- seed 区间(应为 **3_500_000–3_800_091**,与 M1 的 3_407_000–3_415_999 **不重叠**)

### ⛔ Stage A 到此为止 —— 门禁

**把统计表整段贴进 `reports/M4_stageA.md` 并回传,等用户确认后再进 Stage B。**

**不要自作主张开始生成图。** 评测集配比错了却已经烧掉 711 张图,是本阶段唯一真正贵的错误
——Stage A 秒级可重跑,Stage B 要几十分钟 GPU 且人工判读全部作废。

---

## 4. Stage B —— `distill/eval_multiref.py`

### 4.1 直接抄的部分(不要重新发明)

| 要的东西 | 抄哪里 |
|---|---|
| LoRA bank 建立(官方权重备份 + ckpt 加载 + **key 硬校验**) | `distill/smoke_eval.py:160-195` |
| `swap_lora()` | `distill/smoke_eval.py:110` |
| **变体外层循环**(切 LoRA 只搬一次)+ warmup 标志 | `distill/smoke_eval.py:196-240` |
| 拼图 | `multibanana_eval/board.py:101 build_row` / `:136 stack_board` |
| `--shard_idx/--num_shards` 切分 | `distill/gen_data.py` 的 sharding |

**本脚本比 `smoke_eval.py` 多出来的只有四件事**:LoRA bank 从 3 个变 4 个、
任务从硬编码变成读 json、变体按任务的 `variants` 字段过滤、加 sharding 与断点续跑。

### 4.2 CLI

```
--eval_json     datasets/eval_multiref/eval_set.json
--pre_lora      log/ref_isolation/checkpoint-20000/dit_lora.safetensors
--post4000_lora log/ref_distill/checkpoint-4000/dit_lora.safetensors
--post2000_lora log/ref_distill/checkpoint-2000/dit_lora.safetensors
--save_path     output/eval_multiref
--shard_idx N --num_shards M
--strata S0,S1,...        # 可选,只跑指定层,便于分批
--dry_run
```

推理超参**全部按 §2.2 写成 default,不要暴露成需要人填的必选参数**——
默认值就是实验配置,少一个手填就少一个跑歪的机会。

### 4.3 输出布局

```
output/eval_multiref/
  {task_id}__{variant}.png            # 711 张全分辨率
  results_shard{i}.json               # 每 shard 一份
  results.json                        # --merge 合并
  boards/{stratum}_{chunk:02d}.jpg    # 拼图,JPEG q90
```

**拼图必须切块**:每块 ≤12 个任务(一任务一行,行内 = refs + 各变体)。
S1 有 132 个任务,拼成一张会得到一个没法看也没法提交的巨图。

**拼图存 JPEG 不存 PNG**:711 张全分辨率 PNG ≈ 250 MB,进 git 太重。
全分辨率图**留在 H800** 供 viewer 人工判读;进 git 的只有 `eval_set.json`、`results.json`
和 JPEG 拼图。

### 4.4 硬性工程要求(全部来自 `DISTILL_PLAN` §3「运行方式」的既有教训)

1. **断点续跑必须用 `im.load()` 校验**,不能只 `PIL.Image.open`。
   [已验证] `Image.open` 是惰性的、只读文件头,截断到一半的图照样通过并报出正确尺寸,
   只有 `.load()` 才抛 `OSError: image file is truncated`。shard 被杀时写到一半的图正是这个场景。
2. **逐样本 try/except**,失败记日志继续,不许一个坏样本杀掉整个 shard。
3. **同一变体内 LoRA 只搬一次**(变体做外层循环)。模型加载 ~7 min,搬来搬去会主导总耗时。
4. **warmup 一次**再开始计时(`smoke_eval.py` 已有 `warmed` 标志)。
5. `results.json` 记录每张图的 `denoise_s`、每变体的 `mean_s / median_s / peak_mem_gb`
   ——用来确认**蒸馏没把 KV cache 加速打坏**(冒烟基线:1.72–1.77x,显存 37 GB)。

### 4.5 成本预算(用来判断你有没有跑歪,别为此过度设计)

冒烟实测:`official_full` ~5.1 s/张,`ours_kv_*` ~2.9 s/张(25 步 512×512)。

- 纯 denoise:232 张全注意力 × 5.1 s + 479 张 KV × 2.9 s ≈ **43 min(单卡)**
- 模型加载 ~7 min/进程
- **8 卡分片:每片 ≈ 7 min 加载 + 6 min 生成 ≈ 15 min 墙钟**
- 单卡跑完也就 ~50 min,**分片失败别硬扛,退回单卡是可接受的**

**这个量级很小。** 如果你发现要跑几小时,说明哪里错了(最可能:LoRA 反复搬、或没做变体外层循环)
——停下上报,别等它跑完。

---

## 5. 验收标准

### 5.1 Stage B 自检:S0 锚点必须复现冒烟结果

`eval_multiref.py` 跑完 S0 后,把 5 张 `S0_*__ours_kv_post4000.png` 与
`output/smoke_eval/case0X__ours_kv_post.png` 逐像素比:

```python
import numpy as np
from PIL import Image
a = np.asarray(Image.open(new).convert("RGB"), dtype=np.int16)
b = np.asarray(Image.open(old).convert("RGB"), dtype=np.int16)
print(f"max={np.abs(a-b).max()} mean={np.abs(a-b).mean():.4f}")
```

- **max ≤ 2**:通过(bf16 非确定性的正常抖动)
- **max 明显更大**:说明 seed / ref 路径 / 变体配置 / 预处理有一处对不上。
  **停下上报,不要继续跑 S1–S4** ——尺子没校准就量,711 张图全部白烧。

参照量级:`bench_kv_cache.py` 实测 iso vs kv 的像素差 max=58 / mean=0.4681 属于**不同实现路径**的差异;
S0 走的是**完全相同**的路径,应当远小于它。

### 5.2 交付清单

进 git(体积可控):

- `distill/build_eval_json.py`、`distill/eval_multiref.py`
- `datasets/eval_multiref/eval_set.json`
- `output/eval_multiref/results.json`
- `output/eval_multiref/boards/*.jpg`
- `reports/M4_stageA.md`、`reports/M4_stageB.md`

留在 H800(体积大,供 viewer 人工判读):

- `output/eval_multiref/*.png` 全部 711 张

**`.gitignore` 需要相应加白名单**——照 `.gitignore:214-218` 已有的
`multibanana_eval_distill` 写法(先 ignore 目录再 `!` 放行指定文件)依样画葫芦。
这一条属于绿灯,自己做。

### 5.3 `reports/M4_stageB.md` 回传内容

**纯文本、能被整段复制、自带上下文**(手册 §3.5)。至少包含:

1. 实际生成图数 / 跳过数 / 失败数(附失败样本的 task_id 与 traceback 摘要)
2. §5.1 锚点复现的 max/mean 数字(5 个 case 逐个列)
3. 每变体 `mean_s / median_s / peak_mem_gb` 表,以及 vs teacher 的加速比
   ——对照冒烟基线 1.72–1.77x / 37 GB,**明显偏离要说明**
4. 各层实际跑了多少任务、多少图
5. 异常现象**原样描述**,不要替我下结论。特别注意并单独报告:
   - 是否出现 PRE 那种**画面级崩坏**(图裂成两半、内容与 prompt 完全无关)
   - S2 里 post2000 与 post4000 **肉眼看复制现象是否有差别**(只描述现象,不做判断)

---

## 6. 明确不在本次范围内

- **人工判读协议**(判据、打分尺度、盲法、判读顺序)—— 后续单独定,本次只交付图与拼图
- **v2 自动度量打分**(GroundingDino + DINO 裁剪)—— 本次不跑。
  注:`filter_data.py:214 _ref_vecs()` 硬绑 dreambooth 目录、`:207 _class_word()` 硬绑
  `prompts_and_classes.txt`,所以它**只能给 S0–S4 打分,给不了 multibanana**
- **multibanana 重跑** —— 已有 24 张产物,沿用
- **任何训练侧改动** —— 本阶段只评测,不训练

---

## 7. 卡住了怎么办

按手册 §3 出诊断包。本文档特有的两个高频歧义,**先看这里再上报**:

| 情形 | 怎么办 |
|---|---|
| §3.2 的任务数算出来对不上 232 / 711 | **不要调规则去凑数**。[已验证] 这两个数字连同 seed 区间、`task_id` 唯一性、S4 覆盖分布,已在本地按 §3.2 规则完整枚举验算过。所以对不上时**先怀疑你的枚举规则与本文档有出入**;把各层实际条数贴出来上报,不要动规则 |
| S4 贪心选法覆盖断言过不了 | 打印实际覆盖分布并上报,**不要放宽断言继续跑** |
| §5.1 锚点复现 max 超标 | **立刻停,不要跑 S1–S4**。附 5 个 case 的 max/mean、以及你实际用的 seed / ref 绝对路径 / 变体五元组 |
| 显存不够 / OOM | 绿灯:降 `--num_shards`、退回单卡。**不许降分辨率或 num_steps**(那是 §2.2 实验语义) |

---

## 8. M5 人工判读判据(**预登记,写死后不许再改**)

> 立此节的日期:2026-08-03。依据:`DISTILL_PLAN.md` §11.3 步骤 2 的 R2 批次
> (198 条,`blind_annotations_m5r1.json`)测出的噪声地板与自洽率。
> §6「明确不在本次范围内」里那条"人工判读协议后续单独定"到此兑现。
>
> **为什么必须先写死再用**:D02「份额失衡比」栽过——判据事后调整,结论就没有可信度。
> 本节的数值在看到任何一次 `student vs teacher` 结果**之前**定下。

### 8.1 主指标

| | 口径 |
|---|---|
| **主报** | 非平局胜率 `S/(S+T)` + **Wilson 95% CI**(z=1.959963984540054) |
| **平局率** | 单列,**不并进主指标** |
| `(S+B)/(T+B)` | 仍算,**只作与 M4 的可比参考,不作判据** |
| 实现 | `distill/blind_eval/report.py`,服务器与离线复算共用同一份 |

### 8.2 达标判据

```
达标 ⟺  非平局胜率的 Wilson 95% CI 下界 ≥ 0.40
        且  非平局样本数 n_nontie ≥ 94
```

**这是非劣性判据,不是优越性判据。** 目标是"学生与 teacher 无实质差距",
对应的零假设是 50%,`0.40` 是**非劣性容限**(允许 10 个百分点的劣势)。
把 X 写成 0.95 那种数是口径错误——0.95 属于旧的 `(S+B)/(T+B)` 尺度,
在非平局胜率上 0.95 意味着"学生几乎场场胜过 teacher",那不是目标。

#### X = 0.40 是怎么定出来的

**约束一:判据必须是一个真·等价的模型能过的。** 否则它不是判据,是摆设。
用 p̂=0.50(完美等价的观测)反推各 X 所需的最小样本:

| X | 所需 n_nontie | 折合总条数(平局率 30%) | 可行性 |
|---|---|---|---|
| 0.30 | 22 | 31 | 太松,见下 |
| 0.35 | 40 | 57 | 太松 |
| **0.40** | **94** | **134** | **≈ 本次 R2 的 post_vs_pre 规模(148 条)** |
| 0.42 | 148 | 211 | 判读量翻倍 |
| 0.45 | 382 | 546 | 人力不可行 |

**约束二:判据必须挡得住已知不达标的模型。** 在 n_nontie=106(本批实际规模)下:

| 真实水平 | CI | X=0.30 | X=0.35 | X=0.40 |
|---|---|---|---|---|
| 0.500(等价) | [0.406, 0.594] | 过 | 过 | **过** |
| 0.450 | [0.361, 0.548] | 过 | 过 | **不过** |
| 0.400 | [0.308, 0.491] | 过 | 不过 | **不过** |
| 0.379(M4 实测 post4000) | [0.291, 0.472] | 不过 | 不过 | **不过** |

X=0.35 会放过一个真实劣 10 个点的模型;X=0.40 在"等价能过、劣 5 个点就不过"
之间,且所需判读量正好落在已经验证可行的批次规模上。

#### `n_nontie ≥ 94` 是判据的一部分,不是建议

样本不足时结论是**「判据不适用」,不是「没达标」**。这条不是形式主义:
R2 的零假设对(30 条 / 20 条非平局)实测 45.0%,CI **[0.258, 0.658]**——
**一个构造上完全等价的比较,在 n=30 下自己就过不了任何候选 X**。
n_nontie=20 时即使观测正好 0.50,CI 下界也只有 0.299。
报告里若出现 n_nontie < 94 的"未达标",那是尺子不够长,不是模型不行。

### 8.3 分层与聚类

- **门控只看总体**;S1 / S3 分层必报,但单层不作判据(M4 的 S3 单层
  CI [0.211, 0.531] 含 0.5,据此动混比就是过度解读)。
- **S1 必须同时报组合级聚类后的版本**:S1 是 44 组合 × 多 seed,
  132 条不是 132 个独立样本。每批**实测** ICC 与设计效应
  `deff = 1 + (m₀−1)·ICC`,报 `n_eff = n/deff` 下的 CI。
  > [2026-08-03 实测] M4 的 S1 非平局 32/87、40 组合:**ICC = −0.041,deff = 1.00**,
  > 聚类后 CI 与朴素 CI 到小数点后三位完全相同 [0.274, 0.473]。
  > **组合内相关实际上不存在。** 但这条要求保留——它必须每批**被测量**,
  > 不能因为测过一次是 0 就以后假设它是 0。

### 8.4 平局率的读法

平局率是**判读分辨率的指标**,不是模型指标。参照点是零假设对:

> [2026-08-03 实测] 零假设对(teacher 两次跑,仅噪声不同)平局率 **33.3%**。
> 即:构造上完全等价的两张图,只有三分之一被打平。

因此「某组平局率 30%」**不能**读成"三成情况下两个模型等价"——
在这把尺子上,等价的东西也只有三成会被判成等价。平局率**只用于**:
(a) 换算 n_nontie;(b) 与零假设对的 33.3% 横向对比。

### 8.5 必须随结论一起声明的局限

1. **单标注者。** 无标注者间一致性,只有自洽率。
2. **指纹可识别。** 「主体复制 = 学生」「画面崩坏 = pre」这类签名是已知的,
   左右盲化挡不住指纹识别。凡是缺陷极端可辨的比较(如 R2 的 post_vs_pre S1 = 62:0),
   报告须写明该数字含"认出是哪个模型"的成分,不是纯偏好强度。
3. **跨批次的尺子会漂。** [2026-08-03 实测] 20 条重放的自洽率 **55.0%**,
   Cohen **κ = 0.274**;硬翻转(T↔S 不沾平局)仅 1/20 = 5%,
   其余 8/20 全发生在平局边界上;9 次改判中 7 次朝 teacher 移动、
   **0 次朝学生移动**(符号检验 p≈0.18,n=20 不显著,但方向干净)。
   ⇒ **不同批次的非平局胜率不得直接并排引用**;跨批次比较必须走重放对校正,
   或干脆在同一批内完成。批内比较不受此影响。
4. **零假设对同时含"换 seed"与"换 run"两个来源。** 本栈非逐位可复现
   (§11.3 步骤 1 的锚点自检失败)。已量化:换 seed 的效应 mean 10.4–42.9,
   run 间抖动 mean 1.6–3.7,相差 3–25 倍,故零假设对成立——
   但不得表述为"纯 seed 差异"。

---

## 9. P-probe 身份留存计数(**预登记,写死后不许再改**)

> 定稿于 2026-08-03,**在任何标注开始之前**。适用对象:`output/probe_iso/` 下
> `official_full` / `official_iso` 两个变体的 384 张产物。

### 9.1 为什么不走 §8 的偏好盲评

P-probe 的产物一眼可辨:`official_iso` 没有参考主体身份,`official_full` 有。
§8.5-2 的「指纹可识别」在这里不是局部限制,是**彻底失效**——每一对都能认出谁是谁。
一个在这种条件下产生的数字会带着「192 对盲评 CI [...]」的出身被引用,
比没有数字更糟。

因此本次**不做偏好比较**,改做**单图客观计数**:对一张生成图和一个参考主体,
只问「这个主体的身份保住了吗」,答案是/否。判定不涉及"哪个更好",
所以"知道这是哪个变体"对答案的污染远小于偏好判断。

**§8.2 的判据在本次未被使用。这不等于 `official_iso` 达标或不达标,是「未评定」。**
后续任何地方引用 P-probe 时都必须带上这句——不许把 §9 的留存率换算成 §8 的胜率。

### 9.2 判定口径(标注界面顶栏常驻此原文)

**是(身份保住)**:生成图里存在一个物体,能与该参考主体认定为**同一个具体个体**
——其特有的花纹 / 配色 / 形状细节 / 文字或图标标识,**至少一项明确对应**。

**否**:只有**类别**一致(都是背包 / 都是毛绒玩具 / 都是碗),或该主体压根没出现。

边界规则(先写死,不许开标时临场解释):

1. 只对上颜色、对不上任何结构或标识 → **否**
2. 主体出现但被严重遮挡 / 过小无法辨认 → **否**(不设第三档)
3. 同一主体在图中出现多次,任一实例对上 → **是**

WHY 要有第 2 条:若给"看不清"单开一档,它会变成一个可以随手倒进去的垃圾桶,
两个变体的垃圾桶容量还不一样(`official_iso` 的主体本来就更小更糊),
结果是把差异吸收进第三档、主指标被稀释。宁可让它计入"没保住"——
方向明确、对两个变体一视同仁。

### 9.3 抽样(确定性,可复现)

- 源:`datasets/eval_multiref/probe_iso_tasks.json`(S1 132 + S3 60 = 192 条)
- 抽 **S1 21 条 + S3 9 条 = 30 条**(按 132:60 层配比),
  每层 task_id 排序后 `random.Random(20260803).sample(...)`
- 每条 × 2 变体 = **60 张图**;S1 每图问 2 个主体、S3 问 1 个 → **102 问**
- 另加 **6 个重放 item**(占 10%)测自洽率,总计 66 个 item
- 界面不显示变体名、不显示 `task_id`,item 顺序按同一随机实例打乱

**重放的摆放规则**(2026-08-03 补写,**在任何标注开始之前**):
重放的**源必须落在清单前半段、重放本身必须落在后半段**,间隔恒 ≥ 22 条,
且不许把 6 个重放堆在末尾。WHY:堆在尾部时标注者一进尾段就会发现"这些我都见过",
于是转去**回忆上次答案**而不是重新判读,自洽率被抬高成假数——
而单标注者条件下自洽率是唯一的质量控制手段,抬高它等于把这个手段废掉。
`build_pairs.py:366` 的既有纪律就是全部条目一起打散,这里对齐并加严。
该规则由 `build_items.py:check_items` 断言,不满足就 `--verify` 失败。

### 9.4 主指标与报法

两个口径**必须都报**,不许只挑一个:

| 口径 | 定义 | 为什么需要 |
|---|---|---|
| **per-subject 留存率** | 保住的主体数 / 提问总数 | 分辨率高,但同图两个主体相关,CI 偏窄 |
| **per-image 留存率** | 该图**所有**主体都保住才计 1 | 对聚类免疫,是保守读数 |

各带 Wilson 95% CI,按层(S1/S3)再分报一次。重放 item **排除在主统计之外**
(复制品不得重复计入),只用于算自洽率。

### 9.5 局限(随结论一起声明)

1. **单标注者**,同 §8.5-1。
2. **非盲。** 变体名虽已隐藏,但 `official_iso` 的输出特征使标注者实际可辨。
   缓解手段只有"判定客观化"这一条,不是消除。
3. **30/192 抽样。** 结论覆盖的是抽中的 30 条,不是全部 192 条;
   全集只有目视印象(2026-08-03 人工看过 S3_00 / S1_00 / S1_05 三块拼图,36 条)。
4. **留存率不是质量分。** 它只回答"身份在不在",不回答"像得有多好"。
   `official_full` 的留存率也不会是 100%,那部分缺口是 teacher 自身的失败率,
   不要读成本实验的噪声。
