# 多参考自蒸馏全流程方案(Stage-2 复刻)

> 本文档是完整实现规格,交给实现者(Opus)按此落地。实现前请先通读「背景与已确认事实」,
> 里面的结论都是已经用代码/实验验证过的,不要重新推翻。

## 0. 背景与已确认事实

**要解决的问题**:我们训的隔离注意力 LoRA(`log/ref_isolation/`)在多参考图生成时
**系统性丢第二主体**(smoke 与 MultiBanana 实验均确认)。

**根因已查明**(不要再花时间归因):

1. 训练数据 100% 单 ref。UNO-1M 原始标签就是 `img_path1 → img_path2` 两图配对,
   官方转换脚本 `uno/utils/filter_uno_1m_dataset.py:50` 和我们的
   `scripts/convert_uno_labels.py:75` 都只产出单元素 `image_paths`。
   模型在隔离注意力下**从未收到过"给多个 ref 段分配注意力"的梯度信号**。
2. 官方 UNO checkpoint 的多 ref 能力来自论文的 stage-2(用合成的多主体配对数据训练),
   该数据与合成管线**未开源**(`train.py:211` 的 TODO 即指此)。
3. KV 缓存本身数学精确无近似(`uno/flux/model.py:205`,ref 调制固定用 t=0),
   质量问题与缓存机制无关,不需要在缓存上做任何修改。
4. 隔离注意力的结构性边界:ref 之间互不可见。**只能学"多个独立主体各自放进场景"**,
   学不了属性迁移/风格参考/背景参考(MultiBanana 那类)。蒸馏数据必须避开后者。

**方案**:自己复刻 stage-2 —— 用官方 full-attention UNO 当 teacher,在 dreambooth
subject 库上合成多主体配对数据,混入训练修复丢主体问题。

**协作约束(必须遵守)**:
- 远程 8×4090 机器只通过 git 同步:本地写代码 → commit/push 到 origin(wenshare71 fork)
  → 远程 `git pull` 执行 → 结果 commit 回来。**不写 scp/rsync 脚本**。
- ByteDance upstream 只读,唯一推送目标是 origin。
- 大文件(生成的图片、JSON)遵循现有模式:小批结果可 commit,大批数据留在远程,
  只 commit 元数据/统计/抽样拼图。

---

## 1. Subject 切分(防评测污染,固定不可改)

30 个 subject 在 `datasets/dreambooth/dataset/`。smoke 评测
(`scripts/smoke_ref_isolation.py` 的 `load_dreambench_cases`)取
`dreambench_multiip.json` 前几组组合,涉及的 subject **必须整体扣出**:

**HELD-OUT(10 个,只准用于评测,严禁进入蒸馏训练数据)**:

```
backpack_dog, bear_plushie, berry_bowl, can, candle,
clock, colorful_sneaker, duck_toy, fancy_boot, grey_sloth_plushie
```

**TRAIN(20 个,蒸馏数据只能从这里采)**:

```
backpack, cat, cat2, dog, dog2, dog3, dog5, dog6, dog7, dog8,
monster_toy, pink_sunglasses, poop_emoji, rc_car, red_cartoon,
robot_toy, shiny_sneaker, teapot, vase, wolf_plushie
```

class 标签见 `datasets/dreambooth/dataset/prompts_and_classes.txt`。

**组合规则**:
- 跳过同 class 组合(dog×dog、cat×cat、toy×toy 等),teacher 自己都容易串身份;
- TRAIN 集里动物占 9/20(7 狗 2 猫),采样时**限制含动物的组合 ≤ 50%**,
  避免蒸馏数据被狗支配;
- 实现里把两个名单写成模块级常量并加启动断言:任何 held-out subject 出现在
  生成任务里直接 `sys.exit`。这是硬约束,不是建议。

---

## 2. 阶段一:蒸馏数据生成 `distill/gen_data.py`

### Teacher 配置

- 官方 UNO checkpoint(`bytedance-research/UNO` 的 dit_lora)+ **full attention**
  (即 smoke 里的 `official_full` 变体:不开 ref_isolation、不开 kv_cache);
- `model_type=flux-dev-fp8`、`offload`、512×512 输出、`num_steps=25`、`guidance=4`,
  与 smoke/multibanana 推理配置一致;
- 模型加载 / LoRA 挂载 / OOM 兜底(t5.cpu、expandable_segments)直接照抄
  `multibanana_eval/infer_multibanana.py` 与 `scripts/smoke_ref_isolation.py`
  已验证的写法,尤其 **swap_lora 必须硬校验 key 完全匹配**(load_state_dict
  strict=False 会静默失效,这个坑踩过)。

### 采样空间

每条样本 = (subject 组合, 每个 subject 选一张视角图, prompt 模板, seed):

- **2-ref : 3-ref ≈ 2 : 1**(2-ref 是主战场;3-ref 防止只会数到二);
- prompt 模板复用 `prompts_and_classes.txt` 的 25 个场景模板,双主体句式仿照
  `dreambench_multiip.json`(如 `a {class1} and a {class2} in the jungle`),
  3-ref 同理扩展;live subject(猫狗)用其 live 模板;
- 视角图从该 subject 的 4–6 张里随机选(记录选了哪张);
- seed 由样本索引决定(如 `seed = base_seed + idx`),**全流程确定性、可复现**;
- 目标量:**~8000 条**(约 5300 条 2-ref + 2700 条 3-ref)。组合空间
  (150+ 对 × 25 模板 × 视角选择)远大于此,采样不放回即可,无需重复。

### 运行方式(远程 8×4090)

- 单进程单卡,用 `--shard_idx/--num_shards` 切分任务列表,8 个进程各跑 1/8
  (仿照仓库现有单卡脚本模式,不引入 accelerate);
- **断点续跑**:输出文件已存在且完整就跳过该样本,脚本可反复重启;
- 每个 shard 写自己的 `manifest_shard{i}.json`,最后由 `distill/merge_manifest.py`
  (或 gen_data.py 的 `--merge` 模式)合并。

### 输出

```
datasets/distill_multiref/
  images/{idx:06d}.jpg              # teacher 生成的 target
  manifest_raw.json                 # 合并后的全量清单
```

manifest 每条(生成后即为 `FluxPairedDatasetV2` 兼容 schema + 额外元数据):

```json
{
  "image_paths": ["<ref1 相对路径>", "<ref2 相对路径>"],
  "prompt": "a toy and a teapot in the jungle",
  "image_tgt_path": "images/000123.jpg",
  "meta": {"subjects": ["monster_toy", "teapot"], "seed": 3407123,
           "template_id": 5, "n_refs": 2}
}
```

注意 `FluxPairedDatasetV2.__getitem__` 会忽略多余的 `meta` 键,无需改 dataset 代码;
路径相对于 manifest 所在目录(`image_root = dirname(json_file)`),ref 路径需要指回
`datasets/dreambooth/dataset/...`,用相对路径 `../dreambooth/dataset/...` 并实测
能被 `os.path.join(image_root, path)` 正确解析。

### 成本预估

official_full 2-ref 512 端到端 ≈ 25–30s/张,3-ref ≈ 31s。8000 张 ≈ 60 GPU 小时,
8 卡并行 **约 8 小时**(一晚)。

---

## 3. 阶段二:质量过滤 `distill/filter_data.py`

teacher 会有失败样本(丢主体、畸形),必须过滤,否则把 teacher 的失败也蒸给 student。

- 用 **DINOv2(facebook/dinov2-base)** 对每条样本算:每张 ref 图与生成图的
  embedding 余弦相似度,取 `min_ref_sim = min(各 ref 的相似度)`
  ——丢主体的样本正是 min 值低;
- 全部分数写回 manifest(`meta.dino_sims`),**先出分数分布直方图再定阈值**:
  跑一个 `--calibrate` 模式,输出分位数表 + 按分数排序抽 40 张拼图
  (复用 `multibanana_eval/board.py`),人眼定阈值后再 `--threshold X` 出
  `manifest_filtered.json`;
- 预期通过率 60–80%;若明显更低,先人工看拼图找系统性原因,不要盲目降阈值;
- 单卡 GPU 即可,8000 张 DINO 推理 < 半小时。

产出:`datasets/distill_multiref/manifest_filtered.json`(训练直接用)。

---

## 4. 阶段三:混合训练

### 数据混合 `distill/build_train_json.py`

生成 `datasets/distill_multiref/train_mixed.json`:

- **UNO-1M 单 ref 部分**:重新过滤,加上官方配方 `vlm_filter_cot.score_final >= 4.0`
  (论文只用满分数据;我们现行 convert 脚本没做分数过滤,这次对齐)。
  基于 `uno_1m_total_labels.json` 重跑转换(融合 `convert_uno_labels.py` 的
  文件存在性检查 + 官方脚本的分数过滤);
- **多 ref 蒸馏部分**:`manifest_filtered.json` 直接并入;
- **混合比例:有效样本流里多 ref ≈ 30–50%**。蒸馏数据只有几千条而单 ref 有几十万,
  简单 concat 会被淹没——通过**重复蒸馏条目 N 遍**(oversample)达到目标比例,
  N 在脚本里按两边条数自动算,打乱后写出;
- 注意:`collate_fn` 断言同 batch ref 数一致,现行 `batch_size=1` 天然满足,
  **不要改 batch_size**,也不要动 collate。

### 训练配置

- 从当前最新 checkpoint **续训**(`RESUME_FROM_CHECKPOINT=latest`),不从头训:
  单 ref 能力已花 2 万步买来,推倒重来浪费且没必要;
- 复制 `scripts/train_ref_isolation.sh` 为 `scripts/train_distill.sh`,只改:
  `--train_data_json datasets/distill_multiref/train_mixed.json`、
  `--project_dir log/ref_distill`(**新目录,别覆盖原实验**)、
  `--max_train_steps` 在续训起点上 **+4000**;
- 其余超参(lora_rank 512、lr 8e-5、resolution 512、`--ref_isolation True`)一律不动,
  单变量原则:这次实验唯一变化的是数据;
- 续训起点的 checkpoint 文件需先从 `log/ref_isolation/` 复制到 `log/ref_distill/`
  (脚本里做,带完整性校验,参考 train_ref_isolation.sh 的残档检查逻辑)。

---

## 5. 阶段四:评测(修复是否成立以此为准)

### 评测集 `distill/build_eval_json.py`

两份,全部来自 **HELD-OUT** subject:

1. **原 smoke 集不动**:与历史结果(ckpt2000/7000)直接可比;
2. **新增 held-out 组合集**:HELD-OUT 10 个 subject 的双/三 ref 组合
   (跳过同 class),每组合 2 个 prompt,共 ~40 条,输出 JSON 同
   `dreambench_multiip.json` 格式。

### 评测脚本 `distill/eval_multiref.py`

在 `infer_multibanana.py` 基础上改(结构几乎一致),新增:

- **多 seed**:每任务 3 个 seed(单 seed 肉眼判读信号弱,已有教训);
- **客观指标**:对每张生成图算每个 ref 的 DINO 相似度(复用 filter_data.py 的
  模块),报告 `min_ref_sim` 的均值——**丢主体直接体现为该值低**;
- 变体跑 `official_full / ours_kv`,拼图复用 `board.py`。

### 验收标准

| 指标 | 现状(ckpt7000) | 目标 |
|---|---|---|
| held-out 双主体 min_ref_sim(ours_kv) | 低(常丢第二主体) | 显著提升,接近 official_full 的 80% 以上 |
| 双主体"两个都在"目测比例(3 seed) | ~50% 或更低 | ≥ 80% |
| 单 ref 质量(smoke 单 ref 案例) | 好 | **不回退**(混采里单 ref 仍占大头,应无碍) |
| 速度(ours_kv denoise) | 2-ref 512 ≈ 7.8s | 不变(蒸馏不动架构) |

蒸馏前后各跑一次同一评测,同 seed 直接对比。

---

## 6. 里程碑(每步远程执行,结果 git 回传)

1. **M1 生成**:gen_data.py 8 卡跑完 8000 条,回传 manifest_raw.json + 抽样拼图
   (40 张,board.py 拼)供人工检查 teacher 质量;
2. **M2 过滤**:calibrate 直方图 + 分数排序拼图回传 → 人工定阈值 → 出
   manifest_filtered.json(回传通过率统计);
3. **M3 训练**:train_distill.sh 续训 +4000 步,每 1000 步 checkpoint;
4. **M4 评测**:ckpt 每 +2000 步跑一次 eval_multiref.py,回传 results.json + 拼图,
   画 min_ref_sim vs steps 曲线,确认在涨且未平台化。

M1/M2 之间、M2/M3 之间需人工确认,不要一口气全跑。

## 7. 风险与备选

- **过拟合 20 个训练 subject**:held-out 评测直接暴露。若组内涨、held-out 不涨,
  走扩量路线:从 UNO-1M 的 target 图用 grounding+分割抽第二主体做交叉配对
  (即官方 stage-2 原始做法),主体多样性升到百万级,管线其余部分不变;
- **teacher 失败率过高**(M1 抽样拼图很差):优先换 prompt 模板组合
  (某些模板 teacher 弱),其次减 3-ref 占比;
- **单 ref 回退**:提高混采中单 ref 比例(50% → 70%),重跑 M3;
- **蒸馏后 held-out 有涨但不达标**:先扩数据量(8000 → 20000)再考虑改比例/步数。

## 8. 交付文件清单

```
distill/
  DISTILL_PLAN.md          # 本文档
  gen_data.py              # M1: teacher 生成(分 shard、断点续跑、held-out 断言)
  filter_data.py           # M2: DINO 打分 + calibrate + 阈值过滤
  build_train_json.py      # M3 前置: UNO-1M(score>=4.0) + 蒸馏数据混合(oversample)
  build_eval_json.py       # M4 前置: held-out 评测集生成
  eval_multiref.py         # M4: 多 seed + DINO 指标 + 拼图评测
scripts/
  train_distill.sh         # M3: 续训脚本(新 project_dir)
```

所有脚本:中文 docstring 说明用途与用法示例(仿 multibanana_eval 现有风格)、
`--dry_run` 模式(不占 GPU 验证任务枚举/路径/落盘)、失败样本跳过并打日志而非静默。
