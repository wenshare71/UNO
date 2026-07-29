# 多参考自蒸馏总体实验计划 v2(8×H800)

> v2 修订说明:吸收了 Opus 评审(逐条代码核对)的 B1–B4 / C1–C4 修正,并把运行环境
> 从 8×4090 切换到 8×H800(见 `docs/H800_REBUILD.md`)。gitignore 问题已由 rebuild
> 会话在 commit `8b8de83` 修复,不在本计划待办内。
>
> 全文用两种标记区分结论性质:
> - **[已验证]** 实际跑过命令/读过代码确认的事实,实现时直接信任;
> - **[假设]** 合理推断但未实测,首次触及时要先小规模标定,不许直接按它做不可逆决策。

## 0. 背景与已确认事实(全部 [已验证],不要重新推翻)

**要解决的问题**:我们训的隔离注意力 LoRA(`log/ref_isolation/`,已训至 13000 步,
目标 20000)在多参考图生成时**系统性丢第二主体**。

1. 训练数据 100% 单 ref:UNO-1M 标签是 `img_path1 → img_path2` 两图配对,官方转换脚本
   `uno/utils/filter_uno_1m_dataset.py:50` 与我们的 `scripts/convert_uno_labels.py:75`
   都只产出单元素 `image_paths`;我们的脚本**没有**按 `score_final` 过滤。
2. 官方 UNO checkpoint 的多 ref 能力来自论文 stage-2(合成多主体配对数据),该数据与
   管线未开源(`train.py:211` TODO)。
3. KV 缓存数学精确(ref 段固定 t=0 调制,`uno/flux/model.py:205`),质量问题与缓存无关。
4. 隔离注意力结构边界:ref 互不可见,只能学"多个独立主体各自入场景",学不了属性/风格/
   背景迁移。蒸馏数据必须避开后者 → 用 dreambooth,不用 multibanana。
5. `FluxPairedDatasetV2`(`uno/dataset/uno.py:60-99`):`image_root=dirname(json_file)`、
   忽略多余键(manifest 可带 `meta`)、`bucket_images` 自动 resize(原图无需预处理)、
   相对路径 `../dreambooth/dataset/...` 可正确解析;`collate_fn:105` 断言同 batch ref 数
   一致,`batch_size=1` 天然满足——不要改 batch_size 和 collate。
6. **方案**:复刻 stage-2——官方 full-attention UNO 当 teacher,在 dreambooth 训练
   subject 上合成多主体数据,混入续训。

**协作约束**:**本地无法 ssh 到 H800**,且**该机器 push 不出去**(代理放行 GET、
吃掉 POST),所以通道是单向的:本地 push 到 origin → H800 `git pull`;
反向只能由 H800 上的 agent **打印文本、经用户转达**。大批图片/数据留在远程,
只交付统计、数字与文字描述(图我看不到)。

**分工(固定)**:**代码全部由本地(Opus)编写**,H800 上的 MiniMax-M3 agent
**只负责运行脚本、观察现象、回传信息**,不写也不改 `.py`/`.sh`。
它的完整行动边界、诊断包格式与 hooks 安装见 **`distill/REMOTE_AGENT_HANDBOOK.md`**,
执行前必读。

**运行环境** [已验证,来自 `docs/H800_REBUILD.md` 实测]:8×H800(143 GB/卡,sm_90),
NVLink 全互联 + IB(**必须开 P2P/IB**,与 4090 相反;`train_ref_isolation.sh` 的 NCCL
开关已在 commit `7023e70` 改为可外部覆盖)。Python 3.10(环境由 `setup_env_h800.sh`
搭好)。权重/数据放本地 NVMe `/code`,checkpoint 落 ceph。UNO-1M 118 GB 已抽样验证
零缺失;官方 UNO 权重已在 HF 缓存。注意:`log/` 下 rsync 来的 checkpoint 是 root 属主,
**M3 前要先 chown**。

---

## 1. 决策记录

### D-1 teacher 精度:用 bf16 `flux-dev`、不 offload(回答评审 F1)

fp8+offload 是 4090 24 GB 下的妥协;蒸馏数据的质量上限由 teacher 直接决定,H800 上
没有理由省精度。**基线可比性处理**:teacher 精度只影响训练数据,不影响评测——M4 的
所有结论都来自**同机、同配置、同 seed 的蒸馏前/后成对对比**(两个 ckpt 都在 H800 上
用同一套 eval 配置重跑),历史 4090 fp8 smoke 数字只作参考、不参与结论。因此换 bf16
不损害任何对比的有效性。[已验证:显存充足;bf16 生成质量 ≥ fp8 属常识性 [假设],
无需专门验证]

### D-2 动物比例:目标 60%,允许 (组合,模板) 槽位复用(回答评审 B2)

TRAIN 集动物占 9/20,自然分布下 2-ref 组合 70% 含动物(113/162)。压到 50% 会把仅有的
49 个非动物组合复用得太狠;完全放开又会被狗支配。取 **60%**,并且**明确允许同一
(组合,模板) 槽位出现多条样本**——唯一性下沉到 (组合,模板,视角元组) 层面(见 §3),
每条样本仍全局唯一。[已验证:配比算术见 §3 表格,均可整除且复用 ≤3 次]

### D-3 双机并行(回答评审 D3)

M1 数据生成只依赖官方 teacher 权重,**不依赖任何 ref_isolation checkpoint**:
H800 立即开跑 M1/M2;旧 4090 机器同时把 ref_isolation 训完 20000 步;最终 ckpt
经 git/HF 同步到 H800 后做 M3。

---

## 2. Subject 切分(固定不可改,[已验证] 互斥且并集=全部 30 个)

**HELD-OUT(10,只用于评测,严禁进蒸馏数据)**:
`backpack_dog, bear_plushie, berry_bowl, can, candle, clock, colorful_sneaker, duck_toy, fancy_boot, grey_sloth_plushie`

**TRAIN(20,蒸馏数据唯一来源)**:
`backpack, cat, cat2, dog, dog2, dog3, dog5, dog6, dog7, dog8, monster_toy, pink_sunglasses, poop_emoji, rc_car, red_cartoon, robot_toy, shiny_sneaker, teapot, vase, wolf_plushie`

规则:同 class 组合跳过(class 表在 `datasets/dreambooth/dataset/prompts_and_classes.txt`);
两个名单写成模块级常量 + 启动断言,held-out 泄漏即 `sys.exit`。

---

## 3. M1:蒸馏数据生成 `distill/gen_data.py`

### 3.0 M0 pre-flight:**先做,在生成任何图之前**(约 10 分钟)

M2 要用的 `dino_vits16` 走 `torch.hub`,它会从 **github.com** 拉仓库 zip、从
**dl.fbaipublicfiles.com** 拉约 85 MB 权重。这台机器出网全经日本代理
(`docs/H800_REBUILD.md:33-42` 实测:HF 0.66 MB/s、PyPI 官方 0.05 MB/s、ubuntu 源
**直接不可达**),**github 可达性未测** [假设];而且环境里的 `HF_HUB_OFFLINE=1` 是 HF
的机制,**管不到 torch.hub**。

风险不在速度(85 MB 就算 0.66 MB/s 也才 2 分钟),在**时序**:这个依赖出现在 M2,
若等到那时才发现拿不到,8000 张图已经白生成了。所以必须前置:

```bash
python -c "import torch; m=torch.hub.load('facebookresearch/dino:main','dino_vits16',pretrained=True); print('dino ok', sum(p.numel() for p in m.parameters()))"
```

- **成功** → 权重已落 `$TORCH_HOME`(默认 `~/.cache/torch/hub`),记下路径,M2 直接用;
- **失败**(github 不可达 / 代理超时)→ **不要卡住 M1**,M1 与此无关,照常开跑;
  同时并行解决权重:① 换 `TORCH_HOME` 指向已有缓存;② 从内网源找 `timm` 的等价
  ViT-S/16 DINO 权重;③ 让本地机器下好后经 git-lfs 或内网中转。
  **三条都不通再知会我**(见 `distill/REMOTE_AGENT_HANDBOOK.md` 的升级规则)。

同时确认:`nvidia-smi` 8 卡可见、`datasets/dreambooth/dataset/` 30 个 subject 齐全、
官方 UNO dit_lora 在 HF 缓存里可读。

### Teacher 配置

`bytedance-research/UNO` 官方 dit_lora + full attention(即 smoke 的 `official_full`
变体,`(False, False, False)`);**`model_type=flux-dev`(bf16)、不 offload**、
512×512、`num_steps=25`、`guidance=4`、**显式 `--ref_size 512`**(与训练侧
`resolution_ref=None→512` 对齐,评审 C3)。模型加载/LoRA 挂载抄
`multibanana_eval/infer_multibanana.py`,**swap_lora 硬校验 key**(`:91-101` 已有实现,
直接复用)。

### 模板规则(评审 B3 修正,[已验证] 逐条读过模板)

只用**场景模板**,剔除单主体属性/换装模板:
- object 模板可用 **1–20 条**(21–25 是 `a red {}` 等属性词,弃用);
- live 模板可用 **1–10 条**(与 object 1–10 相同的场景;11–25 换装/属性词,弃用);
- 组合含动物 → 用 10 条共享场景模板;纯物体组合 → 用 20 条 object 场景模板;
- 双主体句式:`a {class1} and a {class2} in the jungle`;三主体:
  `a {c1}, a {c2} and a {c3} ...`(与 `dreambench_multiip.json` 句式一致)。

### 数量与配比([已验证] 以下算术全部实测核对过)

当前实际配比(2026-07-28 调整):在原有 8000 条 2-ref/3-ref 数据基础上追加 1000 条 1-ref,
总数达到 **9000 条**。1-ref 动物组 470 条是严格不重复视角下的理论上限。

| | 组合数(动物/非动物) | 槽位=组合×可用模板 | 目标条数 | 槽位复用 |
|---|---|---|---|---|
| 1-ref | 9 / 11 | 9×10 + 11×20 = 310 | 470 + 530 = **1000** | 动物 ≤6, 物体 ≤3 |
| 2-ref | 113 / 49 | 113×10 + 49×20 = 2110 | 2400 + 1600 = **4000** | 动物 2-3×, 非动物 1-2× |
| 3-ref | 595 / 119 | 595×10 + 119×20 = 8330 | 2400 + 1600 = **4000** | 均 ≤1, 无需复用 |
| 合计 | | | **9000** | |

唯一性保证:
- 2-ref/3-ref 每个 subject 有 4–6 张视角图 → 2-ref 组合 ≥16 种视角元组,
  每 (组合,模板) 槽位最多用 3 条 → 在 (组合,模板,视角元组) 层面不放回采样;
- 1-ref 每个 (subject,模板) 槽位内视角不重复,动物组容量 47×10 = 470 条刚好用满;
- **每条样本全局唯一**;`seed = base_seed + idx`,全流程确定性可复现。
- 2-ref/3-ref 编号 `idx 0..7999`、`seed 3407000..3414999` 保持不变,
  1-ref 追加在尾部:`idx 8000..8999`、`seed 3415000..3415999`。

### 运行方式

- 单进程单卡,`--shard_idx/--num_shards` 切分任务列表,8 进程各 1/8。
  **这是要新写的代码,仓库里没有现成 sharding 可抄**(评审 C4);`--dry_run` 有先例
  (`infer_multibanana.py:130`)可参考;
- 断点续跑:输出图已存在**且能完整解码**即跳过,脚本可反复重启。
  **必须用 `im.load()`(或 `verify()`)校验,不能只 `PIL.Image.open`**——`Image.open`
  是惰性的,只读文件头:[已验证] 一张截断到一半的 JPEG,`Image.open` 照样通过并报出
  正确尺寸 (512,512),只有 `.load()` 才抛 `OSError: image file is truncated`。
  shard 被杀时写到一半的图正是这个场景,而断点续跑存在的意义就是应对被杀;
- 逐样本 try/except,失败记日志继续,不许一个坏样本杀掉整个 shard
  (2–3 小时无人值守的底线);
- 每 shard 写 `manifest_shard{i}.json`,`--merge` 模式合并出 `manifest_raw.json`。

### 输出 schema(兼容 `FluxPairedDatasetV2`,[已验证])

```json
{
  "image_paths": ["../dreambooth/dataset/monster_toy/02.jpg", "../dreambooth/dataset/teapot/01.jpg"],
  "prompt": "a toy and a teapot in the jungle",
  "image_tgt_path": "images/000123.jpg",
  "meta": {"subjects": ["monster_toy", "teapot"], "seed": 3407123,
           "template_id": 5, "n_refs": 2}
}
```

落盘到 `datasets/distill_multiref/`(已被 gitignore,[已验证] commit `8b8de83`)。

---

## 4. M2:质量过滤 `distill/filter_data.py`

- Backbone 用 **`torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`**——
  与仓库官方评测脚本(`eval/evaluate_clip_dino_score_multi_subject.py:217`)和
  DreamBench/UNO 论文一致,**不用 DINOv2**(评审 B1;换 backbone 会失去与论文数字的
  可比性)。权重需在 **M0 预取**(见 §3.0),M2 阶段假定 `$TORCH_HOME` 已有缓存;
- 特征提取代码**必须复制,不能 import** `eval/evaluate_clip_dino_score_multi_subject.py`。
  [已验证] 两个阻塞点:(a) 该文件第 8 行 `import clip`,而 OpenAI CLIP 包**没装**
  (实测 `ModuleNotFoundError: No module named 'clip'`);(b) 第 199–215 行是**模块级**的
  `parser.parse_args()`(带 `required=True`)和 `clip.load(..., device='cuda')`——
  一 import 就 SystemExit 并往 GPU 加载 CLIP。
  把 `DINOImageDataset` 与 `extract_all_images`(:70 / :109)**抄进 `filter_data.py`**,
  抄的时候顺带甩掉用不上的 clip 依赖,并在 docstring 注明来源文件与行号;
- **聚合方式用 min-over-refs,不用官方的 mean**:对每条样本算每张 ref 与生成图的
  余弦相似度,取 `min_ref_sim`——丢主体正是 min 低、mean 会被另一个高分主体掩盖
  (这是对官方实现的有意偏离,理由要写进脚本 docstring);
- 分数写回 manifest(`meta.dino_sims`, `meta.min_ref_sim`);
- 流程:`--calibrate` 先出分位数表 + 按 `min_ref_sim` 排序抽 40 张拼图
  (复用 `multibanana_eval/board.py`)→ **人工看图定阈值** → `--threshold X` 产出
  `manifest_filtered.json` + 通过率统计;
- 预期通过率 60–80% [假设]。明显更低时先看拼图找系统性原因(某类模板差?3-ref 差?),
  不要盲目降阈值。

### 4.1 度量修订 v2(2026-07-24):整图 DINO 证伪,改为「定位 + 双侧裁剪」

**上面各条里"整图 vs 整图算余弦"的口径已废弃**;backbone(dino_vits16)、复制不 import、
min-over-refs 聚合、calibrate→人工定阈→threshold 的流程全部**保持不变**。

证伪证据([已验证],16 张 smoke + text-only 地板线配对):
- teacher 与 text-only(同 prompt/seed、不喂 ref)整图分数分布几乎重叠:中位差 +0.028,
  6/16 反转(泛型物体比忠实复刻分还高),0/16 越过地板线;而 teacher 在读图有肉眼铁证
  (000000 枣红背包连徽章、白标都复刻对了)——指标与视觉矛盾时,死的是指标;
- 排序与人眼判读接近相反:视觉满分的 005500/006500 垫底,融合失败的 005000 排 14/16;
- 根因:整图 CLS 特征测的是全局场景构图,多主体 + 丰富背景把主体信号稀释光
  (绝对值 0.13–0.36,远低于单主体 DreamBench 的 0.6–0.8);**ref 侧同样被污染**
  (backpack 的 ref 是"人背着包",人和天空占画面主导)。官方 eval 的整图口径是几百样本
  × 全部视角求 mean 的 benchmark 级聚合,能让系统差异浮出噪声;逐样本判定不能沿用。

v2 度量(presence + identity,实现见 `filter_data.py` 头注释):
- **presence**:GroundingDino-tiny(`IDEA-Research/grounding-dino-tiny`,transformers
  4.43.3 自带该架构,权重 ~0.7GB 需一次性预取)用类别词在生成图上开放词表检测;
  无框 → 该主体判掉,sim=0.0;
- **identity**:候选框 crop(12% padding)vs 该主体**全部参考视角**的主体 crop
  (ref 侧同样检测裁剪去污染),DINO 余弦取 max over (框 × 视角);
- 样本分仍为 min over subjects;`meta.det` 记录框数/置信度/胜出框(board 上叠红框核对),
  `meta.metric = "v2-grounded-crop"` 版本标记让旧缓存自动失效;
- `floor_line.py` 同步切 v2(与 filter_data 共用同一 ctx 打分,严格可比);
- 已知盲区:同类双主体共用类别词的框归属、主体融合"两边中等分"——标定时人工盯;
- v2 验收标准(在 16 张已人工判读的 smoke 上):003000/006000 的被掉主体无框记 0 垫底;
  005500/006500/007000 不得落入下四分位;004500/005000 应落入下四分位;
  地板线复验中 teacher 与 text-only 分布显著分开(中位差 ≥0.15 量级)。
  **不达标 → 度量继续回炉,不带着测不准的尺子跑全量。**

### 4.2 实际执行修订(2026-07-29):人工全量标注取代自动阈值过滤

9000 条生成数据已用 `distill/viewer` **全量人工标注**(pass/fail),M2 的
"calibrate → 人工定阈值 → 自动过滤"流程被人工判定直接取代:
`filter_data.py --from_annotations annotations.json` 直出 `manifest_filtered.json`。

**实测通过率(远低于 §4 预期的 60–80%)**:

| 子集 | pass | 总数 | pass 率 |
|---|---|---|---|
| 全体 | 3079 | 9000 | 34.2% |
| 1-ref | 900 | 1000 | 90.0% |
| 2-ref | 2041 | 4000 | 51.0% |
| 3-ref | 138 | 4000 | **3.5%** |

**3-ref 失败定性**(按 §9「teacher 失败率高」路径排查):fail 率在模板间均匀分布
(65–83%,无可剔除的烂模板),subject 间差异由 3-ref 曝光度解释——失败集中在
"3 主体同时入场"任务本身,官方 teacher 在 3-ref 上系统性不行,与要解决的丢主体
问题同源。**决策:保留 138 条 3-ref 但降权,M3 实际多主体数据以 2-ref 为主**;
不追加 3-ref 生成(按 3.5% pass 率补量性价比极低)。

v2 自动度量代码不废弃,保留给 M4 评测指标(§6 验收标准的 min_ref_sim 仍需要它)。

---

## 5. M3:混合训练

**前置**:旧机器训完 20000 步,最终 ckpt 同步到 H800;`log/` 下文件 **chown 到当前用户**
([已验证] rsync 保留了 root 属主,不处理会在保存 checkpoint 时崩)。

### 数据混合 `distill/build_train_json.py`

- **单 ref 部分**:基于 `uno_1m_total_labels.json` 重新转换,融合现有脚本的文件存在性
  检查 + 官方 `score_final >= 4.0` 过滤(论文只用满分数据;对齐官方配方)。
  **[已验证 2026-07-29] 本地 dump 的 score_final 不在顶层,在 `vlm_filter_cot.score_final`
  里**;分布:≥4.0 共 404,259 条(40.0%)、≥3.5 共 528,991 条(52.3%)、有 1 条
  异常值 131184.67 需剔除。**4.0 档充足,直接用 4.0,不需要降 3.5**;
- **多 ref 部分**:`manifest_filtered.json` 直接并入,注意两边路径都要相对于输出 json
  所在目录重算;
- 混合比例:有效样本流里**多 ref ≈ 40%**(30–50% 区间取中,[假设],M4 不达标时的
  第一调节旋钮)。蒸馏数据少、单 ref 几十万条,靠**重复蒸馏条目 N 遍** oversample,
  N 由脚本按两边条数自动算,打乱后写出。
  **[已验证 2026-07-29] 实际盘子:多 ref pass 3079 条(1-ref 900 / 2-ref 2041 /
  3-ref 138,见 §4.2),单 ref ≥4.0 池 404,259 条。若取 40% 混比,N≈88**
  (404259×0.4/0.6÷3079);**3-ref 降权**(§4.2 决策):138 条 3-ref 按独立权重
  oversample,不让它被 2-ref 淹没,也不让它主导——脚本按 n_refs 分组各自算重复数,
  组间比例由 --mix 参数显式控制(默认 2-ref:1-ref:3-ref ≈ 60:30:10)。

### 训练配置(评审 C1 简化)

- **不拷 checkpoint、不算步数偏移**,直接利用 `train.py:145` 的语义
  (传具体 safetensors 路径时 `global_step=0`):

```
--resume_from_checkpoint log/ref_isolation/checkpoint-20000/dit_lora.safetensors
--max_train_steps 4000
--project_dir log/ref_distill
--train_data_json datasets/distill_multiref/train_mixed.json
```

  [已验证] 该路径不恢复 optimizer 状态——但 `latest` 路径同样不恢复(函数只返回
  dit/ema/step),两种写法 Adam 动量都重置,简化无损失;
- `scripts/train_distill.sh` 从 `train_ref_isolation.sh` 复制,**注意自检 heredoc 里
  硬编码的 `project_dir` 和 `labels` 路径要一起改**(评审 C2),NCCL P2P/IB 在 H800
  上保持开启(默认值已可覆盖);
- 其余超参一律不动(lora_rank 512 / lr 8e-5 / res 512 / `--ref_isolation True` /
  batch_size 1 / grad_accum 2):**本实验唯一变量是数据**;
- `--checkpointing_steps 1000`,新目录 `log/ref_distill` 不覆盖原实验。

### 5.1 执行进度(2026-07-29)

**脚本与数据已落地并验证**:

- `distill/build_train_json.py` + `scripts/train_distill.sh` 已写好并实测;
- `datasets/distill_multiref/train_mixed.json` 已生成(29,777 条, 11.84 MB)。

**数据构成修正**(发现原计划 §5 把蒸馏 1-ref 当"多 ref"计权重的错误,已改):

| 分类 | 唯一样本 | 流中条数 | 占比 |
|---|---|---|---|
| UNO-1M 单 ref(score≥4.0, split1-5 磁盘可得) | 16,966 | 16,966 | 56.9% |
| 蒸馏 1-ref(in-domain 桥接, 不 oversample) | 900 | 900 | 3.0% |
| **单 ref 小计** | **17,866** | **17,866** | **60.0%** |
| 蒸馏 2-ref(oversample 5.3×) | 2,041 | 10,720 | 36.0% |
| 蒸馏 3-ref(oversample 8.6×) | 138 | 1,191 | 4.0% |
| **真·多 ref 小计** | **2,179** | **11,911** | **40.0%** |
| **总计** | | **29,777** | |

关键决策(2026-07-29):**蒸馏 1-ref 归入单 ref 池**(任务类型相同,都只有 1 张 ref,
不教"多主体不丢"),只作 dreambooth 主体域的 in-domain 桥接;真·多 ref = 2-ref + 3-ref,
mix 从 `60:30:10`(2r:1r:3r)改为 `90:10`(2r:3r)。修正后**真·多 ref 实占 40.0%**
(修正前表面 40% 但实际仅 28%)。

**磁盘可得性约束** [已验证]:UNO-1M 标签 101 万条引用 102 个 split,但磁盘只解压了
split1-5(5 万对),score≥4.0 池实际 16,966 条(非计划假设的 404,259)。混比 40% 不变,
实验核心结论不受影响——续训自 20000 步,单 ref 已学好,4000 步只补多 ref。

**checkpoint 权限** [已验证]:`log/ref_isolation/checkpoint-20000` 是 root:root 600
(rsync 残留),已 `sudo chown -R wuwenxuan:wuwenxuan log/ref_isolation` 改属主,
加载验证 304 个 LoRA 张量(rank 512, down/up weights)正常。

**100 步标定** [已验证 2026-07-29 17:00]:

| 指标 | 实测 |
|---|---|
| 100 步总耗时 | 10 min 07 s(含 ~7 min 加载 + ~3 min 训练) |
| 稳态速度 | 5.64 s/it |
| loss 范围 | 0.295–0.753(单 ref ~0.3, 多 ref ~0.5–0.7, 混合波动符合预期) |
| GPU 显存 | 25–28 GB/卡(H800 143 GB, 余 115 GB, 无 OOM 风险) |
| GPU 利用率 | 8 卡 100% |
| checkpoint 保存 | `checkpoint-100/{dit_lora.safetensors 3.6GB, optimizer.bin 1.4GB}` ✅ |
| NCCL | P2P/IB 开启, NVLink 通信正常 |

**正式训练已启动** [2026-07-29 17:04]:

- tmux 会话 `m3`,`bash scripts/train_distill.sh`,日志 `log/ref_distill/train.log`;
- 监控会话 `monitor`(claude code, 30 min 周期);
- 预计 4000 步耗时 ~6.3 h(5.25 s/it 实测, 比标定略快),约 21:25 完成;
- 产出 `log/ref_distill/checkpoint-{1000,2000,3000,4000}`。

**4000 步训练覆盖估算**(有效 batch = 8卡×1×2 = 16, 4000 步消耗 64,000 样本):

| 类型 | 唯一样本 | 每条被看次数 |
|---|---|---|
| UNO-1M 单 ref | 16,966 | ~2.1(防遗忘足够) |
| 蒸馏 1-ref | 900 | ~2.1(轻度桥接) |
| 蒸馏 2-ref | 2,041 | ~11.3(学进去足够) |
| 蒸馏 3-ref | 138 | ~18.5(略多但降权保留) |

---

## 6. M4:评测

### 评测集 `distill/build_eval_json.py`(评审 B4 修正)

held-out 实有 44 个合法 2-组合、112 个 3-组合 [已验证]。取法(确定性,固定 seed):

- 2-ref:从 44 对中选 **20 对**(贪心保证每个 held-out subject 至少出现 3 次)× 2 个
  场景 prompt = 40 条;
- 3-ref:从 112 组中选 **10 组**(同样保证覆盖)× 2 prompt = 20 条;
- 共 **60 条**,输出格式同 `dreambench_multiip.json`;
- 原 smoke 集(4 个 dreambench 案例 + 内置案例)保持不动,作为历史连续性参考。

### 评测脚本 `distill/eval_multiref.py`

在 `infer_multibanana.py` 结构上改:变体 `official_full` / `ours_kv`;**每任务 3 个
seed**(单 seed 肉眼判读信号弱,已有教训);指标 = 复用 M2 的 dino_vits16 模块算
**每 ref 相似度 + min_ref_sim**;拼图复用 `board.py`。

**可比性纪律**:结论只从"同机同配置同 seed 的 ckpt-20000(蒸馏前)vs ckpt-distill
(蒸馏后)成对差值"得出;与历史 4090/fp8 数字不做跨机比较。

### 验收标准

| 指标(held-out 60 条,3 seed) | 蒸馏前(预期) | 蒸馏后达标线 |
|---|---|---|
| `min_ref_sim` 均值(ours_kv) | 低 [假设,M4 首跑实测] | 达到同图 official_full 的 **≥80%** |
| 双主体"两个都在"目测比例 | ~50% 或更低 [假设] | **≥80%** |
| 单 ref 质量(smoke 单 ref 案例) | 好 | **不回退**(目测 + 与蒸馏前同 seed 对比) |
| ours_kv 推理速度 | — | 不变(蒸馏不动架构,只需抽查确认) |

---

## 7. 时间与资源预算(8×H800)

| 阶段 | 内容 | 预估 | 性质 |
|---|---|---|---|
| **M0** | pre-flight:dino_vits16 预取 + 环境自检(§3.0) | ~10 min | **必做,在 M1 之前** |
| M1 标定 | 先跑 50 张实测吞吐 | ~10 min | 必做,校准下面的估计 |
| M1 | 8000 张 bf16 生成,8 卡并行 | **2–3 h** | [假设],基于 rebuild 的 H800 3–4× 提速估计 |
| M2 | dino_vits16 打分 8000 张 + calibrate 拼图 | <0.5 h + 人工定阈值 | 打分时长 [假设],量级可靠 |
| M3 前置 | 单 ref 数据重转换(score≥4.0)+ 混合 | <1 h CPU | [假设] |
| M3 标定 | 先跑 100 步实测 it/s | ~15 min | 必做;H800+P2P 的训练速度**无实测数据** |
| M3 | 4000 步,8 卡 | 数小时量级,以标定为准 | [假设] |
| M4 | 60 任务 × 3 seed × 2 变体 = 360 张/次,跑 2 个 ckpt | 每次 1–1.5 h + 打分分钟级 | [假设] |

**M3 实测** [已验证 2026-07-29]:100 步标定 10 min(含加载),稳态 5.64 s/it;正式 4000 步
实测 5.25 s/it(NVLink 通信效率高于标定),预计 ~6.3 h 完成数据。

并行轨道:M1/M2 在 H800 上进行的同时,旧 4090 机器把 ref_isolation 从 13000 训到
20000 步;两条线在 M3 汇合。

---

## 8. 里程碑与人工确认点

0. **M0**:pre-flight(§3.0)——dino_vits16 预取 + 8 卡/数据/权重自检。
   失败**不阻塞 M1**,但要在里程碑记录里写明当前状态。
1. **M1**:先 `--dry_run` 核对任务枚举(条数、动物比例、held-out 断言、路径),再 50 张
   标定,后全量。产出:`manifest_raw.json` + 40 张随机抽样拼图。
   **人工确认点①:看拼图评 teacher 质量,通过才进 M2。**
2. **M2**:calibrate 直方图 + 排序拼图 → **人工确认点②:定阈值** → `manifest_filtered.json`
   + 通过率。通过率 <50% 时按 §4 排查,不盲目放行。
3. **M3**:混合 json → 100 步标定 → 4000 步训练,每 1000 步 checkpoint。
   产出:`log/ref_distill/checkpoint-{1000..4000}`。
   **[进度 2026-07-29] 脚本+数据+标定已通过,正式训练 17:04 启动,预计 21:25 完成。**
4. **M4**:ckpt-2000 与 ckpt-4000 各评一次(蒸馏前 ckpt-20000 同批评),产出
   results.json + 拼图 + min_ref_sim 对比表;若 2000→4000 仍在涨,酌情追加步数。
   **人工确认点③:对照 §6 验收标准判定实验成败。**

每个里程碑结束把统计/拼图/结论 commit 回 fork(图片大文件不 commit)。

---

## 9. 风险与回退

| 风险 | 信号 | 回退动作 |
|---|---|---|
| 过拟合 20 个训练 subject | held-out 不涨、训练组合肉眼好 | 走官方 stage-2 扩量路线:从 UNO-1M target 图用 grounding+分割抽第二主体交叉配对,主体多样性升到百万级,管线其余不变 |
| teacher 失败率高(确认点①不过) | 拼图大面积丢主体/畸形 | 先按模板/组合类型分桶统计失败率,剔除差模板;再减 3-ref 占比;teacher 本身不动(官方权重是上限) |
| 单 ref 能力回退 | smoke 单 ref 案例变差 | 多 ref 混比 40%→25%,重跑 M3(4000 步成本可承受) |
| 有效但不达标 | min_ref_sim 涨但 <80% 线 | 依次:数据量 8000→20000(H800 上仅 +5–7 h)→ 混比上调 → 步数 +4000 |
| H800 首次长任务踩环境坑 | 任一阶段异常中断 | 所有脚本断点续跑 + 逐样本容错是硬要求;疑难环境问题问 rebuild 会话 |
| bf16 teacher 与部署端 fp8 推理的分布差 | 理论上存在 [假设] | 不处理——蒸馏学的是注意力分配行为,不是像素分布;若 M4 达标即证明无碍 |

---

## 10. 交付文件清单

```
distill/
  DISTILL_PLAN.md          # 本文档(v2)
  REMOTE_AGENT_HANDBOOK.md # 远程 H800 上 MiniMax-M3 的行动边界与诊断包格式(执行前必读)
  gen_data.py              # M1:teacher 生成(新写 sharding、断点续跑、held-out 断言、逐样本容错)
  filter_data.py           # M2:dino_vits16 + min-over-refs 打分、calibrate、阈值过滤
  build_train_json.py      # M3 前置:UNO-1M(score>=4.0)重转换 + 蒸馏数据 oversample 混合  [已落地 2026-07-29]
  build_eval_json.py       # M4 前置:held-out 60 条评测集(确定性选取)
  eval_multiref.py         # M4:3 seed + min_ref_sim + 拼图
scripts/
  train_distill.sh         # M3:训练脚本(注意 heredoc 与 accelerate 两处路径都要改)  [已落地 2026-07-29]
datasets/distill_multiref/
  train_mixed.json         # M3 混合训练集(29,777 条, 真·多ref 40%)  [已生成 2026-07-29]
log/ref_distill/
  train.log                # M3 正式训练日志  [运行中 2026-07-29]
  checkpoint-{1000..4000}/ # M3 训练产物(预计 21:25 全部产出)
```

所有脚本:中文 docstring(仿 multibanana_eval 风格)、`--dry_run`、失败打日志不静默。
