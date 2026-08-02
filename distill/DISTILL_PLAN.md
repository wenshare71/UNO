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

**分工** [2026-07-30 修订,取代原「固定」条款]:远程执行者已由 MiniMax-M3 换为
**KIMI K3**,代码授权由「一刀切禁止」改为**三档**——🟢 环境/调用方式/非语义参数自主;
🟡 按规格文档写**新建**脚本;🔴 改实验语义、改既有代码、做判断类决定一律禁止。
**改档依据是产物可机械复核,不是模型能力评级**:M4 Stage A 的 227 条非锚点任务
× 6 个字段由本地独立重算比对,零差异。完整边界、黄档准入三条件与诊断包格式见
**`distill/REMOTE_AGENT_HANDBOOK.md` §2.0**,执行前必读。

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

~~v2 自动度量代码不废弃,保留给 M4 评测指标(§6 验收标准的 min_ref_sim 仍需要它)。~~

**[2026-07-30 修订]** 这句作废:v2 度量硬绑 dreambooth(`_class_word:207`、
`_ref_vecs:214`),multibanana 打不了分,冒烟 39 张里 24 张结构上不可评,
**当不了 M4 的判定尺**。M4 验收改为人工盲评,见 §6.2。代码本身仍留着,
但只作 M2 的历史产物,不再承担任何验收职责。

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

**正式训练已完成** [已验证 2026-07-29 23:21]:

- tmux 会话 `m3`,`bash scripts/train_distill.sh`,日志 `log/ref_distill/train.log`;
- 监控会话 `monitor`(claude code, 每小时 :18 触发),训练完成后自动收尾并删除 cron;
- **4000 步总耗时 6h 14min(5.43 s/it 实测, 比标定 5.64 略快),23:21 完成**;
- 产出 `log/ref_distill/checkpoint-{1000,2000,3000,4000}` 全部落盘
  (每个含 `dit_lora.safetensors` 3.8GB + `optimizer.bin` 1.4GB);
- 全程零异常(无 `anomaly_*.md`),最终 step_loss=0.192,loss 区间 0.18–0.57;
- 保活收尾 `bench_kv_cache.py` 自动跑完:iso vs kv 像素差 max=58 mean=0.4681
  (✅ 等价,蒸馏不动架构已验证),加速比 full→kv 1.30x,peak 显存 34.5 GB/卡。

**4000 步训练覆盖估算**(有效 batch = 8卡×1×2 = 16, 4000 步消耗 64,000 样本):

| 类型 | 唯一样本 | 每条被看次数 |
|---|---|---|
| UNO-1M 单 ref | 16,966 | ~2.1(防遗忘足够) |
| 蒸馏 1-ref | 900 | ~2.1(轻度桥接) |
| 蒸馏 2-ref | 2,041 | ~11.3(学进去足够) |
| 蒸馏 3-ref | 138 | ~18.5(略多但降权保留) |

### 5.2 M4 冒烟评测(2026-07-30)

M3 训练完成后,先跑两轮冒烟评测肉眼确认蒸馏方向,再决定是否进全量 M4。两个脚本
(`distill/smoke_eval.py`、`distill/eval_multibanana.py`)复用 `multibanana_eval/board.py`
拼图,单卡 bf16 不 offload,每任务 3 变体并排对比:

- `official_full`:官方 UNO + 全注意力(teacher 金标准)
- `ours_kv_pre`:ckpt-20000(蒸馏前)+ 隔离注意力 + KV cache
- `ours_kv_post`:ckpt-4000(蒸馏后)+ 隔离注意力 + KV cache

**轮①:dreambench held-out 2-ref**(`smoke_eval.py`,5 个 held-out 2-ref 组合各 1 prompt)
— 产出 `output/smoke_eval/smoke_compare.png`:

| 变体 | denoise 均值 | peak 显存 | vs teacher |
|---|---|---|---|
| official_full | 5.13 s | 36.2 GB | 1.00x |
| ours_kv_pre | 2.90 s | 37.0 GB | 1.77x |
| ours_kv_post | 2.90 s | 37.0 GB | 1.77x |

肉眼初判(待 §4.1 v2 度量定量确认):5 个 case 上 `ours_kv_post` 均比 `ours_kv_pre`
更接近 `official_full`——两个主体都保留住,而 pre 疑似丢一。**这是 in-distribution
信号**(蒸馏数据就是 dreambench 句式),支持"蒸馏让 student 学会不丢主体"。

**轮②:multibanana `add` 子集**(`eval_multibanana.py`,8 个主体合成任务)
— 产出 `output/multibanana_eval_distill/compare_add.png`:

| 变体 | denoise 均值 | peak 显存 | vs teacher |
|---|---|---|---|
| official_full | 4.94 s | 36.2 GB | 1.00x |
| ours_kv_pre | 2.88 s | 37.0 GB | 1.72x |
| ours_kv_post | 2.88 s | 37.0 GB | 1.72x |

肉眼初判:`add` 是 OOD 任务(Image 1 是场景图不是 ref,结构与训练不同),pre/post
**几乎无差别**——都靠隔离注意力的通用泛化撑着,蒸馏没教这个任务模式。这反而是个
**稳健性信号**:蒸馏针对性地改善了"多主体入场"目标能力,没泛化到无关任务上做
奇怪的事(没把 `add` 打坏,也没在 `add` 上虚涨)。

**共同结论**:蒸馏没把架构/速度打坏(KV cache 1.72–1.77x 不变,显存 37 GB 不变),
in-distribution 有正信号,OOD 无负信号。**下一步**:跑 §4.1 v2 度量(GroundingDino
presence + DINO 裁剪)给 39 张图(15 + 24)自动打分,把肉眼判断落地为数字,再决定
是否进全量 M4。

---

## 6. M4:评测

> ⚠️ **本节 §6.0 部分(下面到「验收标准」表为止)是原设计,已被实际执行推翻,
> 保留作历史记录。现行口径直接看 §6.1(评测集)/ §6.2(验收标准)/ §6.3(结果与判定)。**
>
> ⚠️ **§6.3 于 2026-08-02 第二次改写**:首轮判读中的
> 「S3 单 ref 上隔离近似 no-op」→「隔离注意力无可测代价」→「SFT 配方有毒、架构无罪」
> 这条推理链**已被代码证伪并作废**(依据见 §6.3 的作废小节)。
> **数字全部有效,判读换了。** 验收口径也从 `(S+B)/(T+B)` 改为
> 非平局胜率 + Wilson CI(§11.2),阈值待 §11 步骤 3 重定。

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

### 6.1 实际执行修订(2026-07-30):评测集从 60 条扩到 232 条五分层

上面 §6 的原设计(2-ref 40 条 + 3-ref 20 条 = 60 条)在冒烟评测后被推翻,三条理由:

1. **v2 自动度量不能用于全量判定**。`filter_data.py` 的 presence+crop 度量硬绑
   dreambooth(`_class_word:207` → `prompts_and_classes.txt`、`_ref_vecs:214` →
   `data_dir/subject/*.jpg`),multibanana 打不了分;冒烟 39 张里 **24 张结构上不可评**,
   剩下 15 张 n 太小,当门禁不合格。
2. **原设计缺单 ref 回归层**。混入 40% 多 ref 训了 4000 步,**从没有人验证过单 ref
   没退化**——而这正是 §9 预登记的风险之一。
3. **冒烟暴露出新失败模式「主体复制」**(模型学到的是"该有 N 个物体"的计数先验,
   强过"ref j → 槽位 j"的绑定),原设计无对应层。

改为五分层 **232 任务 / 711 图**,完整规格见 `distill/M4_EVAL_SPEC.md` §3.2:

| 层 | 内容 | 任务数 | 目的 |
|---|---|---|---|
| S0 | 锚点(冒烟 5 case 原 seed) | 5 | 回归自检,与已提交基线比对,不过不放行 S1–S4 |
| S1 | 44 个合法 2-组合 × 3 seed | 132 | 主场:多主体主指标 |
| S2 | 主体复制探针(含 ckpt-2000 第四变体) | 15 | 复制是不是过训练产物 |
| S3 | **单 ref 回归** | 60 | 原设计缺失项 |
| S4 | 3-ref | 20 | 3-ref 能力 |

### 6.2 验收口径修订(2026-07-30):人工盲评取代 `min_ref_sim`

上表的 `min_ref_sim ≥ 80%` **作废**(度量 v2 不可用,理由同 6.1-1)。改为
**人工盲评 A/B**(`distill/blind_eval/`,post4000 vs teacher,227 条 = 232 − S0 锚点):
T = 老师更好、S = 学生更好、B = 无明显区别,

```
score = (S + B) / (T + B)        达标线 ≥ 0.95
```

**[2026-08-02 再次修订]** 这个口径**从 M5 起停用为判据**,只保留为与 M4 的可比参考。
三个毛病:含义随平局率漂移(跨层不可比)、`B` 把"都好"和"都烂"合并、
S1 的 132 条不是独立样本(44 组合 × 3 seed)。
新口径见 **§11.2**:主报非平局胜率 `S/(S+T)` + Wilson 95% CI,平局率单列。
**`0.95` 这条线本身从未验证过可达**,新阈值 X 待 §11 步骤 3 依噪声地板重定。

### 6.3 M4 结果 [已验证 2026-07-31,判读于 2026-08-02 改写]:**未达标**

**工程面全绿**:232 任务 / 711 图,`n_fails: 0`,8 卡分片。
**速度验收通过**:teacher 4.86 s/张,ours_kv_post4000 2.88 s/张 = **1.686×**,
显存 37.9 GB 不变——原验收表「推理速度不变」这一行成立。

**盲评结果(227 条全部标注)**:

| 层 | T | S | B | n | score | 平局率 | 非平局 n | 学生胜率 | 符号检验 p |
|---|---|---|---|---|---|---|---|---|---|
| S1 (2-ref) | 55 | 32 | 45 | 132 | 0.770 | 34.1% | 87 | 36.8% | 0.018 |
| S2 (复制探针) | 3 | 6 | 6 | 15 | 1.333 | 40.0% | 9 | 66.7% | 0.508 |
| S3 (单 ref) | 20 | 11 | 29 | 60 | 0.816 | 48.3% | 31 | 35.5% | 0.150 |
| S4 (3-ref) | 4 | 1 | 15 | 20 | 0.842 | 75.0% | 5 | 20.0% | 0.375 |
| **总体** | **82** | **50** | **95** | **227** | **0.819** | 41.9% | 132 | 37.9% | 0.007 |

**总体 0.819 < 0.95 → 人工确认点③ 判定:不达标。**

#### 判读(2026-08-02 第二次改写。**前一版的第 3 条与「隔离无罪」推论已被代码证伪,
不要重新捡回**;下面 1/2 两条经得起复核,保留)

1. **S2 的 1.333 不成立**。非平局仅 9 条,p=0.51。"学生在主体复制上反超老师"
   是噪声,**不许写进任何结论**。
2. **分层 score 的排序(S1<S3<S4)是平局率假象**。平局率 34%/48%/75% 差异巨大,
   而 `(S+B)/(T+B)` 平局越多越靠近 1,跨层直接比 score 无效。剥掉平局看
   **学生在有分歧对子里的胜率**:S1 36.8% vs S3 35.5%,`z=0.13, p=0.90`,
   **统计上无法区分**。
3. **这批数据牢固支持的只有一句话:学生整体、以及在 2-ref 上,显著劣于 teacher。**
   逐层做 Wilson 95% CI(非平局胜率):

   | 层 | 非平局胜率 | Wilson 95% CI | 是否显著 |
   |---|---|---|---|
   | S1 (2-ref) | 36.8% | [0.274, 0.473] | ✅ 显著 |
   | S2 | 66.7% | [0.354, 0.879] | ❌ 含 0.5 |
   | S3 (单 ref) | 35.5% | [0.211, 0.531] | ❌ 含 0.5,**单层不显著** |
   | S4 (3-ref) | 20.0% | [0.036, 0.624] | ❌ n=5 |
   | **总体** | **37.9%** | **[0.301, 0.464]** | ✅ 显著 |

   S3 单层**不显著**。首轮曾用「S1 与 S3 掉幅一致」去论证全局质量税,
   那是把一个不显著的观测当证据用了。

**S4 的 0.842 不可作质量陈述**:75% 平局,而 `B` 把"两边都好"和"两边都烂"混为一谈;
teacher 自己在 3-ref 上人工通过率只有 3.5%(§4.2),这些平局大概率是"一样烂"。

**独立性瑕疵(如实声明,不推翻结论)**:S1 的 132 条 = 44 组合 × 3 seed,
同组合内相关,符号检验按独立样本算会**略高估显著性**。S1 的 p=0.018 做组合级聚类后
会变弱,但 55:32 的方向大概率仍站得住。报告时必须写明。

#### ❌ 作废:原「附带的正面结论:隔离无罪」[2026-08-02 证伪]

> ~~隔离生效的 S1 与隔离失效的 S3 掉幅一致 ⇒ 隔离注意力没有带来可测量的额外代价。~~
> ~~这把问题定义改写了:不是「隔离学不会绑定」,而是「SFT 配方在损伤模型,架构本身是好的」。~~

**证伪依据(两处代码,不是推测)**:

- `uno/flux/ref_attention.py:70-90`:掩码规则是 **ref 行只在自己段上为 True——
  不看 txt、不看 img**,不只是"ref 之间互不可见"。全注意力下单张 ref 每层都能
  attend 到 prompt 和当前生成图,隔离把这条通路整个切断,**N=1 时照样生效**。
- `uno/flux/model.py:206-227`:只要 `ref_isolation` 开着,ref 段就用
  **t=0 / guidance=1 的固定调制**(与去噪步解耦),这与 ref 数无关。

即:**S3 上隔离并不是 no-op,两层机制都在工作**。因此
「S3 测的是纯粹生成质量」不成立,「S1≈S3 ⇒ 隔离无代价」的推理链断在第一环。

**连带后果**:三个混淆**至今没有分离**——
① **底座差距**(ckpt-20000 的 stage-1 数据没按 `score_final` 过滤,官方只用满分数据);
② **隔离的结构代价**;③ **M3 配方损伤**。
下面的机制假说只是 ③ 的一个候选,**不再是已确认的结论**。

**而且 ③ 与冒烟证据方向相反**:PRE 有画面级崩坏(`M4_EVAL_SPEC.md:53`),
POST 39 张一张都没有 ⇒ M3 至少在稳定性上是**改善**。这正是 §11 第一步要测的。

#### 机制假说 ③ [假设,**未验证**,待 §11 S2 批次判定]

混合集的账(由 §5 的实际数字重算):

```
总样本流   29,777
├─ 单 ref  17,866 (60%)   16,966 真实 UNO-1M + 900 蒸馏 1-ref,不重复
└─ 多 ref  11,911 (40%)   仅 2,179 条唯一样本(2041×2-ref + 138×3-ref)
                          重复 5.47 倍填满

训练量 = 4000 步 × 8 卡 × 2 累积 = 64,000 样本 = 2.15 epoch
  → 每张真实图看 2.15 次
  → 每张 teacher 合成图看 ~11.8 次      ← 差 5.5 倍
```

2-ref 人工通过率 51%、3-ref 3.5%(§4.2),合成图本身带 teacher 瑕疵;
对它们重复曝光近 12 次,而 **LoRA 是全局共享的、没有任何东西按 ref 数把它分区**
⇒ 瑕疵渗进整个模型,单 ref 一并遭殃。这与 §9「单 ref 能力回退」的预登记回退动作
(混比 40%→25%)指向同一个旋钮。

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
   **[进度 2026-07-29 23:21] ✅ 已完成。4000 步 6h14min,零异常,4 个 checkpoint 全落盘。**
4. **M4**:评测集构建 → 全量推理 → 人工盲评。
   **[进度 2026-07-30] 冒烟评测(轮①dreambench 2-ref + 轮②multibanana add)已跑完,
   肉眼信号正向(in-dist 有改善、OOD 无回退);评测集与验收口径按 §6.1/§6.2 改写。**
   **[进度 2026-07-31] ✅ 已完成。Stage A/B 工程全绿(232 任务 / 711 图 / 0 失败),
   227 条人工盲评全部标注。**
   **人工确认点③ 判定:❌ 未达标——总体 score 0.819 < 0.95(§6.3)。**
   **[2026-08-02 修订] 首轮对失败模式的判读(「隔离无代价、SFT 有毒」)已被代码证伪
   (§6.3)。当前站得住的只有「学生整体、以及在 2-ref 上显著劣于 teacher」,
   底座 / 隔离 / 配方三个混淆未分离。**
5. **M5**(§11,2026-08-02 改写):**先标定尺子、再分离混淆、最后才上训练**。
   五个确认点,**上一个没打勾不许进下一步**:
   - **确认点 0**:盲评服务器重构 + M4 的 227 条标注迁移后复算出 §6.3 五行逐值吻合。
   - **确认点 1**:30 条噪声地板图 0 失败、拼图目检两侧都是正常 teacher 出图。
   - **确认点 2**:198 条标注批 R2 的三张表(噪声地板 / 自洽率 / post-vs-pre 分层)。
   - **确认点 3**:新达标线 X 写死进 SPEC 并 commit,之后不许改。
   - **确认点 4**:按 post-vs-pre 结果做门控分叉,决定走 P2 / ② / 臂 A-B 中的哪条。
   原「先 P0 再臂 A」的排序作废——训练类实验全部退到确认点 4 之后。

每个里程碑结束把统计/拼图/结论 commit 回 fork(图片大文件不 commit)。

---

## 9. 风险与回退

| 风险 | 信号 | 回退动作 |
|---|---|---|
| 过拟合 20 个训练 subject | held-out 不涨、训练组合肉眼好 | 走官方 stage-2 扩量路线:从 UNO-1M target 图用 grounding+分割抽第二主体交叉配对,主体多样性升到百万级,管线其余不变 |
| teacher 失败率高(确认点①不过) | 拼图大面积丢主体/畸形 | 先按模板/组合类型分桶统计失败率,剔除差模板;再减 3-ref 占比;teacher 本身不动(官方权重是上限) |
| **单 ref 能力回退 [⚠️ 信号存疑,2026-08-02 降级]** | ~~smoke 单 ref 案例变差~~ → S3 学生胜率 35.5%,但 **Wilson CI [0.211, 0.531] 含 0.5,单层不显著**(§6.3) | **不要据此就动混比**——证据不够。先做 §11 步骤 2 的 `post vs pre` 148 条:那是"M3 有没有伤单 ref 能力"的直接检验,零 GPU。确认点 4 门控后才轮到 P2(混比 40%→25%) |
| ~~有效但不达标~~ **无效,病灶未定位** | ~~min_ref_sim 涨但 <80% 线~~ → 实际:总体非平局胜率 37.9%,CI [0.301, 0.464],**显著劣于 teacher** | **原回退动作(加数据/加混比/加步数)全部作废**——它们假设"方向对、量不够"。但**新的归因也还没有**:§6.3 首轮的"配方有毒"判读已被代码证伪,底座/隔离/配方三个混淆未分离。走 §11 分步定位,**在确认点 4 之前不要选旋钮** |
| **达标线 0.95 从未验证可达 [新增 2026-08-02]** | 无——这是个盲区,不是故障 | §11 步骤 1-2 的噪声地板 + 自洽率;步骤 3 据此重定 X 并写死。**在 X 定下来之前,不用 0.95 对任何新实验下"达标/不达标"结论** |
| **判读工具本身出错 [新增 2026-08-02,已部分兑现]** | 盲评服务器启动即崩;marks 按下标键控而 `eval_set.json` 被中部插入 45 条 ⇒ 80 条下标偏移 | §11 步骤 0 重构为配对清单驱动 + 按 `pair_id` 键控;迁移后**必须复算出 §6.3 五行逐值吻合**才算数。今后**任何被声明冻结的文件改动一律新开版本,不就地改** |
| H800 首次长任务踩环境坑 | 任一阶段异常中断 | 所有脚本断点续跑 + 逐样本容错是硬要求;疑难环境问题问 rebuild 会话 |
| bf16 teacher 与部署端 fp8 推理的分布差 | 理论上存在 [假设] | 不处理——蒸馏学的是注意力分配行为,不是像素分布;若 M4 达标即证明无碍 |

---

## 10. 交付文件清单

```
distill/
  DISTILL_PLAN.md          # 本文档(v2)
  REMOTE_AGENT_HANDBOOK.md # 远程 H800 上 KIMI K3 的行动边界(§2.0 三档授权)与诊断包格式(执行前必读)
  M4_EVAL_SPEC.md          # M4 实现规格(黄档规格文档,232 任务五分层)  [已落地 2026-07-30]
  gen_data.py              # M1:teacher 生成(新写 sharding、断点续跑、held-out 断言、逐样本容错)
  filter_data.py           # M2:dino_vits16 + min-over-refs 打分、calibrate、阈值过滤
  build_train_json.py      # M3 前置:UNO-1M(score>=4.0)重转换 + 蒸馏数据 oversample 混合  [已落地 2026-07-29]
  build_eval_json.py       # M4 前置:五分层 232 条评测集(确定性选取,17 条启动断言)  [已落地 2026-07-30]
  eval_multiref.py         # M4:4 变体 × 分片 × 断点续跑 + 锚点自检 + 拼图  [已落地 2026-07-30]
  M5_DECISIONS.md          # M5 的 7 个决策点与你的答复(§11 的依据)  [已落地 2026-08-02]
  blind_eval/              # 人工盲评(FastAPI 单页,A/B 槽位确定性分配)
                           #   v1 [2026-07-30] 任务×两个硬编码变体,揭盲回顾
                           #   v2 [§11 步骤 0] 配对清单驱动、marks 按 pair_id 键控、支持重放
  build_pairs.py           # [§11 步骤 0 新建] 由 eval_set + results 生成配对清单(4 类 pair)
  build_noise_floor.py     # [§11 步骤 1] 生成 noise_floor_tasks.json(同 refs、仅换 seed);--verify 单独复校  [已落地 2026-08-02]
  M5_STEP1_RUN.md          # [§11 步骤 1] H800 执行单(6 步:pull→dry_run→生成→锚点→merge→带回)  [已落地 2026-08-02]
  smoke_eval.py            # M4 冒烟:dreambench held-out 2-ref 三方对比  [已落地 2026-07-30]
  eval_multibanana.py      # M4 冒烟:multibanana 子集三方对比(默认 add)  [已落地 2026-07-30]
scripts/
  train_distill.sh         # M3:训练脚本(注意 heredoc 与 accelerate 两处路径都要改)  [已落地 2026-07-29]
datasets/distill_multiref/
  train_mixed.json         # M3 混合训练集(29,777 条, 真·多ref 40%)  [已生成 2026-07-29]
log/ref_distill/
  train.log                # M3 正式训练日志  [已完成 2026-07-29 23:21]
  checkpoint-{1000..4000}/ # M3 训练产物  [全部落盘 2026-07-29]
output/smoke_eval/
  smoke_compare.png        # M4 冒烟①总览:dreambench 2-ref 5 case × 3 变体  [已产出 2026-07-30]
  results.json             # 冒烟①配置 + 计时 + 案例清单
output/multibanana_eval_distill/
  compare_add.png          # M4 冒烟②总览:multibanana add 8 task × 3 变体  [已产出 2026-07-30]
  results_add.json         # 冒烟②配置 + 计时 + 任务清单
multibanana_eval/
  download_multibanana_single_two.py  # 多子集下载器(single + 11 个 2-ref 子集)  [已落地 2026-07-30]
datasets/eval_multiref/
  eval_set.json            # ⚠️ 两个版本,见 §11.0 偏离③
                           #   v1 = 232 任务 / 711 图 = commit ccd2a36^ ← M4 的 0.819 与 M5 全部标注的定义域
                           #   v2 = 277 任务(ccd2a36 起,S2 扩到 60)← 那 45 条 S2 没有图,M5 不使用
  noise_floor_tasks.json   # [§11 步骤 1] 35 条 = 30 条零假设(同 refs/prompt, seed+7,000,000)
                           #   + 5 条 S0 锚点(逐字复制,只供 --check_anchor,不参与配对)  [已落地 2026-08-02]
output/noise_floor/        # [§11 步骤 1] 45 张图 + results.json + boards/ ← 与 M4 产物**分目录**,理由见步骤 1
output/eval_multiref/
  results.json             # M4 全量配置 + 计时 + 711 条记录(0 失败)  [已产出 2026-07-31]
  boards/                  # 21 张 JPEG 拼图(711 张全分辨率 PNG 留在 H800,不进 git)
  blind_rond1.json         # M4 原始标注(按列表下标键控的旧格式)  [已产出 2026-07-31]
  pairs_m5r1.json          # [§11 步骤 0] R2 批次配对清单(198 条:148 post-vs-pre + 30 null + 20 replay)
  blind_annotations_m4r1.json  # [§11 步骤 0] M4 的 227 条迁移到 pair_id 键控
  blind_annotations_m5r1.json  # [§11 步骤 2] R2 批次标注结果
```

> **冻结纪律 [2026-08-02 新增]**:被声明冻结的文件(`eval_set.json` 等)
> **一律新开版本,不就地改**。`ccd2a36` 就地把 232 改成 277,导致 80 条标注下标偏移
> ——虽未污染 0.819(算于插入之前,已验证),但那是运气。

所有脚本:中文 docstring(仿 multibanana_eval 风格)、`--dry_run`、失败打日志不静默。

---

## 11. M5:尺子标定 + 混淆分离(2026-08-02 改写,**整节取代 07-31 版**)

### 11.0 决策来源、以及我在落地时发现的三处必须偏离

**决策来源**:`distill/M5_DECISIONS.md`,你 2026-08-02 的答复——
Q1 **甲+乙都做** / Q2 **B(分批门控)+ S1 取多 seed** / Q3 **A(先测噪声地板再定 X)** /
Q4 **B(修 3 项 + 换 seed + 支持重放)** / Q5 **A(v1/v2 双版并存)** /
Q6 **C(45 条 S2 不补跑,S2 维持 n=15)** / Q7 **A(臂的优先级等数据)**。

> Q7 你只选了 A,没选 C。**记录在案:与 mentor 的「贡献口径」沟通按你的决定推迟到
> 有数据之后**(原提案是 A+C 并行)。这条不影响下面任何步骤的执行。

**三处偏离**——都是落地时查代码发现原提案行不通,先声明,不同意就改:

**偏离① 甲(噪声地板)的配对方式必须换。**
原提案是「teacher 同组合**跨 seed** 自比」。查 `eval_set.json` 后发现行不通:
`S1_000_s0` 的 refs 是 `backpack_dog/00.jpg + bear_plushie/01.jpg`,
`S1_000_s1` 是 `01.jpg + 02.jpg`——**视角随 seed 变,两张图的参考图根本不是同一组**。
这不只是"偏保守",而是**判读题面不成立**:让你比两张图谁更忠于参考,却给不出
公共的参考图。
**改法**:新建 30 条任务,`image_paths` / `prompt` **逐字沿用**既有任务,只把 `seed`
加一个固定偏移,只跑 `official_full`。得到的是**同 refs、同 prompt、仅噪声不同**的
teacher 双生图——这才是干净的零假设。代价 30 张图 ≈ 3 分钟 GPU。

**偏离② 甲、乙 不单独成批,混进 ① 一起标。**
原提案把甲排成第一批 30 条单独做。这会**自毁**:整批都是同模型自比,你标到第五条
就知道了,判读行为随之改变(多半会多打平局),测出来的"噪声地板"不是真的地板。
**改法**:①(148)+ 甲(30)+ 乙(20)= **198 条打散成一批**,你不知道哪条是哪种。
代价一样,方法论上严格得多。**门控不受影响**:批次做完同时拿到"M3 有没有伤模型"
和"尺子刻度",分叉决策仍在这批之后。

**偏离③ Q5=A 与 Q6=C 在工程上冲突,M5 的标注域只能是 v1=232。**
Q5 选了 v1(232)/v2(277)并存、M5 起用 v2;Q6 选了不给扩出来的 45 条 S2 补跑图。
但 `eval_set.json` 盘上已是 277 条,而盲评服务器**启动时断言每条任务的候选图都存在
且可解码**(`server.py:117-133`)——那 45 条没有图,用 v2 一定启动失败。
**改法**:v2=277 记录为「**已构建、未填充图像、M5 不使用**」;
**M5 全部标注在 v1=232 上做**(= 当前 HEAD 的 `eval_set.json` 去掉那 45 条,
等价于 commit `ccd2a36^` 的版本)。等哪天决定补跑 S2,v2 直接可用。

---

### 11.1 M5 要回答的问题(按依赖排序)

| # | 问题 | 用什么回答 | GPU | 你的判读量 |
|---|---|---|---|---|
| **A** | **尺子有没有刻度**:同模型双生图会被判出多大差距?有没有左右偏好? | 甲 30 条零假设对 | 3 min | 30 |
| **B** | **你的重测信度**:同一对图两次判定一致率多少? | 乙 20 条重放 | 0 | 20 |
| **C** | **M3 到底有没有伤模型**(§6.3 机制假说 ③ 的直接检验) | ① post4000 vs pre 148 条 | 0 | 148 |
| **D** | **底座差距有多大**(混淆 ①) | ② pre vs teacher 148 条 | 0 | 148 |
| **E** | **隔离的净代价**(混淆 ②) | `official_iso` 探针 + 臂 B | 20 min / 6 h | 视门控 |
| **F** | **0.95 是否可达 / 配方是否有毒** | 臂 A | 6 h | 视门控 |

A–C 一批做完(步骤 2),D–F 全部**由这批结果门控**(步骤 4)。
**盘上已有 `ours_kv_pre` 的 232 张图、0 失败**,所以 C 和 D 都是**零 GPU 成本**——
这是 M4 留下的最大的现成资产,`M4_EVAL_SPEC.md:203` 本来就给 S3 预登记了
「不劣于 PRE」的验收标准,只是当初没执行。

---

### 11.2 新的报告口径(Q3 的可落地部分,现在就生效)

`(S+B)/(T+B)` 的三个毛病(§6.3 已列):含义随平局率漂移、`B` 把"都好"和"都烂"
合并、S1 的组合内相关。**从 M5 起改为**:

- **主报**:非平局胜率 `S/(S+T)` + **Wilson 95% CI**;
- **平局率单列**,不并进主指标;
- `(S+B)/(T+B)` 仍算,但**只作与 M4 的可比参考**,不作判据;
- S1 的显著性**必须**同时报组合级聚类后的版本(44 组合,不是 132 条独立样本)。

**达标判据的形式**已定,**数值 X 待步骤 3 定**:

```
预登记形式:  非平局胜率的 Wilson 95% CI 下界 ≥ X%
X 的确定依据: 步骤 2 测出的噪声地板与自洽率(Q3 选了 A)
```

**一个无法修复、只能声明的局限**:你读过 39 张冒烟图,知道「主体复制 = 学生」
「画面崩坏 = pre」这类签名。A/B 位置盲化挡不住**指纹识别**。
结论文档必须如实写明:**单标注者、指纹可识别条件下的盲评**。

---

### 11.3 分步执行计划

> **纪律**:一步一个确认点。**上一步的 ✅ 没打勾,不许开始下一步。**
> 每步末尾列出「产出」和「确认点要看什么」,确认点看的是**产出物**,不是我的口头汇报。

#### 步骤 0 — 盲评服务器重构 + 数据迁移(本地 / 我做 / 零 GPU / 不占你时间)

现状:`create_app`(:105)没有 `rond` 形参而 `main`(:331)传了、`blind_seed_for`(:336)
全文未定义 ⇒ **启动即崩**;`marks` 按 `str(idx)` 键控,而 `eval_set.json` 被中部插入
45 条,**已使 80 条标注的下标偏移**(idx147 标注时是 `S3_000_s0`,现在是 `S2_005_s0`)。

**改动范围**(既有代码,手册 R0,本地改):

1. **从「任务 × 两个硬编码变体」改成「配对清单驱动」。**
   这是把 Q4 的三项修复一次做对的最小结构改动:四类比较
   (post-vs-teacher / post-vs-pre / pre-vs-teacher / 同模型自比)在旧结构里
   是四种特例,在新结构里是同一件事的四个实例,重放也只是多一条 `pair_id`。
   清单 schema:

   ```
   meta:  batch_id, blind_seed, eval_set_version, eval_set_commit, question, created
   pairs[]: pair_id, kind, stratum, prompt, ref_paths[],
            left_key, right_key,          # 服务端语义标签,前端永不可见
            left_img, right_img, src_task_id
   ```
   槽位仍由 `md5(f"{pair_id}|{blind_seed}") % 2` 决定 `left_key` 落 A 还是 B。

2. **marks 改按 `pair_id` 键控**(不再是列表下标)。
3. **`TEACHER`/`STUDENT` 模块级常量删除**,身份来自清单的 `left_key`/`right_key`。
4. **换 blind seed → `m5-blind-v1`**。理由:你已在揭盲回顾模式下逐条看过全部 227 条的
   身份映射,沿用 `m4-blind-v1` 则每个 task 的学生方位与上轮完全相同,**盲法名存实亡**。
5. **重放支持**(Q4 选 B):同一对图允许出现两条不同 `pair_id`,槽位独立随机。
6. **迁移 M4 的 227 条**:旧 marks 的**值**里存了 `task_id`(我已验证),
   可无损重键为 `m4r1::tvs::{task_id}`,落到 `blind_annotations_m4r1.json`。
   **迁移后必须复算出 0.819 五行逐值吻合,否则迁移作废。**
7. 启动断言保留并加强:缺图/不可解码仍然**启动即炸**;另加
   `eval_set_version` 与实际图片数的一致性断言。

**产出**:改后的 `server.py`;`blind_annotations_m4r1.json`(迁移结果);
一份行为测试记录(启动、槽位稳定性、重放独立性、迁移复算)。
**✅ 确认点 0**:你看「迁移复算出的五行 == §6.3 表」这一条。对不上就别往下走。

#### 步骤 1 — 噪声地板图生成(H800 / ~3 min GPU / 绿档)

新建 `datasets/eval_multiref/noise_floor_tasks.json`:**30 条**
(S1 取 20 条、S3 取 10 条,确定性选取)。每条:

- `image_paths` / `prompt` / `meta` **逐字复制**对应的既有任务;
- `task_id` = `NF_{src_task_id}`;
- `seed` = 源任务 seed **+ 7,000,000**(固定偏移,写死);
- `variants` = `["official_full"]` ← 已知标签,**`eval_multiref.py` 不需要改**
  (`VARIANTS` 是模块级硬编码表,加新变体才要动它;这里没加)。

跑:`--eval_json datasets/eval_multiref/noise_floor_tasks.json`。
配对时 `left_img` = 既有的 `{src}__official_full.png`,`right_img` = 新的
`NF_{src}__official_full.png`。

**产出**:30 张图、`results.json`(要求 0 失败)、一张拼图。
**✅ 确认点 1**:拼图目检——两侧都应是**正常的 teacher 出图**。
若新图里出现崩坏,说明 seed 偏移撞上了坏区,换偏移重跑,**不要拿崩坏图当零假设**。

> **[2026-08-02 已落地]** 生成器 `distill/build_noise_floor.py` + 任务单
> `datasets/eval_multiref/noise_floor_tasks.json` 已写好并自检通过;
> 远端执行单见 `distill/M5_STEP1_RUN.md`。落地时相对上文有**两处补强**:
>
> 1. **`--save_path` 必须是 `output/noise_floor/`,不能复用 `output/eval_multiref/`。**
>    `eval_multiref.py:write_shard_results` 往 save_path 写 `results_shard0.json`,
>    同目录跑会**覆盖 M4 的 shard 记录**,随后 `--merge` 还会把 `results.json`
>    重算成只剩这 45 张。M4 产物冻结,不就地动(§10 冻结纪律)。
> 2. **任务单里多带 5 条 S0 锚点(逐字复制,3 变体共 15 张,+1.5 min)。**
>    WHY:零假设对的全部前提是"左右两张只差噪声"。左半边是 M4 期间生成的,
>    右半边是现在生成的——**中间隔着一次环境可能的漂移**。漂了的话差异就不止噪声,
>    地板测出来是环境差 + 噪声差的混合,而这一点在拼图上肉眼看不出来。
>    S0 锚点(`--check_anchor`,与 `output/smoke_eval/` 逐像素比,max ≤ 2)是现成的
>    仪器,直接把这个漏洞堵上。**它不参与盲评配对。**
>
> 选样规则(确定性,写死在脚本里):`md5(task_id|"nf-v1")` 排序后贪心取,
> **同组合/同主体只取一条** —— S1 命中 20 个不同组合、S3 覆盖全部 10 个主体。
> WHY 去重:20 条若挤在 3 个组合上,量的是"这 3 个组合有多稳",不是地板。
> `SEED_OFFSET = 7_000_000` 写死在脚本里、不做成命令行参数——可传就意味着
> 可以一直试到地板"好看"为止。

#### 步骤 2 — 标注批次 R2:198 条(**你的时间,本批是全流程的瓶颈**)

| kind | 比什么 | 条数 | 取样 |
|---|---|---|---|
| `post_vs_pre` | post4000 vs pre | **148** | S3 全 60 + S1 44 组合 × seed_slot {0,1} = 88 |
| `null_floor` | teacher vs teacher(仅噪声不同) | **30** | 步骤 1 的 30 条 |
| `replay` | M4 已标过的原对重放 | **20** | 从 227 条里确定性抽,槽位重掷 |
| | **合计** | **198** | |

S1 取 2 个 seed 是你的决定(Q2)——比 1 个 seed 多 44 条判读,换来组合内噪声可平均。
**S2 / S4 不进本批**:S2 维持 n=15 无功效(Q6=C),S4 平局率 75% 判不出东西。
198 条**打散**(`md5(pair_id|blind_seed)` 确定性排序),你看不出哪条是哪类。

**产出**:`blind_annotations_m5r1.json`。
**✅ 确认点 2**:我出三张表——
① **噪声地板**:null 对的平局率、非平局胜率、以及**左右偏好检验**
(零假设下胜率应为 50%,显著偏离 = 存在位置/顺序偏好,是要从所有其它数字里
扣掉的偏差项);
② **自洽率**:20 条重放与 M4 原判的一致率;
③ **post vs pre 分层结果**(非平局胜率 + Wilson CI + 平局率,S1 另报组合级聚类)。

> **提前写死怎么读 ①**:零假设下 `(S+B)/(T+B)` 的期望**本来就是 1.0**(对称性),
> 所以这一步**不是**在检验"0.95 可不可达"。它测的是三件事:
> **(a) 零假设平局率**——若你在真正等价的两张图上只有 20% 打平,说明大量非平局是
> 抛硬币,有效样本量远小于名义 n;**(b) 位置偏好**——偏离 50% 就是系统偏差;
> **(c) n=30 的分辨率**——决定后续每批要标多少条才够。
> 不要把这一步的结果说成"0.95 被证伪/证实"。

#### 步骤 3 — 定尺子、定 X(你 + mentor / 零成本 / 但必须先做完才准往下)

用步骤 2 的 ①②,把 §11.2 里的 X **写死**进 `M4_EVAL_SPEC.md`(新开一节 M5 判据),
并 commit。**写死之后不许再改**——这是 D02「份额失衡比」栽过的跟头:
判据事后调整,结论就没有可信度了。

**✅ 确认点 3**:判据条文进仓库,commit hash 记在本节。

#### 步骤 4 — 门控分叉(零成本,只是一个决定)

按步骤 2 的③,post4000 vs pre 三种走向,分别指向不同的下一步:

| 若 ③ 显示 | 含义 | 下一步 |
|---|---|---|
| **post 显著劣于 pre** | 机制假说 ③ 坐实,**M3 配方在损伤模型** | 走 **P2**(混比 40%→25% 或减半步数),②(pre vs teacher)不急 |
| **post ≈ pre**(CI 含 0.5) | 差距是**继承来的**,不是 M3 造的 | **②(pre vs teacher)立刻变成最关键的一组**,且臂 B 升级为主线候选 |
| **post 显著优于 pre** | M3 是**改善**,与冒烟证据一致 | 回退全部归因于底座 + 隔离 ⇒ 直接上 `official_iso` 探针 + 臂 A/B |

**✅ 确认点 4**:分叉决定写进本节,附当时的三张表。

#### 步骤 5 — 训练类实验(6 h 起,**排在所有零 GPU 判读之后**)

见 §11.4。**在确认点 4 之前一律不上机**——原因很直白:一次训练 6 小时,
而"M3 有没有伤模型"这个问题用盘上现成的图 + 你 148 条判读就能答,
先花 6 小时训练是拿最贵的资源去赌一个便宜就能知道的答案。

---

### 11.4 训练类实验清单(受确认点 4 门控)

| | 内容 | 成本 | 回答 | 状态 |
|---|---|---|---|---|
| **P-probe** | `official_iso`:官方 LoRA 直接开 `ref_isolation + kv_cache` 跑评测集 | ~20 min,**纯推理** | 隔离在**不重训**下本身损失多少(**从没测过**);官方权重能否当臂 B 的 init | 门控后**最先做**,最便宜 |
| **P0** | ckpt-1000/2000/3000 在 S1+S3 子集上扫一遍 | ~10 min GPU + 判读 | 质量税是否随步数单调;若是,**换 ckpt 即零成本修法** | 需改 `eval_multiref.py`(见下) |
| **P1** | **臂 A**:`official_full` + 同数据同配方 4000 步(`--ref_isolation False`) | ~6 h | 配方有毒 vs 学生有病;标定 0.95 可达性 | 门控 |
| **P3** | **臂 B**:`official_full` + `--ref_isolation True` 同数据 4000 步 | ~6 h | 隔离净代价(init/数据/配方全对齐) | **§6.3 证伪后重新升权**——原先降级为"确证性实验"的理由(隔离已证无代价)已不成立 |
| **P2** | 修复尝试:`--multi_ratio 0.4→0.25` 或减半步数 | ~6 h | 视门控选旋钮(§9 预登记动作) | 门控 |

#### P1 预登记预测(保持 07-31 原文,不因证伪而改)

> 臂 A 与原始 teacher 盲评,**预测 score 落在 0.8 附近**。
> - **落在 0.8** → 机制假说 ③ 成立:配方有毒且与架构无关。此时真正该报的数
>   不是 `student vs 原始 teacher`,而是 **`student vs 臂 A`**——双方交了同一笔
>   质量税,剩下的差才是隔离的代价。
> - **落在 0.95+** → 假说 ③ 被证伪:问题出在学生侧(ckpt-20000 底座或隔离)。

**臂 A 的固有盲区**:`train_mixed.json` 的 40% 多 ref 目标图**就是 teacher 自己生成的**,
余下 60% 是 UNO-1M 真实数据、官方也早训过。所以臂 A 近似"在自己的分布上原地踏步",
它能证明"数据没有毒性",**证明不了"数据足以教会另一个架构"**——学习负担不对等。
这是 P3 存在的理由,不能用臂 A 的结果替代。

#### 工程门禁:两道,不过不上机

**门禁① LoRA 键名核对(必须过,否则 P1/P3 两臂全废)**

`train.py:155` 是 `unwarp_dit.load_state_dict(lora_state, strict=False)`。官方 LoRA
的键名若与训练代码的 LoRA 参数名不符,**一个张量都不会加载、也不会报错**——
我们会以为在官方权重上续训,实际从随机初始化开始,**而且 loss 曲线上看不出来**。

`eval_multiref.py:170-178` 已趟过这条路:它不从文件读官方 LoRA,而是
`get_models(..., only_lora=True, lora_rank=512)` 挂载后**从活模型 state_dict 里按后缀
过滤提取**,所以提出来的键天然就是模型自己的命名。照此导出即可。

> **验收必须是与 `log/ref_isolation/checkpoint-20000/dit_lora.safetensors` 的
> 键集合做严格相等断言,不是"加载没报错"。**

**门禁② `official_iso` 探针**(已提升为 P-probe,见上表)。

#### 已验证的工程事实

- `scripts/train_distill.sh` 的 `RESUME_FROM_CHECKPOINT` / `MAX_TRAIN_STEPS` /
  `PROJECT_DIR` / `TRAIN_DATA_JSON` 均可由环境变量覆盖,末尾 `"$@"` 透传。
  `--ref_isolation True` 虽写死在 accelerate launch 行,但 `HfArgumentParser` 基于
  argparse、**重复 flag 后者胜**,故 `bash scripts/train_distill.sh --ref_isolation False`
  可覆盖——**不需要改既有脚本**(不触手册 R0)。
  [已验证:argparse 层本地实测;`HfArgumentParser` 对 bool 字段有自己的包装,
  上机前用一行 `parse_args_into_dataclasses` 打印确认,只读操作,绿档]
- LoRA 参数形状**不受** `ref_isolation` 影响(隔离只改注意力掩码与 KV 缓存)[已验证]。
- `train.py:145` 语义:传具体 safetensors 路径时 `global_step=0`,不恢复 optimizer
  [已验证,§5 沿用]。

#### P0 的工程障碍(需本地改代码)

`eval_multiref.py:53-59` 的 `VARIANTS` 是**模块级硬编码表**,`load_tasks`(:134)
对表外标签直接 `SystemExit`。要加 ckpt-1000/3000 就**必须动这个文件**——手册 R0,
**由本地改**,不是新建 json 能绕开的。
[修正 07-31 版的一处口误:换任务单的参数是 **`--eval_json`**,不是 `--tasks_json`;
`eval_multiref.py:464` 为准。步骤 1 的噪声地板只用已知标签 `official_full`,
所以**不受这个障碍影响**。]

---

### 11.5 贡献口径的提醒(Q7 = A:推迟沟通,但先记在这里)

若 P3(官方 init + 隔离 SFT)显著优于当前 post4000,主线故事会从
"**从零复现 stage-2**"变成"**从官方权重初始化 + 短程隔离适配 SFT**"。官方权重是
公开的,这不算作弊,工程上更实用——但这是**贡献口径的变化,不只是消融**,
需要与 mentor 明确确认后才改写论文主张。

**为什么它可能发生**:`ckpt-20000` 的 stage-1 数据没按 `score_final` 过滤,
官方只用满分数据 ⇒ 底座差距可能是继承来的。修自训底座是条长路,
而官方权重公开可用——"官方 init + 短程隔离适配"可能是到达
"1.69× 加速且质量对齐"这个工程目标的**更短路径**。

按你的决定(Q7=A),**这次沟通排在步骤 4 的分叉之后**。
若步骤 2 的③ 落在"post ≈ pre",这条提醒会立刻变成需要主动找 mentor 的事项。
