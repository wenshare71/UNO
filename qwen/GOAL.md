# 在 Qwen-Image-Edit-2511 上做隔离注意力 / ref KV 缓存蒸馏 · 目标

> **这份文件说的是「要什么、为什么、已知什么、边界在哪」,不是执行计划。**
> 怎么切分步骤、先做哪一步,由你和用户在这个目录里定。
>
> 写于 2026-08-11。仓库根在 `..`(ByteDance UNO 的 fork),本目录是新 teacher 的地盘。

---

## 1. 一句话

把「参考图段隔离注意力 + 参考 KV 缓存」这套架构,在 **Qwen-Image-Edit-2511** 上重做一遍,
并用仓库里已有的盲评装置量出它的**质量代价**。

---

## 2. 机制:隔离注意力为什么能缓存

多参考图生成里,每张参考图的 token 默认要和噪声图、文本做全注意力。改成**隔离**之后:

1. ref token **只自注意**(mask 掉 ref → txt/img 的注意力);
2. ref 段用**固定 t=0 的调制**,不跟着去噪时间步走。

两条**必须同时成立**。只做 (1),ref 的 hidden state 仍随时间步调制漂移;只做 (2),ref 仍
看得见含噪图像,逐层被带偏。两个原因一起杀掉,ref 的 K/V 才真正**步不变**,于是可以在
第 0 步算一次、后续所有步直接读——无损缓存。

代价是原始权重不认这个结构,**必须重训**。整个项目就是在回答:这笔重训花多少、买回多少。

参考实现(FLUX 版,91 行,别直接搬,架构对不上):`../uno/flux/ref_attention.py`

---

## 3. 为什么换 teacher

上一轮(M0–M6)在 UNO / FLUX.1-dev 上做完了,结论:

| 结论 | 数 |
|---|---|
| 推理加速 | **1.672×**,无条件成立 |
| 原命题「加速且质量对齐 teacher」 | **已被 M4 证伪**(50/132,CI 上界 0.464 < 0.5,p = 0.0067) |
| 隔离 vs 全注意力(M5 臂B vs 臂A) | 2-ref 非平局胜率 **29.9%**(p = 0.00054);1-ref 测不到 |
| 隔离消融收官(M6 P4) | m6_iso 非平局胜率 **33.3%**,CI 下界 0.250 < 判据 0.40,**判据不达标** |
| 身份留存 | 46/51 = 90.2%,与全注意力 45/51 **无法区分** |
| 我们自己复刻的 stage-1 底座 | 50.6%,**不承重**——掉的东西几乎全记在隔离头上 |

细节和原文在 `../distill/OVERVIEW.md`(叙事)与 `../distill/DISTILL_PLAN.md`(法定文本,
两者冲突以后者为准)。上表的口径不完全同批,引用前去核对。

**换 teacher 的理由**:代价确实存在,但底座太弱——UNO 本身在多主体上就不强,
测到的 33% 里分不清多少是「隔离的代价」、多少是「底座本来就到不了」。换一个明显更强的
teacher 重做,这个混淆项才消得掉。

**外部对照(重要)**:BFL 的 `FLUX.2-klein-9b-kv` 已经把同样的工程做出来了
(step-0 写、step-1+ 读,1.21×–2.66× 加速),说明**工程可行性不需要我们再证**。
但 BFL **一个质量数字都没公布**——没有胜率、没有身份留存、没有 FID。
我们的 M4–M6 盲评装置恰好就是补这笔账的仪器。**这是这个项目现在的立足点。**

---

## 4. teacher:Qwen-Image-Edit-2511

| 项 | 值 |
|---|---|
| 规模 | 20B MMDiT,60 层 dual-stream |
| 许可 | **Apache 2.0**(所以蒸出来的权重可以公开) |
| 入口 | `QwenImageEditPlusPipeline`(diffusers **main**,0.40.0.dev0,不是 release) |
| 多参考方式 | ref 图沿**序列维拼接**进 joint attention |
| 位置编码 | `QwenEmbedRope` |
| bf16 权重 | 57.72 GB / 35 文件 |

**结构上和 UNO 同构**(都是 ref token 拼进序列做联合注意力),所以隔离 mask 的思路能搬;
但实现完全不共用——`../uno/flux/` 是自己写的 FLUX,Qwen 走 diffusers。

⚠️ **一个必须记住的混淆项**:Qwen-Image-Edit 除了把 ref 拼进序列,**还有一条
Qwen2.5-VL 的语义通路**在同时喂条件。也就是说参考信息有**两条路**进模型。
只隔离序列那条,VL 那条还在——「隔离」这个操作在这个模型上到底隔离掉了什么,
在动手前必须先想清楚,否则测出来的代价不知道是谁的。UNO 上没有这个问题。

---

## 5. Q1 已经做完的:teacher 裸基线(上界)

用现有 UNO/M6 测试数据在 Qwen 上裸跑了一遍,拿到 teacher 自己的天花板。
后面所有 student 数字都跟这个比。

**runner**:`../scripts/infer_qwen_edit.py`(口径写死在 `CONSTANTS` 块里,不可从命令行调)

| 参数 | 值 |
|---|---|
| 任务表 | `../datasets/eval_multiref/m6_tasks.json` |
| 子集 | `i % 8 == 0` → 320 取 **40** 条(S1 2-ref × 28 + S3 1-ref × 12) |
| steps / true_cfg | 40 / 4.0 |
| negative_prompt | `" "` |
| 分辨率 | 1024×1024(Qwen 原生;UNO 侧是 512²,**刻意不同**) |
| prompt | **原样**,不加前缀不改写 |
| seed | 取自 task 自带的 `seed` |

**结论(目测,2026-08-11)**:整体比 UNO 好不少。产物在 H 机
`/kaimm-distill/wuwenxuan/output/qwen_baseline`,**还没拉回仓库**——见 §9。

### 5.1 teacher 自己就做不到的 7 条

用户逐张看完 40 张后标出来的。**这批是豁免集**:student 在这些样本上做不到,
不算 student 的错,评测时要单独摘出来。

| task_id | 主体 | 失效 |
|---|---|---|
| `M6_S1_003_s1` | backpack_dog + **candle** | 身份漂移 |
| `M6_S1_006_s2` | backpack_dog + **duck_toy** | 身份漂移 |
| `M6_S1_014_s2` | bear_plushie + **duck_toy** | **主体丢失**(整个不见了) |
| `M6_S1_019_s1` | berry_bowl + colorful_sneaker | 身份漂移 |
| `M6_S1_032_s0` | **candle** + fancy_boot | 身份漂移 |
| `M6_S1_033_s3` | **candle** + grey_sloth_plushie | 身份漂移 |
| `M6_S1_035_s1` | clock + **duck_toy** | 身份漂移 |

**分布(算过的,不是印象)**:

- **7 条全部落在 S1(2-ref),12 条 1-ref 零问题**。7/28 = 25%。
- 按主体统计失效率:**candle 3/6 = 50%**、**duck_toy 3/6 = 50%**、backpack_dog 2/6 = 33%,
  其余 7 个主体都在 0–20%(`can` 是 0/5)。
- `M6_S1_032_s0` / `M6_S1_033_s3` 里 candle 是 **ref[0]**,不是第二个。
  **所以失效跟参考图位置无关,跟主体本身有关。**

**失效模式**:7 条里只有 1 条是真丢失,其余 6 条是**退化到类别原型**——用户原话
「数据集中的蜡烛变成了普通的蜡烛」。candle 和 duck_toy 的共同点是**身份靠纹理/图案承载、
类别本身极其通用**;而 `can`(有醒目包装)一次都没坏。

这条观察对蒸馏有用:teacher 的身份保持失效是**漂移**不是**遗漏**,
而隔离注意力削弱的正是 ref → img 的信息通路。**两者是否会叠加,是个该测的问题。**

---

## 6. 一个已知的架构叉口,建议一开始就做成开关

ref 段内部的 mask 有两种写法,上一轮**从没分开测过**:

| | 我们的(`build_isolated_attn_mask`) | BFL klein-9b-kv |
|---|---|---|
| ref 段 | **块对角**,ref_i 看不见 ref_j | **整段自注意**,refs 互相可见 |

1-ref 时两者**完全等价**——这正好解释了 M6 为什么在 1-ref 上什么都测不到。
两者只在 2-ref 及以上分叉,而那恰恰是我们测到 33% 代价的地方。

所以 M6 那个代价里,有多少是「隔离」、有多少是「refs 之间也被隔离」,**目前是混在一起的**。
新 teacher 从零开始,把它做成一个开关,这笔账就能拆开。别再写死。

---

## 7. 上一轮留下的、可以直接用的东西

不要重造。这些是这个项目最值钱的部分:

| 位置 | 是什么 |
|---|---|
| `../distill/` | 12.5k 行:任务构建 / 评测 / **盲评闭环** / 可视化。M4–M6 全靠它 |
| `../distill/blind_eval/` | 盲评服务器 + 配对 + 报告(`report.py` 可离线复算) |
| `../distill/viewer/` | 判读用的图片浏览器 |
| `../multibanana_eval/board.py` | 对比拼图,`infer_qwen_edit.py` 已经在复用 |
| `../datasets/eval_multiref/` | M6 任务表(320 条,分层 S1/S3) |
| `../datasets/dreambooth/` | 主体图源 |
| `../docs/infer_hub/` | 远程推理队列用法 |
| `../distill/REMOTE_AGENT_HANDBOOK.md` | **远程 agent 的红/黄/绿档规程,必读** |
| `../reports/20260811-0951-q1-env-glitch/REPORT.md` | Q1 上机踩的所有坑 + 最终提交命令 |

---

## 8. 机器与环境

| | |
|---|---|
| 本地 4090 | `aiplatform-bjy-ge47-391`,8× RTX 4090 24GB,Ubuntu 20.04,**glibc 2.31** |
| 远程推理 | infer_hub 队列,`--cluster h`(H800 80GB),只认已 push 的 commit |
| env | `/kaimm-distill/wuwenxuan/envs/qwen-edit`(kling-mini 底座,py3.11,torch 2.5.1+cu124,diffusers 0.40.0.dev0) |
| 权重 | `/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511` |

**已知坑**(都在 Q1 REPORT 里,别重踩):

- `aio_n26` / `v4moe` 这两个公共 env 在 4090 上 **import torch 直接挂**(UCX 要 GLIBC_2.33)。用 kling-mini 系。
- 4090 **做不了 Qwen 生成**:20B transformer bf16 ≈ 40GB > 24GB,
  `enable_model_cpu_offload` 是按组件整体搬的粗粒度 offload,必 OOM。4090 只能做加载/验证。
- 提交 job 时 `LD_LIBRARY_PATH` 要自己在 `--cmd` 里导出,别赌 worker 注入。
- `--output-dir` 必须显式给,默认值会往权重目录里写。

---

## 9. 现在缺的东西

**Q1 的产物还在 H 机,没回到仓库。** 需要拉回 `results.json` + 拼图,
并补一份 Q1 结果报告(跑了什么口径、40 条是哪 40 条、§5.1 那 7 条的图)。

「比 UNO 好不少」目前只是口头判断,仓库里零凭据。而这是接下来**所有** student 数字的锚点——
按这个项目一贯的规矩(`OVERVIEW.md` 开头:「所有承重数字都在」),它必须有出处。
**这件事应该排在写训练代码前面。**

---

## 10. 边界

- **不动 `../uno/`**。它是 M4–M6 结果的可复现凭据,冻结。Qwen 的代码全写在本目录。
- **不改 Q1 的口径**(steps / true_cfg / 分辨率 / negative_prompt / prompt 文本)。
  改了这批基线就废了,后面没有东西可比。
- **`.gitignore` 是白名单模式**:`output/` 默认全忽略,逐批显式放行且每条写明理由
  (盲评纪律:未判读的批次,带变体名的拼图不许进 git)。Qwen 系列沿用,别开例外。
- **远程 agent 按 `REMOTE_AGENT_HANDBOOK.md` 的红/黄/绿档走**,R0 = 不许碰已有的 `.py`/`.sh`。
