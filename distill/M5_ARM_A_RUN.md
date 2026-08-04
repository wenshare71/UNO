# M5 臂 A 执行单 —— 官方 init + 同数据同配方 4000 步

> 对应 `DISTILL_PLAN.md` §11.4 的 **P1**。**档位:🟡 黄档**——不改任何既有 `.py`/`.sh`
> (符合手册 R0),但要跑一次 6 小时的 8 卡训练,中途 kill 的代价很高。
> 总耗时:门禁① ~10 min + 标定 ~15 min + 正式训练 **~6.3 h**。
>
> **[2026-08-03 确认点 5] 臂 A 现在排第 1**。臂 B 押后——P-probe 证明它的起点
> 身份≈0,原设计跑出的空结果不可判读(见 §11.6)。臂 A 不受影响:
> 它全程 `--ref_isolation False`,与隔离结构无关。

## 这一步在干什么

~~回退归因还剩 **① 底座 gap**(我们的 `ckpt-20000` 是自训的,官方 stage-1 数据按
`score_final` 过滤过、我们的没有)。~~

**[2026-08-04 修正,原文保留见上]** 这句把臂 A 的作用说大了。臂 A 与 post4000 之间
同时差**底座**和**隔离**两个变量,单独用它读不出底座 gap——那正是 §11.6 作废掉的
可加模型。臂 A 实际测的是**「配方施加在一个已经对齐的底座上有没有代价」**,
即三边链条 `official_full → 臂A → 臂B → post4000` 的**第 1 条边**;底座 gap 是第 3 条边,
要等臂 B 才拿得到。完整表述见 §11.7。

臂 A 把底座换成**官方权重**,数据与配方一字不改地重跑 4000 步:

| | init | 数据 | 配方 | 注意力 |
|---|---|---|---|---|
| M3 post4000(已有) | `ckpt-20000`(自训) | `train_mixed.json` | 同 | 隔离 |
| **臂 A**(本次) | **官方 LoRA** | `train_mixed.json` | 同 | **全注意力** |

**臂 A 的固有盲区(原文见 §11.4,这里重申一遍免得读结果时忘了)**:
`train_mixed.json` 的 40% 多 ref 目标图**就是 teacher 自己生成的**,余下 60% 是
UNO-1M 真实数据、官方也早训过。所以臂 A 近似"在自己的分布上原地踏步",
它能证明"数据没有毒性",**证明不了"数据足以教会另一个架构"**。

## ⛔ 步骤 0:门禁① —— 不过不上机

`train.py:155` 是 `unwarp_dit.load_state_dict(lora_state, strict=False)`。
**键名对不上时一个张量都不会加载、也不会报错**,我们会以为在官方权重上续训,
实际从随机初始化开始,而且 **loss 曲线上看不出来**(两者都从高处平滑下降)。
6 小时白烧且事后无法从产物分辨,所以这道门禁必须先过。

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull

# 先只比对,不写文件(加载 ~7 min)
python distill/export_official_lora.py --compare_only
```

预期最后一行:`✅ 门禁① 通过:键集合严格相等、形状一致、非全零。`

**任何一项不过就停下上报,不要绕过。** 脚本没有 `--force`,这是有意的。
三种失败的含义:

| 报错 | 含义 |
|---|---|
| `ckpt 多出 N 个键` | 那些参数在臂 A 里会是**随机初始化**的 ← 正是要防的 |
| `官方多出 N 个键` | 臂 A 会带上对照里没有的参数,两边不同构 |
| `形状不一致` | 先查 `--lora_rank` 是不是 512 |
| `全零张量` | `only_lora` 挂载没生效,导出它等于随机初始化 |

通过后再导出:

```bash
python distill/export_official_lora.py
```

产出 `log/official_init/dit_lora.safetensors`。脚本会**回读复核**——真正喂给
训练的是磁盘上这个文件,不是内存里那个 dict,dtype 在序列化时被改掉之类的问题
只有回读才看得见。

## ⚠️ HF 离线环境变量:训练侧没有,必须 shell 层补

`train.py` 和两个训练 shell **都没设** `HF_HOME` / `HF_HUB_OFFLINE`
(全仓 16 个加载权重的脚本里只有 `eval_multiref.py` / `run_attn_diag.py` /
`keepalive_infer.py` 这 3 个设了)。H800 上直连 huggingface.co 不通、走代理会**卡死**
——没有报错、也没有超时,只能人工 kill。6 小时的任务卡在第 0 分钟最冤。

**不改 `train.py`**(往训练脚本里硬编码 H800 专有路径,换机器反而更糟),
在 shell 层 export:

```bash
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
```

`os.environ.setdefault` 会给已存在的环境变量让路,所以这与其它脚本里的默认值不冲突。

## 步骤 1:确认 `--ref_isolation False` 真的覆盖得掉(只读,1 min)

`scripts/train_distill.sh` 的 accelerate 行里写死了 `--ref_isolation True`,
末尾 `"$@"` 把我们追加的 `--ref_isolation False` 排在它后面。argparse **重复 flag 后者胜**
(已在本地验过),但 `HfArgumentParser` 对 bool 字段有自己的包装,上机前用一行确认:

```bash
python - <<'EOF'
from transformers import HfArgumentParser
from train import TrainArgs
a, = HfArgumentParser(TrainArgs).parse_args_into_dataclasses(
    ["--ref_isolation", "True", "--ref_isolation", "False"])
print("ref_isolation =", a.ref_isolation, "← 必须是 False")
EOF
```

**打出 `True` 就停下上报**——那说明 bool 覆盖不成立,臂 A 会变成臂 B,
6 小时跑出一个我们没打算做的实验。

## 步骤 2:100 步标定(~15 min,不污染正式目录)

```bash
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
MAX_TRAIN_STEPS=100 \
CHECKPOINTING_STEPS=50 \
PROJECT_DIR=log/arm_a_calibration \
RESUME_FROM_CHECKPOINT=log/official_init/dit_lora.safetensors \
bash scripts/train_distill.sh --ref_isolation False
```

预期(对照 M3 实测):稳态 **~5.6 s/it**,100 步含加载 ~10 min。

**要看的三件事:**
1. `[preflight]` 六行全过,特别是 `resume checkpoint: log/official_init/... (N 个张量)`
   —— N 要与门禁①打印的张量数一致;
2. 日志里 `ref_isolation` 的取值(若脚本有打印)或至少确认步骤 1 已验过;
3. ~~s/it 与 5.6 同量级。**明显更快要警惕**——全注意力比隔离注意力**更慢**才对
   (隔离省掉了 ref 段的交叉注意力),若反而快很多,先怀疑 ref 没被喂进去。~~

   **[2026-08-04 订正,方向写反了]** 括号里那句「隔离省掉了 ref 段的交叉注意力」
   在**训练**里不成立,已逐行核过:

   - `train.py:415-423` 的 loss 前向是 `dit(..., ref_isolation=args.ref_isolation)`,
     **没有 `ref_kv`**,`TrainArgs` 里也根本没有 `kv_cache` 字段
     ⇒ **训练永远不用 KV cache,ref token 每一步都在序列里**,序列长度两边完全一样;
   - `ref_attention.py:83-90` 建的是一个**稠密 (L,L) bool 掩码**,
     `math.py:50` 把它当 `attn_mask` 交给 `scaled_dot_product_attention`
     ⇒ 隔离**一个 FLOP 都没省**,只是把能走 FlashAttention 融合路径的调用
     降级成了带任意 bool 掩码的慢后端;
   - `model.py:206-227` 还额外给 ref 段算一套 t=0 调制向量。

   ⇒ **训练里隔离严格更贵。1.672× 那个加速全部来自推理侧的 KV cache,与掩码无关。**

   **正确的判据方向:全注意力(臂 A)应当比隔离(M3 的 5.61 s/it 壁钟)更快。
   若臂 A 跑出 5.6 附近或更慢,那才该怀疑 `--ref_isolation False` 没覆盖住。**
   臂 B 是隔离,读它的 s/it 时同理反过来看。

标定完把 `log/arm_a_calibration` 留着别删,它是"配置确实生效过"的证据。

## 步骤 3:正式 4000 步(~6.3 h,后台 + 日志)

```bash
mkdir -p logs
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
nohup env \
  PROJECT_DIR=log/arm_a \
  RESUME_FROM_CHECKPOINT=log/official_init/dit_lora.safetensors \
  MAX_TRAIN_STEPS=4000 \
  CHECKPOINTING_STEPS=1000 \
  bash scripts/train_distill.sh --ref_isolation False \
  > logs/m5_arm_a.log 2>&1 &
echo "pid=$!"
```

- `PROJECT_DIR=log/arm_a` —— **不要用 `log/ref_distill`**,那是 M3 的产物,
  是冻结的。同名会把 checkpoint 覆盖掉。
- 每 1000 步一个 checkpoint,共 4 个。
- 按手册惯例每 ~30 min 报一行心跳(step / loss / s-per-it)。

**中途异常的处理**:训练崩了不要自己重启。把最后 50 行日志带回来——
6 小时的任务重启一次就是半天,得先判断是配置问题还是偶发。

## 步骤 4:带回来

1. `logs/m5_arm_a.log` 的**首 40 行**(preflight 六项 + 加载 + 前几步 loss)
   与**末 20 行**(总步数 / 总耗时 / 最终 loss);
2. `ls -la log/arm_a/` —— 4 个 checkpoint 目录都在、`dit_lora.safetensors` 大小正常;
3. 稳态 s/it,与 M3 的 5.25 对照;
4. 中途有无 NaN / OOM / NCCL 警告。

## 之后要做的(**本执行单不含,列在这里免得漏**)

臂 A 训完只是拿到了权重,还没有读数。后续需要:

1. ~~**本地**给 `eval_multiref.py` 的 `VARIANTS` 加一行~~ **✅ 已完成 [2026-08-04,本地]**:
   `VARIANTS` 新增 `("arm_a_full", False, False, False, "arm_a")`、新增 `--arm_a_lora`
   (默认 `log/arm_a/checkpoint-4000/dit_lora.safetensors`),并把 bank 的存在性检查
   改为**只查本批任务用到的**——否则没跑过臂 A 的机器连 P-probe 都启动不了。
   三个任务单的回归已本地验过(P-probe / 臂 A / M4 各自只检查该查的 bank)。
2. ~~生成臂 A 的评测图~~ **✅ 任务单已生成**:`distill/build_arm_a_tasks.py`
   → `datasets/eval_multiref/arm_a_tasks.json`(192 条 = S1 132 + S3 60,
   `official_full` + `arm_a_full` 两变体,seed 零偏移,自检 + 回读复核通过)。
   设计理由与预登记性质的说明见 `DISTILL_PLAN.md` **§11.7**。
   **两个变体必须在同一次会话里一起生成**,不许复用 M4 的 `official_full`。
3. 判读 192 条,按 SPEC §8.1/§8.2 现有口径,**不新增判据**。
   平局率 > 51.0% 时 n_nontie 跌破 94 ⇒ 结论是「判据不适用」,
   **且平局率本身是主读数**(对照零假设对的 33.3%),见 §11.7。
4. 读到结果后**才**定臂 B 的 init 与步数,并在上机前预登记(§11.6 末尾的三个候选,
   §11.7 补了第四个:分段——先纯单 ref 做隔离结构适配,再进混合数据)。

### 上机命令(**等确认点全过之后**再发)

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull

# ① 任务单已随 git 带过来,先自检一遍(纯 CPU,秒级)
python distill/build_arm_a_tasks.py --verify

# ② 成本核对(不碰 GPU)
python distill/eval_multiref.py --eval_json datasets/eval_multiref/arm_a_tasks.json --dry_run
#    预期:official_full 192 张 + arm_a_full 192 张,合计 ≈ 32.6 min 单卡

# ③ 8 shard 并行生成(--save_path 换新目录,不覆盖 M4/P-probe 产物)
#    注意 official_full 与 arm_a_full 必须在**同一次运行**里出,不许分两次
```

## 步骤 4 实际读数 [2026-08-04,`logs/m5_arm_a.log` 889 行,本地复核]

| 项 | 实测 | 判定 |
|---|---|---|
| preflight 训练 json | `train_mixed.json`(29777 条) | ✅ 与 M3 同一份 |
| **preflight resume** | `log/official_init/dit_lora.safetensors`(**304 个张量**) | ✅ 与 `ckpt-20000` 的 304 一致 ⇒ init 是官方权重 |
| 日志正文 | `Resuming from checkpoint log/official_init/dit_lora.safetensors` | ✅ |
| 步数 | **4000/4000** | ✅ |
| 壁钟 | **5:23:50**(平均 **4.86 s/it**,稳态 4.60–5.42) | ✅ 见下 |
| checkpoint | `Saved state` × 4,`checkpoint-{1000,2000,3000,4000}` | ✅ 四个齐全 |
| loss | 起 0.859 → 末 **0.245**,全程区间 **0.159–1.020** | ✅ 平滑下降 |
| 异常 | NaN / OOM / Traceback / RuntimeError **各 0 次** | ✅ |
| NCCL | 唯一一处是 `Backend: nccl` 的环境声明,非警告 | ✅ |

**s/it 的读法(按上面订正后的方向)**:

```
M3 post4000(隔离)  6:14:00  =  5.61 s/it
臂 A     (全注意力) 5:23:50  =  4.86 s/it      ← 快 13.4%
```

**方向正确。** 隔离在训练里更贵(稠密掩码挡掉 FlashAttention + 多一套 t=0 调制,
且序列长度两边一样),所以臂 A 更快是预期内的;**跑出 5.6 附近才是警报**。

**第二个独立信号(佐证,非铁证)**:`train.py:505/514` 的预览推理传的是
`kv_cache=args.ref_isolation`。日志里预览稳定在 **1.93–1.94 it/s**;若 `ref_isolation`
其实是 `True`,预览会连带开 KV cache 而明显更快。**但这条只能算佐证**——预览跑在
ZeRO-3 参数切分下,耗时被 all-gather 主导,KV cache 的 1.672× 会被稀释,
稀释到什么程度没有实测,而且本地没有 M3 的日志可作同尺对比。

**仍缺一项(不阻塞,但要补进记录)**:步骤 1 那条
`HfArgumentParser` bool 覆盖自检当时打印的是不是 `False`。日志里不含它(是另一条命令)。

## ✅ 确认点(用户来判)

- 门禁① 通过 ⇒ 臂 A 的 init 确实是官方权重,不是随机初始化;
- 步骤 1 打出 `False` ⇒ 跑的确实是臂 A 而不是臂 B;
- 4000 步零异常、4 个 checkpoint 落盘 ⇒ 可以进入评测环节。
