# M5 臂 A 执行单 —— 官方 init + 同数据同配方 4000 步

> 对应 `DISTILL_PLAN.md` §11.4 的 **P1**。**档位:🟡 黄档**——不改任何既有 `.py`/`.sh`
> (符合手册 R0),但要跑一次 6 小时的 8 卡训练,中途 kill 的代价很高。
> 总耗时:门禁① ~10 min + 标定 ~15 min + 正式训练 **~6.3 h**。
>
> **[2026-08-03 确认点 5] 臂 A 现在排第 1**。臂 B 押后——P-probe 证明它的起点
> 身份≈0,原设计跑出的空结果不可判读(见 §11.6)。臂 A 不受影响:
> 它全程 `--ref_isolation False`,与隔离结构无关。

## 这一步在干什么

回退归因还剩 **① 底座 gap**(我们的 `ckpt-20000` 是自训的,官方 stage-1 数据按
`score_final` 过滤过、我们的没有)。臂 A 把底座换成**官方权重**,数据与配方
一字不改地重跑 4000 步:

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
3. s/it 与 5.6 同量级。**明显更快要警惕**——全注意力比隔离注意力**更慢**才对
   (隔离省掉了 ref 段的交叉注意力),若反而快很多,先怀疑 ref 没被喂进去。

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

1. **本地**给 `eval_multiref.py` 的 `VARIANTS` 加一行
   `("arm_a_full", False, False, False, "arm_a")` 并把 `log/arm_a/checkpoint-4000`
   加进 LoRA bank ——**在本地改、push、H800 pull**(R0)。
   注意 `--dry_run` 的成本已改为按 `kv_cache` 位推导,新增变体天然覆盖,不会再有
   `KeyError`(commit `032957b` 的教训)。
2. 生成臂 A 的评测图(与 `official_full` 同批 seed、同一次会话)。
3. 新一批盲评配对清单 + 判读。**跨批次的尺子会漂**(SPEC §8.5-3,κ=0.274),
   所以臂 A vs teacher 必须**在同一批内**完成,不许拿 M4 的 37.9% 直接并排比。
4. 读到结果后**才**定臂 B 的 init 与步数,并在上机前预登记(§11.6 末尾的三个候选)。

## ✅ 确认点(用户来判)

- 门禁① 通过 ⇒ 臂 A 的 init 确实是官方权重,不是随机初始化;
- 步骤 1 打出 `False` ⇒ 跑的确实是臂 A 而不是臂 B;
- 4000 步零异常、4 个 checkpoint 落盘 ⇒ 可以进入评测环节。
