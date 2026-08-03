# M5 P-probe 执行单 — 隔离代价探针图生成

> 对应 `DISTILL_PLAN.md` §11.4 P-probe。**档位:🟢 绿档**——不改任何既有代码,
> 只是换一份任务单调用 `eval_multiref.py`。总耗时 ~32 min(其中 GPU denoise ~25 min,单卡)。

## 这一步在干什么(先看懂再跑)

M4 的回退归因里有一整项是"隔离本身有没有代价",但这个组合——
**官方 LoRA(未重训)+ ref_isolation + KV cache**——从没跑过。
`official_full`(不隔离)和 `ours_kv_*`(隔离 + 重训)之间的差距,
分不清有多少是"隔离这个动作本身"的代价、多少是"没重训"的代价。

所以同一次会话里生成两个变体:
- `official_full`——不开隔离(M4 已知配置)
- `official_iso`——开隔离 + KV cache,但**权重还是官方的、没重训**

**seed 完全相同**,于是两张图之间只剩 `ref_isolation` 这一个 flag 的差别,
谁减谁都干净。`official_iso` 已经登记在 `eval_multiref.py:VARIANTS` 里
(commit `dfdb74f` 之后新增的一行),本次不需要改那个文件。

任务单选样:`eval_set.json` 的 S1 全部 132 条 + S3 全部 60 条 = 192 条,
不抽样、不打散。选这个数是为了让非平局样本数折合后 ≥94,
够得上 `M4_EVAL_SPEC.md` §8.2 的判据线(见 `build_probe_iso.py` 顶部注释,
--verify 输出里也会打印折算值)。

**顺带产出**:两个变体用同一批 seed、同一批参考图跑,`--merge` 打出来的
`speedup_vs_teacher` 就是这次能同场测出的**隔离本身的加速比**——
是这次上机的第二个产出,不用额外操作。

## 产出去哪

**`--save_path output/probe_iso`,不要用 `output/eval_multiref`。**

`eval_multiref.py:write_shard_results` 会往 `save_path` 写
`results_shard{N}.json`,`--merge` 再把它们汇总成 `save_path/results.json`。
用 `output/eval_multiref` 会**覆盖掉 M4 的 shard 记录**,`--merge` 还会把
M4 的 `results.json` 连带重算成只剩本次这 192 条。这条教训是从步骤 1
(`M5_STEP1_RUN.md`)照搬过来的——那一步险些因为存到同一目录把 M4 的
产物毁掉,这次直接把独立目录写进 runbook,不留犯错空间。M4 的产物是
冻结的,不许就地动。

## ⚠️ H800 网络的坑(必须先确认,不确认就不要跑)

H800 上**直连 huggingface.co 不通,走代理会卡死**——这个坑已经在
`scripts/keepalive_infer.py` 踩过一次。加载权重必须走本地离线缓存,
靠的是 `distill/eval_multiref.py` 文件顶部(约第 44-46 行)这两行:

```python
os.environ.setdefault("HF_HOME", "/kaimm-distill/wuwenxuan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
```

**开跑前先确认这两行还在**(`git pull` 之后可能被别的改动误删过)。
本次不需要新写任何加载模型权重的脚本,`eval_multiref.py` 里已经有——
但如果之后有人在这次探针的基础上加新脚本,**只要那个脚本会加载模型权重,
就必须照抄这两行**,否则会在拉 huggingface.co 时直接卡死,没有报错、
也没有超时,只能人工 kill。

## 步骤

### 1. 拉代码,生成并校验任务单

```bash
cd ~/UNO && git pull
python distill/build_probe_iso.py --verify
```

预期:`✓ 自检通过` + `任务 192 条 / 出图 384 张` + `分层:{'S1': 132, 'S3': 60}`。

**如果 `probe_iso_tasks.json` 还不存在**(本地生成但没 commit 上来),
先跑一次不带 `--verify` 的 `python distill/build_probe_iso.py` 把它生成出来,
再跑 `--verify` 确认。**对不上就停下来报告**,不要凭感觉改 `build_probe_iso.py`
里的常量重新生成——那些数(192 = S1 132 + S3 60)是按判据倒推出来的硬编码,
改了就是换一批任务,后面的分析对不上号。

### 2. 空跑一遍(不碰 GPU)

```bash
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/probe_iso_tasks.json \
  --save_path output/probe_iso --dry_run
```

预期:`official_full 192 张`、`official_iso 192 张`、
`分层:{'S1': 132, 'S3': 60}`,纯 denoise 合计与
`build_probe_iso.py --dry_run` 打印的单卡估算(~25 min)量级一致。
**若发现要跑几小时,说明哪里错了(LoRA 反复搬 / 没做变体外层循环),停下上报。**

### 3. 生成(后台 + 日志)

```bash
mkdir -p logs
nohup python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/probe_iso_tasks.json \
  --save_path output/probe_iso \
  > logs/m5_probe_iso.log 2>&1 &
echo "pid=$!"
```

模型加载 ~7 min 期间日志是静的,这是正常的;`模型就绪` 之后每张图一行。
按手册惯例每 ~5 min 报一行心跳。

**验收:`失败 0`。** 有任何一张失败,把 `results_shard0.json` 的 `fails`
字段原样带回来,不要自己重跑掉的那几张——先报告,等判断是否要整体重跑。

### 4. 合并 + 拼图(纯 CPU)

```bash
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/probe_iso_tasks.json \
  --save_path output/probe_iso --merge
```

产出 `output/probe_iso/results.json` 与 `output/probe_iso/boards/*.jpg`。

这一步会打印一张表,`official_iso` 那一行的 `vs teacher` 列**这次是有意义的**
(不像步骤 1 里 KV 变体只有 5 张 S0 样本那种"忽略即可")——两个变体在同一批
192 条任务上跑满,这就是隔离本身的加速比读数。

### 5. 带回来

把这几样贴回来(图走用户手动下载,不用 push):

1. `logs/m5_probe_iso.log` 的**末尾 20 行**(生成数 / 失败数 / 耗时);
2. `--merge` 打印的那张表——两个变体各自的 **mean / median 耗时、peak 显存、
   vs teacher 加速比**;
3. `output/probe_iso/results.json` 的 `n_records` / `n_fails` / `timing`;
4. `output/probe_iso/boards/` 下的 jpg 文件名列表(用户自己取图目检)。

## ✅ 确认点(用户来判)

拼图目检:`official_full` 与 `official_iso` 两侧应该都是正常的官方 LoRA
出图,画面主体、构图应高度接近(同 seed、同参考图,只差一个隔离 flag)——
明显的画面级差异本身就是"隔离代价"的直接证据,记录下来,不要当成异常。
若出现解码失败或画面崩坏(纯灰块、纯噪声),那是生成失败而不是"隔离的代价",
连同 `fails` 字段一起报告,不要放进后续统计。
