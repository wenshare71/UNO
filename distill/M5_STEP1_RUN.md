# M5 步骤 1 执行单 — 噪声地板图生成

> 对应 `DISTILL_PLAN.md` §11.3 步骤 1。**档位:🟢 绿档**——不改任何既有代码,
> 只是换一份任务单调用 `eval_multiref.py`。总耗时 ~11 min(其中 GPU denoise 3.5 min)。

## 这一步在干什么(先看懂再跑)

M4 得出 0.819,但**没人知道"两张真的没有差异的图"在盲评里会得几分**——
不知道地板在哪,0.819 是"学生差"还是"尺子抖"分不清。

所以造 30 组**零假设对**:同一个 teacher、同一组参考图、同一个 prompt,
**只有噪声 seed 不同**。这 30 组的标注结果就是尺子的刻度。

- 左半边 = M4 已有的 `output/eval_multiref/{src}__official_full.png`(**不重新生成**)
- 右半边 = 本次要生成的 `output/noise_floor/NF_{src}__official_full.png`

另带 5 条 S0 锚点(15 张图)。**它们不参与盲评**,只回答一个问题:
*从 M4 跑完到现在,环境有没有漂移?* 如果漂了,右半边和左半边的差异就不只是噪声了,
整个零假设对的前提就塌了。这是本步唯一的自动化闸门。

## 产出去哪

**`--save_path output/noise_floor`,不要用 `output/eval_multiref`。**
`eval_multiref.py:write_shard_results` 会往 save_path 写 `results_shard0.json`,
用同一个目录会**覆盖掉 M4 的 shard 记录**,之后 `--merge` 还会连带把
`results.json` 重算成只剩这 45 张。M4 的产物是冻结的,不许就地动。

## 步骤

### 1. 拉代码,确认任务单在

```bash
cd ~/UNO && git pull
python distill/build_noise_floor.py --verify
```

预期:`✓ 自检通过` + `任务 35 条 / 出图 45 张 (零假设对 30 组 + S0 锚点 5×3)`。
**对不上就停下来报告**,不要自己重新生成任务单(`build_noise_floor.py` 不带 `--verify`
会重写文件;那份 json 是本地生成并 commit 过的,远端只做校验)。

### 2. 空跑一遍(不碰 GPU)

```bash
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/noise_floor_tasks.json \
  --save_path output/noise_floor --dry_run
```

预期:`official_full 35 张`、`ours_kv_pre 5 张`、`ours_kv_post4000 5 张`、
`分层:{'S0': 5, 'S1': 20, 'S3': 10}`、`纯 denoise 合计 ≈ 3.5 min`。

### 3. 生成(后台 + 日志)

```bash
mkdir -p logs
nohup python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/noise_floor_tasks.json \
  --save_path output/noise_floor \
  > logs/m5_noise_floor.log 2>&1 &
echo "pid=$!"
```

模型加载 ~7 min 期间日志是静的,这是正常的;`模型就绪` 之后每张图一行。
按手册 §4 每 ~5 min 报一行心跳。

**验收:`失败 0`。** 有任何一张失败,把 `results_shard0.json` 的 `fails` 字段原样带回来。

### 4. 锚点自检(纯 CPU,必须过)

```bash
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/noise_floor_tasks.json \
  --save_path output/noise_floor --check_anchor
```

预期:15 行全 `✓`(max ≤ 2)。

**这一条超标就停,不要往下走,也不要"再跑一遍看看"。**
超标意味着环境相对 M4 漂了,那 30 组右半边图就不是纯噪声差异,
零假设失效——这时要报告的是 max/mean 的具体数值和环境信息(torch / 驱动版本),不是重跑。

### 5. 合并 + 拼图(纯 CPU)

```bash
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/noise_floor_tasks.json \
  --save_path output/noise_floor --merge
```

产出 `output/noise_floor/results.json` 与 `output/noise_floor/boards/*.jpg`。

> 这一步打印的 `vs teacher` 加速比**没有意义**(KV 变体只有 5 张,还都是 S0),
> 忽略即可。本步不测性能。

### 6. 带回来

把这几样贴回来(图走用户手动下载,不用 push):

1. `logs/m5_noise_floor.log` 的**末尾 20 行**(生成数 / 失败数 / 耗时);
2. 第 4 步锚点自检的**完整 15 行输出**;
3. `output/noise_floor/results.json` 的 `n_records` / `n_fails` / `timing`;
4. `output/noise_floor/boards/` 下的 jpg 文件名列表(用户自己取图目检)。

## ✅ 确认点 1(用户来判)

拼图目检:**左右两侧都应该是正常的 teacher 出图**。
若新图里出现画面级崩坏,说明 seed 偏移撞上了坏区——
改 `build_noise_floor.py` 的 `SEED_OFFSET` 重跑(**这个改动归本地,不在远端做**),
**不要拿崩坏图当零假设**:那会把地板测得比真实值低,后面所有比较都被系统性放宽。
