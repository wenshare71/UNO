# M6 步骤 4 执行单 — 两腿出图 + 350 对盲评

> 对应 `distill/M6_ABLATION_SPEC.md` §8 的 P4。**档位:🟢 绿档**——不改任何既有代码,
> 只是用已经提交好的脚本投两个 infer_hub 任务、再在本机建清单起盲评服务。
> GPU ~50 min(两个单卡 job),判读 ~25 min。P2/P3 已收官,见 `M6_STEP2_REPORT.md`。

## 这一步在干什么

把两条从零训出来的腿摆到同一批图里,交给人判。**这是 M6 唯一的一次评测**
(SPEC §7-2:只做一次,用最后一个 checkpoint),跑完就是结论。

```
m6_iso   log/ref_distill_iso/checkpoint-4000    隔离 + KV cache   ← key_0 被检验方
m6_full  log/ref_distill_full/checkpoint-4000   全注意力,不开 KV  ← key_1 基线
```

`m6_full` **不许开 KV**(SPEC §3 硬约束:它没有隔离,cache 是有损的;速度对比是另一件事,
不许混进质量批)。其余推理超参(512×512 / ref_size 512 / 25 步 / guidance 4.0 / bf16 不 offload)
全是 `eval_multiref.py` 的 default,**一个都不要手填**——手填就有填错的机会,而填错了整批不可比。

---

## 本单的三条预登记(**落盘于任何一张图生成之前**)

SPEC §5 已冻结,以下三条是它留白处的细则,写在这里、在出图前提交,**结果出来后一律不许改**。
待 P5 出报告时按房规(§7.4)以带日期的条目并入 SPEC。

### 1. 任务池 192 → 320 的扩张规则

SPEC §5.2 登记了"主对 320 条",没登记这 320 条怎么构成。规则是**把既有循环的取值范围
拉长,不引入任何新公式**:

| 层 | 构成 | 条数 | 与 M4 任务单的关系 |
|---|---|---|---|
| S1 | 44 组合 × `object_tpl[k % 20]` × s∈{0,1,2} | 132 | **逐字相同** |
| S1x | 同 44 组合 × `object_tpl[(k+10) % 20]` × s∈{3,4} | 88 | 新增 |
| S3 | 10 主体 × c∈{0,1,2} × s∈{0,1} | 60 | **逐字相同** |
| S3x | 同 10 主体 × c∈{3,4} × s∈{0,1} | 40 | 新增 |

- **seed 公式原样**(S1 `3_500_000+k*10+s`、S3 `3_700_000+i*100+c*10+s`),只是 s/c 的
  取值范围变大 ⇒ 与既有 seed 零碰撞,仍落在 `SEED_RANGE_S1_S4` 内、与 M1 区间不重叠。
- `(k+10) % 20` 是 20 条模板轮转的**对跖点**:每个 k 都换到另一个场景,且没有可挑的
  自由度(选 +1 还是 +7 就成了一次无规则的自由选择)。
- 多 ref : 单 ref = **220 : 100 = 68.75% : 31.25%**,与 192 批的 132:60 **完全同比**
  ⇒ "扩了池子"与"换了题型"这两件事不会混在一起。
- 核心 192 条的 prompt / refs / seed 与臂 A/B 批**逐字相同**,`build_m6_tasks.py --verify`
  拿 `eval_set.json` 逐字段对一遍 ⇒ 满足 SPEC §3「推理 seed 与臂 A/B 批逐字相同」。

`stratum` 仍只写 `S1` / `S3`,扩张与否记在 `meta.m6_ext`——`stratum` 是 `report.py` 的
分层键也是前端可见字段,新造 `S1x` 会让本批分层表与历史批次对不上。

### 2. run_floor 的第二侧走**另一个 job**

臂 B 的天花板是跨会话的,因为它的主对两侧本来就来自两个批次。M6 不是:`m6_iso` 与
`m6_full` 是**同一进程、同一张卡、变体外层循环**里先后生成的。在同一个 job 里再生成一次
同权重同 seed 的图,大概率**逐位相同** ⇒ 天花板退化成 100% 平局,尺子等于没有。

所以第二侧另起一个 job(另一进程/另一张卡),权重取 `m6_full`(主对的基线侧)。
**随结论必须一起声明**:

> 主对两侧同进程(run 噪声 ≈ 0)、天花板两侧跨进程 ⇒ **本批天花板是噪声的上界,偏保守。**

天花板不取臂 B 用的 `official_full`:那批选官方权重是因为它由 pipeline 自带备份提供、
不依赖任何 checkpoint 在不在盘上;M6 两腿的 checkpoint 本来就必须在盘上(它们是主对本身),
那条理由在这里不成立。取基线侧,量的正是骑在主对上的那份噪声。

### 3. 判据一个字不改

SPEC §5.1,`key_0` = `m6_iso`、`key_1` = `m6_full`:

```
非平局胜率 p̂ = win_0 / (win_0 + win_1) 的 Wilson 95% CI 下界 ≥ 0.40
且  n_nontie ≥ 94
```

按边 ③ 实测 53.6% 平局率折算 `n_nontie ≈ 148`。平局率若 > **70.6%**,`n_nontie` 跌破 94
⇒ 结论是**「判据不适用」**而非「不达标」,**不许事后追加样本**(SPEC §5.2 / §11.7)。

---

## 阶段 0 · 上机前的五条核对

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull

echo "=== 1 commit 已 push? ===";  SHA=$(git rev-parse HEAD); echo $SHA; \
                                   git branch -r --contains $SHA
echo "=== 2 两腿 ckpt ===";        ls -lh log/ref_distill_{iso,full}/checkpoint-4000/dit_lora.safetensors
echo "=== 3 ckpt 字节级校验 ===";  for L in iso full; do \
                                     python3 scripts/check_lora_ckpt.py \
                                       log/ref_distill_$L/checkpoint-4000/dit_lora.safetensors; done
echo "=== 4 任务单复核 ===";       python3 distill/build_m6_tasks.py --verify
echo "=== 5 配对清单预演 ===";     python3 distill/build_pairs.py m6 --dry_run
echo "=== 6 输出目录必须是空的 ==="; ls -d output/eval_m6 output/eval_m6_floor 2>&1
```

逐条对:

| 检查 | 期望 |
|---|---|
| 1 commit | 第二行要列出 `origin/main`。列不出来 = 没 push,推理机会报 `prepare_commit_not_found`(冒烟第一次就栽在这)。**把这个 40 位 sha 抄下来,后面写死用它,不要在提交命令里 `rev-parse HEAD` 现取** |
| 2 ckpt | 两个文件都在,各约 1.8 G |
| 3 校验 | 两次都是 `304 张量,空分片 0,全零 0`。任意一条不过就停,报回来 |
| 4 任务单 | `✓ 自检通过(… 核心 192 条与 M4 逐字一致 …)`;主池 320 / 天花板 30 |
| 5 清单 | `350 对 {'m6_iso_vs_m6_full': 320, 'run_floor': 30}`,`achieved_min_gap` **≥ 116**,三分位大致 10/11/9 |
| 6 目录 | **两个都应该不存在**。已经存在说明这批跑过一次——停下报回来,不要在旧目录上续跑 |

> 第 4 条是本单最重要的一次校验:它把"核心 192 条与臂 A/B 同 seed"这条前提**在出图前**
> 钉死。本地(4090)已经跑过一遍并通过,H800 上再跑一次是因为 `eval_set.json` 不进 git,
> 两台机器上的它是各自 build 出来的。

## 阶段 1 · 投主 job(640 张,~50 min)

提交端要 root 写队列、且必须保持 wuwenxuan 的 HOME 让 git 读到代理配置
(冒烟实测,两条都踩过):

```bash
U=/kaimm-distill/wuwenxuan/UNO
SHA=<阶段 0 第 1 条抄下来的 40 位 sha>

sudo env HOME=/kaimm-distill/wuwenxuan PATH=/kaimm-distill/infer_hub/lib:$PATH \
  python3 /kaimm-distill/infer_hub/lib/infer_submit \
  --owner wuwenxuan --project m2v-aio --cluster h \
  --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
  --weights     $U/log \
  --output-dir  $U/output/eval_m6 \
  --uv-env      $U/.venv-uno \
  --label m6eval_Iter4000 \
  --gpus 1 --timeout 75 \
  --prep-cmd    'true' \
  --prep-marker $U/log \
  --cmd 'python distill/eval_multiref.py --eval_json datasets/eval_multiref/m6_tasks.json --save_path $INFER_OUTPUT_DIR --m6_iso_lora $INFER_WEIGHTS_DIR/ref_distill_iso/checkpoint-4000/dit_lora.safetensors --m6_full_lora $INFER_WEIGHTS_DIR/ref_distill_full/checkpoint-4000/dit_lora.safetensors' \
  --dry-run
```

**先带 `--dry-run` 看 job json,没问题再去掉重投。** 被拒就停下报原样错误。

### 几个参数为什么这么填

- `--weights $U/log`:一个 job 要同时挂两腿的 LoRA,而 `--weights` 只收一个路径。
  指到 `log/` 之后两条腿都在 `$INFER_WEIGHTS_DIR` 底下,`--cmd` 里就不需要出现任何
  `/kaimm-distill/` 字面量——**硬规矩 3 禁止 `--cmd` 出现未声明的共享盘绝对路径**。
  `log/` 按约定只读,不会被动。
  > 冒烟那次 `--weights` 指的是单个 checkpoint 目录,这次指的是整个 `log/`(~640 GB,
  > 两腿 stage-1 各 100 个 ckpt 都在里面)。提交端只做路径前缀校验、worker 只把它注成
  > 环境变量,理论上与大小无关。**万一提交端或 worker 因为目录太大报错,停下报回来**
  > ——处置是本地建一个只含两腿最终 ckpt 的小目录再指过去,不要自己在远端改脚本。
- `--eval_json datasets/eval_multiref/m6_tasks.json` 是**代码目录相对路径**:任务单已经
  提交进 git,worker checkout 出来就有。ref 图走 `datasets/dreambooth` 子模块,
  worker 自动 `submodule update --init --recursive`(冒烟实测 9s 全部就绪)。
- `--prep-cmd 'true' --prep-marker $U/log`:m2v-aio 队列 `submit_require_prep: true`,
  强制两阶段。本任务没有独立切分步骤,按官方解法给 `true` + 一个必然存在的目录,
  提交时即判 `prep_done=true`,直接排卡。
- `--timeout 75`:640 张 × 5.1s/2.9s ≈ **43 min** denoise + 96s 加载(冒烟实测)。
  到点若 GPU 还在算会自动延长(最多 3 倍),不会被误杀。
- `--gpus 1` **不分片**:denoise 只有 43 min,分片省下的时间抵不过每片各付一次加载,
  而且 shard 越多越容易出「某片漏跑」的对账麻烦(臂 B 那批的既有判断)。
- `--label m6eval_Iter4000` 是**中性名**,不带 iso/full 臂名——控制台任务树对所有人可见。

## 阶段 2 · 投天花板 job(30 张,~5 min)

主 job 跑完再投,**不要合成一个 job**——分开正是"另一次 run"的全部意义所在(见预登记 2)。

```bash
sudo env HOME=/kaimm-distill/wuwenxuan PATH=/kaimm-distill/infer_hub/lib:$PATH \
  python3 /kaimm-distill/infer_hub/lib/infer_submit \
  --owner wuwenxuan --project m2v-aio --cluster h \
  --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
  --weights     $U/log \
  --output-dir  $U/output/eval_m6_floor \
  --uv-env      $U/.venv-uno \
  --label m6floor_Iter4000 \
  --gpus 1 --timeout 20 \
  --prep-cmd    'true' \
  --prep-marker $U/log \
  --cmd 'python distill/eval_multiref.py --eval_json datasets/eval_multiref/m6_floor_tasks.json --save_path $INFER_OUTPUT_DIR --m6_full_lora $INFER_WEIGHTS_DIR/ref_distill_full/checkpoint-4000/dit_lora.safetensors'
```

天花板 job 只用到 `m6_full` 一个 bank,`--m6_iso_lora` 不传(脚本只检查本批用到的 bank)。

### 盯任务

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
infer_status --owner wuwenxuan
tail -f /kaimm-distill/infer_hub/queues/m2v-aio/logs/wuwenxuan__m6eval_Iter4000__*.log
```

## 阶段 3 · 合并 + 三件必看的事(纯 CPU,在本机跑)

生成那一步只写 `results_shard0.json`,`results.json` 由 `--merge` 产出;
`build_pairs.py m6` 读的是 `results.json`,不跑这步会直接报缺文件。

```bash
cd $U
python3 distill/eval_multiref.py --eval_json datasets/eval_multiref/m6_tasks.json \
  --save_path output/eval_m6 --merge --no_board
python3 distill/eval_multiref.py --eval_json datasets/eval_multiref/m6_floor_tasks.json \
  --save_path output/eval_m6_floor --merge --no_board
```

> ⚠️ **`--no_board` 一个都不能省。** 不加的话 `--merge` 会顺手拼出带**变体名列头**的
> 并排图 —— 那就是揭盲图。SPEC §7-5 明写"判读完成前不得生成带变体名的拼图",
> 而"看过"不可撤销。`.gitignore` 也不放行本批的 `boards/`。

```bash
python3 - <<'PY'
import json, statistics, collections
for d in ("output/eval_m6", "output/eval_m6_floor"):
    r = json.load(open(f"{d}/results.json"))
    by, st = collections.defaultdict(list), collections.Counter()
    for x in r["records"]:
        st[(x["variant"], x["status"])] += 1
        if x.get("denoise_s"):
            by[x["variant"]].append(x["denoise_s"])
    print(f"\n== {d} ==\n状态: {dict(st)}")
    for v, xs in by.items():
        print(f"  {v:<10} n={len(xs):>3}  中位 {statistics.median(xs):.3f}s")
    print("失败:", r.get("fails", [])[:5])
PY
```

**1. 状态**:主批 `m6_iso` 320 ok + `m6_full` 320 ok、天花板 `m6_full` 30 ok,`fails` 为空。
任何一个不齐就停下报回来——320 对不完整就不许开评。

**2. 单张耗时必须一腿一档,这是唯一能抓住"隔离没生效"的信号**:

```
m6_full  预期 ≈ 5.1 s   ← 全注意力档(臂 A 批 192 张实测中位;H200 上可能快到 ~4.4s)
m6_iso   预期 ≈ 2.9 s   ← 隔离 + KV 档(P-probe 192 张实测中位)
```

⇒ **`m6_iso` 中位若贴到 `m6_full` 那一档,说明隔离/KV 没开,整批作废,停下上报。**
两者比值应在 **1.6–1.8×** 之间。(方向与训练侧**相反**:训练里隔离更慢,加速全部来自
推理侧 KV cache。别记反。)

**3. `swap_lora` 应当发生 2 次**——日志里搜 `swap` 或看 `LoRA bank:` 那行确认。
臂 B 那批是 1 次(它有一侧用官方 bank),M6 两腿都是自己的 checkpoint,所以是 2 次,
**别照着旧日志对**。

## 阶段 4 · 建清单 + 盲评(判读 ~25 min)

```bash
python3 distill/build_pairs.py m6
python3 -m distill.blind_eval.server --pairs output/eval_m6/pairs_m6.json --port 8765
```

标注完:

```bash
python3 -m distill.blind_eval.report \
  output/eval_m6/pairs_m6.json output/eval_m6/blind_annotations_m6.json
```

组成:

| kind | 内容 | 条数 |
|---|---|---|
| `m6_iso_vs_m6_full` | `m6_iso`(key_0 被检验方)vs `m6_full`(key_1 基线) | **320** |
| `run_floor` | 主批的 `m6_full` ↔ 天花板 job 的同 30 张 | **30** |

新盲种 `m6-iso-v1`,问题逐字沿用(`哪一张更好?(综合参考图忠实度与画面质量)`)。

### 三条上机前就知道的事(不是看到结果才说的)

1. **蒸馏 target 是官方全注意力 teacher 生成的 ⇒ 目标分布对基线腿有利**
   (SPEC §6 的方向性偏置,在任何一腿启动之前就写下了)。隔离腿打平或胜出 ⇒ 结论更硬;
   隔离腿落败 ⇒ **不能**直接归因于隔离本身,必须在报告里写明这条偏置。
2. **本批天花板偏保守**(预登记 2)。
3. **那 30 条锚点的 prompt/refs 在批内出现两次**(一次主对、一次 run_floor),躲不掉。
   处置:`spread_runfloor` 在 30 个槽位之间做确定性重排,最小间距 **136**(批长 350 的 39%,
   超过既有纪律 `len//3` = 116),三分位 10/11/9 不聚堆。即便如此,这件事本身仍要写进 §8.5 局限。

## 带回来

1. `infer_status` 里两个 job 的终态行(状态 / duration / avg_gpu_util / 执行机);
2. 主 job 日志的**首 30 行**(bank 加载 + 变体循环)与**末 20 行**;
3. 阶段 0 六条核对的完整输出(尤其第 4、5 条);
4. 阶段 3 那段状态/耗时脚本的**完整输出**——`m6_iso` 与 `m6_full` 的中位耗时和比值;
5. `python3 distill/build_pairs.py m6` 的完整输出(构成表 + 间距 + 折算的 `n_nontie`);
6. `python3 -m distill.blind_eval.report …` 的完整表格;
7. `output/eval_m6/results.json`、`output/eval_m6_floor/results.json`、`pairs_m6.json`、
   `blind_annotations_m6.json` —— **提交进 git**(图不提交,拼图更不提交);
8. 任何一步的原样报错(不要转述)。

## 不要做的事

| 不要 | 为什么 |
|---|---|
| 把主 job 拆成 iso / full 两个 job | 主对两侧同进程是本批的设计(run 噪声 ≈ 0),拆开就把 run 噪声灌进主对,天花板那条声明也随之失效 |
| 把天花板并进主 job | 同权重同 seed 同进程大概率逐位相同,天花板退化成 100% 平局,尺子作废 |
| 开 watchdog 常驻 | `USAGE.md` §3 推荐,但对本项目有毒:自动投递会把中途结果摆上控制台,破 SPEC §7.1 |
| `--merge` 不带 `--no_board` / 打开 `boards/` | 带变体名列头 = 已揭盲,SPEC §7-5;"看过"不可撤销 |
| 因为 `n_nontie` 不够就补样本 | SPEC §5.2 / §11.7 明禁。不够就报「判据不适用」 |
| 改 `distill/*.py` 或 `scripts/*.sh` | R0:既有文件无例外,不看改动大小。报上来本地改完 push |
| 在提交命令里 `rev-parse HEAD` 现取 sha | 冒烟栽过一次:本地 commit 没 push,推理机 mirror 找不到,报 `prepare_commit_not_found` |
| label 里带 iso / full | 控制台任务树对所有人可见,盲评约束 |
| 在已存在的 `output/eval_m6*` 上续跑 | `already_done()` 会跳过旧图,那批图来自哪个 commit 说不清 |

## ✅ 确认点(用户来判)

- 阶段 0 第 4 条 `✓ 自检通过` 且含「核心 192 条与 M4 逐字一致」⇒ 与臂 A/B 同 seed 这条前提成立;
- `m6_iso` 中位耗时 ≈ 2.9 s 而不是 ≈ 5.1 s,比值 1.6–1.8× ⇒ 跑的确实是隔离 + KV;
- 670 张零失败 ⇒ 可以开评;
- `achieved_min_gap ≥ 116` ⇒ 锚点不会被标注者认出来;
- 判读完的 `n_nontie` 落在 94 的哪一侧 ⇒ 决定报「读数」还是报「判据不适用」。
