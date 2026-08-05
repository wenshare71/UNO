# M5 臂 B 执行单(二)—— 出图 → §9 身份留存门 → 终批 222 对

> 对应 `DISTILL_PLAN.md` §11.11(c) 的**步 3 / 4 / 5**。训练那一段见 `M5_ARM_B_RUN.md`。
> **档位:🟡 黄档** —— 要用 GPU(~19 min)与判读预算(5 min 或 ~21 min),但不改任何既有 `.py`/`.sh`。
>
> **前置**:`log/arm_b/checkpoint-4000/dit_lora.safetensors` 已落盘(4000 步已训完)。
>
> **本单的判读总预算按预登记就是两档**:门失败 **5 分钟**、门通过 **~21 分钟**。
> **没有第四个批次。**

---

## 这一步在干什么

链条第 ②′ 边、也是 08-04 之后的主命题:

> 同一份数据、同一个配方、同一个 init 下,
> **隔离相对全注意力的代价是多少**,换来 1.672× 加速。

`arm_b_iso`(隔离 + KV cache)对 `arm_a_full`(全注意力),**只差一个 flag**。

**但先过门**:§11.11(d) 预登记了一道客观关卡——如果臂 B 连主体身份都没保住,
两张图「一眼可辨」,§8 的偏好盲评就名不副实(P-probe 上已判过一次这个先例),
那 16 分钟判读预算不该花。

---

## 步骤 0:拉代码 + 确认产物(~1 min,只读)

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull

ls -la log/arm_b/checkpoint-4000/dit_lora.safetensors
export HF_HOME=/kaimm-distill/wuwenxuan/hf_cache
export HF_HUB_OFFLINE=1
```

本次 `git pull` 会带来三样新东西(**全是新文件,除 `build_pairs.py` 外没动过任何既有代码**):

| 文件 | 是什么 |
|---|---|
| `distill/build_arm_b_tasks.py` + `datasets/eval_multiref/arm_b_tasks.json` | 出图任务单(192 任务 / 222 张) |
| `distill/build_arm_b_gate.py` + `output/eval_arm_b/gate_items.json` | §9 门的标注清单(30 item / 51 问) |
| `distill/build_pairs.py` 新增 `arm-b` 子命令 | 终批 222 对(**门通过才建**) |

两个 json 已经在本地生成并自检过,H800 上**不需要重新生成**,只要校验:

```bash
python distill/build_arm_b_tasks.py --verify
```

预期:`✓ 自检通过(… seed 零偏移 … 对侧臂A产物齐备 …)` + 192 条 / 222 张 / 锚点 30 条。

---

## 步骤 1:出图(~19 min,**单卡**)

```bash
mkdir -p logs
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/arm_b_tasks.json \
  --save_path output/eval_arm_b \
  2>&1 | tee logs/m5_arm_b_eval.log
```

`--num_shards` 默认就是 1,单卡不用传。`--arm_b_lora` 默认已指向
`log/arm_b/checkpoint-4000/dit_lora.safetensors`,也不用传。
推理超参(512×512 / `ref_size` 512 / 25 步 / guidance 4.0 / bf16 不 offload)
全部是 default,**一个都不要手填**——手填就有填错的机会,而填错了整批不可比。

**WHY 单卡不分片**:纯 denoise 只有 11.9 min,而每个 shard 各自要付 ~7 min 模型加载。
分 8 片端到端反而更慢(8.5 min),还多出"某片漏跑"的对账麻烦。

本批生成 **222 张**:

| 变体 | 张数 | 单张 | 小计 |
|---|---|---|---|
| `arm_b_iso` | 192 | 2.907 s | 9.3 min |
| `official_full` | 30 | 5.132 s | 2.6 min |

`official_full` 那 30 张是 `run_floor` 锚点(同会话天花板),不是多余的。

### 合并结果(**必跑**,否则 `build_pairs.py arm-b` 会报 `❌ 缺少 output/eval_arm_b/results.json`)

生成那一步只写 `results_shard0.json`,`results.json` 由 `--merge` 产出。

```bash
python distill/eval_multiref.py \
  --eval_json datasets/eval_multiref/arm_b_tasks.json \
  --save_path output/eval_arm_b --merge --no_board
```

> ⚠️ **`--no_board` 不能省。** 不加的话 `--merge` 会顺手拼出
> `output/eval_arm_b/boards/*.jpg` —— 那是**带变体名列头**的并排图,即已揭盲图。
> 本批还要走 §8 盲评,把它拼出来摆在盘上就是个随时会被打开的坑,而"看过"不可撤销。
> (旧批次拼图是判读**之后**的事,不是同一情形。`.gitignore` 也不放行本批的 `boards/`。)

### 要看的三件事(**这是本单唯一能抓住"隔离没生效"的信号**)

```bash
python - <<'PY'
import json, statistics, collections
d = json.load(open("output/eval_arm_b/results.json"))
by = collections.defaultdict(list)
st = collections.Counter()
for r in d["records"]:
    st[(r["variant"], r["status"])] += 1
    if r.get("denoise_s"):
        by[r["variant"]].append(r["denoise_s"])
print("状态:", dict(st))
for v, xs in by.items():
    print(f"{v:<16} n={len(xs):>3}  中位 {statistics.median(xs):.3f}s")
print("失败:", d.get("fails", [])[:5])
PY
```

**1. 状态:`arm_b_iso` 192 个 ok、`official_full` 30 个 ok,`fails` 为空。**

**2. 单张耗时必须落在隔离档,不是全注意力档。**

```
official_full(全注意力)   5.132 s   ← 臂 A 批 192 张实测中位
official_iso (隔离+KV)     2.907 s   ← P-probe 192 张实测中位
arm_b_iso                  预期 ≈ 2.9 s
```

⇒ **`arm_b_iso` 中位若贴到 5.1 s,说明隔离/KV 没开,整批作废,停下上报。**
(方向与训练侧**相反**:训练里隔离更慢,加速全部来自推理侧 KV cache。别记反。)

**3. 本批同时出现两个变体,`swap_lora` 应当只发生 1 次**——日志里搜 `swap` 确认。
两次以上说明变体循环被打乱了,不致命但要报。

---

## 步骤 2:§9 身份留存门(判读 **5 min**,**在任何偏好盲评之前**)

```bash
python distill/build_arm_b_gate.py --verify     # 30 item / 51 问 + 图片可解码
python -m distill.idcount.server \
  --items output/eval_arm_b/gate_items.json \
  --marks output/eval_arm_b/gate_marks.json --port 8011
```

标注完:

```bash
python -m distill.idcount.report \
  output/eval_arm_b/gate_items.json output/eval_arm_b/gate_marks.json
```

### 判定口径(**预登记,出图之后不许改**)

判定看 **per-subject 留存的点估计**,现成锚点同尺同题:

| | per-subject | 含义 |
|---|---|---|
| `official_iso` | **0 / 51 = 0.0%** | 隔离未适配的地板 |
| `official_full` | **45 / 51 = 88.2%** | 全注意力的天花板 |
| **`arm_b_iso`** | **?** | 本次 |

| 留存(**点估计**) | 判定 |
|---|---|
| **< 60%** | **终批取消,项目在此收口。** 结论直接写:**4000 步补不完隔离适配,边 ②′ 承重、边 ③ 小,留存率就是读数。** 判读总支出 5 分钟 |
| **≥ 60%** | 进步骤 3 |

**用点估计不用 CI**:51 问在 60% 处 Wilson 半宽约 ±13pp,用 CI 判两边都不确定,
门就失去决断力。**这是决策规则,不是测量值。**

> **诚实声明(随门一起引用,不许省)**:这把 §9 尺子只在**极端对比**(0% vs 88%)上
> 验证过,§11.6「不许外推到细微差别」明写在案。落在 **50–70%** 这个带里时它**最不可靠**。

---

## 步骤 3:终批 222 对(**门通过才做**,判读 ~16 min)

```bash
python distill/build_pairs.py arm-b
python -m distill.blind_eval.server \
  --pairs output/eval_arm_b/pairs_m5ab.json --port 8765
```

标注完:

```bash
python -m distill.blind_eval.report \
  output/eval_arm_b/pairs_m5ab.json \
  output/eval_arm_b/blind_annotations_m5ab.json
```

**组成(§11.11(e) 预登记)**:

| kind | 内容 | 条数 |
|---|---|---|
| `arm_b_vs_arm_a` | `arm_b_iso`(key_0 = 被检验方)vs `arm_a_full`(key_1 = 基线) | **192** |
| `run_floor` | 本批同会话的 30 张 `official_full` ↔ 臂 A 批对应 30 张 | **30** |
| ~~`replay`~~ | **砍掉**(自洽率已测两次,第三次不承重) | 0 |

新盲种 `m5-arm-b-v1`,问题逐字沿用(`哪一张更好?(综合参考图忠实度与画面质量)`)。

### 三条上机前就知道的事(不是看到结果才说的)

1. **判据很可能「不适用」**:§8.2 要求 `n_nontie ≥ 94`,而 192 对在平局率 > 51.0% 时就跌破。
   臂 A 那批实测 **59.4%** 已经越线。**不许事后追加样本**(那等于事后调判据)。
   §11.11(e) 只登记了批次组成,**没有**登记"平局率是主读数"——判读完按 §8.2 原样算,
   落在哪一档报哪一档;可主张的句式见 §11.11(g)⑥。
2. **那 30 条锚点的 prompt/refs 会在批内出现两次**(一次 `arm_b` 对、一次 `run_floor` 对)。
   躲不掉:天花板必须跨"与被检验那一对相同的会话间隔",而臂 A 批只覆盖 S1+S3。
   **处置**:`build_pairs.py:spread_runfloor` 在 30 个 `run_floor` **槽位之间**做确定性重排,
   把两次出现的最小间距从 3 拉到 **89**(批长 222 的 40%,超过既有纪律 `len//3`=74),
   而槽位本身不动 ⇒ 锚点三分位分布仍是 10/13/7,**尾段不聚堆**。
   即便如此,"同一 prompt 出现两次"这件事本身**仍要写进 §8.5 局限**。
3. **拼图不看、不传**(同步骤 1 的警告)。

---

## 步骤 4:带回来

1. `logs/m5_arm_b_eval.log` 的**首 30 行**(bank 加载 + 变体循环)与**末 20 行**;
2. 步骤 1 那段状态/耗时脚本的完整输出(尤其 `arm_b_iso` 的中位耗时);
3. `python -m distill.idcount.report …` 的完整表格;
4. 门通过的话,再加 `python -m distill.blind_eval.report …` 的完整表格;
5. `output/eval_arm_b/results.json`、`gate_marks.json`、`pairs_m5ab.json`、
   `blind_annotations_m5ab.json` —— **提交进 git**(图片不提交,拼图更不提交)。

### ✅ 确认点(用户来判)

- `arm_b_iso` 中位耗时 ≈ 2.9 s 而不是 ≈ 5.1 s ⇒ 跑的确实是隔离 + KV,不是又一个全注意力;
- 222 张零失败 ⇒ 可以进门;
- 门的点估计落在 60% 的哪一侧 ⇒ 决定项目是在这里收口,还是再花 16 分钟判读。
