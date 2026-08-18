# PPT 示例图取物清单 —— 给远程会话的执行单

> **档位:🟢 绿档。** 不改任何 `.py` / `.sh`,不跑任何模型,不产生任何新的实验数字。
> 就是把已经存在的 PNG 挑 62 张出来、压小、交回。
>
> 用途:给 `reports/FINAL_REPORT.md` 配图做 PPT。**清单里每一张图对应报告里的哪个数字,
> 下面每组的表头都写了。** 不要自己换样本 —— 换了就对不上报告。

---

## 0. 你在做什么

最终报告已经交了,但里面所有的生成结果都还在机器上,仓库里一张主结果的图都没有。
这一单只做一件事:**按下面写死的 62 个文件名去取图**。

**为什么是写死的 62 个,而不是「挑一些有代表性的」**:这些样本是本地按**已经冻结的盲评标注**
选出来的 —— 哪一张是标注者判平局、哪一张判 teacher 赢、哪一张判学生赢,全部来自 git 里的
`blind_annotations_*.json`,不是印象。你在机器上重新挑会挑出另一批,报告里的数字就配不上图了。
**这一单没有留白给你填。**

### 盲评纪律在这一单上已经解除(先读这段,免得你被 `.gitignore` 那条规矩卡住)

`qwen/PLAN.md` §5 / `.gitignore` 里那条「**判读完成前,带变体名的拼图不许进 git**」是防揭盲的。
本单涉及的四个批次**判读全部完成并冻结**,标注文件本身就在 git 里:

| 批次 | 标注文件(已在 git) | 判读完成 |
|---|---|---|
| P-probe(§9 客观计数) | `output/probe_iso/idcount_marks.json` | 2026-08-03 |
| M4 | `output/eval_multiref/blind_annotations_m4r1.json` | 2026-07-30 |
| 臂 B vs 臂 A | `output/eval_arm_b/blind_annotations_m5ab.json` | 2026-08-05 |
| P3(第二轮主批) | `output/p3_eval/blind_annotations_p3.json` | 2026-08-15 |

⇒ **带变体名的文件现在可以进 git。** 这不是给你开的例外,是那条规矩的前提已经不成立了。
但 `.gitignore` 的白名单模式照旧:新放行的路径要**逐条写明理由**(§4)。

---

## 1. 什么时候停

**默认动作是继续。** 取不到的记下来接着取,最后在报告里列缺哪些。

🔴 真要停的只有两条:

1. **需要改任何 `.py` / `.sh`**(R0)。这一单用不到任何仓库里的脚本,要真需要,说明理解错了。
2. **清单里的文件在机器上找不到,而你想拿别的替上。** 缺就缺,**照实报缺**——
   第一轮那几个批次是 7 月底 8 月初的产物,`output/` 有没有被清理过我在本地看不到。
   替换样本 = 图和报告里的数字对不上,而这种错**看不出来**。

🟡 其余全部「记下来,继续」。⚪ **清单不是权威,机器上看到的才是**;
路径对不上以机器为准,做完在报告里指出来。

---

## 2. 取物清单(62 个文件,4 组)

仓库根:`/kaimm-distill/wuwenxuan/UNO`。下面所有源路径都是**相对仓库根**的。

参考图(`datasets/dreambooth/dataset/<主体>/*.jpg`)**不用取**,本地已经有 submodule。

### A 组 · 实验 1:不重训直接开隔离 → 身份归零(报告 §2.2,`0/51` vs `45/51`)

同一个 task_id 两张图并排:左边官方权重全注意力(身份保住),右边同一份权重直接开隔离(**归零**)。
这四条都是「客观计数里 full 拿满分、iso 全灭」的干净对照,不是我挑好看的。

| task_id | 参考主体 | prompt | full | iso |
|---|---|---|---|---|
| `PI_S1_005_s1` | backpack_dog + colorful_sneaker | a backpack and a sneaker on top of a wooden floor | 2/2 | **0/2** |
| `PI_S1_028_s0` | can + grey_sloth_plushie | a can and a stuffed animal with a blue house in the background | 2/2 | **0/2** |
| `PI_S3_017_s0` | clock(1-ref) | a clock on top of the sidewalk in a crowded street | 1/1 | **0/1** |
| `PI_S3_022_s1` | duck_toy(1-ref) | a toy on the beach | 1/1 | **0/1** |

### B 组 · 实验 2:两段 SFT 之后 vs 官方(报告 §2.3,主指标 0.819)

故意混了三种判读结果,**别只取 teacher 赢的那几张**——报告里这一批的故事就是「追不平但不是全崩」。

| task_id | 参考主体 | prompt | 标注者判 |
|---|---|---|---|
| `S1_000_s0` | backpack_dog + bear_plushie | a backpack and a stuffed animal in the jungle | teacher 赢 |
| `S1_011_s0` | bear_plushie + candle | a stuffed animal and a candle with a tree and autumn leaves… | 平局 |
| `S1_001_s0` | backpack_dog + berry_bowl | a backpack and a bowl in the snow | **学生赢** |
| `S3_000_s1` | backpack_dog(1-ref) | a backpack in the jungle | teacher 赢 |

### C 组 · 实验 4:隔离 vs 全注意力,只差一个开关(报告 §2.4,0.842;2-ref 0.716)

这一组是第一轮最承重的那条边 —— 两侧同数据、同配方、同 4000 步、同 seed,**只差 `--ref_isolation`**。
注意两侧图在**两个不同目录**、而且 task_id 前缀不一样(`AB_` vs `AA_`),别搞混。

| task_id(去前缀) | 参考主体 | prompt | 标注者判 |
|---|---|---|---|
| `S1_021_s1` | berry_bowl + fancy_boot | a bowl and a boot in the snow | 全注意力赢 |
| `S1_037_s1` | clock + grey_sloth_plushie | a clock and a stuffed animal on top of the sidewalk… | 全注意力赢 |
| `S1_039_s1` | colorful_sneaker + fancy_boot | a sneaker and a boot on top of a white rug | 平局 |
| `S1_035_s0` | clock + duck_toy | a clock and a toy on top of green grass with sunflowers… | **隔离赢** |
| `S3_001_s1` | backpack_dog(1-ref) | a backpack in the snow | 平局(1-ref 测不到代价) |

### D 组 · 第二轮主结果:三臂同 task_id(报告 §3.3,主指标 0.946)

**这组是 PPT 的主角,36 张,一张都不能少。** 三个变体同一批、同 seed、同口径:

- `p3_full` = 官方权重全注意力 = teacher
- `p3_iso_pre` = 不重训直接开隔离(**参考图特征根本看不到**,报告 §3.3 末段)
- `p3_iso_post` = 隔离学生(2000 步速度匹配)

| task_id | 层 | 参考主体 | 标注者判 |
|---|---|---|---|
| `M6_S1_042_s2` | 2-ref | duck_toy + grey_sloth_plushie | 平局 |
| `M6_S1_012_s2` | 2-ref | bear_plushie + clock | 平局 |
| `M6_S1_024_s2` | 2-ref | can + clock | 平局 |
| `M6_S1_028_s0` | 2-ref | can + grey_sloth_plushie | 平局 |
| `M6_S1_039_s2` | 2-ref | colorful_sneaker + fancy_boot | teacher 赢 |
| `M6_S1_037_s1` | 2-ref | clock + grey_sloth_plushie | teacher 赢 |
| `M6_S1_005_s1` | 2-ref | backpack_dog + colorful_sneaker | teacher 赢 |
| `M6_S1_020_s2` | 2-ref | berry_bowl + duck_toy | **学生赢** |
| `M6_S1_040_s2` | 2-ref | colorful_sneaker + grey_sloth_plushie | **学生赢** |
| `M6_S3_03c2_s0` | 1-ref | can | 平局 |
| `M6_S3_01c1_s1` | 1-ref | bear_plushie | 平局 |
| `M6_S3_06c3_s0` | 1-ref | colorful_sneaker | 平局 |

> 顺带一提,`M6_S1_005_s1`(D 组)和 `PI_S1_005_s1`(A 组)是**同一个 prompt、同一对参考主体**
> ——backpack + sneaker on a wooden floor。两轮、两个底座、同一个场景,PPT 里可以并成一页。
> 我没有把它做成额外的取物项,你按各自组取就行。

### 已经在 git 里的,**不要重复取**

| 已有 | 是什么 |
|---|---|
| `output/p3_floor/a/*.png` + `b/*.png`(30+30) | 第二轮的**满分刻度**锚点(同权重异 run),报告 §3.3 那个 1.000 |
| `output/qwen_baseline/ALL_COMPARISON_part*.png` | Q1 teacher 裸基线 40 条拼图 |
| `output/qwen_3ref/ALL_COMPARISON_part*.png` | Q1-B 3-ref 探底 122 条拼图(报告 §3.2 的 73.0%) |
| `output/attn_diag/*.png`、`output/multibanana_eval*/*.png` | 注意力可视化与早期对比 |

---

## 3. 怎么取

### 3.1 写清单文件

整段复制。**别手敲**,62 行手敲必错一行,而错一行的后果是图和数字对不上。

清单**只有一列源路径**,目标文件名由 §3.3 的脚本按规则算出来 ——
这样做是因为这份单子要经过聊天窗口转达一次,**带分隔符的表格在转达途中会被吃掉空白**。

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull
mkdir -p output/ppt_assets
cat > /tmp/ppt_srcs.txt <<'EOF'
output/probe_iso/PI_S1_005_s1__official_full.png
output/probe_iso/PI_S1_005_s1__official_iso.png
output/probe_iso/PI_S1_028_s0__official_full.png
output/probe_iso/PI_S1_028_s0__official_iso.png
output/probe_iso/PI_S3_017_s0__official_full.png
output/probe_iso/PI_S3_017_s0__official_iso.png
output/probe_iso/PI_S3_022_s1__official_full.png
output/probe_iso/PI_S3_022_s1__official_iso.png
output/eval_multiref/S1_000_s0__official_full.png
output/eval_multiref/S1_000_s0__ours_kv_post4000.png
output/eval_multiref/S1_011_s0__official_full.png
output/eval_multiref/S1_011_s0__ours_kv_post4000.png
output/eval_multiref/S1_001_s0__official_full.png
output/eval_multiref/S1_001_s0__ours_kv_post4000.png
output/eval_multiref/S3_000_s1__official_full.png
output/eval_multiref/S3_000_s1__ours_kv_post4000.png
output/eval_arm_b/AB_S1_021_s1__arm_b_iso.png
output/eval_arm_a/AA_S1_021_s1__arm_a_full.png
output/eval_arm_b/AB_S1_037_s1__arm_b_iso.png
output/eval_arm_a/AA_S1_037_s1__arm_a_full.png
output/eval_arm_b/AB_S1_039_s1__arm_b_iso.png
output/eval_arm_a/AA_S1_039_s1__arm_a_full.png
output/eval_arm_b/AB_S1_035_s0__arm_b_iso.png
output/eval_arm_a/AA_S1_035_s0__arm_a_full.png
output/eval_arm_b/AB_S3_001_s1__arm_b_iso.png
output/eval_arm_a/AA_S3_001_s1__arm_a_full.png
output/p3_full/M6_S1_042_s2.png
output/p3_iso_pre/M6_S1_042_s2.png
output/p3_iso_post/M6_S1_042_s2.png
output/p3_full/M6_S1_012_s2.png
output/p3_iso_pre/M6_S1_012_s2.png
output/p3_iso_post/M6_S1_012_s2.png
output/p3_full/M6_S1_024_s2.png
output/p3_iso_pre/M6_S1_024_s2.png
output/p3_iso_post/M6_S1_024_s2.png
output/p3_full/M6_S1_028_s0.png
output/p3_iso_pre/M6_S1_028_s0.png
output/p3_iso_post/M6_S1_028_s0.png
output/p3_full/M6_S1_039_s2.png
output/p3_iso_pre/M6_S1_039_s2.png
output/p3_iso_post/M6_S1_039_s2.png
output/p3_full/M6_S1_037_s1.png
output/p3_iso_pre/M6_S1_037_s1.png
output/p3_iso_post/M6_S1_037_s1.png
output/p3_full/M6_S1_005_s1.png
output/p3_iso_pre/M6_S1_005_s1.png
output/p3_iso_post/M6_S1_005_s1.png
output/p3_full/M6_S1_020_s2.png
output/p3_iso_pre/M6_S1_020_s2.png
output/p3_iso_post/M6_S1_020_s2.png
output/p3_full/M6_S1_040_s2.png
output/p3_iso_pre/M6_S1_040_s2.png
output/p3_iso_post/M6_S1_040_s2.png
output/p3_full/M6_S3_03c2_s0.png
output/p3_iso_pre/M6_S3_03c2_s0.png
output/p3_iso_post/M6_S3_03c2_s0.png
output/p3_full/M6_S3_01c1_s1.png
output/p3_iso_pre/M6_S3_01c1_s1.png
output/p3_iso_post/M6_S3_01c1_s1.png
output/p3_full/M6_S3_06c3_s0.png
output/p3_iso_pre/M6_S3_06c3_s0.png
output/p3_iso_post/M6_S3_06c3_s0.png
EOF
wc -l < /tmp/ppt_srcs.txt          # 必须是 62
sort -u /tmp/ppt_srcs.txt | wc -l  # 也必须是 62(少了说明贴漏了行)
```

### 3.2 先只查存在性,不要先转换

**先取证再动手**:第一轮那几个目录是 7 月底的产物,可能被清过。先知道缺什么,再决定怎么办。

```bash
missing=0
while read -r src; do
  [ -f "$src" ] || { echo "缺: $src"; missing=$((missing+1)); }
done < /tmp/ppt_srcs.txt
echo "缺 $missing / 62"
```

**缺了照实报,不要找替代品。** 缺的那一组在报告里写明缺哪几个 task_id。

> 这 62 条路径**已经在本地对着 git 里的 `results.json` 逐条核过**,全部是当初那几次 run
> 真正写出来的文件名(`probe_iso` 384 条 / `eval_arm_a` 384 / `eval_arm_b` 222 /
> `eval_multiref` 711 / `p3_*` 各 240,后者 `n_fail=0`、`n_missing_png=0`)。
> ⇒ **文件不在,只可能是磁盘被清理过,不会是路径写错了。** 报告里就这么写,
> 别去猜别的命名方式、别去 `find` 相似文件名顶上。

### 3.3 转换 + 压小(改名规则写在脚本里)

PPT 用不上 1024² 无损 PNG(一页幻灯片 1920×1080,三图并排每格才 600px),
而 62 张原图约 70 MB,推进 git 会把仓库拖垮(`.git` 已经 523 MB)。
统一转 **JPEG q90、长边不超过 1024(不放大)**,同时按源路径改成自解释的名字。

把下面这段存成 `/tmp/ppt_conv.py` 再跑(**别用 heredoc 套 heredoc**,会咬到引号):

```python
import re, pathlib
from PIL import Image

RULES = [(r'^output/probe_iso/(.+)__official_full\.png$',        r'A_probe__\1__full'),
         (r'^output/probe_iso/(.+)__official_iso\.png$',         r'A_probe__\1__iso'),
         (r'^output/eval_multiref/(.+)__official_full\.png$',    r'B_m4__\1__teacher'),
         (r'^output/eval_multiref/(.+)__ours_kv_post4000\.png$', r'B_m4__\1__student'),
         (r'^output/eval_arm_b/AB_(.+)__arm_b_iso\.png$',        r'C_arm__\1__isoB'),
         (r'^output/eval_arm_a/AA_(.+)__arm_a_full\.png$',       r'C_arm__\1__fullA'),
         (r'^output/p3_full/(.+)\.png$',                         r'D_p3__\1__full'),
         (r'^output/p3_iso_pre/(.+)\.png$',                      r'D_p3__\1__iso_pre'),
         (r'^output/p3_iso_post/(.+)\.png$',                     r'D_p3__\1__iso_post')]

def dst(s):
    for pat, rep in RULES:
        if re.match(pat, s):
            return re.sub(pat, rep, s) + '.jpg'
    return None

out = pathlib.Path('output/ppt_assets'); out.mkdir(parents=True, exist_ok=True)
srcs = [l.strip() for l in open('/tmp/ppt_srcs.txt') if l.strip()]
ok = skip = 0
for s in srcs:
    d = dst(s)
    if d is None:                       # 规则没覆盖 = 清单贴坏了,不要猜
        print('[规则未覆盖] ' + s); skip += 1; continue
    if not pathlib.Path(s).exists():
        print('[缺] ' + s); skip += 1; continue
    im = Image.open(s).convert('RGB')
    w, h = im.size; m = max(w, h)
    if m > 1024:
        im = im.resize((round(w * 1024 / m), round(h * 1024 / m)), Image.LANCZOS)
    im.save(out / d, 'JPEG', quality=90, optimize=True)
    ok += 1
print('转换 %d / %d,跳过 %d' % (ok, len(srcs), skip))
```

```bash
python3 /tmp/ppt_conv.py
ls output/ppt_assets | wc -l      # 期望 62(减去缺的)
du -sh output/ppt_assets          # 期望 8–14 MB
```

🟡 **超过 40 MB 就停下来先报一声**,别直接推 —— 那说明尺寸假设错了,我重新给参数。
🔴 出现 `[规则未覆盖]` 说明清单在转达途中贴坏了,**停下重贴**,别手工补名字。

### 3.4 留一份取证清单

```bash
{ echo "# ppt_assets 取证  $(date -u +%FT%TZ)  commit $(git rev-parse --short HEAD)"
  cd output/ppt_assets && sha256sum *.jpg | sort -k2
} > output/ppt_assets/MANIFEST.txt
```

---

## 4. 怎么交

### 主通道:推一个分支

```bash
git checkout -b assets/ppt-figs
```

`.gitignore` 是白名单模式,得先放行。**把这段追加到 `.gitignore` 末尾,理由一起写进去**
(这是这个仓库的规矩,不是形式):

```
# ─── PPT 配图(docs/PPT_ASSETS_RUN.md)────────────────────────────────────
# 62 张示例图,对应 reports/FINAL_REPORT.md §2.2/§2.3/§2.4/§3.3 的四个批次。
# 样本按已冻结的 blind_annotations_*.json 选定,不是随手挑的,所以要能被审计。
# 带变体名可以进 git 的理由:这四批**判读全部完成并冻结**,
# 「判读完成前不得生成带变体名的拼图」那条纪律的前提已经不成立。
# 统一 JPEG q90 / 长边 ≤1024,约 10 MB;全分辨率 PNG 仍留在共享盘,不进 git。
!/output/ppt_assets/
/output/ppt_assets/*
!/output/ppt_assets/*.jpg
!/output/ppt_assets/MANIFEST.txt
```

```bash
git add .gitignore output/ppt_assets
git commit -m "assets: PPT 示例图 62 张 —— 四个批次按冻结标注选样"
git push -u origin assets/ppt-figs
```

### 推不出去怎么办(`REMOTE_AGENT_HANDBOOK.md` §3.5:代理吃掉 POST)

**不要去找隧道**,那是 R10 级别的浪费。改成打一个包放在共享盘上,把路径报回来让用户自己取:

```bash
tar czf /kaimm-distill/wuwenxuan/ppt_assets.tgz -C output ppt_assets
ls -lh /kaimm-distill/wuwenxuan/ppt_assets.tgz
sha256sum /kaimm-distill/wuwenxuan/ppt_assets.tgz
```

然后把**绝对路径 + 大小 + sha256 + `MANIFEST.txt` 全文**打印出来。
本地 commit 照做(审计线索),推不出去不影响。

---

## 5. 已知的坑

- **两侧在不同目录**:C 组的 `AB_*`(`output/eval_arm_b/`)和 `AA_*`(`output/eval_arm_a/`)
  是**同一个任务**的两条腿,前缀不同是历史原因。清单里已经写死了完整路径,照抄就行。
- **D 组三个变体是三个目录、同一个文件名**(`output/p3_{full,iso_pre,iso_post}/<task_id>.png`),
  别只取到一个目录就以为齐了。三臂缺一张,那一页 PPT 就废了。
- **worker 建出来的目录是 root 属主**,写不进去先
  `sudo chown -R wuwenxuan03:wuwenxuan03 <目录>`。
- **PIL 可能不在当前 env 里**。`source` 一个有 pillow 的 env(`qwen-edit` 有);
  实在没有就用 `ffmpeg -i in.png -vf scale=... -q:v 3 out.jpg`,在报告里写明换了工具。
- **别用 `nohup` 跑长任务**,这一单几分钟跑完,用不上后台;但真要后台就用
  `setsid ... < /dev/null &`。
- 沉默上限 10 分钟(`REMOTE_AGENT_HANDBOOK.md` §5)。这一单不该超过,超了说明卡住了,报一声。

---

## 6. 明确不做

| 不做 | 理由 |
|---|---|
| 自己换样本 / 加样本 | 样本是按冻结标注选的,换了就和报告里的数字对不上 |
| 拼拼图(board) | 排版是 PPT 那边的事,拼死了反而没法调。**只交单张** |
| 加标题、加水印、加变体名文字 | 同上,而且烧进图里之后改不掉 |
| 重新出图 / 重跑任何模型 | 这一单一张新图都不生成 |
| 推全分辨率 PNG | `.git` 已经 523 MB |
| 顺手把别的批次也一起取了 | 想加就先说,别自己扩范围 |

---

## 7. 交回什么

落 `reports/<UTC日期>-ppt-assets/REPORT.md`,并**整段打印到 stdout**(用户会转给我):

1. **状态**:绿灯 / 黄灯 —— 以及走的是哪条通道(推分支 / 打包)
2. **62 个里取到几个**,缺的逐条列 task_id + 源路径 + 你查到的原因(目录不在?文件不在?)
3. `du -sh output/ppt_assets` 与 `ls | wc -l`
4. `MANIFEST.txt` 全文(sha256 清单)
5. 分支名 + commit sha,或 tgz 的绝对路径 + 大小 + sha256
6. **你顺手看到的任何异常**:比如某张图明显是坏的、某个目录里文件数和 `results.json` 对不上

⚠️ 报告里**不要替我判读这些图好不好看**。取物就是取物,判读是作者的事。
