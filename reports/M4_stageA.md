# M4 Stage A 回传:评测集构建统计表

> 日期 2026-07-30 · 执行者 H800 agent · 上游 `distill/M4_EVAL_SPEC.md` §3 / §3.5
>
> 交付物:`distill/build_eval_json.py`(新建)、`datasets/eval_multiref/eval_set.json`
> (232 任务 / 711 图,断言全部通过后写出)。**Stage A 到此为止,等确认后进 Stage B。**

## 执行摘要

- 命令:`.venv-uno/bin/python distill/build_eval_json.py`(先 `--dry_run` 核对,再正式写出)
- 启动断言(规格 §3.4)全部通过:无 TRAIN 泄漏、task_id 全局唯一、53 个唯一 ref 文件全部存在、
  总数 232/711 逐格对上、S1 恰好 44 个不重复组合、S1–S4 seed 唯一且与 M1 区间不重叠。
- S4 贪心选法实测**每个 held-out subject 恰好覆盖 3 次**,与规格 [已验证] 的参考实现一致。
- 一处与规格文本的出入(不影响任何条数/规则):§3.3 的 JSON 示例里 `S1_000_s0` 写的是
  `backpack_dog + berry_bowl`,而按 §3.2 伪代码"字典序 44 组合"枚举,k=0 是
  `backpack_dog + bear_plushie`(bear_plushie < berry_bowl)。示例应只是 schema 示意,
  本文按 §3.2 伪代码执行(它在 §2.5 不可改清单的保护范围内)。其余字段与示例 schema 一致。

## 统计表(命令输出原文)

```text
ref 图验存:53 个唯一文件全部可读 ✓

============================================================================
M4 评测集统计(M4_EVAL_SPEC §3.5)
============================================================================

[1] 分层任务数 / 图数(与 §3.2 表格逐格对照)
  层         任务      期望      图数      期望  对照
  S0         5       5      15      15  ✓
  S1       132     132     396     396  ✓
  S2        15      15      60      60  ✓
  S3        60      60     180     180  ✓
  S4        20      20      60      60  ✓
  合计       232     232     711     711  ✓

[2] S1 的 44 个组合 + 各自分到的场景模板(k % 20 轮转)
  k=00  backpack_dog + bear_plushie  →  tpl00 "in the jungle"
  k=01  backpack_dog + berry_bowl  →  tpl01 "in the snow"
  k=02  backpack_dog + can  →  tpl02 "on the beach"
  k=03  backpack_dog + candle  →  tpl03 "on a cobblestone street"
  k=04  backpack_dog + clock  →  tpl04 "on top of pink fabric"
  k=05  backpack_dog + colorful_sneaker  →  tpl05 "on top of a wooden floor"
  k=06  backpack_dog + duck_toy  →  tpl06 "with a city in the background"
  k=07  backpack_dog + fancy_boot  →  tpl07 "with a mountain in the background"
  k=08  backpack_dog + grey_sloth_plushie  →  tpl08 "with a blue house in the background"
  k=09  bear_plushie + berry_bowl  →  tpl09 "on top of a purple rug in a forest"
  k=10  bear_plushie + can  →  tpl10 "with a wheat field in the background"
  k=11  bear_plushie + candle  →  tpl11 "with a tree and autumn leaves in the background"
  k=12  bear_plushie + clock  →  tpl12 "with the Eiffel Tower in the background"
  k=13  bear_plushie + colorful_sneaker  →  tpl13 "floating on top of water"
  k=14  bear_plushie + duck_toy  →  tpl14 "floating in an ocean of milk"
  k=15  bear_plushie + fancy_boot  →  tpl15 "on top of green grass with sunflowers around it"
  k=16  berry_bowl + can  →  tpl16 "on top of a mirror"
  k=17  berry_bowl + candle  →  tpl17 "on top of the sidewalk in a crowded street"
  k=18  berry_bowl + clock  →  tpl18 "on top of a dirt road"
  k=19  berry_bowl + colorful_sneaker  →  tpl19 "on top of a white rug"
  k=20  berry_bowl + duck_toy  →  tpl00 "in the jungle"
  k=21  berry_bowl + fancy_boot  →  tpl01 "in the snow"
  k=22  berry_bowl + grey_sloth_plushie  →  tpl02 "on the beach"
  k=23  can + candle  →  tpl03 "on a cobblestone street"
  k=24  can + clock  →  tpl04 "on top of pink fabric"
  k=25  can + colorful_sneaker  →  tpl05 "on top of a wooden floor"
  k=26  can + duck_toy  →  tpl06 "with a city in the background"
  k=27  can + fancy_boot  →  tpl07 "with a mountain in the background"
  k=28  can + grey_sloth_plushie  →  tpl08 "with a blue house in the background"
  k=29  candle + clock  →  tpl09 "on top of a purple rug in a forest"
  k=30  candle + colorful_sneaker  →  tpl10 "with a wheat field in the background"
  k=31  candle + duck_toy  →  tpl11 "with a tree and autumn leaves in the background"
  k=32  candle + fancy_boot  →  tpl12 "with the Eiffel Tower in the background"
  k=33  candle + grey_sloth_plushie  →  tpl13 "floating on top of water"
  k=34  clock + colorful_sneaker  →  tpl14 "floating in an ocean of milk"
  k=35  clock + duck_toy  →  tpl15 "on top of green grass with sunflowers around it"
  k=36  clock + fancy_boot  →  tpl16 "on top of a mirror"
  k=37  clock + grey_sloth_plushie  →  tpl17 "on top of the sidewalk in a crowded street"
  k=38  colorful_sneaker + duck_toy  →  tpl18 "on top of a dirt road"
  k=39  colorful_sneaker + fancy_boot  →  tpl19 "on top of a white rug"
  k=40  colorful_sneaker + grey_sloth_plushie  →  tpl00 "in the jungle"
  k=41  duck_toy + fancy_boot  →  tpl01 "in the snow"
  k=42  duck_toy + grey_sloth_plushie  →  tpl02 "on the beach"
  k=43  fancy_boot + grey_sloth_plushie  →  tpl03 "on a cobblestone street"

[3] 每个 held-out subject 在各层的出现次数(按任务计,一个任务里出现一次记 1)
  subject                  S0   S1   S2   S3   S4     合计
  backpack_dog              1   27    0    6    6     40
  bear_plushie              1   24   15    6    6     52
  berry_bowl                1   27    0    6    6     40
  can                       1   27    0    6    6     40
  candle                    1   27    0    6    6     40
  clock                     1   27    0    6    6     40
  colorful_sneaker          1   27    0    6    6     40
  duck_toy                  1   27    0    6    6     40
  fancy_boot                1   27    0    6    6     40
  grey_sloth_plushie        1   24   15    6    6     52

[4] 场景模板使用直方图(按任务计)
  tpl00 ( 20) ████████████████████ "in the jungle"
  tpl01 ( 19) ███████████████████ "in the snow"
  tpl02 ( 20) ████████████████████ "on the beach"
  tpl03 ( 18) ██████████████████ "on a cobblestone street"
  tpl04 ( 15) ███████████████ "on top of pink fabric"
  tpl05 ( 12) ████████████ "on top of a wooden floor"
  tpl06 ( 12) ████████████ "with a city in the background"
  tpl07 ( 12) ████████████ "with a mountain in the background"
  tpl08 ( 12) ████████████ "with a blue house in the background"
  tpl09 ( 12) ████████████ "on top of a purple rug in a forest"
  tpl10 (  8) ████████ "with a wheat field in the background"
  tpl11 (  8) ████████ "with a tree and autumn leaves in the background"
  tpl12 (  8) ████████ "with the Eiffel Tower in the background"
  tpl13 (  8) ████████ "floating on top of water"
  tpl14 (  8) ████████ "floating in an ocean of milk"
  tpl15 (  8) ████████ "on top of green grass with sunflowers around it"
  tpl16 (  8) ████████ "on top of a mirror"
  tpl17 (  8) ████████ "on top of the sidewalk in a crowded street"
  tpl18 (  8) ████████ "on top of a dirt road"
  tpl19 (  8) ████████ "on top of a white rug"

[5] S4 选中的 10 个 3-组合 + 覆盖分布
  k=0  backpack_dog + bear_plushie + berry_bowl  →  tpl00
  k=1  can + candle + clock  →  tpl01
  k=2  colorful_sneaker + duck_toy + fancy_boot  →  tpl02
  k=3  backpack_dog + berry_bowl + grey_sloth_plushie  →  tpl03
  k=4  bear_plushie + can + candle  →  tpl04
  k=5  clock + colorful_sneaker + duck_toy  →  tpl05
  k=6  backpack_dog + fancy_boot + grey_sloth_plushie  →  tpl06
  k=7  bear_plushie + berry_bowl + can  →  tpl07
  k=8  candle + clock + colorful_sneaker  →  tpl08
  k=9  duck_toy + fancy_boot + grey_sloth_plushie  →  tpl09
  覆盖分布(覆盖次数:subject 数):{3: 10}(参考实现应恰好每个 subject 3 次;断言下限 ≥2)
  逐 subject:{'backpack_dog': 3, 'bear_plushie': 3, 'berry_bowl': 3, 'can': 3, 'candle': 3, 'clock': 3, 'colorful_sneaker': 3, 'duck_toy': 3, 'fancy_boot': 3, 'grey_sloth_plushie': 3}

[6] seed 区间
  S0(锚点,复刻冒烟):[3407]
  S1–S4:3500000–3800091(规格要求 3500000–3800091)
  与 M1 区间 3407000–3415999 重叠检查:✓ 不重叠(断言已过)
============================================================================

已写出 datasets/eval_multiref/eval_set.json:232 个任务,711 张图。
⛔ Stage A 到此为止。把统计表贴进 reports/M4_stageA.md 回传,等确认后再进 Stage B(distill/eval_multiref.py)。
```
