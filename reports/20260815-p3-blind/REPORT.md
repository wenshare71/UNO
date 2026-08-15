# P3 盲评启动执行报告

> 执行单:`qwen/P3_BLIND_RUN.md` · 日期:2026-08-15
> 机器:`aiplatform-bjy-ge47-391.idchb1az2.hb1.kwaidc.com`(10.51.149.117)

**要记的数**:270 对 = 240 主对 + 30 run_floor;S1 165 / S3 75;盲种 `p3-qwen-iso-20260815`。

---

## 1. 三臂 n_missing_png

| 臂 | n_missing_png | 任务 | 失败 |
|---|---|---|---|
| full | **0** | 240 | 0 |
| iso_pre | **0** | 240 | 0 |
| iso_post | **0** | 240 | 0 |

三臂各合并 8 个 shard,存档 `results.json` 均写入。

## 2. build_pairs_p3.py 完整 stdout

```
写入 output/p3_eval/pairs_p3.json:270 对 {'p3_iso_post_vs_p3_full': 240, 'run_floor': 30}

条数:270 对 {'p3_iso_post_vs_p3_full': 240, 'run_floor': 30}
  p3_iso_post_vs_p3_full  S1     165
  p3_iso_post_vs_p3_full  S3      75
  run_floor               S1      20
  run_floor               S3      10

方向约定:key_0 = p3_iso_post(被检验方)/ key_1 = p3_full(基线)
  ⇒ 非平局胜率 < 0.5 一律读作「隔离腿更差」,不翻转。
  §8.2 要 n_nontie ≥ 94 ⇒ 平局率 > 60.8% 时结论是「判据不适用」而非「不达标」,**不许事后追加样本**

锚点孪生间距:纯 md5 最小 16 → 槽位内重排后 最小 115 / 中位 127 / 最大 200(批长 270;8 条锚点不在 240 子集内,批里只出现一次)
run_floor 三分位分布 {'头': 12, '中': 9, '尾': 9}(槽位未动,不聚堆)

盲种 p3-qwen-iso-20260815
⚠️ 随结论必须一起声明:
  ① 蒸馏 target 由官方**全注意力** teacher 生成 ⇒ 目标分布对基线腿有利;
  ② 本批 run_floor 两侧**逐位相同**(流水线位级确定),它量的是「标注者对一模一样的两张图的打平率」,**不是** run 噪声天花板;
  ③ 单标注者,无标注者间一致性(§8.5-1)。
```

## 3. 服务地址 + 连通性

- **服务**:`http://10.51.149.117:8766`(机器名 `aiplatform-bjy-ge47-391.idchb1az2.hb1.kwaidc.com`)
- 端口 **8766**(手册给的 8765 被占,见坑 ②)
- 服务常驻运行中,自检通过(540 张图全解码成功,无缺图/坏图),根路径 HTTP 200。
- 2026-08-15 按用户要求**重启过一次**(pid 256919),改用 `python -u` 无缓冲,
  启动 INFO 日志实时可见;重启后再次确认:配对清单 `pairs_p3.json`、盲种
  `p3-qwen-iso-20260815`、标注落盘 `blind_annotations_p3.json` 均正确。
- 连通性:**待用户确认**。若用户到这台机器无直连,走隧道:
  `ssh -L 8766:localhost:8766 aiplatform-bjy-ge47-391.idchb1az2.hb1.kwaidc.com`,再开 `http://localhost:8766`。

## 4. 踩到的坑

1. **results.json 写不进三臂输出目录** —— `output/p3_*/` 是上一单 root 属主写的(`drwxr-xr-x root`),
   当前用户 `wuwenxuan03` 无权写。手册说这一单不用 sudo,但前提(全本机新产出)不成立;
   按手册「前提错了以机器为准」,用**免密 sudo** 补跑 `--merge` 写存档。仅写了 `results.json`,
   未改动任何目录权限或 .py。
2. **端口 8765 被别的用户占着** —— `yanglingxiao` 的 M6 盲评服务(pid 71638,`pairs_m6.json`,
   `8月11` 起,dpdmd 环境)一直占着 8765。我的服务 bind 失败(exit 3),**第一次 curl 到的 200
   是别人的服务**。改为空闲端口 **8766** 重起,才是我这份 P3 服务。标注前务必确认开的是 8766,
   不是 8765(那是别人的 M6,别把标注写进别人的清单)。
3. **启动日志被 Python stdout 缓冲** —— 自检期(解码 540 张,~99% CPU 十几秒)后台输出文件一直
   是空的,端口绑定完成前看不出任何日志;以端口监听 + 未崩为准,别被空日志吓到。
4. 依赖/网络无坑:`uv` 建 `/tmp/blind` venv + `fastapi/uvicorn/pillow` 直连 pypi 一次成功,无需代理。

## 5. report.py 输出(原样)

标注完成于 2026-08-15(标注文件 `output/p3_eval/blind_annotations_p3.json`,270/270)。
命令:`/tmp/blind/bin/python -m distill.blind_eval.report output/p3_eval/pairs_p3.json output/p3_eval/blind_annotations_p3.json`
以下为脚本 stdout 原样,**未做任何解读**;达标判据在 `distill/M4_EVAL_SPEC.md` §8.2,由作者判读。

```
批次 p3  已标 270/270  (完整)

分组                            n  win0  win1  tie     平局率      非平局胜率  Wilson 95% CI          旧口径
p3_iso_post_vs_p3_full      240    16    28  196   81.7%      36.4%  [0.238, 0.511]       0.946
run_floor                    30     0     0   30  100.0%          —  —                    1.000
p3_iso_post_vs_p3_full/S1   165    14    27  124   75.2%      34.1%  [0.216, 0.495]       0.914
p3_iso_post_vs_p3_full/S3    75     2     1   72   96.0%      66.7%  [0.208, 0.939]       1.014
run_floor/S1                 20     0     0   20  100.0%          —  —                    1.000
run_floor/S3                 10     0     0   10  100.0%          —  —                    1.000
    p3_iso_post_vs_p3_full win0=p3_iso_post   win1=p3_full
    run_floor      win0=p3_full_run_a   win1=p3_full_run_b

左右偏好(位置偏差检验)        L    R  tie     选左率  Wilson 95% CI
  p3_iso_post_vs_p3_full       24   20  196   54.5%  [0.401, 0.683]
  run_floor                     0    0   30       —  —
  总体                           24   20  226   54.5%  [0.401, 0.683]
  判读:CI 含 0.5 = 无位置偏好;不含 = 所有其它数字都要扣掉这个偏差项
```
