# x₀ 分布诊断 · 执行报告

日期:2026-08-17
脚本:`qwen/diag_x0_shift.py`(commit `1f15f6ca242cca72714d268b1f7e5a78be11bdbf`,分支 `diag/x0-shift`,已 push)
任务类型:infer_hub 提交式推理(本机 4090×8,推理走 H 卡集群)

**本报告只放数据与自检,不做判读。判读由作者来。**

---

## 0. 执行概述

| 阶段 | 说明 | 机器 | 状态 |
|---|---|---|---|
| embeds 首投 | `--n 24` 单卡 | ge90-26 | ❌ 失败(tee 目录顺序 bug,详见 §4-1) |
| embeds 重投 | 同上,`--force` | ge90-26 | ✅ |
| run | `--n 24 --shard_idx 0..7 --num_shards 8` 8 卡 | ge90-10 | ✅ |
| merge | 本机,无需 GPU/QWEN_WEIGHTS | 本机 | ✅ |

- 提交参数:`--owner wuwenxuan --project diag-x0 --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 --output-dir /kaimm-distill/wuwenxuan/UNO/output --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit --prep-cmd 'true' --prep-marker /kaimm-distill/wuwenxuan/UNO/datasets --gpus 1/8`
- 墙钟(run,含轨迹+残差):约 23–24 分钟(执行单预估 15 分钟,实测偏慢,见 §3)

---

## 1. embeds `[自检]` 行

```
[自检] 选中 24 条 | n_refs 分布 {1-ref: 8, 2-ref: 8, 3-ref: 8}
```

分层抽样生效:24 条 = 1/2/3-ref 各 8 条。产物 48 个 `.pt`(正/负各 24),落共享盘 `output/diag_x0_shift/embeds/`。

---

## 2. merge 完整 stdout(原样)

```
合并 8 个分片 | 24 条样本 | 1920 行 → /kaimm-distill/wuwenxuan/UNO/output/diag_x0_shift/rows.json

======================================================================================
前提:x₀ 到底缺不缺高频(高频能量占比 ‖hi‖/‖all‖)
--------------------------------------------------------------------------------------
      样本  n_refs     x₀_UNO     on-policy 末端       比值
     143       2     0.1970           0.2294    0.859
    4064       3     0.2523           0.1831    1.378
    8007       1     0.1994           0.1721    1.159
     343       2     0.2009           0.2006    1.001
    4071       3     0.1763           0.1795    0.982
    8092       1     0.1859           0.2529    0.735
     403       2     0.2080           0.2216    0.938
    4126       3     0.2088           0.1588    1.315
    8119       1     0.2076           0.2934    0.708
    2384       2     0.2387           0.1709    1.397
    4969       3     0.1924           0.2355    0.817
    8451       1     0.1677           0.2987    0.561
    2750       2     0.1804           0.2663    0.677
    5417       3     0.1860           0.1444    1.288
    8796       1     0.2126           0.3053    0.696
    2838       2     0.2470           0.2137    1.156
    5452       3     0.1868           0.2296    0.814
    8932       1     0.1668           0.2478    0.673
    3305       2     0.2240           0.2264    0.990
    7139       3     0.1859           0.1210    1.536
    8965       1     0.2228           0.2819    0.790
    3847       2     0.2345           0.2037    1.151
    7230       3     0.2239           0.2491    0.899
    8989       1     0.2155           0.2585    0.834
    平均比值                                        0.973
  比值 ≈ 1 ⇒ x₀ 的高频并不比部署末端少,机制链第一步就不成立,下表不用解释。

======================================================================================
主表:同一条 (prompt, refs),只差 x_t 从哪来
--------------------------------------------------------------------------------------
               σ 段           组     n   rel_pre  rel_post      修复率   rel_lo   rel_hi       残差高频占比
    高 σ [0.75,1.0]  train_dist   384    0.0754    0.0367    0.513   0.0596   0.0125        0.302
    高 σ [0.75,1.0]   on_policy   384    0.1174    0.0514    0.562   0.0740   0.0267        0.394
    中高 [0.50,0.75)  train_dist   264    0.0417    0.0214    0.487   0.0289   0.0153        0.566
    中高 [0.50,0.75)   on_policy   264    0.1013    0.0535    0.472   0.0603   0.0481        0.666
    中低 [0.25,0.50)  train_dist   168    0.0348    0.0209    0.400   0.0244   0.0187        0.702
    中低 [0.25,0.50)   on_policy   168    0.1063    0.0713    0.330   0.0707   0.0719        0.753
  低 σ  [0.00,0.25)  train_dist   144    0.0930    0.0396    0.574   0.0346   0.0473        0.780
  低 σ  [0.00,0.25)   on_policy   144    0.1824    0.1299    0.288   0.1033   0.1618        0.818

  rel_pre  = 未训练的隔离腿(=iso_pre)在这一段的相对残差,批内标尺
  修复率   = 1 − rel_post/rel_pre,训练在这一段补回了多少
  rel_lo/hi= 低频/高频**各自**的相对误差;闹钟「保住颜色保不住数字」对应 rel_hi ≫ rel_lo
  两组修复率重合 ⇒ 对齐点没偏,换 x₀ 重训无收益;B 在低 σ 显著更低 ⇒ 假说立住
  **判据由作者定,本脚本不下结论。**
======================================================================================

✅ σ₀ 机制自检:24 条样本两组逐位相同。
```

---

## 3. 每片墙钟耗时与峰值显存

| 分片 | 样本 idx | 墙钟(秒) | 峰值显存 |
|---|---|---|---|
| shard0 | 143 / 4064 / 8007 | 1411 | 41.0 GB |
| shard1 | 343 / 4071 / 8092 | 1408 | 41.0 GB |
| shard2 | 403 / 4126 / 8119 | 1402 | 41.0 GB |
| shard3 | 2384 / 4969 / 8451 | 1408 | 41.0 GB |
| shard4 | 2750 / 5417 / 8796 | 1406 | 41.0 GB |
| shard5 | 2838 / 5452 / 8932 | 1406 | 41.0 GB |
| shard6 | 3305 / 7139 / 8965 | 1399 | 41.0 GB |
| shard7 | 3847 / 7230 / 8989 | 1408 | 41.0 GB |

- 墙钟 ≈ 23–24 分钟/片(执行单预估 15 分钟)。轨迹采样 ~205s/片,量残差每条样本 ~550–600s(40σ × 2组 × 3 前向 = 240 次 20B bf16 前向)。
- 峰值显存 41.0 GB(H800 143GB,余量充足)。

---

## 4. 踩坑记录(原样)

1. **embeds 首投失败:tee 目标目录顺序 bug。**
   `--cmd` 里 `python ... | tee $INFER_OUTPUT_DIR/diag_x0_shift/embeds_stdout.log` 的 tee 目标目录当时**还没 mkdir**(mkdir 写在 tee 之后)。tee 直接 `No such file or directory` 退出 → `set -o pipefail` 下整条失败(exit 1)。24 条 embeds 计算其实成功落盘推理机本地缓存,但没拷回共享盘。
   修正:`mkdir -p $INFER_OUTPUT_DIR/diag_x0_shift` 前移到 tee 之前;`infer_submit --force` 重投(同 label+commit 幂等跳过需绕过)。

2. **推理机 checkout 数据集不完整。**
   `datasets/distill_multiref` 在 git 只追踪 34 个文件,`datasets/dreambooth` 是 submodule;manifest 引用的 9000 张 refs 图和 `image_tgt_path`(x₀ 目标图)都在共享盘完整版。解法:`--cmd` 开头 `rm -rf $INFER_CODE_DIR/datasets/{dreambooth,distill_multiref} && ln -s /kaimm-distill/wuwenxuan/UNO/datasets/{dreambooth,distill_multiref} ...`。软链源路径用 `--prep-marker /kaimm-distill/wuwenxuan/UNO/datasets` 声明,同时满足 v3 prep 门槛(`--prep-cmd 'true'`)。

3. **lora ckpt 共享盘路径过不了泄漏检查。**
   `--cmd` 硬规矩:不允许出现未声明的共享盘绝对路径。2.2G 的 `output/train_iso/step002000.pt` 无法直接写进 `--cmd`。解法:`--output-dir` 设为 `/kaimm-distill/wuwenxuan/UNO/output`(覆盖 train_iso 的 declared root),`--cmd` 里用 `--lora $INFER_OUTPUT_DIR/train_iso/step002000.pt`(注入变量,无字面量)。

4. **merge 写 rows.json PermissionError。**
   `output/diag_x0_shift` 是推理机 worker(root)建的,当前用户无写权限。`sudo chown -R wuwenxuan03:wuwenxuan03 output/diag_x0_shift` 修复(执行单 §3 预期坑)。

5. **量残差阶段日志静默,易误判卡死。**
   第二阶段每条样本 240 次前向是纯 `torch.no_grad()`,无 progress bar,第一次 `[残差]` 在第一条样本完成后才打印,中间有几钟静默。判活性靠 worker 心跳(`claimed/*.hb` 持续刷新)+ 各片 mtime。8 片最终全部完成、σ₀ 自检全过。

6. **infer_submit 权限/代理(记忆已有)。**
   当前 shell 用户 `wuwenxuan03` 写不了 `infer_hub/queues/`,注册成员是 `wuwenxuan`。用 `sudo -E env PATH=... http_proxy=... https_proxy=... /kaimm-distill/infer_hub/lib/infer_submit` 模板。

---

## 5. 产物清单

`output/diag_x0_shift/`:

| 文件 | 说明 |
|---|---|
| `embeds/` | 48 个 `.pt`(24 条 × 正/负 prompt_embeds) |
| `embeds_stdout.log` | embeds 完整 stdout |
| `rows_shard0..7.json` | 8 片原始读数(各 240 行) |
| `rows.json` | merge 合并结果(1920 行) |
| `shard0..7.log` | 8 片完整 stdout(墙钟/峰值/自检) |

## 6. 备注

- 机器:embeds 在 `ge90-26`,run 在 `ge90-10`,merge 在本机。
- 提交的 job:`wuwenxuan__diag_x0_embeds__...`(首投 failed / 重投 `...__1786956697` done)、`wuwenxuan__diag_x0_run__...`(首投被取消/重投同 id done)。run 首投被取消是因为当时 embeds 失败,重投的 run `--cmd` 内置了 embeds 产物等待循环,两个 job 可并行排队。
