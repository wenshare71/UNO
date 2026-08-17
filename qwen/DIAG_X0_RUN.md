# x₀ 分布诊断 —— 给远程会话的执行单

> 脚本 `qwen/diag_x0_shift.py`,模块头写了完整设计,**跑之前读一遍那个 docstring**,这里不重复。
> 本单只讲怎么在机器上把它跑起来。**不训练、不出图、不进盲评。**

---

## 0. 你在做什么

量一件事:**训练时对齐 teacher 的那些点,和部署时真正经过的点,差多远。**

起因是 §9 判读——student 特有的退化集中在小尺度高频细节(罐头汉字、闹钟数字、蓝莓碗)。
待检验的是:manifest 的 `x₀` 是 UNO 的 512² 产物上采样来的,高频段是插值伪影,
而低 σ 时 `x_t ≈ x₀` ⇒ student 可能根本没在"带真实高频细节的图像"上对齐过。

产物是一张表,**不是一个结论**。判读作者来做。

约 15 分钟墙钟(8 片并行),每片 3 条样本 × 320 次前向。

---

## 1. 什么时候停

**默认动作是继续。** 这份单里的任何"预期"都不是门禁,数没对上就记进报告接着做。

🔴 真要停的只有两条:

1. **σ₀ 机制自检没过**(脚本自己会打 `❌ σ₀ 机制自检未过`)。
   两组在 σ₀ 用的是同一个 `x_t`,读数本应逐位相同——这是构造出来的恒等式,不是猜的阈值。
   不过就说明装置坏了(轨迹取错位、`encode_sample` 与 pipeline 的 `image_latents` 对不上),
   后面的表没有意义。把整段 stdout 贴回来。
2. 需要改任何 `.py`。

🟡 其余全部"记下来,继续"。⚪ **清单不是权威,机器上看到的才是**;前提错了以机器为准,做完在报告里指出来。

---

## 2. 做什么

机器要 **H800**。4090 做不了 Qwen 生成(20B bf16 ≈ 40 GB > 24 GB),这一单要采 teacher 轨迹,
所以 4090 上跑不了。怎么拿 8 张卡你定(P2 那台、或排 8 个单卡 job 都行)。

⚠️ **仓库 2026-08-17 起转为 private**,`git pull` / infer_hub clone 都需要凭据。
拉不动就先解决这个,别绕。

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
PY=/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python
```

### 2.1 预算 embeds(单卡,只载 7B VL)

```bash
$PY qwen/diag_x0_shift.py embeds --n 24
```

写 `output/diag_x0_shift/embeds/`,每条两个文件(正/负)。已存在的会跳过,断了直接重跑。
把 `[自检]` 那行的 n_refs 分布记下来(分层抽样有没有生效,看这个)。

### 2.2 量残差(8 片并行,每片一张卡)

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i $PY qwen/diag_x0_shift.py run \
    --lora /kaimm-distill/wuwenxuan/UNO/output/train_iso/step002000.pt \
    --n 24 --shard_idx $i --num_shards 8 &
done; wait
```

每片写 `output/diag_x0_shift/rows_shard$i.json`。
脚本启动时会自查 ckpt(`lora_B 非零` 应为 478,复用 `infer_iso.apply_lora_ckpt` 的断言),
自查 embeds 齐不齐,缺了会指路。

### 2.3 合并 + 判读表

```bash
$PY qwen/diag_x0_shift.py merge
```

`merge` 不加载模型,不需要 GPU,也不需要 `QWEN_WEIGHTS`。
它会核分片有没有重叠、ckpt 是不是同一份,不过就 `SystemExit`。

**整段 stdout 原样贴回报告,不要自己解读。**

---

## 3. 已知的坑(P2/P3 踩过的,撞上了照这个处理)

- `output/` 下某些目录是上一单 root 属主建的,当前用户写不进去。用免密 sudo 补,只写产物文件。
- ceph 偶发 EIO,某一片可能中途挂。**重跑那一片就行**——`embeds` 会跳过已存在的,
  `run` 是整片重算(15 分钟量级,不值得做断点)。
- 提交 job 时 `LD_LIBRARY_PATH` 要自己在 `--cmd` 里 export,别赌 worker 注入。
- `aio_n26` / `v4moe` 这两个公共 env 在这套机器上 import torch 直接挂,用 `qwen-edit`。

---

## 4. 明确不做

- 不改 `qwen/` 下任何既有 `.py`(R0)。`diag_x0_shift.py` 本身也不要改——
  它本地干跑过 25 项(下标 math、分层、分片、自检三分支、merge 两个守卫)。
- 不出图、不建拼图、不碰 `output/p3_*`。这一单与盲评无关。
- 不下结论。表里没有"达标/不达标",判读是作者的事。
- 不为了凑数改 `--n` / `--seed`。要改先说,改了两次读数就不可比。

---

## 5. 回报

一份 `reports/<日期>-diag-x0/REPORT.md`:

1. `embeds` 的 `[自检]` 行;
2. `merge` 的完整 stdout(前提表 + 主表 + σ₀ 自检那一行);
3. 每片的墙钟耗时与峰值显存;
4. 踩到的坑,原样记。
