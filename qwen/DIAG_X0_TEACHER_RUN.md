# x₀ 换成 teacher 自己的输出 —— 给远程会话的执行单

> 脚本 `qwen/diag_x0_teacher.py`,模块头写了完整设计和**预登记读法**,
> **跑之前读一遍那个 docstring**,这里不重复。本单只讲怎么在机器上跑起来。
> **不训练、不出图、不进盲评。**

上一单是 `reports/20260817-diag-x0/REPORT.md`(commit 28009f2)。本单接着它做,
踩过的坑都在下面 §3 兑现成命令了。

---

## 0. 你在做什么

上一单发现:部署真正经过的点(on_policy),student 的残差随 σ→0 越来越修不动
(修复率 0.562 → 0.288),而训练喂的那些点看不出这个趋势。原假说「x₀ 太糊」被前提表
否掉了(比值 0.973)。本单验的是替代解释:**x₀ 不是 teacher 自己的不动点**。

做法:同一批 24 条、同一份 ε,加第三条臂 `teacher_x0`(x₀ 换成 teacher 40 步采样
自己的输出),**三臂同批重测**。teacher 的输出是白拿的 —— 采轨迹那次 `pipe(...)`
本来就跑满 40 步,上一版把返回值丢了,这版捡起来,所以没有额外生成开销。

产物是三张表,**不是一个结论**。判读作者来做。

约 35 分钟墙钟(8 片并行),每片 3 条样本 × 440 次前向(上一单是 320,涨 1.4×)。

---

## 1. 什么时候停

**默认动作是继续。** 这份单里的任何"预期"都不是门禁,数没对上就记进报告接着做。

🔴 真要停的只有两条:

1. **σ₀ 机制自检没过**(脚本自己会打 `❌ σ₀ 机制自检未过`)。
   三臂在 σ₀ 用的是同一个 `x_t`,读数本应逐位相同、`dx` 本应全 0 ——
   这是构造出来的恒等式,不是猜的阈值。不过就说明装置坏了,后面的表没有意义。
   把整段 stdout 贴回来。
2. 需要改任何 `.py`。

🟡 其余全部"记下来,继续"。⚪ **清单不是权威,机器上看到的才是**;
前提错了以机器为准,做完在报告里指出来。

---

## 2. 做什么

一个 infer_hub job,8 卡。机器要 **H800**(20B bf16 峰值 41 GB,4090 装不下)。

⚠️ 仓库是 private,`git pull` / infer_hub clone 需要凭据。拉不动先解决这个,别绕。

**embeds 不用重算。** 本单复用 `output/diag_x0_shift/embeds/` 里那 48 个 `.pt`
(同 `--n 24` 同 `--seed 1234` ⇒ `pick_samples` 选出同一批 24 条)。
脚本启动时会逐条查,缺了会指路。

### 提交

提交参数**沿用上一单**(`wuwenxuan__diag_x0_run__*`),只改两处:`--project` 和 `--cmd`。

```
--owner wuwenxuan --project diag-x0-teacher
--weights   /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
--output-dir /kaimm-distill/wuwenxuan/UNO/output
--uv-env    /kaimm-distill/wuwenxuan/envs/qwen-edit
--prep-cmd 'true' --prep-marker /kaimm-distill/wuwenxuan/UNO/datasets
--gpus 1/8
```

`--cmd` 直接拿上一单 `diag_x0_run` 的来改 —— **分片变量怎么取,原样照抄那一份**,
别自己猜 infer_hub 暴露的是哪个环境变量。要改的只有三件事:

```bash
# ① mkdir 必须在 tee 之前(上一单 §4-1 就栽在这)
mkdir -p $INFER_OUTPUT_DIR/diag_x0_teacher

# ② 数据集软链照旧(git 里 distill_multiref 只有 34 个文件,dreambooth 是 submodule)
rm -rf $INFER_CODE_DIR/datasets/{dreambooth,distill_multiref}
ln -s /kaimm-distill/wuwenxuan/UNO/datasets/{dreambooth,distill_multiref} $INFER_CODE_DIR/datasets/

# ③ 换脚本名,lora 仍走注入变量(共享盘字面量过不了泄漏检查,上一单 §4-3)
python qwen/diag_x0_teacher.py run \
    --lora $INFER_OUTPUT_DIR/train_iso/step002000.pt \
    --n 24 --shard_idx <照抄> --num_shards 8 \
  | tee $INFER_OUTPUT_DIR/diag_x0_teacher/shard<照抄>.log
```

`--lora` 必须和上一单是**同一个 ckpt**(`step002000.pt`)。换了就没法和 28009f2 对话。

### 合并

```bash
sudo chown -R wuwenxuan03:wuwenxuan03 /kaimm-distill/wuwenxuan/UNO/output/diag_x0_teacher
cd /kaimm-distill/wuwenxuan/UNO && $PY qwen/diag_x0_teacher.py merge
```

`merge` 不加载模型,不需要 GPU / `QWEN_WEIGHTS`。它会核分片重叠、ckpt 一致、
**每条样本三臂齐全**(有一片是旧的两臂脚本跑的就会炸),不过就 `SystemExit`。

**整段 stdout 原样贴回报告,不要自己解读。**

---

## 3. 已知的坑(上一单 §4 的原样兑现)

- **量残差阶段静默 10 分钟是正常的。** 每条样本 360 次纯 `no_grad` 前向、无进度条,
  `[残差]` 要等一条跑完才打印。判活性看 worker 心跳(`claimed/*.hb`)+ 各片 mtime,
  别当卡死杀掉。
- `output/diag_x0_teacher` 会被 worker(root)建出来,当前用户写不进去 —— merge 前先 chown。
- `infer_submit` 用 `sudo -E env PATH=... http_proxy=... https_proxy=...` 那个模板
  (shell 用户 `wuwenxuan03` 写不了 `infer_hub/queues/`,注册成员是 `wuwenxuan`)。
- ceph 偶发 EIO,某一片可能中途挂。**重跑那一片就行**,15 分钟量级不值得做断点。
- `aio_n26` / `v4moe` 这两个公共 env import torch 直接挂,用 `qwen-edit`。

---

## 4. 明确不做

- 不改 `qwen/` 下任何既有 `.py`(R0)。`diag_x0_teacher.py` 本身也不要改 ——
  它本地干跑过 37 项(σ₀ 自检四分支、三臂聚合、merge 四个守卫、8 片 CLI 全路径)。
  `diag_x0_shift.py` **一个字都别动**,28009f2 那批读数得能原样复现。
- 不重算 embeds。不换 `--n` / `--seed` / `--lora`,换了就和上一单不可比。
- 不出图、不建拼图、不碰 `output/p3_*`。与盲评无关。
- 不下结论。判读法印在脚本输出末尾,是**跑之前**定死的,别在报告里替作者读。

---

## 5. 回报

一份 `reports/<日期>-diag-x0-teacher/REPORT.md`:

1. `merge` 的完整 stdout(三张表 + 预登记读法 + σ₀ 自检那一行);
2. 每片的墙钟耗时与峰值显存;
3. 踩到的坑,原样记;
4. 与上一单的差异项(用了哪个 commit、ckpt 是不是同一份)。
