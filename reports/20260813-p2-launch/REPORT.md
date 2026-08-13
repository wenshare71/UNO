# P2 训练路径自检 · 执行报告

> 对应 `qwen/P2_LAUNCH_RUN.md`。执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),执行时间 2026-08-13。
> **§1 未全过(S5c),按手册「§1 不过就别往下走」,§2 / §3 未执行。** 输出原样全文,不转述、不修代码。

---

## §1 · 训练路径自检(纯 CPU)

### 命令原样

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
cd $R && git pull            # $R = /kaimm-distill/wuwenxuan/UNO;Already up to date.
$E/bin/python -u qwen/test_train_smoke.py
```

`git pull`:`Already up to date.`

> 注:手册原命令无 `-u`。实测裸 `| tee` 时 stdout 块缓冲,进度被吞;改 `-u` 无缓冲重跑(其余完全一致)。初次 5 分钟超时被杀即此因,非卡死。

### stdout 原样全文

进程退出码 `1`(有项不过)。

```text
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[自检] 训练路径 | 小模型 2 层 | fp32 CPU | LoRA targets 8 个
There are modules in QwenImageTransformer2DModel that should be kept in float32: []. Casting directly with `to()` can lead to inconsistent results; set `torch_dtype` in `from_pretrained()` instead to keep these modules in float32.
  ✅ S1 add_adapter 认得 LORA_TARGETS 那 8 个名字(期望 32 个张量)  实得 32 | 例:transformer_blocks.0.attn.to_q.lora_A.default.weight
  ✅ S1b 只有 LoRA 要梯度,主干全冻  非 LoRA 的可训张量 无
  ✅ S2 enable_gradient_checkpointing 落到顶层模型上(fork 的 forward 读的是它)  gradient_checkpointing=True
  ✅ S3 LoRA 初始为恒等(lora_B=0),disable/enable 逐位相同  max=0.000e+00
  ✅ S4 第 0 步 loss 非零 —— 差异来自 mask 而非 LoRA(LoRA 此刻还是恒等)  mse=4.938599e-08
  ✅ S5a student 输出形状 = 噪声段(ref 位置不进 loss)  (1, 16, 64)
  ✅ S5b teacher 已 detach,不带梯度  requires_grad=False
  ❌ S5c 梯度检查点下能反传,且梯度传到了每一层 LoRA  14/32 个张量拿到非零梯度 (lora_A 初始高斯 / lora_B 初始 0,约一半非零属正常)
  ✅ S6 反传之后 cache 仍为空(store=False 而非事后 clear)  cache 里有 0 层
  ✅ S7a optimizer.step() 之后 LoRA 参数确实动了  29/32 个张量变化
  ✅ S7b 更新后 teacher 回到原样(disable 是真的关掉,不是近似)  max=0.000e+00
  ✅ S7c 更新后 student 确实变了(否则 LoRA 没接进前向)  max=1.563e-03
  ✅ S8 lora_state / set_lora_state 往返一致(断点续跑靠它)  max=0.000e+00
  ✅ S9 sigma 网格 = 推理实际那 40 个(单调降,首 1.0000 末 0.0200)  n=40
  ✅ S9b 网格与 ref 数无关(seq_len 只算噪声段)  同 seq_len 复现一致

====================================================================
❌ 1/15 项不通过。别申请机器,先修。
====================================================================
```

### 不过的那一项

| 项 | 断言 | 实测 |
|---|---|---|
| **S5c** | 梯度检查点下能反传,且**梯度传到每一层 LoRA** | **14/32 个张量拿到非零梯度**(脚本自带注:lora_A 初始高斯 / lora_B 初始 0,约一半非零属正常) |

对照:S5a/S5b 通过(student 形状对、teacher 已 detach),S6 通过(反传后 cache 空),S7 通过(参数确实动、teacher 回原样)。**唯独"梯度到达每一层 LoRA"这一项没满足。**

---

## §2 · 真 pipeline 段(CPU 加载 54 GB)

**未执行。** 被 §1 S5c 阻塞(手册:「§1 不过就别往下走」)。待 §1 修复通过后:`QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 $E/bin/python qwen/test_train_smoke.py --pipe`。

---

## §3 · prompt_embeds 预算(1 卡)

**未执行。** 同上被 §1 S5c 阻塞。待 §1/§2 通过后:`QWEN_WEIGHTS=... $E/bin/python qwen/train_iso.py precompute 2>&1 | tee /tmp/precompute.log`(跑前确认磁盘,输出 ~32 GB 到 `cache/prompt_embeds/`)。

---

## 待办(判读归你)

- S5c:14/32 个 LoRA 张量拿到非零梯度 —— 需要你判断是梯度检查点那 8 个位置参数传错(PLAN §3.2 里 S5 的注),还是判据本身对 lora_B=0 初始化太苛刻。
