# P2 训练 · §1–§4 回报

> 执行单 `qwen/P2_TRAIN_RUN.md`,起点 `aa2a97a`(实际 HEAD `0afc48a`,更新)。
> 日期:2026-08-13。机器:8×H800(每卡 143 GB)。
> 环境:`E=/kaimm-distill/wuwenxuan/envs/qwen-edit`(python 3.11.15)、
> `QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511`。

---

## §1 训练路径自检(纯 CPU)

**命令:**

```bash
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
$E/bin/python -u qwen/test_train_smoke.py
```

**stdout 全文:**

```
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[自检] 训练路径 | 小模型 2 层 | fp32 CPU | LoRA targets 8 个
There are modules in QwenImageTransformer2DModel that should be kept in float32: []. Casting directly with `to()` can lead to inconsistent results; set `torch_dtype` in `from_pretrained()` instead to keep these modules in float32.
  ✅ S1 add_adapter 认得 LORA_TARGETS 那 8 个名字(期望 32 个张量)  实得 32 | 例:transformer_blocks.0.attn.to_q.lora_A.default.weight
  ✅ S1b 只有 LoRA 要梯度,主干全冻  非 LoRA 的可训张量 无
  ✅ S2 enable_gradient_checkpointing 落到顶层模型上(fork 的 forward 读的是它)  gradient_checkpointing=True
  ✅ S3 LoRA 初始为恒等(lora_B=0),disable/enable 逐位相同  max=0.000e+00
  ✅ S4 第 0 步 loss 非零 —— 差异来自 mask 而非 LoRA(LoRA 此刻还是恒等)  mse=4.938758e-08
  ✅ S5a student 输出形状 = 噪声段(ref 位置不进 loss)  (1, 16, 64)
  ✅ S5b teacher 已 detach,不带梯度  requires_grad=False
  ✅ S5c 检查点下能反传,且拿到梯度的恰好是该拿的那 14 个  非零 14/32 | 逐个点名一致
  ✅ S6 反传之后 cache 仍为空(store=False 而非事后 clear)  cache 里有 0 层
  ✅ S7a optimizer.step() 之后 LoRA 参数确实动了  29/32 个张量变化
  ✅ S7b 更新后 teacher 回到原样(disable 是真的关掉,不是近似)  max=0.000e+00
  ✅ S7c 更新后 student 确实变了(否则 LoRA 没接进前向)  max=1.562e-03
  ✅ S8 lora_state / set_lora_state 往返一致(断点续跑靠它)  max=0.000e+00
  ✅ S10 训练的 t 与推理逐位相同(不是只舍入一次的那个)  与推理一致=True | 朴素写法差 3.906e-03
  ✅ S11 set_lora_state 遇到对不上的 key 会 raise(不是静默丢掉)  raise 了
[自检] LoRA 已加载 /tmp/tmprpoat2dq/step000001.pt | step 1 | rank 8 | 32 个张量,其中 lora_B 非零 14 个
  ✅ S12a apply_lora_ckpt 能吃下 train_iso.save() 的格式  OK
  ✅ S12b 未训练的 ckpt(lora_B 全 0)会被拦下,不会冒充 iso_post  拦下了
  ✅ S9 sigma 网格 = 推理实际那 40 个(单调降,首 1.0000 末 0.0200)  n=40
  ✅ S9b 网格与 ref 数无关(seq_len 只算噪声段)  同 seq_len 复现一致

====================================================================
✅ 19/19 项通过。
   还没验的三样只能上真机:峰值显存 / s-it / DDP all_reduce。
====================================================================
```

**结论:19/19 通过**(执行单预期 18/18,实际 19 项,全部通过;S10/S11/S12a/S12b 四条回归钉的均在)。

---

## §2 补完 prompt_embeds(GPU 0,~1 h)

**命令:**

```bash
CUDA_VISIBLE_DEVICES=0 nohup $E/bin/python -u qwen/train_iso.py precompute > /tmp/precompute.log 2>&1 &
```

**结果:** 完成 8488 条,跳过 512 条 → `/kaimm-distill/wuwenxuan/UNO/cache/prompt_embeds`。

- **`txt token 实测:min 213 / 中位 426 / max 635`**(执行单 §4.4 账上「估 400–600」的实测值,中位 426)
- `ls cache/prompt_embeds | wc -l` → **9000**
- `du -sh cache/prompt_embeds` → **30G**(预期 ~32GB)

---

## §3 5 步冒烟(GPU 1)

**命令:**

```bash
CUDA_VISIBLE_DEVICES=1 $E/bin/python -u qwen/train_iso.py train \
    --steps 5 --log_every 1 --allow_partial_embeds \
    --out /tmp/smoke_iso 2>&1 | tee /tmp/smoke5.log
```

**开头自检两行(原样):**

```
[自检] embeds 覆盖 512/9000 | 1-ref 0/1000 2-ref 512/4000 3-ref 0/4000
[自检] LoRA rank 64 | 可训参数 188.7 M | dtype torch.float32 | seed 20260813(各 rank 一致) | target ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
```

- `dtype torch.float32` ✅(修的第 ② 条已在——打出 bf16 就是没修进去)
- `seed 20260813(各 rank 一致)` ✅(修的第 ① 条在)
- `可训参数 188.7 M` ✅(rank 64 × 8 模块 × 60 层 × 2,精确命中)

**逐步输出(原样):**

```
[17:01:31] step 1/5 | loss 0.00328 | 9.2 s/it | 峰值 46.6 GB
[17:01:39] step 2/5 | loss 0.00745 | 8.5 s/it | 峰值 48.0 GB
[17:01:47] step 3/5 | loss 0.00308 | 8.3 s/it | 峰值 48.0 GB
[17:01:54] step 4/5 | loss 0.00400 | 8.1 s/it | 峰值 48.0 GB
[17:02:02] step 5/5 | loss 0.00331 | 8.1 s/it | 峰值 48.0 GB
  ✓ 存 /tmp/smoke_iso/step000005.pt

====================================================================
训练结束 | 5 步 | 0.7 min | 8.6 s/it
峰值显存 48.0 GB
loss 首 0.00422 → 末 0.00422
====================================================================
```

**三个数的判读:**
- **峰值显存 48.0 GB** —— 远低于 130 GB 红线(3-ref 更长,§4 再量真实尖峰)。
- **s/it 8.6 s** —— 比执行单粗估 20–30 s **快 2–3 倍**。注意 §3 只在 512 条 2-ref 上跑,单一桶;
  §4 全分布才是定步数的真数。若 §4 也快这么多,正式跑 1000 步成本会显著低于预估。
- **loss 小且非零**(0.003–0.007),无 NaN/inf;5 步内无单调趋势,属执行单「5 步太短看不准,
  记下往下走,§4 再看」的情形。

---

## §4 100 步标定(GPU 0,§2 §3 过后)

**命令:**

```bash
CUDA_VISIBLE_DEVICES=0 $E/bin/python -u qwen/train_iso.py train \
    --steps 100 --log_every 10 --out /tmp/calib_iso 2>&1 | tee /tmp/calib100.log
```

**开头自检行(原样,确认没带 `--allow_partial_embeds`):**

```
[自检] embeds 覆盖 9000/9000 | 1-ref 1000/1000 2-ref 4000/4000 3-ref 4000/4000
[自检] LoRA rank 64 | 可训参数 188.7 M | dtype torch.float32 | seed 20260813(各 rank 一致) | target ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
```

**10 个 loss 采样点(每 10 步一行,原样):**

```
[17:40:41] step 10/100 | loss 0.00372 | 7.7 s/it | 峰值 50.8 GB
[17:41:53] step 20/100 | loss 0.01648 | 7.5 s/it | 峰值 50.8 GB
[17:43:14] step 30/100 | loss 0.00475 | 7.7 s/it | 峰值 50.8 GB
[17:44:13] step 40/100 | loss 0.00246 | 7.2 s/it | 峰值 50.8 GB
[17:45:42] step 50/100 | loss 0.00290 | 7.6 s/it | 峰值 50.8 GB
[17:46:53] step 60/100 | loss 0.00254 | 7.5 s/it | 峰值 50.8 GB
[17:48:07] step 70/100 | loss 0.00949 | 7.5 s/it | 峰值 50.8 GB
[17:49:11] step 80/100 | loss 0.00544 | 7.3 s/it | 峰值 50.8 GB
[17:50:14] step 90/100 | loss 0.00375 | 7.2 s/it | 峰值 50.8 GB
[17:51:25] step 100/100 | loss 0.00325 | 7.2 s/it | 峰值 50.8 GB
  ✓ 存 /tmp/calib_iso/step000100.pt

====================================================================
训练结束 | 100 步 | 12.1 min | 7.2 s/it
峰值显存 50.8 GB
loss 首 0.00372 → 末 0.00325
====================================================================
```

**判读:**

- **s/it 7.2 s** —— 执行单 §5 粗估「修正后约 60 s/it」,实测快 **8 倍多**。1000 步 ≈ **2 小时**,
  不是 17 小时。这是决定步数/accum 的头号依据。
- **峰值显存 50.8 GB** —— 100 步里 1/2/3-ref 都混入,3-ref 最长档(≈16.8k 序列)没有 OOM,
  全程恒定 50.8 GB,距 130 GB 红线还有 ~80 GB。write 模式 mask 的显存侧这次量到了。
- **loss 形状** —— 小且非零,整体微降(0.00372 → 0.00325),带噪声;step20(0.0165)与
  step70(0.0095)两个尖峰疑似 1-ref/3-ref 档位 loss 尺度差异,标定只读数不做收敛判断。
  无恒 0、无 NaN/inf。
