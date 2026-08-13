# P2 训练路径自检 + prompt_embeds 预算 · 执行报告

> 对应 `qwen/P2_LAUNCH_RUN.md`。执行机:先 `aiplatform-bjy-ge47-391`(4090),后
> 切 **`aiplatform-wlf3-ge90-8`(H800,8×143GB)**,执行时间 2026-08-13。
> 结果:**§1 ✅、§2 ✅、§3 ⚠️ 中途被用户叫停(512/9000)**。§4 训练按用户要求立即开跑(见末尾)。

---

## §1 · 训练路径自检(纯 CPU)

**结果:✅ 15/15。** 首跑(4090)曾 S5c 不过(14/32 计数),用户修复 `8b26cbc`
(S5c 改点名而非计数),H800 上随 §2 `--pipe` 一起重跑全过。§1 与 §2 的 tiny 段共用同一份输出。

## §2 · 真 pipeline 段(CPU 加载 54 GB)

### 命令原样

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
$E/bin/python -u qwen/test_train_smoke.py --pipe
```

> 注:`-u` 无缓冲(手册原命令无,裸管道下 stdout 块缓冲,进度会被吞)。脚本 `run_pipe` 段**硬编码 CPU**(docstring"只验 API 接线,不上卡"),无 GPU 开关。H800 上跑了 ~15 分钟(CPU VAE encode),过程中被用户改判跳过,但进程未被杀,**自行跑完,21/21 全过**。

### stdout(检查项原样全文,退出码 0)

```text
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
✅ S9 sigma 网格 = 推理实际那 40 个(单调降,首 1.0000 末 0.0200)  n=40
✅ S9b 网格与 ref 数无关(seq_len 只算噪声段)  同 seq_len 复现一致
✅ P1 manifest 读入 + 泄漏断言通过  9000 条
✅ P2 prompt_embeds 缓存不存在,只验 latent 那一半  (缺 /kaimm-distill/wuwenxuan/UNO/cache/prompt_embeds/000000.pt)
✅ P3 x₀ 打包成 4096 个 token(1024² / 8 / 2 的平方)  (1, 4096, 64)
✅ P4 2-ref 的 image_latents = 8192 个 token  (1, 8192, 64)
✅ P5 img_shapes 第一项是噪声图 (1,64,64)  [(1, 64, 64), (1, 64, 64), (1, 64, 64)]
✅ P6 img_shapes 的 token 数与 latents 对得上  8192 vs 8192

====================================================================
✅ 21/21 项通过。
====================================================================
```

完整 stdout(含 tqdm 进度条)在 worker/本机日志:`/tmp/p2launch_s2_h800.log`。

## §3 · prompt_embeds 预算(1 卡)

### 命令原样

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
$E/bin/python -u qwen/train_iso.py precompute 2>&1 | tee /tmp/precompute.log
```

### 状态:⚠️ 中途被用户叫停

- VL 7B 已上 H800 GPU 0(约 16 GB),pipeline 加载完成,开始写 embeds。
- **写到 512/9000 个** `cache/prompt_embeds/{000000..000511}.pt` 时,用户指令"停止一切进程,写报告,立刻开训",任务被停(后台任务 `brokmij0n` 已停,无残留进程)。
- **`txt token 实测:min / 中位 / max` 未打出**(没跑完,§4.4 的 token 账仍缺这个数)。
- 断点续跑:重跑同一条 precompute 命令即可续(已存在的文件跳过)。

```text
du -sh cache/prompt_embeds   →   §3 停时 ≈ 1.7 GB(512 个 pt)
```

## §4 · 训练(用户要求立刻开跑)

用户明确指令"要求立刻开启训练"。按手册 §4 顺序从 ① 单卡 5 步开始。

**⚠️ 前置风险(判读归你)**:缓存只完成 512/9000,`train_iso.py` 的缓存门槛只看目录存在
(`os.path.isdir`,能启动),但训练 `Batcher` 是**全量随机抽样**(按 n_refs 分桶后随机),
5 步冒烟大概率撞上缺失的 embed 文件(`FileNotFoundError`)。若崩:贴 traceback,等判读。
若要完整训练,先把 §3 precompute 补完。
