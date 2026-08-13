# P1 门禁 · 等价性自检执行报告

> 对应 `qwen/P1_GATE_RUN.md`。纯 CPU 结构自检,零 GPU、零权重加载。
> 执行机器:`aiplatform-bjy-ge47-391`(4090 开发机),执行时间 2026-08-13。
> 命令原样 + stdout 原样全文 + 版本,不转述、不判读。

---

## 1. 命令原样

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
cd $R && git pull          # $R = /kaimm-distill/wuwenxuan/UNO
$E/bin/python qwen/test_iso_equiv.py -v
```

`git pull` 输出:`Already up to date.`(HEAD `b3c0524`)。

---

## 2. stdout 原样全文

退出码 `0`。

```text
[自检] 小模型 2 层 / hidden 32 | txt 7(pad 2) | noise 16 | refs [16, 24] | fp32 CPU
  ✅ T4 RoPE 陷阱:摘掉 ref 会平移 txt 频率(所以 read 必须传完整 img_shapes)  两份 txt_freqs 不同 ⇒ 陷阱确实存在
  ✅ T1 改写没改语义:stock processor == off 模式(含 txt padding mask)  max=0.000e+00 mean=0.000e+00
  ✅ T2 mask 不是摆设:全注意力 != 隔离注意力  max=1.250e-03 mean=1.999e-04
  ✅ T3 【硬门禁】隔离-无缓存 == 隔离-有缓存  max=0.000e+00 mean=0.000e+00  (fp32 判据 max<1e-5)
  ✅ T5 ref K/V 与去噪步无关(t=0.8999999761581421 vs t=0.10000000149011612,2 层全比)  max=0.000e+00
  ✅ T6 ref K/V 与 prompt 无关 ⇒ cond/uncond 共享缓存(80 次前向 1 写 79 读)  max=0.000e+00

====================================================================
✅ 6/6 项通过。P1 结构门禁过。
   下一步:全权重 bf16 那一遍走 infer_hub 确认(弱判据,像素差 mean<0.5)。
====================================================================
```

### 2.1 stderr 附注(未计入 stdout)

stderr 另有两行无害 warning,原样附上:

```text
/kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
There are modules in QwenImageTransformer2DModel that should be kept in float32: []. Casting directly with `to()` can lead to inconsistent results; set `torch_dtype` in `from_pretrained()` instead to keep these modules in float32.
```

---

## 3. 环境版本

```text
$E/bin/python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__)"
2.5.1+cu124 0.40.0.dev0
```
