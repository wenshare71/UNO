# MultiBanana 评测

在 [MultiBanana](https://github.com/matsuolab/multibanana)(CVPR 2026 多参考生成 benchmark)
的一小批任务上,验证我们训的 UNO ref_isolation LoRA 在**没见过的多主体组合**上的泛化。

## 为什么选它

- 纯多参考生成(无编辑污染),任务目录按参考图数分组(`3_*`/`4_*`…),可精确挑 3-参考子集
- 主体是 concrete object,接近我们的训练分布
- 覆盖 domain/scale mismatch、rare concept 等难例,是压力测试

**注意:数据集没有 ground-truth 目标图**。`_generated` 后缀是留给我们写模型输出的,
benchmark 靠 VLM-as-judge(`judge.py`,用 Gemini/GPT/Qwen3-VL)打 5 维分,不是和真值比像素。
所以定量只能靠它的 judge 或身份相似度,不能算 GT 距离。

## 两步流程(都在远程跑)

```bash
# 1) 只下一小批(联网,不需要 GPU)。默认三个 3-参考子目录各取前 5 个任务
python multibanana_eval/download_multibanana.py
#   自定义:--task_dirs 3_back 3_local --max_per_dir 8

# 2) 用我们的 LoRA 推理(单卡)。默认对照 official_full vs ours_kv,ref_size=512
python multibanana_eval/infer_multibanana.py \
    --lora_path log/ref_isolation/checkpoint-7000/dit_lora.safetensors \
    --model_type flux-dev-fp8
```

产出(在 `output/multibanana_eval/`):
- `ALL_COMPARISON.png` —— 每行 `[参考图 | 各变体]`,变体标签带 denoise 耗时
- `compare__<task>.png` / `<task>__<variant>.png` —— 单例
- `results.json` —— 逐任务计时 + 加速比 + 峰值显存

同时把主变体(ours_kv)结果写回各任务目录 `<num>_generated.jpg`,可直接跑 MultiBanana 的
`judge.py` 用 VLM 打分。

## 参数备忘

- `--ref_size 512`:上一轮 ckpt7000 对照实验里 512 的多主体保留明显好于 320(默认已是 512)
- `--max_refs 4`:只处理 ≤4 参考的任务,跳过的会打印;更多参考显存吃紧且超训练分布
- `--variants`:可选 `official_full ours_full ours_kv ours_iso`,语义同 `scripts/smoke_ref_isolation.py`
- `--dry_run`:不用 GPU,验证任务发现 / 拼图 / 落盘流程
