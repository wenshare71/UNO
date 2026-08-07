# M6 步骤 1 标定报告 — ZeRO-2 checkpoint 保存失败

> 2026-08-07。对应 `distill/M6_STEP1_RUN.md` 步骤 2/3 的执行结果。
> **结论:闸门 A 失败,不能进 P2。** 根因是 `train.py` 的代码 bug(混档 ZeRO-2/3 + prepare 顺序),
> 不是机器或内存问题。文档说本步"绿档不改代码",故本次未修,待决策。

## 1. 现状

- 只跑了 **2a(iso 腿)**。第 50 步第一次存 checkpoint 时全部 8 rank 崩溃。
- **2b(baseline 腿)未跑**:两腿走同一条保存路径,必然同样崩,不必浪费 8 卡。
- 标定目录残留空壳 `log/zero2_calib/checkpoint-50/`,重跑前需清掉。
- GPU 已释放,无残留进程。数据下载两个进程不在本机,与本报告无关。

## 2. 复现命令(2a)

```bash
source .venv-uno/bin/activate   # 必须,否则脚本第一行 python 都找不到
MAX_TRAIN_STEPS=100 CHECKPOINTING_STEPS=50 REF_ISOLATION=True \
TRAIN_DATA_JSON=datasets/UNO-1M/uno_1m_total_labels_convert.json \
PROJECT_DIR=log/zero2_calib bash scripts/train_stage1_official.sh \
2>&1 | tee logs/zero2_calib_iso.log
```

preflight 全部通过:`50000 条 = 官方满分池的 12.4%`、8 卡、grad_accum=1、自检通过。
模型加载正常(FLUX.1-dev 23G 本地缓存,代理可达),步进正常跑到 50/100。

## 3. 错误

全部 8 rank 同位置抛错(调用栈):

```
train.py:538  main()
train.py:468  unwrapped_model_state = accelerator.get_state_dict(dit)
accelerate/accelerator.py:3365
ValueError: Cannot get 16bit model weights because
`stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False.
```

## 4. 根因

1. `train.py:235-237` 是**混档**配置:
   - `dit`  → `zero2_config.json`(ZeRO-2,commit `3172b67` 从 zero3 切回)
   - `t5`/`clip` → `zero3_config.json`(ZeRO-3,官方原样)
2. `accelerator.prepare()` 每次把**全局** `self.deepspeed_config` 覆盖成当前插件
   的配置(`accelerator.py:1822`)。prepare 顺序是 dit → t5 → clip(`train.py:362/364/366`),
   所以最后全局 config 停在 **stage 3**。
3. 存 checkpoint 时 `get_state_dict(dit)` 只读全局 config(`accelerator.py:3361`)判断
   `stage == 3` → 走 ZeRO-3 分支;但 dit 引擎实际是 stage-2,
   `zero_gather_16bit_weights_on_model_save()` 返回 False → 直接抛错。
4. **以前为什么没事**:ZeRO-3 时代三个插件全是 zero3,`zero3_config.json` 里
   `stage3_gather_16bit_weights_on_model_save: true`,走 `_zero3_consolidated_16bit_state_dict()`
   聚合路径。切回 dit=zero2 后没处理这个残留交互。
5. **修复无副作用**:`self.deepspeed_config` 在训练期只有 `get_state_dict` 一个读取点
   (`accelerator.py:3361`),prepare 之外没人读它。

## 5. 修复建议(待决策,本次未改)

| 方案 | 改动 | 说明 |
|---|---|---|
| **A(首选)** | `train.py` prepare 顺序改为 t5 → clip → dit(dit 最后) | 全局 config 停在 zero2,`get_state_dict` 走文档预期的 `clone_tensors_for_torch_save` 普通路径,继续用 accelerate 官方 API |
| B | `train.py:468` 直接 `clone_tensors_for_torch_save(unwrapped_model.state_dict())` | 绕过出错的 `get_state_dict` 分支,改动面最小,属症状修复 |

两方案等价(都落到 `clone_tensors_for_torch_save(model.state_dict())`),修完重跑 2a/2b 再过闸门 A。

## 6. 顺带观察(未完成采样,不算数)

崩溃前步进已到 ~50 步,稳态 s/it ≈ **1.12–1.73**(ZeRO-2 + grad_accum=1)。
对照旧账:5.3–5.9 s/it(隔离)/ 4.86 s/it(全注意力)是 **ZeRO-3 + grad_accum=2** 的数。
若修完重跑稳态确认,ZeRO-2 明显更快,SPEC §8 的 P2 172 h 总账可大砍。

## 7. 回到 M6_STEP1_RUN.md 的确认点

- **闸门 A:未过。** 两个标定目录都不存在有效 checkpoint(2a 崩在 save,2b 未跑),
  不是文档预期的 OOM,是上面的代码 bug。
- **闸门 B / 重估:未执行**,依赖 A 修好后重跑。
