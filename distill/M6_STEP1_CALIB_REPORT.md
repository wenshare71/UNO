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

---

## 8. 本地复核 [2026-08-07]

对着 `accelerate==1.1.1` 的 wheel 逐行核过(H800 上 pin 的就是这个版本)。

**§4 根因成立。** 三处原文:

- `accelerator.py:1822` `self.deepspeed_config = deepspeed_plugin.deepspeed_config`
  ——普通属性,每次 `_prepare_deepspeed` 覆盖一次。
- `accelerator.py:3361` `if self.deepspeed_config["zero_optimization"]["stage"] == 3:`
  ——确实读全局,不读 `model` 自己的引擎。
- 全文件 `deepspeed_config` 只出现在 1667 / 1683 / 1720-1742 / 1821-1832 / 3361。
  训练期唯一读取点就是 3361,§4.5「修复无副作用」成立。

**§5 方案 A(调换 prepare 顺序)不能用,已否决。**

```python
# accelerator.py:1868-1870
# pointing for deepspeed_engine_wrapped.backward()
if self.deepspeed_engine_wrapped is None:
    self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
else:
    logger.warning("A wrapped DeepSpeed engine reference is currently tied ...")

# accelerator.py:2233
self.deepspeed_engine_wrapped.backward(loss, **kwargs)
```

`deepspeed_engine_wrapped` 绑的是**第一个** prepare 的引擎,而 `accelerator.backward()`
只认它。改成 t5 → clip → dit 之后 backward 会打到 t5 的引擎上;t5 是纯推理、没有 optimizer,
`engine.backward()` 里 `self.optimizer.backward(...)` 会 AttributeError。
**dit 必须第一个 prepare**,这是 accelerate 的硬约束,不是风格问题。
(旁证:那条 warning 现在每次启动都会打两次——t5 和 clip 各一次。)

因此 §5 末尾「两方案等价」不成立:B 只改算什么,A 还改了引擎绑定。

**已按方案 B 修。** commit 见下,`train.py:468` 改为

```python
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
unwrapped_model_state = clone_tensors_for_torch_save(unwrapped_model.state_dict())
```

即 `get_state_dict` 在 `stage != 3` 时的原样分支(`accelerator.py:3371-3373`)。

**闸门 A 原本要验的那件事仍然没验。** 2a 崩在 `get_state_dict` 入口,
`clone_tensors_for_torch_save` 一次都没执行过,所以
「8 个 rank 各克隆一份 25.8 GB 到 CPU」这个 RUN.md 里写的真风险
(峰值 ~206 GB host RAM)**目前是零证据**。重跑时这才是要盯的东西。

**重跑前**:`rm -rf log/zero2_calib log/zero2_calib_full`(§1 那个空壳会让
闸门 A 的校验脚本读到半个目录)。命令与 §2 完全一致,2a / 2b 都要跑。

**§6 的时间账**:1.12–1.73 s/it 若稳态坐实,单腿 100k 步 = **31–48 h**,
两腿 62–96 h,对比 SPEC §8 现在写的 172 h。**但这是崩溃前不完整采样,不进 SPEC**,
等重跑拿到干净的稳态值再改 §8。
