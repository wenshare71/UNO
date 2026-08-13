# P1 门禁:跑一遍等价性自检 · 临时执行单

> 给远程 agent。**这一单只有一条命令。** 全部落绿档(G3:调用方式,不改代码里写了什么)。
> 上下文:`qwen/PLAN.md` §3.1「等价性自检 —— 这是唯一的硬门禁」。

---

## 0. 一句话

`qwen/test_iso_equiv.py` 是我写好的,**你只跑它、把输出原样贴回来**。
纯 CPU、几秒、不占 GPU、不加载任何权重。

---

## 1. 跑

在 4090 开发机上,仓库根目录:

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
cd $R && git pull
$E/bin/python qwen/test_iso_equiv.py -v
```

`QWEN_WEIGHTS` **不需要**设置——这个自检不碰权重。

---

## 2. 它测什么

六项,`-v` 会逐项打印实测差值。**T3 是硬门禁,其余五项是它的前提或旁证。**

| | 断言 | 不过说明什么 |
|---|---|---|
| T4 | 摘掉 ref 会平移 txt 的 RoPE 频率 | 测试用例没构造出陷阱(该项是回归保险,不是实现的 bug) |
| T1 | stock processor == 我的 `off` 模式,**逐位相同** | 我重写 processor 时改动了语义,或把 txt padding mask 弄丢了 |
| T2 | 全注意力 != 隔离注意力 | mask 没生效,于是 T3 是空的 |
| **T3** | **隔离-无缓存 == 隔离-有缓存**(fp32,max < 1e-5) | **缓存实现有错。门禁不过,后面所有数字都没有意义** |
| T5 | ref K/V 在 t=0.9 与 t=0.1 下逐位相同 | `PLAN.md` §1 的归纳在实现上不成立 |
| T6 | ref K/V 换 prompt 后逐位相同 | cond/uncond 不能共享缓存,加速比要重算 |

退出码 0 = 全过,1 = 有不过。

---

## 3. 回报

新建 `reports/20260813-p1-gate/REPORT.md`(绿档,你可以写),内容只要三样:

1. 命令原样;
2. **stdout 原样全文**(不要摘要、不要只贴"通过"两个字——我要看每一项的实测差值);
3. `$E/bin/python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__)"`。

然后 `git add reports/20260813-p1-gate/ && git commit && git push`。
commit message:`test(qwen): P1 等价性自检结果`。

**报错了不要修。** 把 traceback 原样贴进报告的「未通过」一节,push,停下等我。
`qwen/*.py` 是我写的,修它是红档(R0 对新文件的等价约束:代码是我的判断,不是你的)。

---

## 4. 明确不要做的

| 不要 | 为什么 |
|---|---|
| 改 `qwen/*.py` 任何一行 | 报错了我改 |
| 加载真权重、跑 GPU、跑推理 | 这一单是纯 CPU 结构自检。全权重那一遍走 infer_hub,是下一单 |
| 调 `TINY` / `IMG_SHAPES` / `SEED` / 容差 | 判据的一部分。调了就不是同一个测试 |
| 写"通过了所以没问题"之类的结论 | 你交 stdout,判读是我的事 |
