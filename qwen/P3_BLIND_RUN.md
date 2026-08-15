# P3 盲评启动 —— 给远程会话的执行单

> 前一单 `qwen/P3_EVAL_RUN.md` 已完成:三臂各 240 张出齐,diag_kv v2 PASS
> (`reports/20260814-p3-eval/REPORT.md`)。本单只做一件事:**把盲评界面架起来,交给用户标注。**

---

## 0. 你在做什么

`p3_iso_post`(隔离注意力 + ref KV 缓存,训练后)对 `p3_full`(stock 全注意力 teacher),
240 对,同批混入 30 条 `run_floor`,共 **270 对**。标注者是**用户本人**,不是你。

**这一单不用 GPU、不用 infer_hub、不用 sudo、不用排队。** 全是本机 CPU 上的几条命令。
图已经在盘上(`output/p3_full/` `output/p3_iso_post/` `output/p3_floor/{a,b}/`)。

---

## 1. 什么时候停

**默认动作是继续。** 这份单里的任何"预期值"都是作者的猜测,不是门禁;数没对上就记进报告接着做。

🔴 真要停的只有两条:

1. `qwen/build_pairs_p3.py` 报 **缺图** —— 720 张没齐,配不出 270 对。把它打印的缺失清单贴过来。
2. 需要改任何 `.py`。

🟡 其余全部是"记下来,继续"。⚪ **清单不是权威,机器上看到的才是**;发现前提错了以机器为准,做完在报告里指出来。

---

## 2. 做什么

### 2.1 拉代码 + 合 results(约 1 分钟)

```bash
cd /kaimm-distill/wuwenxuan/UNO && git pull
for v in full iso_pre iso_post; do          # --merge 不加载模型,不需要 QWEN_WEIGHTS(本地验过)
  /kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python qwen/infer_iso.py \
    --merge --variant $v --out output/p3_$v
done
```

把每臂打印的 `n_missing_png` 记下来(**应为 0**)。这一步只是给三臂各留一份 `results.json`
存档,不是门禁 —— 真正的门禁是下一步按 png 逐张查。

### 2.2 生成配对清单(约 1 分钟)

```bash
/kaimm-distill/wuwenxuan/envs/qwen-edit/bin/python qwen/build_pairs_p3.py
```

写出 `output/p3_eval/pairs_p3.json`。脚本自己核条数(270 = 240 + 30)、查 540 张图能否解码、
核锚点孪生间距,不过就 `SystemExit`,**不用你对着数字判**。本地已干跑过全路径。

把它打印的「条数 / 孪生间距 / 三分位分布」原样抄进报告。

### 2.3 起盲评服务(常驻)

`fastapi` / `uvicorn` 大概率不在 `qwen-edit` 环境里。**别往推理环境里装东西** —— 另起一个临时的:

```bash
uv venv /tmp/blind && uv pip install --python /tmp/blind/bin/python fastapi uvicorn pillow
cd /kaimm-distill/wuwenxuan/UNO && /tmp/blind/bin/python -m distill.blind_eval.server \
    --pairs output/p3_eval/pairs_p3.json \
    --marks output/p3_eval/blind_annotations_p3.json \
    --host 0.0.0.0 --port 8765
```

后台跑,让它一直开着。启动时服务器会自检清单(缺图/坏图当场炸,这是设计如此)。

然后**回报给用户**:机器名、`http://<机器名或IP>:8765`。如果用户到这台机器没有直连,
让他用 `ssh -L 8765:localhost:8765 <机器>` 打隧道,再开 `http://localhost:8765`。

**到这里你这一单就做完了。** 标注是用户的事,不要替他点。

### 2.4 标注结束后(用户会告诉你)

```bash
/tmp/blind/bin/python -m distill.blind_eval.report \
    output/p3_eval/pairs_p3.json output/p3_eval/blind_annotations_p3.json
```

原样贴回结果,**不要自己解读达标与否** —— 判据在 `distill/M4_EVAL_SPEC.md` §8.2,作者来读。

---

## 3. 明确不做

- **不要看图。** 不打开 `output/p3_*/` 里的任何 png,不生成 `boards/`(带变体名的并排图)。
  盲评没结束之前看图会污染判读,这是纪律不是建议。
- **不要访问 `/api/stats?reveal=1`。** 那是揭盲接口,标注中途看它就是提前拆盲。
- **不要把 `output/p3_eval/` 提交进 git。** 清单里有 `key_0`/`key_1` 语义标签,
  判读结束前进库等于把答案放进仓库(`.gitignore` 是白名单模式,默认已挡住,别显式放行)。
- 不改 `PLAN.md`,不改 `M4_EVAL_SPEC.md`,不动 `distill/` 下任何既有 `.py`(R0)。
- 不做 §9 客观身份留存计数(`iso_pre` 那一层),那是另一单。

---

## 4. 回报

一份 `reports/20260815-p3-blind/REPORT.md`,要这几样:

1. 三臂 `n_missing_png`;
2. `build_pairs_p3.py` 的完整 stdout;
3. 服务地址 + 用户能不能连上;
4. 踩到的坑(原样记,尤其环境/依赖/网络这类);
5. 标注跑完后 `report.py` 的输出原样。

**要记的数**:270 对 = 240 主对 + 30 run_floor;S1 165 / S3 75;盲种 `p3-qwen-iso-20260815`。
