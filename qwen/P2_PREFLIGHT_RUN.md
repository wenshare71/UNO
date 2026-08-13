# 申请 H800 之前的两件事 · 临时执行单

> 给远程 agent。**两个独立单元,A 在 4090 上,B 投 infer_hub。** 全部绿档
> (G3:调用方式;不改代码里写了什么)。
> 上下文:P1 四个文件已齐(`124167e`),结构门禁已过(`reports/20260813-p1-gate/`)。
> 这两件都**不需要 H800**,而且都是「申下机器之前必须知道答案」的问题。

---

## A · 探内网源有没有 cp310 的 torch 2.5.1(4090,纯 CPU,几分钟)

### 为什么

H800 上**只有 python3.10.12**(`docs/H800_REBUILD.md` §1 实测),而 4090 的
`qwen-edit` 环境是 **python3.11**。cp311 的轮子在 cp310 上一个都跑不起来——
上一轮踩的就是这个坑(原话:「旧机器是 3.12,cp312 轮子全部作废」)。

所以 H800 上得**重建**一套 qwen 栈。整件事只有一个真变量:内网源
`pypi.corp.kuaishou.com` 上有没有 cp310 的 torch 2.5.1。有就是十几分钟,
没有就得另想办法——**这个答案要在申请机器之前拿到,不然申下来卡在装环境上。**

### 跑

```bash
E=/kaimm-distill/wuwenxuan/envs/qwen-edit
IDX=https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/
D=$(mktemp -d)

# 1. 源上 torch 有哪些版本
$E/bin/pip index versions torch --index-url $IDX

# 2. 关键一问:cp310 的 2.5.1 能不能下(--dry-run 不真下载)
$E/bin/pip download --no-deps --dry-run --only-binary=:all: \
    --python-version 3.10 --implementation cp --abi cp310 \
    --platform manylinux1_x86_64 \
    torch==2.5.1 -d $D --index-url $IDX

# 3. 同样问法过一遍其余几个(这几个是纯 python 包,大概率没问题,确认一下)
for pkg in "transformers==5.14.1" "accelerate==1.14.0" "peft==0.20.0" \
           "safetensors" "sentencepiece" "einops"; do
  echo "=== $pkg ==="
  $E/bin/pip download --no-deps --dry-run --only-binary=:all: \
      --python-version 3.10 --implementation cp --abi cp310 \
      --platform manylinux1_x86_64 \
      "$pkg" -d $D --index-url $IDX 2>&1 | tail -3
done
```

**不要真装,不要建 venv,不要碰 H800。** 这一单只回答「源上有没有」。

### 两件顺带确认的

```bash
# a. 4090 上 qwen-edit 的确切 python 版本(报告里要有,别靠路径猜)
$E/bin/python -V

# b. diffusers 是不是 editable / 源码树在哪 —— H800 上要复用这份源码,
#    而不是过日本代理去 GitHub 拉(0.66 MB/s 那一档)
$E/bin/pip show diffusers | grep -i "location\|version\|editable"
ls -d /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 && du -sh $_
```

**flash-attn 不用管。** diffusers 默认后端是 `native`,走 torch 自带的 SDPA
(自带 flash kernel),不依赖 flash-attn 这个包。而 H800 没有 nvcc,
这正好去掉唯一一个必须现场编译的依赖。

---

## B · bf16 确认 + 加速比(infer_hub,1 卡)

### 为什么

到目前为止**整个 iso 栈没碰过真权重**:门禁那一遍是 2 层随机权重 / fp32 / CPU,
它证的是结构。60 层 bf16 累积、12.7k 真实序列长度、`(1,1,L,L)` mask 在
真 SDPA 上走哪个 kernel —— 一样都没测过。

同时这一单顺带产出 §4.4 预登记要对的那个数(2-ref 预测实测 1.9–2.0×)。

### 投

先确认 `124167e`(或更新的 HEAD)已经 push,infer_hub 只认已 push 的 commit。

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
SHA=$(git rev-parse HEAD)

infer_submit --owner wuwenxuan --project default --cluster h --gpus 1 --timeout 90 \
  --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
  --weights /kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511 \
  --output-dir /kaimm-distill/wuwenxuan/UNO/output/p2_preflight \
  --uv-env /kaimm-distill/wuwenxuan/envs/qwen-edit \
  --label p2_preflight_bf16 \
  --cmd 'set -e
export QWEN_WEIGHTS=$INFER_WEIGHTS_DIR
python qwen/infer_iso.py --variant full    --limit 6 --out $INFER_OUTPUT_DIR/full
python qwen/infer_iso.py --variant iso_pre --limit 6 --cache_check 3 --out $INFER_OUTPUT_DIR/iso_pre'
```

`--limit 6` 取的是 m6 那 240 条子集的前 6 条,**全是 S1 = 2-ref**,正好对上
§4.4 预测表的 2-ref 那一行。

耗时估算(Q1-B 实测 2-ref 全注意力 60.3 s/img):
full 6 张约 6 min + iso 6 张约 3 min + 无缓存对照 3 张约 3 min + 权重加载 2×约 7 min
≈ **27 min**。`--timeout 90` 留足。

### 三个数

跑完 stdout 里有三样,报告要原样带上:

| 数 | 从哪看 | 判据 |
|---|---|---|
| **像素差** | `[缓存确认]` 行的 `max=` / `mean=` | `mean < 0.5`(沿用 `../scripts/bench_kv_cache.py` 口径)。**这是这一单的主判据** |
| **加速比** | 两次 `report_speed` 的 `中位 s/img` 相除 | §4.4 预测 1.9–2.0×。对不上不改代码,记数就行 |
| **前向次数** | `前向次数:write N / read M` | 6 张图应是 write 6 / read 474(每张 1 写 79 读);带 `--cache_check` 的那次 write 会多出来,正常 |

---

## 回报

新建 `reports/20260813-p2-preflight/REPORT.md`(绿档),分 A / B 两节:

- **A 节**:每条命令 + **原样输出**。最后单独一行写清楚:
  `torch 2.5.1 cp310:能 / 不能`。
- **B 节**:`infer_submit` 输出、job 状态、**完整 stdout**(不要摘要)、
  以及两个 `results_shard0.json` 的 `meta` 段。

然后 `git add reports/20260813-p2-preflight/ && git commit && git push`。
commit message:`test(qwen): P2 前置 — 内网源 cp310 探测 + bf16 确认`。

**出图不进 git**(`.gitignore` 白名单模式,`output/` 默认全忽略)。

---

## 明确不要做的

| 不要 | 为什么 |
|---|---|
| 改 `qwen/*.py` 任何一行 | 报错了我改。贴 traceback 原文,停下等我 |
| 在 H800 上装任何东西 / 申请 H800 | 这一单的目的正是「申请之前先知道答案」 |
| 真装 cp310 的包到 4090 | A 单只探,不装。装了会污染 `qwen-edit` |
| 跑 `precompute` / 跑训练 | 不是这一单的事 |
| 判据没达标时调参数重跑 | 交数字,判读是我的事 |
