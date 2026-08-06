# M6 步骤 1 执行单 — 数据补齐 + ZeRO-2 标定

> 对应 `distill/M6_ABLATION_SPEC.md` §8 的 P1。**档位:🟢 绿档**——不写不改任何代码,
> 只是按环境变量调用既有脚本。总耗时 ~36 h,其中人要盯着的只有 ~30 min,
> GPU 占用 ~20 min(标定),其余是无人值守下载。

## 这一步在干什么(先看懂再跑)

P1 要交付**两件**东西,缺一件都不能进 P2(那是 7 天 8 卡):

1. **官方口径的 stage-1 训练集** —— 404,258 条 `score_final ≥ 4`。
   现在磁盘上只有 6/102 个分片,可用满分样本 16,966 条 = **4.2%**。
   用 4.2% 的数据训 40k 步 = 走 18.9 个 epoch,比的是"谁更会背这 17k 张图",
   不是隔离的效应(算式见 SPEC §4.0)。所以数据必须补齐,这不是奢侈品。

2. **ZeRO-2 能在这台机器上正常存 checkpoint 的证据** ——
   `train.py` 的 DiT 刚从 ZeRO-3 切回官方的 ZeRO-2(commit `3172b67`)。
   ZeRO-2 下 `accelerator.get_state_dict(dit)` 走
   `clone_tensors_for_torch_save(model.state_dict())`,先取**完整 25.8 GB**
   state_dict 再过滤成 304 个 LoRA 张量,**8 个 rank 同时来一次整模型量级的内存峰值**。
   上游就是这么跑的,但**没在这台机器上验过**。
   172 GPU-小时的长跑上第一次用没验过的并行配置——这是全程唯一的真风险,
   而验它只要 20 分钟。

两件事**互不占用**:下载不吃 GPU,标定不吃网。所以**先起下载,再跑标定**。

## 产出去哪

| | 路径 | 说明 |
|---|---|---|
| 数据 | `datasets/UNO-1M/images/split*/` | 约 2.0 TB,`.gitignore` 已忽略 |
| 训练集 | `datasets/UNO-1M/stage1_official_score4.json` | **文件名带 `_partial` 就是没过关** |
| 标定 | `log/zero2_calib/`、`log/zero2_calib_full/` | 一次性产物,验完可删,**不要**用 `log/stage1_official*` |

⚠️ **标定的 `PROJECT_DIR` 绝不能用 `log/stage1_official`**——那是 P2 student 腿的正式目录。
`train_stage1_official.sh` 只在 `REF_ISOLATION=False` 时硬拦这个名字,`True` 时拦不住。

---

## 步骤

### 0. 拉代码,确认四个 commit 到位

```bash
cd ~/UNO && git pull
git log --oneline -4
```

预期(从新到旧):

```
3172b67 fix(train): DiT 的 DeepSpeed 切回官方 ZeRO-2,删掉 4090 遗留的 zero.Init
ba5bf3c docs(m6): §4 定档——选 A(全量 404,258 条 + 100000 步),本节冻结
3212558 fix(m6): UNO-1M 实测是 1.9 TB 不是 118 GB,速率 40–50 MB/s 不是 0.33
3ff83f8 feat(m6): 隔离消融的预登记 + 两腿共用一个脚本所需的最小改动
```

**对不上就停下来报告。** 少了 `3172b67` 的话下面的标定测的还是 ZeRO-3,白跑。

顺手确认改动真的在:

```bash
grep -n "zero2_config\|zero.Init" train.py
```

预期:`zero2_config.json` 出现在 plugin 行;`zero.Init` **只出现在注释里**,没有真调用。

### 1. 起下载(后台,~35 h,先起这个)

> **⚠️ 不要放在训练机上跑。** 2026-08-06 实测:训练机有 GPU 利用率考核,
> 这个纯 CPU 的长任务跑到第 5 片就被强杀。放在挂同一块 ceph 的 4090 机器上。
>
> **⚠️ 时间账已修正。** 一次性测试拿到的 63.9 MB/s 是走运的突发值,连跑 4 片的
> 实测是 **18.7–25.5 MB/s**,单片下载 958–1191 s + 解压 306–347 s ≈ **1370 s**。
> 92 片 ⇒ **约 35 h**,不是原文写的 16 h。

```bash
cd <4090 上的仓库> && git pull
mkdir -p logs

export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export HF_HUB_ENABLE_HF_TRANSFER=1
unset HF_HUB_OFFLINE

# 换机器时**必须**先空跑确认路径:脚本默认写 <仓库>/datasets/UNO-1M,
# 4090 上若是另一个 checkout,不显式 --dir 就会在本地盘从零开始下。
python scripts/fetch_uno1m.py --dir /kaimm-distill/wuwenxuan/UNO/datasets/UNO-1M --dry_run
```

预期第一行:`分片 102 个,已解压 10 个,待处理 92 个(下载量约 1.9TB)`。
**显示 5 或 0 就是路径指错了,停下来。**

确认无误再起:

```bash
setsid python scripts/fetch_uno1m.py \
  --dir /kaimm-distill/wuwenxuan/UNO/datasets/UNO-1M \
  --rm_tar --min_free_gb 500 \
  > logs/fetch_uno1m.log 2>&1 < /dev/null &
echo "pid=$!"
```

- `--rm_tar` **必须带**,否则占 3.9 TB 而不是 1.9 TB
- `--min_free_gb 500`:两次 `df` 之间共享 ceph 少了 7 TB(别的租户在写),
  默认的 40 GB 门槛在这种盘上太贴地。低于 500 GB 会自己停,腾出空间重跑即可
- **别用 `nohup`**,`setsid` 才躲得开 SIGHUP(§11.12(a) 的教训)
- **同一时刻只能有一个进程碰同一个 split** —— 两个进程同时写 `<name>.part` 会互相踩烂

断点安全性:`already_done()` 看 `images/<name>/` 有没有 ≥100 个文件;
`safe_extract` 先写 `<name>.part` 再原子改名,残留的 `.part` 下次会被 `rmtree` 清掉。
不管是下载途中还是解压途中被杀,重跑同一条命令即可。

#### 可选:双实例(零代码改动)

从 63.9 掉到 22 MB/s,可能是代理侧的**单进程限速**而非链路跑满。是的话开两个进程
处理不相交的分片能翻倍;不是的话两边各跑一半,总量不变也没损失。

```bash
setsid python scripts/fetch_uno1m.py --dir <共享路径> --rm_tar --min_free_gb 500 \
  --only $(for i in $(seq 11 56);  do echo split$i; done) > logs/fetch_A.log 2>&1 < /dev/null &
setsid python scripts/fetch_uno1m.py --dir <共享路径> --rm_tar --min_free_gb 500 \
  --only $(for i in $(seq 57 102); do echo split$i; done) > logs/fetch_B.log 2>&1 < /dev/null &
```

**两个列表必须不相交。** 跑 20 min 看两边各自的 MB/s,加起来明显高于 22 就留着,
否则杀掉一个。

之后**不用管它**,直接做第 2 步。查进度:

```bash
grep -c '✅' logs/fetch_uno1m.log      # 已完成几片 / 96
tail -2 logs/fetch_uno1m.log
```

### 2. ZeRO-2 标定(GPU,两次各 ~15 min,与下载并行)

用**旧的** `uno_1m_total_labels_convert.json`(split1-5 那份,磁盘上已经有)。
新数据还没下完,而这一步只验并行配置,不产出任何要用的东西。

```bash
# 2a) student 腿的配置
MAX_TRAIN_STEPS=100 CHECKPOINTING_STEPS=50 \
REF_ISOLATION=True \
TRAIN_DATA_JSON=datasets/UNO-1M/uno_1m_total_labels_convert.json \
PROJECT_DIR=log/zero2_calib \
bash scripts/train_stage1_official.sh 2>&1 | tee logs/zero2_calib_iso.log

# 2b) baseline 腿的配置
MAX_TRAIN_STEPS=100 CHECKPOINTING_STEPS=50 \
REF_ISOLATION=False \
TRAIN_DATA_JSON=datasets/UNO-1M/uno_1m_total_labels_convert.json \
PROJECT_DIR=log/zero2_calib_full \
bash scripts/train_stage1_official.sh 2>&1 | tee logs/zero2_calib_full.log
```

preflight 预期(2a):

```
[preflight] 训练集: datasets/UNO-1M/uno_1m_total_labels_convert.json
[preflight] NNNNN 条 = 官方满分池的 X.X%  [官方口径]
[preflight] GPU 8 卡 / ref_isolation=True / grad_accum=1
[preflight] 100 步 ≈ 0 小时(0.0 天) —— 粗估,只看数量级
[preflight] === 自检通过,开始训练 ===
```

> `[官方口径]` 这个标签在标定里是**假的**——它只看文件名里有没有 `_partial`,
> 而旧的 convert json 两者都不是。标定不产出正式底座,忽略即可。

模型加载几分钟内日志是静的,正常。每存一次 checkpoint 会顺带跑一遍 dreambooth
样例推理(`train.py:505`),所以 100 步实际比"100 × s/it"久,这是预期的。

### 3. 标定验收 —— ⛔ 这是闸门

```bash
for d in log/zero2_calib log/zero2_calib_full; do
  echo "=== $d ==="
  python -c "
from safetensors.torch import load_file
s = load_file('$d/checkpoint-100/dit_lora.safetensors')
empty = sum(1 for v in s.values() if v.numel() == 0)
zero  = sum(1 for v in s.values() if v.numel() and v.float().abs().sum() == 0)
print(f'{len(s)} 个张量,空分片 {empty} 个,全零 {zero} 个')
print('样例:', list(s)[:2])
"
done
```

**必须是 `304 个张量,空分片 0 个,全零 0 个`。** 两个目录都要过。

- **空分片 > 0** 是这一步要抓的主要失败模式:ZeRO 把参数切了却没聚合回来,
  `state_dict()` 返回 numel=0 的占位张量,存出来的 LoRA 是壳。
- **全零 > 0** 也不行。`LoRALinearLayer` 的 `up.weight` 是
  `nn.init.zeros_` 起步的(`uno/flux/modules/layers.py:144`),第 1 步之后就该被
  梯度推离 0;跑了 100 步还全零说明这一半根本没在训。
- 张量数不是 304 ⇒ `requires_grad` 过滤或聚合出了问题。

任何一条不满足就**停下报告**,不要进 P2。
- 训练过程中 host OOM 被杀 ⇒ 同上,停下报告,把 `dmesg | tail -30` 一起带回。

顺便把吞吐读出来 —— 这是重估 P2 时间预算的依据:

```bash
for f in logs/zero2_calib_iso.log logs/zero2_calib_full.log; do
  echo "=== $f ==="
  grep -oE '[0-9.]+s/it' "$f" | tail -20
done
```

> 现有的 5.3–5.9 s/it(隔离)/ 4.86 s/it(全注意力)是 **ZeRO-3 + grad_accum=2** 的数。
> 这次是 **ZeRO-2 + grad_accum=1**,两个维度都变了,所以要重新测。
> 取**后 20 步**的稳态值,前面几步含 warmup 和 CUDA graph 预热,不算数。

验完可以删掉标定目录省盘(可选):

```bash
rm -rf log/zero2_calib log/zero2_calib_full
```

### 4. 等下载跑完 → 出训练集 —— ⛔ 第二个闸门

下载日志出现 `✅ 全部分片就位` 之后:

```bash
python distill/build_stage1_official.py --strict
```

⚠️ 这一步会对 404,258 条记录做 ~808k 次 `os.path.exists`,在 ceph 上可能
**10–30 分钟没有任何输出**,别以为卡死了。

预期结尾:

```
[对官方满分池] 404258 / 404259 = **100.0%**
写入 datasets/UNO-1M/stage1_official_score4.json:404258 条
```

- **文件名不带 `_partial` 才算过关。**
- 带了 `_partial`,或 `--strict` 直接拒绝出文件 ⇒ 磁盘没全。
  把 `[磁盘覆盖率]` 那一整段贴回来,**不要**自己改 `--min_coverage` 凑数,
  也**不要**加 `--allow_partial`(SPEC §4.0 写死了这条)。

## 5. 带回来

1. `git log --oneline -4` 的输出(确认四个 commit);
2. 两次标定的 **preflight 全段** + 第 3 步张量检查的**两段完整输出**;
3. 两次标定的 **s/it 后 20 行**;
4. `logs/fetch_uno1m.log` 的**末尾 15 行**(含 `实测速率` 与是否 `全部分片就位`);
5. `build_stage1_official.py --strict` 的 **`[跳过原因]` 起到结尾**的完整输出。

## ✅ 确认点(我来判,判完才开 P2)

- **闸门 A**:两个标定目录都是 `304 个张量,空或全零 0 个`,且训练没被 OOM 杀掉。
- **闸门 B**:`stage1_official_score4.json` 落盘,文件名**不带** `_partial`,覆盖率 ≥ 95%。
- **重估**:拿两条 s/it 把 P2 的 172 h 重算一遍,写进 SPEC §8。
  ZeRO-2 如果明显更快,总账会往下掉。

两个闸门都过了,才动 P2 的第一条命令。
