# infer_hub 冒烟执行单 — 验推理队列可用性

> 2026-08-10。**档位:🟢 绿档**——不改任何既有代码,只往共享推理队列投一个任务。
> 队列文档见 `docs/infer_hub/`(README / FUNCTION / USAGE)。
> 单卡、一次生成,墙钟 ~15 min,不占训练机的卡。

## 这一步在干什么

P2 两腿还在跑(iso ~71000 步 / full ~77000 步),P3 未开始。趁这个空档验一件事:
**P4 的评测能不能搬到 infer_hub 上跑**。要证的是通路,不是画质:

1. worker 能不能从 github 拉到我们的 commit(机房走 `git_proxy`);
2. `/kaimm-distill/wuwenxuan/UNO/.venv-uno` 在推理机上能不能激活、torch 能不能看到卡;
3. `hf_cache` 里那 76 GB 权重能不能从推理集群读通(`HF_HUB_OFFLINE=1` 不联网);
4. **训练机写的 checkpoint,推理机能不能看见并读通**——这条是 P4 的命门;
5. FLUX 从 ceph 加载要多久 —— 这个数决定 P4 的 `--timeout` 怎么给、要不要拆 shard。

## ⚠️ 不用我们的 ckpt 出图,这是有意的

本单里 checkpoint 会被**完整读一遍**(证 4),但**出图走官方 LoRA**。原因在
`uno/flux/util.py:274-279`:

```python
if hf_download:                       # 默认 True,UNOPipeline 没覆盖
    try:
        lora_ckpt_path = hf_hub_download("bytedance-research/UNO", "dit_lora.safetensors")
    except:
        lora_ckpt_path = os.environ.get("LORA", None)
```

官方 LoRA 在 `hf_cache` 里,`HF_HUB_OFFLINE=1` 下这行**照样命中**——所以 `LORA=`
环境变量是**静默失效**的,设了也不会生效,还会让人以为验过了。真要拿我们的 ckpt
出图,得走 `distill/eval_multiref.py` 的 `--*_lora` swap 路径,那是 P4 的活。

> 顺带这也让本单对 SPEC §7.1 完全无害:**不生成任何一条腿的图**,不存在"看过"的问题。
> 这是本单的设计约束之一,不许为了"顺便看看效果"改掉。

---

## 阶段 0 · 前置检查(H800 上,纯 CPU,秒级)

**第一条是 blocker。** 执行单一直写 `cd ~/UNO`,但 `docs/infer_hub/USAGE.md` §0 说
`/home/<user>` 是各机独立的 mmu_ssd 挂载,**推理机上看不见**,提交端会直接拒绝。
如果 `~/UNO` 不在 `/kaimm-distill/` 下,这件事从根上做不了,把 `readlink` 的输出报回来。

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH

echo "=== 1 仓库在不在共享盘(blocker) ===";  readlink -f ~/UNO
echo "=== 2 venv ===";                       ls -d ~/UNO/.venv-uno/bin/python
echo "=== 3 HF 缓存四件套 ===";              ls -d /kaimm-distill/wuwenxuan/hf_cache/hub/models--*
echo "=== 4 最新 ckpt ===";                  ls -d ~/UNO/log/stage1_official/checkpoint-* | sort -V | tail -2
echo "=== 5 commit 已 push? ===";            SHA=$(git -C ~/UNO rev-parse HEAD); echo $SHA; \
                                             git -C ~/UNO branch -r --contains $SHA
echo "=== 6 CLI ===";                        which infer_submit infer_status
```

逐条对:

| 检查 | 期望 |
|---|---|
| 1 仓库路径 | `/kaimm-distill/wuwenxuan/UNO`。**不在 `/kaimm-distill/` 下就停,报回来** |
| 2 venv | 文件存在。不存在说明 venv 建在了本地 NVMe,推理机看不见,报回来 |
| 3 HF 缓存 | 四个 `models--*` 目录齐:FLUX.1-dev / bytedance-research--UNO / clip-vit-large-patch14 / xflux_text_encoders |
| 4 ckpt | 记下**最新那个编号**,下一步要填进去(不一定是 71000) |
| 5 commit | 第二行要列出 `origin/main`。列不出来 = 这个 commit 没 push,推理机会报 `commit_not_found` |
| 6 CLI | 两个路径都在 |

## 阶段 1 · 先 dry-run

`--dry-run` 只打印将要投的 job json,不真写队列。**`checkpoint-71000` 换成阶段 0
第 4 条看到的实际编号。**

```bash
export PATH=/kaimm-distill/infer_hub/lib:$PATH
U=/kaimm-distill/wuwenxuan/UNO
SHA=$(git -C $U rev-parse HEAD)

infer_submit --owner wuwenxuan --project m2v-aio --cluster h \
  --commit-url https://github.com/wenshare71/UNO/commit/$SHA \
  --weights    $U/log/stage1_official/checkpoint-71000 \
  --output-dir /kaimm-distill/wuwenxuan/hub_smoke/20260810 \
  --uv-env     $U/.venv-uno \
  --label hubsmoke_Iter71000 \
  --gpus 1 --timeout 45 \
  --cmd 'ls -lh $INFER_WEIGHTS_DIR && cat $INFER_WEIGHTS_DIR/dit_lora.safetensors > /dev/null && echo "[smoke] 1/3 ckpt 全量可读" && KEEPALIVE_MAX_ROUNDS=1 KEEPALIVE_SAVE_DIR=$INFER_OUTPUT_DIR python scripts/keepalive_infer.py && echo "[smoke] 2/3 进程正常退出" && test -f $INFER_OUTPUT_DIR/latest.png && grep -q "| OK |" $INFER_OUTPUT_DIR/keepalive.log && echo "[smoke] 3/3 出图成功"' \
  --dry-run
```

**dry-run 被拒绝就停下报回来**,尤其是抱怨 `--cmd` 里有路径的那类错。
硬规矩 3 禁止 `--cmd` 出现未声明的共享盘绝对路径——上面这条命令里一个
`/kaimm-distill/` 都没有(`HF_HOME` 由 `scripts/keepalive_infer.py:43` 自己
`setdefault`,烤在代码里),理论上过得去,但这正是要验的东西之一。

### 几个参数为什么这么填

- `--cluster h` **硬绑定 H 卡**。`m2v-aio` 默认虽然就走 H 卡,但 venv 是在 H800 上编的,
  不能留下回落到 `5kpro` 的可能。
- `--output-dir` 显式指到仓库外。默认值是 `<weights>/infer_results`,那会往**正在训练的
  checkpoint 目录里写**。
- `--label hubsmoke_Iter71000` 是中性名,控制台任务树上不出现 iso / full。
  P4 的盲评标注人是你本人,控制台会主动展示 label 和输出目录,从现在起就别让它带臂名。
- `--gpus 1`:单卡够了,还能让装箱路由把它塞进别人剩下的零头卡里,不占整机。
- `--timeout 45`:H800 上 FLUX 从 ceph 加载 ~7 min,推理集群未必一样,给足余量。
  预计实际 12–15 min 就结束,到不了超时检查那一刻。

## 阶段 2 · 正式投

dry-run 的 json 没问题,**去掉最后一行 `--dry-run`** 原样重投。

## 阶段 3 · 看

```bash
infer_status --owner wuwenxuan
tail -f /kaimm-distill/infer_hub/queues/m2v-aio/logs/wuwenxuan__hubsmoke_Iter71000__*.log
```

**判据是日志里的 `[smoke] 3/3 出图成功`,不是图好不好看。**
出的是官方 LoRA 的图,跟我们两条腿没有任何关系,不要对它做任何质量判断。

---

## 三个预判的失败点

### 一、submodule 卡在 git 准备阶段(最可能)

`.gitmodules` 里有 `datasets/dreambooth → https://github.com/google/dreambooth.git`。
`docs/infer_hub/FUNCTION.md` §3.8 说 worker 会无条件
`submodule update --init --recursive`,机房走 `git_proxy`。

本单**根本用不到 dreambooth**(ref 图取自 `assets/`,随仓库分发),但它拉不下来会让
git 准备整个失败——10 分钟 prep 超时,报错落在 checkout 而不是推理。
**看到准备阶段失败先查这个**,把 worker 日志里 git 那段原样贴回来。不要试图改
`.gitmodules` 绕过(R0)。

### 二、`--cmd` 被提交端的静态检查拦下

见阶段 1 的说明。被拦就停,把拒绝信息原样贴回来。**不要**为了绕过它去改
`keepalive_infer.py` 里的 `HF_HOME` 默认值(R0)。

### 三、出图失败也会 exit 0

`scripts/keepalive_infer.py` 的循环 catch 住异常记 fail,然后
`round_idx >= MAX_ROUNDS` 直接 `return`——**退出码 0**,infer_hub 会把它判成 `done`。
所以 `--cmd` 尾巴上挂了 `test -f latest.png && grep -q "| OK |"`,靠 infer_hub 自动加的
`set -e -o pipefail` 把它翻成真失败。

> **所以别只看 `infer_status` 的 done/failed,一定要 grep 日志里的 `[smoke] 3/3`。**

---

## 带回来

不要贴整份日志,贴这几样:

1. 阶段 0 六条检查的**原样输出**;
2. `[smoke] 1/3` `2/3` `3/3` 三行在不在;
3. **`模型加载完成,耗时 X.Xs`** —— keepalive 日志里的这一行,直接决定 P4 的 `--timeout`;
4. `round      0 | OK | XX.Xs | seed 3407 | case0 | peak XX.XGB` 这一行;
5. `cat $INFER_WEIGHTS_DIR/dit_lora.safetensors` 那步花了多久(从日志时间戳估即可),
   以及 `ls -lh` 打出的文件大小;
6. `infer_status --owner wuwenxuan` 的最终状态行;
7. 任何失败的**原样报错**,不要转述。

## 不要做的事

| | 为什么 |
|---|---|
| 改 `keepalive_infer.py` / `uno/flux/util.py` / 任何既有 `.py` `.sh` | R0。要改报上来,本地改完 push |
| 设 `LORA=` 想让它用我们的 ckpt 出图 | 静默失效,见开头。而且违反本单"不生成任何一条腿的图"的设计约束 |
| 顺手起 watchdog 常驻自动投 ckpt | `USAGE.md` §3 推荐这么用,但 M6 **不许**——SPEC §7.1 禁止中途看结果挑 checkpoint,两腿各 100 个 ckpt 自动投出来就是 200 批中途结果摆在控制台上,"看过"不可逆 |
| 为了省时间跳过 dry-run | dry-run 本身就是要验的东西之一(硬规矩 3) |
| 动 P2 两腿的训练进程 | 本单全程不碰训练机的卡,`--gpus 1` 跑在推理集群上 |

## 确认点

跑完回一句:阶段 0 六条全绿 / `[smoke] 3/3` 出现 / 模型加载耗时 X s。
三样齐了我把实测数字填回本单,并据此写 P4 的投递方式。
