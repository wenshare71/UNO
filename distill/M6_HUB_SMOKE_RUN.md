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

---

## 实测记录(2026-08-10,首次尝试)

**结论:阶段 0 全绿、阶段 1 dry-run 过,阶段 2 卡在提交端权限,已停。**

### 阶段 0 · 六条全绿

1. 仓库 `/kaimm-distill/wuwenxuan/UNO` ✅(在共享盘下,blocker 通过)
2. `.venv-uno/bin/python` 存在 ✅
3. HF 缓存四件套齐(外加 `models--Yuanshi--OminiControl`)✅
4. 最新 ckpt = **`checkpoint-77000`**(阶段 1/2 按它填,不是 71000)
5. commit `d46a2f197336177ca925c6beae8daf685c5e8048` 已在 `origin/main` ✅
6. `infer_submit` / `infer_status` 均在 `$PATH` ✅

### 阶段 1 · dry-run——多出一个「第四类失败点」,已按官方解法绕过

首次 dry-run 被拒,但**不是执行单预判的三类**:

```
[infer_submit] 错误: 本队列已全量切换两阶段流水线:必须声明 --prep-cmd/--prep-marker(切权重命令与完成标志)。
           没有独立切分步骤的任务,--prep-cmd 'true' --prep-marker <权重目录> 即可。
```

`config.json:42` 的 `submit_require_prep: true` 已生效——**m2v-aio 队列强制两阶段**。
按报错信息给的官方解法补上:

```
--prep-cmd 'true' \
--prep-marker $U/log/stage1_official/checkpoint-77000
```

marker 指向必然存在的权重目录 → 提交时即判 `prep_done=true`,任务按「已切分」入队直接排卡;
`prep_cmd 'true'` 不真跑切分,**被验证的通路完全不变**。dry-run 的 job json 核对无误:
repo/commit 正确、cmd 无共享盘绝对路径、`cluster=default`(`--cluster h` 规范化)、
`format=v3` + `prep_done=true`。

> 若执行单原文不含 prep 参数,P4 写投递方式时 **m2v-aio 队列默认要带 `--prep-cmd 'true'`**。

### 阶段 2 · 正式投——卡在提交端写队列权限(新 blocker)

去掉 `--dry-run` 后报 `PermissionError`,不是任务本身的问题:

```
PermissionError: [Errno 13] Permission denied:
'/kaimm-distill/infer_hub/queues/m2v-aio/tmp/wuwenxuan__hubsmoke_Iter77000__d46a2f197336.json'
```

事实链(已逐项核实):

| 项 | 发现 |
|---|---|
| 队列目录权限 | `queues/m2v-aio/*` 全部 **root:root 755** |
| 历史 job 属主 | done/failed/logs 里**全是 root** |
| 本会话身份 | `wuwenxuan03`(uid 1001),无 sudo / su / docker / nsenter 提权 |
| 提交通道 | 控制台**没有 submit 路由**(仅 cancel/kill/pin/restore,走 root web 进程);提交只能 CLI `infer_submit` |
| 系统权限假设 | DESIGN:组内成员互有 root、共享盘大家可写——**提交端本该以 root 跑** |
| wuwenxuan 现状 | 名单里有,但历史 0 个 job、无 watchdog——还没投过任务 |

**执行单原文没预判到「提交端要以 root 跑」这一前提。** 在 H800/开发机上若以
root 登录执行本单即可继续;若会话是普通用户,需先切 root(或确认提交端机器
以 root 起 `infer_submit`)。

### 下一轮怎么续

1. 拿到 root 提交通道后,阶段 2 命令 = 阶段 1 同款 + 去掉 `--dry-run`(已含 prep 两参数);
2. 阶段 3 判据不变:`[smoke] 3/3` + `模型加载耗时 X.Xs` + `round 0 | OK |` 行;
3. 本单成功后再回头验证「m2v-aio 强制两阶段」是否要写进 P4 投递模板。

---

## 实测记录(2026-08-10,第二次——**成功**)

**确认点三样齐:阶段 0 六条全绿 / `[smoke] 3/3` 出现 / 模型加载耗时 96.1s。**
任务 `wuwenxuan__hubsmoke_Iter77000__d46a2f197336`,最终 **done**,exit_code=0,
duration 118.6s,跑在 `aiplatform-wlf3-ge90-10`(worker v2.4.2)。

### 阶段 2 实际走的命令(记录备查)

sudo + HOME 保持(提交端要 root 写队列 + git 走 wuwenxuan 的代理配置):

```bash
sudo env HOME=/kaimm-distill/wuwenxuan PATH=/kaimm-distill/infer_hub/lib:$PATH \
  python3 /kaimm-distill/infer_hub/lib/infer_submit \
  --owner wuwenxuan --project m2v-aio --cluster h \
  --commit-url https://github.com/wenshare71/UNO/commit/<40位sha> \
  --weights    /kaimm-distill/wuwenxuan/UNO/log/stage1_official/checkpoint-77000 \
  --output-dir /kaimm-distill/wuwenxuan/hub_smoke/20260810 \
  --uv-env     /kaimm-distill/wuwenxuan/UNO/.venv-uno \
  --label hubsmoke_Iter77000 --gpus 1 --timeout 45 \
  --prep-cmd 'true' \
  --prep-marker /kaimm-distill/wuwenxuan/UNO/log/stage1_official/checkpoint-77000 \
  --cmd '...'   # 同阶段 1
```

> **root 的 git 环境不带 github 凭据/代理**(`/root/.git-credentials` 只有内网 gitlab),
> 直接 `sudo infer_submit` 会 git 超时。必须 `sudo env HOME=<wuwenxuan>` 让 git 读
> wuwenxuan 的 gitconfig(代理 `oversea-squid1.jp.txyun:11080`)。wuwenxuan 名下此后
> 提交都用这组命令。

### 首次重投的坑(已避免的失败)

第二次投错 commit 报 `prepare_commit_not_found`(job `...__5404d59cd1cb`):
**用 `rev-parse HEAD` 取的 commit 是本地的、没 push**。推理机 mirror 只 fetch 远端,
找不到本地 commit。修法:commit 写死为阶段 0 验证过已 push 的 `d46a2f197336…`。
教训:watchdog/提交脚本里的 COMMIT 必须用 `git ls-remote` 确认过在 `origin/main` 的 sha,
不能 `rev-parse HEAD` 现取。

### 带回来的数字

| 项 | 实测 |
|---|---|
| git 准备(fetch+checkout+submodule) | **9s**(首次,含全量 clone + dreambooth submodule) |
| checkpoint 全量读(1.8G dit_lora) | 通过(`[smoke] 1/3`),ls 到 total 3.2G,dit_lora 1.8G / optimizer.bin 1.4G |
| **FLUX 从 ceph 加载** | **96.1s**,显存 33.4GB |
| 单轮推理(512×512×25 步,单 ref) | **4.4s**,peak 34.5GB |
| 任务总时长 | **118.6s**(含加载+推理) |
| 执行机 | `aiplatform-wlf3-ge90-10`(NVIDIA **H200**,worker v2.4.2) |
| venv 激活 | `.venv-uno/bin/activate` 推理机可见,sourced 正常 |
| `HF_HOME` / offline | `/kaimm-distill/wuwenxuan/hf_cache` + `HF_HUB_OFFLINE=1`,脚本默认,命中 |

### 五条验证结论(对应开头「这一步在干什么」)

1. **worker 拉 github commit 通**(走 git_proxy),首次 clone 9s;
2. **`.venv-uno` 在推理机可激活**,torch 看到卡(H200);
3. **hf_cache 76GB 权重读通**,`HF_HUB_OFFLINE=1` 不联网;
4. **训练机写的 checkpoint,推理机可读通**——P4 命门通过;
5. **FLUX 从 ceph 加载 96.1s** → P4 的 `--timeout` 给 **15 min 足够**(加载 96s + 每批
   推理 4.4s/图,余量拉满),无需拆 shard。

### 预判失败点核销

- 失败点一(submodule/dreambooth 卡 git):**未发生**,9s 就绪;
- 失败点二(`--cmd` 被静态检查拦):**未发生**(命令里无未声明共享盘路径,HF_HOME 烤在代码里);
- 失败点三(出图失败 exit 0):**未发生**,且 `--cmd` 尾部 `test -f && grep` 兜底逻辑本身
  也验证走通(`[smoke] 2/3 → 3/3` 只在这两检查通过后才打);
- 额外发现:提交端 root 权限(已解决)、commit 必须已 push(已记教训)。

### P4 投递方式(据此定稿)

- **环境**:`--uv-env /kaimm-distill/wuwenxuan/UNO/.venv-uno`;
- **commit**:写死已 push 的 sha,不用 `rev-parse HEAD`;推理代码有改动就 push 后更新;
- **卡型**:`--cluster h` 硬绑定 H 卡(H200);
- **超时**:`--timeout 15`(加载 96s + 4.4s/图,留足余量);批量任务按批图数×4.4s + 96s 估,
  必要时逐批投;
- **两阶段**:m2v-aio 强制,无独立切分步骤的任务带 `--prep-cmd 'true' --prep-marker <权重目录>`;
- **label**:中性名,不带 iso/full 臂名(盲评约束);
- **提交方式**:sudo + HOME 保持(见上)。
