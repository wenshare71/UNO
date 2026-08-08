# UNO-1M 下载运行手册(RUNBOOK)

> 2026-08-07 起,在 4090 机器(aiplatform-bjy-ge47-391)上补 UNO-1M 全量 102 split。
> 本文是**运行期踩坑记录 + 重跑参照**,给无人值守时自我重跑用。

## 一、当前部署形态

- **数据目录**: `/kaimm-distill/wuwenxuan/UNO/datasets/UNO-1M/images/`(共享 ceph,598T 可用)
- **下载器**: `scripts/download_splits.py`(curl + 256MB range 分块 + 每块大小校验)
- **解释器**: `/tmp/uno-fetch-hf/bin/python`(huggingface_hub 0.36.2 老环境,但 download_splits 不依赖 HF 库,只有 fetch_uno1m.py 依赖)
- **HF_ENDPOINT**: `https://hf-mirror.com`(必须 export;curl 直接打 URL)
- **必须 `unset HF_HUB_OFFLINE`**(训练环境常开 offline=1)

### 三进程分区(互不相交,这是硬约束)

| 进程 | 代理 | 分片区间 | 日志 |
|---|---|---|---|
| P1 | `10.66.29.113:11080` | split15–43 | `logs/dl_p1.log` |
| P2 | `10.66.37.111:11080` | split44–72 | `logs/dl_p2.log` |
| P3 | `10.66.72.150:11080` | split73–102 | `logs/dl_p3.log` |

启动模板(每进程一条,分开跑):
```bash
export HF_ENDPOINT=https://hf-mirror.com; unset HF_HUB_OFFLINE
setsid /tmp/uno-fetch-hf/bin/python scripts/download_splits.py \
  --splits $(seq -f 'split%.0f' 15 43) --proxy 10.66.29.113:11080 --threads 8 \
  > logs/dl_p1.log 2>&1 < /dev/null &
```

## 二、已验证的经验(别推翻)

### 1. 代理会截断 >1GB 的 range 请求(最关键)
- 实测:请求 2.3GB range,只回 1.05GB 就停。导致拼接的 tar.gz 内容错位 → **gzip 解压卡死**(D 状态)。
- **修复**:256MB 固定小块 + 每块下载后校验大小,不符立即 `raise RuntimeError`。
- 现在所有块都是精确 268435456 字节,无截断。

### 2. hf_transfer 走国内代理会 D 状态挂死
- `fetch_uno1m.py`(hf_hub_download + hf_transfer)走国内代理 → 进程卡 uninterruptible sleep,永不返回。
- **国内代理必须用 curl**,不能用 hf_transfer。这就是 download_splits.py 存在的原因。
- 海外 squid(`oversea-squid1.jp.txyun:11080`)带宽被共享租户吃光(实测 0–40MB/s 波动),且按进程限速 ~22MB/s。

### 3. 国内代理是独立带宽通道
- 每个国内代理 ~10–100MB/s 波动,多进程(不同代理)并行可叠加带宽。
- 3 进程 × 8 线程 = 24 curl 并发。实测单进程 ~16MB/s(受 ceph 写入/代理当前负载影响)。
- 3 代理并行合计期望 40–50MB/s,86 片 ~1.5TB ≈ 8–10 小时。

### 4. 幂等性:已解压自动跳过
- `already_done(images_dir, name)`: `images/<name>/` 有 ≥100 个文件即视为完成。
- `safe_extract`: 解压到 `<name>.part` 再原子改名;残留 `.part` 下次进门会被 rmtree。
- **断点重跑同一命令即可**,已完成的不重下。

### 5. 环境细节
- `.venv-uno` 的 python 软链指向 H800 的解释器(4090 上不存在)→ **必须用 `/tmp/uno-fetch-hf/bin/python`**。
- 建 venv 用内网 PyPI `pypi.corp.kuaishou.com/kuaishou/prod/+simple/`(241MB/s),不能用 pypi.org(0.33MB/s)。
- 权限系统禁 pkill/kill;要用 `python -c "import os,signal; os.kill(pid, SIGTERM)"` 杀进程。

## 三、可能遇到的问题与处置(无人值守自跑参照)

### A. 进程退出,日志尾行是 `❌ RuntimeError`
- 看报错内容:
  - `块 N 期望 XB 实际 YB(range 被截断)` → 代理瞬断。**直接重跑该进程**的同一命令即可(残留 part 会被清理重下)。
  - `拼完 N != 预期 M` → 罕见,同样重跑。
- 重跑前先 `ps` 确认没有残留 download_splits 进程,再起。

### B. 进程退出,日志尾行是 `ConnectionError` / `ReadTimeout` / curl 报错
- 代理抖动。**重跑同一命令**。curl 已带 `-sL` + 1800s timeout,重连失败会异常冒泡到主循环 → 单 split 失败,进程继续下一个。
- 如果整进程挂了(没到下一个 split),看 stderr,重跑。

### C. 进程退出,日志显示全部完成但 split 目录数 < 102
- 说明分区列表没覆盖全(编号有洞)。对照 §一 的区间补跑缺失编号。

### D. 磁盘告急(<500GB)
- ceph 是共享的,别的租户在写。`df -h /kaimm-distill` 看。
- download_splits.py **没有** `--min_free_gb`(fetch_uno1m.py 才有)。盘不够时会解压失败(ENOSPC),重跑会因 part 清理失败而报错 → 先腾空间再重跑。

### E. 单进程 8 线程太慢 / 代理带宽突然掉到 0
- 可能代理被其他租户压满。先 `du -sb images/split*.tar.gz.part*` 看该进程是否还在涨;
- 还在涨(>5MB/s)就没事,继续等;
- 掉到 0 且超过 5 分钟 → 该进程的代理挂了,换成另一个国内代理重跑该进程(代理池:10.66.29.113 / 10.66.37.111 / 10.66.72.150 之外还有 10.66.22.211 等,见 scripts/probe_net.sh 注释)。

### F. 三个进程都还没动(启动后 60s 日志只有 3 行 header)
- 卡在第一个 `get_size()` HEAD 请求(代理拒连)。换代理重跑。

### G. hf-mirror 返回 HTTP 429 Too Many Requests(实测踩过)
- **现象**(2026-08-07):P1 用 cn-29.113 代理 8 线程猛拉(峰值 74MB/s),hf-mirror 触发限流,
  split33–43 连续 16 片全部 429,进程退出。同一批里偶尔混着 curl exit 18(部分传输后断)。
- **处置**:429 是按**代理出口 IP** 限流的。**换一个代理重跑**该进程(代理池里挑空闲的,
  如 10.66.22.211),已解压的会自动跳过。同时把线程降到 6 保守。
- **⚠️ 别信 HEAD 探测**:429 后 `curl -I -r 0-1048575` 各代理都返回 302(正常重定向),
  **但实际 range 下载仍 429**——限流针对并发 range GET,HEAD 不触发。判断"恢复没"只看
  重跑后日志有没有再打 429。实测 8 线程全 429 → 6 线程换代理后立刻正常。
- **预防**:快代理(cn-29.113 这种峰值 74MB/s)别开满 8 线程,6 是安全上限;
  慢代理(P2/P3 的 17MB/s)8 线程没触发过限流,可以继续 8。
- **补充(2026-08-07 晚)**:换到 22.211 后虽然不再 429,但只有 6.76MB/s(split23 用了 53 分钟),
  还出现 curl exit 18(连接断)导致 split24/27 失败。**慢代理不值得留** —— 429 通常
  半小时到一小时后自动解除,回来探测(用真下载测 256MB range,看是否 206 而非 429,
  不要用 HEAD)后**切回最快的代理**。当前最优:29.113 恢复后单进程 6 线程跑最快。
  所以处置排序:429 → 换别的代理应急 → 定期回头试原代理,恢复就切回。

### H. curl exit 92(HTTP/2 流错误,三个进程同时挂)
- **现象**(2026-08-07 晚):三进程同一时刻各自在最后一片报 `curl exit 92`,
  即 HTTP/2 framing 层错误 —— hf-mirror/国内代理的 HTTP/2 层不稳定。
- **修复**:`curl_range()` 加 `--http1.1`,降级到 HTTP/1.1 后稳定。已写进
  `download_splits.py`,以后都用 HTTP/1.1(range 下载不受影响)。
- **处置**:直接重跑对应进程即可(脚本已带 --http1.1),已解压的自动跳过。

### I. 三进程同时 exit 18(全局网络抖动)
- **现象**(2026-08-07 晚,两次):三个进程**同时**在各自最后一片报 curl exit 18(连接断),
  不是单进程的事 —— 机房网络抖动,所有代理的 TCP 连接一起断。
- **特征**:三进程同一时刻退出、各自日志尾行都是 exit 18、已解压的片全部完好。
- **处置**:三个进程**一起重跑**(不用换代理,代理本身没坏)。用 `comm` 或直接
  `ls images/split*/` 对比 1–102 找缺片;重跑后已解压自动跳过。
- 幂等性保证:断点续传,重跑不重下已完成的。
- **补充(2026-08-07 深夜)**:抖动会**反复发作**——当晚出现"三进程同时 exit 18 → 我重跑 →
  又立刻在同一批片 exit 18(part0 就断)"的两连击。处置节奏:
  1. 先测网络(`curl -r 0-1048575`,看是否 206 + 完整数据),**通了再重跑**;
  2. 抖动中不要反复重跑,等网络确认恢复再起(否则白跑且日志被刷);
  3. 每次重跑前核对缺片清单(`for i in $(seq 1 102)`),别让某个进程区间错位。

### J. hf-mirror 缓存过期 → 全仓库转 Xet bridge(2026-08-08 凌晨,最坑的坑)
- **现象**:三个进程在最后 3 片集体 exit 18,且**反复重跑全部失败**(含 HTTP/1.1 改成
  HTTP/2 也一样 0 字节)。检测发现:hf-mirror 对 UNO-1M 的 resolve 现在**一律 302 到
  Xet bridge**(`us.aws.cdn.hf.co/xet-bridge-us/...`),不再是直接给数据。签名 URL
  有效期 1 小时。之前能下 99 片是因为 hf-mirror 缓存着旧文件直接出,现在缓存过期回源。
- **症状特征**:
  - `curl -I`(HEAD)拿不到 Location,但 `curl -s -D -`(GET)能拿到;
  - 响应头带 `X-Linked-Size:`(真实大小)和 `rel="xet-auth"` / `xet-reconstruction-info`;
  - **签名 URL 本体必须 HTTP/2**:`--http1.1` 会 `206/0 字节`(curl exit 18),去掉 `--http1.1` 就正常。
- **已修复**:`download_splits.py` 新增 `_signed_url()`:
  1. 每个 split 先 GET hf-mirror,正则抓 `Location`(302 签名 URL)+ `X-Linked-Size`;
  2. 所有 range 块都走签名 URL,curl **不带 `--http1.1`**(HTTP/2);
  3. 签名过期(403/CalledProcessError)→ 线程锁下重新取签名重试一次。
- **验证**:1MB / 256MB / 1GB range 均完整下载(206 + 精确字节数)。split102 走新流程正常。
- **处置**:直接重跑缺片进程即可,脚本自动走 Xet。若又 exit 18,先确认签名 URL
  是否拿到(日志应显示"分 N 块"而不是先报错)。

## 四、健康检查命令(30 分钟定时任务用)

```bash
# 进程存活
ps aux | grep download_splits | grep -v grep | wc -l    # 期望 3

# 每进程完成数 + 最近速率
for f in logs/dl_p1.log logs/dl_p2.log logs/dl_p3.log; do
  echo "=== $f 完成 $(grep -c '✅' "$f") ==="
  grep -oE '\[split[0-9]+\] .*' "$f" | tail -1
  grep -oE '\([0-9.]+ MB/s\)' "$f" | tail -1
done

# 已解压 split 数 / 102
ls -d datasets/UNO-1M/images/split*/ | wc -l

# 当前在下的 part 合计
du -sb datasets/UNO-1M/images/split*.tar.gz.part* 2>/dev/null | awk '{s+=$1} END {print s/1073741824 " GB"}'

# 磁盘
df -h /kaimm-distill | tail -1
```

## 五、全部完成后

```bash
ls -d datasets/UNO-1M/images/split*/ | wc -l   # 必须 = 102
python distill/build_stage1_official.py --strict   # 产出 stage1_official_score4.json,不带 _partial
```

**必须 `✅ 全部分片都已就位` / 102 split**,才到下一步。见 `distill/M6_STEP1_RUN.md` 闸门 B。
