# M6 步骤 2 完成报告 — 两腿 stage-1 双机并行 + P3 蒸馏

> 对应 `distill/M6_STEP2_RUN.md` 的 P2 与 P3。**2026-08-11,两腿 stage-1 与蒸馏全部完成。**
> 本报告取代 P2 执行单(commit `8573e35`)的"待办"状态,两腿链路收官,可进 P4。

## 结论先行

`M6_STEP2_RUN.md` §确认点,三条全过:

| 确认点 | 判据 | 结果 |
|---|---|---|
| 两腿 stage-1 跑满 | 各 **100000** 步,无中途挑 checkpoint | ✅ A=100000, B=100000 |
| ref_isolation / PROJECT_DIR | A=True→`log/stage1_official`, B=False→`log/stage1_official_full`,未写反 | ✅ preflight 逐行核对 |
| checkpoint 张量检查 | `304 张量,空分片 0,全零 0` | ✅ 两腿 stage-1 与蒸馏最终 ckpt 全过 |
| P3 蒸馏 | 各 4000 步,最终 ckpt 校验 | ✅ A=4000, B=4000 |
| 保活 | GPU 满载防利用率考核强杀 | ✅ 两机各 8 进程运行中 |

**P2/P3 完成,进 P4。**

---

## 1. 这一步要交付什么(回顾)

P2 交付消融的**两个底座**(stage-1),P3 交付两腿**蒸馏**(4000 步)。两腿只差
`REF_ISOLATION` 一个 flag:

```
A 腿(iso student)  REF_ISOLATION=True   stage-1 → log/stage1_official   → P3 → log/ref_distill_iso
B 腿(full baseline) REF_ISOLATION=False  stage-1 → log/stage1_official_full → P3 → log/ref_distill_full
```

SPEC §3 的同一性条件:两腿同 `stage1_official_score4.json`(404,258 条,同一次
build)、同 `train_mixed.json`(29777 条,冻结未重生成)、**world size=8**(seed 按
process_index 依赖卡数)、同 seed/lr/rank/batch/res。双机并行不违反 §3——钉的是
"每腿 8 卡",不是"同一台机器"。

---

## 2. P2 stage-1(双机并行)

两台 H800 各 8 卡,并行跑。**stage-1 步数 100000 / grad_accum=1 / ckpt 每 1000 步**:

| 腿 | 机器 | preflight | 步数 | 总耗时 | 末尾 s/it | 最终 ckpt 校验 |
|---|---|---|---|---|---|---|
| A (iso) | ge90-95 | `ref_isolation=True / grad_accum=1` | 100000 | **32h51m** | 1.18 | `304/0/0` ✅ |
| B (full) | ge90-85 | `ref_isolation=False / grad_accum=1` | 100000 | **28h45m** | 1.04 | `304/0/0` ✅ |

- 两腿各落盘 **100 个 checkpoint**(1000–100000),单 ckpt 3.2 GB,共 ~640 GB。
- A 腿(iso)比标定 1.09 s/it 略慢(实测 ~1.1–1.2),符合"隔离在训练侧更慢"的预期;
  B 腿(full)接近标定 1.00。**GPU 总账 = 32h51m + 28h45m ≈ 61.6 h**(串行口径,
  比 SPEC §8 重估的 58 h 略高,差量来自 A 腿实际步速,量级一致)。

### 2.1 双机改造(P2 执行单 commit `8573e35`)

原 SPEC §8 是"串行 student 30 h + baseline 28 h"。双机并行只动墙钟
(58 h → ~32 h),不动 GPU 总账,不动同一性条件。硬约束:两台都正好 8 卡——脚本对
卡数只警告不退出,这条靠人工把关(本步两机 `nvidia-smi -L | wc -l` 均为 8)。

### 2.2 B 腿首启的坑(已修)

B 腿(ge90-85)第一次启动用 **root 身份**跑的:留下 root 所有的 `log/stage1_official_full`
目录与 `logs/p2_full.log`,wuwenxuan 无写权限;且训练被 **SIGTERM 杀在模型加载阶段**
(webshell 会话中断连坐,未用 setsid 脱离)。

修复:`rmdir` 重建目录 + `rm` 旧 root 日志,以 **wuwenxuan 身份 + setsid** 起跑。
**教训:B 腿必须 wuwenxuan + setsid,别用 root / 裸跑。**

---

## 3. P3 蒸馏(两腿并行)

stage-1 完成后同一台机器接各自蒸馏。**4000 步 / grad_accum=2 / ckpt 每 1000 步**
(`train_distill.sh` 写死,与 SPEC §3 一致):

| 腿 | 机器 | preflight | 步数 | 总耗时 | 稳态 s/it | 最终 ckpt 校验 |
|---|---|---|---|---|---|---|
| A (iso) | ge90-95 | `ref_isolation=True / out=log/ref_distill_iso` | 4000 | **3h40m** | 3.31 | `304/0/0` ✅ |
| B (full) | ge90-85 | `ref_isolation=False / out=log/ref_distill_full` | 4000 | **2h59m** | 2.70 | `304/0/0` ✅ |

- **三个变量一个都不能漏**(`train_distill.sh` 默认值全是旧的):`PROJECT_DIR`
  (默认 `log/ref_distill` 会覆盖 M3 结果)、`RESUME_FROM_CHECKPOINT`(默认 4090
  旧底座)、`REF_ISOLATION`(默认 True)。本步逐条显式传入并 preflight 核对。
- `train_mixed.json` **未重新生成**(冻结,单 ref 池 16,966 条时代产物;重跑会变
  404,258 条,60/40 混比彻底改掉)。
- **蒸馏 s/it 实测回填**:SPEC §8 旧账"合计 ~12 h"是 ZeRO-3 时代。ZeRO-2 实测
  A 3h40m + B 2h59m ≈ **6.6 h**,快 ~45%。回填 SPEC §8。

---

## 4. 保活与无人值守巡检

两腿链路完成后,两台机器各启动保活(`KEEPALIVE_INTERVAL=0 bash scripts/keepalive_8gpu.sh start`),
8 进程各占一卡、显存满载,防"GPU 空转"被利用率考核强杀(2026-08-06 实测被杀过)。
保活运行中(本机 B 腿:gpu0 心跳已推进至 round ~4000,每轮 ~10.2 s)。

无人值守巡检(2026-08-10 授权):每小时 P3/保活巡检 + 每 2 小时保活状态巡检
(cron),异常自修复(续训 / 补起保活进程)。

---

## 5. 本步的工程产物与踩过的坑

| 产物 | 说明 |
|---|---|
| `scripts/check_lora_ckpt.py` | 字节级 safetensors 校验(不依赖 torch),`304 张量/空 0/全零 0`,校验失败退出码非 0 |
| `run_A_iso.sh` / `run_B_full.sh` / `run_P3_full.sh` | 各腿启动脚本,防呆(setsid + 变量显式) |
| `logs/p2_{iso,full}.log` / `logs/p3_{iso,full}.log` | 四份训练日志(共享盘) |

**坑(已识别,不影响运行)**:

1. **keepalive 共享盘 pid 文件互相覆盖**:`output/keepalive_train/gpuN/` 是共享盘,
   A 腿(ge90-95)启动保活时用其 PID 覆盖了 B 腿的 pid 文件 → 本机 `keepalive_8gpu.sh
   status` 读到不存在的 PID,误报"未运行"。**真实健康以「pgrep 进程数(8)+ 显存满载
   + 心跳推进」为准,不信 status 的 pid 判断**。pid 文件冲突未动(改会干扰 A 腿)。
2. **ceph 瞬时 IO 抖动**:保活写日志偶发 `OSError: Errno 5`(写 stdout.log),
   round 标记 `FAIL(1/10)` 后下一轮自动继续——保活脚本容错设计正常工作,训练与显存
   占用从未中断。
3. **GPU 利用率瞬时 0%**:保活训练轮间间隙,进程 CPU 满载 + 心跳推进 = 正常。

---

## 6. 带回来的证据

1. 两腿 stage-1 preflight(`ref_isolation=True/False / grad_accum=1`)与日志末尾
   (100000/100000 + 总耗时,见 §2);
2. 两腿蒸馏 preflight(`ref_isolation=... / out=log/ref_distill_*`)与日志末尾
   (4000/4000 + 总耗时,见 §3);
3. 四个最终 checkpoint 的字节级校验输出(`check_lora_ckpt.py`,均 `304 张量/空 0/全零 0`);
4. 两腿 stage-1 各 100 个 checkpoint 落盘(`ls | grep -c checkpoint` = 100);
5. 保活状态(8 进程 + 8 卡显存满载,见 §4)。

---

## 7. 下一步

P2/P3 完成,进 **P4**:扩任务池 192→320 → 两腿生图 → `build_pairs.py m6` → 盲评
(~25 min,320 主对 + 30 in-batch run_floor)→ 报告。判据与样本量见
`M6_ABLATION_SPEC.md` §5(预登记,结果出来前不许改)。
