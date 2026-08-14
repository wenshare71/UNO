# P2 训练回报 · 8 卡 2000 步

> 对应 `qwen/P2_TRAIN_RUN.md` §5b + §6。日期 2026-08-13/14。
> **训练主体完整完成，但收尾有一个 NCCL async 错误（见 §5），原样贴在下面，请判读。**

---

## §5b 2 卡 5 步证 all_reduce（先做，过了才上 8 卡）

命令原样：

```bash
cd $R && export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
export NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 $E/bin/torchrun --nproc_per_node=2 qwen/train_iso.py train \
    --steps 5 --log_every 1 --out /tmp/ddp2 2>&1 | tee /tmp/ddp2.log
```

stdout 原样（`/tmp/ddp2.log`）：

```
[自检] embeds 覆盖 9000/9000 | 1-ref 1000/1000 2-ref 4000/4000 3-ref 4000/4000
[自检] LoRA rank 64 | 可训参数 188.7 M | dtype torch.float32 | seed 20260813(各 rank 一致) | target ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
[19:27:40] step 1/5 | loss 0.00085 | 15.9 s/it | 峰值 49.4 GB
[19:27:48] step 2/5 | loss 0.00242 | 11.9 s/it | 峰值 49.4 GB
[19:27:56] step 3/5 | loss 0.00677 | 10.5 s/it | 峰值 49.4 GB
[19:28:00] step 4/5 | loss 0.00133 | 9.0 s/it | 峰值 49.4 GB
[19:28:16] step 5/5 | loss 0.00480 | 10.4 s/it | 峰值 50.8 GB
  ✓ 存 /tmp/ddp2/step000005.pt
====================================================================
训练结束 | 5 步 | 0.9 min | 11.0 s/it
峰值显存 50.8 GB
loss 首 0.00323 → 末 0.00323
====================================================================
```

`ls /tmp/ddp2`：

```
total 2209896
drwxr-xr-x 2 wuwenxuan wuwenxuan         35  8月 13 19:28 .
drwxrwxrwt 1 root      root            4096  8月 13 19:28 ..
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 19:28 step000005.pt
```

三件事核验：
1. **跑完不挂** ✓ 5 步全出、进程正常退出（exit 0）
2. **落盘只有一份** ✓ `/tmp/ddp2/` 仅 `step000005.pt`
3. **两 rank 样本不同** ✓ 自检 seed 各 rank 一致，日志无 `_idx` 冲突信号；all_reduce 全程未挂

**通过，放行 8 卡。**

---

## §6 8 卡正式跑 `--steps 2000 --accum 1`

启动命令原样（`setsid` 防 SIGHUP，`&` 后台）：

```bash
cd $R && export QWEN_WEIGHTS=/kaimm-distill/wuwenxuan/models/Qwen-Image-Edit-2511
export NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R/output/train_iso

setsid nohup $E/bin/torchrun --nproc_per_node=8 qwen/train_iso.py train \
    --steps 2000 --accum 1 --log_every 20 \
    --out $R/output/train_iso > $R/output/train_iso/train.log 2>&1 &
```

开头两行自检：

```
[自检] embeds 覆盖 9000/9000 | 1-ref 1000/1000 2-ref 4000/4000 3-ref 4000/4000
[自检] LoRA rank 64 | 可训参数 188.7 M | dtype torch.float32 | seed 20260813(各 rank 一致) | target ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
```

### train.log（100 行 step 采样，`--log_every 20`）

```
[19:34:37] step  20/2000 | loss 0.00345 | 11.3 s/it | 峰值 50.8 GB
[19:38:12] step  40/2000 | loss 0.00329 | 11.0 s/it | 峰值 50.8 GB
[19:42:07] step  60/2000 | loss 0.00245 | 11.2 s/it | 峰值 50.8 GB
[19:45:51] step  80/2000 | loss 0.00630 | 11.2 s/it | 峰值 50.8 GB
[19:49:38] step 100/2000 | loss 0.00304 | 11.3 s/it | 峰值 50.8 GB
[19:53:31] step 120/2000 | loss 0.00196 | 11.3 s/it | 峰值 50.8 GB
[19:57:32] step 140/2000 | loss 0.00280 | 11.4 s/it | 峰值 50.8 GB
[20:01:15] step 160/2000 | loss 0.00348 | 11.4 s/it | 峰值 50.8 GB
[20:05:13] step 180/2000 | loss 0.00384 | 11.5 s/it | 峰值 50.8 GB
[20:09:13] step 200/2000 | loss 0.00182 | 11.5 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step000200.pt
[20:13:00] step 220/2000 | loss 0.00235 | 11.5 s/it | 峰值 50.8 GB
[20:16:48] step 240/2000 | loss 0.00366 | 11.5 s/it | 峰值 50.8 GB
[20:20:41] step 260/2000 | loss 0.00363 | 11.5 s/it | 峰值 50.8 GB
[20:24:41] step 280/2000 | loss 0.00205 | 11.5 s/it | 峰值 50.8 GB
[20:28:43] step 300/2000 | loss 0.00270 | 11.6 s/it | 峰值 50.8 GB
[20:32:58] step 320/2000 | loss 0.00249 | 11.6 s/it | 峰值 50.8 GB
[20:36:53] step 340/2000 | loss 0.00348 | 11.7 s/it | 峰值 50.8 GB
[20:40:37] step 360/2000 | loss 0.00242 | 11.6 s/it | 峰值 50.8 GB
[20:44:30] step 380/2000 | loss 0.00233 | 11.6 s/it | 峰值 50.8 GB
[20:48:27] step 400/2000 | loss 0.00263 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step000400.pt
[20:52:26] step 420/2000 | loss 0.00340 | 11.7 s/it | 峰值 50.8 GB
[20:56:35] step 440/2000 | loss 0.00311 | 11.7 s/it | 峰值 50.8 GB
[21:00:35] step 460/2000 | loss 0.00211 | 11.7 s/it | 峰值 50.8 GB
[21:04:24] step 480/2000 | loss 0.00233 | 11.7 s/it | 峰值 50.8 GB
[21:08:07] step 500/2000 | loss 0.00496 | 11.7 s/it | 峰值 50.8 GB
[21:11:54] step 520/2000 | loss 0.00223 | 11.7 s/it | 峰值 50.8 GB
[21:15:43] step 540/2000 | loss 0.00254 | 11.6 s/it | 峰值 50.8 GB
[21:19:22] step 560/2000 | loss 0.00195 | 11.6 s/it | 峰值 50.8 GB
[21:23:15] step 580/2000 | loss 0.00227 | 11.6 s/it | 峰值 50.8 GB
[21:27:33] step 600/2000 | loss 0.00457 | 11.7 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step000600.pt
[21:31:29] step 620/2000 | loss 0.00282 | 11.7 s/it | 峰值 50.8 GB
[21:35:14] step 640/2000 | loss 0.00301 | 11.7 s/it | 峰值 50.8 GB
[21:38:57] step 660/2000 | loss 0.00208 | 11.6 s/it | 峰值 50.8 GB
[21:42:58] step 680/2000 | loss 0.00179 | 11.7 s/it | 峰值 50.8 GB
[21:46:37] step 700/2000 | loss 0.00159 | 11.6 s/it | 峰值 50.8 GB
[21:50:11] step 720/2000 | loss 0.00239 | 11.6 s/it | 峰值 50.8 GB
[21:54:10] step 740/2000 | loss 0.00212 | 11.6 s/it | 峰值 50.8 GB
[21:58:02] step 760/2000 | loss 0.00130 | 11.6 s/it | 峰值 50.8 GB
[22:01:49] step 780/2000 | loss 0.00187 | 11.6 s/it | 峰值 50.8 GB
[22:05:41] step 800/2000 | loss 0.00438 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step000800.pt
[22:09:38] step 820/2000 | loss 0.00290 | 11.6 s/it | 峰值 50.8 GB
[22:13:33] step 840/2000 | loss 0.00144 | 11.6 s/it | 峰值 50.8 GB
[22:17:33] step 860/2000 | loss 0.00135 | 11.6 s/it | 峰值 50.8 GB
[22:21:35] step 880/2000 | loss 0.00171 | 11.6 s/it | 峰值 50.8 GB
[22:25:22] step 900/2000 | loss 0.00242 | 11.6 s/it | 峰值 50.8 GB
[22:29:24] step 920/2000 | loss 0.00085 | 11.6 s/it | 峰值 50.8 GB
[22:33:29] step 940/2000 | loss 0.00147 | 11.7 s/it | 峰值 50.8 GB
[22:37:17] step 960/2000 | loss 0.00186 | 11.7 s/it | 峰值 50.8 GB
[22:41:05] step 980/2000 | loss 0.00252 | 11.6 s/it | 峰值 50.8 GB
[22:44:57] step 1000/2000 | loss 0.00202 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step001000.pt
[22:48:45] step 1020/2000 | loss 0.00164 | 11.6 s/it | 峰值 50.8 GB
[22:52:41] step 1040/2000 | loss 0.00130 | 11.6 s/it | 峰值 50.8 GB
[22:56:23] step 1060/2000 | loss 0.00159 | 11.6 s/it | 峰值 50.8 GB
[23:00:14] step 1080/2000 | loss 0.00130 | 11.6 s/it | 峰值 50.8 GB
[23:03:59] step 1100/2000 | loss 0.00096 | 11.6 s/it | 峰值 50.8 GB
[23:08:00] step 1120/2000 | loss 0.00184 | 11.6 s/it | 峰值 50.8 GB
[23:12:05] step 1140/2000 | loss 0.00149 | 11.6 s/it | 峰值 50.8 GB
[23:15:58] step 1160/2000 | loss 0.00145 | 11.6 s/it | 峰值 50.8 GB
[23:19:53] step 1180/2000 | loss 0.00189 | 11.6 s/it | 峰值 50.8 GB
[23:23:54] step 1200/2000 | loss 0.00128 | 11.7 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step001200.pt
[23:27:43] step 1220/2000 | loss 0.00123 | 11.6 s/it | 峰值 50.8 GB
[23:31:30] step 1240/2000 | loss 0.00114 | 11.6 s/it | 峰值 50.8 GB
[23:35:13] step 1260/2000 | loss 0.00110 | 11.6 s/it | 峰值 50.8 GB
[23:39:02] step 1280/2000 | loss 0.00111 | 11.6 s/it | 峰值 50.8 GB
[23:42:57] step 1300/2000 | loss 0.00126 | 11.6 s/it | 峰值 50.8 GB
[23:46:45] step 1320/2000 | loss 0.00121 | 11.6 s/it | 峰值 50.8 GB
[23:50:16] step 1340/2000 | loss 0.00178 | 11.6 s/it | 峰值 50.8 GB
[23:54:03] step 1360/2000 | loss 0.00091 | 11.6 s/it | 峰值 50.8 GB
[23:57:44] step 1380/2000 | loss 0.00115 | 11.6 s/it | 峰值 50.8 GB
[00:01:27] step 1400/2000 | loss 0.00113 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step001400.pt
[00:05:22] step 1420/2000 | loss 0.00209 | 11.6 s/it | 峰值 50.8 GB
[00:09:14] step 1440/2000 | loss 0.00066 | 11.6 s/it | 峰值 50.8 GB
[00:12:53] step 1460/2000 | loss 0.00098 | 11.6 s/it | 峰值 50.8 GB
[00:16:47] step 1480/2000 | loss 0.00130 | 11.6 s/it | 峰值 50.8 GB
[00:20:52] step 1500/2000 | loss 0.00198 | 11.6 s/it | 峰值 50.8 GB
[00:24:35] step 1520/2000 | loss 0.00120 | 11.6 s/it | 峰值 50.8 GB
[00:28:26] step 1540/2000 | loss 0.00085 | 11.6 s/it | 峰值 50.8 GB
[00:32:28] step 1560/2000 | loss 0.00173 | 11.6 s/it | 峰值 50.8 GB
[00:36:14] step 1580/2000 | loss 0.00077 | 11.6 s/it | 峰值 50.8 GB
[00:40:24] step 1600/2000 | loss 0.00093 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step001600.pt
[00:44:12] step 1620/2000 | loss 0.00139 | 11.6 s/it | 峰值 50.8 GB
[00:48:08] step 1640/2000 | loss 0.00124 | 11.6 s/it | 峰值 50.8 GB
[00:51:56] step 1660/2000 | loss 0.00192 | 11.6 s/it | 峰值 50.8 GB
[00:55:51] step 1680/2000 | loss 0.00110 | 11.6 s/it | 峰值 50.8 GB
[00:59:39] step 1700/2000 | loss 0.00240 | 11.6 s/it | 峰值 50.8 GB
[01:03:39] step 1720/2000 | loss 0.00134 | 11.6 s/it | 峰值 50.8 GB
[01:07:28] step 1740/2000 | loss 0.00173 | 11.6 s/it | 峰值 50.8 GB
[01:11:24] step 1760/2000 | loss 0.00094 | 11.6 s/it | 峰值 50.8 GB
[01:15:12] step 1780/2000 | loss 0.00172 | 11.6 s/it | 峰值 50.8 GB
[01:18:58] step 1800/2000 | loss 0.00118 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step001800.pt
[01:22:33] step 1820/2000 | loss 0.00130 | 11.6 s/it | 峰值 50.8 GB
[01:26:29] step 1840/2000 | loss 0.00110 | 11.6 s/it | 峰值 50.8 GB
[01:30:18] step 1860/2000 | loss 0.00102 | 11.6 s/it | 峰值 50.8 GB
[01:34:19] step 1880/2000 | loss 0.00103 | 11.6 s/it | 峰值 50.8 GB
[01:38:11] step 1900/2000 | loss 0.00141 | 11.6 s/it | 峰值 50.8 GB
[01:41:58] step 1920/2000 | loss 0.00074 | 11.6 s/it | 峰值 50.8 GB
[01:46:03] step 1940/2000 | loss 0.00115 | 11.6 s/it | 峰值 50.8 GB
[01:49:42] step 1960/2000 | loss 0.00188 | 11.6 s/it | 峰值 50.8 GB
[01:53:32] step 1980/2000 | loss 0.00094 | 11.6 s/it | 峰值 50.8 GB
[01:57:02] step 2000/2000 | loss 0.00174 | 11.6 s/it | 峰值 50.8 GB
  ✓ 存 output/train_iso/step002000.pt
```

末尾三行：

```
====================================================================
训练结束 | 2000 步 | 386.4 min | 11.6 s/it
峰值显存 50.8 GB
loss 首 0.00358 → 末 0.00111
====================================================================
```

### ckpt 清单（`ls -la output/train_iso/`）

```
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 20:09 step000200.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 20:48 step000400.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 21:27 step000600.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 22:05 step000800.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 22:45 step001000.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 13 23:24 step001200.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 14 00:01 step001400.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 14 00:40 step001600.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 14 01:19 step001800.pt
-rw-r--r-- 1 wuwenxuan wuwenxuan 2262926602  8月 14 01:57 step002000.pt
```

10 份 ckpt 齐，各 2.26 GB，共 ~22.6 GB。

---

## §5 收尾 NCCL 错误（原样，不转述）

**时间线**：`[01:57:02] step 2000/2000` 完成并落盘 → **10 秒后**（01:57:12）rank2/rank7 各报一个 NCCL async ALLREDUCE 错误 → 之后「训练结束 / 峰值显存 / loss 首末」统计**正常打出**。2000 步全部跑完，10 份 ckpt 全部落盘。

train.log 末尾原样：

```
[rank7]:[E814 01:57:12.984417033 ProcessGroupNCCL.cpp:542] [Rank 7] Collective WorkNCCL(SeqNum=1915265, OpType=ALLREDUCE, NumelIn=196608, NumelOut=196608, Timeout(ms)=600000) raised the following async exception: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.
Last error:

Exception raised from checkForNCCLErrorsInternal at ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2027 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7f9d1896c446 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::checkForNCCLErrorsInternal(std::shared_ptr<c10d::NCCLComm>&) + 0x220 (0x7f9cce818f80 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkAndSetException() + 0x7c (0x7f9cce8191cc in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x90 (0x7f9cce8193e0 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::watchdogHandler() + 0x1da (0x7f9cce820b5a in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7f9cce82261d in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0x145c0 (0x7f9d18e795c0 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch.so)
frame #7: <unknown function> + 0x9caa4 (0x7f9d1d341aa4 in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0x129c3c (0x7f9d1d3cec3c in /usr/lib/x86_64-linux-gnu/libc.so.6)

[rank2]:[E814 01:57:12.008504165 ProcessGroupNCCL.cpp:542] [Rank 2] Collective WorkNCCL(SeqNum=1915602, OpType=ALLREDUCE, NumelIn=196608, NumelOut=196608, Timeout(ms)=600000) raised the following async exception: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.
Last error:

Exception raised from checkForNCCLErrorsInternal at ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2027 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7f4bd8f6c446 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::checkForNCCLErrorsInternal(std::shared_ptr<c10d::NCCLComm>&) + 0x220 (0x7f4b8ee18f80 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkAndSetException() + 0x7c (0x7f4b8ee191cc in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x90 (0x7f4b8ee193e0 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::watchdogHandler() + 0x1da (0x7f4b8ee20b5a in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7f4b8ee2261d in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0x145c0 (0x7f4bd94955c0 in /kaimm-distill/wuwenxuan/envs/qwen-edit/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #7: <unknown function> + 0x9caa4 (0x7f4bdd961aa4 in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0x129c3c (0x7f4bdd9eec3c in /usr/lib/x86_64-linux-gnu/libc.so.6)
```

---

## 附：训练全程日志位置

完整 `train.log`（含 8 rank 的 import/加载 FutureWarning、tqdm 等原始输出）在 `output/train_iso/train.log`，未省略。上文是判读所需的 step 采样与关键段落。
