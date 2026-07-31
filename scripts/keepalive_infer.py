#!/usr/bin/env python
"""H800 保活推理脚本:模型常驻显存,周期性跑真实推理,防止实例因空闲被回收。

设计要点:
  1. 推理任务(非空转):每轮一次 512x512 / 25 步 UNO 单 ref 生成,配置与
     smoke_eval.py 已验证路径完全一致(flux-dev bf16 / no offload /
     only_lora / pe='d' / guidance 4.0);
  2. 单卡运行:由外部 CUDA_VISIBLE_DEVICES 指定,脚本只占 1 张卡(共享机器,遵守 R12);
  3. 磁盘不膨胀:每轮覆盖 output/keepalive/latest.png,日志单行追加;
  4. 健壮:单轮异常不退出,连续失败 MAX_CONSECUTIVE_FAILS 次才退出(避免无限报错空转);
  5. 心跳:每轮打印时间戳 / 耗时 / seed / 峰值显存,tmux 里一眼确认存活。

用法(在 H800、UNO 仓库根目录、激活 .venv-uno 后):
  export CUDA_VISIBLE_DEVICES=3        # 先 nvidia-smi 挑空闲卡
  python scripts/keepalive_infer.py

可调环境变量(都有默认值,正常保活无需手动设):
  KEEPALIVE_INTERVAL     每轮间隔秒数(默认 60,不含推理耗时)
  KEEPALIVE_NUM_STEPS     采样步数(默认 25,想更省可降到 10)
  KEEPALIVE_MAX_ROUNDS    跑完 N 轮后退出(默认 0 = 无限循环);首次部署时可设 1 做冒烟自检
  KEEPALIVE_SAVE_DIR      输出目录(默认 output/keepalive);若该目录已被 root 占用,
                         在非 root 下跑会因无写权限而 fail 10 次后退出,此时用本变量改到
                         可写位置(例如 /tmp/keepalive 或 $HOME/keepalive)。
  HF_HOME                 共享 HF 缓存(默认 /kaimm-distill/wuwenxuan/hf_cache,两机都挂载)
  HF_HUB_OFFLINE          默认 1,禁联网探测(否则走日本代理卡死,正是上一次 H800 跑到
                         "启动" 就没下文的原因)
  PYTORCH_CUDA_ALLOC_CONF 默认 expandable_segments:True,显存碎片化时也能捡回显存
"""

import os
import sys
import time
import traceback
from datetime import datetime, timezone

# python scripts/xxx.py 时 sys.path 只有 scripts/,把仓库根目录加进来才能 import uno
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 环境自洽:这些必须在 import torch / huggingface_hub 之前 setdefault ---
# 上一次 H800 跑到 "启动" 就没下文,根因是没设 HF_HUB_OFFLINE,hf_hub_download 走日本
# 代理 hang 住。这里把推荐值都设成默认,用户已 export 则尊重用户值(setdefault 不覆盖)。
# HF_HOME 默认指 kaimm-distill 共享缓存(本机与 H800 都挂载);H800 若已拷到本地 NVMe,
# 用户在 shell 里 export HF_HOME=/code/uno/hf_cache 即可覆盖。
os.environ.setdefault("HF_HOME", "/kaimm-distill/wuwenxuan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# torch 在 import 时读这个变量,必须更早 setdefault。4090 24GB 紧 / H800 143GB 余,都开无害。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- 参数(环境变量可调,默认值与 smoke_eval.py 已验证配置一致) ---
INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "60"))
NUM_STEPS = int(os.environ.get("KEEPALIVE_NUM_STEPS", "25"))
MAX_ROUNDS = int(os.environ.get("KEEPALIVE_MAX_ROUNDS", "0"))  # 0 = 无限循环
WIDTH = HEIGHT = 512
GUIDANCE = 4.0
REF_SIZE = 512
BASE_SEED = 3407
MAX_CONSECUTIVE_FAILS = 10
SAVE_DIR = os.environ.get("KEEPALIVE_SAVE_DIR", "output/keepalive")
LOG_FILE = os.path.join(SAVE_DIR, "keepalive.log")

# ref 图全部取自 assets/(随仓库分发,H800 上必然存在),prompt 错开场景
CASES = [
    {"prompt": "a clock on the beach",
     "image_paths": ["assets/clock.png"]},
    {"prompt": "a cup in the jungle",
     "image_paths": ["assets/cup.png"]},
    {"prompt": "a figurine in the snow",
     "image_paths": ["assets/figurine.png"]},
    {"prompt": "a crystal ball on a wooden table",
     "image_paths": ["assets/crystal_ball.png"]},
    {"prompt": "a cat in a cozy cafe",
     "image_paths": ["assets/cat_cafe.png"]},
]


def log(msg: str) -> None:
    """同时写 stdout(tmux 可见)和日志文件(tmux 丢失后可查)。"""
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass  # 日志写不进去不该弄死保活进程


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 启动自检:SAVE_DIR 必须可写 ---
    # output/keepalive 可能被前一次 sudo 跑留下的 root:root 0600 占住,非 root 用户
    # 写不进去;不做这个检查的话要连续 fail MAX_CONSECUTIVE_FAILS 轮才退出,浪费时间。
    try:
        _probe = os.path.join(SAVE_DIR, ".write_probe")
        with open(_probe, "w") as _f:
            _f.write("ok")
        os.remove(_probe)
    except OSError as _e:
        sys.exit(
            f"❌ 输出目录不可写: {SAVE_DIR} ({_e})。"
            f"用 KEEPALIVE_SAVE_DIR=... 改到可写位置,或清理已有 root 所有权目录。"
        )

    # --- 启动自检:ref 图必须都在 ---
    for c in CASES:
        for p in c["image_paths"]:
            if not os.path.isfile(p):
                sys.exit(f"❌ ref 图不存在: {p}(应在 UNO 仓库根目录运行)")

    # --- 启动自检:HF_HOME 下四个必需模型仓库都得在(否则 hf_hub_download 会去联网) ---
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        sys.exit("❌ HF_HOME 未设置且无默认值,无法定位模型缓存")
    hub_dir = os.path.join(hf_home, "hub")
    required_repos = [
        "models--black-forest-labs--FLUX.1-dev",        # flux1-dev.safetensors + ae.safetensors
        "models--bytedance-research--UNO",               # dit_lora.safetensors
        "models--openai--clip-vit-large-patch14",         # CLIP
        "models--xlabs-ai--xflux_text_encoders",          # T5
    ]
    missing = [r for r in required_repos if not os.path.isdir(os.path.join(hub_dir, r))]
    if missing:
        sys.exit(
            f"❌ HF_HOME={hf_home}/hub 缺模型仓库: {missing}。"
            f"指错缓存了;正确的 hf_cache 应有这 4 个 models--* 目录。"
        )

    import torch
    from PIL import Image
    from uno.flux.pipeline import UNOPipeline, preprocess_ref

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA 不可用,检查 CUDA_VISIBLE_DEVICES 是否设置")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "<未设,默认用卡 0>")
    log(f"启动 | GPU: {gpu_name} | CUDA_VISIBLE_DEVICES={gpu_idx} | 间隔 {INTERVAL}s | "
        f"{WIDTH}x{HEIGHT} x {NUM_STEPS} 步 | 输出: {SAVE_DIR}/latest.png"
        + (f" | 最多 {MAX_ROUNDS} 轮" if MAX_ROUNDS > 0 else " | 无限循环"))

    # --- 模型只加载一次,常驻显存(显存占用本身就是保活信号) ---
    t0 = time.perf_counter()
    pipeline = UNOPipeline("flux-dev", torch.device("cuda"), offload=False,
                           only_lora=True, lora_rank=512)
    log(f"模型加载完成,耗时 {time.perf_counter() - t0:.1f}s,"
        f"显存 {torch.cuda.memory_allocated() / 1024**3:.1f}GB")

    # ref 图预处理也只做一次
    ref_bank = [
        [preprocess_ref(Image.open(p), REF_SIZE) for p in c["image_paths"]]
        for c in CASES
    ]

    round_idx = 0
    consec_fails = 0
    while True:
        case_i = round_idx % len(CASES)
        seed = BASE_SEED + round_idx
        try:
            t0 = time.perf_counter()
            img = pipeline(
                prompt=CASES[case_i]["prompt"],
                width=WIDTH, height=HEIGHT,
                guidance=GUIDANCE, num_steps=NUM_STEPS,
                seed=seed, ref_imgs=ref_bank[case_i], pe="d",
            )
            dt = time.perf_counter() - t0
            img.save(os.path.join(SAVE_DIR, "latest.png"))
            peak = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            log(f"round {round_idx:>6} | OK | {dt:5.1f}s | seed {seed} | "
                f"case{case_i} | peak {peak:.1f}GB")
            consec_fails = 0
        except KeyboardInterrupt:
            log("收到 Ctrl-C,保活退出")
            return
        except Exception:
            consec_fails += 1
            log(f"round {round_idx:>6} | FAIL ({consec_fails}/{MAX_CONSECUTIVE_FAILS})\n"
                + traceback.format_exc())
            if consec_fails >= MAX_CONSECUTIVE_FAILS:
                log(f"连续失败 {MAX_CONSECUTIVE_FAILS} 次,环境大概率已坏,退出避免空转")
                sys.exit(1)
        round_idx += 1
        if MAX_ROUNDS > 0 and round_idx >= MAX_ROUNDS:
            log(f"已跑完 {MAX_ROUNDS} 轮,正常退出")
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
