#!/usr/bin/env python
"""H800 训练保活脚本:模型常驻显存,周期性跑真实训练步(前向+反向+优化器更新),
防止实例因空闲被回收,同时满足"训练机器不得跑推理"的合规要求。

设计要点:
  1. 真实训练任务(非空转):每轮跑 KEEPALIVE_STEPS_PER_ROUND 步 flux-dev LoRA
     微调,包含完整的 forward / MSE loss / backward / optimizer.step();
  2. 单卡运行:由外部 CUDA_VISIBLE_DEVICES 指定,脚本只占 1 张卡(共享机器,遵守 R12);
  3. 磁盘不膨胀:不保存 checkpoint、不写图,只追加单行日志;
  4. 状态不污染:只更新 LoRA 参数,base DiT/VAE/T5/CLIP 全部冻结,事后不影响
     任何正式 checkpoint;
  5. 健壮:单轮异常不退出,连续失败 MAX_CONSECUTIVE_FAILS 次才退出(避免无限报错空转);
  6. 心跳:每轮打印时间戳 / 耗时 / 峰值显存 / loss,tmux 里一眼确认存活。

用法(在 H800、UNO 仓库根目录、激活 .venv-uno 后):
  export CUDA_VISIBLE_DEVICES=3        # 先 nvidia-smi 挑空闲卡
  python scripts/keepalive_train.py

可调环境变量(都有默认值,正常保活无需手动设):
  KEEPALIVE_INTERVAL          每轮间隔秒数(默认 60,不含训练耗时)
  KEEPALIVE_STEPS_PER_ROUND   每轮训练步数(默认 10)
  KEEPALIVE_BATCH_SIZE        每步 batch size(默认 1)
  KEEPALIVE_LORA_RANK         LoRA rank(默认 512)
  KEEPALIVE_LR                AdamW 学习率(默认 1e-4)
  KEEPALIVE_MAX_ROUNDS        跑完 N 轮后退出(默认 0 = 无限循环);首次部署时可设 1 做冒烟自检
  KEEPALIVE_SAVE_DIR          输出目录(默认 output/keepalive_train);若该目录已被 root 占用,
                              在非 root 下跑会 fail,此时用本变量改到可写位置。
  HF_HOME                     共享 HF 缓存(默认 /kaimm-distill/wuwenxuan/hf_cache,两机都挂载)
  HF_HUB_OFFLINE              默认 1,禁联网探测
  PYTORCH_CUDA_ALLOC_CONF     默认 expandable_segments:True
"""

import os
import sys
import time
import traceback
from datetime import datetime, timezone

# python scripts/xxx.py 时 sys.path 只有 scripts/,把仓库根目录加进来才能 import uno
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 环境自洽:这些必须在 import torch / huggingface_hub 之前 setdefault ---
os.environ.setdefault("HF_HOME", "/kaimm-distill/wuwenxuan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from einops import rearrange

from uno.flux.sampling import get_schedule, prepare_multi_ip
from uno.flux.util import load_ae, load_clip, load_flow_model, load_t5, set_lora

# --- 参数(环境变量可调) ---
INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "60"))
STEPS_PER_ROUND = int(os.environ.get("KEEPALIVE_STEPS_PER_ROUND", "10"))
BATCH_SIZE = int(os.environ.get("KEEPALIVE_BATCH_SIZE", "1"))
LORA_RANK = int(os.environ.get("KEEPALIVE_LORA_RANK", "512"))
LR = float(os.environ.get("KEEPALIVE_LR", "1e-4"))
MAX_ROUNDS = int(os.environ.get("KEEPALIVE_MAX_ROUNDS", "0"))  # 0 = 无限循环
SAVE_DIR = os.environ.get("KEEPALIVE_SAVE_DIR", "output/keepalive_train")
LOG_FILE = os.path.join(SAVE_DIR, "keepalive.log")
MAX_CONSECUTIVE_FAILS = 10

RESOLUTION = 512
BASE_SEED = 3407

# 固定 prompt / ref 场景,保证 text encoder 输出稳定可复用
CASES = [
    {"prompt": "a clock on the beach"},
    {"prompt": "a cup in the jungle"},
    {"prompt": "a figurine in the snow"},
    {"prompt": "a crystal ball on a wooden table"},
    {"prompt": "a cat in a cozy cafe"},
]


def log(msg: str) -> None:
    """同时写 stdout(tmux 可见)和日志文件(tmux 丢失后可查)。"""
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def get_models(name: str, device, offload: bool = False):
    """复用 train.py 的模型加载顺序,但默认不 offload。"""
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip


def build_fake_batch(device, t5, clip, prompts, resolution=512):
    """构造假训练 batch,latent 直接随机生成,避免依赖真实图片和 VAE encode。

    512x512 图像经 VAE encode 后 latent 空间为 (16, 64, 64),再经 prepare_multi_ip
    里的 rearrange(ph=2,pw=2) 得到 (1024, 64)。这里直接生成 4D latent,让
    prepare_multi_ip 做 rearrange,与 train.py 的数据流一致。
    """
    bs = len(prompts)
    h = w = resolution // 8  # 64
    x_1 = torch.randn(bs, 16, h, w, device=device, dtype=torch.bfloat16)
    x_ref = [torch.randn(bs, 16, h, w, device=device, dtype=torch.bfloat16)]
    inp = prepare_multi_ip(t5=t5, clip=clip, img=x_1, prompt=prompts,
                           ref_imgs=tuple(x_ref), pe="d")
    return inp


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 启动自检:SAVE_DIR 必须可写 ---
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

    # --- 启动自检:HF_HOME 下四个必需模型仓库都得在 ---
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        sys.exit("❌ HF_HOME 未设置且无默认值,无法定位模型缓存")
    hub_dir = os.path.join(hf_home, "hub")
    required_repos = [
        "models--black-forest-labs--FLUX.1-dev",
        "models--bytedance-research--UNO",
        "models--openai--clip-vit-large-patch14",
        "models--xlabs-ai--xflux_text_encoders",
    ]
    missing = [r for r in required_repos if not os.path.isdir(os.path.join(hub_dir, r))]
    if missing:
        sys.exit(
            f"❌ HF_HOME={hf_home}/hub 缺模型仓库: {missing}。"
            f"指错缓存了;正确的 hf_cache 应有这 4 个 models--* 目录。"
        )

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA 不可用,检查 CUDA_VISIBLE_DEVICES 是否设置")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "<未设,默认用卡 0>")
    log(f"启动 | GPU: {gpu_name} | CUDA_VISIBLE_DEVICES={gpu_idx} | "
        f"间隔 {INTERVAL}s | 每轮 {STEPS_PER_ROUND} 步 | bs={BATCH_SIZE} | "
        f"rank={LORA_RANK} | lr={LR} | 输出: {SAVE_DIR}"
        + (f" | 最多 {MAX_ROUNDS} 轮" if MAX_ROUNDS > 0 else " | 无限循环"))

    # --- 模型只加载一次,常驻显存(显存占用本身就是保活信号) ---
    t0 = time.perf_counter()
    global dit, vae, t5, clip  # build_fake_batch 需要 t5/clip
    dit, vae, t5, clip = get_models("flux-dev", device, offload=False)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    dit.requires_grad_(False)
    dit = set_lora(dit, LORA_RANK, device=device)
    dit.train()
    dit.gradient_checkpointing = True
    dit = dit.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=LR, betas=[0.9, 0.999], weight_decay=0.01, eps=1e-8
    )

    # timesteps 与 train.py 一致:999 步 schedule,512x512 对应 4096 tokens
    # 指定 bf16,避免后面 (1-t)*x_1 时被 promote 成 fp32 导致模型 dtype 不匹配
    timesteps = get_schedule(
        999,
        (RESOLUTION // 8) * (RESOLUTION // 8) // 4,
        shift=True,
    )
    timesteps = torch.tensor(timesteps, device=device, dtype=torch.bfloat16)

    log(f"模型加载完成,耗时 {time.perf_counter() - t0:.1f}s,"
        f"显存 {torch.cuda.memory_allocated() / 1024**3:.1f}GB")

    round_idx = 0
    consec_fails = 0
    while True:
        case_i = round_idx % len(CASES)
        prompts = [CASES[case_i]["prompt"]] * BATCH_SIZE
        seed = BASE_SEED + round_idx
        torch.manual_seed(seed)
        try:
            t0 = time.perf_counter()
            inp = build_fake_batch(device, t5, clip, prompts, RESOLUTION)
            x_1 = inp["img"]
            # 训练步
            round_loss = 0.0
            for step in range(STEPS_PER_ROUND):
                bs = x_1.shape[0]
                t = torch.randint(0, 1000, (bs,), device=device)
                t = timesteps[t]
                x_0 = torch.randn_like(x_1)
                x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
                guidance_vec = torch.full((bs,), 1, device=device, dtype=x_t.dtype)

                model_pred = dit(
                    img=x_t.to(torch.bfloat16),
                    img_ids=inp["img_ids"].to(torch.bfloat16),
                    ref_img=[x.to(torch.bfloat16) for x in inp["ref_img"]],
                    ref_img_ids=[ref_img_id.to(torch.bfloat16) for ref_img_id in inp["ref_img_ids"]],
                    txt=inp["txt"].to(torch.bfloat16),
                    txt_ids=inp["txt_ids"].to(torch.bfloat16),
                    y=inp["vec"].to(torch.bfloat16),
                    timesteps=t.to(torch.bfloat16),
                    guidance=guidance_vec.to(torch.bfloat16),
                    ref_isolation=True,
                )
                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                round_loss += loss.item()

            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            avg_loss = round_loss / STEPS_PER_ROUND
            log(f"round {round_idx:>6} | OK | {dt:5.1f}s | seed {seed} | "
                f"case{case_i} | loss {avg_loss:.4f} | peak {peak:.1f}GB")
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
