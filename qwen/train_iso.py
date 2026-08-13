"""P2 速度匹配训练:teacher = 全注意力,student = 隔离注意力 + LoRA。

规格:`qwen/PLAN.md` §3.2。**只新建本文件,既有 `.py` 一个字不动(R0)。**

    teacher = 同一份权重, disable_adapters(), 全注意力, no_grad
    student = 同一份权重, LoRA 开,          隔离 mask

    L = ‖ v_iso(x_t, t) − v_full(x_t, t).detach() ‖²      只在噪声图 token 位置上取

**不是图像 SFT。** 目标是补回被 mask 切断的那条通路,不是教新能力,所以
manifest 里 `x₀` 好不好不构成约束(§3.2:M2 那 34.2% 的通过率在这里不适用)——
`x₀` 在这里只用来定义 `x_t` 落在哪条轨迹上,监督信号完全来自 teacher。

权重共用一份(LoRA 开关切身份),显存里不多一份 40 GB,只多一次 no_grad 前向的激活。

两个子命令:

    # 1. 单卡,离线预算 prompt_embeds(依赖 ref 图,但在去噪循环外只算一次)
    python qwen/train_iso.py precompute

    # 2. 8×H800
    torchrun --nproc_per_node=8 qwen/train_iso.py train --steps 100   # 先标定
"""
import argparse
import json
import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE, os.path.join(_REPO_ROOT, "distill")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 固定切分。**import 不改**(R0)。泄漏之后从结果里看不出来,所以启动就断言。
from gen_data import HELD_OUT, TRAIN  # noqa: E402

MANIFEST = os.path.join(_REPO_ROOT, "datasets/distill_multiref/manifest_raw.json")
DATA_DIR = os.path.join(_REPO_ROOT, "datasets/distill_multiref")
EMBED_CACHE = os.path.join(_REPO_ROOT, "cache/prompt_embeds")

# ------------------------------------------------------------------ 起手配置
# PLAN §3.2 的表。都是默认值,100 步标定后按实测调。
LORA_RANK = 64
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0",
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
RESOLUTION = 1024                       # 与评测一致(§5:不改 Q1 口径)
NUM_INFERENCE_STEPS = 40                # t 从推理用的 40 步 sigma 网格上取
REF_MIX = {1: 3, 2: 5, 3: 2}            # §3.2「照评测分布压 3-ref,但别压到 0」

# pipeline 里的常量,原样搬(pipeline L66-67)。改了就与推理对不上。
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def calculate_dimensions(target_area, ratio):
    """逐字照抄 pipeline L158-165。ref 的 token 数由它决定,不许自己写一个。"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def calculate_shift(seq_len, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
    """逐字照抄 pipeline L71-83。"""
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq_len * m + base_shift - m * base_seq


def load_manifest() -> list[dict]:
    with open(MANIFEST, "rt", encoding="utf-8") as f:
        items = json.load(f)

    # 泄漏断言:R1。宁可退出也不要带着污染的数据往下跑。
    held = set(HELD_OUT)
    leaked = {s for it in items for s in it["meta"]["subjects"] if s in held}
    if leaked:
        raise SystemExit(f"❌ held-out 主体出现在训练集里:{sorted(leaked)}\n"
                         f"   泄漏之后从评测结果里看不出来(见过的主体当然生成得好),停。")
    unknown = {s for it in items for s in it["meta"]["subjects"]} - set(TRAIN)
    if unknown:
        raise SystemExit(f"❌ 训练集里有 TRAIN 20 名单之外的主体:{sorted(unknown)}")

    for i, it in enumerate(items):
        it["_idx"] = i          # prompt_embeds 缓存按这个下标命名,抽样后还得找回来
    return items


def resolve(path: str) -> str:
    return os.path.normpath(os.path.join(DATA_DIR, path))


# ==================================================================== 预算 embeds

def cmd_precompute(args):
    """单卡跑一遍,把 9000 条的 prompt_embeds 落盘。

    为什么值得离线做:它依赖 ref 图(VL 模板把 384² 的 ref 编进 prompt),
    但**在去噪循环外只算一次**。离线之后训练时不用挂 7B VL,省约 15 GB。
    """
    from diffusers import QwenImageEditPlusPipeline

    weights = require_weights()
    items = load_manifest()
    os.makedirs(args.out, exist_ok=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    # 只要 VL + processor,transformer / vae 不上卡
    pipe.text_encoder.to("cuda")

    t0 = time.perf_counter()
    n_done = n_skip = 0
    lens = []
    for i, it in enumerate(items):
        out_path = os.path.join(args.out, f"{i:06d}.pt")
        if os.path.exists(out_path):
            n_skip += 1
            continue
        refs = [Image.open(resolve(p)).convert("RGB") for p in it["image_paths"]]
        cond = []
        for img in refs:
            w, h = img.size
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
            cond.append(pipe.image_processor.resize(img, ch, cw))
        with torch.no_grad():
            embeds, mask = pipe.encode_prompt(image=cond, prompt=it["prompt"], device="cuda")
        # batch=1 时 mask 全 1 会被 encode_prompt 置 None(pipeline L327-328),
        # 所以只存 embeds;训练侧同样按无 padding 处理。
        assert mask is None, f"第 {i} 条出现了 padding mask,与推理路径不一致,停"
        torch.save(embeds[0].to(torch.bfloat16).cpu(), out_path)
        lens.append(embeds.shape[1])
        n_done += 1

        if n_done % 200 == 0:
            el = time.perf_counter() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(items)} | "
                  f"{el / n_done:.2f} s/条 | 剩 {(len(items) - i - 1) * el / n_done / 60:.0f} min",
                  flush=True)

    print(f"\n完成 {n_done} 条,跳过 {n_skip} 条 → {args.out}")
    if lens:
        # §4.4 的 token 账把 txt 估成 400–600,这里是实测值,标定报告要带上
        print(f"txt token 实测:min {min(lens)} / 中位 {int(np.median(lens))} / max {max(lens)}")


# ==================================================================== 训练

def require_weights() -> str:
    w = os.environ.get("QWEN_WEIGHTS")
    if not w:
        raise SystemExit("❌ 环境变量 QWEN_WEIGHTS 未设置(无默认值,防止误读到别的权重)。")
    return w


class Batcher:
    """按 REF_MIX 的比例抽样。batch 恒为 1 —— pipeline 本身只支持 batch_size=1,
    而且不同 n_refs 的序列长度不同,凑 batch 要 padding,不值得。
    有效 batch 靠 8 卡 × 梯度累积。
    """

    def __init__(self, items: list[dict], seed: int, rank: int, world: int):
        self.by_n: dict[int, list[dict]] = {}
        for it in items:
            self.by_n.setdefault(it["meta"]["n_refs"], []).append(it)
        self.buckets = sorted(self.by_n)
        self.weights = [REF_MIX.get(n, 0) for n in self.buckets]
        # 每个 rank 一条独立的流,否则 8 张卡抽到同一批样本
        self.rng = random.Random(seed * 1000 + rank)
        self.world = world

    def __call__(self) -> dict:
        n = self.rng.choices(self.buckets, weights=self.weights, k=1)[0]
        pool = self.by_n[n]
        return pool[self.rng.randrange(len(pool))]


def encode_sample(pipe, item: dict, embeds_dir: str, idx: int, device):
    """返回 (x0_latents, image_latents, prompt_embeds, img_shapes)。

    全部沿用 pipeline 的预处理:ref 走 `calculate_dimensions(VAE_IMAGE_SIZE, ar)`,
    x₀ 直接 1024²。**x₀ 是 UNO 的 512² 产物,这里上采样到 1024²**——
    保评测分辨率一致,不引入第二个变量(§3.2 的两条将就之一)。
    """
    vae_images, vae_sizes = [], []
    for p in item["image_paths"]:
        img = Image.open(resolve(p)).convert("RGB")
        w, h = img.size
        vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, w / h)
        vae_sizes.append((vw, vh))
        vae_images.append(pipe.image_processor.preprocess(img, vh, vw).unsqueeze(2))

    tgt = Image.open(resolve(item["image_tgt_path"])).convert("RGB")
    tgt_t = pipe.image_processor.preprocess(tgt, RESOLUTION, RESOLUTION).unsqueeze(2)

    n_ch = pipe.transformer.config.in_channels // 4
    with torch.no_grad():
        # prepare_latents 的 image_latents 那一半:VAE encode + pack + 按 ref 拼接
        _, image_latents = pipe.prepare_latents(
            vae_images, 1, n_ch, RESOLUTION, RESOLUTION,
            torch.bfloat16, device, None, None)
        # x₀ 走同一条 VAE 路径,拿到与 latents 同形状的干净 latent
        x0 = pipe._encode_vae_image(tgt_t.to(device=device, dtype=torch.bfloat16), None)
        x0 = pipe._pack_latents(x0, 1, n_ch, x0.shape[3], x0.shape[4])

    embeds = torch.load(os.path.join(embeds_dir, f"{idx:06d}.pt"),
                        map_location=device).to(torch.bfloat16)[None]

    vsf = pipe.vae_scale_factor
    img_shapes = [[
        (1, RESOLUTION // vsf // 2, RESOLUTION // vsf // 2),
        *[(1, vh // vsf // 2, vw // vsf // 2) for vw, vh in vae_sizes],
    ]]
    return x0, image_latents, embeds, img_shapes


def build_sigma_grid(pipe, seq_len: int) -> torch.Tensor:
    """推理实际会走的那 40 个 sigma。t 从这上面取 —— 训练分布 = 部署分布(§3.2)。

    注意 `image_seq_len` 只算噪声图(pipeline L765 用 `latents.shape[1]`),
    所以 1/2/3-ref 共用同一条网格,不随 ref 数变。
    """
    from copy import deepcopy
    sched = deepcopy(pipe.scheduler)
    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        seq_len,
        sched.config.get("base_image_seq_len", 256),
        sched.config.get("max_image_seq_len", 4096),
        sched.config.get("base_shift", 0.5),
        sched.config.get("max_shift", 1.15),
    )
    sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
    return sched.sigmas[:NUM_INFERENCE_STEPS].clone()


def cmd_train(args):
    from diffusers import QwenImageEditPlusPipeline
    from peft import LoraConfig

    from pipeline_iso import IsoRunner

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    if world > 1:
        torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    weights = require_weights()
    items = load_manifest()
    if not os.path.isdir(args.embeds):
        raise SystemExit(f"❌ prompt_embeds 缓存不存在:{args.embeds}\n"
                         f"   先跑:python qwen/train_iso.py precompute")

    # 只查目录在不在不够:Batcher 是全量随机抽样,缺一条就崩在 torch.load,
    # 而且崩在第几步是随机的。启动时点清楚,顺带按桶报——manifest 按 n_refs 排序,
    # 预算跑一半的话缺口全压在某一两个桶上。
    have = {it["_idx"] for it in items
            if os.path.exists(os.path.join(args.embeds, f"{it['_idx']:06d}.pt"))}
    avail = [it for it in items if it["_idx"] in have]
    if is_main:
        by_n: dict[int, list] = {}
        for it in items:
            by_n.setdefault(it["meta"]["n_refs"], []).append(it)
        cover = " ".join(f"{n}-ref {sum(1 for it in v if it['_idx'] in have)}/{len(v)}"
                         for n, v in sorted(by_n.items()))
        print(f"[自检] embeds 覆盖 {len(avail)}/{len(items)} | {cover}", flush=True)
    if len(avail) < len(items):
        if not args.allow_partial_embeds:
            raise SystemExit(
                f"❌ embeds 只算了 {len(avail)}/{len(items)} 条。\n"
                f"   补完:python qwen/train_iso.py precompute(已存在的会跳过)\n"
                f"   只想冒烟就加 --allow_partial_embeds —— 那样训练只看得到这 {len(avail)} 条,\n"
                f"   而 manifest 按 n_refs 排序,缺口通常整桶整桶地缺,**不能拿来出正式结果**。")
        items = avail

    pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    pipe.text_encoder = None         # 离线预算过了,7B VL 不上卡(§3.2:省约 15 GB)
    pipe.vae.to(device).eval()
    transformer = pipe.transformer.to(device)
    assert transformer.zero_cond_t is True, "zero_cond_t 没生效,隔离的地基不成立,停"

    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=LORA_TARGETS))
    transformer.enable_gradient_checkpointing()
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if is_main:
        n_lora = sum(p.numel() for p in lora_params)
        print(f"[自检] LoRA rank {args.rank} | 可训参数 {n_lora / 1e6:.1f} M | "
              f"target {LORA_TARGETS}", flush=True)

    runner = IsoRunner(transformer, block_diag=args.block_diag, store=False)
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)

    noise_len = (RESOLUTION // pipe.vae_scale_factor // 2) ** 2
    sigmas = build_sigma_grid(pipe, noise_len).to(device)
    batcher = Batcher(items, args.seed, rank, world)

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        # 长任务会被打断(§3.2:上一轮臂 B 被 SIGHUP 打断两次)
        ck = torch.load(args.resume, map_location="cpu")
        set_lora_state(transformer, ck["lora"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        if is_main:
            print(f"[自检] 从 {args.resume} 续跑,step {start_step}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    losses: list[float] = []
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(args.accum):
            item = batcher()
            x0, image_latents, embeds, img_shapes = encode_sample(
                pipe, item, args.embeds, item["_idx"], device)

            # flow matching:x_t = (1-σ)·x₀ + σ·ε,σ 从推理网格上取
            si = batcher.rng.randrange(len(sigmas))
            sigma = sigmas[si]
            noise = torch.randn_like(x0)
            x_t = (1.0 - sigma) * x0 + sigma * noise
            timestep = sigma[None].to(x_t.dtype)      # pipeline 传的是 t/1000 = sigma

            with torch.no_grad():
                transformer.disable_adapters()
                v_full = runner.teacher_forward(x_t, image_latents, embeds, None,
                                                timestep, img_shapes)
                transformer.enable_adapters()

            v_iso = runner.student_forward(x_t, image_latents, embeds, None,
                                           timestep, img_shapes)

            # ref 位置的输出推理时本来就被丢掉,不进 loss(runner 已只返回噪声段)
            loss = F.mse_loss(v_iso.float(), v_full.float()) / args.accum
            loss.backward()
            step_loss += loss.item()

        # 手动同步梯度,不套 DDP。原因:前向走的是 fork 出来的
        # `iso_transformer_forward`,不经过 `DistributedDataParallel.forward`,
        # DDP 的 autograd hook 根本挂不上,套了也不会同步——那是静默错。
        # 只有 LoRA 有梯度(rank 64 约 189 M 参数),一步一次 all_reduce 够便宜。
        if world > 1:
            for p in lora_params:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        losses.append(step_loss)

        if is_main and (step + 1) % args.log_every == 0:
            el = time.perf_counter() - t0
            done = step + 1 - start_step
            print(f"[{time.strftime('%H:%M:%S')}] step {step + 1}/{args.steps} | "
                  f"loss {np.mean(losses[-args.log_every:]):.5f} | "
                  f"{el / done:.1f} s/it | "
                  f"峰值 {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)

        if is_main and (step + 1) % args.save_every == 0:
            save(transformer, opt, step + 1, args.out)

    if is_main:
        save(transformer, opt, args.steps, args.out)
        el = time.perf_counter() - t0
        print("\n" + "=" * 68)
        print(f"训练结束 | {args.steps - start_step} 步 | {el / 60:.1f} min | "
              f"{el / max(1, args.steps - start_step):.1f} s/it")
        print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
        print(f"loss 首 {np.mean(losses[:10]):.5f} → 末 {np.mean(losses[-10:]):.5f}")
        print("=" * 68)
    if world > 1:
        torch.distributed.destroy_process_group()


def lora_state(transformer) -> dict:
    return {k: v.detach().cpu() for k, v in transformer.state_dict().items() if "lora" in k}


def set_lora_state(transformer, state: dict) -> None:
    transformer.load_state_dict(state, strict=False)


def save(transformer, opt, step: int, out_dir: str) -> None:
    path = os.path.join(out_dir, f"step{step:06d}.pt")
    torch.save({"step": step, "lora": lora_state(transformer), "opt": opt.state_dict()}, path)
    print(f"  ✓ 存 {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("precompute", help="离线预算 prompt_embeds(单卡)")
    pre.add_argument("--out", default=EMBED_CACHE)

    tr = sub.add_parser("train", help="速度匹配训练(torchrun)")
    tr.add_argument("--steps", type=int, default=100, help="先跑 100 步标定(§3.2)")
    tr.add_argument("--accum", type=int, default=1, help="梯度累积;有效 batch = accum × 卡数")
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--rank", type=int, default=LORA_RANK)
    tr.add_argument("--seed", type=int, default=20260813)
    tr.add_argument("--block_diag", action="store_true", help="ref 之间也隔离,本轮不训")
    tr.add_argument("--embeds", default=EMBED_CACHE)
    tr.add_argument("--allow_partial_embeds", action="store_true",
                    help="embeds 没算全也开跑。只用于冒烟,出不了正式结果")
    tr.add_argument("--out", default=os.path.join(_REPO_ROOT, "output/train_iso"))
    tr.add_argument("--resume", default=None)
    tr.add_argument("--log_every", type=int, default=10)
    tr.add_argument("--save_every", type=int, default=200)

    args = p.parse_args()
    (cmd_precompute if args.cmd == "precompute" else cmd_train)(args)


if __name__ == "__main__":
    main()
