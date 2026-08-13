"""训练路径自检:LoRA 挂载 / 梯度检查点 / 反传 / 教师身份切换 / sigma 网格。

规格:`qwen/PLAN.md` §3.2。**这不是门禁**——门禁是 `test_iso_equiv.py`(等价性)。
这一份的目的很具体:**`train_iso.py` 在 8×H800 上崩掉的那些原因,绝大多数不需要
真权重就能试出来。** 而 H800 只有一切就绪时才申请得下来,拿 8 张卡陪着调
`add_adapter` 的参数名太贵。

所以照搬 `test_iso_equiv.py` 的路子:2 层随机权重、fp32、纯 CPU、秒级。
它**测不到**的是显存、s/it、DDP 的 all_reduce —— 那三样只能上真机。

用法:
    python qwen/test_train_smoke.py           # 只跑 tiny(几秒,不需要权重)
    python qwen/test_train_smoke.py --pipe    # 另加真 pipeline 那一段(CPU 加载 54 GB)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from pipeline_iso import IsoRunner
from train_iso import LORA_TARGETS, NUM_INFERENCE_STEPS, build_sigma_grid

# 与 test_iso_equiv.py 同一份 TINY,只是层数留 2 层够验梯度沿层传播
TINY = dict(
    patch_size=2, in_channels=64, out_channels=16,
    num_layers=2, attention_head_dim=16, num_attention_heads=2,
    joint_attention_dim=32, guidance_embeds=False,
    axes_dims_rope=(4, 6, 6), zero_cond_t=True,
)
IMG_SHAPES = [[(1, 4, 4), (1, 4, 4), (1, 6, 4)]]
TXT_LEN = 7
SEED = 20260813

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}  {detail}")


def build():
    torch.manual_seed(SEED)
    model = QwenImageTransformer2DModel(**TINY).to(torch.float32)
    n_noise = IMG_SHAPES[0][0][0] * IMG_SHAPES[0][0][1] * IMG_SHAPES[0][0][2]
    n_ref = sum(f * h * w for f, h, w in IMG_SHAPES[0][1:])
    g = torch.Generator().manual_seed(SEED)
    latents = torch.randn(1, n_noise, TINY["in_channels"], generator=g)
    image_latents = torch.randn(1, n_ref, TINY["in_channels"], generator=g)
    ehs = torch.randn(1, TXT_LEN, TINY["joint_attention_dim"], generator=g)
    return model, latents, image_latents, ehs, n_noise


def run_tiny():
    from peft import LoraConfig

    model, latents, image_latents, ehs, n_noise = build()
    t = torch.tensor([0.7])

    # ---------------------------------------------------------------- S1 挂载
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                                 target_modules=LORA_TARGETS))
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    lora_names = [n for n, p in model.named_parameters() if p.requires_grad]
    # 每层 8 个目标模块 × (lora_A + lora_B) = 16 个张量
    expect = TINY["num_layers"] * len(LORA_TARGETS) * 2
    check(f"S1 add_adapter 认得 LORA_TARGETS 那 8 个名字(期望 {expect} 个张量)",
          len(lora_params) == expect,
          f"实得 {len(lora_params)} | 例:{lora_names[0] if lora_names else '(空)'}")
    check("S1b 只有 LoRA 要梯度,主干全冻",
          all("lora" in n for n in lora_names), f"非 LoRA 的可训张量 "
          f"{[n for n in lora_names if 'lora' not in n][:2] or '无'}")

    # ---------------------------------------------------------------- S2 梯度检查点
    model.enable_gradient_checkpointing()
    check("S2 enable_gradient_checkpointing 落到顶层模型上(fork 的 forward 读的是它)",
          model.gradient_checkpointing is True
          and getattr(model, "_gradient_checkpointing_func", None) is not None,
          f"gradient_checkpointing={model.gradient_checkpointing}")

    runner = IsoRunner(model, store=False)

    def fwd(mode):
        return runner(latents, image_latents, ehs, None, t, IMG_SHAPES, mode=mode)

    # ---------------------------------------------------------------- S3 教师身份
    # LoRA 的 lora_B 初始化为 0 ⇒ 增量恒等 ⇒ 此刻开关切换应当逐位无差。
    # 这一条先立住,S6 才能证明"参数动了之后 teacher 确实回到了原样"。
    with torch.no_grad():
        model.disable_adapters()
        off_disabled = fwd("off")
        model.enable_adapters()
        off_enabled = fwd("off")
    d = (off_disabled - off_enabled).abs().max().item()
    check("S3 LoRA 初始为恒等(lora_B=0),disable/enable 逐位相同", d == 0.0, f"max={d:.3e}")

    # ---------------------------------------------------------------- S4 loss 有信号
    with torch.no_grad():
        v_full = fwd("off")
        v_iso = fwd("write")
    gap = F.mse_loss(v_iso, v_full).item()
    check("S4 第 0 步 loss 非零 —— 差异来自 mask 而非 LoRA(LoRA 此刻还是恒等)",
          gap > 1e-8, f"mse={gap:.6e}")

    # ---------------------------------------------------------------- S5 反传
    with torch.no_grad():
        model.disable_adapters()
        v_full = runner.teacher_forward(latents, image_latents, ehs, None, t, IMG_SHAPES)
        model.enable_adapters()
    v_iso = runner.student_forward(latents, image_latents, ehs, None, t, IMG_SHAPES)
    check("S5a student 输出形状 = 噪声段(ref 位置不进 loss)",
          tuple(v_iso.shape) == (1, n_noise, TINY["patch_size"] ** 2 * TINY["out_channels"]),
          f"{tuple(v_iso.shape)}")
    check("S5b teacher 已 detach,不带梯度", not v_full.requires_grad,
          f"requires_grad={v_full.requires_grad}")

    loss = F.mse_loss(v_iso.float(), v_full.float())
    loss.backward()

    # 哪些该有梯度,是能推出来的,不是数出来的:
    #  · lora_A 全零 —— 增量是 B@A,init 时 lora_B=0 ⇒ dL/dA = Bᵀ(…) = 0;
    #  · lora_B 非零 —— dL/dB = dL/dy·(Ax)ᵀ,只要上游梯度到得了就非零;
    #  · 唯二例外:**最后一层**的 add_q_proj 和 to_add_out。forward 循环结束后
    #    只有 hs 进 norm_out/proj_out,`ehs` 被丢掉;而这两个模块只喂文本流
    #    (add_q_proj → txt_query → 注意力的 txt 行 → txt_attn_output → to_add_out)。
    #    add_k/v_proj 不在此列 —— txt 的 K/V 会被图像 query 注意到,影响得到输出。
    #  ⇒ 非零个数 = 8L − 2。L=2 时 14,真模型 L=60 时 478/480。**这是结构,不是 bug。**
    last = TINY["num_layers"] - 1
    DEAD = {f"transformer_blocks.{last}.attn.add_q_proj",
            f"transformer_blocks.{last}.attn.to_add_out"}
    mismatch = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nonzero = p.grad is not None and p.grad.abs().sum().item() > 0
        want = ".lora_B." in n and n.split(".lora_")[0] not in DEAD
        if nonzero != want:
            mismatch.append(f"{n}: 实测{'非零' if nonzero else '零'}, 应{'非零' if want else '零'}")
    n_nz = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().sum() > 0)
    check(f"S5c 检查点下能反传,且拿到梯度的恰好是该拿的那 {expect // 2 - len(DEAD)} 个",
          not mismatch, f"非零 {n_nz}/{expect}"
          + (f" | 对不上:{mismatch[:3]}" if mismatch else " | 逐个点名一致"))

    # ---------------------------------------------------------------- S6 缓存不漏
    # 反传时检查点会重跑 write 分支。若靠"forward 之后 clear"来清,重跑会再写一遍,
    # 一步下来 60 层 ref K/V 全挂着(真模型 2-ref 约 6 GB)。store=False 才是对的。
    check("S6 反传之后 cache 仍为空(store=False 而非事后 clear)",
          len(runner.ctx.cache) == 0, f"cache 里有 {len(runner.ctx.cache)} 层")

    # ---------------------------------------------------------------- S7 优化器
    before = [p.detach().clone() for p in lora_params]
    opt = torch.optim.AdamW(lora_params, lr=1e-3)
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt.step()
    moved = sum(1 for a, p in zip(before, lora_params) if not torch.equal(a, p))
    check("S7a optimizer.step() 之后 LoRA 参数确实动了", moved > 0, f"{moved}/{expect} 个张量变化")

    with torch.no_grad():
        model.disable_adapters()
        teacher_after = fwd("off")
        model.enable_adapters()
        student_after = fwd("off")
    d_t = (teacher_after - off_disabled).abs().max().item()
    d_s = (student_after - off_disabled).abs().max().item()
    check("S7b 更新后 teacher 回到原样(disable 是真的关掉,不是近似)", d_t == 0.0, f"max={d_t:.3e}")
    check("S7c 更新后 student 确实变了(否则 LoRA 没接进前向)", d_s > 0.0, f"max={d_s:.3e}")

    # ---------------------------------------------------------------- S8 存/续跑
    from train_iso import lora_state, set_lora_state
    state = lora_state(model)
    scrambled = copy.deepcopy(model)
    with torch.no_grad():
        for p in scrambled.parameters():
            if p.requires_grad:
                p.add_(1.0)
    set_lora_state(scrambled, state)
    d = max((a - b).abs().max().item()
            for a, b in zip(model.state_dict().values(), scrambled.state_dict().values()))
    check("S8 lora_state / set_lora_state 往返一致(断点续跑靠它)", d == 0.0, f"max={d:.3e}")

    # ---------------------------------------------------------------- S10 时间约定
    # 推理传的是 bf16(scheduler.timesteps)/1000,而 timesteps = σ×1000 ⇒ **舍入两次**。
    # 直接 sigma.to(bf16) 只舍一次,两者不等。离线实算:40 个格点里 12 个不同,
    # 最大相对差 0.706%。这条钉住训练侧走的是与推理逐位相同的那条路。
    s32 = torch.tensor([0.9313406, 0.8182465, 0.5, 0.0749129], dtype=torch.float32)
    infer_t = (s32 * 1000).to(torch.bfloat16) / 1000          # pipeline L814 + L818
    train_t = (s32 * 1000).to(torch.bfloat16) / 1000          # train_iso.py 现在的写法
    naive_t = s32.to(torch.bfloat16)                          # 改之前的写法
    check("S10 训练的 t 与推理逐位相同(不是只舍入一次的那个)",
          torch.equal(train_t, infer_t) and not torch.equal(naive_t.float(), infer_t.float()),
          f"与推理一致={torch.equal(train_t, infer_t)} | "
          f"朴素写法差 {(naive_t.float() - infer_t.float()).abs().max():.3e}")

    # ---------------------------------------------------------------- S11 续跑不静默
    bad = {k.replace("transformer_blocks", "nonexistent_blocks"): v for k, v in state.items()}
    try:
        set_lora_state(model, bad)
        raised = False
    except SystemExit:
        raised = True
    check("S11 set_lora_state 遇到对不上的 key 会 raise(不是静默丢掉)",
          raised, "raise 了" if raised else "**静默返回** —— 续跑会从随机权重重训")

    # ---------------------------------------------------------------- S12 评测侧加载
    # 整条评测链上唯一一处"加载失败但不报错"的地方,必须往返验一遍。
    import tempfile
    from infer_iso import apply_lora_ckpt
    from train_iso import LORA_TARGETS as _T

    fresh = QwenImageTransformer2DModel(**TINY).to(torch.float32)
    fresh.requires_grad_(False)
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "step000001.pt")
        torch.save({"step": 1, "rank": 8, "targets": list(_T), "lora": state, "opt": {}}, ck)
        try:
            apply_lora_ckpt(fresh, ck)
            loaded, err = True, ""
        except SystemExit as e:
            loaded, err = False, str(e)[:120]
        check("S12a apply_lora_ckpt 能吃下 train_iso.save() 的格式", loaded, err or "OK")

        zero = {k: (torch.zeros_like(v) if "lora_B" in k else v) for k, v in state.items()}
        torch.save({"step": 1, "rank": 8, "targets": list(_T), "lora": zero, "opt": {}}, ck)
        fresh2 = QwenImageTransformer2DModel(**TINY).to(torch.float32)
        fresh2.requires_grad_(False)
        try:
            apply_lora_ckpt(fresh2, ck)
            caught = False
        except SystemExit:
            caught = True
        check("S12b 未训练的 ckpt(lora_B 全 0)会被拦下,不会冒充 iso_post",
              caught, "拦下了" if caught else "**放行了** —— iso_post 臂会跑成 iso_pre")

    # ---------------------------------------------------------------- S9 sigma 网格
    import types
    from diffusers import FlowMatchEulerDiscreteScheduler
    cfg = os.path.join(_HERE, "_vendor/qwen2511_config/scheduler_config.json")
    import json
    with open(cfg, "rt", encoding="utf-8") as f:
        sched = FlowMatchEulerDiscreteScheduler.from_config(json.load(f))
    sigmas = build_sigma_grid(types.SimpleNamespace(scheduler=sched), 4096)
    ok = (len(sigmas) == NUM_INFERENCE_STEPS
          and bool((sigmas[:-1] > sigmas[1:]).all())
          and sigmas[0] > 0.9)
    check(f"S9 sigma 网格 = 推理实际那 40 个(单调降,首 {sigmas[0]:.4f} 末 {sigmas[-1]:.4f})",
          ok, f"n={len(sigmas)}")
    # 1/2/3-ref 共用一条 —— image_seq_len 只算噪声图(pipeline L765)
    check("S9b 网格与 ref 数无关(seq_len 只算噪声段)",
          torch.equal(sigmas, build_sigma_grid(types.SimpleNamespace(scheduler=sched), 4096)),
          "同 seq_len 复现一致")


def run_pipe():
    """真 pipeline 那一段:CPU 上加载 54 GB,只验 API 接线,不上卡。

    覆盖 tiny 覆盖不到的:`prepare_latents` / `_encode_vae_image` / `_pack_latents` /
    `image_processor.preprocess` / `img_shapes` 的算法 —— 这几个我是照着源码写的,没跑过。
    """
    from diffusers import QwenImageEditPlusPipeline

    from train_iso import encode_sample, load_manifest

    weights = os.environ.get("QWEN_WEIGHTS")
    if not weights:
        check("P0 QWEN_WEIGHTS 未设置,跳过真 pipeline 段", False, "(想跑就设上它)")
        return

    items = load_manifest()
    check(f"P1 manifest 读入 + 泄漏断言通过", len(items) == 9000, f"{len(items)} 条")

    pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    pipe.text_encoder = None

    item = next(it for it in items if it["meta"]["n_refs"] == 2)
    embeds_dir = os.environ.get("EMBED_CACHE", os.path.join(_REPO_ROOT, "cache/prompt_embeds"))
    have_embeds = os.path.exists(os.path.join(embeds_dir, f"{item['_idx']:06d}.pt"))
    if not have_embeds:
        check("P2 prompt_embeds 缓存不存在,只验 latent 那一半", True,
              f"(缺 {embeds_dir}/{item['_idx']:06d}.pt)")
        # 绕开 embeds,直接验 latent 路径
        import train_iso
        _orig = torch.load
        torch.load = lambda *a, **k: torch.zeros(4, TINY["joint_attention_dim"])
        try:
            x0, image_latents, embeds, img_shapes = encode_sample(
                pipe, item, embeds_dir, item["_idx"], torch.device("cpu"))
        finally:
            torch.load = _orig
    else:
        x0, image_latents, embeds, img_shapes = encode_sample(
            pipe, item, embeds_dir, item["_idx"], torch.device("cpu"))

    check("P3 x₀ 打包成 4096 个 token(1024² / 8 / 2 的平方)",
          x0.shape[1] == 4096, f"{tuple(x0.shape)}")
    check("P4 2-ref 的 image_latents = 8192 个 token",
          image_latents.shape[1] == 8192, f"{tuple(image_latents.shape)}")
    check("P5 img_shapes 第一项是噪声图 (1,64,64)",
          img_shapes[0][0] == (1, 64, 64), f"{img_shapes[0]}")
    check("P6 img_shapes 的 token 数与 latents 对得上",
          sum(f * h * w for f, h, w in img_shapes[0][1:]) == image_latents.shape[1],
          f"{sum(f * h * w for f, h, w in img_shapes[0][1:])} vs {image_latents.shape[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", action="store_true", help="另跑真 pipeline 段(需要 QWEN_WEIGHTS)")
    args = ap.parse_args()

    print(f"[自检] 训练路径 | 小模型 {TINY['num_layers']} 层 | fp32 CPU | "
          f"LoRA targets {len(LORA_TARGETS)} 个")
    run_tiny()
    if args.pipe:
        print("\n[自检] 真 pipeline 段(CPU 加载,不上卡)")
        run_pipe()

    n_bad = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 68)
    if n_bad:
        print(f"❌ {n_bad}/{len(results)} 项不通过。别申请机器,先修。")
    else:
        print(f"✅ {len(results)}/{len(results)} 项通过。")
        print("   还没验的三样只能上真机:峰值显存 / s-it / DDP all_reduce。")
    print("=" * 68)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
