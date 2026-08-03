#!/usr/bin/env python3
"""门禁①:导出官方 LoRA,并与 `ckpt-20000` 做**键集合严格相等**断言。

WHY 这个脚本必须存在(`DISTILL_PLAN.md` §11.4「工程门禁」):
`train.py:155` 是 `unwarp_dit.load_state_dict(lora_state, strict=False)`。
`strict=False` 的语义是**多余的键静默忽略、缺失的键静默保留原值**——
如果官方 LoRA 的键名与训练代码里 LoRA 参数的命名对不上,
**一个张量都不会被加载,而且不会报错**。我们会以为臂 A 是在官方权重上续训,
实际是从随机初始化开始,**loss 曲线上看不出任何异常**(两者都从高处平滑下降)。
臂 A / 臂 B 各 6 小时,这个坑一旦踩中就是 12 GPU 小时白烧,且事后无法从产物分辨。

WHY 不从文件读官方 LoRA:官方没有单独发布 LoRA safetensors,它是随 `only_lora=True`
挂载进模型的。`eval_multiref.py:181-190` 已趟过这条路——从**活模型的 state_dict**
里按后缀过滤提取,提出来的键天然就是模型自己的命名,不存在"命名规范猜错"的可能。
本脚本照抄那段提取逻辑(**故意逐字照抄,不做"改进"**:两处提取结果必须一致,
否则 M4 的 `official_full` 与臂 A 的 init 就不是同一个东西)。

WHY 断言的是"严格相等"而不是"没报错":子集关系不够。
  - 官方键 ⊋ ckpt 键 → 臂 A 会带上一批 ckpt 里没有的参数,与我们的对照不同构;
  - 官方键 ⊊ ckpt 键 → 臂 A 有一部分参数是随机初始化的,而这正是要防的那件事。
只有集合完全相等,"官方 init"与"我们的 init"才在同一个参数空间里,差的只有取值。

用法(H800,需要一张空闲 GPU;加载 ~7 min):
    # 只比对、不写文件(推荐先跑这个)
    python distill/export_official_lora.py --compare_only

    # 比对通过后导出到 log/official_init/dit_lora.safetensors
    python distill/export_official_lora.py

导出的文件可直接喂给臂 A:
    RESUME_FROM_CHECKPOINT=log/official_init/dit_lora.safetensors \
    bash scripts/train_distill.sh --ref_isolation False ...
(`train.py:145` 语义:传具体 safetensors 路径时 `global_step=0`、不恢复 optimizer。)
"""
from __future__ import annotations

import argparse
import os
import sys

# 与 eval_multiref.py:43-44 一致:H800 上直连 huggingface.co 不通、走代理会卡死,
# 必须离线加载本地缓存。任何会加载模型权重的新脚本都要照抄这两行(§11.4 runbook 的预警)。
os.environ.setdefault("HF_HOME", "/kaimm-distill/wuwenxuan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 后缀白名单与 eval_multiref.py:182-184 逐字相同,不许在这里"顺手扩一个"
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight",
                 ".lora_A.default.weight", ".lora_B.default.weight")


def extract_official(model_sd: dict) -> dict:
    """从活模型 state_dict 提官方 LoRA。逻辑与 eval_multiref.py:181-190 一致。"""
    sd = {k: model_sd[k].detach().clone().cpu() for k in model_sd
          if any(k.endswith(s) for s in LORA_SUFFIXES)}
    if not sd:  # only_lora 模式下官方 LoRA 已挂载,兜底备份全部 lora key
        sd = {k: v.detach().clone().cpu() for k, v in model_sd.items()
              if "lora" in k.lower()}
    if not sd:
        raise SystemExit("❌ 没找到任何 LoRA key——only_lora 挂载可能没生效")
    return sd


def gate(official: dict, ref: dict, ref_path: str) -> None:
    """门禁①本体。**任何一项不过就 exit 1,不给"警告后继续"的选项**——
    这个门禁的全部价值就在于它会拦住上机,能被绕过的门禁等于没有。"""
    errs: list[str] = []

    only_official = sorted(set(official) - set(ref))
    only_ref = sorted(set(ref) - set(official))
    if only_official:
        errs.append(f"官方多出 {len(only_official)} 个键,例如 {only_official[:3]}")
    if only_ref:
        errs.append(f"{os.path.basename(ref_path)} 多出 {len(only_ref)} 个键,"
                    f"例如 {only_ref[:3]}  ← 这些参数在臂 A 里会是随机初始化的")

    # 形状也要比:键名相同而形状不同时 load_state_dict(strict=False) 同样静默跳过,
    # 症状与键名不符完全一样(§11.4 已记:LoRA 形状不受 ref_isolation 影响,
    # 所以这里出现差异只可能是 lora_rank 传错)。
    shape_bad = [(k, tuple(official[k].shape), tuple(ref[k].shape))
                 for k in sorted(set(official) & set(ref))
                 if official[k].shape != ref[k].shape]
    if shape_bad:
        errs.append(f"{len(shape_bad)} 个键形状不一致,例如 {shape_bad[:2]}"
                    f"(先查 --lora_rank 是不是 512)")

    # 全零兜底:提取路径若走偏(比如挂载失败但张量已分配),键集合可能完全正确
    # 而取值全是 0——那同样是"从零开始训",且比键名不符更难发现。
    n_zero = sum(1 for v in official.values() if float(v.abs().max()) == 0.0)
    if n_zero == len(official):
        errs.append("官方 LoRA 全部张量为 0——挂载没生效,导出它等于随机初始化")
    elif n_zero:
        # LoRA 的 B 矩阵按惯例零初始化,但官方权重是训练过的,不该还有整片零。
        errs.append(f"官方 LoRA 有 {n_zero}/{len(official)} 个全零张量,可疑,停下人工确认")

    print(f"\n{'项':<28}{'结果'}")
    print(f"{'官方 LoRA 张量数':<28}{len(official)}")
    print(f"{'ckpt 张量数':<28}{len(ref)}")
    print(f"{'键集合':<28}{'相等 ✓' if not (only_official or only_ref) else '不相等 ✗'}")
    print(f"{'形状':<28}{'全部一致 ✓' if not shape_bad else f'{len(shape_bad)} 处不一致 ✗'}")
    print(f"{'全零张量':<28}{n_zero}")

    if errs:
        print("\n❌ 门禁① 未通过——**不要上机**:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("\n✅ 门禁① 通过:键集合严格相等、形状一致、非全零。臂 A 的 init 是真的官方权重。")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref_lora",
                   default="log/ref_isolation/checkpoint-20000/dit_lora.safetensors",
                   help="用来比对键集合的参照 checkpoint(默认 pre,即我们两臂的共同祖先)")
    p.add_argument("--out", default="log/official_init/dit_lora.safetensors")
    p.add_argument("--compare_only", action="store_true", help="只跑门禁,不写文件")
    p.add_argument("--model_type", default="flux-dev", choices=["flux-dev", "flux-dev-fp8"])
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lora_rank", type=int, default=512)
    args = p.parse_args()

    if not os.path.exists(args.ref_lora):
        raise SystemExit(f"❌ 参照 checkpoint 不存在:{args.ref_lora}")

    import torch  # noqa: PLC0415 — 放在 env 设置之后再 import,与 eval_multiref 同序
    from safetensors.torch import load_file, save_file  # noqa: PLC0415
    from uno.flux.pipeline import UNOPipeline  # noqa: PLC0415

    print("加载模型(~7 min,日志期间是静的)...", flush=True)
    pipeline = UNOPipeline(args.model_type, torch.device("cuda"), offload=args.offload,
                           only_lora=True, lora_rank=args.lora_rank)
    official = extract_official(pipeline.model.state_dict())
    ref = load_file(args.ref_lora, device="cpu")

    gate(official, ref, args.ref_lora)

    if args.compare_only:
        print("(--compare_only,未写文件)")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    save_file(official, tmp)      # 原子写,同 eval_multiref / build_probe_iso 的套路
    os.replace(tmp, args.out)
    size_mb = os.path.getsize(args.out) / 1024**2
    print(f"\n写入 {args.out}({len(official)} 张量,{size_mb:.0f} MB)")

    # 回读复核:save_file 之后立刻重新读一遍并再过一次门禁。WHY——真正喂给臂 A 的是
    # **磁盘上这个文件**,不是内存里那个 dict;dtype 在序列化时被改掉之类的问题
    # 只有回读才看得见,而它一旦发生就是 6 小时白烧。
    back = load_file(args.out, device="cpu")
    print("\n回读复核:")
    gate(back, ref, args.ref_lora)


if __name__ == "__main__":
    main()
