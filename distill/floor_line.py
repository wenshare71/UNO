"""text-only 地板线诊断:证明 teacher 的身份保真来自 ref 图,而非文本先验蒙对。

━━ 为什么做这个 ━━
有个合理的担心:prompt 形如 "a cat, a cartoon and a vase on top of pink fabric",里头的
vase 会不会是 flux 靠文本先验直接画出的一只**泛型**花瓶,而根本没照着 ref 那只复刻?
若真如此,M2 的高分就不能证明 teacher 在"读图",确认点①的前提就虚了。

━━ 怎么测(和 fp8 红灯同理:一次只动一个变量)━━
不去改 prompt 格式(改成 "image 1, image 2..." 会让 teacher 撞上训练分布外的说法而退化,
退化到底是"没在用 ref"还是"没见过这种话"两个变量混在一起,不可归因)。真正干净的对照是:
**同样 16 条 prompt + 同样 seed,唯一区别是不给 ref 图**(gen_data.py --text_only,model 把
空 ref 归一化成 None → 纯 flux-dev 文生图)。对每张 text-only 图,用与 M2 **完全相同**的
口径打分,取 min_ref_sim。这就是"文本先验独自能拿多少分"的地板线。

━━ 口径修订(2026-07-24):v1 整图 DINO 已证伪,本脚本随 M2 切到 v2 ━━
第一轮地板线(整图 DINO)恰好证明了整图口径自身不可用:teacher 与 text-only 中位差仅
+0.028、6/16 反转、0/16 越过地板线,而 teacher 读图有肉眼铁证(枣红背包连徽章都复刻对了)
——指标与视觉相反,死的是指标。完整尸检与 v2 设计见 `filter_data.py` 头注释。
本脚本现在直接复用 filter_data 的 v2 打分(GroundingDino 定位 + 双侧裁剪 + DINO
全视角比对),teacher 与 text-only 共用同一个 ctx,数值严格可比。

━━ 判读(v2 口径下的预期)━━
  * text-only 的"泛型同类物体 crop vs 特定 ref crop"应落在 ~0.35–0.5;teacher 忠实复刻
    应落在 ~0.6+;分布应显著分开(中位差 ≥0.15 量级),干净样本应越过 text-only max。
  * teacher 掉主体的样本(003000/006000)该主体检不到框 → sim=0.0,直接垫底。
  * 若 v2 口径下两分布仍重叠 → 度量再回炉,**不要**带着测不准的尺子跑全量。
  * 阈值下锚不变:确认点②的阈值必须设在地板线之上,否则留下的可能是"文本碰巧生对"
    的污染样本。

━━ 用法(远程,单卡几分钟)━━
    # 0) 预取 grounding-dino-tiny(见 filter_data.py 头注释,一次性)
    # 1) 先要有 teacher manifest(C 步已 --merge 过就跳过)
    python distill/gen_data.py --merge
    # 2) 生成 16 张 text-only 地板线图(同 prompt/seed,无 ref;已生成过会被 already_done 跳过)
    CUDA_VISIBLE_DEVICES=0 python distill/gen_data.py --text_only --num_shards 500 --shard_idx 0
    # 3) 配对打分 + 分布对比 + 拼图
    CUDA_VISIBLE_DEVICES=0 python distill/floor_line.py
把 stdout 全文回传即可判读。board_floorline.png 想看需自行提交(images_textonly 默认 gitignore)。
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 复用 M2 完全同一套 v2 打分(检测 + 裁剪 + DINO),保证地板线与 teacher 分数可比。
from distill.filter_data import annotate_det, load_manifest, make_ctx, resolve, score_records

DEFAULT_OUT = "datasets/distill_multiref"
DEFAULT_TEACHER = os.path.join(DEFAULT_OUT, "manifest_raw.json")
DEFAULT_TEXTONLY = os.path.join(DEFAULT_OUT, "manifest_textonly_shard0.json")


def _q(arr, ps=(10, 25, 50, 75, 90)):
    return {p: float(np.percentile(arr, p)) for p in ps}


def print_report(pairs, floor_hi):
    """pairs: [(idx, subjects, n_refs, teacher_min, textonly_min, rec_t, rec_o)]。"""
    pairs = sorted(pairs, key=lambda x: x[3])  # 按 teacher_min 升序:掉主体的排最上面
    t_arr = np.array([p[3] for p in pairs])
    o_arr = np.array([p[4] for p in pairs])

    print("\n" + "=" * 92)
    print("逐样本配对(按 teacher_min 升序;Δ = teacher_min − textonly_min,越大越说明 ref 带来的增量越大)")
    print("-" * 92)
    print(f"{'idx':>6}  {'n':>1}  {'subjects':<34}{'teacher':>9}{'text-only':>11}{'Δ':>8}")
    print("-" * 92)
    for idx, subs, n, tmin, omin, rec_t, _ in pairs:
        s = "+".join(subs)
        s = s if len(s) <= 33 else s[:32] + "…"
        drop = [str(j + 1) for j, d in enumerate(rec_t["meta"].get("det", []))
                if d["n_boxes"] == 0]
        flag = f"  ← 主体{','.join(drop)}无框(判掉)" if drop else (
            "  ← 塌到地板" if tmin <= floor_hi else "")
        print(f"{idx:>6}  {n:>1}  {s:<34}{tmin:>9.3f}{omin:>11.3f}{tmin - omin:>8.3f}{flag}")
    print("-" * 92)

    tq, oq = _q(t_arr), _q(o_arr)
    print(f"{'分位':>6}     {'':<34}{'teacher':>9}{'text-only':>11}")
    for p in (10, 25, 50, 75, 90):
        print(f"{'p' + str(p):>6}     {'':<34}{tq[p]:>9.3f}{oq[p]:>11.3f}")
    print("-" * 92)
    print(f"中位数间隔 median(teacher) − median(text-only) = {tq[50] - oq[50]:+.3f}")
    print("=" * 92)

    # 阈值下锚:地板线的上界。max 最严(任何 text-only 都过不了),p90 抗单点离群更稳。
    print("\n【阈值下锚】过滤阈值必须设在文本先验地板线**之上**:")
    print(f"  text-only 的 max = {o_arr.max():.3f}  |  p90 = {oq[90]:.3f}")
    n_clear = int((t_arr > o_arr.max()).sum())
    print(f"  teacher 中 min_ref_sim 高过整条地板线(> text-only max)的样本:"
          f"{n_clear}/{len(pairs)} —— 这些是可证明「ref 驱动」的干净样本。")
    print(f"  建议:确认点②的阈值取 ≥ {max(oq[90], float(o_arr.max())):.3f}(不低于地板线上界),"
          "再结合 board 视觉微调。")


def make_compare_board(pairs_full, out_dir, save_path):
    """每行 [各 ref | 红线 | teacher(叠检测框) | text-only],标题带两个 min。"""
    from multibanana_eval.board import build_row, stack_board

    rows = []
    for idx, subs, n, tmin, omin, rec_t, rec_o in sorted(pairs_full, key=lambda x: x[3]):
        refs = [Image.open(resolve(out_dir, p)).convert("RGB") for p in rec_t["image_paths"]]
        g_t = annotate_det(
            Image.open(resolve(out_dir, rec_t["image_tgt_path"])).convert("RGB"),
            rec_t["meta"].get("det"))
        g_o = Image.open(resolve(out_dir, rec_o["image_tgt_path"])).convert("RGB")
        title = (f"{idx:06d}  {n}-ref  teacher_min={tmin:.3f}  text-only_min={omin:.3f}  "
                 f"Δ={tmin - omin:+.3f}")
        rows.append(build_row(title, rec_t["prompt"], refs,
                              {"teacher": g_t, "text-only": g_o}))
    stack_board(rows).save(save_path)
    print(f"\n拼图已存:{save_path}({len(rows)} 行,按 teacher_min 从低到高)", flush=True)


def main():
    from distill.gen_data import DATA_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--teacher_manifest", default=DEFAULT_TEACHER,
                   help="带 ref 的 teacher 产出 manifest(需先 --merge)")
    p.add_argument("--textonly_manifest", default=DEFAULT_TEXTONLY,
                   help="--text_only 产出的 manifest")
    p.add_argument("--out_dir", default=DEFAULT_OUT, help="图路径的基准目录")
    p.add_argument("--data_dir", default=DATA_DIR, help="dreambooth 参考图目录")
    p.add_argument("--board_out", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--box_thr", type=float, default=0.22, help="与 filter_data 同义")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    teacher = load_manifest(args.teacher_manifest)
    textonly = load_manifest(args.textonly_manifest)

    # 同一个 ctx 给两边打分:模型、类别词、ref 特征库、阈值全部一致,数值严格可比
    ctx = make_ctx(args.device, args.data_dir, args.box_thr, args.batch_size)
    teacher, t_skip = score_records(teacher, args.out_dir, ctx)
    textonly, o_skip = score_records(textonly, args.out_dir, ctx)
    if t_skip or o_skip:
        print(f"⚠️ 缺生成图跳过:teacher {len(t_skip)} 条,text-only {len(o_skip)} 条", flush=True)
    if not textonly:
        raise SystemExit("❌ text-only 一条图都没有——先跑 "
                         "`python distill/gen_data.py --text_only --num_shards 500 --shard_idx 0`")

    # 按 seed 配对(seed = base_seed + idx,一一对应且全局唯一)
    by_seed = {r["meta"]["seed"]: r for r in teacher}
    pairs_full = []
    missing = 0
    for r in textonly:
        t = by_seed.get(r["meta"]["seed"])
        if t is None:
            missing += 1
            continue
        pairs_full.append((
            t["meta"]["seed"] - 3407000,  # idx,仅用于显示;3407000 是 gen_data 的 base_seed
            r["meta"]["subjects"], r["meta"]["n_refs"],
            t["meta"]["min_ref_sim"], r["meta"]["min_ref_sim"], t, r,
        ))
    if missing:
        print(f"⚠️ {missing} 条 text-only 样本在 teacher manifest 里找不到对应 seed"
              "(teacher 是否漏跑/漏 merge?),已跳过。", flush=True)
    if not pairs_full:
        raise SystemExit("❌ teacher 与 text-only 没有共同 seed,无法配对——检查两个 manifest。")

    print(f"配对成功 {len(pairs_full)} 组(teacher {len(teacher)} 条 × text-only {len(textonly)} 条)。")

    # 地板线上界(text-only 的 p90):阈值判读与"塌到地板"标记都用它
    floor_hi = float(np.percentile([x[4] for x in pairs_full], 90))
    print_report(pairs_full, floor_hi)

    board_out = args.board_out or os.path.join(args.out_dir, "board_floorline.png")
    make_compare_board(pairs_full, args.out_dir, board_out)


if __name__ == "__main__":
    main()
