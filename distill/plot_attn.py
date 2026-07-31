"""把 `output/attn_diag/*.npz` 渲染成一张自包含 HTML 报告。**纯 CPU,本地跑。**

## 为什么不用 matplotlib

只依赖 numpy + pillow(本地已有)。曲线走内联 SVG,热力图走内联 PNG data-URI,
生成图缩到 256 后 JPEG 内联。产物是单个 .html,能直接提交、直接发人看,
不需要对方装任何东西。也符合仓库约定(展示型产出用 HTML)。

## 报告结构

  1. 总览表:每样本 × 每变体的 late-block ref 份额,以及失衡比 min/max
     —— 失衡比是"第二主体是否被冷落"的一个标量代理
  2. per-head 分辨(D04 新增):同样的 late-block 份额,但**不对 24 个头取平均**
     —— 回答"是不是只有少数几个头在做身份绑定,平均把它们摊薄了"
  3. 每样本一节:
     a. 四变体出图并排(肉眼判丢失/复制)
     b. 曲线网格:行 = block 组(early/mid/late),列 = ref 段,每格 4 条变体线
        —— 回答"什么时候、在哪一层掉下去"
     c. per-head 条带:24 格 × 4 变体,看单样本上哪些头突出
     d. 热力图:指定 block × step,每 ref 一张主图区域图
        —— 区分**丢失**(ref2 份额低 + 单一热区)与**复制**(ref2 份额低 + 两处热区都热在 ref1)

## 为什么加 per-head(D04)

首轮 8 样本跑完后,全 head 平均的份额**测不出主体有没有真的出现**:S1_022 里
`ours_kv_pre` 肉眼确认丢了 grey_sloth_plushie,份额却比渲染正确的 `ours_kv_post4000`
还高;S1_000 里 pre 整张图崩坏,失衡比照样报"均衡"。所以录制侧把 curve 拆到了
head 维度,这边负责把它读出来看。**这是在验证一个假设,不是已经成立的结论。**

npz 兼容:D04 之前录的 curve 是 3 维 `(step, block, seg)`,之后是 4 维
`(step, block, head, seg)`。`load_runs` 统一成两份——`curve`(永远是 head 平均,
老报告的所有视图靠它,数值与 D04 之前逐位一致)和 `curve_head`(4 维时才有,
老 npz 上为 None,per-head 各节自动跳过并注明)。

## 解读红线(沿用 D01,报告里会原样印出来)

注意力份额高 ≠ 身份信息被转移。身份保真可能主要由 K/V **内容**承载,
份额只是"看了多久",不是"看到了什么"。所以这些图是诊断线索,不是因果证据。
**per-head 拆开也不改变这条**:某个头份额高,只说明它在看,不说明它看懂了。

用法:
  python distill/plot_attn.py                      # → output/attn_diag/report.html
  python distill/plot_attn.py --heat_block 52 --heat_step 12
"""
import argparse
import base64
import glob
import html
import io
import json
import os

import numpy as np
from PIL import Image

# block 分组:double 全段 / single 前半 / single 后半。
# 19 是 double→single 的架构分界(model.py:95/97),不是随手切的。
GROUPS = [("early · double 0-18", 0, 19),
          ("mid · single 0-18", 19, 38),
          ("late · single 19-37", 38, 57)]

VARIANT_ORDER = ["official_full", "ours_kv_pre", "ours_iso_nocache", "ours_kv_post4000"]

# late block 段(flat 38-56)。份额在这里最能反映"身份细化阶段还看不看这张 ref"。
LATE_B0, LATE_B1 = 38, 57


def step_windows(n_steps):
    """把 timestep 切成前/后三分之一。

    WHY 不对全部 step 取均值:D02 要回答的是"第二主体是**布局阶段**(早期 step)就没被
    放置,还是**后期细化阶段**衰减"。把时间轴平均掉这个问题就答不了了,而且只发生在
    后段的塌陷会被前段稀释掉一半——最初版本正是这么写的,合成数据测试里
    pre 的失衡比被摊成 0.507,几乎和 teacher 的 0.52 分不开。
    """
    k = max(1, n_steps // 3)
    return slice(0, k), slice(n_steps - k, n_steps)


def seg_shares(curve, seg_names, win):
    """给定 timestep 窗口,返回 {段名: late-block 平均份额}。"""
    m = curve[win, LATE_B0:LATE_B1, :].mean(axis=(0, 1))
    return {n: float(m[i]) for i, n in enumerate(seg_names)}


def imbalance(shares):
    """min(ref) / max(ref):1.0 = 各 ref 同等被关注,→0 = 某个 ref 基本没被看。"""
    refs = [v for n, v in shares.items() if n.startswith("ref")]
    return (min(refs) / max(refs)) if refs and max(refs) > 0 else 0.0


def head_shares(curve_head, win, b0=LATE_B0, b1=LATE_B1):
    """(head, seg) —— 与 seg_shares 同样的窗口,但 head 维度留着不平均(D04)。"""
    return curve_head[win, b0:b1].mean(axis=(0, 1))


def head_imbalance(hs, seg_names):
    """(head,) —— 每个 head 各自算一次 min(ref)/max(ref)。

    注意这**不等于**对 head 平均后再算失衡比:min/max 是非线性的,
    先平均会把"A 头只看 ref1、B 头只看 ref2"这种互补分工洗成"两个头都很均衡"。
    D04 想看的正是这种被洗掉的结构。
    """
    idx = [i for i, n in enumerate(seg_names) if n.startswith("ref")]
    r = np.asarray(hs, dtype=np.float64)[:, idx]
    mx = r.max(axis=1)
    return np.where(mx > 0, r.min(axis=1) / np.maximum(mx, 1e-12), 0.0)


COLORS ={"official_full": "#2f7ed8", "ours_kv_pre": "#d94e4e",
          "ours_iso_nocache": "#e0a030", "ours_kv_post4000": "#3aa655"}
LABELS = {"official_full": "teacher (full attn)",
          "ours_kv_pre": "pre · ckpt-20000",
          "ours_iso_nocache": "pre · 隔离无缓存",
          "ours_kv_post4000": "post · ckpt-4000"}


def _s(v):
    """npz 里的 0-d / bytes 标量还原成 python 值。"""
    a = np.asarray(v)
    if a.ndim == 0:
        a = a.reshape(1)
    out = [x.decode() if isinstance(x, bytes) else x for x in a.tolist()]
    return out[0] if len(out) == 1 else out


def split_curve(raw):
    """npz 里的 curve → (head 平均的 3 维, 按 head 分辨的 4 维或 None)。

    D04 之前录的是 3 维,之后是 4 维。老 npz 拿不出 head 维度,但 head 平均那份视图
    两种格式都有——所以老数据照样能出完整的旧报告,只是 per-head 各节会缺席。
    """
    a = np.asarray(raw)
    if a.ndim == 4:
        return a.mean(axis=2), a
    if a.ndim == 3:
        return a, None
    raise ValueError(f"curve 维数 {a.ndim} 既不是 3(head 平均)也不是 4(按 head),"
                     f"shape={a.shape}")


def load_runs(save_path):
    runs = []
    for p in sorted(glob.glob(os.path.join(save_path, "*.npz"))):
        z = np.load(p, allow_pickle=False)
        stem = p[:-4]
        curve, curve_head = split_curve(z["curve"])
        runs.append({
            "npz": p,
            "png": stem + ".png" if os.path.exists(stem + ".png") else None,
            "task_id": _s(z["meta_task_id"]),
            "variant": _s(z["meta_variant"]),
            "prompt": _s(z["meta_prompt"]),
            "subjects": [_s(x) for x in np.atleast_1d(z["meta_subjects"])],
            "seed": int(_s(z["meta_seed"])),
            "curve": curve,
            "curve_head": curve_head,
            "seg_names": [_s(x) for x in np.atleast_1d(z["seg_names"])],
            "spatial": {(int(k[0]), int(k[1])): z[f"spatial_{int(k[0]):03d}_{int(k[1]):03d}"]
                        for k in z["spatial_keys"]},
            "q_len_by_step": z["q_len_by_step"].tolist(),
        })
    return runs


# ------------------------------------------------------------------ 画图基元

def svg_panel(series, title, ylab, w=250, h=150, pad=32):
    """series: [(label, color, y_array)];x 轴是 timestep 序号。"""
    if not series:
        return ""
    n = len(series[0][2])
    ymax = max(0.02, max(float(np.max(y)) for _, _, y in series)) * 1.15
    iw, ih = w - pad - 6, h - pad - 16

    def X(i):
        return pad + (iw * i / max(1, n - 1))

    def Y(v):
        return 16 + ih - (ih * v / ymax)

    out = [f'<svg viewBox="0 0 {w} {h}" class="pn">']
    for frac in (0, 0.5, 1.0):
        y = Y(ymax * frac)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w - 6}" y2="{y:.1f}" class="gr"/>')
        out.append(f'<text x="{pad - 4}" y="{y + 3:.1f}" class="ax" '
                   f'text-anchor="end">{ymax * frac:.2f}</text>')
    out.append(f'<text x="{pad}" y="10" class="tt">{html.escape(title)}</text>')
    out.append(f'<text x="{w - 6}" y="{h - 3}" class="ax" text-anchor="end">step</text>')
    out.append(f'<text x="{pad}" y="{h - 3}" class="ax">0</text>')
    for label, color, y in series:
        pts = " ".join(f"{X(i):.1f},{Y(float(v)):.1f}" for i, v in enumerate(y))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="1.6"><title>{html.escape(label)}</title></polyline>')
    out.append(f'<text x="2" y="{h / 2:.0f}" class="ax" '
               f'transform="rotate(-90 8,{h / 2:.0f})">{html.escape(ylab)}</text>')
    out.append("</svg>")
    return "".join(out)


def _rgb(x01):
    """[0,1] → RGB。三段线性插值,够读图用,不引 colormap 依赖。"""
    stops = np.array([[12, 24, 90], [40, 120, 200], [245, 220, 90], [200, 40, 30]],
                     dtype=np.float64)
    pos = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(x01, pos, stops[:, c]) for c in range(3)], axis=-1)


def _png_uri(rgb):
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat_uri(arr, vmax, scale=6):
    """(h,w) → 内联 PNG。蓝→黄→红,vmin 固定 0(份额本来就非负)。"""
    x = np.clip(np.asarray(arr, dtype=np.float64) / max(vmax, 1e-9), 0, 1)
    rgb = _rgb(x)
    return _png_uri(np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1))


def strip_uri(vec, vmax, cw=13, ch=20):
    """(n,) → 一条 n 格横向色带 PNG,per-head 用。色标与热力图共用同一套。

    vmax 由调用方跨变体统一给,否则每条自己归一化后"哪条更亮"就没意义了。
    """
    x = np.clip(np.asarray(vec, dtype=np.float64) / max(vmax, 1e-9), 0, 1)
    rgb = _rgb(x.reshape(1, -1))
    return _png_uri(np.repeat(np.repeat(rgb, ch, axis=0), cw, axis=1))


def img_uri(path, box=232):
    if not path or not os.path.exists(path):
        return None
    im = Image.open(path)
    im.load()
    im = im.convert("RGB")
    im.thumbnail((box, box), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ 报告

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--card:#fafafa}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e8e8e8;--mut:#9aa0a6;
--line:#2c3036;--card:#1c1f24}}
*{box-sizing:border-box}
body{margin:0;padding:26px;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:21px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 6px;
border-bottom:1px solid var(--line);padding-bottom:5px}
h3{font-size:13px;margin:18px 0 6px;color:var(--mut);font-weight:600;
text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mut);margin:0 0 18px}
.note{background:var(--card);border-left:3px solid #d9a13b;padding:10px 14px;
margin:16px 0;border-radius:0 5px 5px 0}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:12.5px;white-space:nowrap}
th,td{border:1px solid var(--line);padding:4px 9px;text-align:right}
th{background:var(--card);font-weight:600}td.l,th.l{text-align:left}
.pn{width:250px;height:150px;background:var(--card);border:1px solid var(--line);
border-radius:4px}
.gr{stroke:var(--line);stroke-width:1}
.ax{font-size:8px;fill:var(--mut)}.tt{font-size:9px;fill:var(--fg);font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
.strip{align-items:center;gap:10px;margin:3px 0}
.strip img{border:1px solid var(--line);border-radius:3px;image-rendering:pixelated;
max-width:100%}
.sl{width:150px;font-size:12px;text-align:right;color:var(--mut);flex:none}
.sn{font-size:11px;color:var(--mut)}
.leg{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 14px;font-size:12px}
.leg i{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:5px}
figure{margin:0;text-align:center}
figure img{display:block;border:1px solid var(--line);border-radius:4px;max-width:100%}
figcaption{font-size:11px;color:var(--mut);margin-top:3px}
code{background:var(--card);padding:1px 5px;border-radius:3px;font-size:12px}
"""


def per_head_overview(by_task, tasks):
    """D04 全局节:24 个头各自的 late-block ref 份额,跨样本平均。

    要回答的是一个**是非题**:全 head 平均之所以测不出主体丢没丢,是不是因为
    只有少数几个头在做身份绑定、它们的信号被另外二十来个头摊薄了?
    如果是——那些头上 pre 与 post4000 应当分得开,而平均值分不开。
    如果不是——24 个头看起来都差不多,那这条路走到头,该去查 K/V 内容而不是份额。
    """
    usable = [r for t in tasks for r in by_task[t].values() if r["curve_head"] is not None]
    if not usable:
        return ['<h2>per-head 分辨</h2>',
                '<p class="sub">当前 npz 是 D04 之前录的(curve 只有 3 维,head 已被平均掉),'
                '本节无数据。重跑 <code>run_attn_diag.py</code> 后即可。</p>']

    n_head = usable[0]["curve_head"].shape[2]
    # 每变体累计 (head,) 两条:ref 份额均值、per-head 失衡比
    acc = {v: {"share": [], "imb": []} for v in VARIANT_ORDER}
    for tid in tasks:
        for v in VARIANT_ORDER:
            r = by_task[tid].get(v)
            if r is None or r["curve_head"] is None:
                continue
            _, w_late = step_windows(r["curve_head"].shape[0])
            hs = head_shares(r["curve_head"], w_late)          # (head, seg)
            ridx = [i for i, n in enumerate(r["seg_names"]) if n.startswith("ref")]
            acc[v]["share"].append(np.asarray(hs, dtype=np.float64)[:, ridx].mean(axis=1))
            acc[v]["imb"].append(head_imbalance(hs, r["seg_names"]))
    got = {v: {k: np.mean(np.stack(a), axis=0) for k, a in d.items() if a}
           for v, d in acc.items() if d["share"]}

    P = ["<h2>per-head 分辨:少数几个头在做身份绑定吗?</h2>"]
    P.append('<p class="sub">窗口与总览表一致(late block flat 38-56 × timestep 后 1/3),'
             '唯一区别是 <b>不对 24 个头取平均</b>,并在 %d 个样本上取均值。<br>'
             '「ref 份额」= 该 head 对所有 ref 段的平均份额;'
             '「失衡比」= 该 head <b>自己</b>的 min(ref)/max(ref)——'
             '先平均再算失衡比会把"A 头只看 ref1、B 头只看 ref2"洗成"都很均衡"。</p>'
             % len(tasks))

    # 条带:一眼看份额是否集中在少数头上
    vmax = max(float(d["share"].max()) for d in got.values())
    P.append("<h3>ref 份额 · 每格一个 head(0→23,左起)· 跨变体共用色标</h3>")
    for v in VARIANT_ORDER:
        if v not in got:
            continue
        s = got[v]["share"]
        P.append(f"<div class='row strip'><div class='sl'>{html.escape(LABELS[v])}</div>"
                 f"<img src='{strip_uri(s, vmax)}' alt='{v} per-head share'>"
                 f"<div class='sn'>max head {int(np.argmax(s))} · "
                 f"{s.max():.4f} / 最低 {s.min():.4f} · 比值 {s.max() / max(s.min(), 1e-9):.1f}×"
                 f"</div></div>")

    base = "ours_kv_pre", "ours_kv_post4000"
    have_delta = all(b in got for b in base)
    P.append("<h3>逐 head 明细</h3>")
    if have_delta:
        P.append('<p class="sub">按 <b>Δ失衡比 = post4000 − pre</b> 的绝对值降序。'
                 '若真有"身份头",它们应当排在最上面,而且 Δ 明显大于末行的全 head 平均值。</p>')
    cols = [v for v in VARIANT_ORDER if v in got]
    th = ("<tr><th>head</th>"
          + "".join(f"<th>{html.escape(LABELS[v])}<br>ref份额</th>" for v in cols)
          + "".join(f"<th>{html.escape(LABELS[v])}<br>失衡比</th>" for v in cols)
          + ("<th>Δ失衡比<br>post−pre</th>" if have_delta else "") + "</tr>")
    order = (np.argsort(-np.abs(got[base[1]]["imb"] - got[base[0]]["imb"]))
             if have_delta else np.arange(n_head))
    body = []
    for h in order:
        d = (got[base[1]]["imb"][h] - got[base[0]]["imb"][h]) if have_delta else None
        body.append(
            f"<tr><td>{int(h)}</td>"
            + "".join(f"<td>{got[v]['share'][h]:.4f}</td>" for v in cols)
            + "".join(f"<td>{got[v]['imb'][h]:.3f}</td>" for v in cols)
            + (f"<td style='color:{'#3aa655' if d > 0 else '#d94e4e'}'>{d:+.3f}</td>"
               if have_delta else "") + "</tr>")
    # 末行:D04 之前报告看到的那个数,放这里做参照——per-head 的散布相对它有多大
    avg = ("<tr style='font-weight:700'><td>全 head 平均</td>"
           + "".join(f"<td>{got[v]['share'].mean():.4f}</td>" for v in cols)
           + "".join(f"<td>{got[v]['imb'].mean():.3f}</td>" for v in cols)
           + (f"<td>{got[base[1]]['imb'].mean() - got[base[0]]['imb'].mean():+.3f}</td>"
              if have_delta else "") + "</tr>")
    P.append(f'<div class="scroll"><table>{th}{"".join(body)}{avg}</table></div>')

    if have_delta:
        dl = np.abs(got[base[1]]["imb"] - got[base[0]]["imb"])
        P.append('<div class="note"><b>怎么读这张表</b>:最大的单头 Δ失衡比是 '
                 f'<b>{dl.max():+.3f}</b>(head {int(np.argmax(dl))}),全 head 平均后是 '
                 f'<b>{got[base[1]]["imb"].mean() - got[base[0]]["imb"].mean():+.3f}</b>。'
                 '若前者远大于后者,说明确实存在被平均摊薄的少数头,值得单独追;'
                 '若两者量级相当,那 pre/post 的差别就不在注意力份额这一层,'
                 '再往下拆 head 也不会有新东西——该转去查 K/V 内容。'
                 '<b>无论哪种,份额高仍然不等于身份被转移</b>(见页首红线)。</div>')
    return P


def build_html(runs, heat_block, heat_step):
    by_task = {}
    for r in runs:
        by_task.setdefault(r["task_id"], {})[r["variant"]] = r
    tasks = sorted(by_task)

    P = [f"<style>{CSS}</style>", "<h1>ref → 主图 注意力演化诊断</h1>"]
    n_ph = sum(1 for r in runs if r["curve_head"] is not None)
    P.append(f'<p class="sub">{len(tasks)} 个样本 · {len(runs)} 次录制 · '
             f'上游 D01/D02 + D03 修正 + D04 per-head · '
             f'{n_ph}/{len(runs)} 次带 head 维度</p>')
    P.append('<div class="note"><b>解读红线</b>(D01):注意力份额高 <b>≠</b> 身份信息被转移。'
             '身份保真可能主要由 K/V <b>内容</b>承载,份额只反映"看了多久",不反映"看到了什么"。'
             '以下所有图是<b>诊断线索,不是因果证据</b>;措辞请限于"注意力层面观察到…"。</div>')
    P.append('<div class="leg">' + "".join(
        f'<span><i style="background:{COLORS[v]}"></i>{html.escape(LABELS[v])}</span>'
        for v in VARIANT_ORDER) + "</div>")

    # ---------------- 总览表 ----------------
    P.append("<h2>总览:late block 上的 ref 份额</h2>")
    P.append('<p class="sub">late block = single 19-37(flat 38-56)。'
             '<b>失衡比</b> = min(ref份额) / max(ref份额):1.0 = 各 ref 同等被关注,'
             '→0 = 某个 ref 基本没被看。<br>'
             '<b>早/晚两列各算一次</b>(timestep 前 1/3 与后 1/3):这是 D02 的核心问题——'
             '早段就低 = <b>布局阶段没放置</b>;早段正常而晚段塌 = <b>细化阶段衰减</b>。'
             '两者的修复方向完全不同,所以不能把时间轴平均掉。</p>')
    max_ref = max(len([s for s in r["seg_names"] if s.startswith("ref")]) for r in runs)
    head = ("<tr><th class='l'>样本</th><th class='l'>subjects</th><th class='l'>变体</th>"
            + "".join(f"<th>ref{i + 1}</th>" for i in range(max_ref))
            + "<th>txt</th><th>img</th><th>失衡比<br>早 1/3</th>"
              "<th>失衡比<br>晚 1/3</th><th class='l'>形态</th></tr>")
    rows = []
    for tid in tasks:
        for v in VARIANT_ORDER:
            r = by_task[tid].get(v)
            if r is None:
                continue
            w_early, w_late = step_windows(r["curve"].shape[0])
            sh_e, sh_l = (seg_shares(r["curve"], r["seg_names"], w)
                          for w in (w_early, w_late))
            b_e, b_l = imbalance(sh_e), imbalance(sh_l)
            refs = [sh_l[n] for n in r["seg_names"] if n.startswith("ref")]
            cells = "".join(f"<td>{refs[i]:.4f}</td>" if i < len(refs) else "<td>–</td>"
                            for i in range(max_ref))

            if b_e < 0.5 and b_l < 0.5:
                shape, col = "布局阶段就没放置", "#d94e4e"
            elif b_l < 0.5:
                shape, col = "细化阶段衰减", "#d98a2b"
            elif b_e < 0.5:
                shape, col = "早段低但后期追回", "#2f7ed8"
            else:
                shape, col = "均衡", "var(--mut)"
            fl = lambda b: (" style='color:#d94e4e;font-weight:700'" if b < 0.5 else "")
            rows.append(
                f"<tr><td class='l'>{html.escape(tid)}</td>"
                f"<td class='l'>{html.escape('+'.join(r['subjects']))}</td>"
                f"<td class='l'>{html.escape(LABELS[v])}</td>{cells}"
                f"<td>{sh_l['txt']:.4f}</td><td>{sh_l['img']:.4f}</td>"
                f"<td{fl(b_e)}>{b_e:.3f}</td><td{fl(b_l)}>{b_l:.3f}</td>"
                f"<td class='l' style='color:{col}'>{shape}</td></tr>")
    P.append(f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>')

    P.extend(per_head_overview(by_task, tasks))

    # ---------------- 每样本 ----------------
    for tid in tasks:
        got = by_task[tid]
        any_r = next(iter(got.values()))
        P.append(f"<h2>{html.escape(tid)} — {html.escape('+'.join(any_r['subjects']))}</h2>")
        P.append(f'<p class="sub"><code>{html.escape(any_r["prompt"])}</code> · '
                 f'seed {any_r["seed"]}</p>')

        P.append("<h3>出图</h3><div class='row'>")
        for v in VARIANT_ORDER:
            r = got.get(v)
            uri = img_uri(r["png"]) if r else None
            if uri:
                P.append(f"<figure><img src='{uri}' alt='{html.escape(v)}'>"
                         f"<figcaption>{html.escape(LABELS[v])}</figcaption></figure>")
        P.append("</div>")

        P.append("<h3>份额曲线 · 行 = block 组,列 = ref 段</h3>")
        n_ref = len([s for s in any_r["seg_names"] if s.startswith("ref")])
        for gname, g0, g1 in GROUPS:
            P.append("<div class='row'>")
            for j in range(n_ref):
                seg = f"ref{j + 1}"
                series = []
                for v in VARIANT_ORDER:
                    r = got.get(v)
                    if r is None or seg not in r["seg_names"]:
                        continue
                    si = r["seg_names"].index(seg)
                    series.append((LABELS[v], COLORS[v],
                                   r["curve"][:, g0:g1, si].mean(axis=1)))
                P.append(svg_panel(series, f"{gname} → {seg}", "share"))
            P.append("</div>")

        if any(r["curve_head"] is not None for r in got.values()):
            P.append("<h3>per-head · late block × 后 1/3 step,每格一个 head(0→23)</h3>")
            P.append('<p class="sub">同一 ref 段内 4 条共用色标。'
                     '若某个变体"丢了主体"是注意力层面的事,它那条在对应 ref 上应当整体偏暗;'
                     '若只是个别头暗,那丢失就不在份额这一层。</p>')
            for j in range(n_ref):
                seg = f"ref{j + 1}"
                vecs = []
                for v in VARIANT_ORDER:
                    r = got.get(v)
                    if r is None or r["curve_head"] is None or seg not in r["seg_names"]:
                        continue
                    _, w_late = step_windows(r["curve_head"].shape[0])
                    hs = head_shares(r["curve_head"], w_late)
                    vecs.append((v, np.asarray(hs)[:, r["seg_names"].index(seg)]))
                if not vecs:
                    continue
                vmax = max(float(x.max()) for _, x in vecs)
                P.append(f"<div class='row strip'><div class='sl'>{seg} · "
                         f"max {vmax:.4f}</div></div>")
                for v, x in vecs:
                    P.append(f"<div class='row strip'><div class='sl'>"
                             f"{html.escape(LABELS[v])}</div>"
                             f"<img src='{strip_uri(x, vmax)}' alt='{seg} {v}'>"
                             f"<div class='sn'>head {int(np.argmax(x))} 最高 "
                             f"{x.max():.4f}</div></div>")

        key = (heat_step, heat_block)
        if any(key in r["spatial"] for r in got.values()):
            P.append(f"<h3>热力图 · block {heat_block} · step {heat_step}"
                     f" — 主图哪些区域在看该 ref</h3>")
            P.append('<p class="sub">同一行内共用色标上限,所以格间明暗可直接比。'
                     '<b>丢失</b> = ref2 图整体暗;<b>复制</b> = ref1 图出现两处热区。</p>')
            for j in range(n_ref):
                seg = f"ref{j + 1}"
                mats = []
                for v in VARIANT_ORDER:
                    r = got.get(v)
                    if r is None or key not in r["spatial"] or seg not in r["seg_names"]:
                        continue
                    mats.append((v, r["spatial"][key][:, :, r["seg_names"].index(seg)]))
                if not mats:
                    continue
                vmax = max(float(m.max()) for _, m in mats)
                P.append(f"<div class='row'><div style='width:96px;padding-top:70px;"
                         f"font-size:12px'>{seg}<br><span style='color:var(--mut)'>"
                         f"max {vmax:.4f}</span></div>")
                for v, m in mats:
                    P.append(f"<figure><img src='{heat_uri(m, vmax)}' alt='{seg} {v}'>"
                             f"<figcaption>{html.escape(LABELS[v])}</figcaption></figure>")
                P.append("</div>")

        # kv 变体应当只有 step 0 带 ref token,这条是 sampling.py:239 的实证
        for v in ("ours_kv_pre", "ours_kv_post4000"):
            r = got.get(v)
            if r and len(set(r["q_len_by_step"])) > 1:
                first, rest = r["q_len_by_step"][0], set(r["q_len_by_step"][1:])
                P.append(f'<p class="sub">{html.escape(LABELS[v])}:step0 query 长 {first},'
                         f'其余步 {sorted(rest)} —— 与 write/read 切换一致。</p>')
    return "\n".join(P)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_path", default="output/attn_diag")
    p.add_argument("--out", default=None, help="默认 <save_path>/report.html")
    p.add_argument("--heat_block", type=int, default=52)
    p.add_argument("--heat_step", type=int, default=12)
    p.add_argument("--title", default="注意力演化诊断")
    args = p.parse_args()

    runs = load_runs(args.save_path)
    if not runs:
        raise SystemExit(f"❌ {args.save_path} 下没有 *.npz——先在 H800 上跑 run_attn_diag.py")

    avail = sorted({k for r in runs for k in r["spatial"]})
    if (args.heat_step, args.heat_block) not in avail:
        print(f"⚠️  (step {args.heat_step}, block {args.heat_block}) 没录到热力图。"
              f"可用的 (step, block):{avail}")
        if avail:
            args.heat_step, args.heat_block = avail[len(avail) // 2]
            print(f"   改用 (step {args.heat_step}, block {args.heat_block})")

    body = build_html(runs, args.heat_block, args.heat_step)
    doc = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(args.title)}</title></head><body>{body}</body></html>")
    out = args.out or os.path.join(args.save_path, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"✓ {len(runs)} 次录制 → {out}  ({os.path.getsize(out) / 1024:.0f} KB)")

    # 顺手把总览表也存成 json,方便后续拿去和人工判定对齐
    summary = []
    for r in runs:
        w_early, w_late = step_windows(r["curve"].shape[0])
        sh_e = seg_shares(r["curve"], r["seg_names"], w_early)
        sh_l = seg_shares(r["curve"], r["seg_names"], w_late)
        row = {
            "task_id": r["task_id"], "variant": r["variant"], "subjects": r["subjects"],
            "late_block_ref_share_early_steps":
                [sh_e[n] for n in r["seg_names"] if n.startswith("ref")],
            "late_block_ref_share_late_steps":
                [sh_l[n] for n in r["seg_names"] if n.startswith("ref")],
            "imbalance_early_steps": imbalance(sh_e),
            "imbalance_late_steps": imbalance(sh_l),
        }
        if r["curve_head"] is not None:
            hs = head_shares(r["curve_head"], w_late)
            ridx = [i for i, n in enumerate(r["seg_names"]) if n.startswith("ref")]
            # 逐 head 落进 json,方便不开报告就直接做后续统计
            row["per_head_ref_share_late_steps"] = (
                np.asarray(hs, dtype=np.float64)[:, ridx].round(6).tolist())
            row["per_head_imbalance_late_steps"] = (
                head_imbalance(hs, r["seg_names"]).round(6).tolist())
        summary.append(row)
    sp = os.path.join(args.save_path, "summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ 总览 → {sp}")


if __name__ == "__main__":
    main()
