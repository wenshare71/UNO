#!/usr/bin/env python3
"""盲评工具的行为测试(§11.3 步骤 0 的验收产物之一)。

**为什么这个文件必须存在**:盲评服务器是 M5 的判读工具,它出错的方式是**静默的**——
槽位算错、清单漏检,产出的标注看上去完全正常,只有事后重算才可能发现。
M4 已经这样栽过一次(T/S 错位,score 0.92→0.82,靠重算才捞回来)。

要证明的七件事,每一件都是跑出来对,不是"读代码看着对":
  1. 槽位跨进程稳定       —— 换 PYTHONHASHSEED 起子进程,结果必须逐条相同
  2. 槽位无系统偏向       —— 198 条上左右大致各半
  3. 重放确实独立         —— 重放对的槽位与原对无关(约一半翻面)
  4. 清单自检真的会炸     —— 重复 id / 缺字段 / 同标签 / 缺图 / 路径逃逸,全部必须被抓到
  5. 统计只读语义胜者     —— 把 choice 全改乱而 winner 不变,数字必须一个不变
  6. 迁移复算五行吻合     —— 新代码路径独立重现 §6.3 的表
  7. 换批次则图片 URL 变  —— 否则浏览器拿上一批的缓存图充数(2026-08-03 实际发生过)

    python distill/blind_eval/test_blind.py

HTTP 层的测试需要 FastAPI,本地通常没有;缺了会**明确跳过并提示**,不会假装通过。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from PIL import Image  # noqa: E402

from distill.blind_eval.pairing import (  # noqa: E402
    asset_tag, check_manifest, key0_on_left, slot_of,
)
from distill.blind_eval.report import (  # noqa: E402
    full_report, replay_agreement, wilson,
)

_fail: list[str] = []
_skip: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _fail.append(name)


def skip(name: str, why: str) -> None:
    print(f"  ⊘ {name}  — 跳过:{why}")
    _skip.append(name)


# ------------------------------------------------------------------ 夹具

def tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path, format="PNG")


def fixture_manifest(root: Path, n: int = 4) -> dict:
    pairs = []
    for i in range(n):
        for rel in (f"img/a{i}.png", f"img/b{i}.png", f"img/r{i}.png"):
            tiny_png(root / rel)
        pairs.append({
            "pair_id": f"t::p::{i}", "kind": "post_vs_pre", "stratum": "S1",
            "prompt": "p", "ref_paths": [f"img/r{i}.png"],
            "key_0": "focus", "img_0": f"img/a{i}.png",
            "key_1": "base", "img_1": f"img/b{i}.png",
            "src_task_id": f"T{i}",
        })
    return {"meta": {"batch_id": "t", "blind_seed": "seed-x"}, "pairs": pairs}


# ------------------------------------------------------------------ 1. 槽位

print("\n[1] 槽位分配")

R2 = _REPO / "output/eval_multiref/pairs_m5r1.json"
M4P = _REPO / "output/eval_multiref/pairs_m4r1.json"
M4M = _REPO / "output/eval_multiref/blind_annotations_m4r1.json"
have_real = R2.exists() and M4P.exists() and M4M.exists()

if not have_real:
    skip("真实清单相关的检查", "先跑 build_pairs.py migrate-m4 && build_pairs.py r2")
else:
    man = json.loads(R2.read_text(encoding="utf-8"))
    pairs, seed = man["pairs"], man["meta"]["blind_seed"]

    # 跨进程稳定:内置 hash 会被 PYTHONHASHSEED 随机化,md5 不会。
    # 这条如果坏了,重启一次服务器全部槽位就变,已标注的数据当场作废。
    ids = [p["pair_id"] for p in pairs[:40]]
    code = ("import sys,json;sys.path.insert(0,%r);"
            "from distill.blind_eval.pairing import key0_on_left as f;"
            "print(json.dumps([f(i,%r) for i in json.loads(sys.argv[1])]))"
            % (str(_REPO), seed))
    outs = []
    for hs in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", code, json.dumps(ids)],
                           capture_output=True, text=True, env=env)
        outs.append(r.stdout.strip())
    local = json.dumps([key0_on_left(i, seed) for i in ids])
    check("跨 PYTHONHASHSEED 逐条相同", len({*outs, local}) == 1,
          f"3 个子进程 + 本进程共 {len({*outs, local})} 种结果")

    n_left = sum(key0_on_left(p["pair_id"], seed) for p in pairs)
    frac = n_left / len(pairs)
    check("槽位无系统偏向", 0.40 <= frac <= 0.60,
          f"key_0 在左 {n_left}/{len(pairs)} = {frac:.1%}")

    p0 = pairs[0]
    kl, il = slot_of(p0, seed, "L")
    kr, ir = slot_of(p0, seed, "R")
    check("L/R 取到不同候选且覆盖两者",
          kl != kr and {kl, kr} == {p0["key_0"], p0["key_1"]}
          and {il, ir} == {p0["img_0"], p0["img_1"]}, f"L={kl} R={kr}")

    # 重放独立性:重放对与原对是不同的 pair_id + 不同的盲种,槽位应当无关。
    # 若相关,重放测的就不是"重新判一次",而是"记不记得上次点了哪边"。
    rp = [p for p in pairs if p["kind"] == "replay"]
    m4_seed = json.loads(M4P.read_text(encoding="utf-8"))["meta"]["blind_seed"]
    same = sum(key0_on_left(p["pair_id"], seed) ==
               key0_on_left(p["replay_of"], m4_seed) for p in rp)
    check("重放槽位与原对无关", len(rp) > 0 and 0.2 <= same / len(rp) <= 0.8,
          f"{same}/{len(rp)} 条槽位与 M4 时相同(应≈一半)")

    # 回归:M5 首次上机时 `/api/img?k=ref:0:0` 与 M4 完全同名,浏览器把 M4 第 0 对的
    # 参考图(书包)喂给了 M5 第 0 对(闹钟)。修法是 URL 带清单内容摘要——
    # **两批清单的图片 URL 空间必须不相交**,这一条就是在断言那件事。
    m4_text = M4P.read_text(encoding="utf-8")
    m5_text = R2.read_text(encoding="utf-8")
    check("换批次则图片 URL 版本戳必变",
          asset_tag(m4_text) != asset_tag(m5_text),
          f"m4={asset_tag(m4_text)} m5={asset_tag(m5_text)}")
    check("同一份清单版本戳稳定", asset_tag(m5_text) == asset_tag(m5_text))
    # 内容改一个字节就必须换戳:batch_id 不变但清单重新生成过,正是缓存再次骗人的场景
    check("清单内容变则版本戳变",
          asset_tag(m5_text) != asset_tag(m5_text.replace("m5r1::pvp", "m5r1::pvpX", 1)))


# ------------------------------------------------------------------ 2. 清单自检

print("\n[2] 清单自检必须抓到的错")

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    good = fixture_manifest(root)
    check("干净的清单能过", check_manifest(good, root) == [],
          str(check_manifest(good, root))[:80])

    cases = {}

    m = json.loads(json.dumps(good)); m["pairs"][1]["pair_id"] = m["pairs"][0]["pair_id"]
    cases["重复 pair_id"] = m

    m = json.loads(json.dumps(good)); del m["pairs"][0]["key_1"]
    cases["缺字段"] = m

    m = json.loads(json.dumps(good)); m["pairs"][0]["key_1"] = m["pairs"][0]["key_0"]
    cases["两候选标签相同"] = m

    m = json.loads(json.dumps(good)); m["pairs"][0]["img_0"] = "img/nope.png"
    cases["图不存在"] = m

    m = json.loads(json.dumps(good)); m["pairs"][0]["img_0"] = "../../etc/passwd"
    cases["路径逃出仓库"] = m

    m = json.loads(json.dumps(good)); del m["meta"]["blind_seed"]
    cases["缺 blind_seed"] = m

    m = json.loads(json.dumps(good)); m["pairs"] = []
    cases["空清单"] = m

    # 截断图:Image.open 只读文件头会放过,必须 load() 才炸
    m = json.loads(json.dumps(good))
    trunc = root / "img/trunc.png"
    data = (root / "img/a0.png").read_bytes()
    trunc.write_bytes(data[: len(data) // 2])
    m["pairs"][0]["img_0"] = "img/trunc.png"
    cases["截断图(文件头完好)"] = m

    for name, bad in cases.items():
        errs = check_manifest(bad, root)
        check(f"抓到:{name}", len(errs) > 0, (errs[0][:60] if errs else "**没抓到**"))


# ------------------------------------------------------------------ 3. 统计

print("\n[3] 统计口径")

# Wilson 区间对照公开数值(Brown/Cai/DasGupta 常用算例),不是自己跟自己比
for k, n, lo, hi in [(1, 10, 0.0179, 0.4041), (0, 10, 0.0, 0.2775),
                     (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634)]:
    w = wilson(k, n)
    ok = w is not None and abs(w[0] - lo) < 5e-4 and abs(w[1] - hi) < 5e-4
    check(f"Wilson({k}/{n})", ok, f"得 {w[0]:.4f},{w[1]:.4f} / 期望 {lo},{hi}")
check("Wilson(n=0) 返回 None 而不是 0", wilson(0, 0) is None)

if have_real:
    man4 = json.loads(M4P.read_text(encoding="utf-8"))
    marks4 = json.loads(M4M.read_text(encoding="utf-8"))["marks"]
    rep = full_report(man4, marks4)

    want = {"总体": (82, 50, 95, 227, 0.8192090395480226),
            "S1": (55, 32, 45, 132, 0.77),
            "S2": (3, 6, 6, 15, 1.3333333333333333),
            "S3": (20, 11, 29, 60, 0.8163265306122449),
            "S4": (4, 1, 15, 20, 0.8421052631578947)}
    got = {"总体": rep["by_kind"]["teacher_vs_student"]}
    for st in ("S1", "S2", "S3", "S4"):
        got[st] = rep["by_kind_stratum"][f"teacher_vs_student/{st}"]
    for name, (T, S, B, n, score) in want.items():
        g = got[name]
        ok = (g["win_1"] == T and g["win_0"] == S and g["tie"] == B and g["n"] == n
              and abs(g["legacy_score"] - score) < 1e-12)
        check(f"迁移复算 {name}", ok,
              f"T={g['win_1']} S={g['win_0']} B={g['tie']} n={g['n']} "
              f"score={g['legacy_score']:.4f}")

    # 统计必须只看 winner。把 choice 全打乱而 winner 不动,任何数字都不许变。
    scrambled = {k: {**v, "choice": "tie"} for k, v in marks4.items()}
    rep2 = full_report(man4, scrambled)
    same = (rep2["by_kind"]["teacher_vs_student"]["win_0"]
            == rep["by_kind"]["teacher_vs_student"]["win_0"]
            and rep2["by_kind"]["teacher_vs_student"]["legacy_score"]
            == rep["by_kind"]["teacher_vs_student"]["legacy_score"])
    check("打乱 choice 不影响胜负统计(只读 winner)", same)
    moved = rep2["position_overall"]["L"] != rep["position_overall"]["L"]
    check("但左右偏好统计确实读 choice", moved,
          f"L: {rep['position_overall']['L']} → {rep2['position_overall']['L']}")

# 重放一致率的小造例:2 同 1 异 1 未标
rp_pairs = [{"pair_id": f"n{i}", "kind": "replay", "replay_of": f"o{i}",
             "src_task_id": f"T{i}"} for i in range(4)]
now = {"n0": {"winner": "a"}, "n1": {"winner": "b"}, "n2": {"winner": "a"}}
prev = {"o0": {"winner": "a"}, "o1": {"winner": "b"}, "o2": {"winner": "b"},
        "o3": {"winner": "a"}}
ra = replay_agreement(rp_pairs, now, prev)
check("重放一致率", ra["same"] == 2 and ra["diff"] == 1 and ra["unannotated"] == 1
      and abs(ra["agreement"] - 2 / 3) < 1e-12,
      f"same={ra['same']} diff={ra['diff']} 未标={ra['unannotated']}")


# ------------------------------------------------------------------ 4. HTTP 层

print("\n[4] HTTP 层(需要 FastAPI)")
try:
    from fastapi.testclient import TestClient  # noqa: F401
    _has_fastapi = True
except Exception as e:  # noqa: BLE001
    _has_fastapi = False
    skip("HTTP 接口测试", f"{type(e).__name__} — 在 H800 上跑本文件即可覆盖")

if _has_fastapi:
    from fastapi.testclient import TestClient

    from distill.blind_eval.server import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man = fixture_manifest(root, n=6)
        mp = root / "pairs.json"
        mp.write_text(json.dumps(man), encoding="utf-8")
        app = create_app(mp, root / "marks.json", root)
        c = TestClient(app)

        rows = c.get("/api/tasks").json()["tasks"]
        blob = json.dumps(rows, ensure_ascii=False)
        leaks = [w for w in ("pair_id", "kind", "key_0", "key_1", "img_0", "img_1",
                             "focus", "base", "post_vs_pre", ".png") if w in blob]
        check("/api/tasks 不泄漏身份", not leaks, f"泄漏 {leaks}" if leaks else "")

        c.post("/api/mark", json={"idx": 0, "choice": "L"})
        saved = json.loads((root / "marks.json").read_text(encoding="utf-8"))["marks"]
        pid = man["pairs"][0]["pair_id"]
        exp = slot_of(man["pairs"][0], man["meta"]["blind_seed"], "L")[0]
        check("标注按 pair_id 键控且解出正确胜者",
              list(saved) == [pid] and saved[pid]["winner"] == exp,
              f"{list(saved)} winner={saved.get(pid, {}).get('winner')}")

        # 图片 URL 的版本戳:少了它或戳过期,必须拒绝而不是返回一张图。
        # 返回图 = 静默给出别批次的参考图,这正是 M5 首次上机踩的坑。
        tag = c.get("/api/tasks").json()["asset_tag"]
        check("/api/tasks 下发版本戳", bool(tag), f"tag={tag!r}")
        check("带正确版本戳能取到图",
              c.get(f"/api/img?v={tag}&k=ref:0:0").status_code == 200)
        check("版本戳过期被拒(409)",
              c.get("/api/img?v=deadbeef&k=ref:0:0").status_code == 409)
        check("缺版本戳被拒", c.get("/api/img?k=ref:0:0").status_code == 422)
        r_img = c.get(f"/api/img?v={tag}&k=ref:0:0")
        check("图片响应显式声明缓存策略",
              "max-age" in r_img.headers.get("cache-control", ""),
              r_img.headers.get("cache-control", "(无)"))

        check("非法 choice 被拒",
              c.post("/api/mark", json={"idx": 0, "choice": "A"}).status_code == 400)
        check("越界 idx 被拒",
              c.post("/api/mark", json={"idx": 999, "choice": "L"}).status_code == 400)
        check("未揭盲的 stats 不含统计",
              set(c.get("/api/stats").json()) == {"n_total", "n_annotated", "complete"})
        check("揭盲的 stats 才有 by_kind", "by_kind" in c.get("/api/stats?reveal=1").json())

        # 缺图必须启动即炸,而不是评到一半才发现
        bad = json.loads(json.dumps(man))
        bad["pairs"][0]["img_0"] = "img/gone.png"
        bp = root / "bad.json"
        bp.write_text(json.dumps(bad), encoding="utf-8")
        try:
            create_app(bp, root / "m2.json", root)
            check("缺图时启动即炸", False, "居然起来了")
        except SystemExit:
            check("缺图时启动即炸", True)


# ------------------------------------------------------------------

print("\n" + "=" * 62)
if _skip:
    print(f"⊘ 跳过 {len(_skip)} 项:{_skip}")
if _fail:
    print(f"❌ {len(_fail)} 项失败:{_fail}")
    sys.exit(1)
print("✓ 全部通过" + ("(含跳过项,上机后请重跑本文件)" if _skip else ""))
