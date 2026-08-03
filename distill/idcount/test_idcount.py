#!/usr/bin/env python3
"""身份留存计数工具的行为测试。

跑法(两种都行,效果相同):
    python distill/idcount/test_idcount.py
    python -m distill.idcount.test_idcount

要证明的六件事:
  1. 抽样确定性       —— 同种子跑两次,输出逐字节相同;层配比 21/9;66 个 item;
                        重放 6 个且每个都能对回原 item(字段完全一致)
  2. 版本戳换清单必变   —— 否则浏览器会拿上一批的缓存图(抄 blind_eval 踩过的坑)
  3. /api/items 不泄漏  —— 这是本工具存在的意义:不含 variant / replay_of,
                        也不含 img_path / image_paths(文件名里嵌着 variant,原样
                        下发等于换个地方泄漏,这条比前一条更容易被漏掉)
  4. /api/img 版本戳校验 —— 对得上 200、过期 409、缺失 422
  5. /api/mark 长度校验 —— answers 长度必须等于该 item 的参考图数,否则 400
  6. report 算术        —— 手算一组已知答案的留存率和 Wilson 区间,断言一致

HTTP 层测试需要 FastAPI,本地没有会明确跳过,不会假装通过。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from PIL import Image  # noqa: E402

from distill.blind_eval.pairing import asset_tag  # noqa: E402
from distill.idcount.build_items import (  # noqa: E402
    N_S1, N_S3, SEED, TASKS_JSON, build_manifest, check_items,
)
from distill.idcount.report import full_report, image_tally, subject_tally  # noqa: E402
from distill.blind_eval.report import wilson  # noqa: E402  与 report.py 用的是同一份实现

_fail: list[str] = []
_skip: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _fail.append(name)


def skip(name: str, why: str) -> None:
    print(f"  ⊘ {name}  — 跳过:{why}")
    _skip.append(name)


def tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path, format="PNG")


def fixture_manifest(root: Path, n: int = 6) -> dict:
    """HTTP 层测试用的小清单:字段结构照抄 build_items.build_manifest 的输出形状,
    但图是本地现造的 8×8 纯色 PNG——不依赖 output/probe_iso 里那 384 张只在 H800 上
    才有的真实生成图(见 build_items.py --verify 的已知本地限制)。"""
    items = []
    for i in range(n):
        tiny_png(root / f"img/gen{i}.png")
        tiny_png(root / f"img/ref{i}_0.png")
        tiny_png(root / f"img/ref{i}_1.png")
        items.append({
            "item_id": f"IC_{i:04d}",
            "task_id": f"T{i}",
            "variant": "official_full" if i % 2 == 0 else "official_iso",
            "stratum": "S1",
            "prompt": f"prompt {i}",
            "image_paths": [f"img/ref{i}_0.png", f"img/ref{i}_1.png"],
            "ref_names": [f"subj{i}a", f"subj{i}b"],
            "img_path": f"img/gen{i}.png",
            "replay_of": None,
        })
    return {"meta": {"spec": "M5-idcount-v1", "seed": SEED, "n_tasks": n, "n_items": n,
                     "n_replay": 0, "n_tasks_by_stratum": {"S1": n}},
            "items": items}


# ------------------------------------------------------------------ 1. 抽样

print("\n[1] 抽样确定性")

have_tasks = TASKS_JSON.exists()
if not have_tasks:
    skip("抽样相关的检查", f"{TASKS_JSON} 不存在")
else:
    m1 = build_manifest()
    m2 = build_manifest()
    check("同种子跑两次,输出逐字节相同",
          json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True))

    meta = m1["meta"]
    check("层配比 21/9", meta["n_tasks_by_stratum"] == {"S1": N_S1, "S3": N_S3},
          str(meta["n_tasks_by_stratum"]))
    check("item 总数 66", meta["n_items"] == 66, str(meta["n_items"]))
    check("重放 6 个", meta["n_replay"] == 6, str(meta["n_replay"]))

    items = m1["items"]
    by_id = {it["item_id"]: it for it in items}
    replay_items = [it for it in items if it["replay_of"]]
    check("重放 item 数与 meta 一致", len(replay_items) == 6, str(len(replay_items)))
    all_matched = True
    for it in replay_items:
        src = by_id.get(it["replay_of"])
        if src is None:
            all_matched = False
            continue
        for key in ("task_id", "variant", "stratum", "prompt", "image_paths",
                    "ref_names", "img_path"):
            if it[key] != src[key]:
                all_matched = False
    check("每个重放 item 都能对回原 item(字段完全一致)", all_matched)

    # check_items 是 --verify 用的同一份纯谓词:剔掉需要真实生成图的那条检查后单独测逻辑本身。
    errs = check_items(m1, _REPO)
    non_file_errs = [e for e in errs if "缺失或不可解码" not in e]
    check("check_items 对刚生成的清单只报缺图(结构性检查全过)",
          non_file_errs == [], str(non_file_errs)[:200])


# ------------------------------------------------------------------ 2. 版本戳

print("\n[2] 图片 URL 版本戳")

t1 = json.dumps({"items": [{"a": 1}]})
t2 = json.dumps({"items": [{"a": 2}]})
check("换清单则版本戳必变", asset_tag(t1) != asset_tag(t2))
check("同一份清单版本戳稳定", asset_tag(t1) == asset_tag(t1))


# ------------------------------------------------------------------ 3/4/5. HTTP 层

print("\n[3-5] HTTP 层(需要 FastAPI)")
try:
    from fastapi.testclient import TestClient  # noqa: F401
    _has_fastapi = True
except Exception as e:  # noqa: BLE001
    _has_fastapi = False
    skip("HTTP 接口测试", f"{type(e).__name__}(本地未装 fastapi,上机后请重跑本文件)")

if _has_fastapi:
    from fastapi.testclient import TestClient

    from distill.idcount.server import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man = fixture_manifest(root, n=6)
        ip = root / "items.json"
        ip.write_text(json.dumps(man), encoding="utf-8")
        app = create_app(ip, root / "marks.json", root)
        c = TestClient(app)

        data = c.get("/api/items").json()
        rows = data["items"]
        blob = json.dumps(rows, ensure_ascii=False)
        # variant / replay_of 按字段名查;img_path / image_paths 按"文件名里嵌着 variant
        # 的具体字符串"查——只查字段名会漏掉"字段删了但值以别的方式混进去"这种泄漏。
        leaks = [w for w in ("variant", "replay_of", "img_path", "image_paths",
                             "official_full", "official_iso", ".png", "gen0", "ref0_0")
                if w in blob]
        check("/api/items 不泄漏 variant/replay_of/图片路径", not leaks,
              f"泄漏 {leaks}" if leaks else "")
        check("/api/items 保留标注需要的字段",
              set(rows[0]) == {"item_id", "task_id", "stratum", "prompt", "ref_names", "n_refs"},
              str(set(rows[0])))

        tag = data["tag"]
        check("带正确版本戳能取到生成图",
              c.get(f"/api/img?v={tag}&k=gen:0").status_code == 200)
        check("带正确版本戳能取到参考图",
              c.get(f"/api/img?v={tag}&k=ref:0:0").status_code == 200)
        check("版本戳过期被拒(409)",
              c.get("/api/img?v=deadbeef&k=gen:0").status_code == 409)
        check("缺版本戳被拒(422)", c.get("/api/img?k=gen:0").status_code == 422)
        r_img = c.get(f"/api/img?v={tag}&k=gen:0")
        check("图片响应显式声明缓存策略",
              "max-age" in r_img.headers.get("cache-control", ""),
              r_img.headers.get("cache-control", "(无)"))

        item0_id = man["items"][0]["item_id"]
        ok_mark = c.post("/api/mark", json={"item_id": item0_id,
                                            "answers": [True, False], "dwell_ms": 1200})
        check("正确长度的 answers 被接受", ok_mark.status_code == 200, str(ok_mark.status_code))
        saved = json.loads((root / "marks.json").read_text(encoding="utf-8"))["marks"]
        check("标注按 item_id 键控且原样落盘",
              saved.get(item0_id, {}).get("answers") == [True, False])

        bad_mark = c.post("/api/mark", json={"item_id": item0_id,
                                             "answers": [True], "dwell_ms": 500})
        check("answers 长度不符被拒(400)", bad_mark.status_code == 400,
              str(bad_mark.status_code))

        check("未知 item_id 被拒",
              c.post("/api/mark", json={"item_id": "nope", "answers": [True, False]})
              .status_code == 404)


# ------------------------------------------------------------------ 6. report 算术

print("\n[6] report 算术")

# 手算一个 4-item(每个 2 个 subject)的小造例。
# item0: [T, T] → 2/2 主体保住,整图保住
# item1: [T, F] → 1/2 主体保住,整图不算(不是全 T)
# item2: [F, F] → 0/2
# item3: 未标注(不进 n)
report_items = [
    {"item_id": "a", "task_id": "TA", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
    {"item_id": "b", "task_id": "TB", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
    {"item_id": "c", "task_id": "TC", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
    {"item_id": "d", "task_id": "TD", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
]
report_marks = {
    "a": {"answers": [True, True]},
    "b": {"answers": [True, False]},
    "c": {"answers": [False, False]},
}
st = subject_tally(report_items, report_marks)
check("per-subject: k/n = 3/6", st["k"] == 3 and st["n"] == 6, f"k={st['k']} n={st['n']}")
check("per-subject: rate = 0.5", abs(st["rate"] - 0.5) < 1e-12)
w_expect = wilson(3, 6)
check("per-subject: Wilson CI 与直接调用一致", st["wilson95"] == w_expect)

it_ = image_tally(report_items, report_marks)
check("per-image: k/n = 1/3(只有 item a 全 True)", it_["k"] == 1 and it_["n"] == 3,
      f"k={it_['k']} n={it_['n']}")
check("per-image: rate = 1/3", abs(it_["rate"] - 1 / 3) < 1e-12)

# 重放自洽率造例:同 1、异 1、未标 1
rep_manifest = {"items": [
    {"item_id": "a", "task_id": "TA", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
    {"item_id": "b", "task_id": "TB", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
    {"item_id": "c", "task_id": "TC", "variant": "official_full", "stratum": "S1",
     "replay_of": None},
    {"item_id": "a2", "task_id": "TA", "variant": "official_full", "stratum": "S1",
     "replay_of": "a"},
    {"item_id": "b2", "task_id": "TB", "variant": "official_full", "stratum": "S1",
     "replay_of": "b"},
    {"item_id": "c2", "task_id": "TC", "variant": "official_full", "stratum": "S1",
     "replay_of": "c"},
]}
rep_marks = {
    "a": {"answers": [True, True]}, "a2": {"answers": [True, True]},     # 一致
    "b": {"answers": [True, False]}, "b2": {"answers": [False, False]},  # 不一致
    "c": {"answers": [False, False]},                                    # c2 未标 → missing
}
rep = full_report(rep_manifest, rep_marks)
r = rep["replay"]
check("重放自洽率:1 同 1 异 1 未标", r["same"] == 1 and r["diff"] == 1 and r["missing"] == 1,
      f"same={r['same']} diff={r['diff']} missing={r['missing']}")
check("重放自洽率数值 = 1/2", abs(r["agreement"] - 0.5) < 1e-12)
check("重放主统计已剔除(n_main = 3,不含 a2/b2/c2)", rep["n_main"] == 3, str(rep["n_main"]))


# ------------------------------------------------------------------

print("\n" + "=" * 62)
if _skip:
    print(f"⊘ 跳过 {len(_skip)} 项:{_skip}")
if _fail:
    print(f"❌ {len(_fail)} 项失败:{_fail}")
    sys.exit(1)
print("✓ 全部通过" + ("(含跳过项,上机后请重跑本文件)" if _skip else ""))
