#!/usr/bin/env python3
"""人工盲评服务:**配对清单驱动**的双图 A/B 盲评。

与 M4 版(2026-07-30)的区别,以及为什么要改:

  v1 把"比什么"写死成两个模块级常量 `TEACHER` / `STUDENT`,于是
  ① 每换一种比较就得改服务器——而服务器是判读工具,改它就是改尺子;
  ② 标注按**列表下标**键控,任务单一增删就整体偏移(`ccd2a36` 就地把 232 改成 277,
     80 条标注下标错位);
  ③ 那两个常量本身错位过一次,score 从 0.92 变成 0.82。

  v2 只认一份配对清单(`distill/build_pairs.py` 生成):
  ① 服务器不知道也不关心"teacher"和"student"是什么,它只有 `key_0` / `key_1`;
  ② 标注按 `pair_id` 键控,与任何列表顺序解耦;
  ③ 统计口径搬到 `report.py`,服务器与离线复算共用同一份实现。

盲法设计(**防泄漏是本工具的核心功能,不是附加项**):
  - 左右由 `md5(f"{pair_id}|{blind_seed}") % 2` 确定性决定,刷新/重启不变;
  - 前端只拿到不透明 key(`cand:{idx}:{L|R}` / `ref:{idx}:{i}`),
    `pair_id` / `kind` / `key_*` / 图片路径**从不**出现在任何接口返回里;
    尤其 `kind` 必须瞒住——知道哪些是零假设对,标注行为就会变;
  - `/api/stats` 默认只回进度,`?reveal=1` 才揭盲。

用法:
    python -m distill.blind_eval.server \
        --pairs output/eval_multiref/pairs_m5r1.json \
        --marks output/eval_multiref/blind_annotations_m5r1.json \
        --prev_marks output/eval_multiref/blind_annotations_m4r1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
from .pairing import CHOICES, SLOTS, slot_of, validate
from .report import full_report

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ------------------------------------------------------------------ 标注存储

class MarkStore:
    """JSON 落盘 + 每次写前备份(仿 `distill/viewer/storage.py` 的纪律)。"""

    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.backup_dir = path.parent / "blind_annotations_backups"
        self._lock = threading.Lock()

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "rt", encoding="utf-8") as f:
            return json.load(f).get("marks", {})

    def set(self, pair_id: str, choice: str, winner: str) -> int:
        with self._lock:
            marks = self.load()
            marks[pair_id] = {"choice": choice, "winner": winner,
                              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._write(marks)
            return len(marks)

    def clear(self) -> None:
        with self._lock:
            self._write({})

    def _write(self, marks: dict) -> None:
        if self.path.exists():
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            bak = self.backup_dir / f"{self.path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.json"
            bak.write_bytes(self.path.read_bytes())
        payload = {"meta": {**self.meta, "n_marks": len(marks)}, "marks": marks}
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ App

def create_app(pairs_path: Path, marks_path: Path, root: Path = ROOT,
               prev_marks_path: Path | None = None) -> FastAPI:
    manifest = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs, blind_seed = validate(manifest, root)   # 不过就启动即炸,见 pairing.validate
    meta = manifest["meta"]

    prev_marks = None
    if prev_marks_path is not None:
        prev_marks = json.loads(prev_marks_path.read_text(encoding="utf-8"))["marks"]

    store = MarkStore(marks_path, {"batch_id": meta.get("batch_id"),
                                   "blind_seed": blind_seed,
                                   "pairs": str(pairs_path.relative_to(root))
                                   if pairs_path.is_relative_to(root) else str(pairs_path)})

    app = FastAPI(title=f"Blind A/B Eval — {meta.get('batch_id')}")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    def pair_at(idx: int) -> dict:
        try:
            return pairs[idx]
        except IndexError:
            raise HTTPException(status_code=404, detail="no such pair")

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "batch_id": meta.get("batch_id"), "n_pairs": len(pairs)}

    @app.get("/api/tasks")
    async def get_tasks() -> dict:
        """前端要的全部信息。**刻意不含 pair_id / kind / key_* / 图路径**——
        任何能反推候选身份、或能认出"这条是零假设对"的字段都不下发。"""
        marks = store.load()
        rows = []
        for idx, p in enumerate(pairs):
            m = marks.get(p["pair_id"])
            rows.append({"idx": idx, "stratum": p["stratum"],
                         "n_refs": len(p["ref_paths"]), "prompt": p["prompt"],
                         "annotated": m is not None,
                         "choice": m["choice"] if m else None})
        return {"n_total": len(pairs), "n_annotated": len(marks),
                "question": meta.get("question", "哪一张更好?"), "tasks": rows}

    @app.get("/api/img")
    async def get_img(k: str = Query(..., description="不透明图片 key")) -> FileResponse:
        """key 两种形态:`cand:{idx}:{L|R}` / `ref:{idx}:{i}`。
        语义标签只在服务端这一层解析,绝不进 URL。"""
        parts = k.split(":")
        if len(parts) != 3:
            raise HTTPException(status_code=400, detail="bad key")
        kind, idx_s, sub = parts
        try:
            p = pair_at(int(idx_s))
        except ValueError:
            raise HTTPException(status_code=400, detail="bad idx")

        if kind == "cand":
            if sub not in SLOTS:
                raise HTTPException(status_code=400, detail="bad slot")
            rel = slot_of(p, blind_seed, sub)[1]
        elif kind == "ref":
            try:
                rel = p["ref_paths"][int(sub)]
            except (ValueError, IndexError):
                raise HTTPException(status_code=404, detail="no such ref")
        else:
            raise HTTPException(status_code=400, detail="bad kind")

        target = (root / rel).resolve()
        try:                     # 启动时已查过一遍,这里是纵深防御(清单可能被换掉)
            target.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=404, detail="not allowed")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(target)

    @app.post("/api/mark")
    async def post_mark(payload: dict) -> dict:
        idx, choice = payload.get("idx"), payload.get("choice")
        if not isinstance(idx, int) or not 0 <= idx < len(pairs):
            raise HTTPException(status_code=400, detail="bad idx")
        if choice not in CHOICES:
            raise HTTPException(status_code=400, detail=f"choice must be one of {CHOICES}")
        p = pairs[idx]
        winner = "tie" if choice == "tie" else slot_of(p, blind_seed, choice)[0]
        n = store.set(p["pair_id"], choice, winner)
        return {"ok": True, "n_annotated": n, "n_total": len(pairs)}

    @app.post("/api/reset")
    async def post_reset() -> dict:
        store.clear()
        return {"ok": True}

    @app.get("/api/stats")
    async def get_stats(reveal: int = Query(0, description="1 才揭盲")) -> dict:
        marks = store.load()
        if not reveal:
            return {"n_total": len(pairs), "n_annotated": len(marks),
                    "complete": len(marks) == len(pairs)}
        return full_report(manifest, marks, prev_marks)

    @app.get("/api/review")
    async def get_review() -> dict:
        """揭盲后的逐条回看。盲评阶段前端不会调用本接口,调用即视为揭盲。"""
        marks = store.load()
        rows = []
        for i, p in enumerate(pairs):
            m = marks.get(p["pair_id"])
            rows.append({
                "idx": i, "pair_id": p["pair_id"], "kind": p["kind"],
                "stratum": p["stratum"], "prompt": p["prompt"],
                "src_task_id": p["src_task_id"],
                "left_key": slot_of(p, blind_seed, "L")[0],
                "right_key": slot_of(p, blind_seed, "R")[0],
                "focus_key": p["key_0"], "baseline_key": p["key_1"],
                "choice": m["choice"] if m else None,
                "winner": m["winner"] if m else None,
            })
        return {"batch_id": meta.get("batch_id"), "rows": rows}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def no_cache_static(request, call_next):
        response: Response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="人工盲评服务(配对清单驱动)")
    ap.add_argument("--pairs", type=Path,
                    default=ROOT / "output/eval_multiref/pairs_m5r1.json")
    ap.add_argument("--marks", type=Path, default=None,
                    help="标注落盘路径,默认 <pairs 同目录>/blind_annotations_<batch_id>.json")
    ap.add_argument("--prev_marks", type=Path, default=None,
                    help="重放对照用的旧标注(算自洽率);不给就不算")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    pairs_path = args.pairs.resolve()
    if not pairs_path.exists():
        print(f"[ERROR] 配对清单不存在: {pairs_path}", file=sys.stderr)
        return 1
    manifest_meta = json.loads(pairs_path.read_text(encoding="utf-8"))["meta"]
    marks_path = (args.marks or
                  pairs_path.parent / f"blind_annotations_{manifest_meta['batch_id']}.json"
                  ).resolve()
    prev = args.prev_marks.resolve() if args.prev_marks else None
    if prev and not prev.exists():
        print(f"[ERROR] --prev_marks 不存在: {prev}", file=sys.stderr)
        return 1

    app = create_app(pairs_path, marks_path, ROOT, prev)
    print(f"[INFO] 配对清单: {pairs_path}")
    print(f"[INFO] 批次    : {manifest_meta.get('batch_id')} "
          f"({manifest_meta.get('n_pairs')} 对 {manifest_meta.get('by_kind')})")
    print(f"[INFO] 盲种    : {manifest_meta.get('blind_seed')}")
    print(f"[INFO] 标注落盘: {marks_path}")
    print(f"[INFO] 重放对照: {prev or '(未提供,不算自洽率)'}")
    print(f"[INFO] URL     : http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
