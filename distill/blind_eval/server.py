#!/usr/bin/env python3
"""M4 人工盲评服务:teacher(official_full) vs student(ours_kv_post4000) 双图盲评。

盲法设计(防泄漏是本工具的核心,不是附加功能):
  - 每任务的 A/B 槽位由 md5(f"{task_id}|{BLIND_SEED}") 确定性决定,重启/刷新不变;
  - 前端只能拿到不透明图片 key(cand:{idx}:{A|B} / ref:{idx}:{i}),
    变体名、task_id 从不出现在 URL / DOM / 接口返回里;
  - 统计接口 /api/stats 默认只回进度,带 ?reveal=1 才揭盲给出 T/S/B 与分数。

计分口径(用户确认 2026-07-30):
  T = teacher 更好条数,S = student 更好条数,B = 一样好(含一样烂)条数,
  分数 = (S + B) / (T + B)。范围 S1–S4 共 227 条(不含 S0 锚点)。

用法:
  python -m distill.blind_eval.server --port 8765
  # 浏览器经 SSH 端口转发打开 http://localhost:8765
"""
from __future__ import annotations

import argparse
import hashlib
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_EVAL_JSON = ROOT / "datasets" / "eval_multiref" / "eval_set.json"
DEFAULT_SAVE_DIR = ROOT / "output" / "eval_multiref"

TEACHER = "official_full"
STUDENT = "ours_kv_post4000"
BLIND_SEED = "m4-blind-v1"          # 改动它 = 全部槽位重排 = 已标注数据作废,不许改
DEFAULT_STRATA = ("S1", "S2", "S3", "S4")   # 用户确认:S0 锚点不参与人评
CHOICES = ("A", "B", "tie")


def student_on_left(task_id: str) -> bool:
    """槽位分配:纯函数、跨进程/跨次运行稳定(不能用内置 hash,PYTHONHASHSEED 随机化)。"""
    h = hashlib.md5(f"{task_id}|{BLIND_SEED}".encode()).hexdigest()
    return int(h, 16) % 2 == 0


def slot_variant(task_id: str, slot: str) -> str:
    """slot "A"=左图,"B"=右图 → 变体名。只存在于服务端。"""
    sol = student_on_left(task_id)
    if slot == "A":
        return STUDENT if sol else TEACHER
    return TEACHER if sol else STUDENT


# ------------------------------------------------------------------ 标注存储

class MarkStore:
    """JSON 落盘 + merge + 备份(仿 distill/viewer/storage.py 的纪律)。"""

    def __init__(self, path: Path):
        self.path = path
        self.backup_dir = path.parent / "blind_annotations_backups"
        self._lock = threading.Lock()

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "rt", encoding="utf-8") as f:
            return json.load(f).get("marks", {})

    def set(self, idx: int, choice: str, task_id: str) -> int:
        with self._lock:
            marks = self.load()
            marks[str(idx)] = {"choice": choice, "task_id": task_id,
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
        payload = {"meta": {"blind_seed": BLIND_SEED, "teacher": TEACHER, "student": STUDENT},
                   "marks": marks}
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ App

def create_app(eval_json: Path, save_dir: Path, marks_path: Path,
               strata: tuple[str, ...]) -> FastAPI:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    all_tasks = payload["tasks"]
    tasks = [t for t in all_tasks if t["stratum"] in strata]
    if not tasks:
        raise SystemExit(f"❌ {eval_json} 里没有层 {strata} 的任务")
    json_dir = eval_json.parent

    # 启动断言:全部候选图存在且能完整解码(im.load(),截断图会在解码时才炸)。
    # 评到一半发现缺图,前面的人工判断就白费了,所以宁可启动炸。
    missing = []
    for t in tasks:
        for v in (TEACHER, STUDENT):
            p = save_dir / f"{t['task_id']}__{v}.png"
            ok = p.exists()
            if ok:
                try:
                    with Image.open(p) as im:
                        im.load()
                except Exception:
                    ok = False
            if not ok:
                missing.append(str(p))
        for rel in t["image_paths"]:
            if not (json_dir / rel).resolve().exists():
                missing.append(f"{t['task_id']} ref: {rel}")
    if missing:
        raise SystemExit(f"❌ {len(missing)} 个图片缺失或不可解码,例如:\n  " +
                         "\n  ".join(missing[:5]))

    store = MarkStore(marks_path)
    allowed_bases = [save_dir.resolve(), (ROOT / "datasets").resolve()]

    app = FastAPI(title="M4 Blind A/B Eval")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "n_tasks": len(tasks), "strata": list(strata)}

    @app.get("/api/tasks")
    async def get_tasks() -> dict:
        """任务列表。注意:刻意不含 task_id / variants / image_paths——任何能反推
        变体身份的字段都不下发,前端只有 idx + A/B。"""
        marks = store.load()
        rows = []
        for idx, t in enumerate(tasks):
            m = marks.get(str(idx))
            rows.append({
                "idx": idx,
                "stratum": t["stratum"],
                "n_refs": len(t["image_paths"]),
                "prompt": t["prompt"],
                "annotated": m is not None,
                "choice": m["choice"] if m else None,
            })
        return {"n_total": len(tasks), "n_annotated": len(marks), "tasks": rows}

    @app.get("/api/img")
    async def get_img(k: str = Query(..., description="不透明图片 key")) -> FileResponse:
        """key 两种形态:cand:{idx}:{A|B}(候选图) / ref:{idx}:{i}(第 i 张 ref)。
        变体名只在服务端这一层解析,绝不出现在 URL 里。"""
        parts = k.split(":")
        if len(parts) != 3:
            raise HTTPException(status_code=400, detail="bad key")
        kind, idx_s, sub = parts
        try:
            idx = int(idx_s)
            t = tasks[idx]
        except (ValueError, IndexError):
            raise HTTPException(status_code=404, detail="no such task")

        if kind == "cand":
            if sub not in ("A", "B"):
                raise HTTPException(status_code=400, detail="bad slot")
            variant = slot_variant(t["task_id"], sub)
            target = (save_dir / f"{t['task_id']}__{variant}.png").resolve()
        elif kind == "ref":
            try:
                rel = t["image_paths"][int(sub)]
            except (ValueError, IndexError):
                raise HTTPException(status_code=404, detail="no such ref")
            target = (json_dir / rel).resolve()
        else:
            raise HTTPException(status_code=400, detail="bad kind")

        for base in allowed_bases:
            try:
                target.relative_to(base)
            except ValueError:
                continue
            if target.exists() and target.is_file():
                return FileResponse(target)
        raise HTTPException(status_code=404, detail="image not found or not allowed")

    @app.post("/api/mark")
    async def post_mark(payload: dict) -> dict:
        idx = payload.get("idx")
        choice = payload.get("choice")
        if not isinstance(idx, int) or not 0 <= idx < len(tasks):
            raise HTTPException(status_code=400, detail="bad idx")
        if choice not in CHOICES:
            raise HTTPException(status_code=400, detail=f"choice must be one of {CHOICES}")
        n = store.set(idx, choice, tasks[idx]["task_id"])
        return {"ok": True, "n_annotated": n, "n_total": len(tasks)}

    @app.post("/api/reset")
    async def post_reset() -> dict:
        store.clear()
        return {"ok": True}

    @app.get("/api/stats")
    async def get_stats(reveal: int = Query(0, description="1 才揭盲")) -> dict:
        marks = store.load()
        out = {"n_total": len(tasks), "n_annotated": len(marks),
               "complete": len(marks) == len(tasks)}
        if not reveal:
            return out

        def tally(idxs):
            T = S = B = 0
            for i in idxs:
                m = marks.get(str(i))
                if not m:
                    continue
                if m["choice"] == "tie":
                    B += 1
                    continue
                winner = slot_variant(tasks[i]["task_id"], m["choice"])
                if winner == STUDENT:
                    S += 1
                else:
                    T += 1
            score = (S + B) / (T + B) if (T + B) > 0 else None
            return {"T": T, "S": S, "B": B, "n": T + S + B, "score": score}

        out["overall"] = tally(range(len(tasks)))
        out["by_stratum"] = {st: tally([i for i, t in enumerate(tasks)
                                      if t["stratum"] == st])
                             for st in strata}
        # 揭盲后的逐条映射,供审计/写报告
        out["detail"] = [{
            "idx": i,
            "task_id": tasks[i]["task_id"],
            "stratum": tasks[i]["stratum"],
            "choice": marks[str(i)]["choice"],
            "winner": ("tie" if marks[str(i)]["choice"] == "tie"
                       else slot_variant(tasks[i]["task_id"], marks[str(i)]["choice"])),
            "student_on": ("A" if student_on_left(tasks[i]["task_id"]) else "B"),
        } for i in range(len(tasks)) if str(i) in marks]
        return out

    @app.get("/api/review")
    async def get_review() -> dict:
        """揭盲后的逐条回看:身份映射 + 用户标注 + 判定结果。

        盲评阶段前端不会调用本接口(/api/tasks 刻意不含任何身份信息);
        调用即视为揭盲。图片仍走 /api/img 的不透明 key,身份揭示只发生在前端展示层。
        """
        marks = store.load()
        rows = []
        for i, t in enumerate(tasks):
            m = marks.get(str(i))
            choice = m["choice"] if m else None
            if choice == "tie":
                winner = "tie"
            elif choice in ("A", "B"):
                winner = slot_variant(t["task_id"], choice)
            else:
                winner = None
            rows.append({
                "idx": i,
                "task_id": t["task_id"],
                "stratum": t["stratum"],
                "prompt": t["prompt"],
                "choice": choice,
                "student_on": "A" if student_on_left(t["task_id"]) else "B",
                "winner": winner,
            })
        return {"teacher": TEACHER, "student": STUDENT, "rows": rows}

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
    ap = argparse.ArgumentParser(description="M4 人工盲评服务(teacher vs student A/B)")
    ap.add_argument("--eval_json", type=Path, default=DEFAULT_EVAL_JSON)
    ap.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    ap.add_argument("--marks", type=Path, default=None,
                    help="标注落盘路径,默认 <save_dir>/blind_annotations.json")
    ap.add_argument("--strata", type=str, default=",".join(DEFAULT_STRATA),
                    help="逗号分隔,默认 S1,S2,S3,S4(227 条)")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    eval_json = args.eval_json.resolve()
    save_dir = args.save_dir.resolve()
    if not eval_json.exists():
        print(f"[ERROR] eval json 不存在: {eval_json}", file=sys.stderr)
        return 1
    marks_path = (args.marks or save_dir / "blind_annotations.json").resolve()
    strata = tuple(s.strip() for s in args.strata.split(",") if s.strip())

    app = create_app(eval_json, save_dir, marks_path, strata)
    n = len([t for t in json.loads(eval_json.read_text(encoding="utf-8"))["tasks"]
             if t["stratum"] in strata])
    print(f"[INFO] 评测集  : {eval_json}")
    print(f"[INFO] 任务数  : {n} 条({','.join(strata)})")
    print(f"[INFO] 标注落盘: {marks_path}")
    print(f"[INFO] URL     : http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
