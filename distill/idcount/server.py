#!/usr/bin/env python3
"""身份留存计数标注服务:单图是/否,FastAPI + uvicorn。

和 `distill/blind_eval/server.py` 问的问题不同(那边是"哪张更好"的偏好比较,
这里是"这个参考主体的身份保住了没"的单图计数),但会踩的坑是同一类坑,所以
结构照抄 blind_eval:配对/条目清单驱动、图片 URL 带版本戳、marks 原子写。

**必须照抄的一条纪律**(`distill.blind_eval.pairing.asset_tag` 上面那段注释讲的坑):
图片 URL 形如 `/api/img?k=gen:{idx}` / `ref:{idx}:{i}`,**只含下标不含内容**。
换一批 item 清单后同一个下标可能指向完全不同的图,而 URL 一字不变——浏览器会直接
命中上一批的缓存,画面上没有任何异常信号。所以这里的 `/api/img` 同样要求 `v=` 版本戳
必须等于当前清单原文的 md5,不等就 409,绝不"照旧返回一张来路不明的图"。

`/api/items` 的返回**必须剥掉 `variant` 和 `replay_of`**——这是本工具存在的意义,标注
必须是"看图猜是否"而不是"看标签猜答案"。但只删这两个字段名还不够:`img_path` 形如
`{task_id}__{variant}.png`,variant 明晃晃嵌在文件名字符串里,原样下发等于换了个地方
泄漏同一个信息。所以 `img_path` / `image_paths` 这两个**路径字段本身**也和 blind_eval
的 `img_0`/`img_1`/`ref_paths` 一样,从不出现在任何接口返回里——图只能靠 `/api/img` 的
不透明 `gen:{idx}` / `ref:{idx}:{i}` key 去拿,idx 是 items 数组里的位置,不是 item_id。

用法:
    python -m distill.idcount.server \
        --items output/probe_iso/idcount_items.json \
        --marks output/probe_iso/idcount_marks.json \
        --port 8011
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

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distill.blind_eval.pairing import asset_tag  # noqa: E402  复用同一套版本戳机制,不重开一份

ROOT = REPO_ROOT
STATIC_DIR = _HERE / "static"

# 下发给前端的字段白名单。**新增字段前请先想一遍它会不会泄露 variant**——
# 这条清单比"剥掉哪些字段"更不容易漏改,加白名单比减黑名单更难引入新泄漏点。
_PUBLIC_FIELDS = ("item_id", "task_id", "stratum", "prompt", "ref_names")


# ------------------------------------------------------------------ 标注存储

class MarkStore:
    """JSON 落盘,原子写(先写 .tmp 再 os.replace)。理由同 blind_eval:标注是
    人工判读的产物,半写的文件被下一次读取到,代价是让人重标一遍。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "rt", encoding="utf-8") as f:
            return json.load(f).get("marks", {})

    def set(self, item_id: str, answers: list[bool], dwell_ms) -> dict:
        with self._lock:
            marks = self.load()
            marks[item_id] = {"answers": answers, "dwell_ms": dwell_ms,
                              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._write(marks)
            return marks

    def _write(self, marks: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": {"n_marks": len(marks)}, "marks": marks}
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ App

def create_app(items_path: Path, marks_path: Path, root: Path = ROOT) -> FastAPI:
    items_text = items_path.read_text(encoding="utf-8")
    manifest = json.loads(items_text)
    items = manifest["items"]
    tag = asset_tag(items_text)  # 图片 URL 版本戳 = 清单原文摘要,见 pairing.asset_tag

    store = MarkStore(marks_path)

    app = FastAPI(title="身份留存计数标注")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    def item_at(idx: int) -> dict:
        try:
            return items[idx]
        except IndexError:
            raise HTTPException(status_code=404, detail="no such item")

    def item_by_id(item_id: str) -> dict | None:
        for it in items:
            if it["item_id"] == item_id:
                return it
        return None

    def public(it: dict) -> dict:
        """剥离后下发给前端的字段。见模块 docstring:img_path/image_paths 也不下发。"""
        row = {k: it[k] for k in _PUBLIC_FIELDS}
        row["n_refs"] = len(it["image_paths"])
        return row

    @app.get("/api/health")
    async def health() -> dict:
        marks = store.load()
        return {"ok": True, "tag": tag, "n_marked": len(marks), "n_total": len(items)}

    @app.get("/api/items")
    async def get_items() -> dict:
        marks = store.load()
        return {"tag": tag, "n_total": len(items), "n_marked": len(marks),
                "items": [public(it) for it in items], "marks": marks}

    @app.get("/api/img")
    async def get_img(k: str = Query(..., description="不透明图片 key"),
                      v: str = Query(..., description="清单版本戳")) -> FileResponse:
        """key 两种形态:`gen:{idx}` / `ref:{idx}:{i}`,idx 是 items 数组下标。

        `v` 不匹配一律 409:版本戳对不上意味着前端拿的是别批次的下标,此时返回
        任何一张图都是错的图,而错的图**看不出来**——宁可当场红叉,也不要静默
        给一张来路不明的参考图或生成图(理由与 blind_eval.server.get_img 一致)。"""
        if v != tag:
            raise HTTPException(status_code=409,
                                detail="asset_tag 过期,请刷新页面(可能是换了 item 清单)")
        parts = k.split(":")
        kind = parts[0] if parts else ""
        if kind == "gen" and len(parts) == 2:
            it = item_at(int(parts[1])) if parts[1].lstrip("-").isdigit() else None
            if it is None:
                raise HTTPException(status_code=400, detail="bad idx")
            rel = it["img_path"]
        elif kind == "ref" and len(parts) == 3:
            if not (parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit()):
                raise HTTPException(status_code=400, detail="bad idx")
            it = item_at(int(parts[1]))
            try:
                rel = it["image_paths"][int(parts[2])]
            except IndexError:
                raise HTTPException(status_code=404, detail="no such ref")
        else:
            raise HTTPException(status_code=400, detail="bad key")

        target = (root / rel).resolve()
        try:  # 启动时清单已含真实路径,这里是纵深防御(清单文件可能被换掉)
            target.relative_to(root.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="not allowed")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        # 缓存语义写死:URL 已含内容版本戳,不再靠浏览器启发式规则去猜该不该用缓存。
        return FileResponse(target, headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/mark")
    async def post_mark(payload: dict) -> dict:
        item_id = payload.get("item_id")
        answers = payload.get("answers")
        dwell_ms = payload.get("dwell_ms")
        it = item_by_id(item_id) if isinstance(item_id, str) else None
        if it is None:
            raise HTTPException(status_code=404, detail="no such item_id")
        n_refs = len(it["image_paths"])
        if not isinstance(answers, list) or len(answers) != n_refs or \
           not all(isinstance(a, bool) for a in answers):
            raise HTTPException(status_code=400,
                                detail=f"answers 必须是长度 {n_refs} 的布尔数组")
        marks = store.set(item_id, answers, dwell_ms)
        return {"ok": True, "n_marked": len(marks), "n_total": len(items)}

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
    ap = argparse.ArgumentParser(description="身份留存计数标注服务")
    ap.add_argument("--items", type=Path, default=ROOT / "output/probe_iso/idcount_items.json")
    ap.add_argument("--marks", type=Path, default=ROOT / "output/probe_iso/idcount_marks.json")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    items_path = args.items.resolve()
    if not items_path.exists():
        print(f"[ERROR] item 清单不存在: {items_path}", file=sys.stderr)
        print("        先跑 python distill/idcount/build_items.py 生成它", file=sys.stderr)
        return 1

    app = create_app(items_path, args.marks.resolve(), ROOT)
    print(f"[INFO] item 清单: {items_path}")
    print(f"[INFO] 标注落盘: {args.marks.resolve()}")
    print(f"[INFO] URL     : http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
