#!/usr/bin/env python3
"""FastAPI-backed viewer for distill_multiref quality inspection.

This turns the static ``distill/inspect_html.py`` output into a small
front/back-end service: the browser fetches the manifest and annotations via
REST API, posts annotation updates back to the server, and images are served
safely through ``/api/image``.

Usage:
    python -m distill.viewer.server
    python -m distill.viewer.server --limit 200 --shuffle
    python -m distill.viewer.server --shard 0 --port 8080

Then open http://localhost:8000 in a browser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from distill.viewer.manifest import describe_source, load_manifest
from distill.viewer.storage import AnnotationStore


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST = ROOT / "datasets" / "distill_multiref" / "manifest_raw.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    manifest_path: Path,
    annotations_path: Path,
    *,
    shard: int | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 20260727,
) -> FastAPI:
    manifest, manifest_dir = load_manifest(
        manifest_path,
        shard=shard,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
    )
    store = AnnotationStore(annotations_path)

    # Resolve allowed base directories for image serving.
    allowed_bases = [
        (ROOT / "datasets" / "distill_multiref").resolve(),
        (ROOT / "datasets" / "dreambooth" / "dataset").resolve(),
    ]

    source_desc = describe_source(manifest_path, shard, limit, shuffle, seed)

    # Prepare lightweight metadata rows for the frontend.  We do NOT include
    # prompts/refs in every row payload; instead we expose ``meta`` plus the
    # original relative paths so the frontend can render rows on demand.
    manifest_rows = []
    for idx, rec in enumerate(manifest):
        meta = rec.get("meta", {})
        manifest_rows.append(
            {
                "id": idx,
                "name": str(meta.get("seed", idx) - 3407000),
                "prompt": rec.get("prompt", ""),
                "image_paths": rec.get("image_paths", []),
                "image_tgt_path": rec.get("image_tgt_path", ""),
                "meta": meta,
            }
        )

    app = FastAPI(title="UNO Distill Viewer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "rows": len(manifest_rows), "source": source_desc}

    @app.get("/api/manifest")
    async def get_manifest(
        page: int = Query(0, ge=0, description="Zero-based page index"),
        per_page: int = Query(5, ge=1, le=200, description="Rows per page"),
    ) -> dict:
        total = len(manifest_rows)
        pages = max(1, (total + per_page - 1) // per_page) if total else 1
        page = min(page, pages - 1)
        start = page * per_page
        end = start + per_page
        return {
            "source": source_desc,
            "manifest_dir": str(manifest_dir),
            "n_rows": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
            "rows": manifest_rows[start:end],
        }

    @app.get("/api/annotations")
    async def get_annotations() -> dict:
        return await store.load()

    @app.post("/api/annotations")
    async def post_annotations(payload: dict) -> dict:
        annotations = payload.get("annotations", {})
        if not isinstance(annotations, dict):
            raise HTTPException(status_code=400, detail="annotations must be an object")
        # Merge into existing data so a stale client can never wipe rows.
        total = await store.save(annotations, merge=True)
        return {"ok": True, "count": total}

    @app.post("/api/annotations/clear")
    async def clear_annotations() -> dict:
        await store.clear()
        return {"ok": True, "count": 0}

    @app.get("/api/annotations/backups")
    async def list_annotation_backups() -> dict:
        return {"backup_dir": str(store.backup_dir), "backups": store.list_backups()}

    @app.get("/api/image")
    async def get_image(rel: str = Query(..., description="Image path relative to manifest dir")) -> FileResponse:
        # Resolve the requested path relative to the manifest directory and then
        # verify it falls under one of the allowed image base directories.
        target = (manifest_dir / rel).resolve()
        for base in allowed_bases:
            try:
                target.relative_to(base)
            except ValueError:
                continue
            if target.exists() and target.is_file():
                return FileResponse(target)
        raise HTTPException(status_code=404, detail="Image not found or path not allowed")

    # Serve static files (index.html, style.css, viewer.js, anno.js).
    # Disable client-side caching so code fixes take effect without a hard refresh.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Start the UNO distill viewer server.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Input manifest path")
    ap.add_argument("--out-annotations", type=Path, default=None, help="Annotations JSON path")
    ap.add_argument("--limit", type=int, default=None, help="Only serve the first N rows")
    ap.add_argument("--shard", type=int, default=None, help="Serve the shard-th 1000-row chunk")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle rows before truncation")
    ap.add_argument("--seed", type=int, default=20260727, help="Shuffle seed")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    ap.add_argument("--port", type=int, default=8000, help="Bind port")
    args = ap.parse_args()

    manifest_path = args.manifest.resolve()
    annotations_path = (
        args.out_annotations
        if args.out_annotations
        else manifest_path.parent / "annotations.json"
    ).resolve()

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    app = create_app(
        manifest_path,
        annotations_path,
        shard=args.shard,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print(f"[INFO] Manifest : {manifest_path}")
    print(f"[INFO] Annotations: {annotations_path}")
    print(f"[INFO] URL      : http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
