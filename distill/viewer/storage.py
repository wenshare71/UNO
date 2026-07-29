"""Persistent annotation storage backed by a JSON file.

File format on disk is always::

    {"meta": {...}, "annotations": {"0": {...}, "1": {...}}}

Two safety nets against data loss:

- **Merge-on-save**: ``save()`` merges the incoming annotations into whatever
  is already on disk (incoming wins per key). A stale browser tab posting an
  older/smaller state can therefore never wipe out rows it doesn't know about.
- **Timestamped backups**: before every write the current file is copied to
  ``<path.parent>/annotations_backups/annotations-<timestamp>.json`` and only
  the newest ``backup_keep`` copies are retained.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class AnnotationStore:
    """File-backed annotation store with asyncio locking.

    Stores annotations keyed by row name (e.g. the seed-based identifier). Each
    value is a dict with at least ``choice`` and ``note`` fields.
    """

    def __init__(self, path: Path, backup_keep: int = 30) -> None:
        self.path = path
        self.backup_dir = path.parent / "annotations_backups"
        self.backup_keep = backup_keep
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers (call with the lock held)
    # ------------------------------------------------------------------

    def _read_unlocked(self) -> dict:
        """Read the file and normalize to {"meta": ..., "annotations": ...}."""
        if not self.path.exists():
            return {"meta": {}, "annotations": {}}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("annotations"), dict):
            raw.setdefault("meta", {})
            return raw
        # Legacy flat format: the whole file is the annotations dict.
        if isinstance(raw, dict):
            return {"meta": {}, "annotations": raw}
        return {"meta": {}, "annotations": {}}

    def _backup_unlocked(self) -> Path | None:
        """Copy the current file into the backup dir and prune old copies."""
        if not self.path.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dst = self.backup_dir / f"annotations-{ts}.json"
        shutil.copy2(self.path, dst)
        backups = sorted(self.backup_dir.glob("annotations-*.json"))
        for old in backups[: max(0, len(backups) - self.backup_keep)]:
            old.unlink(missing_ok=True)
        return dst

    def _write_unlocked(self, data: dict) -> None:
        """Write atomically via a tmp file + rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self) -> dict:
        """Load annotations from disk. Returns {"meta", "annotations"}."""
        async with self._lock:
            return self._read_unlocked()

    async def save(self, annotations: dict[str, dict], *, merge: bool = True) -> int:
        """Persist annotations, merging into existing data by default.

        Returns the total number of annotation rows after the write.
        """
        async with self._lock:
            current = self._read_unlocked()
            if merge:
                merged = dict(current["annotations"])
                merged.update(annotations)
            else:
                merged = dict(annotations)
            self._backup_unlocked()
            meta = dict(current.get("meta") or {})
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_unlocked({"meta": meta, "annotations": merged})
            return len(merged)

    async def clear(self) -> None:
        """Backup the current file, then reset annotations to empty."""
        async with self._lock:
            self._backup_unlocked()
            meta = {"updated_at": datetime.now(timezone.utc).isoformat()}
            self._write_unlocked({"meta": meta, "annotations": {}})

    def list_backups(self) -> list[dict]:
        """List available backups, newest first."""
        if not self.backup_dir.exists():
            return []
        out = []
        for p in sorted(self.backup_dir.glob("annotations-*.json"), reverse=True):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("annotations"), dict):
                    count = len(raw["annotations"])
                elif isinstance(raw, dict):
                    count = len(raw)  # legacy flat format
                else:
                    count = -1
            except Exception:
                count = -1
            out.append({"file": p.name, "count": count, "size": p.stat().st_size})
        return out
