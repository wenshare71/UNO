"""Manifest loading utilities for the distill viewer."""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_manifest(
    manifest_path: Path,
    *,
    shard: int | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 20260727,
) -> tuple[list[dict], Path]:
    """Load a manifest JSON and apply optional shard/limit/shuffle filters.

    Args:
        manifest_path: Path to the manifest JSON file.
        shard: If provided, return the shard-th 1000-row chunk.
        limit: If provided, truncate to the first N rows after other filters.
        shuffle: Whether to randomly shuffle rows before truncation.
        seed: Random seed used when shuffling.

    Returns:
        (filtered_manifest_rows, manifest_directory)
    """
    data: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))

    if shard is not None:
        start = shard * 1000
        data = data[start : start + 1000]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)

    if limit is not None:
        data = data[:limit]

    return data, manifest_path.parent.resolve()


def describe_source(manifest_path: Path, shard, limit, shuffle, seed) -> str:
    """Build a human-readable description of the manifest filtering applied."""
    parts = [f"manifest={manifest_path.name}"]
    if shard is not None:
        parts.append(f"shard={shard}")
    if shuffle:
        parts.append(f"shuffle(seed={seed})")
    if limit is not None:
        parts.append(f"limit={limit}")
    return ", ".join(parts)
