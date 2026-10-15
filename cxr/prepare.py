"""Create resumable CPU JPEG caches without changing competition originals."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

from PIL import Image, ImageOps

from .data import SPLITS, read_frame, resolve_data_root


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _convert_one(job: tuple[str, str, int, int, bool]) -> dict:
    source_text, destination_text, longest_side, quality, overwrite = job
    source, destination = Path(source_text), Path(destination_text)
    result = {"source": source_text, "cache": destination_text}
    temporary = None
    try:
        if destination.is_file() and not overwrite:
            try:
                with Image.open(destination) as cached:
                    cached.verify()
                with Image.open(source) as original:
                    expected_longest = min(max(original.size), longest_side)
                with Image.open(destination) as cached:
                    if max(cached.size) == expected_longest and cached.mode == "RGB":
                        return {**result, "status": "skipped", "width": cached.width, "height": cached.height}
            except (OSError, ValueError, SyntaxError):
                pass  # An interrupted/corrupt cache entry is safely rebuilt.
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((longest_side, longest_side), Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            image.save(temporary, format="JPEG", quality=quality, optimize=True, subsampling=0)
            width, height = image.size
        os.replace(temporary, destination)
        return {**result, "status": "written", "width": width, "height": height}
    except Exception as error:
        return {**result, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(
    data_root: str | Path,
    cache_root: str | Path,
    splits: tuple[str, ...] | list[str] = SPLITS,
    longest_side: int = 512,
    quality: int = 95,
    workers: int = 4,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    """Write an atomic per-split JSONL manifest plus an atomic JSON summary."""
    if longest_side < 16 or not 1 <= quality <= 100 or workers < 1:
        raise ValueError("longest_side >= 16, quality in 1..100, and workers >= 1 are required")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    root = resolve_data_root(data_root)
    cache = Path(cache_root).expanduser().resolve()
    if cache == root or cache.is_relative_to(root):
        raise ValueError("cache_root must be outside data_root to preserve competition originals")
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {
        "data_root": str(root), "cache_root": str(cache), "longest_side": longest_side,
        "quality": quality, "workers": workers,
        "created_at": datetime.now(timezone.utc).isoformat(), "splits": {},
    }
    for split in splits:
        frame = read_frame(root, split)
        if limit is not None:
            frame = frame.head(limit)
        manifest = cache / f"manifest_{split}.jsonl"
        temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
        jobs = (
            (
                row.image_path,
                str(cache / split / row.Subject_id / row.Study_id / f"{row.Dicom_id}.jpg"),
                longest_side, quality, overwrite,
            )
            for row in frame.itertuples(index=False)
        )
        counts = {"total": len(frame), "written": 0, "skipped": 0, "failed": 0}
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for number, result in enumerate(executor.map(_convert_one, jobs), 1):
                        counts[result["status"]] += 1
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        if number % 500 == 0 or number == len(frame):
                            output.flush()
                            print(f"{split}: {number}/{len(frame)} {counts}", flush=True)
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)
        summary["splits"][split] = {**counts, "manifest": str(manifest)}
        summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
        _atomic_json(cache / "prepare_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--longest-side", "--size", dest="longest_side", type=int, default=512)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional smoke-check limit per split")
    args = parser.parse_args()
    summary = prepare(**vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(item["failed"] for item in summary["splits"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
