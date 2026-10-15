"""Audit metadata, study labels, patient separation, and image availability."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from .data import NUM_CLASSES, SPLITS, read_frame, resolve_data_root


def audit(data_root: str | Path, verify_images: bool = False) -> dict:
    root = resolve_data_root(data_root)
    result = {
        "data_root": str(root), "created_at": datetime.now(timezone.utc).isoformat(),
        "class_order": list(range(1, NUM_CLASSES + 1)), "splits": {}, "patient_intersections": {},
        "errors": [],
    }
    patient_sets: dict[str, set[str]] = {}
    for split in SPLITS:
        try:
            frame = read_frame(root, split)
        except Exception as error:
            result["errors"].append(f"{split}: {type(error).__name__}: {error}")
            continue
        patient_sets[split] = set(frame["Subject_id"])
        absent = [path for path in frame["image_path"] if not Path(path).is_file()]
        studies = frame.drop_duplicates("study_key")
        entry = {
            "images": len(frame), "patients": len(patient_sets[split]), "studies": len(studies),
            "csv_sha256": hashlib.sha256((root / f"{split}.csv").read_bytes()).hexdigest(),
            "labeled": "targets" in frame, "missing_images": len(absent), "missing_examples": absent[:10],
            "images_per_study": {
                str(int(key)): int(value)
                for key, value in studies["study_image_count"].value_counts().sort_index().items()
            },
            "views": frame["ViewPosition"].value_counts().to_dict() if "ViewPosition" in frame else {},
        }
        if absent:
            result["errors"].append(f"{split}: {len(absent)} images are missing")
        if "targets" in frame:
            unique_labels = frame.groupby("study_key")["targets"].agg(lambda values: len({tuple(value) for value in values}))
            inconsistent = unique_labels[unique_labels > 1].index.tolist()
            entry["inconsistent_studies"] = len(inconsistent)
            entry["inconsistent_study_examples"] = inconsistent[:10]
            entry["class_positive_images"] = [
                int(sum(target[index] for target in frame["targets"])) for index in range(NUM_CLASSES)
            ]
            entry["class_positive_studies"] = [
                int(sum(target[index] for target in studies["targets"])) for index in range(NUM_CLASSES)
            ]
            if inconsistent:
                result["errors"].append(f"{split}: {len(inconsistent)} studies have conflicting labels")
        if verify_images:
            from PIL import Image
            invalid = []
            for path in frame["image_path"]:
                if not Path(path).is_file():
                    continue
                try:
                    with Image.open(path) as image:
                        image.load()
                except Exception as error:
                    invalid.append({"path": path, "error": f"{type(error).__name__}: {error}"})
            entry["invalid_images"] = len(invalid)
            entry["invalid_image_examples"] = invalid[:10]
            if invalid:
                result["errors"].append(f"{split}: {len(invalid)} images failed decoding verification")
        result["splits"][split] = entry
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if first not in patient_sets or second not in patient_sets:
            continue
        overlap = sorted(patient_sets[first] & patient_sets[second])
        result["patient_intersections"][f"{first}_{second}"] = {"count": len(overlap), "examples": overlap[:10]}
        if overlap:
            result["errors"].append(f"{first}/{second}: {len(overlap)} patients overlap")
    result["ok"] = not result["errors"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="artifacts/data_audit.json")
    parser.add_argument("--verify-images", action="store_true", help="Decode-check every original image (slower)")
    args = parser.parse_args()
    result = audit(args.data_root, verify_images=args.verify_images)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
