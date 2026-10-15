"""Create a small, self-contained inference ZIP without optimizer or source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    status = run / "pipeline_status.json"
    metadata = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
    aggregation = metadata.get("selected_aggregation", "mean")
    checkpoint = run / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size > 500_000_000:
        raise ValueError("Uncompressed model exceeds the 500 MB limit")
    output = args.output or run / "outputs" / "inference_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    readme = f"""# Inference package

Target: Python 3.12, torch 2.10.0, torchvision 0.25.0, CUDA 12.8.
Install requirements.txt with the appropriate CUDA PyTorch wheel index.
Place the official dataset at data/ (train.csv, valid.csv, test.csv and image directories).

python -X utf8 -m cxr.prepare --data-root data --cache-root cache/images512 --workers 4
python -X utf8 -m cxr.predict --checkpoint model.pt --data-root data --cache-root cache/images512 --split test --aggregation {aggregation} --output-dir outputs

The included submission.csv preserves all ten probabilities per Study. The official
handling of omitted classes and F1 thresholds still needs confirmation before upload.
Local latency is not official T4 certification. Never upload data or last.pt as model weights.
"""
    manifest = {"model_sha256": model_hash, "uncompressed_model_bytes": checkpoint.stat().st_size,
                "aggregation": aggregation, "source_run": run.name,
                "official_T4_verified": False, "official_submission_convention_confirmed": False}
    temporary = output.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(checkpoint, "model.pt")
        archive.write(root / "requirements.txt", "requirements.txt")
        for file in sorted((root / "cxr").glob("*.py")):
            archive.write(file, f"cxr/{file.name}")
        for file in sorted((root / "docs" / "third_party").glob("*")):
            if file.is_file():
                archive.write(file, f"licenses/{file.name}")
        provenance = root / "artifacts" / "chexpert_pretraining.json"
        if provenance.exists():
            archive.write(provenance, "provenance/chexpert_pretraining.json")
        for name in ("submission.csv", "submission_threshold_0p5.csv", "validation_analysis.json",
                     "latency.json", "test_report.json", "valid_report.json", "experiment_report.md",
                     "training_curves.png", "per_class_auc.png"):
            path = run / "outputs" / name
            if path.exists():
                archive.write(path, f"results/{name}")
        archive.writestr("README.md", readme)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    print(json.dumps({"output": str(output), "zip_bytes": output.stat().st_size, **manifest}, indent=2))


if __name__ == "__main__":
    main()
