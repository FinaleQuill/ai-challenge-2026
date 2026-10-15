"""Validate and export the two currently plausible submission interpretations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def export_submission(study_frame, probabilities, output, *, mode="full", threshold=0.5):
    p = np.asarray(probabilities)
    if p.shape != (len(study_frame), 10):
        raise ValueError("A complete 10-column probability matrix is required")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite in [0,1]")
    ids = study_frame[["Subject_id", "Study_id"]].astype(str)
    if ids.duplicated().any():
        raise ValueError("Duplicate Study records")
    if mode not in {"full", "positive"}:
        raise ValueError("mode must be full or positive")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    out = pd.DataFrame({
        "Subject_id": np.repeat(ids.Subject_id.to_numpy(), 10),
        "Study_id": np.repeat(ids.Study_id.to_numpy(), 10),
        "Predict_class": np.tile(np.arange(1, 11), len(ids)),
        "Probability": p.reshape(-1),
    })
    if mode == "positive":
        out = out.loc[out.Probability >= threshold].copy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    out.to_csv(temporary, index=False, encoding="utf-8", float_format="%.9g")
    temporary.replace(output)
    check = pd.read_csv(output, dtype={"Subject_id": str, "Study_id": str})
    if len(check) != len(out) or check.duplicated(["Subject_id", "Study_id", "Predict_class"]).any():
        raise RuntimeError("Submission round-trip validation failed")
    report = {"mode": mode, "rows": len(out), "expected_studies": len(ids),
              "represented_studies": len(out[["Subject_id", "Study_id"]].drop_duplicates()),
              "threshold": threshold if mode == "positive" else None,
              "official_scoring_contract_confirmed": False,
              "note": "Official rules have not clarified omitted probabilities or the F1 threshold. "
                      "Full preserves AUC information; positive follows the threshold example. "
                      "Keep complete probabilities and confirm the final upload convention."}
    output.with_suffix(".validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "positive"], default="full")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as z:
        frame = pd.DataFrame({"Subject_id": z["subject_ids"], "Study_id": z["study_ids"]})
        report = export_submission(frame, z["probabilities"], args.output,
                                   mode=args.mode, threshold=args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
