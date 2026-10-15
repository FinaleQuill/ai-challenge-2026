"""Compare pooling rules and estimate patient-clustered validation uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cxr.data import read_frame
from cxr.metrics import evaluate_predictions, score_studies, serializable_metrics


def patient_bootstrap(study_frame, targets, probabilities, repeats=200, seed=42):
    rng = np.random.default_rng(seed)
    patients = study_frame.Subject_id.to_numpy()
    unique = np.unique(patients)
    groups = [np.flatnonzero(patients == patient) for patient in unique]
    scores = []
    for _ in range(repeats):
        index = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        score = score_studies(targets[index], probabilities[index])
        # Never present a different subset of classes as the same Macro-AUC.
        if score["auc_classes_evaluated"] == 10:
            scores.append(score["macro_auc"])
    return {"unit": "patient", "requested_repeats": repeats, "valid_repeats": len(scores),
            "macro_auc_percentile_95_interval": np.percentile(scores, [2.5, 97.5]).tolist() if scores else None,
            "note": "Descriptive validation uncertainty; model selection used this same validation set."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("cxr_first/cxr_first"))
    p.add_argument("--output", type=Path, default=Path("outputs/validation_analysis.json"))
    p.add_argument("--bootstrap", type=int, default=200)
    args = p.parse_args()
    frame = read_frame(args.data_root, "valid")
    with np.load(args.predictions, allow_pickle=False) as archive:
        keys = set(archive.files)
        if "image_probabilities" in keys:
            probs = archive["image_probabilities"]
        elif "probabilities" in keys:
            probs = archive["probabilities"]
        elif "probs" in keys:
            probs = archive["probs"]
        else:
            raise ValueError(f"No image probabilities in {keys}")
        if "dicom_ids" in keys:
            if not np.array_equal(archive["dicom_ids"], frame.Dicom_id.to_numpy(dtype=str)):
                raise ValueError("Prediction and CSV image ordering differ")
        elif "image_paths" in keys:
            if not np.array_equal(archive["image_paths"], frame.image_path.to_numpy(dtype=str)):
                raise ValueError("Prediction and CSV image ordering differ")
        else:
            raise ValueError("Prediction archive lacks image identities; refusing positional-only scoring")
    comparison = {}
    for aggregation in ["mean", "max", "mean_max"]:
        result = evaluate_predictions(frame, probs, aggregation=aggregation)
        comparison[aggregation] = serializable_metrics(result)
    mean = evaluate_predictions(frame, probs)
    output = {"pooling_comparison": comparison,
              "patient_bootstrap_mean_pooling": patient_bootstrap(
                  mean["study_frame"], mean["study_targets"], mean["study_probs"], args.bootstrap)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
