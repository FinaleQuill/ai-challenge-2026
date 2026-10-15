"""Evaluate continuous predictions at the competition's Study level."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from cxr import CLASS_NAMES


def aggregate_studies(frame, probs, targets=None, aggregation="mean"):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(frame), 10):
        raise ValueError(f"Expected ({len(frame)}, 10) probabilities, got {probs.shape}")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    if frame.empty:
        raise ValueError("Cannot aggregate an empty frame")
    if frame[["Subject_id", "Study_id"]].isna().any().any():
        raise ValueError("Missing subject/study identifiers")
    key = pd.MultiIndex.from_frame(frame[["Subject_id", "Study_id"]])
    codes, _ = pd.factorize(key, sort=False)
    n = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=n)
    means = np.zeros((n, 10), dtype=np.float64)
    np.add.at(means, codes, probs)
    means /= counts[:, None]
    maxima = np.full((n, 10), -np.inf)
    np.maximum.at(maxima, codes, probs)
    if aggregation == "mean":
        pooled = means
    elif aggregation == "max":
        pooled = maxima
    elif aggregation == "mean_max":
        pooled = (means + maxima) / 2
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    first = np.full(n, len(frame), dtype=int)
    np.minimum.at(first, codes, np.arange(len(frame)))
    study_frame = frame.iloc[first][["Subject_id", "Study_id"]].reset_index(drop=True)
    study_frame["image_count"] = counts
    if targets is None and "targets" in frame:
        targets = np.asarray(frame["targets"].tolist(), dtype=np.float64)
    study_targets = None
    if targets is not None:
        targets = np.asarray(targets)
        if targets.shape != probs.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("Targets must be a binary (images, 10) matrix")
        study_targets = targets[first].astype(np.float32)
        if not np.array_equal(targets, study_targets[codes]):
            raise ValueError("Images within a Study have inconsistent targets")
    return study_frame, pooled.astype(np.float32), study_targets


def score_studies(targets, probs, threshold=0.5):
    targets, probs = np.asarray(targets), np.asarray(probs)
    if targets.shape != probs.shape or targets.ndim != 2 or targets.shape[1] != 10:
        raise ValueError("Expected matching (studies, 10) targets/probabilities")
    per_auc = []
    for c in range(10):
        per_auc.append(float(roc_auc_score(targets[:, c], probs[:, c]))
                       if np.unique(targets[:, c]).size == 2 else None)
    valid_auc = [v for v in per_auc if v is not None]
    per_f1 = f1_score(targets, probs >= threshold, average=None, zero_division=0)
    return {
        "macro_auc": float(np.mean(valid_auc)) if valid_auc else None,
        "macro_f1": float(per_f1.mean()),
        "per_class_auc": per_auc,
        "per_class_f1": per_f1.tolist(),
        "positive_studies": targets.sum(axis=0).astype(int).tolist(),
        "auc_classes_evaluated": len(valid_auc),
        "study_count": len(targets),
        "threshold": float(threshold),
        "class_names": list(CLASS_NAMES),
    }


def evaluate_predictions(frame, probs, targets=None, aggregation="mean"):
    study_frame, study_probs, study_targets = aggregate_studies(
        frame, probs, targets, aggregation=aggregation)
    result = {"study_frame": study_frame, "study_probs": study_probs,
              "study_targets": study_targets, "aggregation": aggregation}
    if study_targets is not None:
        result.update(score_studies(study_targets, study_probs))
    return result


def serializable_metrics(result):
    return {k: v for k, v in result.items()
            if k not in {"study_frame", "study_probs", "study_targets"}}
