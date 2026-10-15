"""Inference with checkpoints, study pooling, and reproducible probability exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cxr.data import CXRDataset, read_frame
from cxr.metrics import evaluate_predictions, serializable_metrics
from cxr.models import build_model
from cxr.submission import export_submission


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = build_model(cfg["model"], pretrained=False, dropout=cfg.get("dropout", 0.0))
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint.get("state_dict")))
    if state is None:
        raise ValueError("Checkpoint does not contain model weights")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if cfg.get("channels_last", True):
        model.to(memory_format=torch.channels_last)
    return model, cfg


def predict_checkpoint(checkpoint, frame, data_root, cache_root=None, batch_size=16,
                       num_workers=4, device="cuda", output_progress=True):
    model, cfg = load_checkpoint(checkpoint, device)
    dataset = CXRDataset(frame, image_size=cfg["image_size"], training=False, cache_root=cache_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.startswith("cuda"), persistent_workers=num_workers > 0)
    probabilities = np.empty((len(frame), 10), dtype=np.float32)
    start, last_log = time.monotonic(), time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            if cfg.get("channels_last", True):
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.startswith("cuda")):
                probs = model(x).float().sigmoid().cpu().numpy()
            probabilities[batch["index"].numpy()] = probs
            if output_progress and time.monotonic() - last_log > 30:
                print(f"Inference {int(batch['index'][-1]) + 1}/{len(frame)} "
                      f"elapsed={time.monotonic()-start:.1f}s", flush=True)
                last_log = time.monotonic()
    del loader, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return probabilities, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("cxr_first/cxr_first"))
    p.add_argument("--cache-root", type=Path, default=Path("cache/images512"))
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aggregation", choices=["mean", "max", "mean_max"], default="mean")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    frame = read_frame(args.data_root, args.split)
    all_probs, configs = [], []
    for checkpoint in args.checkpoint:
        probs, cfg = predict_checkpoint(checkpoint, frame, args.data_root, args.cache_root,
                                        args.batch_size, args.num_workers, args.device)
        all_probs.append(probs)
        configs.append(cfg)
    image_probs = np.mean(all_probs, axis=0)
    result = evaluate_predictions(frame, image_probs, aggregation=args.aggregation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = result["study_frame"]
    archive = {"subject_ids": study.Subject_id.to_numpy(dtype=str),
               "study_ids": study.Study_id.to_numpy(dtype=str),
               "probabilities": result["study_probs"],
               "image_probabilities": image_probs,
               "dicom_ids": frame.Dicom_id.to_numpy(dtype=str)}
    if result["study_targets"] is not None:
        archive["targets"] = result["study_targets"]
    np.savez_compressed(args.output_dir / f"{args.split}_probabilities.npz", **archive)
    hashes = []
    for path in args.checkpoint:
        with path.open("rb") as stream:
            hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    metadata = {"checkpoints": [str(x.resolve()) for x in args.checkpoint],
                "checkpoint_sha256": hashes,
                "aggregation": args.aggregation, "split": args.split,
                "metrics": serializable_metrics(result), "configs": configs}
    (args.output_dir / f"{args.split}_report.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if args.split == "test":
        full = export_submission(study, result["study_probs"], args.output_dir / "submission.csv")
        export_submission(study, result["study_probs"],
                          args.output_dir / "submission_threshold_0p5.csv", mode="positive")
        print(json.dumps(full, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(serializable_metrics(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
