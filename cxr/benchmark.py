"""Measure local training memory or batch-one inference latency. Not a T4 certification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from cxr.models import build_model


def training_probe(model_name, image_size, batch_sizes, steps):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training probe")
    torch.backends.cudnn.benchmark = True
    rows = []
    for batch_size in batch_sizes:
        model = optimizer = images = targets = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(model_name, pretrained=False).cuda().to(memory_format=torch.channels_last).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            images = torch.randn(batch_size, 3, image_size, image_size, device="cuda").to(memory_format=torch.channels_last)
            targets = torch.randint(0, 2, (batch_size, 10), device="cuda").float()
            durations = []
            for i in range(steps + 3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                if i >= 3:
                    durations.append(time.perf_counter() - start)
            row = {"batch_size": batch_size, "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
                   "step_ms": float(np.median(durations)*1000),
                   "images_per_second": float(batch_size/np.median(durations)), "status": "ok"}
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": batch_size, "status": "out_of_memory"}
        finally:
            del model, optimizer, images, targets, loss
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["status"] != "ok":
            break
    return rows


def inference_probe(checkpoint, data_root, split, samples):
    from cxr.data import CXRDataset, read_frame
    from cxr.predict import load_checkpoint
    frame = read_frame(data_root, split)
    chosen = np.linspace(0, len(frame)-1, min(samples, len(frame)), dtype=int)
    frame = frame.iloc[chosen].reset_index(drop=True)
    model, cfg = load_checkpoint(checkpoint, "cuda")
    # No cache: account for decoding original large JPEGs and preprocessing.
    dataset = CXRDataset(frame, image_size=cfg["image_size"], training=False, cache_root=None)
    with torch.inference_mode():
        x = dataset[0]["image"].unsqueeze(0).cuda()
        if cfg.get("channels_last", True):
            x = x.to(memory_format=torch.channels_last)
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
        durations = []
        for i in range(len(dataset)):
            start = time.perf_counter()
            x = dataset[i]["image"].unsqueeze(0).cuda()
            if cfg.get("channels_last", True):
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                model(x).float().sigmoid().cpu().numpy()
            torch.cuda.synchronize()
            durations.append((time.perf_counter()-start)*1000)
    return {"device": torch.cuda.get_device_name(), "samples": len(durations),
            "batch_size": 1, "original_decode_and_preprocessing_included": True,
            "median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
            "max_ms": float(np.max(durations)), "checkpoint_mib": Path(checkpoint).stat().st_size/2**20,
            "official_T4_latency_verified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="densenet121")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("cxr_first/cxr_first"))
    p.add_argument("--split", default="valid")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = p.parse_args()
    torch.set_num_threads(4)
    result = (inference_probe(args.checkpoint, args.data_root, args.split, args.samples)
              if args.checkpoint else training_probe(args.model, args.image_size, args.batch_sizes, args.steps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
