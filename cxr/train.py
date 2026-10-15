"""Reproducible multi-label training and Study-level model selection.

Run ``python -m cxr.train --config configs/densenet121.json``. A run can be
stopped with Ctrl+C or by creating ``run_dir/STOP``. At the next optimizer
boundary its checkpoint is written atomically, including progress within the
epoch. Remove STOP before resuming with ``--resume path/to/last.pt``. A new run
can warm-start from an inference checkpoint with ``--init-checkpoint best.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cxr.data import CXRDataset, read_frame
from cxr.metrics import evaluate_predictions
from cxr.models import build_model


DEFAULTS: dict[str, Any] = {
    "model": "densenet121",
    "pretrained": True,
    "init_checkpoint": None,
    "dropout": 0.0,
    "image_size": 320,
    "batch_size": 12,
    "valid_batch_size": None,
    "num_workers": 2,
    "cpu_threads": 4,
    "epochs": 12,
    "lr": 0.001,
    "backbone_lr": 0.0001,
    "weight_decay": 0.0001,
    "seed": 2026,
    "amp": True,
    "accumulation": 2,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_epochs": 1.0,
    "min_lr_ratio": 0.01,
    "gradient_clip": 1.0,
    "pos_weight": False,
    "pos_weight_clip": 10.0,
    "study_weighted_loss": False,
    "aggregation": "mean",
    "freeze_bn": False,
    "freeze_backbone": False,
    "channels_last": True,
    "cudnn_benchmark": True,
    "deterministic": False,
    "early_stopping_patience": 4,
    "min_delta": 0.00001,
    "log_every": 100,
    "log_interval_seconds": 30.0,
    "max_train_samples": None,
    "max_valid_samples": None,
    "smoke_steps": None,
    "device": "auto",
}


def json_safe(value: Any) -> Any:
    """Replace non-finite metrics and turn NumPy scalars into JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def emit(event: str, **values: Any) -> None:
    print(json.dumps(json_safe({"event": event, **values}), ensure_ascii=False, allow_nan=False), flush=True)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng(generators: dict[str, torch.Generator]) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else [],
        "generators": {name: generator.get_state() for name, generator in generators.items()},
    }


def restore_rng(state: dict[str, Any], generators: dict[str, torch.Generator]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])
    for name, generator_state in state.get("generators", {}).items():
        if name in generators:
            generators[name].set_state(generator_state.cpu())


def seed_worker(worker_id: int) -> None:
    # A top-level function is pickleable under Windows' spawn start method.
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep scores, supports and per-class vectors, excluding prediction arrays."""
    result = {}
    for key, value in metrics.items():
        if key in {"study_probs", "study_targets", "study_frame"} or key.endswith("probs") or key.endswith("targets") or key.endswith("frame"):
            continue
        if isinstance(value, (str, int, float, bool, np.generic)) or value is None:
            result[key] = json_safe(value)
        elif isinstance(value, (dict, list, tuple)):
            try:
                result[key] = json_safe(value)
            except TypeError:
                pass
        elif isinstance(value, np.ndarray) and value.ndim == 1 and len(value) <= 100:
            result[key] = json_safe(value)
    return result


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    supplied = json.loads(path.read_text(encoding="utf-8-sig"))
    config = {**DEFAULTS, **supplied}
    for key in ("device", "epochs", "smoke_steps", "max_train_samples", "max_valid_samples"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.no_pretrained:
        config["pretrained"] = False
    for override in args.set:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected --set key=value, got {override!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        config[key] = value
    root = config.get("data_root") or config.get("root")
    if not root:
        raise ValueError("The JSON configuration must provide data_root (or root)")
    config["data_root"] = str(Path(root).expanduser().resolve())
    cache = config.get("cache_root") or config.get("cache")
    config["cache_root"] = str(Path(cache).expanduser().resolve()) if cache else None
    config["run_dir"] = str(Path(config.get("run_dir", "runs/baseline")).expanduser().resolve())
    init_checkpoint = config.get("init_checkpoint")
    config["init_checkpoint"] = (str(Path(init_checkpoint).expanduser().resolve())
                                 if init_checkpoint else None)
    for key in ("batch_size", "accumulation", "epochs", "cpu_threads", "log_every"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    if not 0 <= float(config["min_lr_ratio"]) <= 1:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if config["aggregation"] not in {"mean", "max", "mean_max"}:
        raise ValueError("aggregation must be mean, max, or mean_max")
    return json_safe(config)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if config["freeze_backbone"] and not name.startswith("classifier."):
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            (head if name.startswith("classifier.") else backbone).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(config["backbone_lr"] or config["lr"]), "name": "backbone"})
    if head:
        groups.append({"params": head, "lr": float(config["lr"]), "name": "classifier"})
    if config["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    if config["optimizer"].lower() == "sgd":
        return torch.optim.SGD(groups, momentum=float(config.get("momentum", 0.9)),
                               weight_decay=float(config["weight_decay"]))
    raise ValueError("optimizer must be adamw or sgd")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], updates_per_epoch: int):
    if config["scheduler"] in (None, "none"):
        return None
    if config["scheduler"] != "cosine":
        raise ValueError("scheduler must be cosine or none")
    total = max(1, int(config["epochs"]) * updates_per_epoch)
    warmup = int(config.get("warmup_steps", float(config["warmup_epochs"]) * updates_per_epoch))
    warmup = max(0, min(warmup, total - 1))
    floor = float(config["min_lr_ratio"])

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        fraction = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * fraction))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def make_loader(dataset, *, training: bool, config: dict[str, Any], device: torch.device,
                generator: torch.Generator) -> DataLoader:
    workers = int(config["num_workers"])
    extra = {"prefetch_factor": int(config.get("prefetch_factor", 2))} if workers else {}
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] if training else (config["valid_batch_size"] or config["batch_size"])),
        shuffle=training,
        drop_last=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        **extra,
    )


def to_device(images: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    images = images.to(device, non_blocking=device.type == "cuda")
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def _close_iterator(iterator) -> None:
    # A Windows DataLoader's children must leave before a clean stopped exit.
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def validate(model: nn.Module, loader: DataLoader, frame, device: torch.device, config: dict[str, Any],
             stop_requested) -> tuple[np.ndarray, np.ndarray] | None:
    model.eval()
    probs = np.empty((len(frame), 10), dtype=np.float32)
    targets = np.empty_like(probs)
    iterator = iter(loader)
    last_log = time.monotonic()
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(iterator):
                if stop_requested():
                    return None
                images = to_device(batch["image"], device, config["channels_last"])
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=bool(config["amp"]) and device.type == "cuda"):
                    logits = model(images)
                indices = torch.as_tensor(batch["index"]).cpu().numpy()
                probs[indices] = logits.float().sigmoid().cpu().numpy()
                targets[indices] = torch.as_tensor(batch["target"]).cpu().numpy()
                if time.monotonic() - last_log >= float(config["log_interval_seconds"]):
                    emit("valid_progress", batch=batch_idx + 1, batches=len(loader))
                    last_log = time.monotonic()
    finally:
        _close_iterator(iterator)
    return probs, targets


def train(config: dict[str, Any], resume_path: str | None = None,
          init_checkpoint: str | None = None) -> dict[str, Any]:
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_path = run_dir / "STOP"
    if stop_path.exists():
        raise RuntimeError(f"Remove {stop_path} before starting or resuming training")
    if not resume_path and (run_dir / "last.pt").exists():
        raise FileExistsError(f"{run_dir} already contains last.pt; use --resume or choose another run_dir")
    if resume_path and init_checkpoint:
        raise ValueError("Use either --resume for exact continuation or --init-checkpoint for warm start, not both")
    write_json(run_dir / "config.json", config)
    threads = int(config["cpu_threads"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(threads, 4))
    except RuntimeError:
        pass
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if config["device"] == "auto" else config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = bool(config["cudnn_benchmark"]) and not config["deterministic"]
    torch.backends.cudnn.deterministic = bool(config["deterministic"])
    torch.set_float32_matmul_precision("high")
    if config["deterministic"]:
        torch.use_deterministic_algorithms(True, warn_only=True)
    generators = {name: torch.Generator().manual_seed(int(config["seed"]) + offset)
                  for offset, name in enumerate(("train", "valid"))}
    train_frame = read_frame(config["data_root"], "train").reset_index(drop=True)
    valid_frame = read_frame(config["data_root"], "valid").reset_index(drop=True)
    for name, limit in (("train", config["max_train_samples"]), ("valid", config["max_valid_samples"])):
        if limit is not None:
            if int(limit) < 1:
                raise ValueError(f"max_{name}_samples must be positive")
            if name == "train":
                train_frame = train_frame.iloc[:int(limit)].copy().reset_index(drop=True)
            else:
                valid_frame = valid_frame.iloc[:int(limit)].copy().reset_index(drop=True)
    if not len(train_frame) or not len(valid_frame):
        raise ValueError("Both train and valid splits need at least one image")
    train_dataset = CXRDataset(train_frame, int(config["image_size"]), True, config["cache_root"])
    valid_dataset = CXRDataset(valid_frame, int(config["image_size"]), False, config["cache_root"])
    train_loader = make_loader(train_dataset, training=True, config=config, device=device, generator=generators["train"])
    valid_loader = make_loader(valid_dataset, training=False, config=config, device=device, generator=generators["valid"])
    train_targets = np.stack(train_frame["targets"].to_numpy()).astype(np.float32)
    if train_targets.shape != (len(train_frame), 10) or not np.isin(train_targets, [0.0, 1.0]).all():
        raise ValueError("Training targets must be binary arrays with ten classes")
    pos_weight = None
    if config["pos_weight"] is True:
        positives = train_targets.sum(axis=0)
        raw_weights = (len(train_frame) - positives) / np.maximum(positives, 1)
        pos_weight = torch.tensor(np.clip(raw_weights, 1.0, float(config["pos_weight_clip"])), dtype=torch.float32, device=device)
        emit("pos_weight", values=pos_weight.cpu().tolist())
    elif isinstance(config["pos_weight"], list):
        if len(config["pos_weight"]) != 10 or min(config["pos_weight"]) <= 0:
            raise ValueError("Explicit pos_weight must have ten positive values")
        pos_weight = torch.tensor(config["pos_weight"], dtype=torch.float32, device=device)
    elif config["pos_weight"] not in (None, False):
        raise ValueError("pos_weight must be false, true, or a list of ten positive weights")
    sample_weights = None
    if config["study_weighted_loss"]:
        counts = train_frame.groupby("study_key")["study_key"].transform("size").to_numpy(dtype=np.float32)
        if (counts < 1).any():
            raise ValueError("study_image_count must be positive")
        weights = 1.0 / counts
        weights /= weights.mean()
        sample_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path else None
    initial_checkpoint = None
    initialized_from = None
    if init_checkpoint and checkpoint is None:
        initial_path = Path(init_checkpoint).expanduser().resolve()
        if not initial_path.is_file():
            raise FileNotFoundError(f"Initial checkpoint does not exist: {initial_path}")
        initial_checkpoint = torch.load(initial_path, map_location="cpu", weights_only=True)
        initial_config = initial_checkpoint.get("config") if isinstance(initial_checkpoint, dict) else None
        if initial_config and initial_config.get("model") != config.get("model"):
            raise ValueError(
                f"Initial checkpoint model {initial_config.get('model')!r} does not match {config.get('model')!r}"
            )
        initialized_from = str(initial_path)
    if checkpoint:
        old_config = checkpoint["config"]
        immutable = ("model", "dropout", "image_size", "batch_size", "accumulation", "seed", "epochs",
                     "lr", "backbone_lr", "weight_decay", "optimizer", "scheduler", "warmup_epochs",
                     "min_lr_ratio", "freeze_backbone", "freeze_bn", "study_weighted_loss", "pos_weight",
                     "data_root", "cache_root", "max_train_samples", "max_valid_samples", "num_workers")
        changed = [key for key in immutable if old_config.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"Resume requires the same training configuration; changed keys: {changed}")
    # On resume, never re-download pretrained weights that the checkpoint replaces.
    model = build_model(config["model"],
                        pretrained=bool(config["pretrained"]) and checkpoint is None and initial_checkpoint is None,
                        dropout=float(config["dropout"]))
    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    elif initial_checkpoint is not None:
        initial_state = (initial_checkpoint.get("state_dict")
                         or initial_checkpoint.get("model_state_dict")
                         or initial_checkpoint.get("model"))
        if initial_state is None:
            raise ValueError(f"Initial checkpoint has no state_dict: {initialized_from}")
        model.load_state_dict(initial_state, strict=True)
        emit("initialized", path=initialized_from,
             source_epoch=initial_checkpoint.get("epoch"), source_score=initial_checkpoint.get("score"))
    model.to(device)
    if config["channels_last"]:
        model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, config)
    updates_per_epoch = math.ceil(len(train_loader) / int(config["accumulation"]))
    scheduler = build_scheduler(optimizer, config, updates_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]) and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    start_epoch, resume_batch, global_step, stale_epochs = 0, 0, 0, 0
    best_score, best_epoch = None, None
    resume_epoch_start_rng = None
    resume_rng = None
    running_loss, running_samples = 0.0, 0
    if checkpoint:
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires last.pt, not the inference-only best.pt")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["next_epoch"])
        resume_batch = int(checkpoint.get("batch_in_epoch", 0))
        global_step = int(checkpoint["global_step"])
        best_score, best_epoch = checkpoint.get("best_score"), checkpoint.get("best_epoch")
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        running_loss = float(checkpoint.get("epoch_loss_sum", 0.0))
        running_samples = int(checkpoint.get("epoch_samples", 0))
        resume_epoch_start_rng = checkpoint.get("epoch_start_rng")
        resume_rng = checkpoint.get("rng")
        if resume_rng:
            restore_rng(resume_rng, generators)
        emit("resumed", path=str(resume_path), next_epoch=start_epoch + 1,
             batch_in_epoch=resume_batch, best_score=best_score)
    history_path = run_dir / "history.jsonl"
    history = []
    if resume_path and history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if int(row["epoch"]) <= start_epoch:
                    history.append(row)
        # Remove an epoch log written before a crash prevented checkpoint commit.
        history_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in history), encoding="utf-8")
    stop_state = {"requested": False}

    def request_stop(signum, frame):
        del signum, frame
        stop_state["requested"] = True

    def stop_requested():
        return stop_state["requested"] or stop_path.exists()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    versions = {"torch": str(torch.__version__), "numpy": str(np.__version__), "python": sys.version.split()[0]}
    began = time.monotonic()
    summary = {"status": "running", "best_score": best_score, "best_epoch": best_epoch,
               "initialized_from": initialized_from,
               "run_dir": str(run_dir), "config": config, "versions": versions,
               "train_images": len(train_frame), "valid_images": len(valid_frame)}
    write_json(run_dir / "summary.json", summary)
    emit("start", device=str(device), train_images=len(train_frame), valid_images=len(valid_frame),
         batches_per_epoch=len(train_loader), parameters=sum(p.numel() for p in model.parameters()),
         amp=scaler.is_enabled(), channels_last=config["channels_last"], run_dir=str(run_dir))

    def save_last(next_epoch: int, batch_in_epoch: int, epoch_start_rng, loss_sum: float, samples: int):
        state = {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "config": config,
            "versions": versions,
            "epoch": next_epoch if batch_in_epoch == 0 else next_epoch + 1,
            "score": best_score,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "next_epoch": next_epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "epoch_start_rng": epoch_start_rng,
            "epoch_loss_sum": loss_sum,
            "epoch_samples": samples,
            "rng": capture_rng(generators),
        }
        atomic_torch_save(state, run_dir / "last.pt")

    try:
        for epoch in range(start_epoch, int(config["epochs"])):
            epoch_began = time.monotonic()
            model.train()
            if config["freeze_bn"]:
                for layer in model.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            if config["freeze_backbone"]:
                model.features.eval()
            continuing_epoch = epoch == start_epoch and resume_batch > 0
            if continuing_epoch:
                if resume_epoch_start_rng is None:
                    raise ValueError("Mid-epoch resume checkpoint is missing epoch_start_rng")
                restore_rng(resume_epoch_start_rng, generators)
                epoch_start_rng = resume_epoch_start_rng
            else:
                running_loss, running_samples = 0.0, 0
                epoch_start_rng = capture_rng(generators)
            optimizer.zero_grad(set_to_none=True)
            processed_batches = resume_batch if continuing_epoch else 0
            if continuing_epoch and resume_batch == len(train_loader):
                # No training is left: a previous interruption happened in
                # validation. Do not decode an entire epoch just to skip it.
                restore_rng(resume_rng, generators)
                continuing_epoch = False
                iterator = iter(())
            else:
                iterator = iter(train_loader)
                if continuing_epoch:
                    for _ in range(resume_batch):
                        next(iterator)
                    # Rebuild the shuffle and worker streams by replaying data,
                    # then restore model RNG before loading the next CPU batch.
                    restore_rng(resume_rng, generators)
                    continuing_epoch = False
            last_log = time.monotonic()
            stopped = False
            try:
                for batch_idx, batch in enumerate(iterator, start=processed_batches):
                    images = to_device(batch["image"], device, config["channels_last"])
                    targets = batch["target"].to(device, dtype=torch.float32, non_blocking=device.type == "cuda")
                    group_start = (batch_idx // int(config["accumulation"])) * int(config["accumulation"])
                    group_size = min(int(config["accumulation"]), len(train_loader) - group_start)
                    group_sample_count = min(
                        group_size * int(config["batch_size"]),
                        len(train_frame) - group_start * int(config["batch_size"]),
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                        logits = model(images)
                        image_loss = criterion(logits, targets).mean(dim=1)
                        if sample_weights is not None:
                            indices = torch.as_tensor(batch["index"], device=device, dtype=torch.long)
                            image_loss = image_loss * sample_weights[indices]
                        loss = image_loss.mean()
                    loss_value = float(loss.detach())
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(f"Non-finite loss in epoch {epoch + 1}, batch {batch_idx + 1}")
                    scaler.scale(image_loss.sum() / group_sample_count).backward()
                    running_loss += loss_value * len(images)
                    running_samples += len(images)
                    processed_batches = batch_idx + 1
                    update_boundary = processed_batches % int(config["accumulation"]) == 0 or processed_batches == len(train_loader)
                    if update_boundary:
                        if config["gradient_clip"]:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= scale_before:
                            if scheduler is not None:
                                scheduler.step()
                            global_step += 1
                        optimizer.zero_grad(set_to_none=True)
                    now = time.monotonic()
                    if batch_idx == 0 or processed_batches % int(config["log_every"]) == 0 or now - last_log >= float(config["log_interval_seconds"]):
                        emit("train_progress", epoch=epoch + 1, batch=processed_batches, batches=len(train_loader),
                             loss=running_loss / running_samples, lr=[group["lr"] for group in optimizer.param_groups],
                             global_step=global_step, seconds=now - epoch_began,
                             gpu_allocated_mb=torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0)
                        last_log = now
                    smoke_limit = config.get("smoke_steps")
                    if update_boundary and (stop_requested() or (smoke_limit is not None and global_step >= int(smoke_limit))):
                        stopped = True
                        break
            finally:
                _close_iterator(iterator)
            resume_batch = 0
            if stopped:
                if processed_batches == 0:
                    restore_rng(epoch_start_rng, generators)
                save_last(epoch, processed_batches, epoch_start_rng, running_loss, running_samples)
                summary.update(status="stopped", epoch=epoch + 1, batch_in_epoch=processed_batches,
                               best_score=best_score, best_epoch=best_epoch,
                               elapsed_seconds=time.monotonic() - began)
                write_json(run_dir / "summary.json", summary)
                emit("stopped", checkpoint=str(run_dir / "last.pt"), **{k: summary[k] for k in ("epoch", "batch_in_epoch", "best_score")})
                return summary
            result = validate(model, valid_loader, valid_frame, device, config, stop_requested)
            if result is None:
                # The train epoch is complete. On resume, replay no training
                # batches and re-run validation before committing this epoch.
                save_last(epoch, len(train_loader), epoch_start_rng, running_loss, running_samples)
                summary.update(status="stopped", epoch=epoch + 1, batch_in_epoch=len(train_loader),
                               best_score=best_score, best_epoch=best_epoch,
                               elapsed_seconds=time.monotonic() - began)
                write_json(run_dir / "summary.json", summary)
                emit("stopped", checkpoint=str(run_dir / "last.pt"), stage="validation")
                return summary
            probs, valid_targets = result
            metrics = evaluate_predictions(valid_frame, probs, targets=valid_targets, aggregation=config["aggregation"])
            if int(metrics.get("auc_classes_evaluated", 10)) != 10:
                if config["max_valid_samples"] is None:
                    raise ValueError("Full validation must contain positive and negative Studies for all ten AUC classes")
                emit("limited_validation", auc_classes_evaluated=metrics.get("auc_classes_evaluated"),
                     note="Explicit small-sample interface check; its scores are not competition estimates")
            clean_metrics = metric_summary(metrics)
            score = metrics["macro_auc"]
            finite_score = float(score) if score is not None and math.isfinite(float(score)) else None
            improved = finite_score is not None and (best_score is None or finite_score > best_score)
            significant_improvement = finite_score is not None and (best_score is None or finite_score > best_score + float(config["min_delta"]))
            # A tiny smoke validation can contain only one class. Keep an
            # inspectable model, but do not call its undefined AUC a real score.
            if improved or not (run_dir / "best.pt").exists():
                if improved:
                    best_score, best_epoch = finite_score, epoch + 1
                atomic_torch_save({"format_version": 1, "state_dict": model.state_dict(), "config": config,
                                   "versions": versions, "epoch": epoch + 1, "score": finite_score,
                                   "metrics": clean_metrics}, run_dir / "best.pt")
                temporary_npz = run_dir / "best_valid_predictions.tmp.npz"
                np.savez_compressed(temporary_npz, probs=probs.astype(np.float32), targets=valid_targets.astype(np.float32),
                                    image_paths=valid_frame["image_path"].astype(str).to_numpy(dtype=str),
                                    dicom_ids=valid_frame["Dicom_id"].astype(str).to_numpy(dtype=str),
                                    study_keys=valid_frame["study_key"].astype(str).to_numpy(dtype=str))
                os.replace(temporary_npz, run_dir / "best_valid_predictions.npz")
            stale_epochs = 0 if significant_improvement else stale_epochs + 1
            row = json_safe({"epoch": epoch + 1, "train_loss": running_loss / running_samples,
                             **clean_metrics, "best_score": best_score, "best_epoch": best_epoch,
                             "epoch_seconds": time.monotonic() - epoch_began, "global_step": global_step,
                             "lr": [group["lr"] for group in optimizer.param_groups]})
            with history_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            history.append(row)
            save_last(epoch + 1, 0, None, 0.0, 0)
            summary.update(status="running", completed_epochs=epoch + 1, best_score=best_score,
                           best_epoch=best_epoch, last_metrics=clean_metrics,
                           elapsed_seconds=time.monotonic() - began)
            write_json(run_dir / "summary.json", summary)
            emit("epoch_complete", **row)
            patience = int(config["early_stopping_patience"] or 0)
            if patience and stale_epochs >= patience:
                summary["status"] = "early_stopped"
                break
        if summary["status"] == "running":
            summary["status"] = "complete"
        summary["elapsed_seconds"] = time.monotonic() - began
        write_json(run_dir / "summary.json", summary)
        emit("finished", status=summary["status"], best_score=best_score, best_epoch=best_epoch,
             seconds=summary["elapsed_seconds"])
        return summary
    except BaseException as error:
        # An OOM or asynchronous device error cannot safely serialize a
        # half-completed optimizer step; preserve the last committed checkpoint.
        summary.update(status="failed", error=f"{type(error).__name__}: {error}",
                       best_score=best_score, best_epoch=best_epoch,
                       elapsed_seconds=time.monotonic() - began)
        write_json(run_dir / "summary.json", summary)
        emit("failed", error=summary["error"], checkpoint=str(run_dir / "last.pt"))
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        optimizer.zero_grad(set_to_none=True)
        del model, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", help="Full training checkpoint (last.pt)")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Inference checkpoint used to initialize a new run")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-steps", type=int, help="Stop safely after this many optimizer updates")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download pretrained weights")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override one JSON setting")
    args = parser.parse_args()
    config = load_config(args.config, args)
    train(config, args.resume,
          str(args.init_checkpoint) if args.init_checkpoint else config.get("init_checkpoint"))


if __name__ == "__main__":
    main()
