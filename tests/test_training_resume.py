"""Small CPU regression test for real augmentation and partial-epoch resume."""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

import cxr.train as training


class TinyClassifier(nn.Module):
    """Keep the production optimizer's features/classifier parameter groups."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(4, 10))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).flatten(1))


def build_tiny_model(name: str, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    del name
    if pretrained:
        raise AssertionError("The resume regression test must not request pretrained weights")
    return TinyClassifier(dropout)


def write_split(root: Path, split: str, labels: list[str]) -> None:
    """Write a real competition-shaped split with distinct Studies."""
    columns = ["Dicom_id", "Subject_id", "Study_id", "Predict_class"]
    with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, label in enumerate(labels):
            subject = f"{split}_patient_{index}"
            study = f"study_{index}"
            dicom = f"image_{index}"
            directory = root / split / subject / study
            directory.mkdir(parents=True)
            # Nonuniform images exercise actual rotations and color jitter.
            height, width = 31 + index % 5, 33 + index % 3
            pixels = ((np.arange(height * width).reshape(height, width) * (index + 3) + 17) % 256).astype(np.uint8)
            Image.fromarray(pixels).save(directory / f"{dicom}.jpg")
            writer.writerow(dict(zip(columns, [dicom, subject, study, label])))


class TrainingResumeTests(unittest.TestCase):
    def test_partial_epoch_resume_matches_uninterrupted_training(self) -> None:
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(lambda: torch.use_deterministic_algorithms(deterministic, warn_only=warn_only))
        benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", benchmark)
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", cudnn_deterministic)

        with tempfile.TemporaryDirectory(prefix="cxr_resume_test_") as temporary:
            directory = Path(temporary)
            dataset = directory / "data"
            dataset.mkdir()
            # Seven images produce microbatches 2,2,2,1 and groups of 4,3.
            write_split(dataset, "train", ["1;2", "2;3", "3;4", "4;5", "5;6", "6;7", "8;9;10"])
            # All ten classes have both positive and negative validation Studies.
            write_split(dataset, "valid", [str(label) for label in range(1, 11)])
            base = {
                **training.DEFAULTS,
                "data_root": str(dataset),
                "cache_root": None,
                "model": "tiny_test_model",
                "pretrained": False,
                "device": "cpu",
                "num_workers": 0,
                "cpu_threads": 1,
                "image_size": 32,
                "batch_size": 2,
                "accumulation": 2,
                "epochs": 2,
                "dropout": 0.35,
                "channels_last": False,
                "deterministic": True,
                "cudnn_benchmark": False,
                "early_stopping_patience": 0,
            }
            complete_config = dict(base, run_dir=str(directory / "complete"))
            interrupted_config = dict(base, run_dir=str(directory / "resumed"), smoke_steps=1)
            complete_path = Path(complete_config["run_dir"]) / "last.pt"
            resumed_path = Path(interrupted_config["run_dir"]) / "last.pt"

            # Patch only the backbone factory; use the real data, loss, optimizer,
            # scheduler, checkpointing, and Study-level validation throughout.
            with (
                patch.object(training, "build_model", side_effect=build_tiny_model) as factory,
                patch("torch.cuda.is_available", return_value=False),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                complete_summary = training.train(complete_config)
                stopped_summary = training.train(interrupted_config)
                stopped = torch.load(resumed_path, map_location="cpu", weights_only=True)
                self.assertEqual(stopped_summary["status"], "stopped")
                self.assertEqual(stopped["next_epoch"], 0)
                self.assertEqual(stopped["batch_in_epoch"], 2)
                self.assertEqual(stopped["global_step"], 1)
                resumed_config = dict(interrupted_config, smoke_steps=None)
                resumed_summary = training.train(resumed_config, str(resumed_path))
                self.assertEqual(factory.call_count, 3)

            complete = torch.load(complete_path, map_location="cpu", weights_only=True)
            resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)
            self.assertEqual(complete_summary["status"], "complete")
            self.assertEqual(resumed_summary["status"], "complete")
            self.assertEqual(complete["global_step"], 4)
            self.assertEqual(resumed["global_step"], complete["global_step"])
            self.assertEqual(resumed["next_epoch"], 2)
            self.assertEqual(resumed["batch_in_epoch"], 0)
            self.assertEqual(complete["scheduler"], resumed["scheduler"])
            self.assertEqual(complete["state_dict"].keys(), resumed["state_dict"].keys())
            for name, tensor in complete["state_dict"].items():
                with self.subTest(parameter=name):
                    self.assertTrue(torch.equal(tensor, resumed["state_dict"][name]), name)
            self.assertTrue(any(
                not torch.equal(value, stopped["state_dict"][name])
                for name, value in complete["state_dict"].items()
            ), "The remaining training steps must actually update the model")
            best = torch.load(Path(resumed_config["run_dir"]) / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(best["metrics"]["auc_classes_evaluated"], 10)
            self.assertIsInstance(best["score"], float)

    def test_warm_start_from_inference_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cxr_warm_start_test_") as temporary:
            directory = Path(temporary)
            dataset = directory / "data"
            dataset.mkdir()
            write_split(dataset, "train", ["1;2", "3;4", "5;6", "8;9;10"])
            write_split(dataset, "valid", [str(label) for label in range(1, 11)])
            source = build_tiny_model("tiny_test_model", pretrained=False, dropout=0.0)
            initial_path = directory / "best.pt"
            torch.save({"state_dict": source.state_dict(), "config": {"model": "tiny_test_model"},
                        "epoch": 3, "score": 0.7}, initial_path)
            config = {
                **training.DEFAULTS,
                "data_root": str(dataset),
                "cache_root": None,
                "run_dir": str(directory / "warm_started"),
                "model": "tiny_test_model",
                "pretrained": False,
                "device": "cpu",
                "num_workers": 0,
                "cpu_threads": 1,
                "image_size": 32,
                "batch_size": 2,
                "accumulation": 2,
                "epochs": 1,
                "dropout": 0.0,
                "channels_last": False,
                "deterministic": True,
                "cudnn_benchmark": False,
                "early_stopping_patience": 0,
            }
            with (
                patch.object(training, "build_model", side_effect=build_tiny_model),
                patch("torch.cuda.is_available", return_value=False),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = training.train(config, init_checkpoint=str(initial_path))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["initialized_from"], str(initial_path.resolve()))
            best = torch.load(Path(config["run_dir"]) / "best.pt", map_location="cpu", weights_only=True)
            self.assertIn("state_dict", best)


if __name__ == "__main__":
    unittest.main()
