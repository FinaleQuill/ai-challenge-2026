"""Validated competition CSVs and a deterministic, study-aware image dataset.

Only pandas is imported when reading metadata. PyTorch and torchvision are
loaded when a CXRDataset is constructed so CPU preparation does not need them.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


NUM_CLASSES = 10
SPLITS = ("train", "valid", "test")
ID_COLUMNS = ("Dicom_id", "Subject_id", "Study_id")
CACHE_LONGEST_SIDE = 512


def resolve_data_root(data_root: str | Path) -> Path:
    """Accept the CSV directory or either wrapper from the original archive."""
    root = Path(data_root).expanduser().resolve()
    for candidate in (root, root / "cxr_first", root / "cxr_first" / "cxr_first"):
        if any((candidate / f"{split}.csv").is_file() for split in SPLITS):
            return candidate
    raise FileNotFoundError(f"No dataset CSV found in {root} or its cxr_first wrappers")


def parse_targets(value: str) -> list[int]:
    """Parse a nonempty, semicolon-separated set of class IDs from 1 to 10."""
    tokens = str(value).split(";")
    labels: list[int] = []
    for raw in tokens:
        token = raw.strip()
        if re.fullmatch(r"(?:[1-9]|10)", token) is None:
            raise ValueError(f"Invalid class token {raw!r} in {value!r}; expected IDs 1..10")
        labels.append(int(token))
    if len(set(labels)) != len(labels):
        raise ValueError(f"Repeated class ID in {value!r}")
    target = [0] * NUM_CLASSES
    for label in labels:
        target[label - 1] = 1
    return target


def read_frame(data_root: str | Path, split: str) -> pd.DataFrame:
    """Read string IDs, validate labels, and attach image/study metadata.

    The result has a fresh integer index. ``targets`` exists only for labeled
    CSVs; it is a list of ten integers in competition class order (1 through 10).
    ``study_key`` includes the patient ID, so study IDs need not be globally unique.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    root = resolve_data_root(data_root)
    csv_path = root / f"{split}.csv"
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    missing = set(ID_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{csv_path}: empty dataset")
    for name in ID_COLUMNS:
        invalid = frame[name].map(
            lambda value: not value or value in (".", "..") or "/" in value or "\\" in value
            or value != value.strip() or ":" in value
        )
        if invalid.any():
            indices = frame.index[invalid].tolist()[:5]
            raise ValueError(f"{csv_path}: unsafe/empty {name} at rows {indices}")
    if frame.duplicated(list(ID_COLUMNS)).any():
        raise ValueError(f"{csv_path}: duplicate image rows")
    if split in ("train", "valid") and "Predict_class" not in frame:
        raise ValueError(f"{csv_path}: labeled split is missing Predict_class")
    if "Predict_class" in frame:
        targets = []
        for index, value in enumerate(frame["Predict_class"]):
            try:
                targets.append(parse_targets(value))
            except ValueError as error:
                raise ValueError(f"{csv_path}: CSV line {index + 2}: {error}") from error
        frame["targets"] = targets
    frame["split"] = split
    frame["image_path"] = [
        str(root / split / subject / study / f"{dicom}.jpg")
        for dicom, subject, study in frame[list(ID_COLUMNS)].itertuples(index=False, name=None)
    ]
    frame["study_key"] = frame["Subject_id"] + "/" + frame["Study_id"]
    frame["study_image_count"] = frame.groupby("study_key")["Dicom_id"].transform("size").astype(int)
    return frame.reset_index(drop=True)


class CXRDataset:
    """Map-style PyTorch dataset; validation/test transforms are deterministic.

    Cached files are preferred where available, with source images as a safe
    fallback. Subsets are reindexed and counts are recomputed within the subset.
    No horizontal/vertical flips or random crops are applied. Training adds only
    small rotations and mild brightness/contrast changes.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int = 320,
        training: bool = False,
        cache_root: str | Path | None = None,
    ) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.frame = frame.copy().reset_index(drop=True)
        self.frame["study_image_count"] = (
            self.frame.groupby("study_key")["Dicom_id"].transform("size").astype(int)
        )
        self.image_size = int(image_size)
        self.training = bool(training)
        self.cache_root = Path(cache_root).expanduser().resolve() if cache_root else None
        if self.cache_root is not None and self.image_size > CACHE_LONGEST_SIDE:
            raise ValueError("image_size cannot exceed the 512-pixel cache resolution; use original images")
        self._image_paths = self.frame["image_path"].tolist()
        self._targets = (
            self.frame["targets"].tolist() if "targets" in self.frame
            else [[0.0] * NUM_CLASSES for _ in range(len(self.frame))]
        )
        self._cache_paths = None
        if self.cache_root is not None:
            self._cache_paths = [
                self.cache_root / split / subject / study / f"{dicom}.jpg"
                for split, subject, study, dicom in self.frame[
                    ["split", "Subject_id", "Study_id", "Dicom_id"]
                ].itertuples(index=False, name=None)
            ]
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self._augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR, fill=0),
            ], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from PIL import Image, ImageOps
        from torchvision.transforms import functional as functional
        from torchvision.transforms import InterpolationMode

        image_path = Path(self._image_paths[index])
        using_cache = False
        if self._cache_paths is not None:
            cached = self._cache_paths[index]
            if cached.is_file():
                image_path = cached
                using_cache = True
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source)
                if not using_cache and image.mode == "L":
                    # Identical grayscale channels need filtering only once.
                    # Verified pixel-identical to RGB-first on real CXR JPEGs.
                    intermediate_side = max(CACHE_LONGEST_SIDE, self.image_size)
                    image.thumbnail((intermediate_side, intermediate_side), Image.Resampling.LANCZOS)
                image = image.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Unable to read image {image_path}") from error
        if not using_cache:
            # Match prepare.py before the final model-size resize. The JPEG95
            # cache adds a small compression difference, intentionally avoiding
            # a JPEG round trip every time an original image is loaded.
            intermediate_side = max(CACHE_LONGEST_SIDE, self.image_size)
            image.thumbnail((intermediate_side, intermediate_side), Image.Resampling.LANCZOS)
        width, height = image.size
        ratio = self.image_size / max(width, height)
        new_width = max(1, min(self.image_size, round(width * ratio)))
        new_height = max(1, min(self.image_size, round(height * ratio)))
        image = functional.resize(
            image, [new_height, new_width], interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2
        image = functional.pad(
            image,
            [left, top, self.image_size - new_width - left, self.image_size - new_height - top],
            fill=0,
        )
        if self.training:
            image = self._augmentation(image)
        tensor = functional.to_tensor(image)
        tensor = functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target = self._targets[index]
        return {"image": tensor, "target": torch.tensor(target, dtype=torch.float32), "index": int(index)}
