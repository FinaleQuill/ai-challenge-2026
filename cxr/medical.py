"""CheXpert-only TorchXrayVision DenseNet weights with a ten-class logits head.

Only a locally converted, tensor-only checkpoint is accepted. This module never
downloads weights and never deserializes the original historical model pickle.
The original source model has eighteen rows, eleven with CheXpert labels; nine
match competition abnormalities. ``No Finding`` is initialized as a new class.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from cxr import CLASS_NAMES


SOURCE_MODEL = "densenet121-res224-chex"
PRETRAINED_PATH = (
    Path(__file__).resolve().parents[1]
    / "cache" / "pretrained" / "densenet121-res224-chex-state_dict.pt"
)
# Empty entries are untrained outputs in this particular CheXpert-only model.
# They must not be interpreted using another TorchXrayVision model's labels.
SOURCE_PATHOLOGIES = (
    "Atelectasis", "Consolidation", "", "Pneumothorax", "Edema", "", "",
    "Effusion", "Pneumonia", "", "Cardiomegaly", "", "", "",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
)
VALID_SOURCE_INDICES = (0, 1, 3, 4, 7, 8, 10, 14, 15, 16, 17)
_SOURCE_ALIASES = {"Pleural Effusion": "Effusion"}
TARGET_SOURCE_ROWS = tuple(
    None if name == "No Finding" else SOURCE_PATHOLOGIES.index(_SOURCE_ALIASES.get(name, name))
    for name in CLASS_NAMES
)


class InputNormalization(nn.Module):
    """Restore grayscale from dataset RGB normalization, then use XRV units.

    CXRDataset emits three channels normalized with ImageNet means/stds. Its
    first channel is inverted here using mean .485 and std .229; grayscale
    [0, 1] becomes [-1024, 1024], as required by the source X-ray model.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("CheXpert input must have shape (batch, 3, height, width)")
        grayscale = images[:, :1] * 0.229 + 0.485
        return (grayscale - 0.5) * 2048.0


def _read_pretrained(path: Path) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing safe CheXpert state_dict checkpoint: {path}. "
            "Prepare the verified tensor-only conversion first, or construct "
            "with pretrained=False when loading an existing competition checkpoint."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "source" not in payload:
        raise ValueError("CheXpert checkpoint must contain state_dict and source metadata")
    state = payload["state_dict"]
    source = payload["source"]
    if not isinstance(source, dict):
        raise ValueError("CheXpert source metadata must be a dictionary")
    if tuple(source.get("pathologies", ())) != SOURCE_PATHOLOGIES:
        raise ValueError("CheXpert source pathologies do not match the verified eighteen-row order")
    if tuple(source.get("valid_indices", ())) != VALID_SOURCE_INDICES:
        raise ValueError("CheXpert valid_indices do not match its eleven trained outputs")
    for key in ("model", "name"):
        if key in source and source[key] != SOURCE_MODEL:
            raise ValueError(f"Expected source {key} {SOURCE_MODEL!r}, got {source[key]!r}")
    if str(source.get("dataset", "CheXpert")).lower() != "chexpert":
        raise ValueError("Only the verified CheXpert-only source is accepted")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("CheXpert state_dict must be a nonempty tensor mapping")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Every CheXpert state_dict entry must have a string key and tensor value")
    required_shapes = {
        "features.conv0.weight": (64, 1, 7, 7),
        "classifier.weight": (18, 1024),
        "classifier.bias": (18,),
    }
    for name, shape in required_shapes.items():
        if name not in state or tuple(state[name].shape) != shape:
            actual = tuple(state[name].shape) if name in state else None
            raise ValueError(f"CheXpert {name} must have shape {shape}, got {actual}")
    return state, dict(source)


def build_chexpert_densenet(pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """Build a torchvision-compatible DenseNet121 returning ten raw logits.

    ``pretrained=False`` constructs the same returned parameter names and shapes
    without reading any external checkpoint. This is the path for safely loading
    a saved competition ``state_dict``. ``features`` and ``classifier`` retain
    their usual names for training parameter groups and backbone freezing.
    """
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    source_state, source_metadata = _read_pretrained(PRETRAINED_PATH) if pretrained else (None, None)
    model = models.densenet121(weights=None, num_classes=18)
    original_conv = model.features.conv0
    grayscale_conv = nn.Conv2d(
        1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
        stride=original_conv.stride, padding=original_conv.padding,
        dilation=original_conv.dilation, groups=original_conv.groups, bias=False,
    )
    nn.init.kaiming_normal_(grayscale_conv.weight, mode="fan_out", nonlinearity="relu")
    model.features.conv0 = grayscale_conv
    if source_state is not None:
        # No silent dropping of buffers or unmatched features: a different
        # source implementation must first be verified and converted explicitly.
        model.load_state_dict(source_state, strict=True)
    source_head = model.classifier
    target_head = nn.Linear(source_head.in_features, len(CLASS_NAMES))
    if source_state is not None:
        with torch.no_grad():
            for target_row, source_row in enumerate(TARGET_SOURCE_ROWS):
                if source_row is not None:
                    target_head.weight[target_row].copy_(source_head.weight[source_row])
                    target_head.bias[target_row].copy_(source_head.bias[source_row])
    model.classifier = nn.Sequential(nn.Dropout(dropout), target_head) if dropout else target_head
    model.features.conv0 = nn.Sequential(InputNormalization(), grayscale_conv)
    # Plain metadata is intentionally outside the parameter state_dict.
    model.pretraining_source = source_metadata
    return model
