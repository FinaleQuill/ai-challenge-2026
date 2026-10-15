"""Torchvision backbones with ten independent chest radiograph outputs."""

from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(name: str, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """Build a classifier returning logits; apply sigmoid only for prediction.

    ``pretrained=False`` never downloads weights. A checkpoint contains this
    function's configuration and a state_dict, rather than a pickled model.
    """
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    name = name.lower().replace("-", "_")
    if name == "densenet121_chexpert":
        from cxr.medical import build_chexpert_densenet
        return build_chexpert_densenet(pretrained=pretrained, dropout=dropout)
    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        head = nn.Linear(in_features, 10)
        model.classifier = nn.Sequential(nn.Dropout(dropout), head) if dropout else head
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        head = nn.Linear(in_features, 10)
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), head) if dropout else head
    elif name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 10))
    else:
        raise ValueError(
            f"Unsupported model {name!r}; choose densenet121, densenet121_chexpert, convnext_tiny, or efficientnet_v2_s"
        )
    return model
