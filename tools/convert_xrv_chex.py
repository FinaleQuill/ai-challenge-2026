"""Extract the verified author's legacy CheXpert model with an exact allowlist.

The raw file contains legacy module persistent IDs unsupported by weights_only.
Every pickle entry point below uses RestrictedUnpickler, whose find_class maps
only the static globals actually found in this SHA256-pinned official file.
Downloaded source strings are never executed; the original model never runs.
The output is immediately reloaded with weights_only=True and strictly checked
against a newly constructed stock torchvision DenseNet121 on CPU.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import types
import warnings

import torch
from torch import nn
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "cache/pretrained/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
OUTPUT = ROOT / "cache/pretrained/densenet121-res224-chex-state_dict.pt"
EXPECTED = "1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4"
URL = "https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
PATHOLOGIES = ["Atelectasis", "Consolidation", "", "Pneumothorax", "Edema", "", "", "Effusion", "Pneumonia", "", "Cardiomegaly", "", "", "", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]
VALID = [0, 1, 3, 4, 7, 8, 10, 14, 15, 16, 17]

# Empty trusted structural classes replace the author's classes completely.
DenseNet = type("DenseNet", (nn.Module,), {})
DenseBlock = type("_DenseBlock", (nn.Sequential,), {})
DenseLayer = type("_DenseLayer", (nn.Sequential,), {})
Transition = type("_Transition", (nn.Sequential,), {})
ALLOWED = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "LongStorage"): torch.LongStorage,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.nn.modules.activation", "ReLU"): nn.ReLU,
    ("torch.nn.modules.batchnorm", "BatchNorm2d"): nn.BatchNorm2d,
    ("torch.nn.modules.container", "Sequential"): nn.Sequential,
    ("torch.nn.modules.conv", "Conv2d"): nn.Conv2d,
    ("torch.nn.modules.linear", "Linear"): nn.Linear,
    ("torch.nn.modules.pooling", "AvgPool2d"): nn.AvgPool2d,
    ("torch.nn.modules.pooling", "MaxPool2d"): nn.MaxPool2d,
    ("torchxrayvision.models", "DenseNet"): DenseNet,
    ("torchxrayvision.models", "_DenseBlock"): DenseBlock,
    ("torchxrayvision.models", "_DenseLayer"): DenseLayer,
    ("torchxrayvision.models", "_Transition"): Transition,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key not in ALLOWED:
            raise pickle.UnpicklingError(f"Rejected unapproved global {module}.{name}")
        return ALLOWED[key]


def restricted_load(stream, **kwargs):
    return RestrictedUnpickler(stream, **kwargs).load()


def main():
    torch.set_num_threads(2)
    data = RAW.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == EXPECTED, "Only the hash-verified official author file is permitted"
    # Five independent legacy pickle programs precede the raw tensor storages.
    # Reject extra globals statically before any object reconstruction occurs.
    stream = io.BytesIO(data)
    globals_seen = set()
    for _ in range(5):
        for op, argument, _ in pickletools.genops(stream):
            if op.name == "GLOBAL":
                module, name = argument.split(" ", 1)
                assert (module, name) in ALLOWED, (module, name)
                globals_seen.add((module, name))
            assert op.name not in {"EXT1", "EXT2", "EXT4", "STACK_GLOBAL"}, op.name
    restricted = types.ModuleType("restricted_xrv_pickle")
    restricted.Unpickler = RestrictedUnpickler
    restricted.load = restricted_load
    restricted.loads = lambda value, **kwargs: restricted_load(io.BytesIO(value), **kwargs)
    with warnings.catch_warnings(record=True) as captured:
        # False only selects legacy persistent-module handling. All global
        # resolution remains restricted, including the header pickle loads.
        original = torch.load(RAW, map_location="cpu", weights_only=False, pickle_module=restricted)
    for module in original.modules():
        if not hasattr(module, "_non_persistent_buffers_set"):
            module._non_persistent_buffers_set = set()
    state = original.state_dict()
    assert all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())
    assert tuple(state["features.conv0.weight"].shape) == (64, 1, 7, 7)
    assert tuple(state["classifier.weight"].shape) == (18, 1024)
    assert all(value.device.type == "cpu" for value in state.values())
    assert all(torch.isfinite(value).all() for value in state.values())
    source = {"name": "densenet121-res224-chex", "dataset": "CheXpert",
              "url": URL, "sha256": digest, "pathologies": PATHOLOGIES,
              "valid_indices": VALID, "input_channels": 1, "input_resolution": 224,
              "input_normalization": "(2 * (uint8 / 255) - 1) * 1024",
              "license": "Apache-2.0", "original_forward_executed": False,
              "conversion": "Exact-allowlist legacy decoder; empty structural stubs; no downloaded code execution"}
    torch.save({"state_dict": state, "source": source}, OUTPUT)
    checkpoint = torch.load(OUTPUT, map_location="cpu", weights_only=True)
    model = densenet121(weights=None, num_classes=18)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not torch.cuda.is_initialized()
    report = {"status": "verified_plain_state_dict", "source": source,
              "raw_bytes": len(data), "output": str(OUTPUT),
              "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              "state_dict_keys": len(state), "classifier_shape": [18, 1024],
              "first_conv_shape": [64, 1, 7, 7], "strict_torchvision_mapping": True,
              "pure_output_weights_only_reload": True, "cuda_initialized": False,
              "allowlist": [f"{m}.{n}" for m, n in sorted(ALLOWED)],
              "static_globals_seen": [f"{m}.{n}" for m, n in sorted(globals_seen)],
              "warnings": [str(item.message) for item in captured],
              "competition_abnormality_mapping_zero_based": [17, 3, 1, 8, 4, 10, 0, 16, 7],
              "no_finding_pretrained": False,
              "hf_metadata_discrepancy": "HF card dataset tag lists mixed datasets; author GitHub model table, README and HF config identify this chex-only URL. The downloaded GitHub asset matches the HF LFS SHA256 exactly.",
              "evidence": ["https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py",
                           "https://github.com/mlmed/torchxrayvision/blob/master/README.md",
                           "https://github.com/mlmed/torchxrayvision/blob/master/scripts/train_model.py",
                           "https://huggingface.co/torchxrayvision/densenet121-res224-chex/blob/main/config.json",
                           "https://huggingface.co/api/models/torchxrayvision/densenet121-res224-chex/tree/main",
                           "https://github.com/mlmed/torchxrayvision/blob/master/LICENSE",
                           "https://arxiv.org/abs/2111.00595"]}
    (ROOT / "artifacts/chexpert_pretraining.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
