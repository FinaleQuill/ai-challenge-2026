"""CPU-only verification under the isolated competition dependency versions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

# Process-local isolation; leave the user's active training process untouched.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def checkpoint_probe(path: Path):
    import numpy as np
    import torch
    from cxr.data import CXRDataset, read_frame
    from cxr.predict import load_checkpoint

    began = time.monotonic()
    model, config = load_checkpoint(path, "cpu")
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    frame = read_frame(ROOT / "cxr_first/cxr_first", "valid").iloc[:1]
    dataset = CXRDataset(frame, image_size=config["image_size"], training=False,
                         cache_root=ROOT / "cache/images512")
    with torch.inference_mode():
        logits = model(dataset[0]["image"].unsqueeze(0))
        probabilities = logits.float().sigmoid().numpy()
    assert logits.shape == (1, 10)
    assert np.isfinite(probabilities).all()
    assert not torch.cuda.is_initialized()
    return {"path": str(path.resolve()), "model": config["model"],
            "image_size": config["image_size"], "strict_weights_only_load": True,
            "finite_forward": True, "output_shape": list(logits.shape),
            "probabilities": probabilities[0].tolist(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "elapsed_seconds": time.monotonic() - began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/compatibility.json")
    args = parser.parse_args()
    import torch
    import torchvision
    import numpy
    import pandas
    import PIL
    import sklearn
    import matplotlib

    torch.set_num_threads(2)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__), "numpy": numpy.__version__,
                "pandas": pandas.__version__, "Pillow": PIL.__version__,
                "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    assert sys.version_info[:2] == (3, 12), versions
    assert versions["torch"].split("+")[0] == "2.10.0", versions
    assert versions["torchvision"].split("+")[0] == "0.25.0", versions
    assert torch.version.cuda is None, "This verification must use the CPU-only PyTorch build"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.skip_tests and args.report.exists() else {}
    report.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), versions=versions,
                  executable=sys.executable, platform=platform.platform(),
                  scope="Windows CPU dependency/checkpoint compatibility; target T4 latency remains unverified",
                  official_sources=["https://docs.astral.sh/uv/guides/install-python/",
                                    "https://pytorch.org/get-started/previous-versions/"])
    if not args.skip_tests:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        with (args.report.parent / "compatibility_unittest.log").open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report["unittest"] = {"tests_run": result.testsRun, "failures": len(result.failures),
                              "errors": len(result.errors), "skipped": len(result.skipped),
                              "successful": result.wasSuccessful(),
                              "log": str((args.report.parent / "compatibility_unittest.log").resolve())}
    paths = args.checkpoint or [ROOT / "runs/gpu_smoke/best.pt"]
    probes = {item["path"]: item for item in report.get("checkpoints", [])}
    for path in paths:
        print(f"CPU checkpoint validation: {path}", flush=True)
        probe = checkpoint_probe(path)
        probes[probe["path"]] = probe
    report["checkpoints"] = list(probes.values())
    report["cuda_initialized"] = torch.cuda.is_initialized()
    report["status"] = "passed" if report.get("unittest", {}).get("successful") and not report["cuda_initialized"] else "failed"
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
