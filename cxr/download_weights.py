"""Repair official TorchVision downloads without relaxing SHA256 checks.

Uses the installed weight enum as the URL authority, bypasses a stale CDN
response with a unique query string, then atomically installs the verified
file in Torch Hub's normal cache. No CUDA initialization is needed.

Official URL authority / checksum semantics:
https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/convnext.py
https://docs.pytorch.org/docs/2.10/hub.html#torch.hub.download_url_to_file
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import uuid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(model="convnext_tiny", attempts=3, report_path=None):
    import torch.hub
    from torchvision.models import get_model_weights

    weights = get_model_weights(model).DEFAULT
    url = weights.url
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "download.pytorch.org":
        raise ValueError(f"Expected an official PyTorch HTTPS URL, got {url}")
    filename = Path(parts.path).name
    match = re.search(r"-([a-f0-9]{8,64})\.", filename)
    if not match:
        raise ValueError(f"No published SHA256 prefix in {filename}")
    expected = match.group(1)
    destination = Path(torch.hub.get_dir()) / "checkpoints" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": model, "weight_enum": str(weights), "official_url": url,
              "expected_sha256_prefix": expected, "cache_path": str(destination),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "attempts": []}
    if destination.is_file():
        digest = sha256_file(destination)
        if digest.startswith(expected):
            report.update(status="verified_existing", sha256=digest, bytes=destination.stat().st_size)
            if report_path:
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
            return report
        report["existing_invalid_sha256"] = digest
    for attempt in range(attempts):
        query = parse_qsl(parts.query) + [("cachebust", uuid.uuid4().hex)]
        request_url = urlunsplit(parts._replace(query=urlencode(query)))
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
        row = {"url": request_url, "attempt": attempt + 1}
        began = time.monotonic()
        try:
            request = Request(request_url, headers={"User-Agent": "torch.hub", "Cache-Control": "no-cache"})
            digest, count = hashlib.sha256(), 0
            with urlopen(request, timeout=60) as response, temporary.open("xb") as output:
                row["http_status"] = response.status
                row["headers"] = {key: response.headers.get(key) for key in
                                  ("Content-Length", "Content-Type", "ETag", "X-Cache", "X-Amz-Cf-Pop")}
                if response.status != 200:
                    raise RuntimeError(f"Expected complete HTTP 200 response, got {response.status}")
                print(json.dumps({"event": "download_start", **row}), flush=True)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                length = response.headers.get("Content-Length")
                row.update(bytes=count, sha256=digest.hexdigest(), elapsed_seconds=time.monotonic()-began)
                if length is not None and count != int(length):
                    raise RuntimeError(f"Truncated response: expected {length} bytes, got {count}")
                if not row["sha256"].startswith(expected):
                    raise RuntimeError(f"Invalid SHA256: expected prefix {expected}, got {row['sha256']}")
            os.replace(temporary, destination)
            row["status"] = "verified"
            report["attempts"].append(row)
            report.update(status="verified_download", sha256=row["sha256"], bytes=count)
            break
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            report["attempts"].append(row)
            print(json.dumps(row), flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        report["status"] = "failed"
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report["status"] == "failed":
        raise RuntimeError("All official downloads failed strict SHA256 verification")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--report", type=Path, default=Path("artifacts/download_convnext_tiny.json"))
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    download_verified(args.model, args.attempts, args.report)


if __name__ == "__main__":
    main()
