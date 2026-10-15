"""Run training, validation analysis, prediction and local deployment checks."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


class PipelineStopped(Exception):
    """A requested stop has completed and no stage process remains."""


@contextlib.contextmanager
def exclusive_project_run(config: Path):
    """An OS-owned byte lock excludes other configs and releases on exit."""
    path = Path(__file__).resolve().parent.parent / "runs" / ".gpu_pipeline.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another pipeline is already running in this project") from error
        try:
            handle.seek(1)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "config": str(config)}).encode("utf-8"))
            handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def keep_awake():
    """Keep an unattended Windows run awake without changing power settings."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def stop_child(process: subprocess.Popen, stage: str, stop_path: Path, grace: float = 45.0) -> bool:
    """Reap a stage and its worker tree; return whether training was forced."""
    if process.poll() is not None:
        return False
    if stage == "train" and grace:
        stop_path.touch()
        try:
            process.wait(timeout=grace)
            return False
        except subprocess.TimeoutExpired:
            pass
    forced_training = stage == "train"
    if os.name == "nt":
        # terminate() alone would leave Windows DataLoader worker processes.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=10)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    return forced_training


def run_stage(stage: str, module: str, arguments, run_dir: Path, stop_path: Path, save):
    """Poll STOP independently of child output, including silent stages."""
    if stop_path.exists():
        raise PipelineStopped
    command = [sys.executable, "-X", "utf8", "-u", "-m", module, *map(str, arguments)]
    save(stage=stage, child_pid=None)
    print(json.dumps({"stage": stage, "command": command}, ensure_ascii=False), flush=True)
    process = None
    reader = None
    try:
        with (run_dir / f"{stage}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONUTF8": "1"})
            lines = queue.Queue()

            def read_output():
                try:
                    for line in process.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            save(child_pid=process.pid)
            eof = False
            deadline = None
            while not eof or process.poll() is None:
                if stop_path.exists():
                    if stage != "train":
                        stop_child(process, stage, stop_path, grace=0)
                        raise PipelineStopped
                    if deadline is None:
                        deadline = time.monotonic() + 45.0
                    elif time.monotonic() >= deadline and process.poll() is None:
                        stop_child(process, stage, stop_path, grace=0)
                        raise RuntimeError("Training did not stop within 45 seconds; forced exit, use the last committed checkpoint")
                try:
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                else:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f"{stage} failed with exit code {code}; see {run_dir / (stage+'.log')}")
        if stop_path.exists():
            raise PipelineStopped
    except BaseException as error:
        forced = stop_child(process, stage, stop_path) if process is not None else False
        if isinstance(error, KeyboardInterrupt) and not forced:
            raise PipelineStopped from error
        raise
    finally:
        if reader is not None:
            reader.join(timeout=2)
        if process is not None and process.stdout is not None:
            process.stdout.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/convnext_tiny.json"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()
    args.config = args.config.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    run_dir = Path(cfg["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "outputs"
    status_path = run_dir / "pipeline_status.json"
    stop_path = run_dir / "STOP"
    started = time.time()
    state = {"pid": os.getpid(), "config": str(args.config),
             "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}

    def save(**values):
        state.update(values, elapsed_seconds=time.time()-started)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def run(stage, module, arguments):
        run_stage(stage, module, arguments, run_dir, stop_path, save)

    # Rejected duplicate starts must not overwrite the active run's status.
    with exclusive_project_run(args.config):
        try:
            with keep_awake():
                if stop_path.exists():
                    raise PipelineStopped
                run("audit", "cxr.audit", ["--data-root", cfg["data_root"],
                                           "--output", run_dir / "data_audit.json"])
                if not args.skip_prepare:
                    run("prepare", "cxr.prepare", ["--data-root", cfg["data_root"], "--cache-root", cfg["cache_root"],
                                                   "--workers", 4, "--longest-side", 512])
                if not args.skip_train:
                    command = ["--config", args.config]
                    if args.resume:
                        command += ["--resume", run_dir / "last.pt"]
                    run("train", "cxr.train", command)
                    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                    if summary["status"] == "stopped":
                        raise PipelineStopped
                checkpoint = run_dir / "best.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"No validated checkpoint at {checkpoint}")
                run("analysis", "cxr.analyze", ["--predictions", run_dir / "best_valid_predictions.npz",
                                                 "--data-root", cfg["data_root"], "--bootstrap", 200,
                                                 "--output", output_dir / "validation_analysis.json"])
                analysis = json.loads((output_dir / "validation_analysis.json").read_text(encoding="utf-8"))
                choices = analysis["pooling_comparison"]
                aggregation = max(choices, key=lambda key: choices[key]["macro_auc"] or -1)
                save(selected_aggregation=aggregation,
                     selection_note="Pooling selected on official validation; not an independent test estimate")
                common = ["--checkpoint", checkpoint, "--data-root", cfg["data_root"],
                          "--cache-root", cfg["cache_root"], "--batch-size", cfg["batch_size"],
                          "--num-workers", cfg["num_workers"], "--output-dir", output_dir,
                          "--aggregation", aggregation]
                run("valid_predict", "cxr.predict", [*common, "--split", "valid"])
                run("test_predict", "cxr.predict", [*common, "--split", "test"])
                if not args.skip_benchmark:
                    run("benchmark", "cxr.benchmark", ["--checkpoint", checkpoint, "--data-root", cfg["data_root"],
                                                        "--samples", 40, "--output", output_dir / "latency.json"])
                run("report", "cxr.report", ["--run-dir", run_dir])
                run("package", "cxr.package", ["--run-dir", run_dir])
                if stop_path.exists():
                    raise PipelineStopped
                save(status="complete", stage="complete", child_pid=None,
                     submission=str(output_dir / "submission.csv"),
                     bundle=str(output_dir / "inference_bundle.zip"))
        except PipelineStopped:
            save(status="stopped", child_pid=None)
        except BaseException as error:
            save(status="failed", child_pid=None, error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
