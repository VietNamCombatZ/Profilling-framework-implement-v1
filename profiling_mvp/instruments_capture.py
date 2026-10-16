"""Bounded xctrace launch and evidence checks for Level 3A MPS Signposts."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .level2 import Workload
from .level3_mps import capture_signposts, require_commit, select_device

TEMPLATE = "Logging"
INSTRUMENT = "os_signpost"
INTERVAL_XPATH = '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost-interval"]'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, timeout: int, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                          cwd=cwd, check=False)


def discover_templates(*, timeout: int = 20) -> tuple[list[str], str | None]:
    if shutil.which("xcrun") is None:
        return [], "xcrun unavailable"
    try:
        result = _run(["xcrun", "xctrace", "list", "templates"], timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"Template discovery failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return [], f"Template discovery exited {result.returncode}: {result.stderr.strip()}"
    names = [line.strip() for line in result.stdout.splitlines()
             if line.strip() and not line.lstrip().startswith("==")]
    return names, None


def _trace_saved(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("*"))


def _export(command: list[str], output: Path, timeout: int) -> tuple[str | None, str | None]:
    try:
        result = _run(command, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return None, f"xctrace export exited {result.returncode}: {result.stderr.strip()}"
    try:
        ET.fromstring(result.stdout)
    except ET.ParseError as exc:
        return None, f"Invalid export XML: {exc}"
    output.write_text(result.stdout)
    return result.stdout, None


def validate_trace(trace: Path, capture_dir: Path, *, timeout: int = 30) -> dict:
    """Inspect TOC and interval rows; an existing .trace alone is insufficient."""
    result = {"trace_saved": _trace_saved(trace), "trace_structurally_readable": False,
              "expected_signpost_events_present": False, "trace_validated": False,
              "signpost_interval_count": 0, "matching_signpost_interval_count": 0,
              "sample_signpost_names": [], "process_pid": None,
              "trace_toc": None, "interval_export": None,
              "validation_error": None, "gpu_device_time_available": None,
              "hardware_counters_available": None}
    if not result["trace_saved"]:
        result["validation_error"] = "Trace missing or empty"
        return result
    toc_path = capture_dir / "trace_toc.xml"
    toc, error = _export(["xcrun", "xctrace", "export", "--input", str(trace), "--toc"], toc_path, timeout)
    if error:
        result["validation_error"] = error
        return result
    root = ET.fromstring(toc)
    if root.tag != "trace-toc" or root.find("run") is None:
        result["validation_error"] = "No trace run in table of contents"
        return result
    result["trace_structurally_readable"] = True
    result["trace_toc"] = str(toc_path)
    target = root.find(".//target/process")
    if target is not None:
        result["process_pid"] = target.get("pid")
    schemas = {x.get("schema") for x in root.findall(".//data/table")}
    if "os-signpost-interval" not in schemas:
        result["validation_error"] = "Trace has no os-signpost-interval table"
        return result
    intervals_path = capture_dir / "signpost_intervals.xml"
    intervals, error = _export(["xcrun", "xctrace", "export", "--input", str(trace),
                                "--xpath", INTERVAL_XPATH], intervals_path, timeout)
    if error:
        result["validation_error"] = error
        return result
    result["interval_export"] = str(intervals_path)
    interval_root = ET.fromstring(intervals)
    rows = interval_root.findall(".//row")
    result["signpost_interval_count"] = len(rows)
    # Observed in the installed Xcode export: PyTorch uses the PyTorchMPS
    # subsystem and PyTorchOperationIntervals name. Other MPSGraph rows can
    # arise during model setup and are not proof of the scoped PyTorch forward.
    ids = {part.get("id"): part for part in interval_root.iter() if part.get("id")}

    def display(part):
        while part is not None and part.get("ref"):
            part = ids.get(part.get("ref"))
        return part.get("fmt") or part.text if part is not None else None

    matching = [row for row in rows
                if display(row.find("subsystem")) == "PyTorchMPS"
                and display(row.find("string")) == "PyTorchOperationIntervals"]
    result["matching_signpost_interval_count"] = len(matching)
    result["sample_signpost_names"] = [
        next((display(part) for part in row.findall(".//os-log-metadata/string")
              if display(part)), "PyTorchOperationIntervals")
        for row in matching[:10]]
    result["expected_signpost_events_present"] = bool(matching)
    result["trace_validated"] = bool(matching)
    if not matching:
        result["validation_error"] = "No PyTorch/MPS signpost interval found in exported rows"
    return result


def capture_instruments(*, model_id: str, revision: str, workload: Workload,
                        phase: str, output_dir: Path, run_id: str,
                        time_limit_s: int = 90, timeout_s: int = 150) -> dict:
    """Launch a fresh worker under xctrace; never touches benchmark timings."""
    revision = require_commit(revision)
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    if not 1 <= time_limit_s < timeout_s:
        raise ValueError("Require 1 <= time_limit_s < timeout_s")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id must be a safe path component")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", workload.name):
        raise ValueError("workload name must be a safe path component")
    capture_dir = output_dir / "instruments" / run_id / f"{workload.name}_{phase}_{uuid.uuid4().hex[:8]}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    trace = capture_dir / "capture.trace"
    worker_status = capture_dir / "worker_status.json"
    info = {"workload": workload.name, "phase": phase, "device": "mps",
            "status": "unavailable", "error": None, "template": None,
            "instrument": INSTRUMENT,
            "template_discovery_command": ["xcrun", "xctrace", "list", "templates"],
            "templates_available": None, "capture_dir": str(capture_dir),
            "trace": None, "command": None, "target_executable": sys.executable,
            "recording_started": False, "signposts_emitted": None,
            "trace_saved": False, "trace_structurally_readable": False,
            "expected_signpost_events_present": False, "trace_validated": False,
            "gpu_device_time_available": None, "hardware_counters_available": None,
            "start_utc": None, "end_utc": None, "exit_status": None,
            "process_pid": None, "worker_status_path": str(worker_status)}
    if not select_device("auto")[0] == "mps":
        info["error"] = "MPS runtime unavailable; benchmark remains usable without Instruments"
        return info
    templates, error = discover_templates()
    info["templates_available"] = templates
    if error or TEMPLATE not in templates:
        info["error"] = error or f"Required template {TEMPLATE!r} absent; available: {templates}"
        return info
    info["template"] = TEMPLATE
    command = ["xcrun", "xctrace", "record", "--template", TEMPLATE,
               "--instrument", INSTRUMENT, "--time-limit", f"{time_limit_s}s",
               "--no-prompt", "--output", str(trace), "--launch", "--",
               sys.executable, "-m", "profiling_mvp.instruments_capture", "worker",
               "--model-id", model_id, "--revision", revision,
               "--experiment", workload.experiment,
               "--prompt-tokens", str(workload.prompt_tokens),
               "--max-new-tokens", str(workload.max_new_tokens),
               "--batch-size", str(workload.batch_size), "--phase", phase,
               "--status-json", str(worker_status)]
    info["command"] = command
    info["start_utc"] = _now()
    try:
        proc = _run(command, timeout=timeout_s, cwd=Path(__file__).resolve().parents[1])
        info["exit_status"] = proc.returncode
        info["xctrace_stdout"] = proc.stdout[-8000:]
        info["xctrace_stderr"] = proc.stderr[-8000:]
        info["recording_started"] = "Starting recording" in proc.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        info["recording_started"] = "Starting recording" in output
        info["trace_saved"] = _trace_saved(trace)
        if info["trace_saved"]:
            info["trace"] = str(trace)
        info.update(status="failed", error=f"xctrace timeout after {timeout_s}s: {exc}", end_utc=_now())
        return info
    except OSError as exc:
        info.update(status="failed", error=f"xctrace launch failed: {type(exc).__name__}: {exc}", end_utc=_now())
        return info
    info["end_utc"] = _now()
    if worker_status.is_file():
        try:
            worker = json.loads(worker_status.read_text())
            info["signposts_emitted"] = worker.get("status") == "emitted"
            info["worker_error"] = worker.get("error")
            info["worker_pid"] = worker.get("pid")
        except (OSError, json.JSONDecodeError) as exc:
            info["worker_error"] = f"Worker status unreadable: {exc}"
    if proc.returncode != 0:
        info["trace_saved"] = _trace_saved(trace)
        if info["trace_saved"]:
            info["trace"] = str(trace)
        info.update(status="failed", error=f"xctrace exited {proc.returncode}: {proc.stderr.strip()}")
        return info
    details = validate_trace(trace, capture_dir)
    info.update(details)
    if info["process_pid"] and info.get("worker_pid") and str(info["worker_pid"]) != str(info["process_pid"]):
        info.update(trace_validated=False, expected_signpost_events_present=False,
                    validation_error="Trace target PID differs from signpost worker PID")
    if info["trace_saved"]:
        info["trace"] = str(trace)
    if not info["recording_started"] or not info["trace_saved"]:
        info.update(status="failed", error=info.get("validation_error") or "Recording or trace missing")
    elif not info["signposts_emitted"]:
        info.update(status="failed", error=info.get("worker_error") or "Worker did not confirm Signpost emission")
    elif info["trace_validated"]:
        info["status"] = "validated"
    else:
        info.update(status="trace_saved_unverified", error=info.get("validation_error"))
    return info


def worker_main(args: argparse.Namespace) -> int:
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    status_path = Path(args.status_json)
    info = {"status": "failed", "error": None, "pid": os.getpid(),
            "start_utc": _now(), "end_utc": None}
    try:
        select_device("mps")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision).to("mps").eval()
        workload = Workload(args.experiment, args.prompt_tokens, args.max_new_tokens, args.batch_size)
        result = capture_signposts(model, tokenizer, workload, args.phase, "mps", enabled=True)
        info.update(result)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    info["end_utc"] = _now()
    status_path.write_text(json.dumps(info, indent=2))
    return 0 if info["status"] == "emitted" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, type=require_commit)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=["prefill", "decode"])
    parser.add_argument("--status-json", required=True)
    return worker_main(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
