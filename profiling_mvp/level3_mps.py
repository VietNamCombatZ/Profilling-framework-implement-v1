"""Level 3A: pinned-model MPS experiment helpers; no hardware counters."""
from __future__ import annotations

import platform
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import transformers

from .level2 import (Workload, make_inputs, run_once, sync_device,
                     validate_workload, rank_operators)

DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"
DEFAULT_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def require_commit(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Model and tokenizer revision must be an exact 40-character commit SHA")
    return revision.lower()


def select_device(requested: str = "auto") -> tuple[str, str | None]:
    """Explicit CPU fallback; this stage never selects CUDA."""
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError("Level 3A device must be auto, mps, or cpu")
    available = bool(torch.backends.mps.is_available())
    if requested == "cpu":
        return "cpu", None
    if available:
        return "mps", None
    if requested == "mps":
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return "cpu", "MPS runtime unavailable; explicit CPU fallback"


def capabilities() -> dict:
    mps = getattr(torch, "mps", None)
    profiler = getattr(mps, "profiler", None)
    return {
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_runtime_available": bool(torch.backends.mps.is_available()),
        "mps_synchronize_available": callable(getattr(mps, "synchronize", None)),
        "mps_signpost_start_available": callable(getattr(profiler, "start", None)),
        "mps_signpost_stop_available": callable(getattr(profiler, "stop", None)),
        "mps_signpost_context_available": callable(getattr(profiler, "profile", None)),
        "torch_profiler_mps_activity_available": hasattr(torch.profiler.ProfilerActivity, "MPS"),
        "instruments_app_available": Path("/Applications/Xcode.app/Contents/Applications/Instruments.app").exists(),
        "xctrace_available": (shutil.which("xctrace") is not None or
                                Path("/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace").exists()),
    }


@torch.inference_mode()
def capture_signposts(model, tokenizer, workload: Workload, phase: str,
                      device: str, enabled: bool = False,
                      decode_context_steps: int = 0) -> dict:
    """Emit signposts around one forward. Instruments recording is external."""
    info = {"workload": workload.name, "phase": phase, "device": device,
            "status": "disabled" if not enabled else "unavailable", "error": None,
            "capture_kind": "os_signposts", "trace": None,
            "mode": "interval", "wait_until_completed": False}
    if not enabled:
        return info
    if device != "mps" or not torch.backends.mps.is_available():
        info["error"] = "MPS runtime unavailable; no signposts emitted"
        return info
    if not callable(getattr(torch.mps.profiler, "profile", None)):
        info["error"] = "torch.mps.profiler.profile unavailable in installed PyTorch"
        return info
    try:
        validate_workload(model, workload)
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"Unknown phase: {phase}")
        ids, mask = make_inputs(tokenizer, workload.prompt_tokens, workload.batch_size, device)
        run_once(model, ids, mask, workload.max_new_tokens, device)  # warm-up outside capture
        token = past = None
        if phase == "decode":
            if not 0 <= decode_context_steps < workload.max_new_tokens - 1:
                raise ValueError("Decode needs at least one remaining cached step")
            prepared = model(input_ids=ids, attention_mask=mask, use_cache=True)
            past = prepared.past_key_values
            if past is None:
                raise RuntimeError("Model did not return KV cache")
            token = prepared.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(decode_context_steps):
                mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
                prepared = model(input_ids=token, attention_mask=mask,
                                 past_key_values=past, use_cache=True)
                past = prepared.past_key_values
                if past is None:
                    raise RuntimeError("Model stopped returning KV cache")
                token = prepared.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        sync_device(device)
        with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
            if phase == "prefill":
                result = model(input_ids=ids, attention_mask=mask, use_cache=True)
            else:
                result = model(input_ids=token, attention_mask=mask,
                               past_key_values=past, use_cache=True)
            sync_device(device)
        del result
        info["status"] = "emitted"
    except Exception as exc:
        info.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=5))
    return info


def level2_comparison_status(level2_metadata: dict | None, level3_metadata: dict) -> str:
    """Guard against an unjustified numeric CPU/MPS comparison."""
    if level2_metadata is None:
        return "Level 2 metadata unavailable; no cross-level comparison"
    checks = ("model", "model_revision", "tokenizer_revision", "model_dtype",
              "torch_version", "warmup", "repeats", "generation_policy", "timing_scope")
    issues = [key for key in checks if not level2_metadata.get(key)
              or level2_metadata.get(key) != level3_metadata.get(key)]
    if issues:
        return "Cross-level comparison uncontrolled: " + ", ".join(issues)
    return "Metadata controls match; compare individual workload shapes and timing conditions before numeric claims"


def make_metadata(*, model_id: str, revision: str, tokenizer, model,
                  requested_device: str, device: str, fallback_reason: str | None,
                  seed: int, workloads: list[Workload], profile_workloads: list[Workload],
                  warmup: int, repeats: int, signposts_enabled: bool,
                  run_id: str, profile_runs: list[dict], signpost_runs: list[dict],
                  instruments_runs: list[dict] | None = None,
                  instruments_settings: dict | None = None) -> dict:
    revision = require_commit(revision)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    return {
        "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_id, "model_revision": revision,
        "model_revision_resolved_by_library": getattr(model.config, "_commit_hash", None),
        "tokenizer": getattr(tokenizer, "name_or_path", model_id),
        "tokenizer_revision": revision,
        "tokenizer_revision_resolved_by_library": getattr(tokenizer, "_commit_hash", None),
        "model_dtype": ",".join(dtypes) or None, "model_class": type(model).__name__,
        "device_requested": requested_device, "device_selected": device,
        "fallback_reason": fallback_reason, "seed": seed,
        "workloads": [vars(w) for w in workloads],
        "profile_workloads": [vars(w) for w in profile_workloads],
        "warmup": warmup, "repeats": repeats, "signposts_enabled": signposts_enabled,
        "profile_runs": profile_runs, "signpost_runs": signpost_runs,
        "instruments_runs": instruments_runs or [],
        "instruments_settings": instruments_settings or {"enabled": False},
        "capabilities": capabilities(), "python_version": sys.version,
        "macos_version": platform.mac_ver()[0], "platform": platform.platform(),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "pandas_version": pd.__version__,
        "sync_policy": "torch.mps.synchronize" if device == "mps" else "CPU synchronous",
        "generation_policy": "greedy argmax; fixed token count; no EOS stopping",
        "timing_scope": "model forward call plus post-call device synchronization; excludes token selection and mask update",
        "metric_definitions": {
            "prefill_latency_s": "Synchronized wall time for full-prompt model forward; first generated token originates here.",
            "decode_latency_s": "Sum of cached one-token forward wall times for remaining generated tokens; unavailable with one-token generation.",
            "prompt_tokens_per_s": "Batch prompt tokens divided by prefill model-forward wall time.",
            "decode_tokens_per_s": "Batch size times timed cached decode steps divided by their summed wall time.",
            "end_to_end_tokens_per_s": "Actual batch generated tokens divided by prefill plus decode forward wall time; excludes sampling and mask updates.",
            "self_cpu_time_us": "PyTorch profiler host CPU self time in microseconds.",
            "total_cpu_time_us": "PyTorch profiler inclusive host CPU time in microseconds; nested totals overlap.",
            "self_device_time_us": "Unavailable on MPS/CPU with the selected torch.profiler activities; never substituted with CPU time.",
        },
    }


def build_report(meta: dict, summary: pd.DataFrame, operators: pd.DataFrame,
                 profile_runs: list[dict], signpost_runs: list[dict],
                 comparison_status: str, instruments_runs: list[dict] | None = None,
                 resource_summary: pd.DataFrame | None = None) -> str:
    lines = ["# Level 3A MPS profiling report", "",
             f"Run: `{meta['run_id']}`; selected device: `{meta['device_selected']}`; requested: `{meta['device_requested']}`.",
             f"Model/tokenizer: `{meta['model']}` at pinned commit `{meta['model_revision']}`; dtype: `{meta['model_dtype']}`.",
             f"macOS {meta['macos_version']}; Python {meta['python_version'].split()[0]}; PyTorch {meta['torch_version']}.", ""]
    if meta["fallback_reason"]:
        lines += [f"**Observed:** {meta['fallback_reason']}. All results below are CPU results.", ""]
    caps = meta.get("capabilities", {})
    if caps:
        lines += [f"Capabilities: MPS build={caps.get('mps_built')}; runtime={caps.get('mps_runtime_available')}; "
                  f"MPS synchronization={caps.get('mps_synchronize_available')}; "
                  f"OS Signpost context={caps.get('mps_signpost_context_available')}; "
                  f"torch.profiler MPS activity={caps.get('torch_profiler_mps_activity_available')}.", ""]
    lines += ["## Measurement semantics", "",
              f"{meta['warmup']} warm-up and {meta['repeats']} measured repeats per workload. Model loading and tokenization are outside timing.",
              "Prefill times one full-prompt forward and includes the forward that produces the first token's logits. Decode times the remaining cached one-token forwards. Argmax, mask updates, and profiler overhead are outside ordinary timings.",
              "Synchronized wall time includes the Python model call and device synchronization. It is not isolated GPU execution time. Batch rates count tokens over all sequences; per-sequence rates use one sequence over the same batch time.",
              "Fixed-length greedy generation ignores EOS. One-token generation has no Decode timing or rate. Standard deviation uses ddof=1; p90 uses linear interpolation.", "",
              "## Workloads and observed results", ""]
    for _, row in summary.iterrows():
        base = f"`{row.workload}`: {row.runs_ok}/{row.runs_requested} valid; {row.runs_failed} failed, {row.runs_skipped} skipped, {row.runs_unsupported} unsupported"
        if row.runs_ok:
            lines.append(f"- **Observed:** {base}; median Prefill {row.prefill_latency_s_median * 1000:.3f} ms; Decode " +
                         (f"{row.decode_latency_s_median * 1000:.3f} ms" if pd.notna(row.decode_latency_s_median) else "unavailable") +
                         f"; aggregate generated-token rate {row.end_to_end_tokens_per_s_median:.3f} tokens/s.")
        else:
            lines.append(f"- **Observed:** {base}; no valid timing. {row.errors}")
    lines += ["", "## Operator profiles", "",
              "Rankings below sum self CPU time across input-shape groups for a single captured forward. Total CPU time is inclusive and must not be summed across nested events."]
    for info in profile_runs:
        lines.append(f"- `{info['workload']}` {info['phase']}: {info['status']}; trace: `{info['trace']}`; activities: {info['activities_used']}.")
        if info["status"] == "ok":
            for _, op in rank_operators(operators, info["workload"], info["phase"]).head(3).iterrows():
                lines.append(f"  - **Observed:** `{op.operator}` self CPU {op.self_cpu_time_us / 1000:.3f} ms; total CPU {op.total_cpu_time_us / 1000:.3f} ms; {int(op.cpu_calls)} calls.")
    lines += ["", "## MPS Signposts", ""]
    for info in signpost_runs:
        lines.append(f"- `{info['workload']}` {info['phase']}: {info['status']}" + (f" ({info['error']})" if info['error'] else "") + ".")
    lines += ["Signpost emission by itself does not start Instruments or save a trace.",
              "", "## Instruments captures", ""]
    if instruments_runs:
        for info in instruments_runs:
            lines.append(f"- `{info['workload']}` {info['phase']}: {info['status']}; "
                         f"recording started={info['recording_started']}; signposts emitted={info['signposts_emitted']}; "
                         f"trace saved={info['trace_saved']}; PyTorchMPS intervals found="
                         f"{info['expected_signpost_events_present']}; trace validated={info['trace_validated']}; "
                         f"trace=`{info['trace']}`; error={info['error']}.")
    else:
        lines.append("No Instruments recording was attempted in this run. Start a recording manually or enable the separate automatic capture pass.")
    resource_settings = meta.get("resource_monitoring")
    if resource_settings is not None:
        lines += ["", "## Resource monitoring", "",
                  f"Enabled={resource_settings['enabled']}; interval={resource_settings.get('sampling_interval_s')} s; "
                  f"raw observations={resource_settings.get('sample_count')}; "
                  f"sampler errors={resource_settings.get('sampler_errors', [])}."]
        for _, item in resource_summary.iterrows() if resource_summary is not None else []:
            lines.append(f"- `{item.workload}` {item.phase}: {item.status}; "
                         f"periodic samples={item.periodic_sample_count}; "
                         f"RSS={item.process_rss_mib_status}; "
                         f"MPS tensor allocation={item.mps_allocated_mib_status}; "
                         f"Metal driver allocation={item.mps_driver_allocated_mib_status}.")
        lines += ["All memory values use binary MiB (bytes / 2^20). Phase baseline and post snapshots occur outside model-forward timers. Periodic sampling does not synchronize MPS; it can contend with measured execution and miss brief spikes. Short workloads may have no periodic samples.",
                  "Process RSS includes the whole Python process. PyTorch MPS current allocation excludes cached allocator blocks. Metal driver allocation includes allocator cache and MPS/MPSGraph allocations for the process; it is not total physical GPU memory."]
    artifacts = meta.get("artifacts")
    if artifacts:
        lines += ["", "## Output artifacts", ""]
        for name, path in artifacts.items():
            if name == "charts":
                for chart, detail in path.items():
                    lines.append(f"- `{chart}`: {detail['status']}; "
                                 f"{detail.get('path') or detail.get('reason')}.")
            else:
                lines.append(f"- `{name}`: `{path}`.")
    lines += ["A validated trace contains PyTorchMPS operation intervals from the launched worker. Their presence does not establish isolated GPU execution time or hardware-counter values.",
              "PyTorch CPU operator traces and OS Signposts are separate captures; their timings include profiler overhead and are not ordinary benchmark samples.", "",
              "## Evidence and limitations", "",
              f"- **Observed:** {comparison_status}.",
              "- **Interpretation:** Differences across workloads may reflect workload sensitivity, but require matched controls and repeated runs before a causal claim.",
              "- **Unverified hypothesis:** Kernel launch overhead, memory traffic, or compute work may contribute to high wall time. No hardware counters were collected.",
              "- MPS device operator timing, GPU utilization, L2 cache hit/miss, and DRAM bandwidth are unavailable in this workflow. CPU operator time is never GPU time.",
              "- A saved Instruments trace is validated separately from signpost emission; unavailable GPU/device metrics remain unavailable.", ""]
    return "\n".join(lines)
