"""Level 2 workload and operator profiling. No hardware-counter claims."""
from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil
import torch
import transformers
from torch.profiler import ProfilerActivity, profile, record_function


@dataclass(frozen=True)
class Workload:
    experiment: str
    prompt_tokens: int
    max_new_tokens: int
    batch_size: int

    @property
    def name(self) -> str:
        return f"{self.experiment}_p{self.prompt_tokens}_n{self.max_new_tokens}_b{self.batch_size}"


def make_workload_matrix(prompt_lengths: list[int], generation_lengths: list[int],
                         batch_sizes: list[int], base_prompt: int,
                         base_new_tokens: int) -> tuple[list[Workload], pd.DataFrame]:
    """Return labeled one-variable sweeps and every candidate's selection reason."""
    if not prompt_lengths or not generation_lengths or not batch_sizes:
        raise ValueError("Each workload dimension needs at least one value")
    if min(*prompt_lengths, *generation_lengths, *batch_sizes, base_prompt, base_new_tokens) < 1:
        raise ValueError("Workload dimensions must be positive")
    workloads = list(dict.fromkeys(
        [Workload("prompt_length", p, base_new_tokens, 1) for p in prompt_lengths]
        + [Workload("generation_length", base_prompt, n, 1) for n in generation_lengths]
        + [Workload("batch_size", base_prompt, base_new_tokens, b) for b in batch_sizes]
    ))
    labels = {}
    for workload in workloads:
        key = (workload.prompt_tokens, workload.max_new_tokens, workload.batch_size)
        labels.setdefault(key, []).append(workload.name)
    plan = pd.DataFrame([
        {"prompt_tokens": p, "max_new_tokens": n, "batch_size": b,
         "status": "selected" if (p, n, b) in labels else "skipped",
         "reason": None if (p, n, b) in labels else "Not selected by one-variable sweeps",
         "workloads": " | ".join(labels.get((p, n, b), [])) or None}
        for p in sorted(set(prompt_lengths + [base_prompt]))
        for n in sorted(set(generation_lengths + [base_new_tokens]))
        for b in sorted(set(batch_sizes + [1]))
    ])
    assert plan.status.eq("selected").sum() == len(labels)
    return workloads, plan


def choose_device(requested: str = "auto") -> str:
    available = {
        "cpu": True,
        "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        return next(device for device in ("mps", "cuda", "cpu") if available[device])
    if requested not in available or not available[requested]:
        raise ValueError(f"Device {requested!r} is invalid or unavailable: {available}")
    return requested


def sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def make_inputs(tokenizer, prompt_tokens: int, batch_size: int, device: str):
    if prompt_tokens < 1 or batch_size < 1:
        raise ValueError("prompt_tokens and batch_size must be positive")
    seed_text = (
        "Profiling measures the performance of language model inference. "
        "The experiment records latency, throughput, and memory usage. "
    ) * max(8, prompt_tokens // 8 + 4)
    ids = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] < prompt_tokens:
        raise ValueError(f"Only {ids.shape[1]} prompt tokens available; requested {prompt_tokens}")
    # Identical, unpadded sequences make every attention-mask row exactly prompt_tokens long.
    ids = ids[:, :prompt_tokens].clone().repeat(batch_size, 1).to(device)
    mask = torch.ones_like(ids)
    assert ids.shape == mask.shape == (batch_size, prompt_tokens)
    return ids, mask


def validate_workload(model, workload: Workload) -> None:
    if workload.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    limit = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    if limit is not None and workload.prompt_tokens + workload.max_new_tokens - 1 > limit:
        raise ValueError(f"Context needs {workload.prompt_tokens + workload.max_new_tokens - 1} positions; model limit is {limit}")


def memory_snapshot(device: str) -> dict:
    result = {"process_rss_mb": psutil.Process().memory_info().rss / 2**20,
              "torch_allocated_mb": None, "torch_reserved_or_driver_mb": None}
    try:
        if device == "cuda":
            result["torch_allocated_mb"] = torch.cuda.memory_allocated() / 2**20
            result["torch_reserved_or_driver_mb"] = torch.cuda.memory_reserved() / 2**20
        elif device == "mps":
            result["torch_allocated_mb"] = torch.mps.current_allocated_memory() / 2**20
            if hasattr(torch.mps, "driver_allocated_memory"):
                result["torch_reserved_or_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    except (AttributeError, RuntimeError):
        pass
    return result


@torch.inference_mode()
def run_once(model, ids, mask, max_new_tokens: int, device: str) -> dict:
    """Same phase boundary as Level 1; one batch forward per phase/step."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    sync_device(device)
    start = time.perf_counter()
    output = model(input_ids=ids, attention_mask=mask, use_cache=True)
    sync_device(device)
    prefill_s = time.perf_counter() - start
    if not math.isfinite(prefill_s) or prefill_s <= 0:
        raise ValueError("Prefill duration must be finite and positive")
    past = output.past_key_values
    if max_new_tokens > 1 and past is None:
        raise RuntimeError("Model did not return a KV cache for Decode")
    token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del output
    step_times = []
    for _ in range(max_new_tokens - 1):
        mask = torch.cat((mask, torch.ones((ids.shape[0], 1), dtype=mask.dtype, device=device)), dim=1)
        sync_device(device)
        start = time.perf_counter()
        output = model(input_ids=token, attention_mask=mask, past_key_values=past, use_cache=True)
        sync_device(device)
        step_s = time.perf_counter() - start
        if not math.isfinite(step_s) or step_s <= 0:
            raise ValueError("Decode-step duration must be finite and positive")
        step_times.append(step_s)
        past = output.past_key_values
        if past is None and len(step_times) < max_new_tokens - 1:
            raise RuntimeError("Model stopped returning a KV cache during Decode")
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del output
    decode_s = sum(step_times)
    total_s = prefill_s + decode_s
    if not math.isfinite(total_s) or total_s <= 0:
        raise ValueError("Total model-forward duration must be finite and positive")
    steps = len(step_times)
    batch = ids.shape[0]
    return {
        "status": "ok", "error": None,
        "prompt_tokens_actual_per_sequence": ids.shape[1],
        "prompt_tokens_actual_total": batch * ids.shape[1],
        "input_shape": str(list(ids.shape)), "attention_mask_shape": str([batch, ids.shape[1]]),
        "generated_tokens_actual_total": batch * max_new_tokens,
        "decode_steps_timed": steps, "decode_tokens_timed_total": batch * steps,
        "prefill_latency_s": prefill_s, "decode_latency_s": decode_s if steps else None,
        "decode_step_latency_mean_s": statistics.mean(step_times) if step_times else None,
        "decode_step_latency_p50_s": statistics.median(step_times) if step_times else None,
        "end_to_end_model_time_s": total_s,
        "prompt_tokens_per_sequence_per_s": ids.shape[1] / prefill_s,
        "prompt_tokens_per_s": batch * ids.shape[1] / prefill_s,
        "decode_steps_per_s": steps / decode_s if decode_s > 0 else None,
        "decode_tokens_per_s": batch * steps / decode_s if decode_s > 0 else None,
        "end_to_end_tokens_per_s": batch * max_new_tokens / total_s if total_s > 0 else None,
        "end_to_end_tokens_per_sequence_per_s": max_new_tokens / total_s,
        **memory_snapshot(device),
    }


def _base_row(workload: Workload, device: str, run_index: int | None, warmup: int, repeats: int) -> dict:
    return {**asdict(workload), "workload": workload.name, "device": device,
            "run_index": run_index, "warmup_runs": warmup, "repeats_configured": repeats,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def model_context(model, device: str) -> dict:
    config = getattr(model, "config", None)
    try:
        dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
        dtype = ",".join(dtypes) or None
    except AttributeError:
        dtype = None
    return {"model_name": getattr(config, "name_or_path", None) or type(model).__name__,
            "model_revision": getattr(config, "_commit_hash", None),
            "model_dtype": dtype,
            "sync_policy": "torch.mps.synchronize" if device == "mps" else
                           "torch.cuda.synchronize" if device == "cuda" else "CPU synchronous"}


def benchmark_workloads(model, tokenizer, workloads: list[Workload], device: str,
                        warmup: int, repeats: int, run_id: str | None = None) -> pd.DataFrame:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    rows = []
    for workload in workloads:
        try:
            validate_workload(model, workload)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rows.extend({**_base_row(workload, device, i, warmup, repeats),
                         "status": "unsupported", "error": reason}
                        for i in range(1, repeats + 1))
            continue
        try:
            ids, mask = make_inputs(tokenizer, workload.prompt_tokens, workload.batch_size, device)
            for _ in range(warmup):
                run_once(model, ids, mask, workload.max_new_tokens, device)
            sync_device(device)
        except Exception as exc:
            reason = f"Preparation or warm-up failed: {type(exc).__name__}: {exc}"
            rows.extend({**_base_row(workload, device, i, warmup, repeats),
                         "status": "skipped", "error": reason}
                        for i in range(1, repeats + 1))
            continue
        for run_index in range(1, repeats + 1):
            try:
                rows.append({**_base_row(workload, device, run_index, warmup, repeats),
                             **run_once(model, ids, mask, workload.max_new_tokens, device)})
            except Exception as exc:
                rows.append({**_base_row(workload, device, run_index, warmup, repeats),
                             "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    raw = pd.DataFrame(rows)
    if len(raw) != len(workloads) * repeats:
        raise AssertionError("Measured-run record count differs from configured workloads × repeats")
    for name, value in model_context(model, device).items():
        raw[name] = value
    raw["run_id"] = run_id
    return raw


METRICS = ("prefill_latency_s", "decode_latency_s", "decode_step_latency_mean_s",
           "decode_step_latency_p50_s",
           "end_to_end_model_time_s", "decode_steps_per_s", "decode_tokens_per_s",
           "prompt_tokens_per_s", "prompt_tokens_per_sequence_per_s",
           "end_to_end_tokens_per_s", "end_to_end_tokens_per_sequence_per_s",
           "process_rss_mb", "torch_allocated_mb",
           "torch_reserved_or_driver_mb")
GROUP = ["workload", "experiment", "prompt_tokens", "max_new_tokens", "batch_size", "device",
         "model_name", "model_revision", "model_dtype", "sync_policy", "warmup_runs", "run_id"]


def summarize_workloads(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, stability = [], []
    for key, group in raw.groupby(GROUP, dropna=False, sort=False):
        base = dict(zip(GROUP, key))
        expected = int(group.repeats_configured.iloc[0])
        if len(group) != expected or set(group.run_index) != set(range(1, expected + 1)):
            raise ValueError(f"Incomplete run records for {base['workload']}")
        good = group[group.status == "ok"]
        for _, run in good.iterrows():
            expected_steps = int(run.max_new_tokens) - 1
            if (run.decode_steps_timed != expected_steps or
                run.decode_tokens_timed_total != run.batch_size * expected_steps or
                run.generated_tokens_actual_total != run.batch_size * run.max_new_tokens or
                run.prompt_tokens_actual_per_sequence != run.prompt_tokens):
                raise ValueError(f"Incomplete or mismatched generation in {base['workload']}")
            required = ["prefill_latency_s", "end_to_end_model_time_s",
                        "prompt_tokens_per_s", "prompt_tokens_per_sequence_per_s",
                        "end_to_end_tokens_per_s", "end_to_end_tokens_per_sequence_per_s"]
            if expected_steps:
                required += ["decode_latency_s", "decode_step_latency_mean_s",
                             "decode_step_latency_p50_s", "decode_steps_per_s", "decode_tokens_per_s"]
            for metric in required:
                value = run.get(metric)
                if pd.isna(value) or not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(f"Invalid {metric} in successful run {base['workload']}")
            if not expected_steps and any(pd.notna(run.get(metric)) for metric in
                                          ("decode_latency_s", "decode_step_latency_mean_s",
                                           "decode_step_latency_p50_s", "decode_steps_per_s",
                                           "decode_tokens_per_s")):
                raise ValueError(f"Decode metrics present without Decode steps in {base['workload']}")
            decode_s = float(run.decode_latency_s) if expected_steps else 0.0
            expected_rates = {
                "end_to_end_model_time_s": float(run.prefill_latency_s) + decode_s,
                "prompt_tokens_per_s": float(run.prompt_tokens_actual_total) / float(run.prefill_latency_s),
                "prompt_tokens_per_sequence_per_s": float(run.prompt_tokens_actual_per_sequence) / float(run.prefill_latency_s),
                "end_to_end_tokens_per_s": float(run.generated_tokens_actual_total) / float(run.end_to_end_model_time_s),
                "end_to_end_tokens_per_sequence_per_s": float(run.max_new_tokens) / float(run.end_to_end_model_time_s),
            }
            if expected_steps:
                expected_rates.update({"decode_steps_per_s": expected_steps / decode_s,
                                       "decode_tokens_per_s": float(run.decode_tokens_timed_total) / decode_s,
                                       "decode_step_latency_mean_s": decode_s / expected_steps})
            for metric, expected_value in expected_rates.items():
                if not math.isclose(float(run[metric]), expected_value, rel_tol=1e-8, abs_tol=1e-12):
                    raise ValueError(f"Inconsistent {metric} in successful run {base['workload']}")
        counts = group.status.value_counts()
        common = {**base, "runs_requested": expected,
                  "runs_ok": len(good), "runs_failed": int(counts.get("failed", 0)),
                  "runs_skipped": int(counts.get("skipped", 0)),
                  "runs_unsupported": int(counts.get("unsupported", 0)),
                  "status": "ok" if len(good) == expected else (
                      "partial" if len(good) else "unsupported" if counts.get("unsupported", 0) == expected
                      else "skipped" if counts.get("skipped", 0) == expected else "failed"),
                  "errors": " | ".join(sorted(set(group.error.dropna().astype(str)))) or None,
                  "prompt_tokens_actual_per_sequence": (int(good.prompt_tokens_actual_per_sequence.iloc[0]) if len(good) else None),
                  "prompt_tokens_actual_total": (int(good.prompt_tokens_actual_total.iloc[0]) if len(good) else None),
                  "input_shape": (good.input_shape.iloc[0] if len(good) else None),
                  "generated_tokens_actual_total": (int(good.generated_tokens_actual_total.iloc[0]) if len(good) else None)}
        item, spread = common.copy(), common.copy()
        for metric in METRICS:
            if metric in good:
                numeric = pd.to_numeric(good[metric], errors="coerce")
                if (good[metric].notna() & numeric.isna()).any() or not numeric.dropna().map(math.isfinite).all():
                    raise ValueError(f"Invalid {metric} in successful run {base['workload']}")
                values = numeric.dropna()
            else:
                values = pd.Series(dtype=float)
            item[f"{metric}_count"] = len(values)
            for stat in ("median", "mean", "std", "min", "max", "p90"):
                item[f"{metric}_{stat}"] = (getattr(values, stat)() if stat != "p90" else values.quantile(.9)) if len(values) and (stat != "std" or len(values) > 1) else None
            spread[f"{metric}_cv"] = values.std() / values.mean() if len(values) > 1 and values.mean() != 0 else None
            spread[f"{metric}_range"] = values.max() - values.min() if len(values) else None
        summary.append(item)
        stability.append(spread)
    return pd.DataFrame(summary), pd.DataFrame(stability)


def profiler_capabilities(device: str) -> dict:
    activities = [ProfilerActivity.CPU]
    notes = []
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    elif device == "mps":
        notes.append("torch.profiler has no MPS activity in this environment; CPU-side operator times only. No MPS device-time claim.")
    else:
        notes.append("CPU backend; device-time fields are unavailable, not zero.")
    return {"requested_activities": [str(x).split(".")[-1] for x in activities],
            "record_shapes": True, "profile_memory": True, "notes": notes}


def _event_value(event, *names):
    for name in names:
        try:
            value = getattr(event, name, None)
            if value is not None:
                return value
        except (AttributeError, RuntimeError):
            pass
    return None


def operator_row(event, workload: Workload, phase: str, device: str,
                 trace: Path, activities: list) -> dict:
    """Extract only fields exposed by the active profiler activity."""
    device_active = device == "cuda" and ProfilerActivity.CUDA in activities
    return {"workload": workload.name, "phase": phase, "device": device,
            "operator": event.key,
            "input_shapes": json.dumps(_event_value(event, "input_shapes"), default=str),
            "cpu_calls": _event_value(event, "count"),
            "self_cpu_time_us": _event_value(event, "self_cpu_time_total"),
            "total_cpu_time_us": _event_value(event, "cpu_time_total"),
            "self_device_time_us": _event_value(event, "self_device_time_total") if device_active else None,
            "total_device_time_us": _event_value(event, "device_time_total") if device_active else None,
            "device_calls": None,  # FunctionEventAvg has no separate device-call count.
            "cpu_memory_bytes": _event_value(event, "cpu_memory_usage"),
            "self_cpu_memory_bytes": _event_value(event, "self_cpu_memory_usage"),
            "device_memory_bytes": _event_value(event, "device_memory_usage") if device_active else None,
            "self_device_memory_bytes": _event_value(event, "self_device_memory_usage") if device_active else None,
            "trace": str(trace)}


def profile_phase(model, tokenizer, workload: Workload, phase: str, device: str,
                  trace_dir: Path, decode_context_steps: int = 0) -> tuple[list[dict], dict]:
    """Profile exactly one model forward. Decode cache preparation is outside profile."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    info = {"workload": workload.name, "phase": phase, "device": device,
            "status": "failed", "error": None, "trace": None, "activities_used": None}
    rows = []
    try:
        validate_workload(model, workload)
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"Unknown phase: {phase}")
        ids, mask = make_inputs(tokenizer, workload.prompt_tokens, workload.batch_size, device)
        # Warm this exact workload outside the profiler; discard its wall-clock result.
        run_once(model, ids, mask, workload.max_new_tokens, device)
        token, past = None, None
        if phase == "decode":
            if decode_context_steps < 0 or decode_context_steps >= workload.max_new_tokens - 1:
                raise ValueError("decode_context_steps needs one remaining cached decode step")
            with torch.inference_mode():
                prepared = model(input_ids=ids, attention_mask=mask, use_cache=True)
                past = prepared.past_key_values
                token = prepared.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                for _ in range(decode_context_steps):
                    mask = torch.cat((mask, torch.ones((ids.shape[0], 1), dtype=mask.dtype, device=device)), dim=1)
                    prepared = model(input_ids=token, attention_mask=mask, past_key_values=past, use_cache=True)
                    past = prepared.past_key_values
                    token = prepared.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                mask = torch.cat((mask, torch.ones((ids.shape[0], 1), dtype=mask.dtype, device=device)), dim=1)
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
        # A CUDA profiler failure may still permit CPU-side profiling. Preserve the reason.
        try:
            result, activities_used = _capture_forward(model, ids, mask, token, past, phase, activities, device)
        except Exception as exc:
            if device != "cuda":
                raise
            info["activity_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            result, activities_used = _capture_forward(model, ids, mask, token, past, phase, [ProfilerActivity.CPU], device)
        trace = trace_dir / f"{workload.name}_{phase}_{device}.json"
        for event in result.key_averages(group_by_input_shape=True):
            rows.append(operator_row(event, workload, phase, device, trace, activities_used))
        if not rows:
            raise RuntimeError("Profiler returned no operator events")
        result.export_chrome_trace(str(trace))
        info.update(status="ok", trace=str(trace), activities_used=[str(x).split(".")[-1] for x in activities_used])
    except Exception as exc:
        rows = []
        info["status"] = "failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc(limit=5)
    return rows, info


def _capture_forward(model, ids, mask, token, past, phase, activities, device):
    sync_device(device)
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with torch.inference_mode(), record_function(f"model_forward_{phase}"):
            if phase == "prefill":
                output = model(input_ids=ids, attention_mask=mask, use_cache=True)
            else:
                output = model(input_ids=token, attention_mask=mask, past_key_values=past, use_cache=True)
        sync_device(device)
    del output
    return prof, activities


def rank_operators(operators: pd.DataFrame, workload: str, phase: str) -> pd.DataFrame:
    """Combine disjoint input-shape groups for each operator; rank by self CPU time."""
    subset = operators[(operators.workload == workload) & (operators.phase == phase)]
    subset = subset[subset.operator.str.startswith(("aten::", "prim::"), na=False)]
    if subset.empty:
        return pd.DataFrame(columns=["operator", "self_cpu_time_us", "total_cpu_time_us", "cpu_calls"])
    ranked = subset.groupby("operator", as_index=False)[
        ["self_cpu_time_us", "total_cpu_time_us", "cpu_calls"]
    ].sum(min_count=1)
    return ranked.sort_values("self_cpu_time_us", ascending=False)


def comparison_issues(sweep: pd.DataFrame, axis: str) -> list[str]:
    """Describe changed controls; the named sweep axis alone may vary."""
    controls = {"prompt_tokens", "max_new_tokens", "batch_size", "device",
                "model_name", "model_revision", "model_dtype", "sync_policy",
                "warmup_runs", "runs_requested", "runs_ok", "run_id"} - {axis}
    if axis != "prompt_tokens":
        controls.add("prompt_tokens_actual_per_sequence")
    if axis == "max_new_tokens":
        controls.add("input_shape")
    issues = [f"{control} unavailable" for control in sorted(controls - set(sweep.columns))]
    for control in sorted(controls & set(sweep.columns)):
        values = sweep[control].drop_duplicates()
        if len(values) > 1:
            labels = ["unavailable" if pd.isna(v) else str(v) for v in values]
            issues.append(f"{control} varies ({', '.join(labels)})")
    return issues


def build_report(metadata: dict, summary: pd.DataFrame, stability: pd.DataFrame,
                 operators: pd.DataFrame, profile_runs: list[dict]) -> str:
    def measure(value, unit: str, factor: float = 1) -> str:
        return f"{value * factor:.3f} {unit}" if pd.notna(value) else "unavailable"

    lines = ["# Level 2 profiling report", "", f"Run: {metadata['timestamp_utc']}",
             f"Model: `{metadata['model']}`; revision: `{metadata.get('model_revision') or 'unresolved'}`; "
             f"dtype: `{metadata.get('model_dtype') or 'unavailable'}`; device: `{metadata['device_selected']}`; "
             f"PyTorch: `{metadata['torch_version']}`.",
             "", "## Method", "", f"{metadata['warmup']} warm-up and {metadata['repeats']} measured runs per workload. ",
             "Tokenization, model loading, and profiling are outside benchmark timing. Prefill includes the first generated token's model forward. ",
             "Each timed region includes the Python model call and device synchronization. Argmax token selection, attention-mask updates, and profiler overhead are excluded.",
             "Decode times the remaining cached one-token model forwards. Batch throughput counts tokens across all sequences.",
             "Prompt rate is aggregate prompt tokens divided by the timed Prefill forward; it includes logits computation, not tokenization.",
             "The fixed-length greedy loop does not stop at EOS. Profiler traces contain one forward per phase; the benchmark times all configured Decode forwards.",
             "Only traces listed in this report belong to this result set; older trace files may remain on disk.",
             "", "## Workloads and observed measurements", ""]
    for _, row in summary.iterrows():
        label = (f"`{row.workload}` ({row.status}; {row.runs_ok}/{row.runs_requested} successful, "
                 f"{row.runs_failed} failed, {row.runs_skipped} skipped, {row.runs_unsupported} unsupported)")
        if row.runs_ok:
            lines.append(f"- **Observed:** {label}: median prefill {measure(row.prefill_latency_s_median, 'ms', 1000)}; "
                         f"decode {measure(row.decode_latency_s_median, 'ms', 1000)}; "
                         f"prompt rate {measure(row.prompt_tokens_per_s_median, 'tokens/s')}; "
                         f"decode throughput {measure(row.decode_tokens_per_s_median, 'tokens/s')}; "
                         f"total throughput {measure(row.end_to_end_tokens_per_s_median, 'tokens/s')}.")
        else:
            lines.append(f"- **Observed:** {label}: no valid measurement: {row.errors}.")
    lines += ["", "## Changes within controlled sweeps", ""]
    for experiment, axis, metric, units, factor in (
        ("prompt_length", "prompt_tokens", "prefill_latency_s_median", "ms Prefill", 1000),
        ("generation_length", "max_new_tokens", "decode_latency_s_median", "ms timed Decode", 1000),
        ("batch_size", "batch_size", "decode_tokens_per_s_median", "batch decode tokens/s", 1),
    ):
        sweep = summary[(summary.experiment == experiment) & (summary.runs_ok > 0)].sort_values(axis)
        sweep = sweep[pd.notna(sweep[metric])]
        issues = comparison_issues(sweep, axis)
        if issues:
            lines.append(f"- {experiment}: no controlled comparison: {'; '.join(issues)}.")
            continue
        if len(sweep) >= 2:
            first, last = sweep.iloc[0], sweep.iloc[-1]
            lines.append(f"- **Observed:** {experiment}: {axis} {int(first[axis])} to {int(last[axis])}; "
                         f"median {first[metric] * factor:.3f} to {last[metric] * factor:.3f} {units}. "
                         "Other configured dimensions are fixed within this sweep.")
        else:
            lines.append(f"- {experiment}: fewer than two successful points; no comparison available.")
    lines += ["", "## Repeatability", ""]
    for _, row in stability.iterrows():
        cv = row.get("prefill_latency_s_cv")
        lines.append(f"- `{row.workload}`: prefill CV {cv:.3f}." if pd.notna(cv) else f"- `{row.workload}`: prefill CV unavailable (fewer than two valid runs).")
    lines += ["", "## Operator rankings", ""]
    for info in profile_runs:
        lines.append(f"### {info['phase']}: `{info['workload']}`")
        if info["status"] != "ok":
            lines.append(f"Profiling unavailable: {info['error']}")
            continue
        for _, op in rank_operators(operators, info["workload"], info["phase"]).head(5).iterrows():
            lines.append(f"- **Observed:** `{op.operator}`: self CPU {op.self_cpu_time_us / 1000:.3f} ms; total CPU {op.total_cpu_time_us / 1000:.3f} ms; {op.cpu_calls} calls.")
        lines.append(f"Trace: `{info['trace']}`. Activities: {info['activities_used']}.")
        if info.get("activity_fallback_reason"):
            lines.append(f"CUDA profiler activity unavailable; CPU activity fallback: {info['activity_fallback_reason']}.")
    lines += ["", "## Interpretation and limits", "",
              "- **Interpretation:** Changes within the one-variable sweeps may reflect workload sensitivity in this environment. Their magnitude needs more repeats when variability is high; they do not establish a hardware cause.",
              "- **Unverified hypothesis:** Large operator self time may reflect compute, memory traffic, launch overhead, or a combination. Hardware counters and deeper traces are needed to distinguish them.",
              "- **Observed:** Profiler CPU operator time is different from synchronized wall-clock latency. Nested total times overlap and must not be summed.",
              "- This demo model is tiny and does not represent production LLM performance."]
    if metadata.get("model_revision") is None:
        lines.append("- Model commit hash was unavailable from the loaded configuration; cross-run model revision equivalence is unverified.")
    lines += [f"- {note}" for note in metadata["profiler_capabilities"]["notes"]]
    if metadata["device_selected"] == "cuda":
        lines.append("- CUDA device time is shown only where the CUDA profiler activity succeeded; no hardware counters were collected.")
    lines += ["- Device call counts are unavailable from the aggregate event API used here, even when CUDA time is available.",
              "- Profiler memory fields are allocator event totals, not peak process or physical device memory.", ""]
    return "\n".join(lines)


def metadata(model_name: str, tokenizer, device: str, seed: int, workloads: list[Workload],
             profile_workloads: list[Workload], warmup: int, repeats: int,
             decode_context_steps: int, profile_runs: list[dict], model=None) -> dict:
    context = model_context(model, device) if model is not None else {}
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": model_name,
            "model_revision": context.get("model_revision"), "model_dtype": context.get("model_dtype"),
            "model_class": type(model).__name__ if model is not None else None,
            "sync_policy": context.get("sync_policy"),
            "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
            "device_selected": device, "seed": seed, "workloads": [asdict(x) for x in workloads],
            "profile_workloads": [asdict(x) for x in profile_workloads],
            "warmup": warmup, "repeats": repeats, "decode_context_steps": decode_context_steps,
            "profile_runs": profile_runs, "profiler_capabilities": profiler_capabilities(device),
            "python_version": sys.version, "platform": platform.platform(),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "pandas_version": pd.__version__,
            "generation_policy": "greedy argmax; fixed token count; no EOS stopping",
            "timing_scope": "model forward call plus post-call device synchronization; excludes token selection and mask update",
            "statistics": "median/mean/min/max and linearly interpolated p90; sample standard deviation (ddof=1), unavailable for fewer than two values",
            "metric_definitions": {"prefill_latency_s": "Synchronized wall time of full-prompt forward; first generated token originates here.",
                                   "decode_latency_s": "Sum of synchronized cached one-token forward calls, excluding first generated token; unavailable if no Decode call occurs.",
                                   "decode_steps_per_s": "Timed decode calls per second; batch independent, as in Level 1.",
                                   "decode_tokens_per_s": "Batch size times timed decode calls divided by decode latency.",
                                   "prompt_tokens_per_s": "Batch size times actual prompt tokens per sequence divided by Prefill model-forward time; includes logits work, excludes tokenization.",
                                   "prompt_tokens_per_sequence_per_s": "Actual prompt tokens per sequence divided by batch Prefill model-forward time.",
                                   "end_to_end_tokens_per_s": "Batch size times requested generated tokens divided by prefill plus decode model time.",
                                   "end_to_end_tokens_per_sequence_per_s": "Requested generated tokens per sequence divided by prefill plus decode batch model time.",
                                   "operator_time": "PyTorch profiler aggregate microseconds; CPU and device times must not be conflated."}}
