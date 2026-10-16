"""Optional Level 3 resource observations and plots; no GPU telemetry inference."""
from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch

from .level2 import Workload

MIB = 2 ** 20
MEMORY_METRICS = ("process_rss_mib", "mps_allocated_mib", "mps_driver_allocated_mib")
SAMPLE_COLUMNS = ("run_id", "timestamp_utc", "elapsed_s", "workload", "run_index",
                  "device", "phase", "sample_kind", *MEMORY_METRICS,
                  *(f"{metric}_status" for metric in MEMORY_METRICS))


def _availability_status(statuses) -> str:
    values = [str(value) for value in statuses if pd.notna(value)]
    if not values:
        return "unavailable"
    if all(value == "ok" for value in values):
        return "ok"
    reasons = " | ".join(sorted(set(values) - {"ok"}))
    return f"partial: {reasons}" if "ok" in values else reasons


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value_in_mib(getter) -> tuple[float | None, str]:
    if not callable(getter):
        return None, "api_unavailable"
    try:
        byte_count = getter()
        if not isinstance(byte_count, (int, float)) or not math.isfinite(byte_count) or byte_count < 0:
            raise ValueError(f"invalid byte count: {byte_count!r}")
        return byte_count / MIB, "ok"
    except Exception as exc:
        return None, f"error: {type(exc).__name__}: {exc}"


def read_memory(device: str, process=None) -> dict:
    """Current-process RSS and MPS byte counters converted to binary MiB."""
    try:
        proc = process if process is not None else psutil.Process(os.getpid())
        rss = _value_in_mib(lambda: proc.memory_info().rss)
    except Exception as exc:
        rss = (None, f"error: {type(exc).__name__}: {exc}")
    if device == "mps":
        mps = getattr(torch, "mps", None)
        allocated = _value_in_mib(getattr(mps, "current_allocated_memory", None))
        driver = _value_in_mib(getattr(mps, "driver_allocated_memory", None))
    else:
        allocated = driver = (None, "backend_unavailable")
    return {"process_rss_mib": rss[0], "process_rss_mib_status": rss[1],
            "mps_allocated_mib": allocated[0], "mps_allocated_mib_status": allocated[1],
            "mps_driver_allocated_mib": driver[0], "mps_driver_allocated_mib_status": driver[1]}


class ResourceSampler:
    """Samples a single measured run; phase snapshots are outside Level 2 timers."""

    def __init__(self, device: str, run_id: str, workload: Workload,
                 run_index: int, interval_s: float):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("sampling interval must be finite and positive")
        self.device, self.run_id, self.workload = device, run_id, workload
        self.run_index, self.interval_s = run_index, interval_s
        self.rows: list[dict] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase = "run"
        self._start = None
        self._thread = None
        try:
            self._process = psutil.Process(os.getpid())
        except Exception as exc:
            self._process = None
            self.errors.append(f"Process handle unavailable: {type(exc).__name__}: {exc}")

    def _record(self, phase: str, kind: str) -> None:
        values = read_memory(self.device, self._process)
        when = time.monotonic()
        row = {"run_id": self.run_id, "timestamp_utc": _utc(),
               "elapsed_s": when - self._start, "workload": self.workload.name,
               "run_index": self.run_index, "device": self.device,
               "phase": phase, "sample_kind": kind, **values}
        with self._lock:
            self.rows.append(row)

    def _periodic(self) -> None:
        try:
            while not self._stop.wait(self.interval_s):
                with self._lock:
                    phase = self._phase
                self._record(phase, "periodic")
        except Exception as exc:
            self.errors.append(f"Sampler thread failed: {type(exc).__name__}: {exc}")

    def __enter__(self):
        self._start = time.monotonic()
        self._record("run", "baseline")
        self._thread = threading.Thread(target=self._periodic, name="level3-resource-sampler", daemon=True)
        try:
            self._thread.start()
        except Exception as exc:
            self.errors.append(f"Sampler thread could not start: {type(exc).__name__}: {exc}")
            self._thread = None
        return self

    def phase_event(self, phase: str, event: str) -> None:
        if event not in {"start", "end"} or phase not in {"prefill", "decode"}:
            raise ValueError("Invalid phase event")
        if event == "start":
            with self._lock:
                self._phase = phase
        self._record(phase, f"phase_{event}")
        if event == "end":
            with self._lock:
                self._phase = "between_phases"

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
            if self._thread.is_alive():
                self.errors.append("Sampler thread did not stop within timeout")
        self._record("run", "post")
        return False


class ResourceMonitor:
    def __init__(self, device: str, run_id: str, interval_s: float):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("sampling interval must be finite and positive")
        self.device, self.run_id, self.interval_s = device, run_id, interval_s
        self.samplers: list[ResourceSampler] = []

    def observer_factory(self, workload: Workload, run_index: int) -> ResourceSampler:
        sampler = ResourceSampler(self.device, self.run_id, workload, run_index, self.interval_s)
        self.samplers.append(sampler)
        return sampler

    def samples_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([row for sampler in self.samplers for row in sampler.rows],
                            columns=SAMPLE_COLUMNS)

    def run_statuses(self) -> list[dict]:
        rows = []
        for sampler in self.samplers:
            entry = {"run_id": self.run_id, "workload": sampler.workload.name,
                     "run_index": sampler.run_index, "sample_count": len(sampler.rows),
                     "periodic_sample_count": sum(row["sample_kind"] == "periodic" for row in sampler.rows),
                     "sampler_errors": sampler.errors}
            for metric in MEMORY_METRICS:
                entry[f"{metric}_status"] = _availability_status(
                    row[f"{metric}_status"] for row in sampler.rows)
            rows.append(entry)
        return rows


def summarize_resources(samples: pd.DataFrame, raw: pd.DataFrame,
                        workloads: list[Workload], interval_s: float | None,
                        enabled: bool) -> pd.DataFrame:
    """One row per planned workload/phase; medians across successful repeats."""
    rows = []
    for workload in workloads:
        for phase in ("prefill", "decode"):
            good = raw[(raw.workload == workload.name) & (raw.status == "ok")]
            selected = samples[(samples.workload == workload.name) &
                               (samples.run_index.isin(good.run_index)) &
                               (samples.phase == phase)] if not samples.empty else samples
            periodic = selected[selected.sample_kind == "periodic"] if not selected.empty else selected
            entry = {"run_id": raw.run_id.iloc[0] if len(raw) else None,
                     "workload": workload.name, "phase": phase,
                     "device": raw.device.iloc[0] if len(raw) else None,
                     "prompt_tokens": workload.prompt_tokens,
                     "max_new_tokens": workload.max_new_tokens,
                     "batch_size": workload.batch_size,
                     "benchmark_runs_ok": len(good), "benchmark_runs_requested": len(raw[raw.workload == workload.name]),
                     "periodic_sample_count": len(periodic),
                     "sampling_interval_s": interval_s if enabled else None,
                     "status": ("disabled" if not enabled else
                                "not_applicable_no_decode_steps" if phase == "decode" and workload.max_new_tokens == 1 else
                                "no_valid_benchmark_run" if good.empty else
                                "observed" if not selected.empty else "no_resource_samples")}
            for metric in MEMORY_METRICS:
                for kind, suffix, aggregation in (("phase_start", "baseline_median", "median"),
                                                   ("phase_end", "post_median", "median"),
                                                   ("periodic", "sampled_peak_max", "max")):
                    values = pd.to_numeric(selected.loc[selected.sample_kind == kind, metric], errors="coerce").dropna() if not selected.empty else pd.Series(dtype=float)
                    entry[f"{metric}_{suffix}"] = float(getattr(values, aggregation)()) if len(values) else None
                statuses = selected[f"{metric}_status"].dropna().astype(str) if not selected.empty else pd.Series(dtype=str)
                entry[f"{metric}_valid_sample_count"] = int(selected[metric].notna().sum()) if not selected.empty else 0
                entry[f"{metric}_status"] = _availability_status(statuses)
            rows.append(entry)
    return pd.DataFrame(rows)


def save_charts(summary: pd.DataFrame, resources: pd.DataFrame,
                samples: pd.DataFrame, output_dir: Path, device: str,
                run_id: str) -> dict:
    """Save supported PNGs and report an explicit reason for unavailable charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    valid = summary[summary.runs_ok > 0].reset_index(drop=True)
    if valid.empty:
        result["latency_breakdown.png"] = result["throughput.png"] = {"status": "unavailable", "reason": "No valid benchmark rows"}
    else:
        labels = valid.workload.tolist()
        for filename, metrics, titles, factor, ylabel in (
            ("latency_breakdown.png", ("prefill_latency_s_median", "decode_latency_s_median", "end_to_end_model_time_s_median"),
             ("Prefill", "Cached Decode", "Prefill + Decode"), 1000, "Synchronized model-forward wall time (ms)"),
            ("throughput.png", ("prompt_tokens_per_s_median", "decode_tokens_per_s_median", "end_to_end_tokens_per_s_median"),
             ("Prompt tokens", "Cached Decode tokens", "Generated tokens / model time"), 1, "Aggregate tokens/second"),
        ):
            fig, axes = plt.subplots(1, 3, figsize=(max(12, len(valid) * 1.8), 4.6), sharey=False)
            for ax, metric, title in zip(axes, metrics, titles):
                values = pd.to_numeric(valid[metric], errors="coerce") * factor
                for i, value in enumerate(values):
                    if pd.notna(value):
                        ax.bar(i, value, color="#3274a1")
                    else:
                        ax.text(i, 0, "N/A", ha="center", va="bottom", fontsize=9)
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
            fig.suptitle(f"Observed workload metrics — {device}; run {run_id}")
            fig.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=140)
            plt.close(fig)
            result[filename] = {"status": "saved", "path": str(path)}
    observed = samples[samples.sample_kind.isin(["baseline", "phase_start", "periodic", "phase_end", "post"])] if not samples.empty else samples
    if observed.empty or not observed[list(MEMORY_METRICS)].notna().any().any():
        result["memory_over_time.png"] = {"status": "unavailable", "reason": "No available resource observations"}
    else:
        runs = observed[["workload", "run_index"]].drop_duplicates()
        first_runs = runs.groupby("workload", sort=False).first().reset_index()
        fig, axes = plt.subplots(len(first_runs), 3,
                                 figsize=(14, max(3.8, 2.8 * len(first_runs))), squeeze=False)
        colors = {"process_rss_mib": "#3274a1", "mps_allocated_mib": "#dd8452", "mps_driver_allocated_mib": "#55a868"}
        for row_axes, (_, run) in zip(axes, first_runs.iterrows()):
            group = observed[(observed.workload == run.workload) & (observed.run_index == run.run_index)].sort_values("elapsed_s")
            for column, (ax, metric) in enumerate(zip(row_axes, MEMORY_METRICS)):
                series = group[["elapsed_s", metric]].dropna()
                if len(series):
                    ax.scatter(series.elapsed_s, series[metric], s=20, color=colors[metric])
                else:
                    ax.text(.5, .5, "Unavailable", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9)
                for _, boundary in group[group.sample_kind == "phase_start"].iterrows():
                    ax.axvline(boundary.elapsed_s, color="gray", linewidth=.8, linestyle="--")
                name = metric.replace("_mib", "")
                ax.set_title((f"{run.workload} · repeat {int(run.run_index)}\n{name}"
                              if column == 0 else name), fontsize=9)
                ax.set_ylabel("MiB")
                ax.set_xlabel("Elapsed time (s)")
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.suptitle("Memory snapshots and periodic samples; dashed lines mark phase starts", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .965), h_pad=2.5)
        path = output_dir / "memory_over_time.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        result[path.name] = {"status": "saved", "path": str(path)}
    if resources.empty or not resources[[f"{metric}_baseline_median" for metric in MEMORY_METRICS]].notna().any().any():
        result["memory_summary.png"] = {"status": "unavailable", "reason": "No available phase memory snapshots"}
    else:
        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(resources) * 1.25), 10), squeeze=False)
        labels = [f"{row.workload}\n{row.phase}" for _, row in resources.iterrows()]
        for ax, metric in zip(axes[:, 0], MEMORY_METRICS):
            any_value = False
            for offset, suffix, name, color in (
                (-.25, "baseline_median", "Baseline snapshot median", "#4c78a8"),
                (0, "post_median", "Post snapshot median", "#f58518"),
                (.25, "sampled_peak_max", "Periodic sampled maximum", "#54a24b")):
                values = pd.to_numeric(resources[f"{metric}_{suffix}"], errors="coerce")
                first_label = True
                for i, value in enumerate(values):
                    if pd.notna(value):
                        ax.bar(i + offset, value, width=.23, color=color,
                               label=name if first_label else None)
                        first_label = False
                        any_value = True
            if any_value:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "Unavailable", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10)
            for i, row in resources.iterrows():
                if all(pd.isna(row[f"{metric}_{suffix}"]) for suffix in
                       ("baseline_median", "post_median", "sampled_peak_max")):
                    ax.text(i, 0, "N/A", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(f"{metric} (MiB)")
        fig.suptitle("Phase memory observations; sampled maximum may miss brief spikes")
        fig.tight_layout()
        path = output_dir / "memory_summary.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        result[path.name] = {"status": "saved", "path": str(path)}
    return result
