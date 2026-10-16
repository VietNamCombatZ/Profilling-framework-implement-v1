"""Deterministic resource sampling and visualization checks; no real MPS required."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.image as mpimg
import pandas as pd
import torch

from profiling_mvp.level2 import Workload, benchmark_workloads, run_once
from profiling_mvp.level3_mps import build_report
from profiling_mvp.resource_monitor import (MIB, ResourceMonitor, ResourceSampler,
    SAMPLE_COLUMNS, read_memory, save_charts, summarize_resources)


class TinyModel:
    config = SimpleNamespace(n_positions=128)

    def __call__(self, *, input_ids, attention_mask, use_cache, past_key_values=None):
        past = input_ids.shape[1] if past_key_values is None else past_key_values + 1
        return SimpleNamespace(past_key_values=past,
                               logits=torch.ones((*input_ids.shape, 2)))


class ResourceTests(unittest.TestCase):
    def test_mps_values_are_binary_mib_and_distinct(self):
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=5 * MIB))
        with (patch("torch.mps.current_allocated_memory", return_value=2 * MIB),
              patch("torch.mps.driver_allocated_memory", return_value=3 * MIB)):
            values = read_memory("mps", process)
        self.assertEqual([values[k] for k in ("process_rss_mib", "mps_allocated_mib", "mps_driver_allocated_mib")], [5, 2, 3])
        self.assertTrue(all(values[f"{k}_status"] == "ok" for k in
                            ("process_rss_mib", "mps_allocated_mib", "mps_driver_allocated_mib")))

    def test_api_and_rss_failures_remain_null(self):
        process = SimpleNamespace(memory_info=lambda: (_ for _ in ()).throw(OSError("rss denied")))
        with (patch("torch.mps.current_allocated_memory", side_effect=RuntimeError("mps denied")),
              patch("torch.mps.driver_allocated_memory", None)):
            values = read_memory("mps", process)
        self.assertIsNone(values["process_rss_mib"])
        self.assertIsNone(values["mps_allocated_mib"])
        self.assertIsNone(values["mps_driver_allocated_mib"])
        self.assertIn("rss denied", values["process_rss_mib_status"])
        self.assertIn("mps denied", values["mps_allocated_mib_status"])
        self.assertEqual(values["mps_driver_allocated_mib_status"], "api_unavailable")
        cpu = read_memory("cpu", SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=MIB)))
        self.assertIsNone(cpu["mps_allocated_mib"])
        self.assertEqual(cpu["mps_allocated_mib_status"], "backend_unavailable")

    def test_sampler_cleanup_and_phase_snapshots_on_exception(self):
        sampler = ResourceSampler("cpu", "run", Workload("test", 2, 2, 1), 1, .001)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with sampler as active:
                active.phase_event("prefill", "start")
                time.sleep(.004)
                raise RuntimeError("injected")
        self.assertFalse(sampler._thread.is_alive())
        kinds = {row["sample_kind"] for row in sampler.rows}
        self.assertTrue({"baseline", "phase_start", "post", "periodic"}.issubset(kinds))
        self.assertEqual({row["workload"] for row in sampler.rows}, {"test_p2_n2_b1"})
        self.assertEqual({row["run_index"] for row in sampler.rows}, {1})

    def test_one_token_has_no_decode_events_or_rate(self):
        ids = torch.ones((2, 3), dtype=torch.long)
        events = []
        result = run_once(TinyModel(), ids, torch.ones_like(ids), 1, "cpu",
                          phase_observer=lambda phase, event: events.append((phase, event)))
        self.assertEqual(events, [("prefill", "start"), ("prefill", "end")])
        self.assertIsNone(result["decode_latency_s"])
        self.assertIsNone(result["decode_tokens_per_s"])
        self.assertEqual(result["generated_tokens_actual_total"], 2)
        self.assertAlmostEqual(result["end_to_end_tokens_per_s"], 2 / result["end_to_end_model_time_s"])

    def test_observer_only_wraps_measured_repeats_and_preserves_rates(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=torch.ones((1, 256), dtype=torch.long))

        workload = Workload("test", 3, 3, 2)
        monitor = ResourceMonitor("cpu", "run", .1)
        raw = benchmark_workloads(TinyModel(), Tokenizer(), [workload], "cpu",
                                  warmup=1, repeats=2, run_id="run",
                                  observer_factory=monitor.observer_factory)
        samples = monitor.samples_dataframe()
        self.assertTrue(raw.status.eq("ok").all())
        self.assertEqual(len(monitor.samplers), 2)
        self.assertEqual(set(samples.run_index), {1, 2})
        self.assertEqual(samples[samples.sample_kind == "baseline"].shape[0], 2)
        self.assertEqual(samples[samples.sample_kind == "post"].shape[0], 2)
        self.assertEqual(samples[samples.sample_kind == "phase_start"].phase.tolist(),
                         ["prefill", "decode", "prefill", "decode"])
        statuses = monitor.run_statuses()
        self.assertEqual(len(statuses), 2)
        self.assertEqual({item["process_rss_mib_status"] for item in statuses}, {"ok"})
        self.assertEqual({item["mps_allocated_mib_status"] for item in statuses}, {"backend_unavailable"})
        self.assertTrue((raw.generated_tokens_actual_total == 6).all())
        for _, row in raw.iterrows():
            self.assertAlmostEqual(row.decode_tokens_per_s,
                                   row.decode_tokens_timed_total / row.decode_latency_s)
            self.assertAlmostEqual(row.end_to_end_tokens_per_s,
                                   row.generated_tokens_actual_total / row.end_to_end_model_time_s)

    def test_summary_keeps_missing_peak_and_one_token_decode_unavailable(self):
        workload = Workload("one_token", 3, 1, 1)
        raw = pd.DataFrame([{"run_id": "run", "workload": workload.name, "run_index": 1,
                             "status": "ok", "device": "cpu"}])
        monitor = ResourceMonitor("cpu", "run", .01)
        with monitor.observer_factory(workload, 1) as sampler:
            sampler.phase_event("prefill", "start")
            sampler.phase_event("prefill", "end")
        samples = monitor.samples_dataframe()
        summary = summarize_resources(samples, raw, [workload], .01, True)
        prefill = summary[summary.phase == "prefill"].iloc[0]
        decode = summary[summary.phase == "decode"].iloc[0]
        self.assertEqual(prefill.process_rss_mib_status, "ok")
        self.assertTrue(pd.isna(prefill.mps_allocated_mib_baseline_median))
        self.assertTrue(pd.isna(prefill.process_rss_mib_sampled_peak_max))
        self.assertEqual(decode.status, "not_applicable_no_decode_steps")
        self.assertEqual(decode.periodic_sample_count, 0)
        self.assertTrue(pd.isna(decode.mps_driver_allocated_mib_post_median))

    def test_disabled_monitoring_has_explicit_summary(self):
        workload = Workload("test", 2, 2, 1)
        raw = pd.DataFrame([{"run_id": "run", "workload": workload.name,
                             "run_index": 1, "status": "ok", "device": "cpu"}])
        summary = summarize_resources(pd.DataFrame(columns=SAMPLE_COLUMNS), raw,
                                      [workload], None, False)
        self.assertEqual(set(summary.status), {"disabled"})
        self.assertTrue(summary.process_rss_mib_baseline_median.isna().all())
        self.assertTrue(summary.mps_allocated_mib_sampled_peak_max.isna().all())

    def test_charts_skip_missing_metrics_and_use_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "runs" / "specific_run"
            benchmark = pd.DataFrame([{"workload": "one", "runs_ok": 1,
                "prefill_latency_s_median": .01, "decode_latency_s_median": None,
                "end_to_end_model_time_s_median": .01,
                "prompt_tokens_per_s_median": 300, "decode_tokens_per_s_median": None,
                "end_to_end_tokens_per_s_median": 100}])
            resource = pd.DataFrame([{"workload": "one", "phase": "prefill",
                "process_rss_mib_baseline_median": None,
                "mps_allocated_mib_baseline_median": None,
                "mps_driver_allocated_mib_baseline_median": None}])
            result = save_charts(benchmark, resource, pd.DataFrame(columns=SAMPLE_COLUMNS),
                                 target, "cpu", "specific_run")
            for name in ("latency_breakdown.png", "throughput.png"):
                self.assertEqual(result[name]["status"], "saved")
                self.assertGreater(mpimg.imread(result[name]["path"]).size, 0)
                self.assertIn("specific_run", result[name]["path"])
            self.assertEqual(result["memory_over_time.png"]["status"], "unavailable")
            self.assertEqual(result["memory_summary.png"]["status"], "unavailable")
            self.assertFalse((target / "memory_summary.png").exists())

    def test_report_states_resource_status_and_limits(self):
        meta = {"run_id": "run", "device_selected": "mps", "device_requested": "mps",
                "model": "model", "model_revision": "0" * 40,
                "model_dtype": "torch.float32", "macos_version": "26",
                "python_version": "3.12", "torch_version": "2.14",
                "fallback_reason": None, "warmup": 1, "repeats": 1,
                "resource_monitoring": {"enabled": True, "sampling_interval_s": .01,
                                        "sample_count": 3, "sampler_errors": []},
                "resource_artifacts": {"resource_samples_csv": "runs/run/resource_samples.csv",
                                       "resource_summary_csv": "runs/run/resource_summary.csv",
                                       "charts": {"memory_over_time.png": {"status": "unavailable", "reason": "no data"}}},
                "artifacts": {"resource_samples_csv": "runs/run/resource_samples.csv",
                              "charts": {"memory_over_time.png": {"status": "unavailable", "reason": "no data"}}}}
        benchmark = pd.DataFrame([{"workload": "bad", "runs_ok": 0, "runs_requested": 1,
                                   "runs_failed": 1, "runs_skipped": 0,
                                   "runs_unsupported": 0, "errors": "injected"}])
        resources = pd.DataFrame([{"workload": "bad", "phase": "prefill",
                                   "status": "no_valid_benchmark_run", "periodic_sample_count": 0,
                                   "process_rss_mib_status": "unavailable",
                                   "mps_allocated_mib_status": "unavailable",
                                   "mps_driver_allocated_mib_status": "unavailable"}])
        report = build_report(meta, benchmark, pd.DataFrame(), [], [],
                              "uncontrolled", [], resources)
        self.assertIn("sampling", report.lower())
        self.assertIn("periodic samples=0", report)
        self.assertIn("runs/run/resource_samples.csv", report)
        self.assertIn("not total physical GPU memory", report)
        self.assertIn("not isolated GPU execution time", report)


if __name__ == "__main__":
    unittest.main()
