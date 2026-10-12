"""Focused Level 2 correctness tests; no model download needed."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from profiling_mvp import level2


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.arange(512).unsqueeze(0))


class Model:
    def __init__(self, fail_call=None, cache=True):
        self.config = SimpleNamespace(n_positions=128)
        self.calls = []
        self.fail_call = fail_call
        self.cache = cache

    def __call__(self, *, input_ids, attention_mask, use_cache, past_key_values=None):
        self.calls.append((tuple(input_ids.shape), tuple(attention_mask.shape), past_key_values))
        if self.fail_call == len(self.calls):
            raise RuntimeError("injected failure")
        if past_key_values is not None:
            assert attention_mask.shape[1] == past_key_values + 1
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2))
        logits[..., 1] = 1
        past = (0 if past_key_values is None else past_key_values) + input_ids.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=past if self.cache else None)


class Level2Tests(unittest.TestCase):
    def test_matrix_counts_and_skipped_reasons(self):
        workloads, plan = level2.make_workload_matrix([32, 64, 128], [8, 16, 32], [1, 2], 64, 8)
        self.assertEqual(len(workloads), 8)
        self.assertEqual(len(plan), 18)
        self.assertEqual(plan.status.value_counts().to_dict(), {"skipped": 12, "selected": 6})
        selected = plan[plan.status == "selected"]
        self.assertEqual(sum(len(x.split(" | ")) for x in selected.workloads), 8)
        self.assertTrue(plan.loc[plan.status == "skipped", "reason"].notna().all())

    def test_timing_batch_throughput_and_cache_masks(self):
        ids, mask = level2.make_inputs(Tokenizer(), 4, 2, "cpu")
        model = Model()
        with patch.object(level2.time, "perf_counter", side_effect=[0, 2, 3, 5, 6, 9]), \
             patch.object(level2, "memory_snapshot", return_value={}):
            row = level2.run_once(model, ids, mask, 3, "cpu")
        self.assertEqual(model.calls, [((2, 4), (2, 4), None),
                                       ((2, 1), (2, 5), 4),
                                       ((2, 1), (2, 6), 5)])
        self.assertEqual(row["prefill_latency_s"], 2)
        self.assertEqual(row["decode_latency_s"], 5)
        self.assertEqual(row["decode_steps_timed"], 2)
        self.assertEqual(row["decode_steps_per_s"], 2 / 5)
        self.assertEqual(row["decode_tokens_per_s"], 4 / 5)
        self.assertEqual(row["prompt_tokens_per_s"], 8 / 2)
        self.assertEqual(row["prompt_tokens_per_sequence_per_s"], 4 / 2)
        self.assertEqual(row["end_to_end_tokens_per_s"], 6 / 7)
        self.assertEqual(row["end_to_end_tokens_per_sequence_per_s"], 3 / 7)
        self.assertEqual(row["generated_tokens_actual_total"], 6)
        self.assertEqual(row["prompt_tokens_actual_per_sequence"], 4)

    def test_one_token_and_missing_cache(self):
        ids, mask = level2.make_inputs(Tokenizer(), 4, 2, "cpu")
        row = level2.run_once(Model(), ids, mask, 1, "cpu")
        self.assertEqual(row["decode_steps_timed"], 0)
        self.assertIsNone(row["decode_latency_s"])
        self.assertIsNone(row["decode_tokens_per_s"])
        self.assertIsNone(row["decode_step_latency_mean_s"])
        self.assertEqual(row["generated_tokens_actual_total"], 2)
        with self.assertRaisesRegex(RuntimeError, "KV cache"):
            level2.run_once(Model(cache=False), ids, mask, 2, "cpu")
        raw = level2.benchmark_workloads(Model(), Tokenizer(),
                                         [level2.Workload("one", 4, 1, 1)], "cpu", 0, 1)
        raw.loc[0, "decode_step_latency_p50_s"] = 0
        with self.assertRaisesRegex(ValueError, "Decode metrics present"):
            level2.summarize_workloads(raw)

    def test_complete_run_records_and_failure_status(self):
        workloads = [level2.Workload("valid", 4, 1, 1),
                     level2.Workload("invalid", 128, 2, 1)]
        raw = level2.benchmark_workloads(Model(fail_call=2), Tokenizer(), workloads, "cpu", 0, 3)
        self.assertEqual(len(raw), 6)
        self.assertEqual(raw.groupby("workload").run_index.apply(list).iloc[0], [1, 2, 3])
        self.assertEqual(set(raw[raw.experiment == "valid"].status), {"ok", "failed"})
        self.assertEqual(set(raw[raw.experiment == "invalid"].status), {"unsupported"})
        self.assertTrue(raw.loc[raw.status != "ok", "error"].notna().all())
        summary, _ = level2.summarize_workloads(raw)
        valid = summary[summary.experiment == "valid"].iloc[0]
        self.assertEqual((valid.runs_ok, valid.runs_failed, valid.runs_skipped), (2, 1, 0))
        self.assertEqual(summary[summary.experiment == "invalid"].runs_unsupported.iloc[0], 3)
        with patch.object(level2, "make_inputs", side_effect=RuntimeError("preparation failed")):
            skipped = level2.benchmark_workloads(Model(), Tokenizer(), [workloads[0]], "cpu", 1, 2)
        self.assertEqual(skipped.status.tolist(), ["skipped", "skipped"])
        with self.assertRaisesRegex(ValueError, "Incomplete run records"):
            level2.summarize_workloads(raw.iloc[:-1])

    def test_missing_metric_aggregation_and_report(self):
        workload = level2.Workload("one", 4, 1, 1)
        raw = level2.benchmark_workloads(Model(), Tokenizer(), [workload], "cpu", 0, 2)
        summary, stability = level2.summarize_workloads(raw)
        self.assertEqual(summary.decode_tokens_per_s_count.iloc[0], 0)
        self.assertTrue(pd.isna(summary.decode_tokens_per_s_median.iloc[0]))
        meta = {"timestamp_utc": "now", "model": "toy", "device_selected": "cpu", "torch_version": torch.__version__,
                "warmup": 0, "repeats": 2, "profiler_capabilities": level2.profiler_capabilities("cpu")}
        report = level2.build_report(meta, summary, stability, pd.DataFrame(), [])
        self.assertIn("decode throughput unavailable", report)
        self.assertNotIn("nan tokens/s", report)

    def test_invalid_timing_and_inconsistent_raw_rate(self):
        ids, mask = level2.make_inputs(Tokenizer(), 4, 1, "cpu")
        with patch.object(level2.time, "perf_counter", side_effect=[1, 1]):
            with self.assertRaisesRegex(ValueError, "Prefill duration"):
                level2.run_once(Model(), ids, mask, 1, "cpu")
        workload = level2.Workload("invalid_rate", 4, 2, 1)
        raw = level2.benchmark_workloads(Model(), Tokenizer(), [workload], "cpu", 0, 2)
        raw.loc[0, "decode_tokens_per_s"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid decode_tokens_per_s"):
            level2.summarize_workloads(raw)
        raw.loc[0, "decode_tokens_per_s"] = raw.loc[0, "decode_steps_per_s"] * 2
        with self.assertRaisesRegex(ValueError, "Inconsistent decode_tokens_per_s"):
            level2.summarize_workloads(raw)

    def test_statistics_use_sample_std_and_linear_p90(self):
        workload = level2.Workload("stats", 4, 1, 1)
        raw = level2.benchmark_workloads(Model(), Tokenizer(), [workload], "cpu", 0, 4)
        for index, value in enumerate((1., 2., 3., 4.)):
            raw.loc[index, "prefill_latency_s"] = value
            raw.loc[index, "end_to_end_model_time_s"] = value
            raw.loc[index, "prompt_tokens_per_s"] = 4 / value
            raw.loc[index, "prompt_tokens_per_sequence_per_s"] = 4 / value
            raw.loc[index, "end_to_end_tokens_per_s"] = 1 / value
            raw.loc[index, "end_to_end_tokens_per_sequence_per_s"] = 1 / value
        summary, _ = level2.summarize_workloads(raw)
        row = summary.iloc[0]
        self.assertEqual(row.prefill_latency_s_mean, 2.5)
        self.assertEqual(row.prefill_latency_s_median, 2.5)
        self.assertAlmostEqual(row.prefill_latency_s_std, 1.2909944487)
        self.assertEqual(row.prefill_latency_s_min, 1)
        self.assertEqual(row.prefill_latency_s_max, 4)
        self.assertAlmostEqual(row.prefill_latency_s_p90, 3.7)

    def test_report_rejects_changed_comparison_controls(self):
        workloads = [level2.Workload("prompt_length", p, 1, 1) for p in (4, 8)]
        raw = level2.benchmark_workloads(Model(), Tokenizer(), workloads, "cpu", 1, 2, run_id="r1")
        summary, stability = level2.summarize_workloads(raw)
        self.assertEqual(level2.comparison_issues(summary, "prompt_tokens"), [])
        changed = summary.copy()
        changed["model_dtype"] = changed["model_dtype"].astype(object)
        changed.loc[1, "model_dtype"] = "torch.float16"
        changed.loc[1, "warmup_runs"] = 0
        changed.loc[1, "device"] = "cuda"
        changed.loc[1, "run_id"] = "r2"
        issues = level2.comparison_issues(changed, "prompt_tokens")
        self.assertTrue(any("model_dtype varies" in issue for issue in issues))
        self.assertTrue(any("warmup_runs varies" in issue for issue in issues))
        self.assertTrue(any("device varies" in issue for issue in issues))
        self.assertTrue(any("run_id varies" in issue for issue in issues))
        meta = {"timestamp_utc": "now", "model": "toy", "device_selected": "cpu", "torch_version": torch.__version__,
                "warmup": 1, "repeats": 2, "profiler_capabilities": level2.profiler_capabilities("cpu")}
        report = level2.build_report(meta, changed, stability, pd.DataFrame(), [])
        self.assertIn("no controlled comparison", report)
        self.assertIn("model_dtype varies", report)

    def test_report_references_only_selected_trace(self):
        workloads = [level2.Workload("prompt_length", 4, 1, 1)]
        raw = level2.benchmark_workloads(Model(), Tokenizer(), workloads, "cpu", 0, 1)
        summary, stability = level2.summarize_workloads(raw)
        meta = {"timestamp_utc": "now", "model": "toy", "device_selected": "cpu", "torch_version": torch.__version__,
                "warmup": 0, "repeats": 1, "profiler_capabilities": level2.profiler_capabilities("cpu")}
        current = "traces/current/prefill.json"
        operators = pd.DataFrame([{"workload": "w", "phase": "prefill", "operator": "aten::add",
                                   "self_cpu_time_us": 10, "total_cpu_time_us": 12, "cpu_calls": 1}])
        report = level2.build_report(meta, summary, stability, operators,
                                     [{"workload": "w", "phase": "prefill", "status": "ok",
                                       "trace": current, "activities_used": ["CPU"]}])
        self.assertIn(current, report)
        self.assertNotIn("traces/older/prefill.json", report)

    def test_operator_fields_are_backend_specific(self):
        event = SimpleNamespace(key="aten::add", input_shapes=[[2, 4]], count=3,
            self_cpu_time_total=10, cpu_time_total=17, self_device_time_total=6,
            device_time_total=8, cpu_memory_usage=20, self_cpu_memory_usage=12,
            device_memory_usage=40, self_device_memory_usage=30)
        workload = level2.Workload("test", 4, 2, 1)
        trace = Path("trace.json")
        cpu = level2.operator_row(event, workload, "prefill", "cpu", trace, [level2.ProfilerActivity.CPU])
        mps = level2.operator_row(event, workload, "prefill", "mps", trace, [level2.ProfilerActivity.CPU])
        cuda = level2.operator_row(event, workload, "decode", "cuda", trace,
                                   [level2.ProfilerActivity.CPU, level2.ProfilerActivity.CUDA])
        self.assertEqual((cpu["self_cpu_time_us"], cpu["total_cpu_time_us"], cpu["cpu_calls"]), (10, 17, 3))
        self.assertEqual(json.loads(cpu["input_shapes"]), [[2, 4]])
        self.assertIsNone(cpu["self_device_time_us"])
        self.assertIsNone(mps["self_device_time_us"])
        self.assertEqual(cuda["self_device_time_us"], 6)
        self.assertEqual(cuda["total_device_time_us"], 8)
        self.assertIsNone(cuda["device_calls"])

    def test_rank_combines_shape_groups_without_summing_other_operators(self):
        rows = pd.DataFrame([
            {"workload": "w", "phase": "prefill", "operator": "aten::add", "input_shapes": "[2]",
             "self_cpu_time_us": 6, "total_cpu_time_us": 9, "cpu_calls": 1},
            {"workload": "w", "phase": "prefill", "operator": "aten::add", "input_shapes": "[4]",
             "self_cpu_time_us": 7, "total_cpu_time_us": 10, "cpu_calls": 2},
            {"workload": "w", "phase": "prefill", "operator": "aten::mul", "input_shapes": "[2]",
             "self_cpu_time_us": 10, "total_cpu_time_us": 11, "cpu_calls": 1},
            {"workload": "w", "phase": "prefill", "operator": "model_forward_prefill", "input_shapes": "[]",
             "self_cpu_time_us": 100, "total_cpu_time_us": 200, "cpu_calls": 1},
        ])
        ranked = level2.rank_operators(rows, "w", "prefill")
        self.assertEqual(ranked.operator.tolist(), ["aten::add", "aten::mul"])
        self.assertEqual(ranked.iloc[0][["self_cpu_time_us", "total_cpu_time_us", "cpu_calls"]].tolist(), [13, 19, 3])

    def test_profiler_extraction_failure_is_not_marked_ok(self):
        class BrokenProfile:
            def key_averages(self, **kwargs):
                raise RuntimeError("bad event table")

        workload = level2.Workload("trace", 4, 2, 1)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(level2, "_capture_forward", return_value=(BrokenProfile(), [level2.ProfilerActivity.CPU])):
            rows, info = level2.profile_phase(Model(), Tokenizer(), workload, "prefill", "cpu", Path(directory))
        self.assertFalse(rows)
        self.assertEqual(info["status"], "failed")
        self.assertIn("bad event table", info["error"])

    def test_prefill_and_decode_traces_are_separate(self):
        model = Model()
        workload = level2.Workload("trace", 4, 3, 1)
        with tempfile.TemporaryDirectory() as directory:
            for phase in ("prefill", "decode"):
                rows, info = level2.profile_phase(model, Tokenizer(), workload, phase, "cpu", Path(directory))
                self.assertEqual(info["status"], "ok", info["error"])
                self.assertTrue(rows)
                names = {event.get("name") for event in json.loads(Path(info["trace"]).read_text())["traceEvents"]}
                self.assertIn(f"model_forward_{phase}", names)
                other = "decode" if phase == "prefill" else "prefill"
                self.assertNotIn(f"model_forward_{other}", names)
                self.assertTrue(all(row["self_device_time_us"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
