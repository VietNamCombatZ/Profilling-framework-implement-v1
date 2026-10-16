"""Fast Level 3A guards; no model download or MPS device required."""
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from profiling_mvp.level2 import Workload
from profiling_mvp.level3_mps import (DEFAULT_REVISION, build_report,
    capture_signposts, level2_comparison_status, make_metadata, require_commit,
    select_device)


class TinyModel:
    config = SimpleNamespace(_commit_hash=None)

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])


class Level3MPSTests(unittest.TestCase):
    def test_exact_revision_required(self):
        self.assertEqual(require_commit(DEFAULT_REVISION), DEFAULT_REVISION)
        with self.assertRaises(ValueError):
            require_commit("main")

    def test_cpu_fallback_and_required_mps_failure(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"),
                             ("cpu", "MPS runtime unavailable; explicit CPU fallback"))
            with self.assertRaisesRegex(RuntimeError, "MPS requested"):
                select_device("mps")
            self.assertEqual(select_device("cpu"), ("cpu", None))

    def test_unavailable_signposts_never_claim_capture(self):
        w = Workload("test", 2, 2, 1)
        with patch("torch.backends.mps.is_available", return_value=False):
            info = capture_signposts(TinyModel(), None, w, "decode", "cpu", enabled=True)
        self.assertEqual(info["status"], "unavailable")
        self.assertIsNone(info["trace"])
        self.assertIn("MPS runtime unavailable", info["error"])
        self.assertEqual(capture_signposts(TinyModel(), None, w, "prefill", "cpu")["status"], "disabled")

    def test_signpost_decode_scope_has_one_cached_forward(self):
        calls = []
        in_scope = [False]

        class Model:
            config = SimpleNamespace(n_positions=32)

            def __call__(self, *, input_ids, attention_mask, use_cache, past_key_values=None):
                calls.append((in_scope[0], past_key_values, tuple(input_ids.shape)))
                return SimpleNamespace(past_key_values=input_ids.shape[1] if past_key_values is None else past_key_values + 1,
                                       logits=torch.ones((*input_ids.shape, 2)))

        @contextmanager
        def scope(**kwargs):
            self.assertEqual(kwargs, {"mode": "interval", "wait_until_completed": False})
            in_scope[0] = True
            try:
                yield
            finally:
                in_scope[0] = False

        w = Workload("test", 2, 3, 1)
        with (patch("torch.backends.mps.is_available", return_value=True),
              patch("profiling_mvp.level3_mps.make_inputs",
                    return_value=(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long))),
              patch("profiling_mvp.level3_mps.run_once"),
              patch("profiling_mvp.level3_mps.sync_device"),
              patch("torch.mps.profiler.profile", side_effect=scope)):
            result = capture_signposts(Model(), None, w, "decode", "mps", enabled=True)
        self.assertEqual(result["status"], "emitted")
        self.assertEqual(sum(flag for flag, _, _ in calls), 1)
        self.assertEqual([past for flag, past, _ in calls if flag], [2])
        self.assertEqual([shape for flag, _, shape in calls if flag], [(1, 1)])

    def test_metadata_complete_and_device_time_unavailable(self):
        w = Workload("test", 2, 2, 1)
        meta = make_metadata(model_id="model", revision=DEFAULT_REVISION,
            tokenizer=SimpleNamespace(name_or_path="model", _commit_hash=None),
            model=TinyModel(), requested_device="auto", device="cpu",
            fallback_reason="MPS runtime unavailable", seed=3, workloads=[w],
            profile_workloads=[w], warmup=1, repeats=2, signposts_enabled=False,
            run_id="run", profile_runs=[], signpost_runs=[])
        self.assertEqual(meta["model_revision"], DEFAULT_REVISION)
        self.assertEqual(meta["tokenizer_revision"], DEFAULT_REVISION)
        self.assertEqual(meta["device_selected"], "cpu")
        self.assertIsNone(meta["model_revision_resolved_by_library"])
        self.assertIn("Unavailable", meta["metric_definitions"]["self_device_time_us"])
        self.assertIsNotNone(meta["macos_version"])

    def test_cross_level_comparison_rejects_unresolved_revision(self):
        current = {"model": "model", "model_revision": DEFAULT_REVISION,
                   "tokenizer_revision": DEFAULT_REVISION, "model_dtype": "torch.float32",
                   "torch_version": "2.14", "warmup": 1, "repeats": 2,
                   "generation_policy": "fixed", "timing_scope": "forward"}
        old = dict(current, model_revision=None)
        self.assertIn("model_revision", level2_comparison_status(old, current))
        self.assertIn("metadata unavailable", level2_comparison_status(None, current))

    def test_report_preserves_failed_workload_and_no_false_gpu_time(self):
        meta = {"run_id": "abc", "device_selected": "cpu", "device_requested": "auto",
                "model": "model", "model_revision": DEFAULT_REVISION,
                "model_dtype": "torch.float32", "macos_version": "26", "python_version": "3.12",
                "torch_version": "2.14", "fallback_reason": "MPS unavailable",
                "warmup": 1, "repeats": 2}
        summary = pd.DataFrame([{"workload": "bad", "runs_ok": 0, "runs_requested": 2,
            "runs_failed": 2, "runs_skipped": 0, "runs_unsupported": 0,
            "errors": "failure"}])
        report = build_report(meta, summary, pd.DataFrame(), [], [], "uncontrolled")
        self.assertIn("bad", report)
        self.assertIn("2 failed", report)
        self.assertIn("failure", report)
        self.assertIn("All results below are CPU results", report)
        self.assertIn("CPU operator time is never GPU time", report)


if __name__ == "__main__":
    unittest.main()
