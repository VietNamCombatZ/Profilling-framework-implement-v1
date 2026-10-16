"""Instruments capture state checks; all external processes are mocked."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from profiling_mvp.instruments_capture import (capture_instruments,
    discover_templates, validate_trace)
from profiling_mvp.level2 import Workload
from profiling_mvp.level3_mps import DEFAULT_REVISION


class InstrumentsCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kwargs = dict(model_id="sshleifer/tiny-gpt2", revision=DEFAULT_REVISION,
                           workload=Workload("test", 32, 4, 1), phase="prefill",
                           output_dir=self.root, run_id="test_run", time_limit_s=5,
                           timeout_s=10)

    def test_xcrun_unavailable(self):
        with patch("profiling_mvp.instruments_capture.shutil.which", return_value=None):
            names, error = discover_templates()
        self.assertEqual(names, [])
        self.assertIn("xcrun unavailable", error)

    def test_missing_template_keeps_benchmark_independent(self):
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates",
                    return_value=(["Time Profiler"], None))):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["trace_saved"])
        self.assertIsNone(result["command"])
        self.assertIn("Logging", result["error"])

    def test_mps_unavailable_never_starts_recording(self):
        with patch("profiling_mvp.instruments_capture.select_device", return_value=("cpu", "fallback")):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["recording_started"])

    def test_timeout(self):
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates", return_value=(["Logging"], None)),
              patch("profiling_mvp.instruments_capture._run",
                    side_effect=subprocess.TimeoutExpired(["xcrun"], 10))):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timeout", result["error"])
        self.assertIsNone(result["exit_status"])

    def test_nonzero_exit(self):
        completed = subprocess.CompletedProcess([], 7, "Starting recording", "recording denied")
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates", return_value=(["Logging"], None)),
              patch("profiling_mvp.instruments_capture._run", return_value=completed)):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_status"], 7)
        self.assertFalse(result["trace_saved"])
        self.assertIn("recording denied", result["error"])

    def test_missing_trace_and_worker_status(self):
        completed = subprocess.CompletedProcess([], 0, "Starting recording", "")
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates", return_value=(["Logging"], None)),
              patch("profiling_mvp.instruments_capture._run", return_value=completed)):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["trace_saved"])
        self.assertFalse(result["trace_validated"])
        self.assertIsNone(result["signposts_emitted"])

    def test_invalid_trace_is_not_validated(self):
        def record(command, **kwargs):
            directory = Path(command[command.index("--output") + 1]).parent
            trace = directory / "capture.trace"
            trace.mkdir()
            (trace / "marker").write_text("data")
            (directory / "worker_status.json").write_text(json.dumps({"status": "emitted", "pid": 123}))
            return subprocess.CompletedProcess(command, 0, "Starting recording", "")
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates", return_value=(["Logging"], None)),
              patch("profiling_mvp.instruments_capture._run", side_effect=record),
              patch("profiling_mvp.instruments_capture.validate_trace", return_value={
                  "trace_saved": True, "trace_structurally_readable": False,
                  "expected_signpost_events_present": False, "trace_validated": False,
                  "process_pid": None, "validation_error": "Invalid export XML"})):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "trace_saved_unverified")
        self.assertTrue(result["trace_saved"])
        self.assertFalse(result["trace_validated"])

    def test_successful_mocked_capture_has_run_specific_path_and_stages(self):
        def record(command, **kwargs):
            self.assertIn("--launch", command)
            self.assertIn("--time-limit", command)
            self.assertEqual(command[command.index("--time-limit") + 1], "5s")
            directory = Path(command[command.index("--output") + 1]).parent
            trace = directory / "capture.trace"
            trace.mkdir()
            (trace / "marker").write_text("data")
            (directory / "worker_status.json").write_text(json.dumps({"status": "emitted", "pid": 123}))
            return subprocess.CompletedProcess(command, 0, "Starting recording", "")
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates", return_value=(["Logging"], None)),
              patch("profiling_mvp.instruments_capture._run", side_effect=record),
              patch("profiling_mvp.instruments_capture.validate_trace", return_value={
                  "trace_saved": True, "trace_structurally_readable": True,
                  "expected_signpost_events_present": True, "trace_validated": True,
                  "process_pid": "123", "matching_signpost_interval_count": 2,
                  "validation_error": None})):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "validated")
        self.assertTrue(all(result[key] for key in
                            ("signposts_emitted", "recording_started", "trace_saved", "trace_validated")))
        self.assertIn("/instruments/test_run/", result["trace"])
        self.assertEqual(result["exit_status"], 0)
        self.assertIsNone(result["gpu_device_time_available"])
        self.assertIsNone(result["hardware_counters_available"])

    def test_pid_mismatch_never_validates(self):
        def record(command, **kwargs):
            directory = Path(command[command.index("--output") + 1]).parent
            trace = directory / "capture.trace"
            trace.mkdir()
            (trace / "marker").write_text("data")
            (directory / "worker_status.json").write_text(json.dumps({"status": "emitted", "pid": 123}))
            return subprocess.CompletedProcess(command, 0, "Starting recording", "")
        with (patch("profiling_mvp.instruments_capture.select_device", return_value=("mps", None)),
              patch("profiling_mvp.instruments_capture.discover_templates", return_value=(["Logging"], None)),
              patch("profiling_mvp.instruments_capture._run", side_effect=record),
              patch("profiling_mvp.instruments_capture.validate_trace", return_value={
                  "trace_saved": True, "trace_structurally_readable": True,
                  "expected_signpost_events_present": True, "trace_validated": True,
                  "process_pid": "999", "validation_error": None})):
            result = capture_instruments(**self.kwargs)
        self.assertEqual(result["status"], "trace_saved_unverified")
        self.assertFalse(result["trace_validated"])
        self.assertIn("PID differs", result["error"])

    def test_trace_xml_requires_pytorchmps_intervals(self):
        trace = self.root / "capture.trace"
        trace.mkdir()
        (trace / "marker").write_text("data")
        toc = '<trace-toc><run number="1"><info><target><process pid="123"/></target></info><data><table schema="os-signpost-interval"/></data></run></trace-toc>'
        interval = '''<trace-query-result><node><row><string id="1" fmt="PyTorchOperationIntervals"/><subsystem id="2" fmt="PyTorchMPS"/><os-log-metadata><string fmt="aten::add"/></os-log-metadata></row><row><string ref="1"/><subsystem ref="2"/><os-log-metadata><string fmt="aten::mul"/></os-log-metadata></row></node></trace-query-result>'''
        def exported(command, **kwargs):
            return subprocess.CompletedProcess(command, 0,
                toc if "--toc" in command else interval, "")
        with patch("profiling_mvp.instruments_capture._run", side_effect=exported):
            result = validate_trace(trace, self.root)
        self.assertTrue(result["trace_validated"])
        self.assertEqual(result["matching_signpost_interval_count"], 2)
        self.assertEqual(result["process_pid"], "123")
        self.assertIsNone(result["gpu_device_time_available"])

    def test_invalid_export_xml_is_not_structurally_readable(self):
        trace = self.root / "capture.trace"
        trace.mkdir()
        (trace / "marker").write_text("data")
        with patch("profiling_mvp.instruments_capture._run", return_value=
                   subprocess.CompletedProcess([], 0, "not XML", "")):
            result = validate_trace(trace, self.root)
        self.assertTrue(result["trace_saved"])
        self.assertFalse(result["trace_structurally_readable"])
        self.assertFalse(result["trace_validated"])
        self.assertIn("Invalid export XML", result["validation_error"])


if __name__ == "__main__":
    unittest.main()
