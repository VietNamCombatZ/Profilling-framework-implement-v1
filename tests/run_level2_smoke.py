"""Execute Level 2 code cells in process and verify generated CPU artifacts.

Run from the repository root with its virtual environment. This avoids Jupyter's
kernel socket requirement in restricted CI environments.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import nbformat
import pandas as pd

from profiling_mvp.level2 import build_report


def file_hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduced", action="store_true", help="Run a smaller CPU matrix under /private/tmp")
    args = parser.parse_args()
    notebook = nbformat.read("notebooks/llm_profiling_mvp_level2.ipynb", as_version=4)
    nbformat.validate(notebook)
    level1 = [Path("notebooks/llm_profiling_mvp_level1.ipynb")]
    level1 += [path for root in (Path("results/notebook_level1"), Path("notebooks/results"))
               if root.exists() for path in root.rglob("*") if path.is_file()]
    before = file_hashes(level1)
    scope = {"display": lambda value: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            ast.parse(source, filename=f"cell-{index}")
            if args.reduced and index == 2:
                replacements = {
                    'DEVICE_REQUESTED = "auto"': 'DEVICE_REQUESTED = "cpu"',
                    "REPEATS = 3": "REPEATS = 2",
                    "PROMPT_LENGTHS = [32, 64, 128]": "PROMPT_LENGTHS = [32, 64]",
                    "GENERATION_LENGTHS = [8, 16, 32]": "GENERATION_LENGTHS = [8, 16]",
                    'OUTPUT_DIR = Path("results/notebook_level2")':
                        'OUTPUT_DIR = Path("/private/tmp/notebook_level2_validation")',
                }
                for old, new in replacements.items():
                    assert old in source, f"Configuration moved: {old}"
                    source = source.replace(old, new)
            exec(compile(source, f"cell-{index}", "exec"), scope)
    root = Path("/private/tmp/notebook_level2_validation" if args.reduced else "results/notebook_level2")
    expected = ("workload_plan.csv", "raw_runs.csv", "workload_summary.csv", "stability_summary.csv",
                "operator_summary.csv", "profile_runs.json", "metadata.json", "report.md",
                "workload_plots.png", "operator_rankings.png")
    assert all((root / name).is_file() and (root / name).stat().st_size > 0 for name in expected)
    raw = pd.read_csv(root / "raw_runs.csv")
    summary = pd.read_csv(root / "workload_summary.csv")
    stability = pd.read_csv(root / "stability_summary.csv")
    operators = pd.read_csv(root / "operator_summary.csv")
    metadata = json.loads((root / "metadata.json").read_text())
    profiles = json.loads((root / "profile_runs.json").read_text())
    report = (root / "report.md").read_text()
    assert metadata["device_selected"] == "cpu"
    assert len(raw) == len(metadata["workloads"]) * metadata["repeats"]
    assert len(summary) == len(metadata["workloads"])
    assert raw.status.eq("ok").all() and (summary.runs_ok == metadata["repeats"]).all()
    assert raw.run_id.eq(metadata["run_id"]).all()
    assert raw.model_dtype.eq(metadata["model_dtype"]).all()
    assert (raw.generated_tokens_actual_total == raw.batch_size * raw.max_new_tokens).all()
    assert (raw.prompt_tokens_actual_total == raw.batch_size * raw.prompt_tokens_actual_per_sequence).all()
    assert (raw.decode_steps_timed == raw.max_new_tokens - 1).all()
    assert operators.self_device_time_us.isna().all()
    assert len(profiles) == 2 and {item["phase"] for item in profiles} == {"prefill", "decode"}
    for item in profiles:
        assert item["status"] == "ok" and metadata["run_id"] in item["trace"]
        trace = Path(item["trace"])
        assert trace.is_file() and item["trace"] in report
        names = {event.get("name") for event in json.loads(trace.read_text())["traceEvents"]}
        assert f"model_forward_{item['phase']}" in names
        other = "decode" if item["phase"] == "prefill" else "prefill"
        assert f"model_forward_{other}" not in names
    assert report == build_report(metadata, summary, stability, operators, profiles)
    assert before == file_hashes(level1)
    print(f"PASS: {'reduced' if args.reduced else 'default'} CPU run, {len(raw)} measurements, "
          f"2 traces, Level 1 unchanged; run_id={metadata['run_id']}")


if __name__ == "__main__":
    main()
