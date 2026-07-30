"""
MicroInfer - Phase 3 Completion Test Suite
Verifies all Phase 3 code files, artifacts, test suites, JSON outputs, and plot images.
Updated to match the new concurrent-wave output schema.
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phase3_artifacts_exist():
    """
    Verifies that all Phase 3 code, spec files, JSON results, and plot images exist.
    """
    root = Path(__file__).parent.parent

    # 1. Spec & Code Files
    assert (root / "specs" / "PHASE3_SPEC.md").exists(), "PHASE3_SPEC.md missing"
    assert (root / "src" / "scheduler.py").exists(), "src/scheduler.py missing"
    assert (root / "benchmarks" / "benchmark_scheduler.py").exists(), "benchmarks/benchmark_scheduler.py missing"
    assert (root / "analysis" / "plot_phase3.py").exists(), "analysis/plot_phase3.py missing"

    # 2. Results JSON File
    results_file = root / "benchmarks" / "results" / "phase3_scheduler.json"
    assert results_file.exists(), "benchmarks/results/phase3_scheduler.json missing"

    with open(results_file, "r") as f:
        data = json.load(f)
    assert data["phase"] == "Phase 3 - Dynamic Request Scheduler with Lifecycle Management"

    # New schema: wave_results list instead of the old flat "requests" key
    assert "wave_results" in data, (
        "Expected 'wave_results' key in phase3_scheduler.json. "
        "The file on disk may be from an old benchmark run -- re-run "
        "`python benchmarks/benchmark_scheduler.py` to regenerate it."
    )
    assert len(data["wave_results"]) >= 1
    assert data["wave_results"][0]["n_completed"] >= 1

    # aggregate throughput must be present and positive
    assert "aggregate" in data
    assert data["aggregate"]["throughput_tok_per_sec"]["mean"] > 0.0

    # 3. Plot Chart File
    chart = root / "analysis" / "plots" / "phase3_scheduler_performance.png"
    assert chart.exists() and chart.stat().st_size > 0
