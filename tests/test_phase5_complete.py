"""
MicroInfer - Phase 5 Completion Test Suite
Verifies all Phase 5 code files, artifacts, test suites, JSON outputs, and plot images.
Updated to match the new concurrent-wave output schema.
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phase5_artifacts_exist():
    """
    Verifies that all Phase 5 code, spec files, JSON results, and plot images exist.
    """
    root = Path(__file__).parent.parent

    # 1. Spec & Code Files
    assert (root / "specs" / "PHASE5_SPEC.md").exists(), "PHASE5_SPEC.md missing"
    assert (root / "benchmarks" / "baseline_vllm.py").exists(), "benchmarks/baseline_vllm.py missing"
    assert (root / "analysis" / "plot_master.py").exists(), "analysis/plot_master.py missing"
    assert (root / "tests" / "test_master_suite.py").exists(), "tests/test_master_suite.py missing"

    # 2. Results JSON File
    results_file = root / "benchmarks" / "results" / "phase5_vllm.json"
    assert results_file.exists(), "benchmarks/results/phase5_vllm.json missing"

    with open(results_file, "r") as f:
        data = json.load(f)

    assert data["phase"] == "Phase 5 - Fallback Scheduler Under Concurrent Load", (
        f"Unexpected phase string: {data['phase']!r}. "
        "Re-run `python benchmarks/baseline_vllm.py` to regenerate the results file."
    )

    # New schema: wave_results list instead of the old flat "results" key
    assert "wave_results" in data, (
        "Expected 'wave_results' key in phase5_vllm.json. "
        "Re-run `python benchmarks/baseline_vllm.py` to regenerate it."
    )
    assert len(data["wave_results"]) >= 1
    assert data["wave_results"][0]["n_completed"] >= 1

    # aggregate throughput must be present and positive
    assert "aggregate" in data
    assert data["aggregate"]["throughput_tok_per_sec"]["mean"] > 0.0

    # 3. Master Plot Chart Files
    chart1 = root / "analysis" / "plots" / "master_throughput_comparison.png"
    chart2 = root / "analysis" / "plots" / "master_vram_footprint.png"
    assert chart1.exists() and chart1.stat().st_size > 0
    assert chart2.exists() and chart2.stat().st_size > 0
