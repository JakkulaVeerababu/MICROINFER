"""
MicroInfer - Phase 1 Completion Test Suite
Verifies all Phase 1 files, artifacts, test suites, JSON outputs, and plot images.
"""

import sys
import json
import pytest
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phase1_artifacts_exist():
    """
    Verifies that all Phase 1 code, spec files, JSON results, and plot images exist.
    """
    root = Path(__file__).parent.parent

    # 1. Spec & Code Files
    assert (root / "specs" / "PHASE1_SPEC.md").exists(), "PHASE1_SPEC.md missing"
    assert (root / "src" / "naive_generate.py").exists(), "src/naive_generate.py missing"
    assert (root / "benchmarks" / "benchmark_naive.py").exists(), "benchmarks/benchmark_naive.py missing"
    assert (root / "analysis" / "plot_phase1.py").exists(), "analysis/plot_phase1.py missing"

    # 2. Results JSON File
    results_file = root / "benchmarks" / "results" / "phase1_naive.json"
    assert results_file.exists(), "benchmarks/results/phase1_naive.json missing"

    with open(results_file, "r") as f:
        data = json.load(f)
    assert data["phase"] == "Phase 1 - Naive Generator (Uncached)"
    assert len(data["results"]) == 3

    # 3. Plot Chart File
    plot_file = root / "analysis" / "plots" / "phase1_quadratic_scaling.png"
    assert plot_file.exists(), "analysis/plots/phase1_quadratic_scaling.png missing"
    assert plot_file.stat().st_size > 0, "Plot chart file is empty"
