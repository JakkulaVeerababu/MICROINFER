"""
MicroInfer - Phase 2 Completion Test Suite
Verifies all Phase 2 files, artifacts, test suites, JSON outputs, and plot images.
"""

import sys
import json
import pytest
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phase2_artifacts_exist():
    """
    Verifies that all Phase 2 code, spec files, JSON results, and plot images exist.
    """
    root = Path(__file__).parent.parent

    # 1. Spec & Code Files
    assert (root / "PHASE2_SPEC.md").exists(), "PHASE2_SPEC.md missing"
    assert (root / "src" / "kv_cache.py").exists(), "src/kv_cache.py missing"
    assert (root / "src" / "cached_generate.py").exists(), "src/cached_generate.py missing"
    assert (root / "benchmarks" / "benchmark_cached.py").exists(), "benchmarks/benchmark_cached.py missing"
    assert (root / "analysis" / "plot_phase2.py").exists(), "analysis/plot_phase2.py missing"

    # 2. Results JSON File
    results_file = root / "benchmarks" / "results" / "phase2_cached.json"
    assert results_file.exists(), "benchmarks/results/phase2_cached.json missing"

    with open(results_file, "r") as f:
        data = json.load(f)
    assert data["phase"] == "Phase 2 - KV-Cache Generator"
    assert len(data["results"]) == 3

    # 3. Plot Chart Files
    chart1 = root / "analysis" / "plots" / "phase2_throughput_comparison.png"
    chart2 = root / "analysis" / "plots" / "phase2_flat_step_latency.png"
    assert chart1.exists() and chart1.stat().st_size > 0
    assert chart2.exists() and chart2.stat().st_size > 0
