"""
MicroInfer - Phase 4 Completion Test Suite
Verifies all Phase 4 code files, artifacts, test suites, JSON outputs, and plot images.
"""

import sys
import json
import pytest
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phase4_artifacts_exist():
    """
    Verifies that all Phase 4 code, spec files, JSON results, and plot images exist.
    """
    root = Path(__file__).parent.parent

    # 1. Spec & Code Files
    assert (root / "specs" / "PHASE4_SPEC.md").exists(), "PHASE4_SPEC.md missing"
    assert (root / "src" / "quant_loader.py").exists(), "src/quant_loader.py missing"
    assert (root / "src" / "quant_generate.py").exists(), "src/quant_generate.py missing"
    assert (root / "benchmarks" / "benchmark_quant.py").exists(), "benchmarks/benchmark_quant.py missing"
    assert (root / "analysis" / "plot_phase4.py").exists(), "analysis/plot_phase4.py missing"

    # 2. Results JSON File
    results_file = root / "benchmarks" / "results" / "phase4_quant.json"
    assert results_file.exists(), "benchmarks/results/phase4_quant.json missing"

    with open(results_file, "r") as f:
        data = json.load(f)
    assert data["phase"] == "Phase 4 - INT8 Quantized Engine"
    assert len(data["results"]) == 3

    # 3. Plot Chart File
    chart = root / "analysis" / "plots" / "phase4_vram_reduction.png"
    assert chart.exists() and chart.stat().st_size > 0
