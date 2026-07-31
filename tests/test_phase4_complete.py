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


def test_phase4_accuracy_json():
    """
    Validates that the INT8 perplexity accuracy evaluation JSON exists and
    contains the required keys with sensible values.
    Run `python benchmarks/quant_accuracy.py` to regenerate if missing.
    """
    root = Path(__file__).parent.parent
    accuracy_file = root / "benchmarks" / "results" / "phase4_accuracy.json"
    assert accuracy_file.exists(), (
        "benchmarks/results/phase4_accuracy.json missing — "
        "run: python benchmarks/quant_accuracy.py"
    )

    with open(accuracy_file, "r") as f:
        data = json.load(f)

    required_keys = [
        "fp16_perplexity", "int8_perplexity",
        "perplexity_delta_abs", "perplexity_delta_pct",
        "verdict", "corpus_tokens",
    ]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in phase4_accuracy.json"

    assert data["fp16_perplexity"] > 0, "FP16 perplexity must be positive"
    assert data["int8_perplexity"] > 0, "INT8 perplexity must be positive"
    assert data["corpus_tokens"] > 0, "corpus_tokens must be positive"
    assert data["verdict"] in ("negligible", "moderate", "significant"), (
        f"Unexpected verdict: {data['verdict']}"
    )
    # Delta must be arithmetically consistent (allow floating point tolerance)
    expected_delta = data["int8_perplexity"] - data["fp16_perplexity"]
    assert abs(data["perplexity_delta_abs"] - expected_delta) < 0.01, (
        "perplexity_delta_abs is inconsistent with fp16/int8 values"
    )

