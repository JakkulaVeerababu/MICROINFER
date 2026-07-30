"""
MicroInfer - Unit Tests for Sub-Phase 4.4 INT8 Quantized Benchmarking Harness
"""

import sys
import json
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_quant import run_quant_benchmark, TEST_PROMPTS


def test_quant_benchmark_execution():
    """
    Executes a quick INT8 benchmark run (max_new_tokens=5, num_runs=1)
    and verifies that valid per-step latency and JSON metrics are produced.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 4 benchmark test."

    results_data = run_quant_benchmark(
        max_new_tokens=5,
        num_runs=1,
    )

    assert results_data["phase"] == "Phase 4 - INT8 Quantized Engine"
    assert "peak_vram_gb" in results_data
    assert results_data["peak_vram_gb"] > 0.0

    scenarios = results_data["results"]
    assert len(scenarios) == len(TEST_PROMPTS)

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase4_quant.json"
    assert results_file.exists()
