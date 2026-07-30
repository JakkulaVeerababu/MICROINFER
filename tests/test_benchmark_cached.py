"""
MicroInfer - Unit Tests for Sub-Phase 2.4 Cached Generator Benchmarking Harness
"""

import sys
import json
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_cached import run_cached_benchmark, TEST_PROMPTS


def test_cached_benchmark_execution():
    """
    Executes a quick cached benchmark run (max_new_tokens=5, num_runs=1)
    and verifies that valid per-step latency and JSON metrics are produced.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 2 benchmark test."

    results_data = run_cached_benchmark(
        max_new_tokens=5,
        num_runs=1,
    )

    assert results_data["phase"] == "Phase 2 - KV-Cache Generator"
    assert "peak_vram_gb" in results_data
    assert results_data["peak_vram_gb"] > 0.0

    scenarios = results_data["results"]
    assert len(scenarios) == len(TEST_PROMPTS)

    for sc in scenarios:
        assert "per_step_latency_ms" in sc
        assert len(sc["per_step_latency_ms"]) == 5
        assert sc["per_step_latency_ms"][0] > 0.0

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase2_cached.json"
    assert results_file.exists()
