"""
MicroInfer - Unit Tests for Phase 5 Production Reference Benchmark Harness
Updated to match the new concurrent-wave output schema.
"""

import sys
import json
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.baseline_vllm import run_vllm_benchmark, TEST_PROMPTS


def test_vllm_benchmark_execution():
    """
    Executes a quick Phase 5 reference benchmark run (num_runs=1 -> 1 wave, 2 requests)
    and verifies that valid JSON metrics are produced.
    No synthetic multipliers are applied; numbers are real measured values.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 5 benchmark test."

    results_data = run_vllm_benchmark(
        max_new_tokens=5,
        num_runs=1,
    )

    assert results_data["phase"].startswith("Phase 5 -")
    assert "engine" in results_data
    assert "peak_vram_gb" in results_data
    assert results_data["peak_vram_gb"] > 0.0

    # New schema: aggregate dict
    assert "aggregate" in results_data
    agg = results_data["aggregate"]
    assert "throughput_tok_per_sec" in agg
    assert agg["throughput_tok_per_sec"]["mean"] > 0.0

    # wave_results list
    assert "wave_results" in results_data
    assert len(results_data["wave_results"]) >= 1
    wave = results_data["wave_results"][0]
    assert wave["n_completed"] >= 1
    assert wave["total_tokens"] > 0
    assert wave["aggregate_throughput"] > 0.0

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase5_vllm.json"
    assert results_file.exists()
