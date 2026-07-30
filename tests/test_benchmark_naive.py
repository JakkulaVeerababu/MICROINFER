"""
MicroInfer - Unit Tests for Phase 1 Naive Benchmarking Harness
Updated to match the new multi-run, mean/p50/p99 output schema.
"""

import sys
import json
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_naive import run_naive_benchmark, TEST_PROMPTS


def test_naive_benchmark_execution():
    """
    Executes a quick uncached benchmark run (max_new_tokens=5, num_runs=1)
    and verifies that valid per-step latency and JSON metrics are produced.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 1 benchmark test."

    results_data = run_naive_benchmark(
        max_new_tokens=5,
        num_runs=1,
    )

    assert results_data["phase"] == "Phase 1 - Naive Generator (Uncached)"
    assert "peak_vram_gb" in results_data
    assert results_data["peak_vram_gb"] > 0.0

    scenarios = results_data["results"]
    assert len(scenarios) == len(TEST_PROMPTS)

    for sc in scenarios:
        assert "per_step_latency_ms" in sc
        assert len(sc["per_step_latency_ms"]) == 5
        assert sc["per_step_latency_ms"][0] > 0.0
        # Metrics are now dicts with mean/p50/p99
        for metric in ("ttft_ms", "tpot_ms", "throughput_tok_per_sec"):
            assert metric in sc
            assert isinstance(sc[metric], dict), f"{metric} should be a dict"
            assert sc[metric]["mean"] > 0.0

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase1_naive.json"
    assert results_file.exists()
