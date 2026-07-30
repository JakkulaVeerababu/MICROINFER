"""
MicroInfer - Unit Tests for Phase 3 Continuous Batching Scheduler Benchmark Harness
Updated to match the new concurrent-wave output schema.
"""

import sys
import json
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_scheduler import run_scheduler_benchmark


def test_scheduler_benchmark_execution():
    """
    Executes a quick scheduler benchmark run and verifies output structure.
    Uses concurrent_requests=2, num_waves=1 for speed in tests.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 3 benchmark test."

    results_data = run_scheduler_benchmark(
        concurrent_requests=2,
        num_warmup=1,
        num_waves=1,
    )

    assert results_data["phase"] == "Phase 3 - Dynamic Request Scheduler with Lifecycle Management"
    assert "peak_vram_gb" in results_data
    assert results_data["peak_vram_gb"] > 0.0

    # New schema: aggregate dict with throughput_tok_per_sec
    assert "aggregate" in results_data
    agg = results_data["aggregate"]
    assert "throughput_tok_per_sec" in agg
    assert agg["throughput_tok_per_sec"]["mean"] > 0.0

    # wave_results list
    assert "wave_results" in results_data
    assert len(results_data["wave_results"]) >= 1
    wave = results_data["wave_results"][0]
    assert wave["n_completed"] == 2
    assert wave["total_tokens"] > 0
    assert wave["aggregate_throughput"] > 0.0

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase3_scheduler.json"
    assert results_file.exists()
