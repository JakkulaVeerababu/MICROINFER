"""
MicroInfer - Unit Tests for Sub-Phase 3.4 Continuous Batching Benchmarking Harness
"""

import sys
import json
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_scheduler import run_scheduler_benchmark


def test_scheduler_benchmark_execution():
    """
    Executes a quick scheduler benchmark run and verifies output structure.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 3 benchmark test."

    results_data = run_scheduler_benchmark(max_batch_size=2)

    assert results_data["phase"] == "Phase 3 - Continuous Batching Scheduler"
    assert "aggregate_throughput_tok_per_sec" in results_data
    assert results_data["aggregate_throughput_tok_per_sec"] > 0.0
    assert "peak_vram_gb" in results_data
    assert results_data["peak_vram_gb"] > 0.0
    assert len(results_data["requests"]) == 5

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase3_scheduler.json"
    assert results_file.exists()
