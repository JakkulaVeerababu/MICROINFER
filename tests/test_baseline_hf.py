"""
MicroInfer - Unit Tests for Phase 0 Baseline Benchmarking Harness
Updated to match the new multi-run, mean/p50/p99 output schema.
"""

import sys
import json
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.baseline_hf import benchmark_hf_generate, TEST_PROMPTS


def test_baseline_hf_execution():
    """
    Executes a quick baseline benchmark run (max_new_tokens=10, num_runs=1)
    and verifies that valid benchmark metrics and JSON output are produced.
    """
    assert torch.cuda.is_available(), "CUDA required for Phase 0 benchmark test."

    results_data = benchmark_hf_generate(
        max_new_tokens=10,
        num_runs=1,
    )

    assert results_data["phase"] == "Phase 0 - HuggingFace Baseline"
    assert "model_id" in results_data
    assert "peak_vram_gb" in results_data
    assert isinstance(results_data["peak_vram_gb"], float)
    assert results_data["peak_vram_gb"] > 0.0

    scenarios = results_data["results"]
    assert len(scenarios) == len(TEST_PROMPTS)

    for sc in scenarios:
        assert "prompt_idx" in sc
        assert "input_tokens" in sc
        assert sc["input_tokens"] > 0
        assert sc["output_tokens"] == 10

        # Metrics are now dicts with mean/p50/p99
        for metric in ("ttft_ms", "tpot_ms", "throughput_tok_per_sec"):
            assert metric in sc
            assert isinstance(sc[metric], dict), f"{metric} should be a dict"
            assert sc[metric]["mean"] > 0.0, f"{metric}['mean'] should be positive"

    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase0_baseline_hf.json"
    assert results_file.exists(), f"Results file '{results_file}' was not created."

    with open(results_file, "r") as f:
        saved_data = json.load(f)
    assert saved_data["phase"] == "Phase 0 - HuggingFace Baseline"
