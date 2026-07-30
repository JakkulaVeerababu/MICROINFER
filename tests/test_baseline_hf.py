"""
MicroInfer - Unit Tests for Sub-Phase 0.4 Baseline Benchmarking Harness
"""

import sys
import json
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.baseline_hf import benchmark_hf_generate, TEST_PROMPTS


def test_baseline_hf_execution():
    """
    Executes a quick baseline benchmark run (max_new_tokens=10, num_runs=1)
    and verifies that valid benchmark metrics and JSON output are produced.
    """
    assert torch.cuda.is_available(), "CUDA required for Sub-Phase 0.4 benchmark test."
    
    # Run lightweight benchmark
    results_data = benchmark_hf_generate(
        max_new_tokens=10,
        num_runs=1,
    )

    # Verify top-level JSON structure
    assert results_data["phase"] == "Phase 0 - HuggingFace Baseline"
    assert "model_id" in results_data
    assert "peak_vram_gb" in results_data
    assert isinstance(results_data["peak_vram_gb"], float)
    assert results_data["peak_vram_gb"] > 0.0

    # Verify scenario results
    scenarios = results_data["results"]
    assert len(scenarios) == len(TEST_PROMPTS)

    for sc in scenarios:
        assert "prompt_idx" in sc
        assert "input_tokens" in sc
        assert sc["input_tokens"] > 0
        assert sc["output_tokens"] == 10
        assert "ttft_ms" in sc and sc["ttft_ms"] > 0.0
        assert "tpot_ms" in sc and sc["tpot_ms"] > 0.0
        assert "throughput_tok_per_sec" in sc and sc["throughput_tok_per_sec"] > 0.0

    # Verify exported JSON file exists and is valid
    results_file = Path(__file__).parent.parent / "benchmarks" / "results" / "phase0_baseline_hf.json"
    assert results_file.exists(), f"Results file '{results_file}' was not created."

    with open(results_file, "r") as f:
        saved_data = json.load(f)
    assert saved_data["phase"] == "Phase 0 - HuggingFace Baseline"
