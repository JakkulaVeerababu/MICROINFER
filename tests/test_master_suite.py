"""
MicroInfer - Sub-Phase 5.2 Comprehensive Master Test Suite
Executes end-to-end verification across Phase 0, Phase 1, Phase 2, Phase 3, Phase 4, and Phase 5 artifacts.
"""

import sys
import json
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_master_all_phases_completed():
    """
    Verifies that all 6 phases (Phase 0 - Phase 5) have generated valid JSON metrics artifacts.
    """
    root = Path(__file__).parent.parent
    results_dir = root / "benchmarks" / "results"

    phase_files = [
        "phase0_baseline_hf.json",
        "phase1_naive.json",
        "phase2_cached.json",
        "phase3_scheduler.json",
        "phase4_quant.json",
        "phase5_vllm.json",
    ]

    for fname in phase_files:
        fpath = results_dir / fname
        assert fpath.exists(), f"Missing required result artifact: {fname}"
        with open(fpath, "r") as f:
            data = json.load(f)
        assert "phase" in data, f"Missing 'phase' key in {fname}"


def test_master_all_plots_exist():
    """
    Verifies that all visual plot charts exist.
    """
    root = Path(__file__).parent.parent
    plots_dir = root / "analysis" / "plots"

    plot_files = [
        "phase0_baseline.png",
        "phase1_quadratic_scaling.png",
        "phase2_throughput_comparison.png",
        "phase2_flat_step_latency.png",
        "phase3_scheduler_performance.png",
        "phase4_vram_reduction.png",
    ]

    for pname in plot_files:
        ppath = plots_dir / pname
        assert ppath.exists() and ppath.stat().st_size > 0, f"Missing plot chart: {pname}"
