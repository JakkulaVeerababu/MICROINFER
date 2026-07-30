"""
MicroInfer - Phase 1 Visualization Plotter
Generates per-step latency curve chart demonstrating uncached quadratic scaling.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase1_scaling():
    results_path = Path(__file__).parent.parent / "benchmarks" / "results" / "phase1_naive.json"
    
    if not results_path.exists():
        print(f"Error: Results file '{results_path}' not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10, 6))
    
    scenarios = data["results"]
    for sc in scenarios:
        steps = list(range(1, len(sc["per_step_latency_ms"]) + 1))
        latencies = sc["per_step_latency_ms"]
        plt.plot(steps, latencies, marker="o", markersize=3, label=f"Scenario {sc['prompt_idx']} (Prompt Len: {sc['input_tokens']} tok)")

    plt.title("MicroInfer Phase 1: Uncached Naive Generator Per-Step Latency", fontsize=14, fontweight="bold")
    plt.xlabel("Generated Token Step (N)", fontsize=12)
    plt.ylabel("Step Latency (ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    chart_path = plots_dir / "phase1_quadratic_scaling.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[MicroInfer] Latency chart saved to '{chart_path}'.")


if __name__ == "__main__":
    plot_phase1_scaling()
