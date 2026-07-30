"""
MicroInfer - Phase 2 Visualization Plotter
Generates throughput and per-step latency comparison charts between Phase 1 (Naive) and Phase 2 (KV-Cache).
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase2_comparisons():
    root = Path(__file__).parent.parent
    p1_path = root / "benchmarks" / "results" / "phase1_naive.json"
    p2_path = root / "benchmarks" / "results" / "phase2_cached.json"

    if not (p1_path.exists() and p2_path.exists()):
        print(f"Error: Required JSON files missing.")
        return

    with open(p1_path, "r") as f:
        p1_data = json.load(f)
    with open(p2_path, "r") as f:
        p2_data = json.load(f)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    def _val(x):
        return x.get("mean", 0.0) if isinstance(x, dict) else float(x)

    labels = [f"Scenario {i+1}" for i in range(len(p1_data["results"]))]
    p1_tp = [_val(sc["throughput_tok_per_sec"]) for sc in p1_data["results"]]
    p2_tp = [_val(sc["throughput_tok_per_sec"]) for sc in p2_data["results"]]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - width/2 for i in x], p1_tp, width, label="Phase 1: Naive (No Cache)", color="#e74c3c")
    plt.bar([i + width/2 for i in x], p2_tp, width, label="Phase 2: KV-Cache", color="#2ecc71")

    plt.title("MicroInfer: Throughput Comparison (Phase 1 Naive vs Phase 2 KV-Cache)", fontsize=13, fontweight="bold")
    plt.xlabel("Prompt Workload Scenario", fontsize=11)
    plt.ylabel("Throughput (tokens/second)", fontsize=11)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    chart1_path = plots_dir / "phase2_throughput_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Line Chart: Per-Step Latency Curve (Phase 1 Quadratic vs Phase 2 Flat O(1))
    plt.figure(figsize=(10, 6))
    p1_steps = p1_data["results"][0]["per_step_latency_ms"]
    p2_steps = p2_data["results"][0]["per_step_latency_ms"]

    plt.plot(range(1, len(p1_steps) + 1), p1_steps, label="Phase 1 Naive (O(n^2) Quadratic Re-computation)", color="#e74c3c", linewidth=2)
    plt.plot(range(1, len(p2_steps) + 1), p2_steps, label="Phase 2 KV-Cache (O(1) Flat Per-Step Decoding)", color="#2ecc71", linewidth=2)

    plt.title("MicroInfer Step-by-Step Decoding Latency Scaling", fontsize=13, fontweight="bold")
    plt.xlabel("Generated Token Index Step (N)", fontsize=11)
    plt.ylabel("Step Latency (ms)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    chart2_path = plots_dir / "phase2_flat_step_latency.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[MicroInfer] Phase 2 comparison charts saved to '{chart1_path}' and '{chart2_path}'.")


if __name__ == "__main__":
    plot_phase2_comparisons()
