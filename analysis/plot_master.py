"""
MicroInfer - Phase 5 Master Visualization Plotter
Generates master benchmark comparison charts across Phase 0 through Phase 5.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_master_comparisons():
    root = Path(__file__).parent.parent
    results_dir = root / "benchmarks" / "results"

    p0 = json.load(open(results_dir / "phase0_baseline_hf.json"))
    p1 = json.load(open(results_dir / "phase1_naive.json"))
    p2 = json.load(open(results_dir / "phase2_cached.json"))
    p3 = json.load(open(results_dir / "phase3_scheduler.json"))
    p4 = json.load(open(results_dir / "phase4_quant.json"))
    p5 = json.load(open(results_dir / "phase5_vllm.json"))

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # 1. Master Bar Chart: Throughput across all engines
    phases = ["P0: HF Baseline", "P1: Uncached", "P2: KV-Cache", "P3: Continuous", "P4: INT8 Quant", "P5: vLLM Ref"]
    
    def _val(x):
        return x.get("mean", 0.0) if isinstance(x, dict) else float(x)

    # Calculate avg throughput per phase
    p0_tp = sum(_val(r["throughput_tok_per_sec"]) for r in p0["results"]) / len(p0["results"])
    p1_tp = sum(_val(r["throughput_tok_per_sec"]) for r in p1["results"]) / len(p1["results"])
    p2_tp = sum(_val(r["throughput_tok_per_sec"]) for r in p2["results"]) / len(p2["results"])
    p3_tp = _val(p3.get("aggregate", {}).get("throughput_tok_per_sec", p3.get("aggregate_throughput_tok_per_sec", 0)))
    p4_tp = sum(_val(r["throughput_tok_per_sec"]) for r in p4["results"]) / len(p4["results"])
    p5_tp = _val(p5.get("aggregate", {}).get("throughput_tok_per_sec", p5.get("aggregate_throughput_tok_per_sec", 0)))

    throughputs = [p0_tp, p1_tp, p2_tp, p3_tp, p4_tp, p5_tp]
    colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#16a085"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(phases, throughputs, color=colors, width=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.1f} tok/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("MicroInfer: Master Generation Throughput Comparison (RTX 4050 GPU)", fontsize=13, fontweight="bold")
    plt.ylabel("Throughput (tokens/second)", fontsize=11)
    plt.ylim(0, 25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    chart1_path = plots_dir / "master_throughput_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Master Bar Chart: Peak VRAM Allocation
    vram_usages = [
        p0["peak_vram_gb"],
        p1["peak_vram_gb"],
        p2["peak_vram_gb"],
        p3["peak_vram_gb"],
        p4["peak_vram_gb"],
        p5["peak_vram_gb"],
    ]

    plt.figure(figsize=(10, 5))
    bars2 = plt.bar(phases, vram_usages, color=colors, width=0.5)

    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f} GB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("MicroInfer: Peak VRAM Memory Allocation Across All Serving Engines", fontsize=13, fontweight="bold")
    plt.ylabel("Peak VRAM (GB)", fontsize=11)
    plt.ylim(0, 3.8)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    chart2_path = plots_dir / "master_vram_footprint.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[MicroInfer] Master comparison charts saved to '{chart1_path}' and '{chart2_path}'.")


if __name__ == "__main__":
    plot_master_comparisons()
