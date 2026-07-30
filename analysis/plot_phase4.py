"""
MicroInfer - Phase 4 Visualization Plotter
Generates VRAM memory footprint reduction charts comparing FP16 vs INT8 quantization.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase4_vram_reduction():
    root = Path(__file__).parent.parent
    p2_path = root / "benchmarks" / "results" / "phase2_cached.json"
    p4_path = root / "benchmarks" / "results" / "phase4_quant.json"

    if not (p2_path.exists() and p4_path.exists()):
        print(f"Error: Required JSON benchmark files missing.")
        return

    with open(p2_path, "r") as f:
        p2_data = json.load(f)
    with open(p4_path, "r") as f:
        p4_data = json.load(f)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    fp16_vram = p2_data["peak_vram_gb"]
    int8_vram = p4_data["peak_vram_gb"]
    vram_savings_pct = ((fp16_vram - int8_vram) / fp16_vram) * 100.0

    categories = ["Phase 2: FP16 Precision", "Phase 4: INT8 Quantized"]
    vram_usage = [fp16_vram, int8_vram]
    colors = ["#3498db", "#e67e22"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, vram_usage, color=colors, width=0.45)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f"{height:.2f} GB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.title(f"MicroInfer VRAM Memory Allocation (FP16 vs INT8 Quantization)\nSaved {vram_savings_pct:.1f}% VRAM Memory!", fontsize=13, fontweight="bold")
    plt.ylabel("Peak VRAM Allocation (GB)", fontsize=11)
    plt.ylim(0, 3.5)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    chart_path = plots_dir / "phase4_vram_reduction.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[MicroInfer] Phase 4 VRAM chart saved to '{chart_path}'.")


if __name__ == "__main__":
    plot_phase4_vram_reduction()
