"""
MicroInfer - Phase 0 Baseline Visualization Plotter
Generates bar charts for TTFT and TPOT baseline metrics across scenario prompts.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase0_baseline():
    results_path = Path(__file__).parent.parent / "benchmarks" / "results" / "phase0_baseline_hf.json"
    
    if not results_path.exists():
        print(f"Error: Baseline results file '{results_path}' not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    def _val(x):
        return x.get("mean", 0.0) if isinstance(x, dict) else float(x)

    scenarios = data["results"]
    prompt_labels = [f"Prompt {sc['prompt_idx']}\n({sc['input_tokens']} tok)" for sc in scenarios]
    ttft_vals = [_val(sc["ttft_ms"]) for sc in scenarios]
    tpot_vals = [_val(sc["tpot_ms"]) for sc in scenarios]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = range(len(scenarios))
    width = 0.35

    rects1 = ax1.bar([i - width/2 for i in x], ttft_vals, width, label="TTFT (ms)", color="#3182bd")
    rects2 = ax1.bar([i + width/2 for i in x], tpot_vals, width, label="TPOT (ms/tok)", color="#e6550d")

    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title(f"MicroInfer Phase 0: HuggingFace Baseline Metrics ({data['model_id']})", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompt_labels, fontsize=10)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Label bar values
    for bar in rects1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in rects2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    chart_path = plots_dir / "phase0_baseline.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[MicroInfer] Phase 0 baseline chart saved to '{chart_path}'.")


if __name__ == "__main__":
    plot_phase0_baseline()
