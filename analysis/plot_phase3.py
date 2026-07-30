"""
MicroInfer - Phase 3 Visualization Plotter
Generates performance charts for Continuous Batching Scheduler under mixed workload.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase3_performance():
    root = Path(__file__).parent.parent
    p3_path = root / "benchmarks" / "results" / "phase3_scheduler.json"

    if not p3_path.exists():
        print(f"Error: Phase 3 results file '{p3_path}' not found.")
        return

    with open(p3_path, "r") as f:
        data = json.load(f)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    requests = data["requests"]
    labels = [f"Req {r['seq_id']}" for r in requests]
    input_toks = [r["input_tokens"] for r in requests]
    gen_toks = [r["generated_tokens"] for r in requests]

    # Bar Chart: Tokens per concurrent request in mixed workload
    plt.figure(figsize=(9, 5))
    x = range(len(labels))
    width = 0.35

    plt.bar([i - width/2 for i in x], input_toks, width, label="Prompt Tokens", color="#3498db")
    plt.bar([i + width/2 for i in x], gen_toks, width, label="Generated Tokens", color="#2ecc71")

    plt.title(f"MicroInfer Phase 3: Continuous Batching Mixed Workload (Batch Size: {data['max_batch_size']})", fontsize=13, fontweight="bold")
    plt.xlabel("Queued Requests", fontsize=11)
    plt.ylabel("Token Count", fontsize=11)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    chart_path = plots_dir / "phase3_scheduler_performance.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[MicroInfer] Phase 3 chart saved to '{chart_path}'.")


if __name__ == "__main__":
    plot_phase3_performance()
