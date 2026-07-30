"""
MicroInfer - Phase 1 Visualization Plotter

Produces two charts:

1. phase1_quadratic_scaling.png
   Per-step latency curve from the canonical single-N run, showing
   the increasing-per-step cost of uncached attention recomputation.

2. phase1_scaling_crossover.png
   TTFT and TPOT at each N in {16,32,64,128,256} for BOTH the naive
   engine and the HF baseline, side-by-side.  A formatted crossover
   table is embedded as a text annotation in the figure so the
   comparison is visible without running the script in a terminal.
   The master-table row for Phase 1 (taken at N=256) is highlighted.
"""


import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"


def _load_json(path: Path):
    if not path.exists():
        print(f"[plot_phase1] Warning: '{path}' not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chart 1: per-step latency curve (unchanged from original)
# ---------------------------------------------------------------------------
def plot_phase1_scaling():
    data = _load_json(RESULTS_DIR / "phase1_naive.json")
    if data is None:
        return

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    colors = ["#7c3aed", "#2563eb", "#059669"]
    for i, sc in enumerate(data["results"]):
        steps    = list(range(1, len(sc["per_step_latency_ms"]) + 1))
        latencies = sc["per_step_latency_ms"]
        ax.plot(steps, latencies,
                color=colors[i % len(colors)],
                linewidth=1.8,
                marker="o", markersize=2,
                label=f"Prompt {sc['prompt_idx']} ({sc['input_tokens']} tok input)")

    ax.set_title("Phase 1: Uncached Naive Generator — Per-Step Latency Growth",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Generated Token Step (N)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Step Latency (ms)", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)

    chart_path = PLOTS_DIR / "phase1_quadratic_scaling.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot_phase1] Latency curve -> '{chart_path}'")


# ---------------------------------------------------------------------------
# Chart 2: TTFT and TPOT vs N for Naive vs HF baseline (crossover chart)
# ---------------------------------------------------------------------------
def plot_phase1_crossover():
    data = _load_json(RESULTS_DIR / "phase1_scaling.json")
    if data is None:
        print("[plot_phase1] Run `python benchmarks/benchmark_naive.py` first (scaling sweep).")
        return

    rows       = data["scaling_rows"]
    n_vals     = [r["N"] for r in rows]
    crossover  = data.get("crossover_N")
    master_N   = data.get("master_table_N", max(n_vals))

    naive_ttft = [r["naive"]["ttft_ms"]["mean"]        for r in rows]
    naive_tpot = [r["naive"]["tpot_ms"]["mean"]        for r in rows]
    hf_ttft    = [r["hf_baseline"]["ttft_ms"]["mean"]  for r in rows]
    hf_tpot    = [r["hf_baseline"]["tpot_ms"]["mean"]  for r in rows]

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    # Use a 3-panel layout: TTFT chart | TPOT chart | text table
    fig = plt.figure(figsize=(19, 7))
    fig.patch.set_facecolor("#0f0f1a")
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 5, 4], wspace=0.35)
    ax_ttft = fig.add_subplot(gs[0])
    ax_tpot = fig.add_subplot(gs[1])
    ax_text = fig.add_subplot(gs[2])

    NAIVE_COLOR = "#7c3aed"   # purple
    HF_COLOR    = "#f59e0b"   # amber
    MASTER_CLR  = "#22c55e"   # green highlight for master-table row
    BG         = "#0f0f1a"
    PANEL      = "#1e293b"
    GRID       = "#334155"
    TEXT       = "#94a3b8"
    WHITE      = "#f1f5f9"

    def _style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT)
        ax.set_xlabel("Output Tokens Generated (N)", color=TEXT, fontsize=10)
        ax.set_ylabel("Latency (ms)", color=TEXT, fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, linestyle="--", alpha=0.3, color=GRID)

    # --- TTFT panel ---
    _style_ax(ax_ttft, "TTFT vs N  (Time to First Token)")
    ax_ttft.set_ylabel("TTFT (ms)", color=TEXT, fontsize=10)
    ax_ttft.plot(n_vals, naive_ttft, "o-", color=NAIVE_COLOR, linewidth=2,
                 markersize=6, label="Phase 1 — Naive (no cache)")
    ax_ttft.plot(n_vals, hf_ttft,   "s--", color=HF_COLOR, linewidth=2,
                 markersize=6, label="Phase 0 — HF baseline (.generate())")
    ax_ttft.set_xticks(n_vals)
    ax_ttft.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)

    # --- TPOT panel ---
    _style_ax(ax_tpot, "TPOT vs N  (Time Per Output Token)")
    ax_tpot.set_ylabel("TPOT (ms/token)", color=TEXT, fontsize=10)
    ax_tpot.plot(n_vals, naive_tpot, "o-", color=NAIVE_COLOR, linewidth=2,
                 markersize=6, label="Phase 1 — Naive (no cache)")
    ax_tpot.plot(n_vals, hf_tpot,   "s--", color=HF_COLOR, linewidth=2,
                 markersize=6, label="Phase 0 — HF baseline (.generate())")
    ax_tpot.set_xticks(n_vals)
    ax_tpot.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)

    # Highlight master-table N on both panels
    for ax in (ax_ttft, ax_tpot):
        ax.axvline(x=master_N, color=MASTER_CLR, linestyle=":", linewidth=1.5,
                   alpha=0.8, label=f"Master-table N={master_N}")
    ax_tpot.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)

    # Annotate crossover if it occurred mid-sweep
    min_n = min(n_vals)
    if crossover is None:
        note_line = (f"No crossover within N\u2264{max(n_vals)} \u2014 see ANALYSIS.md\n"
                     "(HF uses internal cache; naive O(N\u00b2) growth is visible in TPOT rate)")
    elif crossover == min_n:
        # Naive is slower at ALL tested N: report growth-rate delta instead of a crossover line
        naive_growth = naive_tpot[-1] - naive_tpot[0]
        hf_growth    = hf_tpot[-1]   - hf_tpot[0]
        note_line = (
            f"HF (internal DynamicCache) wins TPOT at all tested N\n"
            f"O(N^2) growth IS visible: naive TPOT +{naive_growth:.1f}ms vs HF +{hf_growth:.1f}ms "
            f"from N={min_n}->N={max(n_vals)}\n"
            f"Absolute crossover lies above N={max(n_vals)} for this model - see ANALYSIS.md"
        )
    else:
        for ax in (ax_ttft, ax_tpot):
            ax.axvline(x=crossover, color="#ef4444", linestyle=":", linewidth=1.5)
        note_line = f"Crossover at N={crossover}: naive becomes slower than HF (TPOT)"

    # ---------------------------------------------------------------
    # Text table panel — side-by-side comparison with step-by-step growth %
    # ---------------------------------------------------------------
    ax_text.set_facecolor(BG)
    ax_text.axis("off")

    header = f"{'N':>5} | {'Naive TPOT':>10} {'Naive Δ%':>8} | {'HF TPOT':>8} {'HF Δ%':>8} | {'Winner':>6}"
    lines  = [header, "-" * len(header)]
    for r in rows:
        nt = r["naive"]["tpot_ms"]["mean"]
        ht = r["hf_baseline"]["tpot_ms"]["mean"]
        naive_delta = r["naive"].get("tpot_delta_pct", 0.0)
        hf_delta    = r["hf_baseline"].get("tpot_delta_pct", 0.0)
        winner = "HF" if ht < nt else "Naive"
        marker = " *" if r["N"] == master_N else ""

        n_delta_str = f"+{naive_delta:.1f}%" if naive_delta > 0 else (f"{naive_delta:.1f}%" if naive_delta < 0 else "--")
        h_delta_str = f"+{hf_delta:.1f}%" if hf_delta > 0 else (f"{hf_delta:.1f}%" if hf_delta < 0 else "--")

        lines.append(
            f"{r['N']:>5} | "
            f"{nt:>8.2f}ms {n_delta_str:>8} | "
            f"{ht:>6.2f}ms {h_delta_str:>8} | "
            f"{winner:>6}{marker}"
        )
    lines.append("")
    lines.append(note_line)

    table_text = "\n".join(lines)
    ax_text.text(
        0.02, 0.97, table_text,
        transform=ax_text.transAxes,
        color=WHITE, fontsize=7.5,
        fontfamily="monospace",
        verticalalignment="top",
        wrap=False,
    )
    ax_text.set_title("Side-by-Side Growth Table", color=WHITE,
                      fontsize=11, fontweight="bold", pad=10)

    # Print crossover table to stdout as well
    print("\n" + "=" * 90)
    print(f"  {'N':>5} | {'Naive TPOT':>10} {'Naive Δ%':>9} | {'HF TPOT':>10} {'HF Δ%':>9} | {'Winner (TPOT)':>14}")
    print("  " + "-" * 88)
    for r in rows:
        nt = r["naive"]["tpot_ms"]["mean"]
        ht = r["hf_baseline"]["tpot_ms"]["mean"]
        naive_delta = r["naive"].get("tpot_delta_pct", 0.0)
        hf_delta    = r["hf_baseline"].get("tpot_delta_pct", 0.0)
        n_delta_str = f"+{naive_delta:.1f}%" if naive_delta > 0 else (f"{naive_delta:.1f}%" if naive_delta < 0 else "--")
        h_delta_str = f"+{hf_delta:.1f}%" if hf_delta > 0 else (f"{hf_delta:.1f}%" if hf_delta < 0 else "--")
        winner = "HF" if ht < nt else "Naive"
        master_tag = "  <- master table" if r["N"] == master_N else ""
        print(f"  {r['N']:>5} | {nt:>8.2f}ms {n_delta_str:>9} | {ht:>8.2f}ms {h_delta_str:>9} | {winner:>14}{master_tag}")
    print("=" * 90 + "\n")
    print(note_line)

    fig.suptitle(
        f"Phase 1 vs Phase 0: O(N\u00b2) Scaling Sweep  |  Master-table row @ N={master_N}\n"
        f"{note_line.splitlines()[0]}",
        color=WHITE, fontsize=11, y=1.02,
    )

    chart_path = PLOTS_DIR / "phase1_scaling_crossover.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot_phase1] Crossover chart -> '{chart_path}'")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 1 plotter. Default: produces both charts."
    )
    parser.add_argument("--crossover", action="store_true",
                        help="Plot only the N-sweep crossover chart (requires phase1_scaling.json).")
    parser.add_argument("--scaling-only", dest="scaling_only", action="store_true",
                        help="Plot only the per-step latency curve (requires phase1_naive.json).")
    parser.add_argument("--all", action="store_true",
                        help="Plot both charts (same as default).")
    args = parser.parse_args()

    if args.scaling_only:
        plot_phase1_scaling()
    elif args.crossover:
        plot_phase1_crossover()
    else:
        # Default: both — the crossover chart is the primary Phase 1 deliverable
        plot_phase1_scaling()
        plot_phase1_crossover()
