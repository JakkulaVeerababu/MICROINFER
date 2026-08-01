"""
MicroInfer - Phase 2 vs Phase 1 Scaling Benchmark
===================================================

Head-to-head comparison of naive_generate (Phase 1) vs cached_generate
(Phase 2, MicroInfer KVCache) across sequence lengths N in {64, 256, 512,
1024, 2048}.

This is the authoritative sweep for the "KV-caching Architecture" milestone
in README.md. It uses THIS PROJECT'S own KVCache — not HF DynamicCache
(that is Phase 0's data) — to find the real crossover point where Phase 2
outperforms Phase 1.

Protocol:
  - 3 warm-up runs (discarded) before any timing
  - 5 timed runs per engine per N  (balance of signal vs wall time at N=2048)
  - Reports TPOT (mean decode step latency), throughput, and % delta
  - Saves results to benchmarks/results/phase2_vs_phase1_scaling.json

Usage:
    python benchmarks/benchmark_p2_vs_p1_scaling.py
"""

import sys
import json
import time
from pathlib import Path
from statistics import mean, stdev

import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.naive_generate import naive_generate
from src.cached_generate import cached_generate

# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------
SEQ_LENGTHS   = [64, 256, 512, 1024, 2048]  # context lengths to sweep
GEN_TOKENS    = 32        # tokens to generate at each N (kept short to isolate TPOT)
NUM_WARMUP    = 3         # discarded runs per engine before timing starts
NUM_TIMED     = 5         # timed runs per (engine, N)

RESULTS_DIR   = ROOT_DIR / "benchmarks" / "results"


def build_prompt_for_n(tokenizer, target_n: int) -> str:
    """Build a prompt that encodes to approximately target_n tokens."""
    base = (
        "The field of artificial intelligence and high-performance GPU serving "
        "algorithms has grown rapidly in recent years. "
    )
    prompt = base
    while len(tokenizer.encode(prompt)) < target_n:
        prompt += base
    # Trim to exactly target_n tokens
    tokens = tokenizer.encode(prompt)[:target_n]
    return tokenizer.decode(tokens)


def run_engine(fn, model, tokenizer, prompt, gen_tokens, n_runs):
    """Run fn n_runs times and collect TPOT and throughput lists."""
    tpots = []
    throughputs = []
    for _ in range(n_runs):
        res = fn(model, tokenizer, prompt, max_new_tokens=gen_tokens, temperature=0.0)
        tpots.append(res["tpot_ms"])
        throughputs.append(res["throughput_tok_per_sec"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return tpots, throughputs


def run_scaling_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 72, flush=True)
    print("  MicroInfer  --  Phase 2 vs Phase 1 Scaling Benchmark", flush=True)
    print("=" * 72, flush=True)
    print(f"  Device       : {gpu_name}", flush=True)
    print(f"  Model        : {DEFAULT_MODEL_ID}", flush=True)
    print(f"  Seq lengths  : {SEQ_LENGTHS}", flush=True)
    print(f"  Gen tokens   : {GEN_TOKENS} per run", flush=True)
    print(f"  Warm-up      : {NUM_WARMUP} discarded runs per engine", flush=True)
    print(f"  Timed runs   : {NUM_TIMED} per (engine, N)", flush=True)
    print("=" * 72, flush=True)

    print("\n[1/3] Loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, device=device)
    model.eval()

    # -----------------------------------------------------------------------
    # Global warm-up — 3 short passes to heat up caches and GPU clocks
    # -----------------------------------------------------------------------
    print("\n[2/3] Global warm-up (3 short passes)...", flush=True)
    short_prompt = "Hello world."
    for _ in range(3):
        naive_generate(model, tokenizer, short_prompt, max_new_tokens=4)
        cached_generate(model, tokenizer, short_prompt, max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Warm-up done.", flush=True)

    # -----------------------------------------------------------------------
    # Main sweep
    # -----------------------------------------------------------------------
    print("\n[3/3] Running scaling sweep...\n", flush=True)

    col_header = f"{'N':>6}  {'P1 TPOT':>10}  {'P2 TPOT':>10}  {'P1 tok/s':>10}  {'P2 tok/s':>10}  {'Delta%':>8}  {'Winner':>8}"
    print(col_header, flush=True)
    print("-" * len(col_header), flush=True)

    all_results = []
    crossover_n = None

    for n in SEQ_LENGTHS:
        prompt = build_prompt_for_n(tokenizer, n)
        actual_n = len(tokenizer.encode(prompt))

        # --- Per-N warm-up ---
        for _ in range(NUM_WARMUP):
            naive_generate(model, tokenizer, prompt, max_new_tokens=4)
            cached_generate(model, tokenizer, prompt, max_new_tokens=4)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Phase 1 timing ---
        p1_tpots, p1_throughputs = run_engine(
            naive_generate, model, tokenizer, prompt, GEN_TOKENS, NUM_TIMED
        )

        # --- Phase 2 timing ---
        p2_tpots, p2_throughputs = run_engine(
            cached_generate, model, tokenizer, prompt, GEN_TOKENS, NUM_TIMED
        )

        p1_tpot_mean  = mean(p1_tpots)
        p2_tpot_mean  = mean(p2_tpots)
        p1_tpot_std   = stdev(p1_tpots) if len(p1_tpots) > 1 else 0.0
        p2_tpot_std   = stdev(p2_tpots) if len(p2_tpots) > 1 else 0.0
        p1_tp_mean    = mean(p1_throughputs)
        p2_tp_mean    = mean(p2_throughputs)
        p1_tp_std     = stdev(p1_throughputs) if len(p1_throughputs) > 1 else 0.0
        p2_tp_std     = stdev(p2_throughputs) if len(p2_throughputs) > 1 else 0.0

        delta_pct = (p2_tp_mean / p1_tp_mean - 1) * 100 if p1_tp_mean > 0 else 0.0
        winner = "Phase2" if p2_tp_mean > p1_tp_mean else "Phase1"

        if crossover_n is None and p2_tp_mean > p1_tp_mean:
            crossover_n = n

        row = (
            f"{n:>6}  "
            f"{p1_tpot_mean:>8.2f}ms  "
            f"{p2_tpot_mean:>8.2f}ms  "
            f"{p1_tp_mean:>8.2f}t/s  "
            f"{p2_tp_mean:>8.2f}t/s  "
            f"{delta_pct:>+7.1f}%  "
            f"{winner:>8}"
        )
        print(row, flush=True)

        all_results.append({
            "n": n,
            "actual_prompt_tokens": actual_n,
            "phase1_naive": {
                "tpot_ms_mean": round(p1_tpot_mean, 3),
                "tpot_ms_std":  round(p1_tpot_std,  3),
                "throughput_mean": round(p1_tp_mean, 3),
                "throughput_std":  round(p1_tp_std,  3),
                "raw_tpots": [round(v, 3) for v in p1_tpots],
            },
            "phase2_cached": {
                "tpot_ms_mean": round(p2_tpot_mean, 3),
                "tpot_ms_std":  round(p2_tpot_std,  3),
                "throughput_mean": round(p2_tp_mean, 3),
                "throughput_std":  round(p2_tp_std,  3),
                "raw_tpots": [round(v, 3) for v in p2_tpots],
            },
            "delta_throughput_pct": round(delta_pct, 2),
            "winner": winner,
        })

    print("-" * len(col_header), flush=True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(flush=True)
    if crossover_n is not None:
        print(f"  Crossover point: Phase 2 first beats Phase 1 at N = {crossover_n}", flush=True)
    else:
        print("  Phase 2 did not beat Phase 1 across any tested N.", flush=True)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "phase2_vs_phase1_scaling.json"

    export = {
        "benchmark": "Phase 2 vs Phase 1 Scaling Sweep",
        "model_id": DEFAULT_MODEL_ID,
        "gpu": gpu_name,
        "seq_lengths": SEQ_LENGTHS,
        "gen_tokens": GEN_TOKENS,
        "num_warmup": NUM_WARMUP,
        "num_timed": NUM_TIMED,
        "crossover_n": crossover_n,
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Results -> '{output_path}'", flush=True)

    return export


if __name__ == "__main__":
    run_scaling_benchmark()
