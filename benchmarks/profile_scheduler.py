"""
MicroInfer - Phase 3: Scheduler Profiler
=========================================

Runs torch.profiler.profile() over exactly 20 decode steps of the Phase 3
scheduler with 8 concurrent sequences to produce profiler evidence for
the Python-overhead vs CUDA-kernel-time claim in analysis/ANALYSIS.md.

Outputs:
  - analysis/profiles/profiler_trace.json  (Chrome trace, load in chrome://tracing)
  - analysis/profiles/profiler_summary.txt (key_averages table sorted by cuda_time_total)

Usage:
    python benchmarks/profile_scheduler.py

The results printed to console and saved to profiler_summary.txt are the
authoritative source for the overhead claims in ANALYSIS.md.
"""

import sys
import time
import json
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.scheduler import ContinuousBatchScheduler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROFILE_BATCH_SIZE = 8       # concurrent sequences during profiling
WARMUP_STEPS = 5             # steps to run before profiler starts
PROFILE_STEPS = 20           # steps to capture in profiler
PROMPT = "Explain how transformer attention mechanism works in simple terms."
MAX_NEW_TOKENS = 128         # long enough for all 20 decode steps to be decode, not prefill

PROFILES_DIR = Path(__file__).parent.parent / "analysis" / "profiles"
TRACE_PATH = PROFILES_DIR / "profiler_trace.json"
SUMMARY_PATH = PROFILES_DIR / "profiler_summary.txt"


def run_profiler():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Profiler] Loading model on {device.upper()}...")
    model, tokenizer = load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, device=device)
    model.eval()

    # -----------------------------------------------------------------------
    # Build scheduler with PROFILE_BATCH_SIZE requests
    # -----------------------------------------------------------------------
    scheduler = ContinuousBatchScheduler(max_batch_size=PROFILE_BATCH_SIZE, device=device)
    for i in range(PROFILE_BATCH_SIZE):
        scheduler.add_request(
            PROMPT, tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            model=model,
        )

    print(f"[Profiler] Scheduler initialized with {PROFILE_BATCH_SIZE} concurrent sequences.")
    print(f"[Profiler] Running {WARMUP_STEPS} warm-up steps...")

    # -----------------------------------------------------------------------
    # Warm-up: run WARMUP_STEPS steps to get all sequences into decode phase
    # -----------------------------------------------------------------------
    for step_i in range(WARMUP_STEPS):
        if scheduler.has_pending_work():
            scheduler.step(model, tokenizer)
        else:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[Profiler] Warm-up complete. Running {PROFILE_STEPS} profiled decode steps...")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Profiled region: exactly PROFILE_STEPS scheduler steps
    # -----------------------------------------------------------------------
    steps_captured = 0
    cpu_times_ms = []
    cuda_times_ms = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(PROFILE_STEPS):
            if not scheduler.has_pending_work():
                break

            t_cpu_start = time.perf_counter()
            with record_function("scheduler_step"):
                scheduler.step(model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_cpu_end = time.perf_counter()

            cpu_times_ms.append((t_cpu_end - t_cpu_start) * 1000.0)
            steps_captured += 1

    print(f"[Profiler] Captured {steps_captured} profiled steps.")

    # -----------------------------------------------------------------------
    # Export Chrome trace
    # -----------------------------------------------------------------------
    prof.export_chrome_trace(str(TRACE_PATH))
    print(f"[Profiler] Chrome trace -> '{TRACE_PATH}'")

    # -----------------------------------------------------------------------
    # Build key_averages table sorted by cuda_time_total
    # -----------------------------------------------------------------------
    avg_table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(f"MicroInfer Phase 3 Scheduler Profiler Summary\n")
        f.write(f"Batch Size: {PROFILE_BATCH_SIZE} concurrent sequences\n")
        f.write(f"Steps profiled: {steps_captured}\n")
        f.write(f"Model: {DEFAULT_MODEL_ID}\n")
        f.write("=" * 80 + "\n\n")
        f.write(avg_table)
        f.write("\n\n")

        # --- CPU vs CUDA time breakdown ---
        all_avgs = prof.key_averages()

        total_cuda_us = sum(
            getattr(item, "self_cuda_time_total", 0) for item in all_avgs
        )
        total_cpu_us = sum(
            getattr(item, "self_cpu_time_total", 0) for item in all_avgs
        )
        total_us = total_cuda_us + total_cpu_us

        cpu_pct = (total_cpu_us / total_us * 100) if total_us > 0 else 0
        cuda_pct = (total_cuda_us / total_us * 100) if total_us > 0 else 0

        # Top ops by self_cuda_time
        sorted_by_cuda = sorted(
            all_avgs,
            key=lambda x: getattr(x, "self_cuda_time_total", 0),
            reverse=True,
        )
        top3_cuda = sorted_by_cuda[:3]

        # Top ops by self_cpu_time
        sorted_by_cpu = sorted(
            all_avgs,
            key=lambda x: getattr(x, "self_cpu_time_total", 0),
            reverse=True,
        )
        top3_cpu = sorted_by_cpu[:3]

        breakdown_lines = [
            "\n--- CPU vs CUDA Time Breakdown ---\n",
            f"Total CPU self-time : {total_cpu_us/1000:.2f} ms  ({cpu_pct:.1f}%)\n",
            f"Total CUDA self-time: {total_cuda_us/1000:.2f} ms  ({cuda_pct:.1f}%)\n",
            f"Ratio CPU:CUDA      : {cpu_pct:.1f}% : {cuda_pct:.1f}%\n",
            "\n--- Top 3 Ops by CUDA time ---\n",
        ]
        for i, op in enumerate(top3_cuda, 1):
            cuda_ms = getattr(op, "self_cuda_time_total", 0) / 1000.0
            calls = getattr(op, "count", "?")
            breakdown_lines.append(
                f"  {i}. {op.key:<40s}  CUDA={cuda_ms:.2f}ms  calls={calls}\n"
            )

        breakdown_lines.append("\n--- Top 3 Ops by CPU time ---\n")
        for i, op in enumerate(top3_cpu, 1):
            cpu_ms = getattr(op, "self_cpu_time_total", 0) / 1000.0
            calls = getattr(op, "count", "?")
            breakdown_lines.append(
                f"  {i}. {op.key:<40s}  CPU={cpu_ms:.2f}ms  calls={calls}\n"
            )

        for line in breakdown_lines:
            f.write(line)

    print(f"[Profiler] Summary  -> '{SUMMARY_PATH}'")

    # Print summary to console
    print("\n" + "=" * 80)
    print("PROFILER SUMMARY (sorted by cuda_time_total)")
    print("=" * 80)
    print(avg_table[:3000])   # Truncate for terminal readability

    print("\n--- CPU vs CUDA Time Breakdown ---")
    print(f"  Total CPU  self-time : {total_cpu_us/1000:.2f} ms  ({cpu_pct:.1f}%)")
    print(f"  Total CUDA self-time : {total_cuda_us/1000:.2f} ms  ({cuda_pct:.1f}%)")
    print(f"  Ratio                : CPU={cpu_pct:.1f}%  CUDA={cuda_pct:.1f}%")

    print("\n--- Top 3 Ops by CUDA time ---")
    for i, op in enumerate(top3_cuda, 1):
        cuda_ms = getattr(op, "self_cuda_time_total", 0) / 1000.0
        print(f"  {i}. {op.key:<40s}  {cuda_ms:.2f} ms CUDA")

    print("\n--- Top 3 Ops by CPU time ---")
    for i, op in enumerate(top3_cpu, 1):
        cpu_ms = getattr(op, "self_cpu_time_total", 0) / 1000.0
        print(f"  {i}. {op.key:<40s}  {cpu_ms:.2f} ms CPU")

    # Return structured data for downstream use
    return {
        "steps_profiled": steps_captured,
        "batch_size": PROFILE_BATCH_SIZE,
        "total_cpu_time_ms": round(total_cpu_us / 1000.0, 2),
        "total_cuda_time_ms": round(total_cuda_us / 1000.0, 2),
        "cpu_pct": round(cpu_pct, 1),
        "cuda_pct": round(cuda_pct, 1),
        "top3_cuda": [
            {
                "op": op.key,
                "cuda_time_ms": round(getattr(op, "self_cuda_time_total", 0) / 1000.0, 2),
                "calls": getattr(op, "count", 0),
            }
            for op in top3_cuda
        ],
        "top3_cpu": [
            {
                "op": op.key,
                "cpu_time_ms": round(getattr(op, "self_cpu_time_total", 0) / 1000.0, 2),
                "calls": getattr(op, "count", 0),
            }
            for op in top3_cpu
        ],
        "trace_path": str(TRACE_PATH),
        "summary_path": str(SUMMARY_PATH),
    }


if __name__ == "__main__":
    result = run_profiler()
    # Save structured JSON for ANALYSIS.md reference
    json_path = PROFILES_DIR / "profiler_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Profiler] Structured results -> '{json_path}'")
    print("\n[Profiler] Done. Use analysis/profiles/profiler_summary.txt for ANALYSIS.md evidence.")
