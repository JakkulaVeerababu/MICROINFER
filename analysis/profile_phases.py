"""
MicroInfer - Phase 2/3/4 Profiler Evidence Collection
======================================================
Runs torch.profiler on a controlled set of decode steps for each phase and
extracts concrete CPU/CUDA time breakdowns, top kernel names, and GPU utilization
data to replace the architectural-guess explanations in analysis/ANALYSIS.md.

Outputs (analysis/profiles/):
  phase2_profiler_summary.json   -- KV-cache decode: CPU/CUDA split + top ops
  phase2_trace.json              -- Chrome trace (loadable in chrome://tracing)
  phase3_profiler_summary.json   -- Scheduler decode: CPU/CUDA split + top ops
  phase3_python_vs_cuda.json     -- Explicit Python overhead vs CUDA forward timing
  phase3_trace.json              -- Chrome trace for Phase 3
  gpu_util_phase3.csv            -- nvidia-smi 1s GPU utilization samples
  phase4_profiler_summary.json   -- INT8 decode: dequant kernel breakdown
  phase4_trace.json              -- Chrome trace for Phase 4

Run: python analysis/profile_phases.py
"""

import csv
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PROFILES_DIR = Path(__file__).parent / "profiles"
PROFILES_DIR.mkdir(exist_ok=True, parents=True)

from src.model_loader import DEFAULT_MODEL_ID, load_model_and_tokenizer
from src.quant_loader import load_quantized_model_and_tokenizer

PROMPT = "Explain how transformer attention mechanism works in simple terms for a software engineer."
MAX_NEW_TOKENS = 32   # short enough to finish quickly, long enough to profile decode


# ---------------------------------------------------------------------------
# Helper: extract summary from profiler key_averages
# ---------------------------------------------------------------------------

def _cuda_us(e) -> float:
    """Return self CUDA/device time in microseconds, compatible with PyTorch 1.x and 2.x."""
    return getattr(e, "self_cuda_time_total", None) or getattr(e, "self_device_time_total", 0) or 0


def extract_summary(prof, top_n=8):
    """Return a list of dicts from profiler key_averages, sorted by self CUDA time."""
    averages = prof.key_averages()
    rows = sorted(averages, key=_cuda_us, reverse=True)
    total_cuda_us = sum(_cuda_us(e) for e in averages)
    total_cpu_us  = sum(e.self_cpu_time_total for e in averages)

    summary = []
    for e in rows[:top_n]:
        c_us = _cuda_us(e)
        summary.append({
            "op_name":      e.key,
            "calls":        e.count,
            "self_cpu_ms":  round(e.self_cpu_time_total / 1000, 3),
            "self_cuda_ms": round(c_us / 1000, 3),
            "cuda_pct":     round(100.0 * c_us / total_cuda_us, 2) if total_cuda_us > 0 else 0,
        })

    return {
        "total_cpu_ms":  round(total_cpu_us  / 1000, 3),
        "total_cuda_ms": round(total_cuda_us / 1000, 3),
        "cpu_pct":  round(100.0 * total_cpu_us  / (total_cpu_us + total_cuda_us), 2) if (total_cpu_us + total_cuda_us) > 0 else 0,
        "cuda_pct": round(100.0 * total_cuda_us / (total_cpu_us + total_cuda_us), 2) if (total_cpu_us + total_cuda_us) > 0 else 0,
        "top_ops":  summary,
    }



# ---------------------------------------------------------------------------
# nvidia-smi GPU utilization sampler (background thread)
# ---------------------------------------------------------------------------

def sample_gpu_util(out_csv: Path, duration_sec: float, interval_sec: float = 1.0):
    """
    Samples GPU SM utilization via nvidia-smi and writes to a CSV file.
    Returns the list of samples when done.
    """
    samples = []
    t_end = time.perf_counter() + duration_sec
    while time.perf_counter() < t_end:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    samples.append({
                        "t_sec": round(time.perf_counter(), 2),
                        "gpu_util_pct":  int(parts[0].strip()),
                        "mem_util_pct":  int(parts[1].strip()),
                    })
        except Exception:
            pass
        time.sleep(interval_sec)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_sec", "gpu_util_pct", "mem_util_pct"])
        writer.writeheader()
        writer.writerows(samples)

    return samples


# ---------------------------------------------------------------------------
# PHASE 2 — KV-Cache Decode Profiling
# ---------------------------------------------------------------------------

def profile_phase2(model, tokenizer):
    print("\n" + "=" * 60)
    print("  PROFILING PHASE 2: KV-Cache Decode Steps")
    print("=" * 60)

    from src.kv_cache import KVCache
    from transformers.cache_utils import DynamicCache

    device = model.device
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    prompt_len  = prompt_ids.shape[1]

    # Prefill outside profiler to populate KV-cache
    max_cap = prompt_len + MAX_NEW_TOKENS + 64
    past_kv = KVCache.from_model(model, max_seq_len=max_cap, batch_size=1)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=past_kv, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # Warmup decode steps (3 outside profiler)
    for _ in range(3):
        with torch.no_grad():
            out = model(input_ids=current, past_key_values=past_kv, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # --- Profile 5 decode steps ---
    trace_path = str(PROFILES_DIR / "phase2_trace.json")
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        for _ in range(5):
            with record_function("phase2_decode_step"):
                with torch.no_grad():
                    out = model(input_ids=current, past_key_values=past_kv, use_cache=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    prof.export_chrome_trace(trace_path)
    summary = extract_summary(prof)
    summary["phase"] = "Phase 2 - KV-Cache Decode"
    summary["profiled_steps"] = 5
    summary["note"] = (
        "Profiled 5 steady-state decode steps (1-token input, KV-cache populated). "
        "CPU time includes Python dispatch overhead; CUDA time is kernel execution only."
    )

    out_path = PROFILES_DIR / "phase2_profiler_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Total CPU time  : {summary['total_cpu_ms']} ms  ({summary['cpu_pct']}% of CPU+CUDA sum)")
    print(f"  Total CUDA time : {summary['total_cuda_ms']} ms  ({summary['cuda_pct']}% of CPU+CUDA sum)")
    print(f"  Top op (CUDA)   : {summary['top_ops'][0]['op_name']}  {summary['top_ops'][0]['self_cuda_ms']} ms")
    print(f"  Trace           : {trace_path}")
    print(f"  Summary         : {out_path}")

    return summary


# ---------------------------------------------------------------------------
# PHASE 3 — Scheduler Decode Profiling + Python vs CUDA timing
# ---------------------------------------------------------------------------

def profile_phase3(model, tokenizer):
    print("\n" + "=" * 60)
    print("  PROFILING PHASE 3: Scheduler Concurrent Decode")
    print("=" * 60)

    from src.scheduler import ContinuousBatchScheduler

    device = str(model.device)
    sched = ContinuousBatchScheduler(max_batch_size=4, device=device)

    # Add 4 concurrent requests
    for _ in range(4):
        sched.add_request(PROMPT, tokenizer, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, model=model)

    # Warmup: run until all 4 sequences have been prefilled
    steps = 0
    while any(len(s.generated_tokens) == 0 for s in sched.running_batch + sched.waiting_queue) or not sched.running_batch:
        sched.step(model, tokenizer)
        steps += 1
        if steps > 20:
            break

    print(f"  Prefill warmup steps: {steps}")
    print(f"  Running batch size  : {len(sched.running_batch)}")

    # ---- Python vs CUDA split via forward-wrapper injection ----
    # We inject a thin timing wrapper around model.forward so we can
    # separately measure: (a) Python bookkeeping before the forward call,
    # and (b) the actual GPU forward + sync. This avoids reproducing any
    # internal scheduler KV stacking logic.
    forward_times_ms = []   # time inside model() + sync
    step_total_ms    = []   # wall-clock time for full sched.step()

    _orig_forward = model.forward

    def _timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_forward(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    model.forward = _timed_forward

    # Profile 5 actual scheduler steps
    trace_path = str(PROFILES_DIR / "phase3_trace.json")
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        for _ in range(5):
            if not sched.running_batch:
                break
            t_step = time.perf_counter()
            with record_function("phase3_scheduler_step"):
                sched.step(model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_total_ms.append((time.perf_counter() - t_step) * 1000.0)

    model.forward = _orig_forward  # restore original forward

    prof.export_chrome_trace(trace_path)

    # Compute Python overhead = step wall-clock minus forward time
    n = min(len(step_total_ms), len(forward_times_ms))
    if n == 0:
        print("  WARNING: no timing data captured (running_batch was empty).")
        python_overhead_ms_list = [0.0]
        cuda_forward_ms_list    = [0.0]
    else:
        cuda_forward_ms_list    = forward_times_ms[:n]
        python_overhead_ms_list = [step_total_ms[i] - forward_times_ms[i] for i in range(n)]

    mean_py_ms  = sum(python_overhead_ms_list) / len(python_overhead_ms_list)
    mean_fwd_ms = sum(cuda_forward_ms_list)    / len(cuda_forward_ms_list)
    total_step_ms = mean_py_ms + mean_fwd_ms
    py_pct  = 100.0 * mean_py_ms  / total_step_ms if total_step_ms > 0 else 0
    fwd_pct = 100.0 * mean_fwd_ms / total_step_ms if total_step_ms > 0 else 0
    B = len(sched.running_batch)

    timing_result = {
        "phase": "Phase 3 - Scheduler Python vs CUDA Timing",
        "batch_size": B,
        "profiled_steps": n,
        "mean_python_overhead_ms":  round(mean_py_ms, 3),
        "mean_cuda_forward_ms":     round(mean_fwd_ms, 3),
        "mean_total_step_ms":       round(total_step_ms, 3),
        "python_pct_of_step":       round(py_pct, 2),
        "cuda_forward_pct_of_step": round(fwd_pct, 2),
        "per_step_raw": {
            "python_ms":       [round(x, 3) for x in python_overhead_ms_list],
            "cuda_forward_ms": [round(x, 3) for x in cuda_forward_ms_list],
        },
        "method": (
            "Python overhead = sched.step() wall-clock minus model.forward() time. "
            "CUDA forward = model.forward() wall-clock including cuda.synchronize(). "
            f"Batch size = {B} concurrent sequences."
        ),
    }

    timing_path = PROFILES_DIR / "phase3_python_vs_cuda.json"
    with open(timing_path, "w") as f:
        json.dump(timing_result, f, indent=2)

    profiler_summary = extract_summary(prof)
    profiler_summary["phase"]          = "Phase 3 - Scheduler Decode"
    profiler_summary["profiled_steps"] = n
    profiler_summary["batch_size"]     = B

    summary_path = PROFILES_DIR / "phase3_profiler_summary.json"
    with open(summary_path, "w") as f:
        json.dump(profiler_summary, f, indent=2)

    print(f"  Batch size          : {B} sequences")
    print(f"  Mean Python overhead: {mean_py_ms:.3f} ms  ({py_pct:.1f}% of step)")
    print(f"  Mean CUDA forward   : {mean_fwd_ms:.3f} ms  ({fwd_pct:.1f}% of step)")
    print(f"  Mean total step     : {total_step_ms:.3f} ms")
    print(f"  Trace               : {trace_path}")
    print(f"  Python/CUDA timing  : {timing_path}")

    return profiler_summary, timing_result



# ---------------------------------------------------------------------------
# PHASE 3 — GPU Utilization Sampling (nvidia-smi during a 16-req wave)
# ---------------------------------------------------------------------------

def sample_gpu_util_phase3(model, tokenizer):
    print("\n  [Phase 3] Sampling GPU utilization via nvidia-smi during 16-req wave...")

    from src.scheduler import ContinuousBatchScheduler

    device = str(model.device)
    sched = ContinuousBatchScheduler(max_batch_size=16, device=device)
    for _ in range(16):
        sched.add_request(PROMPT, tokenizer, max_new_tokens=16, temperature=0.0, model=model)

    csv_path = PROFILES_DIR / "gpu_util_phase3.csv"
    samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        samples.append({
                            "t_sec": round(time.perf_counter(), 3),
                            "gpu_util_pct": int(parts[0].strip()),
                            "mem_util_pct": int(parts[1].strip()),
                        })
            except Exception:
                pass
            time.sleep(0.5)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    # Run scheduler until all sequences complete
    max_iters = 200
    iters = 0
    while (sched.waiting_queue or sched.running_batch) and iters < max_iters:
        sched.step(model, tokenizer)
        iters += 1

    stop_event.set()
    sampler_thread.join(timeout=3)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_sec", "gpu_util_pct", "mem_util_pct"])
        writer.writeheader()
        writer.writerows(samples)

    if samples:
        mean_gpu = sum(s["gpu_util_pct"] for s in samples) / len(samples)
        max_gpu  = max(s["gpu_util_pct"] for s in samples)
        min_gpu  = min(s["gpu_util_pct"] for s in samples)
        print(f"  GPU util samples: {len(samples)}")
        print(f"  Mean GPU util   : {mean_gpu:.1f}%")
        print(f"  Max / Min       : {max_gpu}% / {min_gpu}%")
        print(f"  CSV             : {csv_path}")
        return {"mean_gpu_util_pct": round(mean_gpu, 1), "max_gpu_util_pct": max_gpu, "min_gpu_util_pct": min_gpu, "n_samples": len(samples)}
    else:
        print("  nvidia-smi sampling failed (no output). GPU util data unavailable.")
        return {"mean_gpu_util_pct": None, "error": "nvidia-smi returned no output"}


# ---------------------------------------------------------------------------
# PHASE 4 — INT8 Decode Profiling
# ---------------------------------------------------------------------------

def profile_phase4(model, tokenizer):
    print("\n" + "=" * 60)
    print("  PROFILING PHASE 4: INT8 Quantized Decode Steps")
    print("=" * 60)

    from transformers import DynamicCache

    device = next(model.parameters()).device
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    kv_cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, past_key_values=kv_cache, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # Warmup 3 steps
    for _ in range(3):
        with torch.no_grad():
            out = model(input_ids=current, past_key_values=kv_cache, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # Profile 5 decode steps
    trace_path = str(PROFILES_DIR / "phase4_trace.json")
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        for _ in range(5):
            with record_function("phase4_int8_decode_step"):
                with torch.no_grad():
                    out = model(input_ids=current, past_key_values=kv_cache, use_cache=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                current = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    prof.export_chrome_trace(trace_path)
    summary = extract_summary(prof, top_n=10)
    summary["phase"] = "Phase 4 - INT8 Decode"
    summary["profiled_steps"] = 5
    summary["note"] = (
        "Profiled 5 steady-state INT8 decode steps. "
        "Look for ops containing 'int8', 'dequant', or 'mm' to identify quantization overhead."
    )

    out_path = PROFILES_DIR / "phase4_profiler_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Total CPU time  : {summary['total_cpu_ms']} ms  ({summary['cpu_pct']}% of CPU+CUDA sum)")
    print(f"  Total CUDA time : {summary['total_cuda_ms']} ms  ({summary['cuda_pct']}% of CPU+CUDA sum)")
    if summary['top_ops']:
        print(f"  Top op (CUDA)   : {summary['top_ops'][0]['op_name']}  {summary['top_ops'][0]['self_cuda_ms']} ms")
    print(f"  Trace           : {trace_path}")
    print(f"  Summary         : {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nMicroInfer Profiler Pass — device={device.upper()}")
    print(f"Outputs -> {PROFILES_DIR}\n")

    results = {}

    # ---- Phase 2: FP16 KV-cache ----
    print("[1/4] Loading FP16 model for Phase 2 + Phase 3...")
    fp16_model, tokenizer = load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, device=device)
    fp16_model.eval()

    p2 = profile_phase2(fp16_model, tokenizer)
    results["phase2"] = p2

    p3_prof, p3_timing = profile_phase3(fp16_model, tokenizer)
    results["phase3_profiler"] = p3_prof
    results["phase3_timing"]   = p3_timing

    p3_gpu = sample_gpu_util_phase3(fp16_model, tokenizer)
    results["phase3_gpu_util"] = p3_gpu

    del fp16_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Phase 4: INT8 ----
    print("\n[2/4] Loading INT8 model for Phase 4...")
    int8_model, _ = load_quantized_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, device=device)
    int8_model.eval()

    p4 = profile_phase4(int8_model, tokenizer)
    results["phase4"] = p4

    del int8_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Save combined results
    combined_path = PROFILES_DIR / "all_phases_summary.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  PROFILING COMPLETE — Summary")
    print("=" * 60)
    print(f"  Phase 2 CPU/CUDA ratio : {p2['total_cpu_ms']:.1f} ms CPU / {p2['total_cuda_ms']:.1f} ms CUDA")
    print(f"  Phase 3 Python overhead: {p3_timing['mean_python_overhead_ms']:.1f} ms  ({p3_timing['python_pct_of_step']:.1f}% of step)")
    print(f"  Phase 3 CUDA forward   : {p3_timing['mean_cuda_forward_ms']:.1f} ms  ({p3_timing['cuda_forward_pct_of_step']:.1f}% of step)")
    if p3_gpu.get("mean_gpu_util_pct") is not None:
        print(f"  Phase 3 GPU utilization: mean={p3_gpu['mean_gpu_util_pct']}%  max={p3_gpu['max_gpu_util_pct']}%")
    print(f"  Phase 4 CPU/CUDA ratio : {p4['total_cpu_ms']:.1f} ms CPU / {p4['total_cuda_ms']:.1f} ms CUDA")
    print(f"\n  All files in: {PROFILES_DIR}")


if __name__ == "__main__":
    main()
