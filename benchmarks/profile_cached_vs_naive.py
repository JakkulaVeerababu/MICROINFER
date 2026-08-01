"""
MicroInfer - Phase 2 KV-Cache vs Phase 1 Naive Profiler
=========================================================

Profiles cached_generate.py against naive_generate.py at N=64 using
torch.profiler with record_function isolation for KVCache::update and
KVCacheLayer::update.

Goal: measure exactly how much per-step overhead the custom KVCache adds
on top of the model forward pass, vs. Phase 1's zero-overhead naive loop.

Outputs:
  - analysis/profiles/profile_cached_vs_naive_summary.txt  (human-readable breakdown)
  - analysis/profiles/profile_cached_vs_naive.json         (structured for README)
  - analysis/profiles/trace_p2_decode.json                 (Chrome trace)

Usage:
    python benchmarks/profile_cached_vs_naive.py
"""

import sys
import json
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.naive_generate import naive_generate
from src.cached_generate import cached_generate

# ---------------------------------------------------------------------------
# Config — canonical N=64 condition matching the master benchmark table
# ---------------------------------------------------------------------------
PROFILES_DIR = ROOT_DIR / "analysis" / "profiles"
SUMMARY_PATH = PROFILES_DIR / "profile_cached_vs_naive_summary.txt"
JSON_PATH    = PROFILES_DIR / "profile_cached_vs_naive.json"
TRACE_CACHED = PROFILES_DIR / "trace_p2_decode.json"

PROMPT       = "Explain how transformer attention mechanism works in simple terms."
MAX_NEW_TOKENS = 64   # N=64 decode steps — same as master table canonical condition
NUM_WARMUP   = 3


def _extract_ops(key_avgs, top_n: int = 15):
    """Return list of dicts from key_averages, sorted by self_cuda_time_total."""
    items = sorted(
        key_avgs,
        key=lambda x: getattr(x, "self_cuda_time_total", 0),
        reverse=True,
    )
    result = []
    for item in items[:top_n]:
        result.append({
            "op": item.key,
            "calls": getattr(item, "count", 0),
            "self_cuda_ms": round(getattr(item, "self_cuda_time_total", 0) / 1000.0, 3),
            "self_cpu_ms":  round(getattr(item, "self_cpu_time_total",  0) / 1000.0, 3),
            "cuda_time_total_ms": round(getattr(item, "cuda_time_total", 0) / 1000.0, 3),
        })
    return result


def _totals(key_avgs):
    total_cuda = sum(getattr(i, "self_cuda_time_total", 0) for i in key_avgs)
    total_cpu  = sum(getattr(i, "self_cpu_time_total",  0) for i in key_avgs)
    return total_cuda, total_cpu


def _cache_overhead(key_avgs):
    """Sum self_cuda_time and self_cpu_time for KVCache record_function spans."""
    cache_cuda = sum(
        getattr(i, "self_cuda_time_total", 0) for i in key_avgs
        if "KVCache" in i.key
    )
    cache_cpu = sum(
        getattr(i, "self_cpu_time_total", 0) for i in key_avgs
        if "KVCache" in i.key
    )
    return cache_cuda, cache_cpu


def run_profile():
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("[WARNING] No CUDA GPU detected. Profiling on CPU -- CUDA times will be 0.", flush=True)

    print("=" * 70, flush=True)
    print(" MicroInfer -- Phase 2 vs Phase 1 Overhead Profiler", flush=True)
    print("=" * 70, flush=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device     : {gpu_name}", flush=True)
    print(f"Model      : {DEFAULT_MODEL_ID}", flush=True)
    print(f"Decode     : {MAX_NEW_TOKENS} new tokens", flush=True)

    print("\n[1/5] Loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, device=device)
    model.eval()

    actual_prompt_tokens = len(tokenizer.encode(PROMPT))
    print(f"Prompt token count: {actual_prompt_tokens}", flush=True)

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------
    print(f"\n[2/5] Warm-up ({NUM_WARMUP} runs each, discarded)...", flush=True)
    for _ in range(NUM_WARMUP):
        naive_generate(model, tokenizer, PROMPT, max_new_tokens=4)
        cached_generate(model, tokenizer, PROMPT, max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Profile Phase 1 -- Naive (no cache)
    # -----------------------------------------------------------------------
    print("\n[3/5] Profiling Phase 1 (Naive, no cache)...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof_naive:
        with record_function("naive_full_generation"):
            naive_result = naive_generate(model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    naive_avgs = prof_naive.key_averages()
    naive_total_cuda, naive_total_cpu = _totals(naive_avgs)
    naive_ops = _extract_ops(naive_avgs)
    naive_throughput = naive_result["throughput_tok_per_sec"]
    naive_tpot = naive_result["tpot_ms"]
    print(f"  Phase 1 result: {naive_throughput:.2f} tok/s  |  TPOT={naive_tpot:.2f} ms/tok", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Profile Phase 2 -- KV-Cached
    # -----------------------------------------------------------------------
    print("\n[4/5] Profiling Phase 2 (KV-Cached, custom KVCache)...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof_cached:
        with record_function("cached_full_generation"):
            cached_result = cached_generate(model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    cached_avgs = prof_cached.key_averages()
    cached_total_cuda, cached_total_cpu = _totals(cached_avgs)
    cache_cuda_overhead, cache_cpu_overhead = _cache_overhead(cached_avgs)
    cached_ops = _extract_ops(cached_avgs)
    cached_throughput = cached_result["throughput_tok_per_sec"]
    cached_tpot = cached_result["tpot_ms"]
    print(f"  Phase 2 result: {cached_throughput:.2f} tok/s  |  TPOT={cached_tpot:.2f} ms/tok", flush=True)

    try:
        prof_cached.export_chrome_trace(str(TRACE_CACHED))
        print(f"  Chrome trace -> '{TRACE_CACHED}'", flush=True)
    except Exception as e:
        print(f"  Trace export warning: {e}", flush=True)

    # -----------------------------------------------------------------------
    # Compute breakdown
    # -----------------------------------------------------------------------
    gen_tokens = cached_result["generated_tokens"]
    decode_steps = max(gen_tokens - 1, 1)

    cache_cuda_pct = (cache_cuda_overhead / cached_total_cuda * 100) if cached_total_cuda > 0 else 0.0
    cache_cpu_pct  = (cache_cpu_overhead  / cached_total_cpu  * 100) if cached_total_cpu  > 0 else 0.0

    cache_cuda_per_step_ms = (cache_cuda_overhead / 1000.0) / decode_steps
    cache_cpu_per_step_ms  = (cache_cpu_overhead  / 1000.0) / decode_steps

    delta_pct = (cached_throughput / naive_throughput - 1) * 100 if naive_throughput > 0 else 0.0

    # -----------------------------------------------------------------------
    # Build text report
    # -----------------------------------------------------------------------
    sep = "=" * 70
    lines = [
        "MicroInfer -- Phase 2 vs Phase 1 Overhead Profiler Report",
        f"GPU   : {gpu_name}",
        f"Model : {DEFAULT_MODEL_ID}",
        f"Prompt tokens: {actual_prompt_tokens}  |  Decode steps: {MAX_NEW_TOKENS}",
        "",
        sep,
        "THROUGHPUT COMPARISON (N=64 canonical condition)",
        sep,
        f"  Phase 1 (Naive, no cache)  : {naive_throughput:6.2f} tok/s  |  TPOT = {naive_tpot:.2f} ms/tok",
        f"  Phase 2 (KV-Cache engine)  : {cached_throughput:6.2f} tok/s  |  TPOT = {cached_tpot:.2f} ms/tok",
        f"  Delta                      : {cached_throughput - naive_throughput:+.2f} tok/s  ({delta_pct:+.1f}%)",
        "",
        sep,
        "PHASE 2 -- KVCACHE OVERHEAD BREAKDOWN",
        sep,
        f"  Total Phase 2 CUDA self-time : {cached_total_cuda/1000:.2f} ms  over {gen_tokens} tokens",
        f"  KVCache::update CUDA time    : {cache_cuda_overhead/1000:.2f} ms  ({cache_cuda_pct:.1f}% of total CUDA)",
        f"  KVCache::update CPU time     : {cache_cpu_overhead/1000:.2f} ms  ({cache_cpu_pct:.1f}% of total CPU)",
        f"  Cache overhead per step      : {cache_cuda_per_step_ms:.4f} ms CUDA  |  {cache_cpu_per_step_ms:.4f} ms CPU",
        "",
        sep,
        "PHASE 1 -- NAIVE PROFILER TOTALS",
        sep,
        f"  Total CUDA self-time : {naive_total_cuda/1000:.2f} ms  over {naive_result['generated_tokens']} tokens",
        f"  Total CPU  self-time : {naive_total_cpu/1000:.2f} ms",
        "",
        sep,
        "PHASE 1 -- TOP 15 OPS BY SELF CUDA TIME",
        sep,
    ]
    for op in naive_ops:
        lines.append(
            f"  {op['op']:<45s}  cuda={op['self_cuda_ms']:8.3f}ms  cpu={op['self_cpu_ms']:7.3f}ms  calls={op['calls']}"
        )

    lines += [
        "",
        sep,
        "PHASE 2 -- TOP 15 OPS BY SELF CUDA TIME  (* = KVCache overhead)",
        sep,
    ]
    for op in cached_ops:
        marker = " *" if "KVCache" in op["op"] else "  "
        lines.append(
            f"{marker} {op['op']:<45s}  cuda={op['self_cuda_ms']:8.3f}ms  cpu={op['self_cpu_ms']:7.3f}ms  calls={op['calls']}"
        )

    lines += ["", sep, "DIAGNOSIS", sep]

    if delta_pct >= 0:
        lines.append(
            f"  Phase 2 BEATS Phase 1 by {delta_pct:+.1f}% at N=64 after KVCache.update() optimization.\n"
            f"  KVCache::update overhead: {cache_cpu_per_step_ms:.4f} ms CPU / {cache_cuda_per_step_ms:.4f} ms CUDA per decode step."
        )
    else:
        lines.append(
            f"  Phase 2 is {abs(delta_pct):.1f}% SLOWER than Phase 1 at N=64.\n"
            f"  Root cause: KVCache Python dispatch overhead ({cache_cpu_per_step_ms:.4f} ms/step CPU, {cache_cuda_per_step_ms:.4f} ms/step CUDA)\n"
            f"  adds to identical FFN compute cost; attention savings at N=64 are smaller than this overhead.\n"
            f"  KV-caching advantage emerges at longer N -- see benchmark_p2_vs_p1_scaling.py for crossover."
        )

    report = "\n".join(lines)
    print("\n" + report, flush=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[5/5] Summary -> '{SUMMARY_PATH}'", flush=True)

    result_json = {
        "model_id": DEFAULT_MODEL_ID,
        "gpu": gpu_name,
        "prompt_tokens": actual_prompt_tokens,
        "decode_steps": MAX_NEW_TOKENS,
        "phase1_naive": {
            "throughput_tok_per_sec": naive_throughput,
            "tpot_ms": naive_tpot,
            "total_cuda_ms": round(naive_total_cuda / 1000.0, 3),
            "total_cpu_ms":  round(naive_total_cpu  / 1000.0, 3),
            "top_ops": naive_ops,
        },
        "phase2_cached": {
            "throughput_tok_per_sec": cached_throughput,
            "tpot_ms": cached_tpot,
            "total_cuda_ms": round(cached_total_cuda / 1000.0, 3),
            "total_cpu_ms":  round(cached_total_cpu  / 1000.0, 3),
            "kvcache_overhead": {
                "cuda_total_ms":     round(cache_cuda_overhead / 1000.0, 3),
                "cpu_total_ms":      round(cache_cpu_overhead  / 1000.0, 3),
                "cuda_pct":          round(cache_cuda_pct, 2),
                "cpu_pct":           round(cache_cpu_pct,  2),
                "cuda_per_step_ms":  round(cache_cuda_per_step_ms, 4),
                "cpu_per_step_ms":   round(cache_cpu_per_step_ms,  4),
            },
            "top_ops": cached_ops,
        },
        "delta_tok_per_sec": round(cached_throughput - naive_throughput, 3),
        "delta_pct":         round(delta_pct, 2),
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    print(f"         JSON  -> '{JSON_PATH}'", flush=True)

    return result_json


if __name__ == "__main__":
    run_profile()
