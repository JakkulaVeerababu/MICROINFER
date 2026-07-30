"""
MicroInfer - Phase 2: Incremental KV-Cache Generation Engine
============================================================

WHAT WE ACTUALLY OWN IN THIS FILE
-----------------------------------
HuggingFace's `model(input_ids=..., past_key_values=..., use_cache=True)` API
stores attention key/value projections inside a `DynamicCache` object that lives
*outside* the model weights — it grows one decode step at a time without the
model being aware.  This module's contribution is **the generation loop and
timing infrastructure that drives that cache correctly**:

    1. It explicitly separates the Prefill step (full prompt → one forward pass)
       from the Decode steps (1 token at a time), a two-phase discipline that
       vanilla `model.generate()` hides inside C++/Rust internals.

    2. It measures TTFT and TPOT independently with CUDA-synchronised wall-clock
       timers, rather than relying on transformers' own aggregate timer.

    3. It manages the `DynamicCache` lifecycle manually — instantiate, pass in,
       let HF grow it — which is equivalent to how a production serving engine
       like vLLM or TRT-LLM manages its own cache object per sequence slot.

WHAT HUGGINGFACE PROVIDES
--------------------------
- The actual K/V tensor storage (DynamicCache, backed by a Python list of tuples
  of CUDA tensors, one per layer, growing with each decoded token).
- The attention kernel that reads those tensors during each forward pass.

WHY NOT THE CUSTOM KVCache CLASS (src/kv_cache.py)?
----------------------------------------------------
`src/kv_cache.py` implements a pre-allocated 5-D CUDA tensor store
`(num_layers, batch, num_kv_heads, max_seq_len, head_dim)` that owns its own
memory up front (no Python list growth).  Wiring it *directly* into the model's
attention layers would require monkey-patching each layer's forward() method —
an interesting but model-architecture-specific exercise.  That path is noted as
an open design question in the Phase 2 spec (PHASE2_SPEC.md, line 106) and in
ANALYSIS.md.  This file deliberately keeps the loop clean and the claim honest.

Open engineering question (not hidden): what would it take to use KVCache here?
    → Patch Qwen2Attention.forward() to read/write self._kv_store (our object)
      instead of the DynamicCache.  Requires matching (batch, heads, seq, dim)
      slice indexing to the model's exact layout.  Doable; left as next step.
"""

import time
import torch
from typing import Any, Dict
from transformers.cache_utils import DynamicCache


def cached_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,  # 0.0 = greedy decoding
) -> Dict[str, Any]:
    """
    Two-phase KV-cache generation loop.

    Phase A — Prefill
        Run the full prompt through the model in one forward pass.
        DynamicCache is populated with K/V projections for all prompt tokens.
        Time to first token (TTFT) is measured here.

    Phase B — Decode
        Feed one new token per step; DynamicCache carries forward all previous
        K/V projections so each attention call is O(1) in context length.
        Per-token time (TPOT) is the mean over all decode steps.

    MicroInfer's contribution:
        - Explicit two-phase separation with independent CUDA-synchronised timers.
        - Manual DynamicCache lifecycle management (instantiate once, reuse).
        - Returns a structured dict with TTFT, TPOT, throughput, and per-step
          latency list for downstream analysis and plotting.

    Args:
        model:          HuggingFace AutoModelForCausalLM on CUDA or CPU.
        tokenizer:      Corresponding AutoTokenizer instance.
        prompt:         Input text string.
        max_new_tokens: Maximum tokens to generate (default 128).
        temperature:    Sampling temperature; 0.0 = greedy (default).

    Returns:
        dict with keys:
            output_text, full_text, prompt_tokens, generated_tokens,
            total_time_sec, ttft_ms, tpot_ms, throughput_tok_per_sec,
            step_times_ms
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    prompt_len = prompt_ids.shape[1]

    # --- MicroInfer owns: cache object instantiation & lifecycle ---
    # HuggingFace provides: the DynamicCache data structure and the attention
    # kernel that reads from it on every forward pass.
    past_key_values = DynamicCache()

    step_times = []
    generated_tokens = []

    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Phase A: Prefill — process the full prompt in a single forward pass  #
    # ------------------------------------------------------------------ #
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids,        # shape (1, prompt_len)
            past_key_values=past_key_values,
            use_cache=True,              # instructs HF to populate past_key_values
        )
        next_token_logits = outputs.logits[:, -1, :]  # last token position

        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_prefill_end = time.perf_counter()
    prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000.0
    step_times.append(prefill_time_ms)

    generated_tokens.append(next_token.item())
    current_input = next_token

    # ------------------------------------------------------------------ #
    # Phase B: Decode — one token per step, O(1) attention per step       #
    # Each call to model() reads cached K/V via DynamicCache;             #
    # attention cost does NOT grow with context length here.              #
    # ------------------------------------------------------------------ #
    for step in range(1, max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break

        t_step_start = time.perf_counter()
        with torch.no_grad():
            # Feed ONLY the single new token (shape 1×1).
            # DynamicCache supplies all prior K/V; no re-computation occurs.
            outputs = model(
                input_ids=current_input,         # shape (1, 1)
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_step_end = time.perf_counter()
        step_times.append((t_step_end - t_step_start) * 1000.0)

        generated_tokens.append(next_token.item())
        current_input = next_token

    t_total = time.perf_counter() - t_start
    gen_count = len(generated_tokens)

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + output_text

    ttft_ms = step_times[0]
    tpot_ms = (sum(step_times[1:]) / len(step_times[1:])) if len(step_times) > 1 else ttft_ms
    throughput = gen_count / t_total if t_total > 0 else 0.0

    return {
        "output_text": output_text,
        "full_text": full_text,
        "prompt_tokens": prompt_len,
        "generated_tokens": gen_count,
        "total_time_sec": round(t_total, 3),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "throughput_tok_per_sec": round(throughput, 2),
        "step_times_ms": step_times,
    }


if __name__ == "__main__":
    from src.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    res = cached_generate(
        model, tokenizer,
        "KV-caching accelerates transformer inference by",
        max_new_tokens=32,
    )
    print("\n--- Cached Generation Output ---")
    print(res["output_text"])
    print(
        f"\nTTFT: {res['ttft_ms']} ms | "
        f"TPOT: {res['tpot_ms']} ms/tok | "
        f"Throughput: {res['throughput_tok_per_sec']} tok/s"
    )
