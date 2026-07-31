"""
MicroInfer - Phase 2: Incremental KV-Cache Generation Engine
============================================================

WHAT WE ACTUALLY OWN IN THIS FILE
-----------------------------------
This module drives a two-phase (Prefill + Decode) autoregressive generation loop
using MicroInfer's custom pre-allocated 5D CUDA tensor store (`KVCache` in `src/kv_cache.py`).

1. It explicitly separates the Prefill step (full prompt → one forward pass)
   from Decode steps (1 token at a time).

2. It measures TTFT and TPOT independently with CUDA-synchronised wall-clock
   timers.

3. It manages the `KVCache` lifecycle manually — pre-allocating contiguous
   5D CUDA tensors up front and updating layer slices during model execution.
"""

import time
import torch
from typing import Any, Dict
from src.kv_cache import KVCache


def cached_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,  # 0.0 = greedy decoding
) -> Dict[str, Any]:
    """
    Two-phase KV-cache generation loop using MicroInfer's pre-allocated KVCache.

    Phase A — Prefill
        Run the full prompt through the model in one forward pass.
        KVCache is populated with K/V projections for all prompt tokens.
        Time to first token (TTFT) is measured here.

    Phase B — Decode
        Feed one new token per step; KVCache carries forward all previous
        K/V projections in pre-allocated memory so each attention call is O(1).
        Per-token time (TPOT) is the mean over all decode steps.

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

    # Pre-allocate KVCache upfront matching model architecture
    max_capacity = prompt_len + max_new_tokens + 64
    past_key_values = KVCache.from_model(model, max_seq_len=max_capacity, batch_size=1)

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
            use_cache=True,
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
    # ------------------------------------------------------------------ #
    for step in range(1, max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break

        t_step_start = time.perf_counter()
        with torch.no_grad():
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
    print(f"Output: '{res['output_text']}'")
    print(f"TTFT: {res['ttft_ms']} ms | TPOT: {res['tpot_ms']} ms/tok | Throughput: {res['throughput_tok_per_sec']} tok/s")
