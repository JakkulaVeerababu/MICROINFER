"""
MicroInfer - Phase 4: INT8 Quantized Generator Engine Module
Executes 2-phase generation (Prefill + Decode) using 8-bit quantized linear weights with KV-caching.
"""

import time
import torch
from typing import Dict, Any, List
from transformers import DynamicCache

from src.quant_loader import load_quantized_model_and_tokenizer


def quant_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Generates text using 8-bit quantized weights and KV-caching.
    Measures TTFT, TPOT, tokens/sec throughput, and per-step decoding latency.
    """
    device = next(model.parameters()).device
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[1]

    kv_cache = DynamicCache()
    step_times_ms: List[float] = []
    generated_token_ids: List[int] = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    # --- Phase 1: Prefill Step (Full Prompt) ---
    t_step_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            past_key_values=kv_cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_prefill_end = time.perf_counter()
    prefill_time_ms = (t_prefill_end - t_step_start) * 1000.0
    step_times_ms.append(prefill_time_ms)

    next_token_id = next_token.item()
    generated_token_ids.append(next_token_id)
    current_input_ids = next_token

    # --- Phase 2: Decode Steps (1x1 Tokens) ---
    for step in range(1, max_new_tokens):
        if next_token_id == tokenizer.eos_token_id:
            break

        t_decode_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_decode_end = time.perf_counter()
        decode_time_ms = (t_decode_end - t_decode_start) * 1000.0
        step_times_ms.append(decode_time_ms)

        next_token_id = next_token.item()
        generated_token_ids.append(next_token_id)
        current_input_ids = next_token

    t_total_end = time.perf_counter()
    total_time_sec = t_total_end - t_start

    generated_count = len(generated_token_ids)
    output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    ttft_ms = step_times_ms[0]
    decode_step_times = step_times_ms[1:] if len(step_times_ms) > 1 else [0.0]
    tpot_ms = sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0.0
    throughput = generated_count / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_count,
        "output_text": output_text,
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "throughput_tok_per_sec": round(throughput, 2),
        "total_time_sec": round(total_time_sec, 3),
        "step_times_ms": [round(st, 2) for st in step_times_ms],
    }


if __name__ == "__main__":
    model, tokenizer = load_quantized_model_and_tokenizer()
    res = quant_generate(model, tokenizer, "INT8 quantization in deep learning accelerates", max_new_tokens=20)
    print(f"\nOutput: '{res['output_text']}'")
    print(f"TTFT: {res['ttft_ms']} ms | TPOT: {res['tpot_ms']} ms/tok | Throughput: {res['throughput_tok_per_sec']} tok/s")
