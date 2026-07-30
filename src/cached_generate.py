"""
MicroInfer - Phase 2: Cached Incremental Generator Module
Implements 2-phase generation (Prefill + Decode) using KV-caching to achieve O(n) linear complexity.
"""

import time
import torch
from typing import Optional, List, Dict, Any
from transformers.cache_utils import DynamicCache


def cached_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,  # 0.0 = Greedy Decoding
) -> Dict[str, Any]:
    """
    Generates text using KV-caching.
    Prefill Phase processes prompt tokens; Decode Phase processes 1 single token per step.
    
    Args:
        model: HuggingFace AutoModelForCausalLM instance.
        tokenizer: Corresponding AutoTokenizer instance.
        prompt: Input text string.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for greedy decoding).
        
    Returns:
        dict: Summary containing generated text, token counts, TTFT, TPOT, and per-step timing.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    prompt_len = prompt_ids.shape[1]

    # Initialize dynamic KV-cache
    past_key_values = DynamicCache()
    step_times = []
    generated_tokens = []

    t_start = time.perf_counter()

    # 1. Prefill Phase (Step 0): Process full input prompt
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids,
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

    t_prefill_end = time.perf_counter()
    prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000.0
    step_times.append(prefill_time_ms)

    generated_tokens.append(next_token.item())
    current_input = next_token

    # 2. Decode Phase (Steps 1 ... max_new_tokens - 1): Process 1 single token per step
    for step in range(1, max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break

        t_step_start = time.perf_counter()
        with torch.no_grad():
            # Feed ONLY single token (shape 1x1) with past_key_values
            outputs = model(
                input_ids=current_input,
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
    res = cached_generate(model, tokenizer, "KV-caching accelerates transformer inference by", max_new_tokens=32)
    print("\n--- Cached Generation Output ---")
    print(res["output_text"])
    print(f"\nTTFT: {res['ttft_ms']} ms | TPOT: {res['tpot_ms']} ms/tok | Throughput: {res['throughput_tok_per_sec']} tok/s")
