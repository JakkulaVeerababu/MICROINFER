"""
MicroInfer - Phase 1: Naive Generator (No KV-Cache)
Implements an uncached forward-pass generation loop to demonstrate O(n^2) quadratic scaling penalty.
"""

import time
import torch
from typing import Optional, List, Dict, Any


def naive_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,  # 0.0 = Greedy Decoding
) -> Dict[str, Any]:
    """
    Generates text by re-running the complete forward pass for every generated token.
    
    Args:
        model: HuggingFace AutoModelForCausalLM instance.
        tokenizer: Corresponding AutoTokenizer instance.
        prompt: Input text string.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for greedy decoding).
        
    Returns:
        dict: Summary containing generated text, token counts, per-step timing, and throughput.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs.input_ids
    prompt_len = generated.shape[1]

    step_times = []
    t_start = time.perf_counter()

    for step in range(max_new_tokens):
        t_step_start = time.perf_counter()
        
        with torch.no_grad():
            # Crucial: Full sequence re-computation on EVERY step
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
        generated = torch.cat([generated, next_token], dim=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_step_end = time.perf_counter()
        step_times.append((t_step_end - t_step_start) * 1000.0)

        # Stop on EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    t_total = time.perf_counter() - t_start
    generated_token_count = generated.shape[1] - prompt_len

    output_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    ttft_ms = step_times[0] if step_times else 0.0
    tpot_ms = (sum(step_times[1:]) / len(step_times[1:])) if len(step_times) > 1 else ttft_ms
    throughput = generated_token_count / t_total if t_total > 0 else 0.0

    return {
        "output_text": output_text,
        "full_text": full_text,
        "prompt_tokens": prompt_len,
        "generated_tokens": generated_token_count,
        "total_time_sec": round(t_total, 3),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "throughput_tok_per_sec": round(throughput, 2),
        "step_times_ms": step_times,
    }


if __name__ == "__main__":
    from src.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    res = naive_generate(model, tokenizer, "The future of artificial intelligence in infrastructure is", max_new_tokens=32)
    print("\n--- Naive Generation Output ---")
    print(res["output_text"])
    print(f"\nTTFT: {res['ttft_ms']} ms | TPOT: {res['tpot_ms']} ms/tok | Throughput: {res['throughput_tok_per_sec']} tok/s")
