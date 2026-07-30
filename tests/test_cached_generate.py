"""
MicroInfer - Unit Tests for Sub-Phase 2.2 / 2.3 Cached Generator & Correctness
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.cached_generate import cached_generate


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    return model, tokenizer


def test_cached_generate_execution(loaded_model_and_tokenizer):
    """
    Verifies that cached_generate executes 2-phase generation (prefill + decode) properly.
    """
    model, tokenizer = loaded_model_and_tokenizer
    prompt = "KV-caching optimizes GPU memory bandwidth by"
    max_tokens = 20

    res = cached_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    assert "output_text" in res and len(res["output_text"]) > 0
    assert "generated_tokens" in res and res["generated_tokens"] > 0
    assert len(res["step_times_ms"]) == res["generated_tokens"]
    assert res["ttft_ms"] > 0.0
    assert res["tpot_ms"] > 0.0
    assert res["throughput_tok_per_sec"] > 0.0


def test_cached_generate_matches_hf_baseline(loaded_model_and_tokenizer):
    """
    Verifies token-for-token equivalence between cached_generate() and HF .generate()
    under greedy decoding (temperature=0.0).
    """
    model, tokenizer = loaded_model_and_tokenizer
    prompt = "Artificial Intelligence is transforming software engineering by"
    max_tokens = 15

    # 1. HuggingFace baseline
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        hf_out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    hf_tokens = hf_out[0][inputs.input_ids.shape[1]:].tolist()
    hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=True)

    # 2. Custom cached_generate output
    cached_res = cached_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    print("\n[HF Output]:    ", hf_text)
    print("[Cached Output]:", cached_res["output_text"])

    assert cached_res["output_text"].strip() == hf_text.strip(), (
        f"Mismatch between HF and Cached Generator!\nHF: '{hf_text}'\nCached: '{cached_res['output_text']}'"
    )
