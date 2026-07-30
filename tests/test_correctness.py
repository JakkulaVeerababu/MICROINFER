"""
MicroInfer - Correctness Test Suite
Verifies that custom generation loops match HuggingFace baseline outputs token-for-token.
"""

import sys
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from src.model_loader import load_model_and_tokenizer
from src.naive_generate import naive_generate


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    return model, tokenizer


def test_naive_generate_matches_hf(model_and_tokenizer):
    """
    Verifies that naive_generate (uncached) produces identical tokens to HF .generate()
    when using greedy decoding (use_cache=False).
    """
    model, tokenizer = model_and_tokenizer
    prompt = "Artificial Intelligence is transforming software engineering by"
    max_new_tokens = 15

    # 1. HuggingFace .generate() baseline without caching
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        hf_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    hf_tokens = hf_output[0][inputs.input_ids.shape[1]:].tolist()
    hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=True)

    # 2. Custom naive_generate() output
    naive_res = naive_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    print("\n[HF Output]:   ", hf_text)
    print("[Naive Output]:", naive_res["output_text"])

    # Verify matching token IDs
    assert naive_res["output_text"].strip() == hf_text.strip(), (
        f"Mismatch between HF and Naive Generator!\nHF: '{hf_text}'\nNaive: '{naive_res['output_text']}'"
    )
