"""
MicroInfer - Sub-Phase 2.3: Correctness Verification Suite for Cached Generator
Verifies token-by-token output equivalence between Cached Generator and Naive Generator.
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.naive_generate import naive_generate
from src.cached_generate import cached_generate


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    return model, tokenizer


@pytest.mark.parametrize("prompt", [
    "Artificial Intelligence is transforming software engineering by",
    "Explain how transformer attention mechanism works in simple terms for",
    "What are the key trade-offs between KV-caching and continuous batching in",
])
def test_cached_matches_naive_generator(loaded_model_and_tokenizer, prompt):
    """
    Verifies 100% token-for-token equivalence between cached_generate() (O(n) KV-cached)
    and naive_generate() (O(n^2) uncached) under greedy decoding.
    """
    model, tokenizer = loaded_model_and_tokenizer
    max_tokens = 20

    # 1. Uncached Naive Generator output
    naive_res = naive_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    # 2. KV-Cached Generator output
    cached_res = cached_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    print(f"\n[Prompt]:       '{prompt}'")
    print(f"[Naive Output]: '{naive_res['output_text']}'")
    print(f"[Cached Output]:'{cached_res['output_text']}'")

    assert cached_res["output_text"].strip() == naive_res["output_text"].strip(), (
        f"Mismatch between Naive and Cached Generator for prompt '{prompt}'!\nNaive: '{naive_res['output_text']}'\nCached: '{cached_res['output_text']}'"
    )


def test_cached_generate_sampling_temperature(loaded_model_and_tokenizer):
    """
    Verifies temperature sampling execution (temperature=0.7) produces valid text.
    """
    model, tokenizer = loaded_model_and_tokenizer
    prompt = "The future of LLM serving infrastructure"
    
    res = cached_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.7,
    )

    assert "output_text" in res and len(res["output_text"]) > 0
    assert res["generated_tokens"] == 10
