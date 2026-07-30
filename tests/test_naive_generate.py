"""
MicroInfer - Unit Tests for Sub-Phase 1.1 Naive Generator Module
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.naive_generate import naive_generate


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    return model, tokenizer


def test_naive_generate_functionality(loaded_model_and_tokenizer):
    """
    Verifies that naive_generate executes generation and produces valid timing statistics.
    """
    model, tokenizer = loaded_model_and_tokenizer
    prompt = "The fundamentals of database indexing are"
    max_tokens = 15

    res = naive_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    assert "output_text" in res and len(res["output_text"]) > 0
    assert "generated_tokens" in res and res["generated_tokens"] > 0
    assert "step_times_ms" in res and len(res["step_times_ms"]) == res["generated_tokens"]
    assert res["ttft_ms"] > 0.0
    assert res["tpot_ms"] > 0.0
    assert res["throughput_tok_per_sec"] > 0.0
