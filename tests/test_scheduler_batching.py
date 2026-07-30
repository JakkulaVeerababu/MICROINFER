"""
MicroInfer - Unit Tests for Sub-Phase 3.2 Batched Tensor Forward Execution
Stress tests continuous batching with staggered concurrent requests and varying output lengths.
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.scheduler import ContinuousBatchScheduler, SequenceState


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    return model, tokenizer


def test_staggered_concurrent_requests(loaded_model_and_tokenizer):
    """
    Verifies scheduler handles staggered request additions while batch is actively running.
    """
    model, tokenizer = loaded_model_and_tokenizer
    scheduler = ContinuousBatchScheduler(max_batch_size=3)

    # 1. Add initial 2 requests
    s1 = scheduler.add_request("Explain quantum computing in simple terms", tokenizer, max_new_tokens=15)
    s2 = scheduler.add_request("Write a short poem about space exploration", tokenizer, max_new_tokens=10)

    # Step 1: Start s1 and s2
    scheduler.step(model, tokenizer)
    assert len(scheduler.running_batch) == 2

    # 2. Add 2 more requests while s1 and s2 are running
    s3 = scheduler.add_request("List top 3 Linux terminal commands", tokenizer, max_new_tokens=12)
    s4 = scheduler.add_request("What is the speed of light in vacuum?", tokenizer, max_new_tokens=8)

    # Step 2: s3 should be admitted into slot 3 (max_batch_size = 3)
    scheduler.step(model, tokenizer)
    assert len(scheduler.running_batch) == 3
    assert s4.state == SequenceState.WAITING

    # Step 3..N: Run until all finished
    while scheduler.has_pending_work():
        scheduler.step(model, tokenizer)

    assert len(scheduler.finished_sequences) == 4
    for seq in [s1, s2, s3, s4]:
        assert seq.state == SequenceState.FINISHED
        assert len(seq.generated_tokens) > 0
        decoded = tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
        assert len(decoded.strip()) > 0
