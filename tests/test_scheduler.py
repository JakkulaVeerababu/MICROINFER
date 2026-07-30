"""
MicroInfer - Unit Tests for Sub-Phase 3.1 / 3.3 Continuous Batching Scheduler Module
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


def test_scheduler_lifecycle(loaded_model_and_tokenizer):
    """
    Verifies request admission, step execution, and eviction lifecycle.
    """
    model, tokenizer = loaded_model_and_tokenizer
    scheduler = ContinuousBatchScheduler(max_batch_size=2)

    req1 = scheduler.add_request("Prompt 1", tokenizer, max_new_tokens=10)
    req2 = scheduler.add_request("Prompt 2", tokenizer, max_new_tokens=10)
    req3 = scheduler.add_request("Prompt 3", tokenizer, max_new_tokens=10)

    assert len(scheduler.waiting_queue) == 3
    assert len(scheduler.running_batch) == 0

    # Step 1: Should admit req1 and req2 (max_batch_size = 2)
    scheduler.step(model, tokenizer)
    assert len(scheduler.running_batch) == 2
    assert len(scheduler.waiting_queue) == 1
    assert req1.state == SequenceState.RUNNING
    assert req2.state == SequenceState.RUNNING
    assert req3.state == SequenceState.WAITING

    # Run remaining steps until all finish
    step_count = 1
    while scheduler.has_pending_work() and step_count < 50:
        scheduler.step(model, tokenizer)
        step_count += 1

    assert len(scheduler.finished_sequences) == 3
    assert not scheduler.has_pending_work()
    for seq in scheduler.finished_sequences:
        assert seq.state == SequenceState.FINISHED
        assert len(seq.generated_tokens) > 0
        assert seq.ttft_ms > 0.0
