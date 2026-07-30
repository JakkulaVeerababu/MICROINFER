"""
MicroInfer - Sub-Phase 3.3: Scheduler Correctness & Lifecycle Unit Test Suite
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


def test_scheduler_lifecycle_and_fifo_order(loaded_model_and_tokenizer):
    """
    Verifies request admission order, FIFO queue processing, step execution, and eviction lifecycle.
    """
    model, tokenizer = loaded_model_and_tokenizer
    scheduler = ContinuousBatchScheduler(max_batch_size=2)

    req1 = scheduler.add_request("Explain KV caching", tokenizer, max_new_tokens=10)
    req2 = scheduler.add_request("Explain continuous batching", tokenizer, max_new_tokens=10)
    req3 = scheduler.add_request("Explain weight quantization", tokenizer, max_new_tokens=10)

    assert len(scheduler.waiting_queue) == 3
    assert len(scheduler.running_batch) == 0

    # Step 1: Should admit req1 and req2 (max_batch_size = 2) in FIFO order
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
    
    # Assert finished order preserves seq_id
    assert [seq.seq_id for seq in scheduler.finished_sequences] == [1, 2, 3]
    for seq in scheduler.finished_sequences:
        assert seq.state == SequenceState.FINISHED
        assert len(seq.generated_tokens) > 0
        assert seq.ttft_ms > 0.0


def test_batch_size_one_scheduling(loaded_model_and_tokenizer):
    """
    Verifies scheduler behavior when max_batch_size=1 (sequential processing).
    """
    model, tokenizer = loaded_model_and_tokenizer
    scheduler = ContinuousBatchScheduler(max_batch_size=1)

    r1 = scheduler.add_request("Prompt A", tokenizer, max_new_tokens=5)
    r2 = scheduler.add_request("Prompt B", tokenizer, max_new_tokens=5)

    scheduler.step(model, tokenizer)
    assert len(scheduler.running_batch) == 1
    assert scheduler.running_batch[0].seq_id == r1.seq_id

    while scheduler.has_pending_work():
        scheduler.step(model, tokenizer)

    assert len(scheduler.finished_sequences) == 2
    assert scheduler.finished_sequences[0].seq_id == r1.seq_id
    assert scheduler.finished_sequences[1].seq_id == r2.seq_id
