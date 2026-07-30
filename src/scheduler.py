"""
MicroInfer - Phase 3: Dynamic Request Scheduler with Lifecycle Management Module

Implements iteration-granularity request lifecycle management, dynamic queue
scheduling (WAITING -> RUNNING -> FINISHED), and multi-sequence stepping.

Architecture Note:
Active sequences are stepped in an iteration loop per step to maintain per-sequence
KV-cache independence. True tensor-level batching across concurrent sequences
requires restructuring the forward pass to stack (B, T) inputs — this is a
documented next step.
"""

import time
import torch
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from transformers.cache_utils import DynamicCache


class SequenceState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    """
    Represents an individual generation request lifecycle state.
    """
    seq_id: int
    prompt: str
    prompt_tokens: List[int]
    max_new_tokens: int = 128
    temperature: float = 0.0
    generated_tokens: List[int] = field(default_factory=list)
    state: SequenceState = SequenceState.WAITING
    cache_slot: Optional[Any] = None
    ttft_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def is_finished(self) -> bool:
        return self.state == SequenceState.FINISHED


class ContinuousBatchScheduler:
    """
    In-flight request scheduler managing concurrent requests across WAITING, RUNNING, and FINISHED states.
    """

    def __init__(
        self,
        max_batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.max_batch_size = max_batch_size
        self.device = device
        self.waiting_queue: List[Sequence] = []
        self.running_batch: List[Sequence] = []
        self.finished_sequences: List[Sequence] = []
        self._next_seq_id = 1

    def add_request(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Sequence:
        """
        Submits a new generation request to the waiting queue.
        """
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            state=SequenceState.WAITING,
            cache_slot=DynamicCache(),
        )
        self._next_seq_id += 1
        self.waiting_queue.append(seq)
        return seq

    def step(self, model: torch.nn.Module, tokenizer: Any) -> List[Sequence]:
        """
        Executes 1 continuous batching step:
        1. Admits waiting requests into running_batch if capacity exists.
        2. Executes forward pass across running sequences.
        3. Evicts completed sequences and frees resources.
        """
        # 1. Admission phase: Admit waiting requests if capacity is available
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            seq = self.waiting_queue.pop(0)
            seq.state = SequenceState.RUNNING
            self.running_batch.append(seq)

        if not self.running_batch:
            return []

        finished_this_step: List[Sequence] = []

        # 2. Execution phase: Process each running sequence in current batch
        for seq in self.running_batch:
            t_step_start = time.perf_counter()

            # Determine whether this is Prefill (1st step) or Decode (subsequent steps)
            if len(seq.generated_tokens) == 0:
                input_tensor = torch.tensor([seq.prompt_tokens], device=self.device)
            else:
                input_tensor = torch.tensor([[seq.generated_tokens[-1]]], device=self.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_tensor,
                    past_key_values=seq.cache_slot,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]

                if seq.temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True).item()
                else:
                    probs = torch.softmax(logits / seq.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_step_end = time.perf_counter()
            step_ms = (t_step_end - t_step_start) * 1000.0

            if len(seq.generated_tokens) == 0:
                seq.ttft_ms = step_ms

            seq.generated_tokens.append(next_token)
            seq.total_time_ms += step_ms

            # Check completion criteria (EOS token or max_new_tokens reached)
            if next_token == tokenizer.eos_token_id or len(seq.generated_tokens) >= seq.max_new_tokens:
                seq.state = SequenceState.FINISHED
                finished_this_step.append(seq)

        # 3. Eviction phase: Remove finished sequences from running batch
        for seq in finished_this_step:
            self.running_batch.remove(seq)
            self.finished_sequences.append(seq)

        return finished_this_step

    def has_pending_work(self) -> bool:
        """
        Returns True if there are requests in either waiting_queue or running_batch.
        """
        return len(self.waiting_queue) > 0 or len(self.running_batch) > 0


if __name__ == "__main__":
    from src.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    scheduler = ContinuousBatchScheduler(max_batch_size=2)
    
    seq1 = scheduler.add_request("Continuous batching in LLMs is", tokenizer, max_new_tokens=15)
    seq2 = scheduler.add_request("Transformer KV cache optimization", tokenizer, max_new_tokens=15)

    print(f"[MicroInfer] Scheduler initialized. Waiting requests: {len(scheduler.waiting_queue)}")
    while scheduler.has_pending_work():
        finished = scheduler.step(model, tokenizer)
        for seq in finished:
            print(f"[Finished Seq {seq.seq_id}]: '{tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)}'")
