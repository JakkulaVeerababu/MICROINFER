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
from src.kv_cache import KVCache


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
        model: Optional[torch.nn.Module] = None,
    ) -> Sequence:
        """
        Submits a new generation request to the waiting queue.
        """
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        cache_slot = (
            KVCache.from_model(model, max_seq_len=len(prompt_tokens) + max_new_tokens + 64)
            if model is not None
            else None
        )
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            state=SequenceState.WAITING,
            cache_slot=cache_slot,
        )
        self._next_seq_id += 1
        self.waiting_queue.append(seq)
        return seq

    def step(self, model: torch.nn.Module, tokenizer: Any) -> List[Sequence]:
        """
        Executes 1 continuous batching step:
        1. Admits waiting requests into running_batch if capacity exists.
        2. Executes prefill for new sequences or TRUE BATCHED decode for active running sequences.
        3. Evicts completed sequences and frees resources.
        """
        # 1. Admission phase: Admit waiting requests if capacity is available
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            seq = self.waiting_queue.pop(0)
            seq.state = SequenceState.RUNNING
            if seq.cache_slot is None:
                max_len = len(seq.prompt_tokens) + seq.max_new_tokens + 64
                seq.cache_slot = KVCache.from_model(model, max_seq_len=max_len)
            self.running_batch.append(seq)

        if not self.running_batch:
            return []

        finished_this_step: List[Sequence] = []

        # Separate prefill (un-started) vs decode (in-progress) sequences
        prefill_seqs = [seq for seq in self.running_batch if len(seq.generated_tokens) == 0]
        decode_seqs = [seq for seq in self.running_batch if len(seq.generated_tokens) > 0]

        # 2a. Prefill step for any newly admitted sequence
        for seq in prefill_seqs:
            t0 = time.perf_counter()
            input_tensor = torch.tensor([seq.prompt_tokens], device=self.device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_tensor,
                    past_key_values=seq.cache_slot,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                if seq.temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    probs = torch.softmax(logits / seq.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            step_ms = (time.perf_counter() - t0) * 1000.0
            seq.ttft_ms = step_ms
            seq.generated_tokens.append(next_token)
            seq.total_time_ms += step_ms

            if next_token == tokenizer.eos_token_id or len(seq.generated_tokens) >= seq.max_new_tokens:
                seq.state = SequenceState.FINISHED
                finished_this_step.append(seq)

        # 2b. BATCHED Decode step for all active running sequences
        if decode_seqs:
            t0 = time.perf_counter()
            B = len(decode_seqs)

            if B == 1:
                # Single sequence decode
                seq = decode_seqs[0]
                input_tensor = torch.tensor([[seq.generated_tokens[-1]]], device=self.device)
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_tensor,
                        past_key_values=seq.cache_slot,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    if seq.temperature == 0.0:
                        next_token = torch.argmax(logits, dim=-1).item()
                    else:
                        probs = torch.softmax(logits / seq.temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                step_ms = (time.perf_counter() - t0) * 1000.0
                seq.generated_tokens.append(next_token)
                seq.total_time_ms += step_ms

                if next_token == tokenizer.eos_token_id or len(seq.generated_tokens) >= seq.max_new_tokens:
                    seq.state = SequenceState.FINISHED
                    finished_this_step.append(seq)
            else:
                # Multi-sequence batched decode forward pass in a single GPU call
                S_lens = [seq.cache_slot.current_len for seq in decode_seqs]
                S_max = max(S_lens)
                num_layers = decode_seqs[0].cache_slot.num_layers
                num_kv_heads = decode_seqs[0].cache_slot.num_kv_heads
                head_dim = decode_seqs[0].cache_slot.head_dim

                batched_pkv = []
                for l in range(num_layers):
                    bk = torch.zeros((B, num_kv_heads, S_max, head_dim), device=self.device, dtype=torch.float16)
                    bv = torch.zeros((B, num_kv_heads, S_max, head_dim), device=self.device, dtype=torch.float16)
                    for i, seq in enumerate(decode_seqs):
                        slen = S_lens[i]
                        bk[i, :, :slen, :] = seq.cache_slot.layers[l].k_slice[0, :, :slen, :]
                        bv[i, :, :slen, :] = seq.cache_slot.layers[l].v_slice[0, :, :slen, :]
                    batched_pkv.append((bk, bv))

                batch_input_ids = torch.tensor([[seq.generated_tokens[-1]] for seq in decode_seqs], device=self.device)
                batch_pos_ids = torch.tensor([[slen] for slen in S_lens], device=self.device)

                batch_mask = torch.zeros((B, S_max + 1), device=self.device, dtype=torch.long)
                for i, slen in enumerate(S_lens):
                    batch_mask[i, :slen] = 1
                    batch_mask[i, S_max] = 1

                cache_obj = DynamicCache.from_legacy_cache(tuple(batched_pkv))
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_mask,
                        position_ids=batch_pos_ids,
                        past_key_values=cache_obj,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                step_ms = (time.perf_counter() - t0) * 1000.0

                for i, seq in enumerate(decode_seqs):
                    slen = S_lens[i]
                    if seq.temperature == 0.0:
                        next_token = torch.argmax(logits[i], dim=-1).item()
                    else:
                        probs = torch.softmax(logits[i] / seq.temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()

                    # Update individual KV cache slot for position slen
                    for l in range(num_layers):
                        new_k = outputs.past_key_values[l][0][i:i+1, :, S_max:S_max+1, :]
                        new_v = outputs.past_key_values[l][1][i:i+1, :, S_max:S_max+1, :]
                        seq.cache_slot.layers[l].update(new_k, new_v)

                    seq.generated_tokens.append(next_token)
                    seq.total_time_ms += step_ms

                    if next_token == tokenizer.eos_token_id or len(seq.generated_tokens) >= seq.max_new_tokens:
                        seq.state = SequenceState.FINISHED
                        finished_this_step.append(seq)

        # 3. Eviction phase: Remove finished sequences from running batch
        for seq in finished_this_step:
            if seq in self.running_batch:
                self.running_batch.remove(seq)
            if seq not in self.finished_sequences:
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
