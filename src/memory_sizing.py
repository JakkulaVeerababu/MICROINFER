"""
MicroInfer - Sub-Phase 0.2: Model Selection & VRAM Memory Sizing Profiler
Analyzes model architecture parameters and computes exact VRAM allocation limits for 6GB RTX 4050 GPU.
"""

import sys
import json
import torch
from pathlib import Path
from transformers import AutoConfig


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def profile_memory_sizing(model_id: str = DEFAULT_MODEL_ID):
    """
    Computes architectural memory requirements for model weights and KV-cache tensors.
    """
    print(f"[MicroInfer] Fetching configuration for '{model_id}'...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Extract architectural hyperparameters
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 28))
    hidden_size = getattr(config, "hidden_size", 1536)
    num_attention_heads = getattr(config, "num_attention_heads", 12)
    num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    vocab_size = getattr(config, "vocab_size", 151936)
    max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

    # Total parameters estimate (~1.54B)
    num_params = getattr(config, "num_parameters", 1540000000)

    # Precision calculations (FP16 = 2 bytes/param)
    bytes_per_param_fp16 = 2
    bytes_per_param_int8 = 1

    weight_memory_fp16_gb = (num_params * bytes_per_param_fp16) / (1024 ** 3)
    weight_memory_int8_gb = (num_params * bytes_per_param_int8) / (1024 ** 3)

    # KV-Cache Math per token per sequence (2 = Key + Value)
    # Shape: (num_layers, 2, num_kv_heads, seq_len, head_dim)
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param_fp16
    kv_kb_per_token = kv_bytes_per_token / 1024.0

    # Memory at standard sequence lengths (256, 512, 1024, 2048)
    seq_lengths = [256, 512, 1024, 2048, 4096]
    kv_memory_table = {}
    for seq_len in seq_lengths:
        per_seq_mb = (kv_bytes_per_token * seq_len) / (1024 ** 2)
        kv_memory_table[f"seq_{seq_len}"] = {
            "per_sequence_mb": round(per_seq_mb, 2),
            "batch_8_mb": round(per_seq_mb * 8, 2),
            "batch_16_mb": round(per_seq_mb * 16, 2),
            "batch_32_mb": round(per_seq_mb * 32, 2),
        }

    # Total GPU budget analysis for RTX 4050 (6.0 GB VRAM)
    gpu_total_vram_gb = 6.0
    cuda_overhead_gb = 0.5  # CUDA context & PyTorch runtime buffer
    available_vram_gb = gpu_total_vram_gb - weight_memory_fp16_gb - cuda_overhead_gb

    # Max sequence capacity at Batch Size 1 and Batch Size 16
    max_tokens_b1 = int((available_vram_gb * (1024 ** 3)) / kv_bytes_per_token)
    max_tokens_b16 = int(((available_vram_gb * (1024 ** 3)) / kv_bytes_per_token) / 16)

    report = {
        "sub_phase": "0.2 - Model Selection & VRAM Memory Sizing",
        "model_id": model_id,
        "architectural_specs": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "attention_mechanism": "Grouped-Query Attention (GQA)" if num_kv_heads < num_attention_heads else "Multi-Head Attention (MHA)",
        },
        "weight_memory_gb": {
            "fp16": round(weight_memory_fp16_gb, 2),
            "int8": round(weight_memory_int8_gb, 2),
        },
        "kv_cache_specs": {
            "bytes_per_token": kv_bytes_per_token,
            "kb_per_token": round(kv_kb_per_token, 2),
            "sequence_length_matrix": kv_memory_table,
        },
        "rtx_4050_vram_budget": {
            "gpu_vram_gb": gpu_total_vram_gb,
            "fp16_weights_gb": round(weight_memory_fp16_gb, 2),
            "remaining_headroom_gb": round(available_vram_gb, 2),
            "max_capacity_tokens_batch_1": max_tokens_b1,
            "max_capacity_tokens_batch_16": max_tokens_b16,
        }
    }

    return report


def main():
    print("=" * 60)
    print("  MICROINFER SUB-PHASE 0.2: MODEL SELECTION & VRAM MEMORY SIZING")
    print("=" * 60)

    report = profile_memory_sizing()
    specs = report["architectural_specs"]
    weights = report["weight_memory_gb"]
    kv = report["kv_cache_specs"]
    budget = report["rtx_4050_vram_budget"]

    print(f"\n[Model Architecture Specs]")
    print(f"  Model ID:           {report['model_id']}")
    print(f"  Layers:             {specs['num_layers']}")
    print(f"  Hidden Size:        {specs['hidden_size']}")
    print(f"  Attention Type:     {specs['attention_mechanism']} ({specs['num_attention_heads']} Q heads, {specs['num_kv_heads']} KV heads)")
    print(f"  Head Dimension:     {specs['head_dim']}")

    print(f"\n[Memory Requirements]")
    print(f"  FP16 Model Weights: {weights['fp16']} GB")
    print(f"  INT8 Model Weights: {weights['int8']} GB")
    print(f"  KV-Cache Footprint: {kv['kb_per_token']} KB / token / sequence")

    print(f"\n[KV-Cache Scaling Matrix (FP16)]")
    print(f"  Seq Len 512:        {kv['sequence_length_matrix']['seq_512']['per_sequence_mb']} MB (Batch 16: {kv['sequence_length_matrix']['seq_512']['batch_16_mb']} MB)")
    print(f"  Seq Len 2048:       {kv['sequence_length_matrix']['seq_2048']['per_sequence_mb']} MB (Batch 16: {kv['sequence_length_matrix']['seq_2048']['batch_16_mb']} MB)")

    print(f"\n[RTX 4050 6GB VRAM Budget Allocation]")
    print(f"  Total GPU VRAM:     {budget['gpu_vram_gb']} GB")
    print(f"  Allocated Model:    {budget['fp16_weights_gb']} GB")
    print(f"  Available Headroom: {budget['remaining_headroom_gb']} GB")
    print(f"  Max Cache Tokens:   ~{budget['max_capacity_tokens_batch_1']:,} tokens (Batch 1)")

    # Export report to analysis/memory_sizing.json
    output_dir = Path(__file__).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / "memory_sizing.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[MicroInfer] Memory sizing profile saved to '{out_path}'.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
