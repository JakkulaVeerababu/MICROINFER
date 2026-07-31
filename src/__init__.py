"""
MicroInfer — High-Performance LLM Inference Engine & Production Benchmarking Suite
"""

__version__ = "0.1.0"
__author__ = "Jakkula Veerababu"

# Lazy imports — do not eagerly pull in heavy model/cache modules at package
# load time. This keeps `import src` safe during test collection even when
# optional dependencies (e.g. specific transformers versions) are absent.
# Import individual submodules explicitly where needed:
#   from src.kv_cache import KVCache
#   from src.scheduler import ContinuousBatchScheduler

__all__ = [
    "KVCache",
    "ContinuousBatchScheduler",
    "Sequence",
    "SequenceState",
    "load_model_and_tokenizer",
    "cached_generate",
    "naive_generate",
]
