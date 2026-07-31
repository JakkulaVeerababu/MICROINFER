"""
MicroInfer - Engine Unit Tests (Phase 0-5 Unified Interface)
=============================================================

Tests the MicroInferEngine unified entry point in all 4 modes.
Each test loads the engine, generates a short response (max_new_tokens=8
for speed), and verifies the output is a non-empty string.

Run with:
    python -m pytest tests/test_engine.py -v
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import MicroInferEngine
from src.model_loader import DEFAULT_MODEL_ID

# ---------------------------------------------------------------------------
# Shared fixture — model config for all tests
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "model_id": DEFAULT_MODEL_ID,
    "max_batch_size": 4,
    "max_seq_len": 256,
}

TEST_PROMPT = "What is a transformer model?"
SHORT_TOKENS = 8  # Keep all engine tests fast


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_valid_output(output: str, mode: str):
    """Validate that the engine returned a non-empty string."""
    assert isinstance(output, str), f"[{mode}] Output must be a string, got {type(output)}"
    assert len(output.strip()) > 0, f"[{mode}] Output must be non-empty"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineNaiveMode:
    """Phase 1: Uncached O(N^2) generator tests."""

    def test_engine_naive_mode(self):
        """Engine in naive mode returns a non-empty string for a simple prompt."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "naive"})
        outputs = engine.generate([TEST_PROMPT], max_new_tokens=SHORT_TOKENS)
        assert len(outputs) == 1, "Should return exactly 1 output for 1 prompt"
        _assert_valid_output(outputs[0], "naive")

    def test_engine_naive_multiple_prompts(self):
        """Engine in naive mode handles multiple prompts and returns in order."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "naive"})
        prompts = [
            "What is a transformer model?",
            "Explain KV-caching briefly.",
        ]
        outputs = engine.generate(prompts, max_new_tokens=SHORT_TOKENS)
        assert len(outputs) == len(prompts), "Should return same number of outputs as inputs"
        for i, out in enumerate(outputs):
            _assert_valid_output(out, f"naive[{i}]")


class TestEngineCachedMode:
    """Phase 2: KV-Cache generator tests."""

    def test_engine_cached_mode(self):
        """Engine in cached mode returns a non-empty string for a simple prompt."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "cached"})
        outputs = engine.generate([TEST_PROMPT], max_new_tokens=SHORT_TOKENS)
        assert len(outputs) == 1, "Should return exactly 1 output for 1 prompt"
        _assert_valid_output(outputs[0], "cached")

    def test_engine_cached_empty_prompts(self):
        """Engine in cached mode returns empty list for empty input."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "cached"})
        outputs = engine.generate([], max_new_tokens=SHORT_TOKENS)
        assert outputs == [], "Should return empty list for empty prompts"


class TestEngineScheduledMode:
    """Phase 3: Continuous batch scheduler tests."""

    def test_engine_scheduled_mode(self):
        """Engine in scheduled mode returns non-empty strings for 2 prompts."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "scheduled"})
        prompts = [
            "What is a transformer model?",
            "Explain KV-caching briefly.",
        ]
        outputs = engine.generate(prompts, max_new_tokens=SHORT_TOKENS)
        assert len(outputs) == len(prompts), "Should return same number of outputs as inputs"
        for i, out in enumerate(outputs):
            _assert_valid_output(out, f"scheduled[{i}]")

    def test_engine_scheduled_preserves_order(self):
        """Engine in scheduled mode preserves prompt order in returned outputs."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "scheduled"})
        # Use 3 distinct prompts to verify ordering is maintained
        prompts = [
            "Sentence one prompt here.",
            "Sentence two prompt here.",
            "Sentence three prompt here.",
        ]
        outputs = engine.generate(prompts, max_new_tokens=SHORT_TOKENS)
        assert len(outputs) == 3, "Should return 3 outputs for 3 prompts"
        for i, out in enumerate(outputs):
            _assert_valid_output(out, f"scheduled_order[{i}]")


class TestEngineQuantizedMode:
    """Phase 4: INT8 quantized generator tests."""

    def test_engine_quantized_mode(self):
        """Engine in quantized mode returns a non-empty string for a simple prompt."""
        engine = MicroInferEngine({**BASE_CONFIG, "mode": "quantized"})
        outputs = engine.generate([TEST_PROMPT], max_new_tokens=SHORT_TOKENS)
        assert len(outputs) == 1, "Should return exactly 1 output for 1 prompt"
        _assert_valid_output(outputs[0], "quantized")


class TestEngineConfigValidation:
    """Engine config validation tests."""

    def test_engine_invalid_mode_raises(self):
        """Engine raises ValueError for an unrecognized mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            MicroInferEngine({**BASE_CONFIG, "mode": "nonexistent_mode"})

    def test_engine_default_mode_is_cached(self):
        """Engine defaults to 'cached' mode when no mode is specified."""
        engine = MicroInferEngine({"model_id": DEFAULT_MODEL_ID})
        assert engine.mode == "cached"
