"""Explicit model loading; no remote code execution or model training."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM


def load_engine(model_id, *, revision=None, device="cuda", **kwargs):
    from .engine import DecisionEngine

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a CPU experiment")
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    loader = Qwen3_5ForCausalLM if config.model_type == "qwen3_5" else AutoModelForCausalLM
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, loading = loader.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete model checkpoint load: {loading}")
    engine = DecisionEngine(model, tokenizer, **kwargs)
    engine.model_id = model_id
    engine.revision = getattr(config, "_commit_hash", None) or revision
    return engine
