"""Differentiable scalar judgments and cached inference for the v0.2 contract."""

import hashlib
import inspect
import json
import shutil
import time
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .engine import fork_cache

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_KERNELS = None


def configure_kernels(backend):
    """Select one process-wide Qwen recurrent backend before running a model."""
    global _KERNELS
    if backend not in ("reference", "fla"):
        raise ValueError("kernel backend must be reference or fla")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    names = (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    if _KERNELS is None:
        _KERNELS = {name: getattr(module, name) for name in names}
    if backend == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        module.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
        module.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        for name in names[2:]:
            setattr(module, name, _KERNELS[name])
    else:
        for name, function in _KERNELS.items():
            setattr(module, name, inspect.unwrap(function))


def frozen_digest(model):
    """Hash every frozen parameter, without retaining a full CPU weight copy."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            value = parameter.detach().cpu().contiguous().view(torch.uint8)
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_identity(path, info):
    digest = hashlib.sha256(
        json.dumps({k: info[k] for k in ("format", "model_id", "revision", "lora")}, sort_keys=True).encode()
    )
    for file in sorted(Path(path).rglob("*.safetensors")):
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return "openjev-judgment-v0.2/sha256-" + digest.hexdigest()


class JudgmentModel(nn.Module):
    def __init__(self, causal_lm, *, model_id=MODEL_ID, revision=REVISION, kernel_backend="reference"):
        super().__init__()
        configure_kernels(kernel_backend)
        self.model_id, self.revision, self.kernel_backend = model_id, revision, kernel_backend
        projection = causal_lm.get_output_embeddings().weight
        initial = projection[32].detach().float() - projection[33].detach().float()
        self.backbone = causal_lm.model
        self.backbone.requires_grad_(False)
        self.compatibility = nn.Linear(initial.numel(), 1, bias=False, device=initial.device, dtype=torch.float32)
        self.binary = nn.Linear(initial.numel(), 1, bias=True, device=initial.device, dtype=torch.float32)
        with torch.no_grad():
            self.compatibility.weight.copy_(initial[None])
            self.binary.weight.copy_(initial[None])
            self.binary.bias.zero_()
        self.lora_settings = None
        self.checkpoint_id = None
        self.eval()

    @property
    def device(self):
        return self.compatibility.weight.device

    def _read(self, hidden, kinds):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden.float()
            return torch.where(kinds.bool(), self.binary(h).squeeze(-1), self.compatibility(h).squeeze(-1))

    def forward(self, input_ids, attention_mask, score_positions, readout_kind):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        endpoints = hidden[torch.arange(len(input_ids), device=input_ids.device), score_positions]
        return self._read(endpoints, readout_kind)

    @staticmethod
    def _validate(prompts, kinds, batch_size):
        if not prompts or any(not p for p in prompts) or len(prompts) != len(kinds) or batch_size < 1:
            raise ValueError("nonempty prompts, matching kinds and positive batch size required")
        if any(type(k) is not int or k not in (0, 1) for k in kinds):
            raise ValueError("readout kinds must be integers 0 or 1")

    def _batch(self, prompts):
        length = max(map(len, prompts))
        pad = getattr(self.backbone.config, "pad_token_id", None) or 0
        inputs = torch.tensor([list(p) + [pad] * (length - len(p)) for p in prompts], device=self.device)
        lengths = torch.tensor(list(map(len, prompts)), device=self.device)
        mask = torch.arange(length, device=self.device)[None] < lengths[:, None]
        return inputs, mask, lengths

    def score_prompts(self, prompts, kinds, *, unit_batch_size=4):
        """Differentiable full-input path; complete candidate groups stay in caller."""
        self._validate(prompts, kinds, unit_batch_size)
        results = []
        for start in range(0, len(prompts), unit_batch_size):
            inputs, mask, lengths = self._batch(prompts[start : start + unit_batch_size])
            readout = torch.tensor(kinds[start : start + unit_batch_size], device=self.device)
            results.append(self(inputs, mask, lengths - 1, readout))
        return torch.cat(results)

    @torch.inference_mode()
    def score_cached(self, prompts, kinds, *, unit_batch_size=4):
        self._validate(prompts, kinds, unit_batch_size)
        common = 0
        for tokens in zip(*prompts):
            if len(set(tokens)) != 1:
                break
            common += 1
        common = min(common, min(map(len, prompts)) - 1)
        cache, calls = None, 0
        if common:
            out = self.backbone(torch.tensor([prompts[0][:common]], device=self.device), use_cache=True)
            cache, calls = out.past_key_values, 1
            del out
        batches = []
        sizes = [len(p) - common for p in prompts]
        for start in range(0, len(prompts), unit_batch_size):
            indices = list(range(start, min(start + unit_batch_size, len(prompts))))
            while len(indices) * max(sizes[i] for i in indices) > 2 * sum(sizes[i] for i in indices):
                longest = max(indices, key=sizes.__getitem__)
                batches.append([longest])
                indices.remove(longest)
            batches.append(indices)
        results = [None] * len(prompts)
        for indices in batches:
            inputs, _, lengths = self._batch([prompts[i][common:] for i in indices])
            mask = torch.arange(common + inputs.shape[1], device=self.device)[None] < (common + lengths[:, None])
            branch = fork_cache(cache, len(indices)) if cache is not None else None
            out = self.backbone(inputs, attention_mask=mask, past_key_values=branch, use_cache=True)
            h = out.last_hidden_state[torch.arange(len(indices), device=self.device), lengths - 1]
            scores = self._read(h, torch.tensor([kinds[i] for i in indices], device=self.device))
            for row, index in enumerate(indices):
                results[index] = scores[row].clone()
            calls += 1
            del out, branch
        return torch.stack(results), {"forward_calls": calls, "prefix_tokens": common, "branch_groups": len(batches)}

    def add_lora(self, *, rank=8, alpha=16, gradient_checkpointing=True):
        from peft import LoraConfig, get_peft_model

        if self.lora_settings is not None:
            raise ValueError("LoRA already attached")
        targets = sorted(
            {
                n.rsplit(".", 1)[-1]
                for n, m in self.backbone.named_modules()
                if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in LORA_TARGETS
            }
        )
        if not targets:
            raise ValueError("no eligible adapter projections")
        self.backbone = get_peft_model(
            self.backbone, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=targets)
        )
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lora_settings = {
            "rank": rank,
            "alpha": alpha,
            "targets": targets,
            "gradient_checkpointing": gradient_checkpointing,
        }
        return self.parameter_inventory()

    def parameter_inventory(self):
        named = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        invalid = [n for n in named if not (n.startswith(("compatibility.", "binary.")) or ".lora_" in n)]
        if invalid:
            raise ValueError(f"unexpected trainable base parameters: {invalid}")
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(named.values()),
            "adapter_parameters": sum(v for n, v in named.items() if ".lora_" in n),
            "readout_parameters": sum(v for n, v in named.items() if n.startswith(("compatibility.", "binary."))),
            "trainable_names": named,
            "lora": self.lora_settings,
        }

    def save_checkpoint(self, path, *, metadata=None, optimizer=None, training_state=None):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / ("." + path.name + ".pending-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            heads = {
                "compatibility.weight": self.compatibility.weight.detach().cpu().contiguous(),
                "binary.weight": self.binary.weight.detach().cpu().contiguous(),
                "binary.bias": self.binary.bias.detach().cpu().contiguous(),
            }
            save_file(heads, str(temporary / "readouts.safetensors"))
            if self.lora_settings:
                self.backbone.save_pretrained(temporary / "adapter", safe_serialization=True)
            info = {
                "format": "openjev-judgment-v0.2",
                "model_id": self.model_id,
                "revision": self.revision,
                "kernel_backend": self.kernel_backend,
                "lora": self.lora_settings,
                "metadata": metadata or {},
            }
            info["checkpoint_id"] = checkpoint_identity(temporary, info)
            (temporary / "checkpoint.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
            if optimizer is not None:
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "state": training_state or {},
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    },
                    temporary / "training.pt",
                )
            temporary.rename(path)
            return info["checkpoint_id"]
        except BaseException:
            shutil.rmtree(temporary)
            raise

    @classmethod
    def load_checkpoint(cls, path, *, base_factory=None, device="cuda", trainable=False, kernel_backend=None):
        path = Path(path)
        info = json.loads((path / "checkpoint.json").read_text())
        if info.get("format") != "openjev-judgment-v0.2":
            raise ValueError("unsupported checkpoint format")
        identity = checkpoint_identity(path, info)
        if info.get("checkpoint_id") != identity:
            raise ValueError("checkpoint identity does not match saved weights")
        if base_factory is None:
            from .runtime import load_engine

            causal = load_engine(info["model_id"], revision=info["revision"], device=device).model
        else:
            causal = base_factory()
        model = cls(
            causal,
            model_id=info["model_id"],
            revision=info["revision"],
            kernel_backend=kernel_backend or info["kernel_backend"],
        )
        if info["lora"]:
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(model.backbone, path / "adapter", is_trainable=trainable)
            model.lora_settings = info["lora"]
            if trainable and info["lora"]["gradient_checkpointing"]:
                model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        heads = load_file(str(path / "readouts.safetensors"), device=str(model.device))
        with torch.no_grad():
            model.compatibility.weight.copy_(heads["compatibility.weight"])
            model.binary.weight.copy_(heads["binary.weight"])
            model.binary.bias.copy_(heads["binary.bias"])
        model.train(trainable)
        model.checkpoint_id = identity
        return model, info["metadata"]


class JudgmentEngine:
    def __init__(self, model, tokenizer, *, model_id="openjev-judgment-v0.2", max_input_tokens=8192, unit_batch_size=4):
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id
        self.max_input_tokens, self.unit_batch_size = max_input_tokens, unit_batch_size
        if {c: tokenizer.encode(c, add_special_tokens=False) for c in ("A", "B")} != {"A": [32], "B": [33]}:
            raise ValueError("tokenizer does not match pinned readout anchors")

    def prepare(self, request):
        from .judgments import compile_request, render_unit_messages

        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        requested_model = request.get("model", "openjev-judgment-v0.2")
        if requested_model not in ("openjev-judgment-v0.2", self.model_id):
            raise ValueError("requested model does not match the loaded checkpoint")
        compiled = compile_request({**request, "model": self.model_id})
        prompts, kinds = [], []
        for unit in compiled.units:
            messages = render_unit_messages(unit)
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
            )
            if len(ids) > self.max_input_tokens:
                raise ValueError(f"rendered unit exceeds {self.max_input_tokens} tokens; no truncation")
            prompts.append(ids)
            kinds.append(unit.readout_kind)
        return compiled, prompts, kinds

    def synchronize(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)

    @torch.inference_mode()
    def evaluate(self, request, *, details=False, cached=True):
        from .judgments import assemble_response

        self.model.eval()
        self.synchronize()
        started = time.perf_counter()
        compiled, prompts, kinds = self.prepare(request)
        if cached:
            logits, stats = self.model.score_cached(prompts, kinds, unit_batch_size=self.unit_batch_size)
        else:
            logits = self.model.score_prompts(prompts, kinds, unit_batch_size=self.unit_batch_size)
            stats = {
                "forward_calls": (len(prompts) + self.unit_batch_size - 1) // self.unit_batch_size,
                "prefix_tokens": 0,
            }
        if not torch.isfinite(logits).all():
            raise ValueError("model returned non-finite logits")
        result = assemble_response(compiled, logits.cpu().tolist(), details=details)
        self.synchronize()
        result["usage"] = {
            "input_tokens": sum(map(len, prompts)),
            "max_unit_tokens": max(map(len, prompts)),
            "model_units": len(prompts),
            "output_tokens": 0,
            **stats,
        }
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result
