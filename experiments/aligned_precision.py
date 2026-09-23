"""FP32 follow-up for observed serial/cache disagreements, not prompt tuning."""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from experiments.aligned_fewshot import CONDITIONS, compare_execution, decide
from experiments.answer_ablation import REVISION
from experiments.fewshot_stability import read_cases
from openjev.engine import DecisionEngine


def main():
    source = Path("reports/aligned-holdout.json")
    output = Path("reports/aligned-precision.json")
    if output.exists():
        raise ValueError("Refusing to overwrite reference results")
    report = json.loads(source.read_text())
    selected = set()
    for condition in CONDITIONS:
        a = {
            r["id"]: r
            for r in report["results"][f"parallel_{condition}"]
            if r["layout"] == "s0_forward" and r["repeat"] == 0
        }
        b = {r["id"]: r for r in report["results"][f"serial_{condition}"]}
        delta = {}
        for case_id, cached in a.items():
            delta[case_id] = max(
                abs(x - y)
                for f in cached["probabilities"]
                for x, y in zip(cached["probabilities"][f], b[case_id]["probabilities"][f], strict=True)
            )
            if cached["values"] != b[case_id]["values"]:
                selected.add((condition, case_id))
        selected.add((condition, max(delta, key=delta.get)))
    cases = {c["id"]: c for c in read_cases(Path("data/holdout-v3.json"))}
    examples = read_cases(Path("data/fewshot-examples-v1.json"))
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_id = "Qwen/Qwen3.5-2B"
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_id, revision=REVISION, dtype=torch.float32, device_map="cuda", attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    engine = DecisionEngine(model, tokenizer)
    reference = {
        "dtype": "float32",
        "revision": REVISION,
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "All case/condition pairs with BF16 choice disagreements, plus greatest probability discrepancy per condition; no accuracy/prompt selection",
        "attention_backend": "EFFICIENT_ATTENTION with explicitly repeated KV heads",
        "checks": [],
    }
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        stack.enter_context(patch("transformers.integrations.sdpa_attention.use_gqa_in_sdpa", return_value=False))
        stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
        for condition, case_id in sorted(selected):
            case = cases[case_id]
            a = decide(engine, case, examples, condition)
            b = decide(engine, case, examples, condition, parallel=False)
            check = compare_execution([case], [a], [b])
            check.update(id=case_id, condition=condition, parallel=a, serial=b)
            reference["checks"].append(check)
            reference["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            output.with_suffix(".partial.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
            print(
                condition, case_id, check["max_abs_probability_difference"], check["choice_disagreements"], flush=True
            )
    output.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".partial.json").unlink()


if __name__ == "__main__":
    main()
