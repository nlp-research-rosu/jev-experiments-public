"""Reference the largest numerical discrepancy and any changed choices in FP32."""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from experiments.aligned_fewshot import compare_execution
from experiments.answer_ablation import REVISION
from experiments.compact_fewshot import decide, sha
from experiments.fewshot_stability import read_cases
from openjev.engine import DecisionEngine


def main():
    source, output = Path("reports/compact-holdout.json"), Path("reports/compact-precision.json")
    if output.exists():
        raise ValueError("Refusing to overwrite reference results")
    report = json.loads(source.read_text())
    selected = report["metadata"]["selected"]
    cached = {r["id"]: r for r in report["results"][selected] if r["repeat"] == 0 and r["layout"] == "s0_forward"}
    serial = {r["id"]: r for r in report["results"]["serial_selected"]}
    delta = {
        i: max(
            abs(x - y)
            for f in a["probabilities"]
            for x, y in zip(a["probabilities"][f], serial[i]["probabilities"][f], strict=True)
        )
        for i, a in cached.items()
    }
    ids = {i for i in cached if cached[i]["values"] != serial[i]["values"]}
    ids.add(max(delta, key=delta.get))
    cases = {c["id"]: c for c in read_cases(Path("data/holdout-v4.json"))}
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
    result = {
        "dtype": "float32",
        "revision": REVISION,
        "condition": selected,
        "source_report_sha256": sha(source),
        "selection": "All identity-layout answer disagreements plus the case with largest probability difference; targeted numerical diagnostic, not accuracy selection.",
        "attention_backend": "EFFICIENT_ATTENTION with explicitly repeated KV heads",
        "checks": [],
    }
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        stack.enter_context(patch("transformers.integrations.sdpa_attention.use_gqa_in_sdpa", return_value=False))
        stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
        for case_id in sorted(ids):
            case = cases[case_id]
            a = decide(engine, case, examples, selected)
            b = decide(engine, case, examples, selected, parallel=False)
            check = compare_execution([case], [a], [b])
            check.update(id=case_id, parallel=a, serial=b, bfloat16_max_difference=delta[case_id])
            result["checks"].append(check)
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            output.with_suffix(".partial.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(case_id, check["max_abs_probability_difference"], check["choice_disagreements"], flush=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".partial.json").unlink()


if __name__ == "__main__":
    main()
