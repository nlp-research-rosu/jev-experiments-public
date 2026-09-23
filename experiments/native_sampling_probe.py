"""One post-hoc sampled draw for two greedy-capped questions per primitive.

This six-case termination diagnostic is not an estimate of full-cohort ability.
The completed greedy archive and all its prompts remain immutable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from experiments.architecture_native import ROOT, cached_metadata, load_model
from experiments.calibrated_qwen_teacher import _write_once
from experiments.calibrated_targets import file_sha256
from experiments.native_reasoning_followup import (
    MAX_CONTEXT_TOKENS,
    Budget,
    _gold_label,
    digest,
    encode_json,
    parse_natural_answer,
    prepare_completed_readout,
    probability_metrics,
)

PARENT = ROOT / "reports/consistency-native-v1/native/completed-v1"
OUTPUT = ROOT / "reports/consistency-native-v1/native/sampled-six-v1"
RECIPE = {"temperature": 1., "top_p": .95, "top_k": 20, "min_p": 0.,
          "presence_penalty": 1.5, "repetition_penalty": 1.,
          "order": "FP32 raw logits; subtract presence once per generated token ID; temperature; top-k; top-p; multinomial",
          "presence_history": "generated thought and final tokens only; prompt excluded",
          "top_p_boundary": "keep first token that crosses .95 cumulative mass after top-k renormalization",
          "seed_rule": "42 + original greedy manifest index; fresh per-case generator on model device",
          "draws_per_case": 1, "source": "https://huggingface.co/Qwen/Qwen3.5-2B"}


def sampling_logits(raw_logits, generated_ids):
    """Apply the fixed sampling recipe without altering raw vocabulary logits."""
    import torch

    logits = raw_logits.detach().float().clone()
    if logits.ndim != 1 or not logits.numel() or not torch.isfinite(logits).all():
        raise ValueError("finite full-vocabulary logits required")
    seen = sorted(set(generated_ids))
    if seen:
        if seen[0] < 0 or seen[-1] >= logits.numel():
            raise ValueError("generated token ID outside vocabulary")
        logits[seen] -= 1.5
    # Temperature and repetition penalty are identity; min-p is disabled.
    values, indices = torch.topk(logits, min(20, logits.numel()), sorted=True)
    remove = values.softmax(-1).cumsum(-1) > .95
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    filtered = torch.full_like(logits, -torch.inf)
    filtered[indices[~remove]] = values[~remove]
    return filtered


def case_generator(device, seed):
    import torch

    return torch.Generator(device=device).manual_seed(seed)


def sample_token(raw_logits, generated_ids, generator):
    import torch

    probabilities = sampling_logits(raw_logits, generated_ids).softmax(-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def select_capped(manifest, reasons, gold):
    selected = []
    for primitive in ("noul", "choice", "score"):
        candidates = [index for index, row in enumerate(manifest["inputs"])
                      if row["primitive"] == primitive
                      and reasons[index]["generation"]["stop_reason"] == "scratchpad_cap"
                      and reasons[index]["generation"]["scratchpad_tokens"] == 4096
                      and not reasons[index]["generation"]["completed_thought"]]
        def key(index):
            row = manifest["inputs"][index]
            return hashlib.sha256(("sampled-native-v1:" + row["case_id"] + ":" + row["question_id"]).encode()).hexdigest()
        if len(candidates) < 2:
            raise ValueError("need at least two greedy scratchpad-capped questions per primitive")
        for index in sorted(candidates, key=key)[:2]:
            row = copy.deepcopy(manifest["inputs"][index])
            selected.append({**row, "greedy_index": index, "seed": 42 + index, "selection_hash": key(index),
                             "prepared": copy.deepcopy(manifest["prepared"][index]["reason"]),
                             "gold": copy.deepcopy(gold[str(index)])})
    return selected


def generate_sampled(scorer, input_ids, tokenizer, *, generator, deadline, scratchpad_tokens=4096, final_tokens=32,
                     clock=time.monotonic, on_token=None):
    """Copy of the checked greedy state machine with one sampled selection hook.

    The same sampler and generated-only presence history apply during both phases.
    """
    import torch

    Budget(scratchpad_tokens=scratchpad_tokens, final_tokens=final_tokens)
    scorer._check_head()
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if not closing:
        raise ValueError("native tokenizer must encode a closing-thought marker")
    eos = set()
    for value in (tokenizer.eos_token_id, getattr(getattr(scorer.model, "generation_config", None), "eos_token_id", None)):
        eos.update(value if isinstance(value, list) else [value])
    eos.discard(None)
    started = clock()
    generated, thought, final, past = [], [], [], None
    completed, stop_reason, error = False, None, None
    try:
        with torch.inference_mode():
            while True:
                if clock() >= deadline:
                    stop_reason = "walltime"
                    break
                if not completed and len(thought) >= scratchpad_tokens:
                    stop_reason = "scratchpad_cap"
                    break
                if completed and len(final) >= final_tokens:
                    stop_reason = "final_token_cap"
                    break
                if len(input_ids) + len(generated) >= MAX_CONTEXT_TOKENS:
                    stop_reason = "context_cap"
                    break
                ids = torch.tensor([input_ids if past is None else [generated[-1]]], device=scorer.device)
                mask = torch.ones((1, len(input_ids) + len(generated)), dtype=torch.long, device=scorer.device)
                output = scorer.model.model(input_ids=ids, attention_mask=mask, use_cache=True, past_key_values=past)
                logits = scorer.head(output.last_hidden_state[:, -1, :])[0].float()
                token = sample_token(logits, generated, generator)
                past = output.past_key_values
                phase = "final" if completed else "thought"
                generated.append(token)
                (final if completed else thought).append(token)
                if on_token is not None:
                    on_token({"index": len(generated) - 1, "token_id": token, "phase": phase,
                              "elapsed_seconds": clock() - started})
                if token in eos:
                    stop_reason = "eos_after_think" if completed else "eos_before_think_close"
                    break
                if not completed and thought[-len(closing):] == closing:
                    completed = True
                del output, logits
    except BaseException as exc:
        stop_reason, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        del past
    final_content = final[:-1] if final and final[-1] in eos else final
    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"generated_ids": generated, "generated_text": decode(generated),
            "thought_ids": thought, "thought_text": decode(thought), "scratchpad_tokens": len(thought),
            "final_ids": final, "natural_final_text": decode(final_content), "final_tokens": len(final),
            "completed_thought": completed, "natural_final_complete": stop_reason == "eos_after_think",
            "stop_reason": stop_reason, "error": error, "elapsed_seconds": clock() - started,
            "deadline_reached": clock() >= deadline}


def probe_budget(*, started=0., max_seconds=720., total_tokens=24576):
    if max_seconds > 720 or total_tokens > 24576:
        raise ValueError("sampled probe caps are 720 seconds and 24576 scratchpad tokens")
    return Budget(started=started, max_seconds=max_seconds, total_tokens=total_tokens)


def verify_source_pins(pins):
    for name, expected in pins.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"greedy parent source changed: {name}")


def load_parent(path=PARENT):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    gold = json.loads((path / "gold.json").read_text())
    if len(manifest["inputs"]) != 30 or len(manifest["prepared"]) != 30 or set(gold) != {str(i) for i in range(30)}:
        raise ValueError("completed thirty-question greedy parent required")
    reasons = {index: json.loads((path / f"reason-{index:03d}.json").read_text()) for index in range(30)}
    if any(record["index"] != index or record["manifest_sha256"] != digest(manifest) for index, record in reasons.items()):
        raise ValueError("greedy record identity or manifest binding mismatch")
    metadata = manifest["metadata"]
    verify_source_pins(metadata["followup_sources"])
    verify_source_pins(metadata["original_native"]["source_files"])
    pins = {str(file.relative_to(path)): file_sha256(file) for file in sorted(path.rglob("*")) if file.is_file()}
    return select_capped(manifest, reasons, gold), manifest, pins


def collect(selected, tokenizer, scorer, output, metadata, *, clock=time.monotonic):
    if len(selected) != 6:
        raise ValueError("exactly six post-hoc selected questions required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "recipe": RECIPE, "metadata": metadata,
                "selected": [{key: value for key, value in row.items() if key != "gold"} for row in selected],
                "limits": {"question_seconds": 120, "total_seconds": 720, "thought_tokens_per_question": 4096,
                           "total_thought_tokens": 24576, "final_tokens_per_question": 32},
                "selection": "two greedy scratchpad-cap(4096) questions/primitive by SHA256(sampled-native-v1:+caseid+:+qid)",
                "interpretation": "post-hoc six-case decoding/termination diagnostic; not a full thirty-question benchmark estimate"}
    _write_once(output / "manifest.json", manifest)
    _write_once(output / "gold.json", {str(index): row["gold"] for index, row in enumerate(selected)})
    budget = probe_budget(started=clock())
    records, conditioned, consumed = [], {}, 0
    stop = "six_attempts_finished"
    try:
        for index, row in enumerate(selected):
            allowance = budget.allowance(now=clock(), consumed=consumed)
            if allowance is None:
                stop = "total_scratchpad_cap" if consumed >= budget.total_tokens else "total_walltime_cap"
                break
            prepared = row["prepared"]  # Exact immutable parent token IDs and option mapping.
            generator = case_generator(scorer.device, row["seed"])
            with (output / f"tokens-{index:03d}.jsonl").open("x") as stream:
                def retain(event):
                    stream.write(encode_json(event) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                generated = generate_sampled(scorer, prepared["input_ids"], tokenizer, generator=generator,
                                             **allowance, final_tokens=32, clock=clock, on_token=retain)
            consumed += generated["scratchpad_tokens"]
            _write_once(output / f"generation-{index:03d}.json", {"seed": row["seed"], **generated})
            natural = parse_natural_answer(generated["natural_final_text"], prepared["codes"], prepared["labels"])
            natural["resolved"] = (generated["natural_final_complete"] and natural["format_valid"]
                                   and not generated["deadline_reached"])
            score, prefix, timely = None, None, False
            if generated["completed_thought"] and generated["error"] is None and clock() < allowance["deadline"]:
                prefix = prepare_completed_readout(tokenizer, prepared, generated)
                # Raw conditional readout has no sampling filters or presence penalty.
                score = scorer.score(prefix["input_ids"], prefix["code_token_ids"], prefix["labels"])
                timely = clock() <= allowance["deadline"]
                if timely:
                    conditioned[index] = score["semantic_probabilities"]
            record = {"index": index, "greedy_index": row["greedy_index"], "seed": row["seed"],
                      "generation": generated, "natural_answer": natural, "conditional_readout": score,
                      "conditional_prompt": prefix, "conditional_readout_within_deadline": timely,
                      "manifest_sha256": digest(manifest)}
            _write_once(output / f"question-{index:03d}.json", record)
            records.append(record)
            if generated["error"] is not None:
                stop = "generation_error_no_retry"
                break
    except BaseException as error:
        _write_once(output / "failure.json", {"error": f"{type(error).__name__}: {error}", "attempted": len(records),
                                               "scratchpad_tokens": consumed, "elapsed_seconds": clock() - budget.started})
        raise
    resolved = [record for record in records if record["natural_answer"]["resolved"]]
    correct = sum(record["natural_answer"]["label"] == _gold_label(selected[record["index"]]) for record in resolved)
    result = {"selected_questions": 6, "attempted": len(records), "stop": stop,
              "natural": {"resolved": len(resolved), "unresolved": 6 - len(resolved), "correct": correct,
                          "accuracy_among_resolved": correct / len(resolved) if resolved else None,
                          "correct_fraction_of_selected_six": correct / 6},
              "completed_thoughts": sum(record["generation"]["completed_thought"] for record in records),
              "conditional_readout": probability_metrics(selected, conditioned), "scratchpad_tokens": consumed,
              "natural_final_tokens": sum(record["generation"]["final_tokens"] for record in records),
              "elapsed_seconds": clock() - budget.started, "interpretation": manifest["interpretation"]}
    _write_once(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="validate frozen parent, sources and cached tokenizer; no model load")
    args = parser.parse_args(argv)
    if OUTPUT.exists():
        raise FileExistsError("sampled-six-v1 already exists; no overwrite, retries or regeneration")
    selected, parent_manifest, parent_pins = load_parent()
    tokenizer, original = cached_metadata(body="original", device="cuda")
    if original != parent_manifest["metadata"]["original_native"]:
        raise ValueError("original model/runtime/tokenizer provenance differs from the completed greedy run")
    for row in selected:
        prepared = row["prepared"]
        if (tokenizer.encode(prepared["prompt"], add_special_tokens=False) != prepared["input_ids"]
                or hashlib.sha256(prepared["prompt"].encode()).hexdigest() != prepared["prompt_sha256"]):
            raise ValueError("immutable parent prompt/token binding differs")
    metadata = {"parent": str(PARENT), "parent_files": parent_pins, "original_native": original,
                "sources": {"experiments/native_sampling_probe.py": file_sha256(__file__),
                            "tests/test_native_sampling_probe.py": file_sha256(ROOT / "tests/test_native_sampling_probe.py")}}
    print(encode_json({"preflight": {"selected": [{key: row[key] for key in
          ("greedy_index", "case_id", "question_id", "primitive", "seed")} for row in selected],
          "recipe": RECIPE, "metadata": metadata, "model_weights_loaded": False}}), flush=True)
    if args.preflight:
        return
    from experiments.architecture_diagnostics import GPU_lock

    with GPU_lock():
        scorer = load_model(original)
        result = collect(selected, tokenizer, scorer, OUTPUT, metadata)
    # The parent archive is read-only; certify that its entire tree is unchanged.
    after = {str(file.relative_to(PARENT)): file_sha256(file) for file in sorted(PARENT.rglob("*")) if file.is_file()}
    _write_once(OUTPUT / "parent-integrity.json", {"unchanged": after == parent_pins, "parent_files": parent_pins})
    if after != parent_pins:
        raise RuntimeError("greedy parent archive changed during sampled probe")
    print(encode_json(result), flush=True)


if __name__ == "__main__":
    main()
