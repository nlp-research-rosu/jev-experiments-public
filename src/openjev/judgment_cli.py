"""JSON-file boundary for the local nested judgment API."""

import argparse
import json
from pathlib import Path

from .judgment_model import MODEL_ID, REVISION, JudgmentEngine, JudgmentModel


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def load_judgment_engine(
    *, checkpoint=None, device="cuda", backend="reference", unit_batch_size=4, max_input_tokens=8192, trainable=False
):
    from transformers import AutoTokenizer

    from .runtime import load_engine

    if checkpoint:
        model, _ = JudgmentModel.load_checkpoint(checkpoint, device=device, trainable=trainable, kernel_backend=backend)
        tokenizer = AutoTokenizer.from_pretrained(model.model_id, revision=model.revision)
        identity = model.checkpoint_id
    else:
        base = load_engine(MODEL_ID, revision=REVISION, device=device)
        model = JudgmentModel(base.model, kernel_backend=backend)
        tokenizer = base.tokenizer
        identity = "openjev-judgment-v0.2/untrained"
    return JudgmentEngine(
        model, tokenizer, model_id=identity, unit_batch_size=unit_batch_size, max_input_tokens=max_input_tokens
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backend", choices=("reference", "fla"), default="reference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--unit-batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.unit_batch_size < 1 or args.max_input_tokens < 1:
        parser.error("choose a new output and positive resource bounds")
    request = read_json(args.request)
    engine = load_judgment_engine(
        checkpoint=args.checkpoint,
        device=args.device,
        backend=args.backend,
        unit_batch_size=args.unit_batch_size,
        max_input_tokens=args.max_input_tokens,
    )
    result = engine.evaluate(request, details=args.details, cached=not args.independent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}: {result['usage']['model_units']} units, {result['elapsed_ms']:.1f} ms")


if __name__ == "__main__":
    main()
