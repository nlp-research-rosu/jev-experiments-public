# General structured decisions: discussion brief

**Latest proposal:** [training semantics v0.2](training-contract-v0.2.md) addresses
consistency, primitive-specific model outputs/losses and audited data support.
It remains a design for review, not a launched training run.

**Current gate:** [probability semantics and comparability](../reports/PROBABILITY_DESIGN.md)
are under review. The v0.1 model contract and all dataset-size suggestions remain
proposals; no large corpus conversion or training should begin from them yet.

**Updated direction:** the later model I/O contract draft
proposes isolated criterion inputs and grouped numerical-score supervision. It
supersedes this brief's earlier suggestion to train per-question option-code
outputs. The public-data research below remains relevant; conversion and training
must follow the contract chosen after review.

The user's chosen target is a model that follows schemas supplied at runtime
across domains, field names and decision rules. The five support fields are an
evaluation fixture, not the intended specialization. No training has started.

## Input and output contract

The current engine accepts text context and a schema of independent boolean or
categorical fields, including descriptions and allowed values. It maps values to
single-token codes, evaluates each field, and constructs typed JSON in Python.
Generalization means using unfamiliar descriptions, labels, rules and schemas.
Arbitrary free-text extraction, unbounded numbers and variable-length arrays
require additional output machinery; the current one-token field scorer does
not produce those by itself. Schema validity does not imply semantic correctness.

## Exact prompt examples

`prompt-examples/` contains the actual current renderers' output for `v4_t08`:
"The API is healthy but the desktop client cannot print. Fix the printing problem
today and give me a telephone call."

- Request: `prompt-examples/request.json`.
- Expected/previously measured output: `prompt-examples/expected-output.json`.
- No examples: `prompt-examples/no-examples.txt`, priority prompt 472 tokens.
- Shorter examples: `prompt-examples/shorter-examples.txt`, priority prompt 1,207 tokens.
- Long examples: `prompt-examples/long-examples.txt`, priority prompt 2,182 tokens.
- Readable message arrays and token/hash metadata accompany those files.

All three full prompt-list hashes reproduce the recorded v4 benchmark, and all
three recorded predictions match the displayed output. This was a CPU rendering
check, not a new inference benchmark.

Long examples contain four field-specific native user/assistant demonstrations.
Every demonstration repeats the complete five-field schema. Shorter examples put
the schema in the system message and omit it from each demonstration; the final
user turn remains unchanged and still includes a schema copy. Both keep full
field questions/options and the same four examples. The original no-example
format has no demonstration history. These are prompts, not trained weights.

The five histories diverge at field-specific questions, so only their common
prefix can be reused. Most demonstration tokens still need processing per field.

## Public data candidates, checked 2026-09-19

| Source | Available scale | Relevance | Preparation constraints |
|---|---|---|---|
| [Tasksource Instruct](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) | About 5.31 million training rows, recast from 485 datasets | Diverse classification, inference, logic and multiple-choice tasks | Select compatible tasks; preserve original labels and task provenance; exclude unsupported token tagging/free-form outputs; audit source splits and component licenses. |
| [CLINC150 original](https://github.com/clinc/oos-eval) | 15,100 training examples; 150 in-scope intents and out-of-scope data | Intent/tool-like choice, changing candidate sets, none-of-the-above | Preserve original train/validation/test splits; track excluded intents/domains explicitly. |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | 9,427 training and 3,270 validation examples | Ground a boolean decision in a provided passage and question | Preserve the passage; group shared passages; avoid duplicate copies through Tasksource. |
| [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Five subsets; single-turn function-calling subset has about 1.89k rows | Tool choice and schema following | Most arbitrary arguments need generation; extract only decisions with valid finite candidate sets. |
| [xLAM/APIGen](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60,000 generated examples using 3,673 APIs across 21 categories | Broad function selection and typed arguments | Gated access: current unauthenticated file request returned 401. Requires account acceptance of access conditions. Not needed to start with the other sources. |

Observed Hugging Face revisions:

- Tasksource: `1dee7ed51b87880e37882c2cbed2d50bd114e0df`.
- BoolQ: `35b264d03638db9f4ce671b711558bf7ff0f80d5`.
- Hermes: `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`.
- xLAM: `26d14ebfe18b1f7b524bd39b404b50af5dc97866`.

These are candidate sources, not a downloaded or validated training corpus.
Row counts above describe their available data, not independent retained samples
after filtering. Different datasets can overlap, particularly through Tasksource.

## Proposed pilot

Use a balanced subset of roughly 20,000–50,000 labeled decisions across diverse
tasks; finalize its size after measuring training throughput. Convert each to
context + schema + field question → correct single-token code, matching the
short-prompt inference contract. Randomize option-code assignments and display
order consistently. Train a LoRA adapter and evaluate a merged checkpoint using
the same batched engine.

Reserve complete source tasks, schemas and policy families for evaluation in
addition to ordinary row-level splits. Keep paraphrases/shared source contexts
together. Preserve existing datasets as regression checks. Public benchmarks may
already have appeared in base-model pretraining, so held-out fine-tuning tasks
alone are not proof of complete novelty to the base model.

Add a smaller, explicitly labeled synthetic policy component where the answer is
computed from known facts and a supplied rule. Vary the rule itself: for example,
the same 'fix printing today' context can be urgent under a today-deadline rule
and normal under an outages-only rule. This tests whether the model reads the
schema rather than memorizing a permanent interpretation of `priority`.

Success is improved correctness and option stability with short prompts on held-out
tasks/schemas, while preserving schema validity and low measured latency. General
capability and equivalence to Jev remain hypotheses to test, not established facts.

GPU kernel correctness, backward-pass stability and memory/throughput smoke tests
remain prerequisites for the proposed bounded unattended training run. The earlier
three-hour local budget was proposed, not measured as a completion estimate.
