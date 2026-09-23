# Attribution

The four JSON files under `data/upstream/` were downloaded without modification
from Harsha Gundala's `harshatheg/Qwen-2.5-1B-RLCD` repository, revision
`2af86848be75847ccb3553b0941cc51d6ef7e4e9`.

Source: https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD

The upstream repository declares Apache License 2.0. Its exact provenance is also
recorded in `data/upstream/PROVENANCE.json`; a copy of the license is included at
`data/upstream/LICENSE`. The implementation here was written
for this experiment; it is inspired by the shared-prefix/parallel-field approach.
`reports/harsha-reference-readme.md` retains the README from that same revision
as a source for comparing the published benchmark claims.

Qwen3.5 model weights are supplied by the Qwen team under their model repository's
license (Apache License 2.0 for Qwen3.5-2B). Base weights are downloaded into the
normal Hugging Face cache and are not bundled in this project. Locally trained
LoRA adapters and readout/probe weights are not bundled in this reference
repository; the saved evaluation predictions are included instead.

Source: https://huggingface.co/Qwen/Qwen3.5-2B

## AI-text detection dataset

`data/ai-text-detection-v1/human/` contains text fetched without modification (apart from whitespace,
citation brackets and HTML entities) from Wikipedia revisions dated 2019 and Stack Exchange answers
dated 2012 to 2019, both licensed CC BY-SA, plus public-domain Project Gutenberg essays and a US
National Park Service page. `data/ai-text-detection-v1/human/SOURCES.tsv` records the source URL,
revision or snapshot date and license of every file. The five writing-habit families used in
`experiments/ai_text_detection/` follow the taxonomy of https://github.com/blader/humanizer (MIT),
which is based on Wikipedia's "Signs of AI writing".

## KleverBench proof-spec cases

The 45-case proof-spec audit is derived from reference specifications in
[KleverBench](https://github.com/nlp-research-rosu/KleverBench) at commit
ae3a9fea. The case generator retains the task, program, formal language
semantics and verification file, then creates faithful or flawed candidate
specifications. See [the case-study method](data/kleverbench-judge-v1/README.md)
and its saved manifest for the exact variants and labels.
