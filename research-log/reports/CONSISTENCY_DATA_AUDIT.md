# Data support for supervised and consistency losses

Checked 2026-09-19. Machine-readable evidence is in
consistency-data-audit.json.

We inspected the first 100 training rows for five Hugging Face datasets and counted
the original, pinned CLINC150 JSON splits. These samples are not random and do not
establish overall annotation quality or the final number of usable examples.
No training corpus has been converted and no model has been trained.

| Source | Available training rows | Verified fields/labels | Supported use |
|---|---:|---|---|
| [BoolQ](https://huggingface.co/datasets/google/boolq) | 9,427 | `passage`, `question`, boolean `answer` | Direct binary supervision; derived exact-complement views |
| [CLINC150](https://github.com/clinc/oos-eval) | 15,100 including 100 out-of-scope | Utterance plus intent name; 150 in-scope labels | Choice supervision, with an explicitly defined candidate set and none-of-the-above option |
| [Tasksource Instruct](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) | 5,314,383 | `inputs`, `targets`, source `task` | Diverse categorical/binary supervision after per-task conversion |
| [PAWS-Wiki labeled_final](https://github.com/google-research-datasets/paws) | 49,401 | Two sentences and paraphrase/non-paraphrase label | Equivalence Noul, sentence-swap symmetry, validated semantic transformations |
| [SST5](https://huggingface.co/datasets/SetFit/sst5) | 8,544 | Review, integer 0–4, matching ordered sentiment label | Five-level Score prototype |
| [TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | 45,615 | Text and negative/neutral/positive label | Three-level Score prototype |

These are available row counts, not six disjoint pools. Tasksource already contains
PAWS and BoolQ-derived tasks and may overlap other sources. Counts of augmented
views must not be added as though they were new independently labeled examples.

## Findings that affect conversion

**BoolQ:** all 100 inspected records have the expected passage/question/boolean
shape. One exact-complement view per source row would give up to 18,854 supervised
views before filtering, but still only 9,427 independent original training
questions. Construct the complement of the complete yes/no judgment; do not
replace it with an unrelated opposite-sounding question.

**CLINC:** the pinned file contains 15,000 in-scope train examples plus 100
out-of-scope, 3,000+100 validation and 4,500+1,000 test examples. The source labels
are nominal intents, not ordered Score levels. A full question can have 150
intents plus an out-of-scope option. That is valid for the API but costly for an
isolated-candidate scorer; candidate-subset experiments must be explicit tasks.

**Tasksource:** 98 of the first 100 rows have a target matching exactly one of the
quoted alternatives under a simple format check; the other two are token-tagging
tasks. This is not a 98% usability estimate. Parsing a label does not prove that
the underlying context, definitions or annotation are suitable.

In the inspected `super_glue/boolq` row, the reformatted input contains the question
but omits the original passage. That can be a different knowledge task, but is
not equivalent to passage-grounded BoolQ. Use original BoolQ for that objective.
Also exclude unsupported sequence-tagging outputs and avoid mapping NLI `neutral`
to the falsehood of a hypothesis: absence of entailment is a different event.

**PAWS:** the sample includes both labels, including highly similar strings marked
as different in meaning. Train the explicit equivalence task on both labels.
For that task, exchanging the two sentences preserves the label. Positive pairs
are possible augmentation material for other tasks, but only when that task's
decision is preserved. Negative pairs are not instructions to make every arbitrary
downstream answer differ. The original release has additional raw-sentence mappings
that can help group related variants when constructing splits.

**Score:** SST5's sample covers all five ordered sentiment names; TweetEval's
sentiment configuration explicitly provides three ordered labels. These are
supervision for levels, not examples of true per-record probability distributions.
They cover sentiment, not the full range of arbitrary user-defined rubrics. A pilot
can test mechanics here; general-rubric claims need additional domains and held-out
rubric families. Arbitrary intent IDs must never be treated as an ordinal scale.

## What is missing

- Per-example calibrated model-confidence targets. They are not necessary for
  supervised log losses or population-level calibration evaluation, but should
  not be fabricated from one-hot labels.
- Ready-made negated-question pairs across all tasks. They need controlled
  construction with inverse labels and retained relation metadata.
- A broad supply of verified question/rubric paraphrases that preserve each task's
  exact boundary. Dataset diversity alone is not a paraphrase-consistency label.
- General multi-question logical constraint graphs and diverse custom-policy Score
  tasks. A smaller constructed component with explicit truth rules is needed to
  investigate these beyond the first complement/symmetry experiments.

## Provenance before training use

BoolQ declares CC BY-SA 3.0. CLINC's original repository carries CC BY 3.0. The
PAWS-Wiki repository permits use for any purpose with acknowledgement requested.
Tasksource's top-level Apache label does not substitute for checking selected
component sources. SST5's inspected metadata has no license entry; TweetEval's
upstream README refers users to component datasets and source terms. Resolve and
record those original terms before including an ordered-sentiment subset.

This is a data-content feasibility audit, not a statement that every row/source is
ready for ingestion. Revision IDs, sample hashes, label counts and observed gaps
are retained in the companion JSON.
