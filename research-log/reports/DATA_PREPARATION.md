# Curated judgment corpus v0.2

Prepared 2026-09-19 US/Central. The corpus is built and available at
`data/processed-v0.2/`, with **20,000 training bundles**,
900 validation bundles, 900 calibration bundles, and 1,500 test bundles. These
23,300 original-case bundles contain 36,950 supervised views. No model training or
quality result is implied by this data report.

The complete, machine-readable inventory is
`manifest.json`. Source downloads are
locked by URL, revision where available, byte count, and SHA256 in
`source-lock.json`. The 23 locked artifacts
total 351,146,871 bytes. All downloaded data extends beyond the prior first-100
read-only audit. Original BoolQ, CLINC150, PAWS, BANKING77 verification data and
Wine Quality releases are complete; Tasksource training coverage is explicitly
limited to one complete shard of seven, plus its complete validation/test shards.

## Retained counts

Counts below are original bundles, **not** candidate rows or augmented views.

| Component | Train | Validation | Calibration | Test | Training objective |
|---|---:|---:|---:|---:|---|
| Original BoolQ | 4,000 | 150 | 150 | 250 | Passage-grounded Noul |
| Original PAWS-Wiki labeled_final | 4,000 | 150 | 150 | 250 | Equivalence Noul |
| Original CLINC150 | 6,500 | 150 | 150 | 250 | Restricted 4/8-candidate Choice |
| Tasksource BANKING77 component | 1,500 | 150 | 150 | 250 | Declared 4-candidate Choice |
| UCI Wine Quality | 3,000 | 150 | 150 | 250 | Three-band ordinal Score |
| Constructed policy rubrics | 1,000 | 150 | 150 | 250 | Four-level oracle Score |
| **Total** | **20,000** | **900** | **900** | **1,500** | |

Training contains 8,000 Choice, 8,000 Noul and 4,000 Score bundles. The other
partitions have equal primitive counts. Training has 32,000 supervised views,
8,000 exact-complement relations, and 4,000 PAWS sentence-swap invariant relations.
Its 18,700 distinct context groups require 77,884 isolated model units before any
runtime batching. Candidate counts never exceed eight.

Source quotas are followed by deterministic round-robin selection over task and
original-gold strata; SHA256 ranks select examples within each stratum and
interleave the final file. This is source/label balancing, not natural deployment
prevalence. Both BoolQ and PAWS have 2,000 original positive and 2,000 original
negative training cases. Wine bands have 1,098 / 1,097 / 805 training cases because
the highest band has fewer eligible distinct examples. Manifest label counts
retain the full breakdown.

## Supervision and transformations

Each bundle retains semantic examples separately from source/task IDs, source row,
revision, original partition, labels, grouping keys, and source-file hashes.
Only `state` and `question` are intended for the compiler/model. Gold fields are
`choice`, `level_index`, or `truth`; no confidence gold or soft labels are invented.
The renderer/compiler integration remains the root worker's responsibility.

**BoolQ** preserves the complete passage and asks the original yes/no question
relative to it. The derived negative question scopes over the entire original
judgment and exchanges true/false criteria and boolean labels. It does not edit
source text by inserting an arbitrary negation. Test negatives use a held-out
mechanical wording (“answer NO”) while training/validation/calibration use the
“false that the answer is YES” wording. These are constructed views of one human
annotation, not additional independently annotated cases.

**PAWS** uses both positive and negative human paraphrase labels for the explicit
same-meaning judgment. Every bundle has the original, its exact complement, and a
sentence-swap view. The swap preserves the equivalence label even for negative
pairs; it does not assert that different-meaning sentences are interchangeable in
other tasks.

**CLINC150** includes the source gold plus deterministically hash-ranked negatives
from its 150-intent ontology, alternately declaring four or eight candidates.
The question explicitly asks for the best intent among those supplied and states
that the correct intent is included. All 1,200 out-of-scope source records are
excluded. **This does not reproduce original 150-way classification or OOS
prediction.** The full ontology and negative-selection rule are recorded.

**Tasksource BANKING77** preserves its four declared alternatives. Every selected
utterance and gold label is checked against the complete pinned original
BANKING77 train/test CSVs, and every candidate against the original 77-label
ontology. Original test membership overrides a derivative split name. Different
Tasksource candidate subsets for the same original utterance are deduplicated.
Only this one approved component is included: 2,975 candidate records were found
while scanning 1,052,435 rows. There is no claim of broad Tasksource family
coverage. Other tasks, including passage-omitting reformatted BoolQ, are excluded.

**Wine Quality** supplies real expert sensory ratings with explicitly ordered
meaning. The question predicts a declared band from the wine type and all eleven
physicochemical features: original ratings 0–5 map to level 0, rating 6 to level 1,
and ratings 7–10 to level 2. The original sensory rating is provenance only.
This is expert-band prediction from numeric measurements, not sentiment or a
measurement of general natural-language rubric competence. Identical feature
records with different resulting bands are removed as conflicting gold; repeated
records in the same band are deduplicated.

**Synthetic Score** contributes only 5% of training bundles. Four threshold
rubrics use deterministic arithmetic outcomes, with boundary examples. Inventory,
queue and completion families are available for training. The latency-severity
family and its wording are held out completely: zero training/validation/
calibration cases and 72 retained test cases. This tests an unseen constructed
rubric family, not real-world generalization. The generation version, input-spec
hash, and preparation-code hash are recorded. The examples are explicitly marked
synthetic with `label_origin: deterministic-oracle`.

## Splitting and overlap controls

All 114,070 converted original cases are grouped **before augmentation and bounded
sampling**. Context fingerprints normalize Unicode, whitespace and case. A
union-find connects any shared passage/utterance/sentence across all loaded sources
and partitions. PAWS also groups token-multiset sentence variants to catch
word-swap families; this grouping creates no semantic invariance annotation.
Wine uses complete measurement-state fingerprints. Synthetic grouping includes
the family, measured value and rubric thresholds.

Original test membership wins for a connected component. Original dev/validation
components are held out, then deterministically assigned between validation and
calibration. Remaining components hash to 80% train, 7% validation, 7%
calibration, 6% test. The held-out synthetic latency family is assigned to test.
No related view is split across files. Original training contexts connected to
an original heldout context are promoted out of training: **726 groups** required
this protection. In particular, BoolQ's final local test uses reserved original
training contexts because its publicly labeled original validation partition is
reserved for validation/calibration. These files are a curated local evaluation,
not the complete official benchmark test sets.

Duplicate semantic cases are resolved deterministically, preferring an original
heldout record. All records in a duplicate case with conflicting targets are
excluded. This removed **1,296 duplicate rows and 85 conflicting-label rows**,
leaving 112,689 eligible cases before source quotas. Exact cross-source context
connections were checked; zero were observed in these chosen components. This
means no exact match was found, not proof of no semantic overlap.

The original PAWS raw-to-final lineage archive returned HTTP 403 at
`https://storage.googleapis.com/paws/english/wiki_raw_and_mapping.tar.gz`.
The pinned released pairs remain usable, but the grouping cannot guarantee that
disjoint backtranslations or separate sentences from the same article stay
together without unavailable lineage/document IDs. This limitation is retained
in the manifest. Neither normalization nor public heldouts can rule out backbone
pretraining memorization. Public source evaluations mostly test new examples of
known tasks; only the small synthetic family is intentionally unseen.

## Provenance and terms

Every source retains its original terms; no collection-wide license replaces them.
The downloaded evidence files and content hashes are listed in the lock/manifest.

| Source | Pinned revision / source identity | Recorded terms and attribution |
|---|---|---|
| [BoolQ](https://huggingface.co/datasets/google/boolq) | `35b264d03638db9f4ce671b711558bf7ff0f80d5` | CC BY-SA 3.0; Clark et al., *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions*, NAACL 2019 |
| [CLINC150](https://github.com/clinc/oos-eval) | `828f8093932c8fe6ca7936c3d2e52903b1c523de` | CC BY 3.0; Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019 |
| [PAWS-Wiki](https://github.com/google-research-datasets/paws) | HF `161ece9501cf0a11f3e48bd356eaa82de46d6a09`; original repo `02b29f3af1143620d1b7f352247e65c0bdd2ec18` | Original dataset permission for any purpose, acknowledgement requested; Zhang, Baldridge and He, *PAWS*, NAACL 2019 |
| [Tasksource](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) / [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | Tasksource `1dee7ed51b87880e37882c2cbed2d50bd114e0df`; component `57ec275d8078af65b7731c2a98be812d844a6d6b` | Tasksource formatting Apache 2.0, BANKING77 CC BY 4.0; Sileo (2023), Casanueva et al. (2020) |
| [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) | DOI `10.24432/C56S3T`; archive SHA256 `3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4` | CC BY 4.0; Cortez, Cerdeira, Almeida, Matos and Reis (2009) |
| Constructed rubrics | `oracle-thresholds-v1` and code/input-spec hashes | Project-authored synthetic data; no third-party data |

SST5 and TweetEval are omitted because their original component provenance/terms
were not resolved. The UCI page explicitly supplies the Wine Quality license;
Tasksource's top-level label is not substituted for BANKING77 component terms.
Source-specific licenses, citations and transformed-example attribution are
retained in the manifest for downstream use.

## Reproduction and verification

The preparation code is [`prepare_data.py`](../../src/openjev/prepare_data.py), with
source-shaped tests in [`test_prepare_data.py`](../../tests/test_prepare_data.py).
The existing baseline environment is untouched. The parent-provided Arrow 21
package overlay is used only to read parquet; other preprocessing uses stdlib.

```sh
PYTHONPATH=.runtime-training:. .venv/bin/python -m openjev.prepare_data
# Restore missing locked artifacts, verifying each SHA256:
PYTHONPATH=.runtime-training:. .venv/bin/python -m openjev.prepare_data --download
.venv/bin/python -m pytest tests/test_prepare_data.py -q
.venv/bin/python -m ruff check src/openjev/prepare_data.py tests/test_prepare_data.py
```

The 15 owned tests and owned lint pass. A second complete preparation produced
byte-identical train, validation, calibration, test and manifest files. An
independent full output scan verified counts, file hashes, legal targets,
complement criteria/labels, invariant labels, candidate limits, unique IDs, and
zero cross-split overlaps among the 21,925 groups and 34,074 recorded context keys.
It found no original heldout record or latency-family record in training. Evidence
is in `audit.json`.

Training file SHA256:
`b09b13b6644dddfb095b13a113352d452984268156907bc4f7058a775a237420`.
All other file hashes and the preparation-code/source-lock hashes are in the
manifest. The longest stored semantic example is 5,146 JSON characters; tokenizer
lengths and runtime non-truncation remain integration checks.

The whole-project suite was also attempted during concurrent implementation:
165 tests and 26 subtests passed, with 11 failures and 14 warnings. Six existing
engine tests selected newly available FLA/Triton kernels for CPU tensors:
`test_cache_fork_does_not_mutate_original_or_share_recurrent_rows`,
`test_extreme_length_outlier_does_not_multiply_padding_work`,
`test_padded_endpoints_match_independent_scores_with_and_without_prefix`,
`test_parallel_matches_full_reference_with_variable_lengths_and_chunking`,
`test_selected_head_matches_full_logits_and_preserves_candidate_order`, and
`test_single_field_and_identical_prompts`. Five new training tests ran before
`judgment_training.py` existed: `test_complement_loss_does_not_replace_supervision`,
`test_grouped_choice_loss_not_independent_binary_losses`,
`test_invalid_targets_relations_and_partial_groups_rejected`,
`test_invariant_choice_aligns_names_before_probability_distance`, and
`test_score_loss_uses_distribution_not_only_its_mean`. These observations were
reported to the root worker for integration; this report does not claim a green
whole-project suite.

Subsequent integration resolved those observations: selecting reference kernels
before CPU tests and completing the training modules gives **180 passing tests
plus 45 subtests** in the pinned training overlay. See
[training readiness](TRAINING_READINESS.md) for the definitive smoke and run status.
