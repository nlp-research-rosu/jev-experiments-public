# What changed in individual judgments

These are selected illustrations from the 35 unaffected families. Selection used the two largest NLL regressions among previously correct answers and two largest improvements in claim-versus-completion; they do not estimate failure rates.

| Case | Intended judgment | Starting v0.2 | Natural step 500 |
| --- | --- | --- | --- |
| Export E77 is explicitly approved but never launched; this rubric assigns approval to stage 1 | Stage 1 | 62.9% on stage 1 | 99.4% on stage 0; 0.52% on stage 1 |
| Amina is awarded; Boris is pending; rubric stage 2 means one award binding resolved | Stage 2 | 43.7% on stage 2 | 99.1% on stage 1; 0.51% on stage 2 |
| Ecologist's report is drafted but unsigned | Draft stage 2 | 97.2% on signed-verification stage 3 | 99.9% on draft stage 2 |
| Technician claims completion; required registrar signature absent | Technician claim | 86.0% on accepted | 94.8% on technician claim |

The gains show improved claim/evidence distinctions. The regressions show that accuracy gains coexist with very confident errors in other predicates or rubric definitions.

A plausible explanation for the export regression is rubric substitution: eight of ten new training categories use a common action rubric whose stage 0 means no invocation, even when approval exists. The held-out export rubric instead places approval alone at stage 1. The model's answer is consistent with applying the familiar training rubric instead of the supplied rubric. This is a hypothesis, not proof of the model's internal strategy.

That suggests a precise next contrast axis: keep the same record, vary the explicitly supplied rubric, and require the appropriate score to change. Add families with independently designed scales and thresholds, rather than primarily paraphrasing the same scale. Probability calibration could reduce certainty but cannot by itself correct the wrong class ordering.

Exact records, questions and full probability vectors are saved in `illustrative-errors.json`. Original labels and checkpoints were not changed.
