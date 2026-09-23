# Post-selection input-contract clarification

The original paired-language training and selections remain frozen. This diagnostic
was added after error analysis exposed public definitions that were available to
whole-request reviewers but absent from isolated model scoring units.

Confirmed test scope: 44 judgments in five families. The ordered-rubric families
contain missing numeric band definitions in binary questions, and one family has
higher-band triggers referenced from isolated Choice/Score candidates. The evening
shift family omits its window from two records while defining it in a sibling
question. The validation set has the same band-definition issue in two families.

The correction copies existing Choice/Score definitions into shared state for all
four variants of every ordered-rubric family. For all four test/temporal_scope/02
variants it exposes the already-stated evening window, 14:00–18:00, in shared state.
No factual event, question, target, candidate order, relation, training datum,
checkpoint weight, or original selection changes. Data is versioned separately in
`data/paired-language-clarification-v1`; original hashes remain intact.

All new inputs are frozen before diagnostic model requests. Reviews verified that
every changed compiled unit receives the exact public definitions. New Jev responses
use the durable archive, with no discarded paid results.

Report three views: original frozen scores (with caveat); the conservative untouched
35-family / 700-judgment subset; and full-suite post-hoc clarified-input sensitivity.
The last is not an independently frozen confirmatory result. A single Jev response
per new request also leaves sampling variation unmeasured.

Corrected validation is evaluated at the already-saved updates 0, 250, 500 and 1000,
using the unchanged original broad-validation predictions and unchanged selection
formula. Any alternative selection is locked before diagnostic local test requests.
It is a selection-sensitivity result, not a replacement for the original lock. If it
names a previously untested checkpoint, evaluate that checkpoint on clarified test,
original broad test and the original semantic suite. Never train or tune on these
new test results.

This exposes a review-process failure: whole-request label agreement does not prove
that each isolated scoring unit has enough premises. Future data review must inspect
rendered per-unit inputs as well as the full API request.
