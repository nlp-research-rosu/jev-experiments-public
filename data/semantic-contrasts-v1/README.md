# Frozen semantic diagnostic v1

This is evaluation data, **not a training split**. It was authored and semantically
reviewed before the first model evaluation. The first model outputs were inspected
only after `suite.json` and `manifest.json` were frozen.

The 114 scenarios have 764 hard-label judgments and 260 relations across six
families. `request` is the only model input. IDs, expected labels, rationales,
family/variant metadata and relation definitions must remain outside prompts.

Binary targets describe their exact evidence-relative predicates. Unknown outcomes
use explicit Choice labels; they are not replaced by invented probability 0.5 gold.
Claim and confirmation are distinct and can have all four truth combinations.

Keep the suite unchanged for regression comparisons. If semantics require a fix,
make a new version and retain the original. Any future training examples must be
separate; reserve new families/wordings for an untouched transfer evaluation.
These correlated templates do not support general population accuracy claims.

See [findings and reproduction](../../research-log/reports/SEMANTIC_CONTRASTS.md).
