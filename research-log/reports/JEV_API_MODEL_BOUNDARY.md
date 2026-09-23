# Jev: API contract, model disclosures, and a reproduction boundary

Checked 2026-09-19. This review distinguishes official documentation, published
client/adapter source, and our proposed implementation. No Jev API request or
training run was performed.

## External contract

[HTTP reference](https://docs.typesafe.ai/api): `POST /v1/systemone` takes `model`,
`state` and a named map of `questions`. Each question supplies its `type`,
`instructions`, and primitive-specific `criteria`. The response contains `model`,
an `answers` map under the original IDs, and `usage` counters.

| Question type | Criteria | Answer payload |
|---|---|---|
| Choice | Up to 255 option names and descriptions | Selected name, per-option probabilities, confidence |
| Score | 2–10 ordered descriptive levels | Weighted mean of level indices, per-level probabilities, legend, confidence |
| Noul | Optional descriptions for true/false | Probability of true in `noul` |

The [State page](https://docs.typesafe.ai/concepts/state) accepts a text string,
JSON object or array. Structured instructions and criteria are described in
[Advanced structure](https://docs.typesafe.ai/primitives/advanced). The SDK's
generic EntryType also includes null, while the HTTP table is narrower in places;
the examples here use the common documented subset.

See `docs/api-examples/request.json` and `response-illustrative.json` for a mixed
request/response. All response numbers are illustrative, explicitly recorded in
`docs/api-examples/README.json`; they are not predictions or latency measurements.

## What the docs explicitly say reaches the model

- The question-map IDs are not sent to the underlying model. Complete semantics
  must therefore appear in instructions and criteria, not just in a key such as
  `refund_requested`. [Questions](https://docs.typesafe.ai/primitives)
- Choice option names and their descriptions both reach the model. Arbitrary
  structured-description keys such as `focus` and `examples` are visible content,
  not reserved API features. [Choice](https://docs.typesafe.ai/primitives/choice)
- Score levels are evaluated separately. The documentation says the model does
  not see a level's number or neighbouring levels. This is a meaningful model-side
  constraint; it does not disclose the tensor layout or normalization procedure.
  [Score](https://docs.typesafe.ai/primitives/score)
- Questions share a state but are independent; one answer does not become another
  question's hidden context. The documented behavior says adding/removing questions
  does not change the remaining answers. [Primitives](https://docs.typesafe.ai/primitives)
- The service describes reading state once and evaluating questions in parallel.
  One API request is not evidence of one neural forward call or one CUDA kernel.
  [Models](https://docs.typesafe.ai/models)

## What is not disclosed in the material inspected

The inspected docs, launch post, official SDK and comparison adapter do not provide
Jev's tokenizer, exact serialized model prompt, tensor signature, model parameter
count, attention masks, network architecture implementation, output-head design,
or full RLCD training recipe. They do not establish that Jev emits A/B tokens, uses
an ordinary vocabulary head, or shares KV caches in the same way as our Qwen code.
TypeSafe claims a new architecture, parallel sampler and RLCD in its
[launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

The public SDK at commit `2ce5c65f13646cab6e6f782328194c9d85f3300a` validates/serializes
the external request and sends it over HTTP. It does not expose the server's
model-input encoding. See its
[endpoint preparation](https://github.com/typesafe-ai/typesafe-sdk-python/blob/2ce5c65f13646cab6e6f782328194c9d85f3300a/src/typesafe_sdk/_core/endpoints.py).

## Software versus learned inference

This is an implementation boundary we can adopt; it is not a recovered diagram
of Jev's private internals.

| Component | Responsibility |
|---|---|
| Request wrapper | Validate types; retain question IDs; select model; preserve structured state/rubrics |
| Input preparation | Encode semantic content into our chosen token/tensor representation |
| GPU runtime | Batch work, reuse state representations where valid, manage memory and caches |
| Learned model | Estimate how the supplied state satisfies a question and candidate criterion |
| Probability transformation | Normalize scores as appropriate; apply any independently fitted calibration transform |
| Response wrapper | Select Choice winner, compute Score expectation, expose P(true), attach confidence and original IDs |
| Application | Threshold, fetch data, extract candidates, combine results, validate dependencies and perform actions |

The semantic judgments and useful uncertainty estimates are the learned work.
JSON construction, label mapping, weighted averages, confidence statistics and
control flow do not require the model to generate text. Calibration can involve
both training and fitted post-processing; normalization alone is insufficient.

## Public comparison adapter: useful software, not Jev

TypeSafe's [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)
implements the same outward interface using ordinary LLM APIs. It serializes
structured instructions, asks for discrete answers or generated probabilities,
normalizes returned distributions, and computes derived response fields. This
does not reveal Jev's neural mechanism.

The [confidence source](https://github.com/typesafe-ai/system-one-adapter-python/blob/adffc2eab300a4fa3c0e92252d4ffd6ceaa53700/src/system_one_adapter/_utils/confidence_metrics.py)
publishes deterministic formulas. For K choices it uses
`(max(p) - 1/K) / (1 - 1/K)`. For Score it scales the probability-weighted distance
from the modal level against a uniform reference, clipping below at zero.
These formulas are concrete reference implementations, not proof that the Jev
server uses precisely the same definitions. This refines the earlier finding:
the confidence *documentation page* omits a formula, but the separate adapter's
formula is public.

## Consequences for our training design

Define our learned interface as semantic state + self-contained question/rubric
→ numerical decision scores. Keep HTTP IDs and response construction outside it.
Persist probabilities instead of training the model to write JSON or confidence
numbers as text. Keep structured inputs in a stable representation shared by
training and inference.

Our existing Qwen backend gives a field prompt all its options and scores one code
per option at a single endpoint. It is a useful baseline, but it does not reproduce
the documented Score level-isolation behavior. For a closer reproduction, prototype
a criterion-isolated scorer: state + question + one level description → evidence
for that level; evaluate the levels in a batch and then form the distribution.
The evidence-to-probability mapping is a design choice requiring evaluation, not
something TypeSafe's docs specify. Choice option isolation is not explicitly
promised by the pages reviewed and should not be inferred from the Score page.

Before committing to a long training run, compare this alternative's accuracy,
probability quality, speed and memory at varying numbers of fields and options.
Its extra criterion branches may cost more than our current one-branch-per-field
path. Our current prompts also expose field IDs and the whole schema; API fidelity
would instead require self-contained question meanings and keeping arbitrary IDs
as metadata. These are now explicit design gaps, not hidden assumptions.
