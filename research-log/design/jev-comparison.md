# Compare the local model with live Jev

Access is verified: see the [first live eight-case comparison](../reports/JEV_LIVE_SMOKE.md).

The live replay sends the frozen diagnostic's exact state, instructions and
criteria to TypeSafe. Expected answers, rationales and contrast relations stay in
the local evaluator. Jev responses are comparison observations; they never
replace the authored labels or enter a training set automatically.

The initial selection contains eight scenarios and 51 Noul/Choice judgments:
unsupported completion claim, successful execution, uncertain timeout, claimed
but unverified destination change, verified change without a claim, an unverified
record, conflicting records, and a verified unchanged record. The complete frozen
suite has 114 scenarios and 764 judgments. Those original artifacts contain only
Noul and Choice. The current replay also supports Score in newly frozen suites,
with hard, zero-based rubric indices as targets; it does not rewrite old suites
or results.

## Configure access

Create an API key in the [TypeSafe console](https://console.typesafe.ai/).
Save only the key, without quotes or a variable assignment, in
`.secrets/typesafe_api_key`. The prepared file is readable/writable only by its
owner and the directory is Git-ignored. Alternatively, set `TYPESAFE_API_KEY` in
the environment of the process running the experiment. Keep credentials out of
chat, source files and reports. The key loader never executes the file as shell
code. Environment configuration takes precedence over the file.

The integration uses the documented
[HTTP endpoint](https://docs.typesafe.ai/api), so it needs no additional SDK
installation or change to the training environment. It sends credentials only to
`https://api.typesafe.ai`, does not follow redirects, and does not save outbound
authorization headers or key configuration. It preserves only request IDs, Retry-After and
Content-Type from response headers.

## Billing and throttling

Checked 2026-09-20: the [model documentation](https://docs.typesafe.ai/models)
publishes $0.042 per million input tokens, free output tokens, 1,200 requests per
minute and 250,000 tokens per second. TypeSafe explicitly says these rate limits
are changing dynamically during the rollout; do not treat them as a permanent
account-specific entitlement. The [API reference](https://docs.typesafe.ai/api)
distinguishes rate limiting (429) from temporary overload (529).

The [credit terms](https://typesafe.ai/legal/mca) describe both purchased and
promotional credits, with automatic refills requiring opt-in. An account that has
not requested a payment method may have promotional access; this account's balance
and billing activation have not been verified. Successful inference alone does
not establish free or unlimited use. The first eight requests reported 11,015
input tokens, equivalent to approximately $0.000463 at the published price; this
is a list-price calculation, not an observed account charge.

The user subsequently confirmed a $5 trial credit. The
[contrast pilot](../reports/CONTRAST_PILOT.md) records additional usage and
preservation checks; the API balance itself has not been queried.

The current runner is sequential and stops on 429 or 529, saving completed
results. It does not repeatedly hammer a throttled service. Larger future runs
should add bounded retries that honor Retry-After, back off and preserve completed
requests. Do not use load testing to discover an account's throttle threshold.

## Prepare and run

Run commands from the project directory. Choose a new output directory each time;
existing results cannot be overwritten.

```sh
# Offline preparation: saves the exact requests, makes no API call.
PYTHONPATH=. .venv/bin/python -m experiments.jev_replay \
  --output reports/my-jev-preparation

# Eight-case access check and initial behavior comparison.
PYTHONPATH=. .venv/bin/python -m experiments.jev_replay \
  --run --output reports/my-jev-smoke

# Full frozen diagnostic, when intentionally requested.
PYTHONPATH=. .venv/bin/python -m experiments.jev_replay \
  --run --all --output reports/my-jev-full
```

Use repeated `--case-id` arguments for an explicit selection. The default model
is pinned to `jev-1.13.0`; `--model` can name another supported fixed version. The returned model
identifier is recorded, and a change during a run fails the comparison rather
than silently mixing versions. Alias stability across dates is not assumed.

Requests run sequentially, once per case. A failed request stops the run, retains
completed results, and records its error type and HTTP status when available.
There are no automatic retries, so an ambiguous network failure cannot trigger
unattended duplicate requests. Inspect partial results before rerunning a request
that might already have been processed. Live requests consume account usage;
reported tokens are preserved, but this runner does not estimate billing.

## Persistent response archive and repeated samples

Every live attempt is saved under `data/jev-responses/`, independently of its
per-run report. This is persistent experiment data, not a disposable cache. The
archive never evicts or overwrites a prior sample. Use `--archive` to choose
another persistent location.

The client writes the exact serialized request and hash before sending, then
streams the response into a synced `response.body`. It atomically publishes
`capture.json` before parsing, and adds a separate `validation.json` afterward.
These records retain timestamps, elapsed time, HTTP status, safe headers,
requested and returned model, and parsed usage when available. Literal credential
bytes are redacted, including matches across network chunk boundaries. This
byte-level redaction does not decode arbitrary escaped or encoded credential echoes.
HTTP failures, malformed JSON, distributions invalid under the current declared
policy, and interrupted reads remain archived and cannot satisfy reuse. The full received body is retained
even when it exceeds the 2 MB parsing limit. A hard process kill can leave an
incomplete attempt with only its request, headers and received body; it cannot
be reused. No program can preserve bytes that never arrived from the service.

A pinned request reuses the earliest valid archived response with identical
input and the same returned fixed model version. Every field other than the
requested model must match after deterministic JSON serialization. A historical
`jev-latest` request can satisfy a `jev-1.13.0` request only when its returned
version is exactly `jev-1.13.0`; its original alias and input remain preserved.
Requests for an alias always make a new call. Cache reads recheck the stored
hashes and response schema under the current validation policy. They never
select a sample by its score. If a complete HTTP 200 response rejected under an
older policy passes current validation, its original verdict stays unchanged;
an immutable `revalidation-v3.json` records the current verdict. Earlier
`validation.json` and `revalidation-v2.json` files remain unchanged. Replay metadata
exposes the original status, current status and validation policy version.

For independent stochastic repeats, pass `--fresh`; each call receives its own
archive directory even when its input is identical:

```sh
PYTHONPATH=. .venv/bin/python -m experiments.jev_replay \
  --run --fresh --output reports/my-jev-independent-repeat

# Offline, idempotent import of the first eight paid responses.
PYTHONPATH=. .venv/bin/python -m experiments.jev_archive \
  --import-run reports/jev-smoke-live-v1
```

The legacy import keeps the original request and response JSONL lines byte for
byte, along with source paths and line numbers. The old runner saved parsed JSON,
so the import explicitly marks original HTTP bytes and headers as unavailable;
its `response.body` reconstructs the saved JSON without claiming wire fidelity.

## Results and interpretation

- `requests.jsonl`: exact request bodies with local case IDs outside the bodies.
- `responses.jsonl`: returned distributions/confidence, returned model, usage,
  receive timestamps and client-observed elapsed time. Literal credential bytes
  unexpectedly echoed by the service are redacted. Archive links and cache-hit flags identify
  the original observation when reused.
- `predictions.json` and `relations.json`: question correctness, probability
  quality, evidence-change pairs, same-record question contrasts and invariance.
- `report.json`: status, data/source hashes, token totals, timing summary and
  comparisons against the saved base/trained predictions when the suite hashes
  match. Disagreement details include both probability vectors. Cache-hit and
  live-attempt counts are separate. `usage` includes reused observations;
  `new_usage` includes validated fresh responses only. Neither field establishes
  account charges, and usage from failed responses remains in their archives.
  `normalized_probability_vectors` counts Choice and Score vectors adjusted under the
  declared probability policy.

HTTP success and schema validity do not establish answer correctness. A failed
or partial replay is never reported as a completed benchmark. The offline tests
exercise recorded-format responses, not the live account's authentication.

Validation policy 2 introduced handling for observed Choice probability rounding.
Current policy 3 retains that rule and applies it to Score distributions. It
first requires exactly the requested classes and finite values in `[0, 1]`.
When the reported sum differs from 1 by more than `1e-6`, it accepts a positive
sum only within `0.020000001` of 1, and only when every value lies on a `0.01`
grid within `1e-9`. Accepted vectors are divided by their reported sum for
probability metrics. Predictions retain `raw_probabilities`,
`reported_probability_sum`, and `probability_normalized: true`; the raw response
and Choice confidence remain unchanged. The returned choice must still match a
maximum of the raw vector. Local-model probability validation remains strict.
This formatting policy does not consult gold labels or model correctness.

Score responses must contain exactly the string keys `"0"` through `"k-1"`,
where `k` is the number of declared criteria, and a legend matching the indexed
rubric. Rubric descriptions may be strings, objects, or lists. The returned
`score` must be finite and within `[0, k-1]`; confidence must be finite and within
`[0, 1]`. Both values remain unchanged in the raw response, and predictions retain
the scalar as `returned_score` and confidence as returned. Missing inputs and
Unknown categories are never converted into a probability of `0.5`.

Score mean absolute error uses the mean of the full validated probability vector,
stored as `score_mean`, against the authored integer target. The normalized error
divides by the rubric range. This supplements categorical accuracy, NLL and Brier
score; the reported scalar is not substituted for the distribution-derived mean.
The mean sanity check permits a formatting tolerance of `0.005001`, plus
`0.005 * sum(abs(i - score_mean))` when every reported bin is on the cent grid.
The first term covers scalar cent rounding and numeric tolerance. The second is
a conservative propagation bound for bins independently rounded by at most
`0.005`: because latent probabilities sum to one, the mean difference is
`sum((i - score_mean) * rounding_error_i)`. Each prediction records the tolerance.
It is a declared rounding allowance, not evidence of calibrated confidence or a
guarantee about the service's internal numeric precision.

The first 100-case pilot stopped after 71 completed cases because the 72nd
response reported a Choice sum of `0.99`. Policy 2 revalidated that same archived
response, retaining its original rejection record and avoiding another paid call.

Jev does not expose raw logits through this API. Negative log likelihood is
therefore computed from its returned probabilities after the declared rounding
normalization when needed, with a declared floor of
`1e-12`; exact zero probability on the expected answer is separately counted.
The same floor is applied to saved local probabilities in the comparison. These
losses differ from the earlier uncapped local losses computed directly from
logits. Brier score sums over the full distribution. Tied maxima are ambiguous
rather than silently counted as correct. Returned Choice confidence is preserved
but is not treated as probability of correctness.

Live latency includes connection setup, network, service time and durable
capture. Cached replay latency measures local retrieval and validation, and is
identified separately in each response. Neither is comparable to local GPU
kernel time without those measurement differences. The small, inspected synthetic suite measures particular
behaviors, not general production accuracy or Jev's unpublished full benchmark.

Prepared artifact: eight-case request set.
Its report is explicitly marked
`prepared-not-sent`.
