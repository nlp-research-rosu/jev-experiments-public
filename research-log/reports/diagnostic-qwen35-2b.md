# Parallel decision experiment

Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`. GPU: NVIDIA GeForce RTX 4080.

Suite: **diagnostic**, 32 unique cases; 3 timing repetitions per case.
Quality metrics use the first repetition only. Timing excludes download/loading and includes prompt preparation, GPU execution, and answer assembly.

| Method | Field accuracy | Exact cases | Valid schema | Median ms | p95 ms | Brier |
|---|---:|---:|---:|---:|---:|---:|
| parallel | 78.8% | 31.2% | 100.0% | 206.3 | 230.1 | 0.2769 |
| json | 86.2% | 53.1% | 100.0% | 758.5 | 845.4 | — |

Median paired JSON/parallel latency ratio: **3.67×**.

## Cached versus independent reference

- double_charge: max probability difference 0.012336; choice disagreements 0.
- invoice_copy: max probability difference 0.008896; choice disagreements 0.
- refund_today: max probability difference 0.013449; choice disagreements 0.

## Errors

### parallel

- `double_charge` / `priority`: expected `normal`, got `urgent`.
- `invoice_copy` / `priority`: expected `normal`, got `urgent`.
- `billing_call` / `phone_call_requested`: expected `True`, got `False`.
- `billing_call` / `priority`: expected `normal`, got `urgent`.
- `staging_failure` / `priority`: expected `normal`, got `urgent`.
- `outage_bridge` / `production_outage`: expected `True`, got `False`.
- `resolved_outage` / `priority`: expected `normal`, got `urgent`.
- `account_call` / `phone_call_requested`: expected `True`, got `False`.
- `account_call` / `priority`: expected `normal`, got `urgent`.
- `change_email` / `priority`: expected `normal`, got `urgent`.
- `pricing` / `department`: expected `sales`, got `billing`.
- `pricing` / `priority`: expected `normal`, got `urgent`.
- `sales_today` / `department`: expected `sales`, got `billing`.
- `sales_call` / `department`: expected `sales`, got `billing`.
- `sales_call` / `phone_call_requested`: expected `True`, got `False`.
- `sales_call` / `priority`: expected `normal`, got `urgent`.
- `feature_question` / `department`: expected `sales`, got `billing`.
- `feature_question` / `priority`: expected `normal`, got `urgent`.
- `refund_no_call` / `refund_requested`: expected `True`, got `False`.
- `refund_no_call` / `priority`: expected `normal`, got `urgent`.
- `refund_and_call` / `priority`: expected `normal`, got `urgent`.
- `hypothetical_outage` / `production_outage`: expected `False`, got `True`.
- `hypothetical_outage` / `priority`: expected `normal`, got `urgent`.
- `test_outage_call` / `department`: expected `technical`, got `billing`.
- `test_outage_call` / `phone_call_requested`: expected `True`, got `False`.
- `test_outage_call` / `priority`: expected `normal`, got `urgent`.
- `restored_no_urgency` / `priority`: expected `normal`, got `urgent`.
- `login_not_global` / `priority`: expected `normal`, got `urgent`.
- `refund_policy_before_purchase` / `department`: expected `sales`, got `billing`.
- `refund_policy_before_purchase` / `priority`: expected `normal`, got `urgent`.
- `hypothetical_deadline` / `department`: expected `sales`, got `billing`.
- `hypothetical_deadline` / `priority`: expected `normal`, got `urgent`.
- `sales_call_today` / `phone_call_requested`: expected `True`, got `False`.
- `quoted_instruction` / `priority`: expected `normal`, got `urgent`.

### json

- `refund_today` / `priority`: expected `urgent`, got `normal`.
- `billing_call` / `department`: expected `billing`, got `sales`.
- `billing_call` / `phone_call_requested`: expected `True`, got `False`.
- `resolved_outage` / `production_outage`: expected `False`, got `True`.
- `resolved_outage` / `priority`: expected `normal`, got `urgent`.
- `account_call` / `phone_call_requested`: expected `True`, got `False`.
- `sales_today` / `priority`: expected `urgent`, got `normal`.
- `sales_call` / `phone_call_requested`: expected `True`, got `False`.
- `feature_question` / `department`: expected `sales`, got `account_access`.
- `negated_refund` / `refund_requested`: expected `False`, got `True`.
- `refund_no_call` / `refund_requested`: expected `True`, got `False`.
- `technical_deadline` / `production_outage`: expected `False`, got `True`.
- `hypothetical_outage` / `production_outage`: expected `False`, got `True`.
- `hypothetical_outage` / `priority`: expected `normal`, got `urgent`.
- `test_outage_call` / `production_outage`: expected `False`, got `True`.
- `test_outage_call` / `phone_call_requested`: expected `True`, got `False`.
- `test_outage_call` / `priority`: expected `normal`, got `urgent`.
- `refund_policy_before_purchase` / `department`: expected `sales`, got `billing`.
- `refund_policy_before_purchase` / `refund_requested`: expected `False`, got `True`.
- `sales_call_today` / `phone_call_requested`: expected `True`, got `False`.
- `sales_call_today` / `priority`: expected `urgent`, got `normal`.
- `quoted_instruction` / `department`: expected `billing`, got `sales`.

## Interpretation limits

The diagnostic set is small, synthetic, and hand-authored. Upstream scenarios have no gold labels. Neither establishes production quality or parity with Jev.
Candidate probabilities are normalized scores, not calibrated confidence. The Brier score is the sum over all classes, averaged over labeled fields; ECE uses ten equal-width confidence bins.
The parallel path uses verified single-token option codes and independently answers each field. JSON generation uses original answer values and can condition later fields on earlier ones, so these are different inference objectives and prompts.
The JSON baseline is greedy ordinary generation, without grammar constraints. This is not a comparison against an optimized structured-output serving engine.
Reported speed is for this implementation, model, hardware, input sizes, and current GPU load. All forward calls, including prefill, are counted in the raw report.
