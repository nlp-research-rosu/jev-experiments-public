# AI-text detection with Jev

Detects AI-written text by asking Jev 1.13 eight typed questions about each chunk of text in **one**
request, then combining four of the answers with a fitted logistic formula. This is the detection
core of the Slop Alarm Chrome extension, copied here so the benchmark can be rerun on its own.
Results and discussion: `reports/AI_TEXT_DETECTION.md`.

This folder is a small Node/TypeScript project inside a Python repository. It needs Node 20 or later.

## Run it

```bash
cd experiments/ai_text_detection
npm install
```

Reproduce every table from the committed Jev responses. No API key and no requests are needed:

```bash
npm run eval
```

```bash
npm run fit
```

`npm run eval` reads `data/ai-text-detection-v1`, finds each chunk's response in
`reports/ai-text-detection-v1/answers.json`, and rewrites `results.csv`, `chunks.csv` and the
printed report. `npm run fit` refits the weights offline with leave-one-group-out validation.

To query Jev live, put a key in `.env` (gitignored):

```
TYPESAFE_API_KEY=...
```

Then either score new samples (only uncached chunks cost anything), or re-ask everything into a
separate folder so the committed responses stay untouched:

```bash
npm run eval -- --refresh --out reports/ai-text-detection-v1/rerun-YYYY-MM-DD
```

A full refresh is 123 requests, about 215,000 input tokens, roughly $0.009 at $0.042 per million.

Other runs in the report:

```bash
npm run eval -- data/ai-text-detection-v1/ablations/markdown-stripped --out reports/ai-text-detection-v1/ablation-markdown-stripped
```

```bash
node eval/fit.mjs --features ai_written,tells
```

Relative paths in these flags resolve from the repository root.

## Layout

| Path | Purpose |
|---|---|
| `core/questions.ts` | The eight Jev questions and the pinned model ids |
| `core/detect.ts` | One request per chunk, answer validation, retry, error mapping |
| `core/scoring.ts` | The fitted formula, verdict bands and aggregation over chunks |
| `core/chunking.ts` | Splits text into chunks of at most 300 words |
| `eval/run.ts` | Scores the labelled samples and prints the metrics |
| `eval/fit.mjs` | Fits the weights per chunk, validates leave-one-group-out |
| `test/` | Unit tests: `npm test` |

## Method in one paragraph

Each sample is split into chunks of at most 300 words. Jev receives the chunk as `state` and eight
questions: a direct "was this AI-generated?" Noul; a 4-level Score for how saturated the text is
with stock AI writing habits; five Nouls for families of habits (staged emphasis, rhythm by rule,
inflated language, decorative formatting, chatbot residue); and a Noul for first-hand specifics. The
chunk probability is `sigmoid(bias + w·[ai_written, tells/3, staging, specifics])`, with weights
fitted by L2 logistic regression. The remaining questions only explain the verdict to a user.
A sample's probability is the word-weighted mean of its chunk probabilities. The habit taxonomy
follows github.com/blader/humanizer, which distils Wikipedia's "Signs of AI writing".
