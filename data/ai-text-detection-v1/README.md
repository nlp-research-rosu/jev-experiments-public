# AI-text detection dataset, v1

92 labelled English texts used to calibrate and evaluate AI-written-text detection with Jev.
See `reports/AI_TEXT_DETECTION.md` for results.

| Folder | Count | Label | Source |
|---|---:|---|---|
| `human/` | 44 | human | Fetched verbatim from sources published before November 2022 |
| `ai/` | 48 | ai | Written by Claude models on 2026-09-20 |
| `ablations/markdown-stripped/ai/` | 19 | ai | The AI samples that contain markdown, with the markup removed |

## Human samples

| Source prefix | Count | What it is |
|---|---:|---|
| `wikipedia-` | 15 | Article body paragraphs from the last revision on or before 2019-12-31 |
| `stackexchange-` | 20 | Top-voted answers from 2012 to 2019, eight sites, code blocks removed |
| `gutenberg-` | 8 | Excerpts from public-domain essays (Emerson, Bacon, Thoreau) |
| `govweb-` | 1 | A US National Park Service page, 2019 Wayback Machine snapshot |

Every file is text fetched over HTTP, never written or paraphrased by a model. The only edits are
mechanical: whitespace, citation brackets, HTML entities, and one trailing-boilerplate trim noted
in `human/SOURCES.tsv`. That file records the source URL, revision or snapshot date, license and
word count for each sample. `collect-human.mjs` rebuilds the folder (see `human/README.md`).

Word counts: 168 to 399, median 328.

## AI samples

- **8 stereotyped samples** (`01-` to `08-`): typical assistant output across genres, including one
  email with chatbot residue ("I hope this helps").
- **40 topic-matched samples** (`matched-<human file>`): for 40 of the human samples, a text on the
  same topic, in the same genre and of similar length. Matching removes topic as a shortcut.

`ai/SOURCES.tsv` records the matched human file and the imagined prompt style of each matched
sample:

| Style | Count | Meaning |
|---|---:|---|
| `default` | 24 | Plain assistant output |
| `engaging` | 9 | Written as if asked to "make it engaging" |
| `casual-disguised` | 7 | Written as if asked to "write casually, like a real person, avoid sounding like AI" |

About a third use light markdown (bold labels, bullets, headings), as assistants often do. No human
sample contains markdown, which is why the markdown-stripped ablation exists.

Word counts: 199 to 319. AI samples are shorter and more uniform in length than the human ones.

## Known gaps

- All AI text comes from one model family (Claude). ChatGPT, Gemini and open models are untested.
- No personal blogs, news writing, social media posts or non-native English writers on the human
  side. The Wayback Machine was offline during collection, which cut the blog and government
  categories short.
- English only.

## License

Wikipedia and Stack Exchange text is CC BY-SA, kept here for research evaluation with per-file
attribution in `human/SOURCES.tsv`. Gutenberg texts and the US government page are public domain.
See `THIRD_PARTY.md` at the repository root.
