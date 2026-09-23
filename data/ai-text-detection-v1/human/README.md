# Real human-written samples only

Every `.txt` file in this directory must be text fetched verbatim over HTTP from a source
where human authorship is certain because the text was **published before November 2022**
(before ChatGPT existed and LLM-written text entered general circulation). Nothing here may
be written, paraphrased, "cleaned up", or recalled from memory by an AI — that would make the
eval harness's false-positive rate meaningless in exactly the direction that matters most:
wrongly accusing a real person of using AI.

The only edits applied to any file are mechanical: whitespace collapsing, stripping citation
brackets (`[12]`, `[edit]`), decoding HTML entities, and (in a couple of cases, noted in
`SOURCES.tsv`) trimming trailing navigation/address boilerplate that the extractor didn't
catch. No wording is ever changed.

## How this directory is populated

Run, from the repo root:

```sh
node data/ai-text-detection-v1/collect-human.mjs
```

The script (plain Node 22+, built-in `fetch`, no dependencies) pulls from five source types,
each chosen because authorship and date are independently verifiable:

- **Wikipedia** — the article revision at or before 2019-12-31, fetched via the MediaWiki API
  (`action=query&prop=revisions&rvstart=2019-12-31...` to find the revision id, then
  `action=parse&oldid=...` for that revision's HTML). Encyclopedic, neutral, polished prose —
  a hard negative for detectors that key on "formal" writing.
- **Stack Exchange** — top-voted answers from 2012–2019 across several sites (cooking, travel,
  english, workplace, diy, academia, parenting, outdoors), via `api.stackexchange.com`, with
  code blocks stripped.
- **Project Gutenberg** — excerpts from public-domain essay collections (Emerson, Bacon,
  Thoreau), taken from the book body (never the license header).
- **US federal government web pages** — pinned to a 2019 snapshot via the Wayback Machine
  CDX API (`web.archive.org/cdx/search/cdx`), which is queried at runtime to find snapshots
  that actually exist rather than guessing a timestamp. Promotional-leaning formal prose
  (nps.gov, noaa.gov, etc.) — another hard negative.
- **Pre-2022 blogs** — org/institutional blogs (Mozilla, Wikimedia, Library of Congress, ...)
  discovered the same way via Wayback CDX, for informal first-person voice.

Re-running the script wipes and regenerates every `.txt` file plus `SOURCES.tsv`. Fetches
that fail (404, no revision in range, couldn't assemble 150–400 clean words) are **skipped,
never substituted** — see the console output for what was skipped and why. The Wayback
Machine in particular is sometimes temporarily offline; a full run may need to be retried, or
the CDX candidate lists in the script broadened, to make up any shortfall in the `govweb` and
`blog` categories.

After a run, spot-check a sample of files by eye for markup debris, nav junk, or anything that
isn't continuous prose, and delete/fix anything that slipped through (`SOURCES.tsv` should
then be hand-edited to match).

## Licensing note

Wikipedia and Stack Exchange text is CC BY-SA; it is kept here for internal detector
calibration only, with attribution (source URL, revision/snapshot date) recorded per-file in
`SOURCES.tsv`. Government pages are US Government Works (public domain). Gutenberg texts are
public domain. Blog posts are attributed per their own terms in `SOURCES.tsv`.
