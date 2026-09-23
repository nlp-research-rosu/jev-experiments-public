#!/usr/bin/env node
/**
 * Eval harness: `npm run eval -- <dir>` (dir defaults to the repo's `eval/samples`).
 *
 * Runs the real, byok pipeline against labelled samples: reads `<dir>/human/*.txt` and
 * `<dir>/ai/*.txt`, splits each on blank lines into blocks, chunks them with `chunkBlocks` (same
 * as production), judges each chunk with ONE `askJev` call (all 8 questions together), scores it
 * with `scoreChunk`, and combines a sample's chunks with `aggregate()` — exactly what the
 * extension does, just against `packages/core` directly instead of through a browser.
 *
 * Every validated raw answer is cached at `<dir>/.cache/answers.json`, keyed by
 * sha256(model + sha256(JSON of QUESTIONS) + chunk text) (see src/cache.ts), so a rerun with
 * unchanged questions makes zero API calls. Weights and bands in `packages/core/src/scoring.ts`
 * can change freely: scoring is recomputed from the cached raw answers every run. `--refresh`
 * ignores the cache. `tools/eval/fit.mjs` reads `<dir>/.cache/chunks.csv` to refit those weights
 * offline; this harness only produces the CSVs.
 *
 * Flags:
 *   --refresh              ignore the raw-answer cache, re-ask Jev for every chunk
 *   --json <path>          also write the full report as JSON
 *   --provider <name>      "typesafe" or "openrouter" (default: whichever key is set; see below)
 *
 * See eval/README.md.
 */
import { dirname as pathDirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import {
  JEV_PROVIDERS,
  JevError,
  LIMITS,
  QUESTIONS,
  aggregate,
  askJev,
  chunkBlocks,
  countWords,
  scoreChunk,
  tooShortChunk,
} from '../core/index.js';
import type { Answers, ChunkResult, JevErrorCode, JevProvider, ProviderConfig, TextChunk } from '../core/index.js';
import { type AnswerCache, cacheKeyFor, loadCache, questionsHash, saveCache } from './cache.js';
import { type ChunkRow, type RawSignalKey, type RawSignals, type SampleRow, RAW_SIGNAL_KEYS, humanSourceOf, loadAiStyles, writeChunksCsv, writeResultsCsv } from './csv.js';
import { type GroupBreakdown, auc, breakdown, clopperPearsonUpper, confusionMatrix, mean, round, stddev } from './metrics.js';

const COST_PER_MILLION_INPUT_TOKENS = 0.042;
const CONCURRENCY = 4;
const RETRY_FALLBACK_MS = 400;
const CACHE_FLUSH_EVERY = 20;
const VERDICTS = ['likely_human', 'unclear', 'likely_ai', 'very_likely_ai', 'too_short'] as const;

const RETRYABLE: JevErrorCode[] = ['rate_limited', 'upstream', 'network'];
const FATAL: JevErrorCode[] = ['missing_key', 'invalid_key', 'no_credits'];

// ---------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------

interface Sample {
  label: 'human' | 'ai';
  file: string;
  text: string;
}

/**
 * Git on Windows may check samples out with CRLF line endings. The response cache is keyed by the
 * exact chunk text, so without this every cached response would miss and the scores would drift.
 */
export function normalizeNewlines(text: string): string {
  return text.replace(/\r\n?/g, '\n');
}

function loadSamples(dir: string): Sample[] {
  const samples: Sample[] = [];
  for (const label of ['human', 'ai'] as const) {
    const labelDir = join(dir, label);
    let files: string[] = [];
    try {
      files = readdirSync(labelDir).filter(
        (f) => f.toLowerCase().endsWith('.txt') && !/^readme/i.test(f) && !/^sources/i.test(f),
      );
    } catch {
      files = [];
    }
    for (const file of files.sort()) samples.push({ label, file, text: normalizeNewlines(readFileSync(join(labelDir, file), 'utf8')) });
  }
  return samples;
}

/** Splits on blank lines, like the extension's block extraction feeds chunkBlocks in production. */
function splitBlocks(text: string): string[] {
  return text
    .split(/\r?\n\s*\r?\n/)
    .map((b) => b.trim())
    .filter(Boolean);
}

interface SampleChunks {
  sample: Sample;
  chunks: TextChunk[];
}

function chunksForSample(sample: Sample): SampleChunks {
  const blocks = splitBlocks(sample.text);
  let chunks = chunkBlocks(blocks.length > 0 ? blocks : [sample.text]);
  if (chunks.length === 0 && sample.text.trim()) {
    // chunkBlocks dropped everything (e.g. whitespace-only blocks); fall back to the raw text as
    // a single chunk so the sample still gets a verdict instead of silently vanishing.
    chunks = [{ id: 'c0', text: sample.text.trim(), words: countWords(sample.text), blockIndexes: [] }];
  }
  return { sample, chunks };
}

// ---------------------------------------------------------------------------
// Scoring one chunk: mirrors detect.ts's per-chunk path (length check, cache, ONE askJev call,
// one polite retry, scoreChunk) but with the eval's own answer cache (see src/cache.ts), which is
// keyed on the questions rather than on SCORING_VERSION so a weight/band refit never invalidates it.
// ---------------------------------------------------------------------------

interface Task {
  sampleIndex: number;
  chunk: TextChunk;
}

interface ChunkOutcome {
  chunkResult: ChunkResult;
  raw: Answers | null;
  words: number;
  fromCache: boolean;
  freshTokens: number;
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

async function askWithRetry(config: ProviderConfig, text: string): Promise<{ answers: Answers; inputTokens: number }> {
  try {
    return await askJev(config, text);
  } catch (err) {
    if (!(err instanceof JevError) || !RETRYABLE.includes(err.code)) throw err;
    await sleep(err.retryAfterMs ?? RETRY_FALLBACK_MS);
    return askJev(config, text);
  }
}

async function scoreOneChunk(
  chunk: TextChunk,
  config: ProviderConfig,
  model: string,
  qHash: string,
  cache: AnswerCache,
): Promise<ChunkOutcome> {
  const text = chunk.text.slice(0, LIMITS.maxChunkChars);
  const words = countWords(text);
  if (words < LIMITS.minWords) {
    return { chunkResult: tooShortChunk(chunk.id, words), raw: null, words, fromCache: false, freshTokens: 0 };
  }

  const key = cacheKeyFor(model, qHash, text);
  const cached = cache.get(key);
  if (cached) {
    return { chunkResult: scoreChunk(chunk.id, words, cached.answers), raw: cached.answers, words, fromCache: true, freshTokens: 0 };
  }

  const { answers, inputTokens } = await askWithRetry(config, text);
  cache.set(key, { answers, inputTokens });
  return { chunkResult: scoreChunk(chunk.id, words, answers), raw: answers, words, fromCache: false, freshTokens: inputTokens };
}

/** Runs `worker` over `items` with bounded concurrency, preserving order in the result array. */
async function runPool<T, R>(items: T[], concurrency: number, worker: (item: T) => Promise<R>): Promise<(R | { error: unknown })[]> {
  const results: (R | { error: unknown })[] = new Array(items.length);
  let next = 0;
  let fatal: unknown;
  const lane = async (): Promise<void> => {
    while (next < items.length) {
      if (fatal) return;
      const i = next++;
      try {
        results[i] = await worker(items[i] as T);
      } catch (err) {
        if (err instanceof JevError && FATAL.includes(err.code)) fatal = err;
        results[i] = { error: err };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, lane));
  if (fatal) throw fatal;
  return results;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

interface ScoreRunOutcome {
  rows: SampleRow[];
  chunkRows: ChunkRow[];
  freshTokens: number;
  requests: number;
  cacheHits: number;
  droppedChunks: number;
}

async function scoreSamples(
  sampleChunksList: SampleChunks[],
  config: ProviderConfig,
  model: string,
  qHash: string,
  cache: AnswerCache,
  cachePath: string,
): Promise<ScoreRunOutcome> {
  const tasks: Task[] = [];
  sampleChunksList.forEach((sc, sampleIndex) => {
    for (const chunk of sc.chunks) tasks.push({ sampleIndex, chunk });
  });

  let freshSinceFlush = 0;
  const worker = async (task: Task): Promise<ChunkOutcome> => {
    const outcome = await scoreOneChunk(task.chunk, config, model, qHash, cache);
    if (!outcome.fromCache && outcome.raw) {
      freshSinceFlush++;
      if (freshSinceFlush >= CACHE_FLUSH_EVERY) {
        freshSinceFlush = 0;
        saveCache(cachePath, cache);
      }
    }
    return outcome;
  };

  let taskResults: (ChunkOutcome | { error: unknown })[];
  try {
    taskResults = await runPool(tasks, CONCURRENCY, worker);
  } finally {
    saveCache(cachePath, cache); // always persist whatever we learned, even on a fatal abort
  }

  const rows: SampleRow[] = [];
  const chunkRows: ChunkRow[] = [];
  let freshTokens = 0;
  let requests = 0;
  let cacheHits = 0;
  let droppedChunks = 0;
  let cursor = 0;

  for (const { sample, chunks } of sampleChunksList) {
    const outcomes: ChunkOutcome[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const r = taskResults[cursor++];
      if (r && 'error' in r) {
        droppedChunks++;
        console.error(`  [${sample.label}] ${sample.file}: chunk failed, dropping it (${(r.error as Error)?.message ?? r.error})`);
      } else if (r) {
        outcomes.push(r);
      }
    }

    for (const o of outcomes) {
      if (!o.raw) continue; // too-short chunk: no answers, not a request or a cache hit
      if (o.fromCache) cacheHits++;
      else {
        requests++;
        freshTokens += o.freshTokens;
      }
      chunkRows.push({
        file: sample.file,
        label: sample.label,
        chunkId: o.chunkResult.id,
        words: o.words,
        probability: o.chunkResult.probability,
        ai_written: o.raw.ai_written.noul,
        tells: o.raw.tells.score,
        staging: o.raw.staging.noul,
        rhythm: o.raw.rhythm.noul,
        inflation: o.raw.inflation.noul,
        formatting: o.raw.formatting.noul,
        chat_residue: o.raw.chat_residue.noul,
        specifics: o.raw.specifics.noul,
      });
    }

    const chunkResults = outcomes.map((o) => o.chunkResult);
    const aggregated = aggregate(chunkResults);
    const judged = outcomes.filter((o): o is ChunkOutcome & { raw: Answers } => o.raw !== null);
    rows.push({
      file: sample.file,
      label: sample.label,
      words: aggregated.words,
      chunkCount: chunks.length,
      judgedChunkCount: judged.length,
      probability: aggregated.probability,
      verdict: aggregated.verdict,
      confidence: aggregated.confidence,
      signals: weightedSignals(judged),
    });
  }

  return { rows, chunkRows, freshTokens, requests, cacheHits, droppedChunks };
}

/** Word-weighted mean across a sample's judged chunks, raw scale (tells stays 0..3). */
function weightedSignals(judged: { raw: Answers; words: number }[]): RawSignals | null {
  if (judged.length === 0) return null;
  const totalWords = judged.reduce((s, e) => s + e.words, 0) || judged.length;
  const wmean = (get: (e: (typeof judged)[number]) => number) => round(judged.reduce((s, e) => s + get(e) * e.words, 0) / totalWords);
  return {
    ai_written: wmean((e) => e.raw.ai_written.noul),
    tells: wmean((e) => e.raw.tells.score),
    staging: wmean((e) => e.raw.staging.noul),
    rhythm: wmean((e) => e.raw.rhythm.noul),
    inflation: wmean((e) => e.raw.inflation.noul),
    formatting: wmean((e) => e.raw.formatting.noul),
    chat_residue: wmean((e) => e.raw.chat_residue.noul),
    specifics: wmean((e) => e.raw.specifics.noul),
  };
}

interface Report {
  dir: string;
  provider: string;
  model: string;
  sampleCounts: { human: number; ai: number };
  perSignalByLabel: Record<'human' | 'ai', Record<RawSignalKey, { mean: number; sd: number; n: number }>>;
  aucOverall: number;
  aucPerSignal: Record<RawSignalKey, number>;
  confusionMatrix: Record<'human' | 'ai', Record<string, number>>;
  falsePositiveRateHuman: number;
  falsePositiveRateHumanUpper95: number;
  falseNegativeRateAi: number;
  misclassified: SampleRow[];
  unclear: SampleRow[];
  byAiStyle: GroupBreakdown[];
  byHumanSource: GroupBreakdown[];
  tokens: { requests: number; cacheHits: number; droppedChunks: number; inputTokens: number; estimatedCostUsd: number };
}

function buildReport(dir: string, provider: string, model: string, rows: SampleRow[], run: ScoreRunOutcome, aiStyles: Map<string, string>): Report {
  const humanRows = rows.filter((r) => r.label === 'human');
  const aiRows = rows.filter((r) => r.label === 'ai');

  const perSignalByLabel = {} as Report['perSignalByLabel'];
  for (const label of ['human', 'ai'] as const) {
    const subset = rows.filter((r) => r.label === label && r.signals);
    const bySignal = {} as Record<RawSignalKey, { mean: number; sd: number; n: number }>;
    for (const key of RAW_SIGNAL_KEYS) {
      const values = subset.map((r) => r.signals![key]);
      bySignal[key] = { mean: round(mean(values)), sd: round(stddev(values)), n: values.length };
    }
    perSignalByLabel[label] = bySignal;
  }

  const aucPerSignal = {} as Record<RawSignalKey, number>;
  for (const key of RAW_SIGNAL_KEYS) aucPerSignal[key] = round(auc(rows.map((r) => ({ label: r.label, score: r.signals?.[key] }))));

  const humanFlagged = humanRows.filter((r) => r.verdict === 'likely_ai' || r.verdict === 'very_likely_ai').length;
  const fpr = humanRows.length ? humanFlagged / humanRows.length : NaN;
  const fprUpper = humanRows.length ? clopperPearsonUpper(humanFlagged, humanRows.length) : NaN;
  const fnr = aiRows.length ? aiRows.filter((r) => r.verdict === 'likely_human').length / aiRows.length : NaN;

  const misclassified = rows.filter(
    (r) => (r.label === 'human' && (r.verdict === 'likely_ai' || r.verdict === 'very_likely_ai')) || (r.label === 'ai' && r.verdict === 'likely_human'),
  );
  const unclear = rows.filter((r) => r.verdict === 'unclear');

  const byAiStyle = breakdown(
    aiRows.map((r) => ({
      key: aiStyles.get(r.file) ?? 'stereotyped',
      probability: r.probability,
      flagged: r.verdict === 'likely_ai' || r.verdict === 'very_likely_ai',
    })),
  );
  const byHumanSource = breakdown(
    humanRows.map((r) => ({ key: humanSourceOf(r.file), probability: r.probability, flagged: r.verdict === 'likely_human' })),
  );

  return {
    dir,
    provider,
    model,
    sampleCounts: { human: humanRows.length, ai: aiRows.length },
    perSignalByLabel,
    aucOverall: round(auc(rows.map((r) => ({ label: r.label, score: r.probability })))),
    aucPerSignal,
    confusionMatrix: confusionMatrix(rows, VERDICTS),
    falsePositiveRateHuman: round(fpr),
    falsePositiveRateHumanUpper95: round(fprUpper),
    falseNegativeRateAi: round(fnr),
    misclassified,
    unclear,
    byAiStyle,
    byHumanSource,
    tokens: {
      requests: run.requests,
      cacheHits: run.cacheHits,
      droppedChunks: run.droppedChunks,
      inputTokens: run.freshTokens,
      // Not run through metrics' round() (3 decimal places, meant for 0..1 probabilities): a
      // sub-cent run would round straight to $0.000.
      estimatedCostUsd: Math.round((run.freshTokens / 1_000_000) * COST_PER_MILLION_INPUT_TOKENS * 1e8) / 1e8,
    },
  };
}

function fmt(n: number): string {
  return Number.isNaN(n) ? 'n/a' : n.toFixed(3);
}
function fmtPct(n: number): string {
  return Number.isNaN(n) ? 'n/a' : `${(n * 100).toFixed(1)}%`;
}

function printReport(report: Report): void {
  console.log(`\n=== Samples ===`);
  console.log(`human: ${report.sampleCounts.human}   ai: ${report.sampleCounts.ai}   (dir: ${report.dir}, provider: ${report.provider}, model: ${report.model})`);
  if (report.sampleCounts.human === 0) console.warn('No human samples: false-positive rate, per-signal stats and the human breakdown cannot be computed.');
  if (report.sampleCounts.ai === 0) console.warn('No AI samples: false-negative rate, per-signal stats and the AI-style breakdown cannot be computed.');

  console.log(`\n=== Per-signal mean (sd, n) by label ===`);
  for (const label of ['human', 'ai'] as const) {
    const bySignal = report.perSignalByLabel[label];
    const parts = RAW_SIGNAL_KEYS.map((k) => `${k}=${fmt(bySignal[k].mean)}(${fmt(bySignal[k].sd)},n=${bySignal[k].n})`);
    console.log(`${label.padEnd(6)} ${parts.join('  ')}`);
  }

  console.log(`\n=== ROC AUC (ai = positive class) ===`);
  console.log(`overall (final probability): ${fmt(report.aucOverall)}`);
  for (const key of RAW_SIGNAL_KEYS) console.log(`  ${key.padEnd(12)} ${fmt(report.aucPerSignal[key])}`);

  console.log(`\n=== Confusion matrix (verdict bands) ===`);
  for (const label of ['human', 'ai'] as const) {
    const counts = report.confusionMatrix[label];
    console.log(`${label.padEnd(6)} ${VERDICTS.map((b) => `${b}=${counts[b]}`).join('  ')}`);
  }

  console.log(`\n=== The key numbers ===`);
  console.log(
    `Human false-positive rate (likely_ai or above): ${
      report.sampleCounts.human
        ? `${fmtPct(report.falsePositiveRateHuman)} (${report.sampleCounts.human} human samples), one-sided 95% upper bound ${fmtPct(report.falsePositiveRateHumanUpper95)}`
        : 'n/a (no human samples)'
    }`,
  );
  console.log(
    `AI false-negative rate (read as likely_human): ${
      report.sampleCounts.ai ? `${fmtPct(report.falseNegativeRateAi)} (${report.sampleCounts.ai} ai samples)` : 'n/a (no ai samples)'
    }`,
  );

  if (report.byAiStyle.length > 0) {
    console.log(`\n=== By AI prompt style (ai/SOURCES.tsv; unlisted = stereotyped) ===`);
    for (const s of report.byAiStyle) console.log(`${s.key.padEnd(18)} n=${s.count}   meanP=${fmt(s.meanProbability)}   flagged(likely_ai+)=${fmtPct(s.share)}`);
  }
  if (report.byHumanSource.length > 0) {
    console.log(`\n=== By human source (filename prefix) ===`);
    for (const s of report.byHumanSource) console.log(`${s.key.padEnd(18)} n=${s.count}   meanP=${fmt(s.meanProbability)}   likely_human=${fmtPct(s.share)}`);
  }

  if (report.misclassified.length > 0) {
    console.log(`\n=== Misclassified (${report.misclassified.length}) ===`);
    for (const r of report.misclassified) console.log(`  [${r.label}] ${r.file}  p=${r.probability ?? 'n/a'}  verdict=${r.verdict}  signals=${JSON.stringify(r.signals)}`);
  }
  if (report.unclear.length > 0) {
    console.log(`\n=== Unclear (${report.unclear.length}) ===`);
    for (const r of report.unclear) console.log(`  [${r.label}] ${r.file}  p=${r.probability ?? 'n/a'}  signals=${JSON.stringify(r.signals)}`);
  }

  console.log(`\n=== Requests and cost (this run) ===`);
  console.log(`requests: ${report.tokens.requests}   cache hits: ${report.tokens.cacheHits}   dropped chunks: ${report.tokens.droppedChunks}`);
  console.log(
    `input tokens: ${report.tokens.inputTokens}   estimated cost: $${report.tokens.estimatedCostUsd.toFixed(6)} (at $${COST_PER_MILLION_INPUT_TOKENS}/million; cache hits are free)`,
  );
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface Args {
  dir: string;
  outDir: string;
  refresh: boolean;
  jsonPath: string | undefined;
  provider: JevProvider | undefined;
}

function here(): string {
  return pathDirname(fileURLToPath(import.meta.url));
}

function repoRoot(): string {
  // experiments/ai_text_detection/eval/run.ts -> eval -> ai_text_detection -> experiments -> repo root
  return resolve(here(), '../../..');
}

function parseArgs(argv: string[]): Args {
  let dirArg: string | undefined;
  let refresh = false;
  let jsonPath: string | undefined;
  let outArg: string | undefined;
  let provider: JevProvider | undefined;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (a === '--refresh') refresh = true;
    else if (a === '--json') jsonPath = argv[++i];
    else if (a === '--out') outArg = argv[++i];
    else if (a === '--provider') {
      const p = argv[++i];
      if (p !== 'typesafe' && p !== 'openrouter') throw new Error(`--provider must be "typesafe" or "openrouter", got ${JSON.stringify(p)}`);
      provider = p;
    } else if (!a.startsWith('--') && dirArg === undefined) dirArg = a;
  }
  const raw = dirArg ?? join(repoRoot(), 'data/ai-text-detection-v1');
  // Relative paths resolve against the repo root, not process.cwd().
  const dir = isAbsolute(raw) ? raw : resolve(repoRoot(), raw);
  const rawOut = outArg ?? join(repoRoot(), 'reports/ai-text-detection-v1');
  const outDir = isAbsolute(rawOut) ? rawOut : resolve(repoRoot(), rawOut);
  const json = jsonPath === undefined ? undefined : isAbsolute(jsonPath) ? jsonPath : resolve(repoRoot(), jsonPath);
  return { dir, outDir, refresh, jsonPath: json, provider };
}

const ENV_KEY_NAME: Record<JevProvider, string> = { typesafe: 'TYPESAFE_API_KEY', openrouter: 'OPENROUTER_API_KEY' };

function resolveProvider(argProvider: JevProvider | undefined): ProviderConfig {
  if (argProvider) {
    const apiKey = process.env[ENV_KEY_NAME[argProvider]];
    if (!apiKey) throw new Error(`--provider ${argProvider} was given but ${ENV_KEY_NAME[argProvider]} is not set (add it to experiments/ai_text_detection/.env).`);
    return { provider: argProvider, apiKey };
  }
  if (process.env.TYPESAFE_API_KEY) return { provider: 'typesafe', apiKey: process.env.TYPESAFE_API_KEY };
  if (process.env.OPENROUTER_API_KEY) return { provider: 'openrouter', apiKey: process.env.OPENROUTER_API_KEY };
  // No key: fine for reproducing results from the committed responses in answers.json. Any chunk
  // that is not cached will fail with missing_key and the run stops with a clear message.
  console.warn('No TYPESAFE_API_KEY or OPENROUTER_API_KEY set: using cached responses only. Add a key to experiments/ai_text_detection/.env to query Jev.');
  return { provider: 'typesafe', apiKey: '' };
}

async function main(): Promise<void> {
  const { dir, outDir, refresh, jsonPath, provider: argProvider } = parseArgs(process.argv.slice(2));

  const samples = loadSamples(dir);
  if (samples.length === 0) {
    console.error(`No samples found under ${dir}. Expected ${dir}/human/*.txt and ${dir}/ai/*.txt`);
    process.exitCode = 1;
    return;
  }

  const config = resolveProvider(argProvider);
  const model = config.model ?? JEV_PROVIDERS[config.provider].model;
  console.log(`Scoring ${samples.length} samples from ${dir} with provider=${config.provider} model=${model}...`);

  const qHash = questionsHash();
  const cachePath = join(outDir, 'answers.json');
  const cache: AnswerCache = refresh ? new Map() : loadCache(cachePath);
  const sampleChunksList = samples.map(chunksForSample);
  const aiStyles = loadAiStyles(join(dir, 'ai', 'SOURCES.tsv'));

  const outcome = await scoreSamples(sampleChunksList, config, model, qHash, cache, cachePath);

  const report = buildReport(dir, config.provider, model, outcome.rows, outcome, aiStyles);
  printReport(report);
  writeResultsCsv(outcome.rows, join(outDir, 'results.csv'));
  console.log(`\nPer-sample results written to ${join(outDir, 'results.csv')}`);
  // Per judged chunk, not per sample: production scores and gates per chunk, so fit.mjs needs
  // chunk-level rows rather than the sample-level word-weighted means in results.csv.
  writeChunksCsv(outcome.chunkRows, join(outDir, 'chunks.csv'));
  console.log(`Per-chunk results written to ${join(outDir, 'chunks.csv')}`);
  if (jsonPath) {
    writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`Full report written to ${jsonPath}`);
  }
}

// Only run the CLI when this file is the actual entry point (`tsx src/run.ts ...`), not when it's
// imported (e.g. by tests, to exercise the pure exports without a real API key).
const isDirectRun = (() => {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return fileURLToPath(import.meta.url) === resolve(entry);
  } catch {
    return false;
  }
})();

if (isDirectRun) {
  main().catch((err) => {
    if (err instanceof JevError) console.error(`\nJev error (${err.code}): ${err.message}`);
    else console.error(err);
    process.exitCode = 1;
  });
}
