/**
 * Fits the scoring formula in core/scoring.ts from the eval harness output.
 *
 *   npm run eval                     fills reports/ai-text-detection-v1/chunks.csv (answers are cached)
 *   npm run fit                           offline: prints validated accuracy and the weights to paste
 *
 * Model: p(ai) = sigmoid(bias + Σ w_i · x_i), every x_i in 0..1. L2 logistic regression, no deps.
 *
 * It mirrors production exactly: the model is fitted PER CHUNK, and a sample's probability is the
 * word-weighted mean of its chunk probabilities (what `aggregate()` in core does).
 *
 * Validation is leave-one-GROUP-out. A group is a sample file together with its topic-matched
 * twin from data/ai-text-detection-v1/ai/SOURCES.tsv, so neither a sibling chunk nor a twin text can leak.
 *
 * Flags: --features a,b,c   --l2 0.1   --bands 0.3,0.7   --csv <path>
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const csvPath = arg('--csv', path.join(here, '../../../reports/ai-text-detection-v1/chunks.csv'));
const SCORE_MAX = { tells: 3 };
const features = arg('--features', 'ai_written,tells,staging,specifics').split(',');
const l2 = Number(arg('--l2', '0.1'));
const [bandHuman, bandAi] = arg('--bands', '0.3,0.75').split(',').map(Number);

if (!fs.existsSync(csvPath)) {
  console.error(`Missing ${csvPath}. Run: npm run eval`);
  process.exit(1);
}
const [header, ...lines] = fs.readFileSync(csvPath, 'utf8').trim().split(/\r?\n/);
const cols = header.split(',');
const chunks = lines
  .map((line) => {
    const cells = line.split(',');
    const get = (name) => cells[cols.indexOf(name)];
    return {
      file: get('file'),
      y: get('label') === 'ai' ? 1 : 0,
      words: Number(get('words')),
      x: features.map((f) => Number(get(f)) / (SCORE_MAX[f] ?? 1)),
    };
  })
  .filter((r) => r.x.every(Number.isFinite) && r.words > 0);

// Group = human file + its matched AI twin.
const groupOf = new Map();
const sources = path.join(here, '../../../data/ai-text-detection-v1/ai/SOURCES.tsv');
if (fs.existsSync(sources)) {
  for (const row of fs.readFileSync(sources, 'utf8').trim().split(/\r?\n/).slice(1)) {
    const [aiFile, humanFile] = row.split('\t');
    if (aiFile && humanFile) groupOf.set(aiFile.trim(), humanFile.trim());
  }
}
for (const c of chunks) c.group = groupOf.get(c.file) ?? c.file;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const predict = (m, x) => sigmoid(m.bias + x.reduce((s, v, i) => s + m.w[i] * v, 0));

function fit(data) {
  const m = { bias: 0, w: features.map(() => 0) };
  // Each SAMPLE counts once (a long page must not outvote a short one), and classes are balanced.
  const perFile = new Map();
  for (const r of data) perFile.set(r.file, (perFile.get(r.file) ?? 0) + 1);
  const files = [...perFile.keys()];
  const nPos = files.filter((f) => data.find((r) => r.file === f).y === 1).length;
  const nNeg = files.length - nPos;
  const weightOf = (r) => (1 / perFile.get(r.file)) * (r.y === 1 ? files.length / (2 * nPos) : files.length / (2 * nNeg));
  const n = files.length;
  for (let iter = 0; iter < 4000; iter++) {
    let gb = 0;
    const gw = m.w.map((w) => (l2 * w) / n);
    for (const r of data) {
      const err = ((predict(m, r.x) - r.y) * weightOf(r)) / n;
      gb += err;
      r.x.forEach((v, i) => (gw[i] += err * v));
    }
    m.bias -= 0.5 * gb;
    m.w = m.w.map((w, i) => w - 0.5 * gw[i]);
  }
  return m;
}

/** Production aggregation: word-weighted mean of chunk probabilities. */
function sampleScores(scoredChunks) {
  const by = new Map();
  for (const c of scoredChunks) {
    const s = by.get(c.file) ?? { file: c.file, y: c.y, pw: 0, w: 0 };
    s.pw += c.p * c.words;
    s.w += c.words;
    by.set(c.file, s);
  }
  return [...by.values()].map((s) => ({ file: s.file, y: s.y, p: s.pw / s.w }));
}

function auc(scored) {
  const pos = scored.filter((s) => s.y === 1), neg = scored.filter((s) => s.y === 0);
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p.p > n.p ? 1 : p.p === n.p ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

/** One-sided 95% Clopper-Pearson upper bound for k successes in n trials (exact for k = 0). */
function upper95(k, n) {
  if (k === 0) return 1 - Math.pow(0.05, 1 / n);
  let lo = k / n, hi = 1;
  const cdf = (p) => { let s = 0, c = 1; for (let i = 0; i <= k; i++) { s += c * p ** i * (1 - p) ** (n - i); c = (c * (n - i)) / (i + 1); } return s; };
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (cdf(mid) > 0.05) lo = mid; else hi = mid; }
  return hi;
}

const groups = [...new Set(chunks.map((c) => c.group))];
const held = [];
for (const g of groups) {
  const model = fit(chunks.filter((c) => c.group !== g));
  for (const c of chunks.filter((c) => c.group === g)) held.push({ ...c, p: predict(model, c.x) });
}
const samples = sampleScores(held);
const human = samples.filter((s) => !s.y), ai = samples.filter((s) => s.y);
const pct = (n) => `${(n * 100).toFixed(1)}%`;

console.log(`chunks: ${chunks.length}   samples: ${samples.length} (${ai.length} ai, ${human.length} human)   groups: ${groups.length}   l2=${l2}`);
console.log(`validation: leave-one-group-out, scored per chunk then word-weighted per sample (as production does)\n`);
console.log(`AUC: ${auc(samples).toFixed(3)}`);
for (const t of [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]) {
  const fp = human.filter((r) => r.p >= t).length, fn = ai.filter((r) => r.p < t).length;
  console.log(`  flag at p>=${t.toFixed(2)}: human flagged ${fp}/${human.length} (${pct(fp / human.length)})   ai missed ${fn}/${ai.length} (${pct(fn / ai.length)})`);
}
const fpAtBand = human.filter((r) => r.p >= bandAi).length;
const caught = ai.filter((r) => r.p >= bandAi).length / ai.length;
const fpUpper = upper95(fpAtBand, human.length);
console.log(`\nAt the likely_ai band (${bandAi}):`);
console.log(`  human flagged: ${fpAtBand}/${human.length}. With this few samples the true rate could be as high as ${pct(fpUpper)} (95% upper bound).`);
console.log(`  The band was chosen on these same samples, so treat even that as optimistic.`);
for (const prevalence of [0.05, 0.2, 0.5]) {
  for (const [name, fpr] of [['observed', fpAtBand / human.length], ['upper bound', fpUpper]]) {
    const precision = (prevalence * caught) / (prevalence * caught + (1 - prevalence) * fpr);
    console.log(`  if ${pct(prevalence)} of checked text is AI and the false-positive rate is the ${name}: ${pct(precision)} of "likely AI" flags are right`);
  }
}
console.log(`  humans reading "likely human" (p<${bandHuman}): ${pct(human.filter((r) => r.p < bandHuman).length / human.length)}   ai reading "likely human": ${pct(ai.filter((r) => r.p < bandHuman).length / ai.length)}`);
console.log('\nhighest-scoring human samples:', human.sort((a, b) => b.p - a.p).slice(0, 6).map((r) => `${r.file} ${r.p.toFixed(2)}`).join(' | '));
console.log('lowest-scoring ai samples:    ', ai.sort((a, b) => a.p - b.p).slice(0, 6).map((r) => `${r.file} ${r.p.toFixed(2)}`).join(' | '));

const final = fit(chunks);
const round = (n) => Math.round(n * 100) / 100;
console.log('\nPaste into SCORING in core/scoring.ts, then bump SCORING_VERSION:');
console.log(`  bias: ${round(final.bias)},`);
console.log(`  weights: { ${features.map((f, i) => `${f}: ${round(final.w[i])}`).join(', ')} },`);
