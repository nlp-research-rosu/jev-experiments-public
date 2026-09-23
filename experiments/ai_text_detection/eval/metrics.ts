/**
 * Pure statistics used by the eval report: no I/O, no knowledge of files or CSV shapes.
 */

export function mean(nums: number[]): number {
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : NaN;
}

export function stddev(nums: number[]): number {
  if (nums.length === 0) return NaN;
  const m = mean(nums);
  return Math.sqrt(nums.reduce((s, x) => s + (x - m) ** 2, 0) / nums.length);
}

export const round = (n: number): number => Math.round(n * 1000) / 1000;

export interface RocRow {
  label: 'human' | 'ai';
  score: number | null | undefined;
}

/**
 * ROC AUC (ai = positive class): probability that a random AI score beats a random human score,
 * ties counting half. NaN when either class is empty after dropping missing scores.
 */
export function auc(rows: RocRow[]): number {
  const positives: number[] = [];
  const negatives: number[] = [];
  for (const r of rows) {
    if (r.score === null || r.score === undefined || Number.isNaN(r.score)) continue;
    (r.label === 'ai' ? positives : negatives).push(r.score);
  }
  if (positives.length === 0 || negatives.length === 0) return NaN;
  let wins = 0;
  for (const p of positives) {
    for (const n of negatives) {
      if (p > n) wins++;
      else if (p === n) wins += 0.5;
    }
  }
  return wins / (positives.length * negatives.length);
}

/**
 * One-sided 95% Clopper-Pearson upper bound for k successes in n Bernoulli trials.
 * Exact for k = 0 (the common case here: zero human false positives observed).
 */
export function clopperPearsonUpper(k: number, n: number, alpha = 0.05): number {
  if (n === 0) return NaN;
  if (k >= n) return 1;
  if (k === 0) return 1 - Math.pow(alpha, 1 / n);
  const cdf = (p: number): number => {
    let s = 0;
    let c = 1; // binomial coefficient C(n, i), built incrementally
    for (let i = 0; i <= k; i++) {
      s += c * p ** i * (1 - p) ** (n - i);
      c = (c * (n - i)) / (i + 1);
    }
    return s;
  };
  let lo = k / n;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) > alpha) lo = mid;
    else hi = mid;
  }
  return hi;
}

export interface ConfusionRow {
  label: 'human' | 'ai';
  verdict: string;
}

/** Counts of verdict by label, in the given verdict order. */
export function confusionMatrix(rows: ConfusionRow[], verdicts: readonly string[]): Record<'human' | 'ai', Record<string, number>> {
  const out = {} as Record<'human' | 'ai', Record<string, number>>;
  for (const label of ['human', 'ai'] as const) {
    const subset = rows.filter((r) => r.label === label);
    out[label] = Object.fromEntries(verdicts.map((v) => [v, subset.filter((r) => r.verdict === v).length]));
  }
  return out;
}

export interface BreakdownRow {
  key: string;
  probability: number | null;
  /** Whichever share this breakdown reports: "flagged" for AI samples, "read as human" for human ones. */
  flagged: boolean;
}

export interface GroupBreakdown {
  key: string;
  count: number;
  meanProbability: number;
  share: number;
}

/** Groups rows by `key`, reporting count, mean probability and the share with `flagged` true. */
export function breakdown(rows: BreakdownRow[]): GroupBreakdown[] {
  const byKey = new Map<string, BreakdownRow[]>();
  for (const r of rows) {
    if (!byKey.has(r.key)) byKey.set(r.key, []);
    byKey.get(r.key)!.push(r);
  }
  return [...byKey.entries()]
    .map(([key, subset]) => ({
      key,
      count: subset.length,
      meanProbability: round(mean(subset.map((r) => r.probability).filter((p): p is number => p !== null))),
      share: round(subset.filter((r) => r.flagged).length / subset.length),
    }))
    .sort((a, b) => b.count - a.count);
}
