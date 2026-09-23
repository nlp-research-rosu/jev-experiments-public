import { describe, expect, it } from 'vitest';
import { auc, breakdown, clopperPearsonUpper, confusionMatrix, mean, round, stddev } from '../eval/metrics.js';

describe('mean / stddev', () => {
  it('computes mean and population stddev', () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    expect(stddev([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(2, 5);
  });

  it('returns NaN for an empty array', () => {
    expect(mean([])).toBeNaN();
    expect(stddev([])).toBeNaN();
  });
});

describe('round', () => {
  it('rounds to 3 decimal places', () => {
    expect(round(0.123456)).toBe(0.123);
    expect(round(0.1235)).toBeCloseTo(0.124, 5);
  });
});

describe('auc', () => {
  it('is 1 when every ai score beats every human score', () => {
    const rows = [
      { label: 'human' as const, score: 0.1 },
      { label: 'human' as const, score: 0.2 },
      { label: 'ai' as const, score: 0.8 },
      { label: 'ai' as const, score: 0.9 },
    ];
    expect(auc(rows)).toBe(1);
  });

  it('is 0 when every ai score loses to every human score', () => {
    const rows = [
      { label: 'human' as const, score: 0.9 },
      { label: 'ai' as const, score: 0.1 },
    ];
    expect(auc(rows)).toBe(0);
  });

  it('counts ties as half a win', () => {
    const rows = [
      { label: 'human' as const, score: 0.5 },
      { label: 'ai' as const, score: 0.5 },
    ];
    expect(auc(rows)).toBe(0.5);
  });

  it('ignores missing scores and returns NaN when a class is empty', () => {
    const rows = [
      { label: 'human' as const, score: null },
      { label: 'ai' as const, score: 0.9 },
    ];
    expect(auc(rows)).toBeNaN();
  });

  it('matches a hand-computed value on a mixed set', () => {
    // human: 0.1, 0.6 | ai: 0.4, 0.9
    // pairs: (0.4>0.1)=1 (0.4>0.6)=0 (0.9>0.1)=1 (0.9>0.6)=1 -> 3/4
    const rows = [
      { label: 'human' as const, score: 0.1 },
      { label: 'human' as const, score: 0.6 },
      { label: 'ai' as const, score: 0.4 },
      { label: 'ai' as const, score: 0.9 },
    ];
    expect(auc(rows)).toBeCloseTo(0.75, 10);
  });
});

describe('clopperPearsonUpper', () => {
  it('is the closed-form 1 - alpha^(1/n) when k = 0', () => {
    const n = 44;
    expect(clopperPearsonUpper(0, n)).toBeCloseTo(1 - Math.pow(0.05, 1 / n), 10);
  });

  it('grows with k for a fixed n', () => {
    const n = 50;
    const u0 = clopperPearsonUpper(0, n);
    const u1 = clopperPearsonUpper(1, n);
    const u5 = clopperPearsonUpper(5, n);
    expect(u1).toBeGreaterThan(u0);
    expect(u5).toBeGreaterThan(u1);
  });

  it('is 1 when k = n', () => {
    expect(clopperPearsonUpper(10, 10)).toBe(1);
  });

  it('shrinks toward the observed rate as n grows, for fixed k/n', () => {
    // Sanity: a bigger sample at the same observed proportion has a tighter (smaller) upper bound.
    const small = clopperPearsonUpper(5, 20); // 25%
    const large = clopperPearsonUpper(25, 100); // 25%
    expect(large).toBeLessThan(small);
    expect(large).toBeGreaterThan(0.25);
  });
});

describe('confusionMatrix', () => {
  it('counts verdicts per label over the given verdict order', () => {
    const rows = [
      { label: 'human' as const, verdict: 'likely_human' },
      { label: 'human' as const, verdict: 'likely_human' },
      { label: 'human' as const, verdict: 'unclear' },
      { label: 'ai' as const, verdict: 'likely_ai' },
      { label: 'ai' as const, verdict: 'very_likely_ai' },
    ];
    const m = confusionMatrix(rows, ['likely_human', 'unclear', 'likely_ai', 'very_likely_ai']);
    expect(m.human).toEqual({ likely_human: 2, unclear: 1, likely_ai: 0, very_likely_ai: 0 });
    expect(m.ai).toEqual({ likely_human: 0, unclear: 0, likely_ai: 1, very_likely_ai: 1 });
  });
});

describe('breakdown', () => {
  it('groups by key and reports count, mean probability and flagged share', () => {
    const rows = [
      { key: 'default', probability: 0.9, flagged: true },
      { key: 'default', probability: 0.3, flagged: false },
      { key: 'engaging', probability: 0.6, flagged: true },
    ];
    const groups = breakdown(rows);
    const byKey = Object.fromEntries(groups.map((g) => [g.key, g]));
    expect(byKey.default).toEqual({ key: 'default', count: 2, meanProbability: 0.6, share: 0.5 });
    expect(byKey.engaging).toEqual({ key: 'engaging', count: 1, meanProbability: 0.6, share: 1 });
  });

  it('sorts groups by count, descending', () => {
    const rows = [
      { key: 'a', probability: 0.5, flagged: false },
      { key: 'b', probability: 0.5, flagged: false },
      { key: 'b', probability: 0.5, flagged: false },
    ];
    expect(breakdown(rows).map((g) => g.key)).toEqual(['b', 'a']);
  });

  it('excludes null probabilities from the mean', () => {
    const rows = [
      { key: 'a', probability: null, flagged: false },
      { key: 'a', probability: 0.4, flagged: false },
    ];
    expect(breakdown(rows)[0]!.meanProbability).toBe(0.4);
  });
});
