import { describe, expect, it } from 'vitest';
import {
  QUESTIONS,
  SCORING,
  aggregate,
  chunkBlocks,
  countWords,
  probability,
  sampleChunks,
  scoreChunk,
  tooShortChunk,
  truncateChars,
  type Answers,
} from '../core/index.js';

/** Builds an answer set. `tells` is on Jev's 0..3 score scale, everything else is 0..1. */
const answers = (over: Partial<Record<keyof Answers, number>> = {}): Answers => {
  const v = { ai_written: 0.5, tells: 0.6, staging: 0.2, rhythm: 0.1, inflation: 0.1, formatting: 0.1, chat_residue: 0.03, specifics: 0.2, ...over };
  const noul = (n: number) => ({ type: 'noul' as const, noul: n });
  return {
    ai_written: noul(v.ai_written), tells: { type: 'score', score: v.tells, confidence: 0.8 },
    staging: noul(v.staging), rhythm: noul(v.rhythm), inflation: noul(v.inflation),
    formatting: noul(v.formatting), chat_residue: noul(v.chat_residue), specifics: noul(v.specifics),
  };
};
// Signal profiles taken from the calibration run.
const AI = { ai_written: 0.92, tells: 2.6, staging: 0.9, inflation: 0.9, specifics: 0.03 };
const WIKIPEDIA = { ai_written: 0.78, tells: 0.3, staging: 0.05, specifics: 0.04 };
const FORUM_POST = { ai_written: 0.4, tells: 0.7, staging: 0.3, specifics: 0.6 };

describe('questions', () => {
  it('asks exactly the questions the scorer and the tells need, in one flat list', () => {
    expect(Object.keys(QUESTIONS).sort()).toEqual(
      ['ai_written', 'chat_residue', 'formatting', 'inflation', 'rhythm', 'specifics', 'staging', 'tells'],
    );
    for (const k of Object.keys(SCORING.weights)) expect(QUESTIONS).toHaveProperty(k);
  });
  it('score levels are standalone descriptions', () => {
    for (const q of Object.values(QUESTIONS)) {
      if (q.type !== 'score') continue;
      for (const level of q.criteria) expect(level.length).toBeGreaterThan(20);
    }
  });
});

describe('probability', () => {
  it('separates the calibrated profiles', () => {
    expect(probability(answers(AI))).toBeGreaterThan(0.9);
    expect(probability(answers(WIKIPEDIA))).toBeLessThan(0.3);
    expect(probability(answers(FORUM_POST))).toBeLessThan(0.3);
  });
  it('does not flag formal human prose just because the direct question leans AI', () => {
    expect(scoreChunk('c0', 300, answers(WIKIPEDIA)).verdict).toBe('likely_human');
  });
  it('first-hand specifics pull the score down', () => {
    expect(probability(answers({ ...AI, specifics: 0.95 }))).toBeLessThan(probability(answers(AI)));
  });
  it('stays inside (0,1) for degenerate input', () => {
    for (const a of [answers({ ai_written: NaN, tells: NaN }), answers({ ai_written: 5, tells: 99 }), answers({ ai_written: -1, tells: -3 })]) {
      const p = probability(a);
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
  });
  it('chat residue floors the probability', () => {
    const r = scoreChunk('c0', 250, answers({ ...FORUM_POST, chat_residue: 0.97 }));
    expect(r.probability).toBeGreaterThanOrEqual(0.9);
    expect(r.confidence).toBe('high');
    expect(r.tells[0]?.family).toBe('chat_residue');
  });
});

describe('scoreChunk', () => {
  it('uses the fitted bands', () => {
    expect(SCORING.bands).toEqual({ human: 0.3, ai: 0.75, veryAi: 0.9 });
    expect(scoreChunk('c0', 250, answers(AI)).verdict).toBe('very_likely_ai');
    expect(scoreChunk('c0', 250, answers({ ai_written: 0.7, tells: 1.0, staging: 0.3 })).verdict).toBe('unclear');
  });
  it('unclear is always low confidence, short text too, and short text is never very_likely_ai', () => {
    expect(scoreChunk('c0', 250, answers({ ai_written: 0.7, tells: 1.0, staging: 0.3 })).confidence).toBe('low');
    const short = scoreChunk('c0', 60, answers(AI));
    expect(short.confidence).toBe('low');
    expect(short.verdict).toBe('likely_ai');
  });
  it('lists tells strongest first, above the display threshold only', () => {
    const r = scoreChunk('c0', 250, answers({ ...AI, staging: 0.7, inflation: 0.92, rhythm: 0.3 }));
    expect(r.tells.map((t) => t.family)).toEqual(['inflation', 'staging']);
  });
});

describe('aggregate', () => {
  it('returns too_short when nothing was judged', () => {
    const r = aggregate([tooShortChunk('c0', 12)]);
    expect(r.verdict).toBe('too_short');
    expect(r.probability).toBeNull();
  });
  it('weights by words and flags mixed pages as unclear', () => {
    const r = aggregate([scoreChunk('c0', 300, answers(AI)), scoreChunk('c1', 300, answers(FORUM_POST))]);
    expect(r.mixed).toBe(true);
    expect(r.verdict).toBe('unclear');
    expect(r.confidence).toBe('low');
    expect(r.words).toBe(600);
  });
  it('agreeing chunks keep their verdict and confidence', () => {
    const r = aggregate([0, 1, 2].map((i) => scoreChunk(`c${i}`, 250, answers(AI))));
    expect(r.verdict).toBe('very_likely_ai');
    expect(r.confidence).toBe('high');
  });
  it('hides tells on text that reads human, and deduplicates by family otherwise', () => {
    expect(aggregate([scoreChunk('c0', 300, answers({ ...FORUM_POST, staging: 0.7 }))]).tells).toEqual([]);
    const r = aggregate([scoreChunk('c0', 250, answers({ ...AI, staging: 0.7, inflation: 0.1 })), scoreChunk('c1', 250, answers({ ...AI, staging: 0.95, inflation: 0.1 }))]);
    expect(r.tells).toEqual([{ family: 'staging', strength: 0.95 }]);
  });
});

describe('chunking', () => {
  const para = (n: number, tag = 'w') => Array.from({ length: n }, (_, i) => `${tag}${i}`).join(' ') + '.';
  it('counts words', () => {
    expect(countWords('  two words ')).toBe(2);
    expect(countWords('')).toBe(0);
  });
  it('packs blocks up to the limit and tracks block indexes', () => {
    const chunks = chunkBlocks([para(120), para(120), para(120), '', para(50)], { maxWords: 300 });
    expect(chunks.map((c) => c.words)).toEqual([240, 170]);
    expect(chunks[0]?.blockIndexes).toEqual([0, 1]);
    expect(chunks[1]?.blockIndexes).toEqual([2, 4]);
  });
  it('splits an overlong block at sentence ends', () => {
    const long = Array.from({ length: 10 }, () => para(50)).join(' ');
    const chunks = chunkBlocks([long], { maxWords: 120 });
    expect(chunks.length).toBeGreaterThan(3);
    for (const c of chunks) expect(c.words).toBeLessThanOrEqual(120);
    expect(chunks.every((c) => c.blockIndexes[0] === 0)).toBe(true);
  });
  it('merges a tiny trailing chunk into the previous one', () => {
    const chunks = chunkBlocks([para(290), para(20)], { maxWords: 300, minWords: 40 });
    expect(chunks.length).toBe(1);
    expect(chunks[0]?.words).toBe(310);
  });
  it('hard-splits a sentence with no punctuation', () => {
    const chunks = chunkBlocks([para(500).replace('.', '')], { maxWords: 200 });
    expect(chunks.every((c) => c.words <= 200)).toBe(true);
  });
  it('samples evenly and keeps first and last', () => {
    const items = Array.from({ length: 10 }, (_, i) => i);
    expect(sampleChunks(items, 3)).toEqual([0, 5, 9]);
    expect(sampleChunks(items, 20)).toEqual(items);
    expect(sampleChunks(items, 1)).toEqual([0]);
  });
  it('truncates on a word boundary', () => {
    const t = truncateChars('alpha beta gamma delta', 12);
    expect(t.length).toBeLessThanOrEqual(12);
    expect(t).toBe('alpha beta');
  });
});
