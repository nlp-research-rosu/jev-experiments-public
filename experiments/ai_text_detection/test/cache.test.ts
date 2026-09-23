import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { Answers } from '../core/index.js';
import { cacheKeyFor, loadCache, questionsHash, saveCache } from '../eval/cache.js';

const ANSWERS: Answers = {
  ai_written: { type: 'noul', noul: 0.8 },
  tells: { type: 'score', score: 2, confidence: 0.6 },
  staging: { type: 'noul', noul: 0.7 },
  rhythm: { type: 'noul', noul: 0.4 },
  inflation: { type: 'noul', noul: 0.3 },
  formatting: { type: 'noul', noul: 0.2 },
  chat_residue: { type: 'noul', noul: 0.0 },
  specifics: { type: 'noul', noul: 0.1 },
};

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'slop-eval-cache-'));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('questionsHash', () => {
  it('is a stable 64-char hex digest', () => {
    const a = questionsHash();
    const b = questionsHash();
    expect(a).toBe(b);
    expect(a).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('cacheKeyFor', () => {
  it('is deterministic for the same inputs', () => {
    expect(cacheKeyFor('jev-1.13.0', 'qhash', 'some text')).toBe(cacheKeyFor('jev-1.13.0', 'qhash', 'some text'));
  });

  it('changes when the model, question hash or text changes', () => {
    const base = cacheKeyFor('jev-1.13.0', 'qhash', 'some text');
    expect(cacheKeyFor('jev-1.14.0', 'qhash', 'some text')).not.toBe(base);
    expect(cacheKeyFor('jev-1.13.0', 'other-qhash', 'some text')).not.toBe(base);
    expect(cacheKeyFor('jev-1.13.0', 'qhash', 'different text')).not.toBe(base);
  });
});

describe('loadCache / saveCache', () => {
  it('returns an empty cache when the file does not exist', () => {
    expect(loadCache(join(dir, 'nope.json')).size).toBe(0);
  });

  it('round-trips a saved cache', () => {
    const path = join(dir, 'answers.json');
    const cache = new Map([['key1', { answers: ANSWERS, inputTokens: 123 }]]);
    saveCache(path, cache);
    const loaded = loadCache(path);
    expect(loaded.size).toBe(1);
    expect(loaded.get('key1')).toEqual({ answers: ANSWERS, inputTokens: 123 });
  });

  it('ignores a file that is not valid JSON, without throwing', () => {
    const path = join(dir, 'broken.json');
    writeFileSync(path, '{not json');
    expect(() => loadCache(path)).not.toThrow();
    expect(loadCache(path).size).toBe(0);
  });

  it('ignores entries from the old backend-era cache shape (nested, includes a removed "voice" question)', () => {
    const path = join(dir, 'old.json');
    const oldShapeEntry = {
      screen: {
        ai_written: { type: 'noul', noul: 0.57 },
        voice: { type: 'score', score: 1.21, confidence: 0.3 },
        tells: { type: 'score', score: 0.48, confidence: 0.52 },
      },
    };
    writeFileSync(path, JSON.stringify({ someOldKey: oldShapeEntry }));
    const loaded = loadCache(path);
    expect(loaded.size).toBe(0);
  });

  it('drops individual malformed entries but keeps valid ones', () => {
    const path = join(dir, 'mixed.json');
    writeFileSync(
      path,
      JSON.stringify({
        good: { answers: ANSWERS, inputTokens: 10 },
        bad: { answers: { ai_written: { type: 'noul', noul: 0.5 } }, inputTokens: 10 }, // missing questions
        alsoBad: 'not an object',
      }),
    );
    const loaded = loadCache(path);
    expect(loaded.size).toBe(1);
    expect(loaded.has('good')).toBe(true);
  });
});
