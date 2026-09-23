/**
 * Raw-answer cache for the eval harness: <dir>/.cache/answers.json.
 *
 * Keyed by sha256(model + "\n" + sha256(JSON of QUESTIONS) + "\n" + chunk text), so a change to
 * the questions (or the model) invalidates every entry automatically — weights and bands can
 * still change freely, because scoring is recomputed from the cached raw answers every run.
 *
 * The cache from the old (backend-hosted) harness has a different shape (nested under a "screen"
 * key, and it includes a now-removed "voice" question). loadCache() treats anything that doesn't
 * match the current { answers, inputTokens } shape as absent rather than throwing, so an old file
 * just means every chunk is a cache miss on the first run under the new scheme.
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import type { Answers } from '../core/index.js';
import { QUESTIONS, validateAnswers } from '../core/index.js';

export interface CachedAnswer {
  answers: Answers;
  inputTokens: number;
}

export type AnswerCache = Map<string, CachedAnswer>;

const sha256hex = (text: string): string => createHash('sha256').update(text).digest('hex');

/** Stable across runs as long as questions.ts is unchanged. */
export function questionsHash(): string {
  return sha256hex(JSON.stringify(QUESTIONS));
}

/** Cache key: sha256(model + questionsHash + chunk text). */
export function cacheKeyFor(model: string, qHash: string, text: string): string {
  return sha256hex(`${model}\n${qHash}\n${text}`);
}

function isCachedAnswer(value: unknown): value is CachedAnswer {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return validateAnswers(v.answers) !== null && typeof v.inputTokens === 'number';
}

/** Reads the cache. Anything that doesn't match the current shape is dropped, never thrown. */
export function loadCache(path: string): AnswerCache {
  const cache: AnswerCache = new Map();
  if (!existsSync(path)) return cache;
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return cache; // unreadable / not JSON: start fresh
  }
  if (typeof raw !== 'object' || raw === null) return cache;
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (isCachedAnswer(value)) cache.set(key, value);
  }
  return cache;
}

/** Writes the cache. Called incrementally so an interrupted run loses little. */
export function saveCache(path: string, cache: AnswerCache): void {
  mkdirSync(dirname(path), { recursive: true });
  const obj: Record<string, CachedAnswer> = {};
  for (const [key, value] of cache) obj[key] = value;
  writeFileSync(path, JSON.stringify(obj));
}
