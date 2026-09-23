import { describe, expect, it } from 'vitest';
import { JEV_PROVIDERS, JevError, QUESTIONS, askJev, cacheKey, detectChunks, testKey, validateAnswers, type ChunkResult } from '../core/index.js';

const WORDS = (n: number) => Array.from({ length: n }, (_, i) => `word${i}`).join(' ') + '.';
const SECRET_TEXT = `zebra-marker ${WORDS(80)}`;

const goodAnswers = (ai = 0.9) => ({
  ai_written: { type: 'noul', noul: ai },
  tells: { type: 'score', score: ai * 3, confidence: 0.8 },
  staging: { type: 'noul', noul: ai },
  rhythm: { type: 'noul', noul: 0.1 },
  inflation: { type: 'noul', noul: ai },
  formatting: { type: 'noul', noul: 0.1 },
  chat_residue: { type: 'noul', noul: 0.02 },
  specifics: { type: 'noul', noul: 0.05 },
});
const ok = (answers: unknown = goodAnswers(), tokens = 1700) =>
  new Response(JSON.stringify({ model: 'jev-1.13.0', answers, usage: { input_tokens: tokens } }), { status: 200 });
const status = (code: number, headers: Record<string, string> = {}) => new Response('{"error":"x"}', { status: code, headers });

function fakeFetch(responses: (Response | Error)[]) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fn = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    const next = responses.shift() ?? ok();
    if (next instanceof Error) throw next;
    return next;
  }) as unknown as typeof fetch;
  return { fn, calls };
}
const noSleep = async () => {};
const ts = { provider: 'typesafe', apiKey: 'key-123' } as const;

describe('askJev', () => {
  it('sends one request with every question, the pinned model and the bearer key', async () => {
    const { fn, calls } = fakeFetch([ok()]);
    const out = await askJev(ts, SECRET_TEXT, { fetch: fn });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe(JEV_PROVIDERS.typesafe.url);
    const body = JSON.parse(String(calls[0]?.init.body));
    expect(body.model).toBe('jev-1.13.0');
    expect(Object.keys(body.questions)).toEqual(Object.keys(QUESTIONS));
    expect(body.state).toBe(SECRET_TEXT);
    expect((calls[0]?.init.headers as Record<string, string>).authorization).toBe('Bearer key-123');
    expect(out.inputTokens).toBe(1700);
  });
  it('targets OpenRouter with its own URL and model slug', async () => {
    const { fn, calls } = fakeFetch([ok()]);
    await askJev({ provider: 'openrouter', apiKey: ' sk-or-1 ' }, SECRET_TEXT, { fetch: fn });
    expect(calls[0]?.url).toBe(JEV_PROVIDERS.openrouter.url);
    expect(JSON.parse(String(calls[0]?.init.body)).model).toBe('typesafe/jev-1.13');
    expect((calls[0]?.init.headers as Record<string, string>).authorization).toBe('Bearer sk-or-1');
  });
  it.each([
    [401, 'invalid_key'], [403, 'invalid_key'], [402, 'no_credits'], [429, 'rate_limited'], [503, 'upstream'], [400, 'bad_request'],
  ])('maps HTTP %i to %s', async (code, expected) => {
    const { fn } = fakeFetch([status(code)]);
    await expect(askJev(ts, SECRET_TEXT, { fetch: fn })).rejects.toMatchObject({ code: expected });
  });
  it('maps a thrown fetch to network, and never puts the text or key in the error', async () => {
    const { fn } = fakeFetch([new Error(`boom ${SECRET_TEXT} key-123`)]);
    const err = await askJev(ts, SECRET_TEXT, { fetch: fn }).catch((e: unknown) => e as JevError);
    expect(err).toBeInstanceOf(JevError);
    expect((err as JevError).code).toBe('network');
    expect(JSON.stringify({ m: (err as JevError).message })).not.toMatch(/zebra-marker|key-123/);
  });
  it('rejects malformed answers', async () => {
    const broken = { ...goodAnswers(), tells: { type: 'score', score: 'high', confidence: 1 } };
    const { fn } = fakeFetch([ok(broken)]);
    await expect(askJev(ts, SECRET_TEXT, { fetch: fn })).rejects.toMatchObject({ code: 'bad_response' });
    expect(validateAnswers({ ...goodAnswers(), specifics: undefined })).toBeNull();
    expect(validateAnswers(goodAnswers())).not.toBeNull();
  });
  it('refuses to call out without a key', async () => {
    const { fn, calls } = fakeFetch([]);
    await expect(askJev({ provider: 'typesafe', apiKey: '  ' }, SECRET_TEXT, { fetch: fn })).rejects.toMatchObject({ code: 'missing_key' });
    expect(calls).toHaveLength(0);
  });
});

describe('detectChunks', () => {
  const chunks = [{ id: 'c0', text: SECRET_TEXT }, { id: 'c1', text: `second ${WORDS(90)}` }];

  it('makes exactly one request per chunk and aggregates', async () => {
    const { fn, calls } = fakeFetch([ok(goodAnswers(0.95)), ok(goodAnswers(0.95))]);
    const out = await detectChunks(ts, chunks, { fetch: fn, sleep: noSleep });
    expect(calls).toHaveLength(2);
    expect(out.requests).toBe(2);
    expect(out.inputTokens).toBe(3400);
    expect(out.result.verdict).toMatch(/likely_ai/);
    expect(out.result.chunks.map((c) => c.id)).toEqual(['c0', 'c1']);
  });
  it('skips short chunks without a request', async () => {
    const { fn, calls } = fakeFetch([]);
    const out = await detectChunks(ts, [{ id: 'c0', text: 'too short to judge' }], { fetch: fn });
    expect(calls).toHaveLength(0);
    expect(out.result.verdict).toBe('too_short');
  });
  it('uses the cache, which holds scores under a hash and never the text', async () => {
    const store = new Map<string, ChunkResult>();
    const cache = { get: async (k: string) => store.get(k), set: async (k: string, v: ChunkResult) => void store.set(k, v) };
    const first = fakeFetch([ok(), ok()]);
    await detectChunks(ts, chunks, { fetch: first.fn, cache });
    const second = fakeFetch([]);
    const out = await detectChunks(ts, chunks, { fetch: second.fn, cache });
    expect(second.calls).toHaveLength(0);
    expect(out.cacheHits).toBe(2);
    expect(JSON.stringify([...store.entries()])).not.toContain('zebra-marker');
    expect([...store.keys()].every((k) => /^[0-9a-f]{64}$/.test(k))).toBe(true);
  });
  it('cache keys differ by provider model, so a model change expires old scores', async () => {
    expect(await cacheKey(ts, 'same text')).not.toBe(await cacheKey({ provider: 'openrouter', apiKey: 'k' }, 'same text'));
    expect(await cacheKey(ts, 'same text')).toBe(await cacheKey({ ...ts, apiKey: 'another key' }, 'same text'));
  });
  it('retries once on 429 and on 5xx', async () => {
    const { fn, calls } = fakeFetch([status(429, { 'retry-after': '1' }), ok()]);
    const waits: number[] = [];
    const out = await detectChunks(ts, [chunks[0]!], { fetch: fn, sleep: async (ms) => void waits.push(ms) });
    expect(calls).toHaveLength(2);
    expect(waits).toEqual([1000]);
    expect(out.result.probability).not.toBeNull();
  });
  it('keeps the good chunks when one fails', async () => {
    const { fn } = fakeFetch([ok(), status(500), status(500)]);
    const out = await detectChunks(ts, chunks, { fetch: fn, sleep: noSleep, concurrency: 1 });
    expect(out.failedChunks).toBe(1);
    expect(out.result.chunks).toHaveLength(1);
  });
  it('throws when every chunk fails', async () => {
    const { fn } = fakeFetch([status(500), status(500), status(500), status(500)]);
    await expect(detectChunks(ts, chunks, { fetch: fn, sleep: noSleep, concurrency: 1 })).rejects.toMatchObject({ code: 'upstream' });
  });
  it('stops at once on a bad key instead of burning requests', async () => {
    const { fn, calls } = fakeFetch([status(401)]);
    await expect(detectChunks(ts, [...chunks, ...chunks], { fetch: fn, sleep: noSleep, concurrency: 1 })).rejects.toMatchObject({ code: 'invalid_key' });
    expect(calls).toHaveLength(1);
  });
});

describe('testKey', () => {
  it('resolves with the answering model on a good key and throws on a bad one', async () => {
    expect(await testKey(ts, { fetch: fakeFetch([ok()]).fn })).toEqual({ model: 'jev-1.13.0' });
    await expect(testKey(ts, { fetch: fakeFetch([status(401)]).fn })).rejects.toMatchObject({ code: 'invalid_key' });
  });
});
