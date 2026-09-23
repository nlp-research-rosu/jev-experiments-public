/**
 * The whole detection pipeline, with no server in the middle.
 *
 * The caller supplies their own API key for TypeSafe or OpenRouter (BYOK). Each chunk costs ONE
 * request carrying all questions. Runs anywhere `fetch` and `crypto.subtle` exist: an MV3 service
 * worker, Node 20+, a test with a fake fetch.
 *
 * Privacy: text goes only to the provider the user chose. Nothing here logs or stores text; the
 * optional cache is keyed by a SHA-256 of the text and holds only the score.
 */
import { LIMITS, type ChunkResult, type DetectChunk, type DetectionResult } from './contract.js';
import { countWords } from './chunking.js';
import { JEV_PROVIDERS, QUESTIONS, type Answers, type JevProvider, type NoulAnswer, type QuestionId, type ScoreAnswer } from './questions.js';
import { SCORING_VERSION, aggregate, scoreChunk, tooShortChunk } from './scoring.js';

export interface ProviderConfig {
  provider: JevProvider;
  apiKey: string;
  /** Overrides the pinned model id. Leave unset unless you have re-run the calibration. */
  model?: string;
}

export type JevErrorCode =
  | 'missing_key'
  | 'invalid_key' // 401 / 403
  | 'no_credits' // 402
  | 'rate_limited' // 429 after the retry
  | 'upstream' // 5xx after the retry
  | 'bad_request' // other 4xx
  | 'bad_response' // 200 with answers we cannot use
  | 'network'; // fetch threw or timed out

export class JevError extends Error {
  constructor(
    public readonly code: JevErrorCode,
    message: string,
    public readonly status?: number,
    public readonly retryAfterMs?: number,
  ) {
    super(message);
    this.name = 'JevError';
  }
}

/** What the person should do about each error. Shown in the UI. */
export const JEV_ERROR_COPY: Record<JevErrorCode, string> = {
  missing_key: 'Add your TypeSafe or OpenRouter API key in settings to start checking text.',
  invalid_key: 'The provider rejected your API key. Check it in settings.',
  no_credits: 'Your provider account is out of credit. Top it up, then try again.',
  rate_limited: 'Your provider is rate limiting requests. Wait a moment and try again.',
  upstream: 'The provider had a problem answering. Try again shortly.',
  bad_request: 'The provider refused the request. If this keeps happening, the API may have changed.',
  bad_response: 'The provider sent an answer Slop Alarm could not read. Try again shortly.',
  network: 'Could not reach the provider. Check your connection and try again.',
};

export interface DetectOptions {
  fetch?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  /** Per-request timeout. Default 20 s. */
  timeoutMs?: number;
  /** Parallel requests. Default 3. */
  concurrency?: number;
  /** Optional result cache. Keys are hashes, values are scores; text is never stored. */
  cache?: {
    get(key: string): Promise<ChunkResult | undefined>;
    set(key: string, value: ChunkResult): Promise<void>;
  };
}

export interface DetectOutcome {
  result: DetectionResult;
  /** Input tokens billed by the provider for this check (0 when everything came from cache). */
  inputTokens: number;
  requests: number;
  cacheHits: number;
  failedChunks: number;
}

// ---------------------------------------------------------------------------

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null;
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isNoul = (v: unknown): v is NoulAnswer => isRecord(v) && v.type === 'noul' && isNum(v.noul);
const isScore = (v: unknown): v is ScoreAnswer => isRecord(v) && v.type === 'score' && isNum(v.score) && isNum(v.confidence);

/** Returns the eight answers, or null if any is missing or malformed. */
export function validateAnswers(raw: unknown): Answers | null {
  if (!isRecord(raw)) return null;
  const out: Record<string, NoulAnswer | ScoreAnswer> = {};
  for (const key of Object.keys(QUESTIONS) as QuestionId[]) {
    const value = raw[key];
    const ok = QUESTIONS[key].type === 'score' ? isScore(value) : isNoul(value);
    if (!ok) return null;
    out[key] = value as NoulAnswer | ScoreAnswer;
  }
  return out as unknown as Answers;
}

function errorForStatus(status: number, retryAfterMs?: number): JevError {
  if (status === 401 || status === 403) return new JevError('invalid_key', `Provider returned ${status}`, status);
  if (status === 402) return new JevError('no_credits', 'Provider returned 402', status);
  if (status === 429) return new JevError('rate_limited', 'Provider returned 429', status, retryAfterMs);
  if (status >= 500) return new JevError('upstream', `Provider returned ${status}`, status);
  return new JevError('bad_request', `Provider returned ${status}`, status);
}

/** One request: all questions about one text. Throws JevError. Never includes the text in an error. */
export async function askJev(
  config: ProviderConfig,
  text: string,
  opts: Pick<DetectOptions, 'fetch' | 'timeoutMs'> = {},
): Promise<{ answers: Answers; inputTokens: number; model: string | null }> {
  if (!config.apiKey.trim()) throw new JevError('missing_key', 'No API key configured');
  const provider = JEV_PROVIDERS[config.provider];
  const doFetch = opts.fetch ?? fetch;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${config.apiKey.trim()}`,
  };
  if (config.provider === 'openrouter') headers['x-title'] = 'Slop Alarm';

  let res: Response;
  try {
    res = await doFetch(provider.url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ model: config.model ?? provider.model, state: text, questions: QUESTIONS }),
      signal: AbortSignal.timeout(opts.timeoutMs ?? 20_000),
    });
  } catch {
    throw new JevError('network', 'Request failed or timed out');
  }
  if (!res.ok) {
    const retryAfter = Number(res.headers.get('retry-after'));
    throw errorForStatus(res.status, Number.isFinite(retryAfter) && retryAfter > 0 ? Math.min(retryAfter, 5) * 1000 : undefined);
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new JevError('bad_response', 'Response was not JSON');
  }
  const answers = validateAnswers(isRecord(body) ? body.answers : undefined);
  if (!answers) throw new JevError('bad_response', 'Response is missing answers');
  const usage = isRecord(body) && isRecord(body.usage) ? body.usage : {};
  return {
    answers,
    inputTokens: isNum(usage.input_tokens) ? usage.input_tokens : 0,
    model: isRecord(body) && typeof body.model === 'string' ? body.model : null,
  };
}

const RETRYABLE: JevErrorCode[] = ['rate_limited', 'upstream', 'network'];
const FATAL: JevErrorCode[] = ['missing_key', 'invalid_key', 'no_credits'];

async function askWithRetry(config: ProviderConfig, text: string, opts: DetectOptions) {
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  try {
    return await askJev(config, text, opts);
  } catch (err) {
    if (!(err instanceof JevError) || !RETRYABLE.includes(err.code)) throw err;
    await sleep(err.retryAfterMs ?? 400);
    return askJev(config, text, opts);
  }
}

/** Hash used as the cache key. Includes the scoring version and model so stale scores expire. */
export async function cacheKey(config: ProviderConfig, text: string): Promise<string> {
  const model = config.model ?? JEV_PROVIDERS[config.provider].model;
  const bytes = new TextEncoder().encode(`${SCORING_VERSION}\n${model}\n${text}`);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Checks a list of chunks and returns one result.
 * Key, credit and "every chunk failed" problems throw a JevError. A partial failure does not:
 * the failed chunks are left out and counted in `failedChunks`.
 */
export async function detectChunks(config: ProviderConfig, chunks: DetectChunk[], opts: DetectOptions = {}): Promise<DetectOutcome> {
  if (!config.apiKey.trim()) throw new JevError('missing_key', 'No API key configured');
  const outcome = { inputTokens: 0, requests: 0, cacheHits: 0, failedChunks: 0 };
  const results: (ChunkResult | null)[] = new Array(chunks.length).fill(null);
  let fatal: JevError | null = null;
  let lastError: JevError | null = null;
  let next = 0;

  const worker = async (): Promise<void> => {
    while (next < chunks.length && !fatal) {
      const i = next++;
      const chunk = chunks[i] as DetectChunk;
      const text = chunk.text.slice(0, LIMITS.maxChunkChars);
      const words = countWords(text);
      if (words < LIMITS.minWords) {
        results[i] = tooShortChunk(chunk.id, words);
        continue;
      }
      const key = opts.cache ? await cacheKey(config, text) : '';
      const hit = opts.cache ? await opts.cache.get(key).catch(() => undefined) : undefined;
      if (hit) {
        results[i] = { ...hit, id: chunk.id };
        outcome.cacheHits++;
        continue;
      }
      try {
        outcome.requests++;
        const { answers, inputTokens } = await askWithRetry(config, text, opts);
        outcome.inputTokens += inputTokens;
        const scored = scoreChunk(chunk.id, words, answers);
        results[i] = scored;
        if (opts.cache) await opts.cache.set(key, scored).catch(() => undefined);
      } catch (err) {
        const jevError = err instanceof JevError ? err : new JevError('network', 'Unexpected failure');
        if (FATAL.includes(jevError.code)) fatal = jevError;
        lastError = jevError;
        outcome.failedChunks++;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(opts.concurrency ?? 3, chunks.length) }, worker));

  if (fatal) throw fatal;
  const ok = results.filter((r): r is ChunkResult => r !== null);
  if (ok.length === 0 && lastError) throw lastError;
  return { result: aggregate(ok), ...outcome };
}

/** Cheapest possible call to confirm a key works. Costs a fraction of a cent. */
export async function testKey(config: ProviderConfig, opts: Pick<DetectOptions, 'fetch' | 'timeoutMs'> = {}): Promise<{ model: string | null }> {
  const sample =
    'This is a short connection test from Slop Alarm. It only checks that the key is accepted and that the provider answers in the expected format.';
  const { model } = await askJev(config, sample, opts);
  return { model };
}
