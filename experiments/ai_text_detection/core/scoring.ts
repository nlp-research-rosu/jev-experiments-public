/**
 * Turns Jev's answers for one chunk into a probability, a verdict and a list of tells.
 * Pure functions, no I/O.
 *
 * The model is deliberately simple: one logistic formula over four signals,
 *
 *     p(ai) = sigmoid(bias + Σ weight_i × signal_i)        every signal in 0..1
 *
 * with weights FITTED from labelled samples by `node eval/fit.mjs` (L2 logistic regression,
 * leave-one-out validated), not chosen by hand. One explicit rule sits on top: text that still
 * contains chatbot leftovers ("I hope this helps!") is scored as AI, because that tell is near
 * certain yet too rare in any sample set for a fit to learn.
 *
 * Design stance: a false "AI" accusation hurts a real person more than a miss does, so the
 * `likely_ai` band starts where no human sample was flagged in cross-validation.
 */
import type { ChunkResult, ConfidenceLevel, DetectionResult, TellFamily, TellHit, Verdict } from './contract.js';
import type { Answers } from './questions.js';

/**
 * Bump whenever questions.ts or any number in SCORING changes. It is part of the result-cache key,
 * so a bump stops the backend from serving scores computed under the old scheme.
 */
export const SCORING_VERSION = '2026-09-20.4';

export const SCORING = {
  /**
   * Fitted 2026-09-20 against jev-1.13.0 on 44 human + 48 AI samples (123 chunks) by eval/fit.mjs,
   * l2=0.1, PER CHUNK, the way production scores. Validated leave-one-group-out (a sample and its
   * topic-matched twin are held out together): AUC 0.97. At the bands below, 0 of 44 human samples
   * were flagged (true rate could still be up to about 7%), 62% of AI samples were flagged, and
   * 68% of human samples read "likely human". Refit whenever the sample set grows.
   */
  bias: -7.8,
  weights: { ai_written: 7.85, tells: 6.32, staging: 2.91, specifics: -2.54 },
  /** Chat residue this certain floors the probability. */
  chatResidue: { trigger: 0.9, floor: 0.9 },
  /** p below `human` reads "likely human"; at or above `ai` reads "likely AI". Between is "unclear". */
  bands: { human: 0.3, ai: 0.75, veryAi: 0.9 },
  /** A tell family is shown to the user at or above this strength. */
  tellDisplayThreshold: 0.6,
  /** Below this many words a chunk can never reach high confidence. */
  shortTextWords: 100,
} as const;

const TELL_FAMILIES: TellFamily[] = ['staging', 'rhythm', 'inflation', 'formatting', 'chat_residue'];

const unit = (n: number): number => Math.min(1, Math.max(0, Number.isFinite(n) ? n : 0.5));
export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));
const round = (n: number): number => Math.round(n * 1000) / 1000;

/** The four scored signals, each on 0..1. */
export function signals(a: Answers): Record<keyof typeof SCORING.weights, number> {
  return {
    ai_written: unit(a.ai_written.noul),
    tells: unit(a.tells.score / 3),
    staging: unit(a.staging.noul),
    specifics: unit(a.specifics.noul),
  };
}

export function hasChatResidue(a: Answers): boolean {
  return a.chat_residue.noul >= SCORING.chatResidue.trigger;
}

/** Probability that the chunk is AI-written. */
export function probability(a: Answers): number {
  const s = signals(a);
  const w = SCORING.weights;
  const z = SCORING.bias + w.ai_written * s.ai_written + w.tells * s.tells + w.staging * s.staging + w.specifics * s.specifics;
  const p = sigmoid(z);
  return hasChatResidue(a) ? Math.max(p, SCORING.chatResidue.floor) : p;
}

export function verdictFor(p: number): Exclude<Verdict, 'too_short'> {
  if (p >= SCORING.bands.veryAi) return 'very_likely_ai';
  if (p >= SCORING.bands.ai) return 'likely_ai';
  if (p < SCORING.bands.human) return 'likely_human';
  return 'unclear';
}

/** Confidence is distance from the nearest band edge, capped for short text. */
function confidenceFor(p: number, words: number): ConfidenceLevel {
  if (words < SCORING.shortTextWords) return 'low';
  const { human, ai } = SCORING.bands;
  const margin = p < human ? human - p : p >= ai ? p - ai : Math.min(p - human, ai - p);
  if (p >= human && p < ai) return 'low'; // "unclear" is by definition not confident
  return margin >= 0.15 ? 'high' : 'medium';
}

export function tellHits(a: Answers): TellHit[] {
  return TELL_FAMILIES.map((family) => ({ family, strength: round(unit(a[family].noul)) }))
    .filter((t) => t.strength >= SCORING.tellDisplayThreshold)
    .sort((x, y) => y.strength - x.strength);
}

/** Scores one chunk from its validated answers. */
export function scoreChunk(id: string, words: number, answers: Answers): ChunkResult {
  const p = probability(answers);
  let verdict: Verdict = verdictFor(p);
  let confidence = hasChatResidue(answers) ? 'high' : confidenceFor(p, words);
  // Short text never earns the strongest label.
  if (confidence === 'low' && verdict === 'very_likely_ai') verdict = 'likely_ai';
  if (words < SCORING.shortTextWords && !hasChatResidue(answers)) confidence = 'low';
  return { id, words, probability: round(p), verdict, confidence, tells: tellHits(answers) };
}

export function tooShortChunk(id: string, words: number): ChunkResult {
  return { id, words, probability: null, verdict: 'too_short', confidence: 'low', tells: [] };
}

const LEVELS: ConfidenceLevel[] = ['low', 'medium', 'high'];

/** Combines chunk results into one result for the page or selection. Word-weighted mean. */
export function aggregate(chunks: ChunkResult[]): DetectionResult {
  const judged = chunks.filter((c): c is ChunkResult & { probability: number } => c.probability !== null);
  const words = chunks.reduce((n, c) => n + c.words, 0);
  if (judged.length === 0) {
    return { probability: null, verdict: 'too_short', confidence: 'low', mixed: false, words, chunks, tells: [] };
  }

  const judgedWords = judged.reduce((n, c) => n + c.words, 0);
  const p = judged.reduce((sum, c) => sum + c.probability * c.words, 0) / judgedWords;
  const mixed =
    judged.some((c) => c.probability >= SCORING.bands.ai) && judged.some((c) => c.probability < SCORING.bands.human);

  let verdict: Verdict = verdictFor(p);
  if (mixed && verdict === 'likely_human') verdict = 'unclear';

  const meanLevel = judged.reduce((sum, c) => sum + LEVELS.indexOf(c.confidence) * c.words, 0) / judgedWords;
  let levelIndex = Math.round(meanLevel);
  if (mixed || verdict === 'unclear') levelIndex = 0;
  if (levelIndex === 0 && verdict === 'very_likely_ai') verdict = 'likely_ai';

  const strongest = new Map<TellFamily, number>();
  for (const c of judged) {
    for (const t of c.tells) strongest.set(t.family, Math.max(strongest.get(t.family) ?? 0, t.strength));
  }
  // Tells explain an AI-leaning verdict. On text that reads human they would only confuse.
  const tells =
    verdict === 'likely_human'
      ? []
      : [...strongest.entries()].map(([family, strength]) => ({ family, strength })).sort((x, y) => y.strength - x.strength);

  return { probability: round(p), verdict, confidence: LEVELS[levelIndex] ?? 'low', mixed, words, chunks, tells };
}
