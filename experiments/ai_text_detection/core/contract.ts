/**
 * Shared result types and limits. There is no server: the extension calls the provider directly
 * with the user's own key (see detect.ts).
 */

// ---------- Verdicts ----------

/** Ordered from most human to most AI. `too_short` means we refused to judge. */
export type Verdict = 'likely_human' | 'unclear' | 'likely_ai' | 'very_likely_ai' | 'too_short';

/** How much the individual signals agreed with each other. */
export type ConfidenceLevel = 'low' | 'medium' | 'high';

/** Families of AI-writing tells, after github.com/blader/humanizer. */
export type TellFamily =
  | 'staging'
  | 'rhythm'
  | 'inflation'
  | 'formatting'
  | 'chat_residue';

export interface TellHit {
  family: TellFamily;
  /** Probability (0..1) that this family of tells is present. */
  strength: number;
}

export interface ChunkResult {
  id: string;
  words: number;
  /** Probability (0..1) that the chunk is AI-written. Absent when verdict is `too_short`. */
  probability: number | null;
  verdict: Verdict;
  confidence: ConfidenceLevel;
  /** Tell families found in this chunk, strongest first. */
  tells: TellHit[];
}

export interface DetectionResult {
  /** Word-weighted probability across judged chunks. Null when nothing could be judged. */
  probability: number | null;
  verdict: Verdict;
  confidence: ConfidenceLevel;
  /** True when some chunks read as AI and others as human. */
  mixed: boolean;
  words: number;
  chunks: ChunkResult[];
  /** Strongest tells across all chunks, strongest first, deduplicated by family. */
  tells: TellHit[];
}

// ---------- Checks ----------

/** How a check was started. Decides how many chunks are sampled from the text. */
export type DetectMode = 'selection' | 'page' | 'auto';

export interface DetectChunk {
  id: string;
  text: string;
}

// ---------- Limits ----------

export const LIMITS = {
  /** Below this many words we refuse to judge a chunk. */
  minWords: 40,
  /** Target upper bound for a chunk. */
  maxChunkWords: 300,
  /** Hard cap per chunk. */
  maxChunkChars: 3000,
  maxChunks: { selection: 5, page: 5, auto: 3 } as Record<DetectMode, number>,
} as const;
