/**
 * Splits extracted text blocks (paragraphs, headings, list items) into chunks for detection.
 * Each chunk remembers which blocks it came from, so the extension can highlight them on the page.
 */
import { LIMITS } from './contract.js';

export interface TextChunk {
  id: string;
  text: string;
  words: number;
  /** Indexes into the `blocks` array passed to chunkBlocks. */
  blockIndexes: number[];
}

export function countWords(text: string): number {
  const m = text.trim().match(/\S+/g);
  return m ? m.length : 0;
}

/** Splits an overlong block at sentence ends, falling back to a hard word split. */
function splitLongBlock(text: string, maxWords: number): string[] {
  const sentences = text.match(/[^.!?。！？]+[.!?。！？]+["')\]]*\s*|[^.!?。！？]+$/g) ?? [text];
  const parts: string[] = [];
  let current = '';
  const flush = () => {
    if (current.trim()) parts.push(current.trim());
    current = '';
  };
  for (const sentence of sentences) {
    if (countWords(sentence) > maxWords) {
      flush();
      const tokens = sentence.trim().split(/\s+/);
      for (let i = 0; i < tokens.length; i += maxWords) parts.push(tokens.slice(i, i + maxWords).join(' '));
      continue;
    }
    if (countWords(current) + countWords(sentence) > maxWords) flush();
    current += sentence;
  }
  flush();
  return parts;
}

/**
 * Greedily packs consecutive blocks into chunks of at most `maxWords`.
 * A trailing chunk under `minWords` is merged into its predecessor when one exists.
 */
export function chunkBlocks(
  blocks: string[],
  opts: { maxWords?: number; minWords?: number } = {},
): TextChunk[] {
  const maxWords = opts.maxWords ?? LIMITS.maxChunkWords;
  const minWords = opts.minWords ?? LIMITS.minWords;

  const pieces: { text: string; index: number }[] = [];
  blocks.forEach((block, index) => {
    const text = block.replace(/[ \t]+/g, ' ').trim();
    if (!text) return;
    if (countWords(text) > maxWords) {
      for (const part of splitLongBlock(text, maxWords)) pieces.push({ text: part, index });
    } else {
      pieces.push({ text, index });
    }
  });

  const chunks: TextChunk[] = [];
  let texts: string[] = [];
  let indexes: number[] = [];
  let words = 0;
  const flush = () => {
    if (texts.length === 0) return;
    chunks.push({ id: `c${chunks.length}`, text: texts.join('\n\n'), words, blockIndexes: [...new Set(indexes)] });
    texts = [];
    indexes = [];
    words = 0;
  };
  for (const piece of pieces) {
    const w = countWords(piece.text);
    if (words > 0 && words + w > maxWords) flush();
    texts.push(piece.text);
    indexes.push(piece.index);
    words += w;
  }
  flush();

  const last = chunks[chunks.length - 1];
  const prev = chunks[chunks.length - 2];
  if (last && prev && last.words < minWords) {
    prev.text += `\n\n${last.text}`;
    prev.words += last.words;
    prev.blockIndexes = [...new Set([...prev.blockIndexes, ...last.blockIndexes])];
    chunks.pop();
  }
  return chunks;
}

/** Picks at most `max` chunks, evenly spread and always including the first and the last. */
export function sampleChunks<T>(chunks: T[], max: number): T[] {
  if (max <= 0) return [];
  if (chunks.length <= max) return chunks;
  if (max === 1) return [chunks[0] as T];
  const picked = new Set<number>();
  for (let i = 0; i < max; i++) picked.add(Math.round((i * (chunks.length - 1)) / (max - 1)));
  return [...picked].sort((a, b) => a - b).map((i) => chunks[i] as T);
}

/** Hard cap applied before sending, mirroring the server-side limit. */
export function truncateChars(text: string, maxChars: number = LIMITS.maxChunkChars): string {
  if (text.length <= maxChars) return text;
  const cut = text.slice(0, maxChars);
  const lastSpace = cut.lastIndexOf(' ');
  return lastSpace > maxChars * 0.8 ? cut.slice(0, lastSpace) : cut;
}
