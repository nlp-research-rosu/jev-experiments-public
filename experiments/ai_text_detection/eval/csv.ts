/**
 * CSV writers for the eval harness output, and the SOURCES.tsv reader for the AI-style
 * breakdown. Pure file-shape logic, no network / scoring here.
 *
 * results.csv and chunks.csv are consumed by tools/eval/fit.mjs (architect-owned) — the column
 * order and names here are load-bearing, do not change them without updating fit.mjs too.
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export const RAW_SIGNAL_KEYS = [
  'ai_written',
  'tells',
  'staging',
  'rhythm',
  'inflation',
  'formatting',
  'chat_residue',
  'specifics',
] as const;
export type RawSignalKey = (typeof RAW_SIGNAL_KEYS)[number];
export type RawSignals = Record<RawSignalKey, number>;

export interface SampleRow {
  file: string;
  label: 'human' | 'ai';
  words: number;
  chunkCount: number;
  judgedChunkCount: number;
  probability: number | null;
  verdict: string;
  confidence: string;
  /** Word-weighted mean over the sample's chunks, raw scale (tells stays 0..3). Null when no chunk was judged. */
  signals: RawSignals | null;
}

export interface ChunkRow {
  file: string;
  label: 'human' | 'ai';
  chunkId: string;
  words: number;
  probability: number | null;
  ai_written: number;
  tells: number;
  staging: number;
  rhythm: number;
  inflation: number;
  formatting: number;
  chat_residue: number;
  specifics: number;
}

const cell = (c: string | number): string => (typeof c === 'string' && c.includes(',') ? `"${c}"` : String(c));

const RESULTS_HEADERS = [
  'file',
  'label',
  'words',
  'chunkCount',
  'judgedChunkCount',
  'probability',
  'verdict',
  'confidence',
  ...RAW_SIGNAL_KEYS,
] as const;

export function writeResultsCsv(rows: SampleRow[], path: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const lines = [RESULTS_HEADERS.join(',')];
  for (const r of rows) {
    const cells: (string | number)[] = [
      r.file,
      r.label,
      r.words,
      r.chunkCount,
      r.judgedChunkCount,
      r.probability ?? '',
      r.verdict,
      r.confidence,
      ...RAW_SIGNAL_KEYS.map((k) => r.signals?.[k] ?? ''),
    ];
    lines.push(cells.map(cell).join(','));
  }
  writeFileSync(path, lines.join('\n'));
}

const CHUNK_HEADERS = ['file', 'label', 'chunkId', 'words', 'probability', ...RAW_SIGNAL_KEYS] as const;

export function writeChunksCsv(rows: ChunkRow[], path: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const lines = [CHUNK_HEADERS.join(',')];
  for (const r of rows) {
    const cells: (string | number)[] = [
      r.file,
      r.label,
      r.chunkId,
      r.words,
      r.probability ?? '',
      r.ai_written,
      r.tells,
      r.staging,
      r.rhythm,
      r.inflation,
      r.formatting,
      r.chat_residue,
      r.specifics,
    ];
    lines.push(cells.map(cell).join(','));
  }
  writeFileSync(path, lines.join('\n'));
}

/** Reads <dir>/ai/SOURCES.tsv: file -> prompt_style. Missing file yields an empty map. */
export function loadAiStyles(sourcesPath: string): Map<string, string> {
  const map = new Map<string, string>();
  if (!existsSync(sourcesPath)) return map;
  const lines = readFileSync(sourcesPath, 'utf8').trim().split(/\r?\n/);
  const header = lines.shift();
  if (!header) return map;
  const cols = header.split('\t');
  const fileIdx = cols.indexOf('file');
  const styleIdx = cols.indexOf('prompt_style');
  if (fileIdx === -1 || styleIdx === -1) return map;
  for (const line of lines) {
    const cells = line.split('\t');
    const file = cells[fileIdx];
    const style = cells[styleIdx];
    if (file && style) map.set(file, style);
  }
  return map;
}

/** Filename prefix before the first "-", e.g. "wikipedia-bologna.txt" -> "wikipedia". */
export function humanSourceOf(file: string): string {
  const idx = file.indexOf('-');
  return idx === -1 ? file.replace(/\.txt$/i, '') : file.slice(0, idx);
}
