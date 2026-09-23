import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { type ChunkRow, type SampleRow, humanSourceOf, loadAiStyles, writeChunksCsv, writeResultsCsv } from '../eval/csv.js';

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'slop-eval-csv-'));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('writeResultsCsv', () => {
  it('writes the exact header fit.mjs expects', () => {
    const row: SampleRow = {
      file: 'wikipedia-bologna.txt',
      label: 'human',
      words: 278,
      chunkCount: 1,
      judgedChunkCount: 1,
      probability: 0.07,
      verdict: 'likely_human',
      confidence: 'high',
      signals: { ai_written: 0.6, tells: 0.3, staging: 0.1, rhythm: 0.1, inflation: 0.1, formatting: 0.1, chat_residue: 0.05, specifics: 0.65 },
    };
    const path = join(dir, 'results.csv');
    writeResultsCsv([row], path);
    const lines = readFileSync(path, 'utf8').trim().split('\n');
    expect(lines[0]).toBe(
      'file,label,words,chunkCount,judgedChunkCount,probability,verdict,confidence,ai_written,tells,staging,rhythm,inflation,formatting,chat_residue,specifics',
    );
    expect(lines[1]).toBe('wikipedia-bologna.txt,human,278,1,1,0.07,likely_human,high,0.6,0.3,0.1,0.1,0.1,0.1,0.05,0.65');
  });

  it('leaves probability and signal cells blank when the sample has no judged chunks', () => {
    const row: SampleRow = {
      file: 'too-short.txt',
      label: 'ai',
      words: 5,
      chunkCount: 1,
      judgedChunkCount: 0,
      probability: null,
      verdict: 'too_short',
      confidence: 'low',
      signals: null,
    };
    const path = join(dir, 'results.csv');
    writeResultsCsv([row], path);
    const line = readFileSync(path, 'utf8').trim().split('\n')[1];
    expect(line).toBe('too-short.txt,ai,5,1,0,,too_short,low,,,,,,,,');
  });

  it('creates the destination directory if missing', () => {
    const path = join(dir, 'nested', 'results.csv');
    writeResultsCsv([], path);
    expect(readFileSync(path, 'utf8').trim()).toContain('file,label,words');
  });
});

describe('writeChunksCsv', () => {
  it('writes one row per judged chunk on the raw scale', () => {
    const row: ChunkRow = {
      file: 'a.txt',
      label: 'ai',
      chunkId: 'c0',
      words: 200,
      probability: 0.91,
      ai_written: 0.95,
      tells: 2.1,
      staging: 0.8,
      rhythm: 0.4,
      inflation: 0.5,
      formatting: 0.2,
      chat_residue: 0.0,
      specifics: 0.1,
    };
    const path = join(dir, 'chunks.csv');
    writeChunksCsv([row], path);
    const lines = readFileSync(path, 'utf8').trim().split('\n');
    expect(lines[0]).toBe('file,label,chunkId,words,probability,ai_written,tells,staging,rhythm,inflation,formatting,chat_residue,specifics');
    expect(lines[1]).toBe('a.txt,ai,c0,200,0.91,0.95,2.1,0.8,0.4,0.5,0.2,0,0.1');
  });
});

describe('loadAiStyles', () => {
  it('maps file to prompt_style', () => {
    const path = join(dir, 'SOURCES.tsv');
    writeFileSync(
      path,
      'file\tmatched_human_file\tprompt_style\tword_count\n' + 'matched-a.txt\ta.txt\tdefault\t280\n' + 'matched-b.txt\tb.txt\tengaging\t260\n',
    );
    const styles = loadAiStyles(path);
    expect(styles.get('matched-a.txt')).toBe('default');
    expect(styles.get('matched-b.txt')).toBe('engaging');
    expect(styles.size).toBe(2);
  });

  it('returns an empty map when the file is missing', () => {
    expect(loadAiStyles(join(dir, 'nope.tsv')).size).toBe(0);
  });
});

describe('humanSourceOf', () => {
  it('takes the filename prefix before the first dash', () => {
    expect(humanSourceOf('wikipedia-bologna.txt')).toBe('wikipedia');
    expect(humanSourceOf('stackexchange-cooking-84325.txt')).toBe('stackexchange');
  });

  it('falls back to the filename without extension when there is no dash', () => {
    expect(humanSourceOf('standalone.txt')).toBe('standalone');
  });
});
