import { describe, expect, it } from 'vitest';
import { normalizeNewlines } from '../eval/run.js';

// Regression: on Windows, git checked samples out with CRLF, so every cached Jev response (keyed by
// the exact chunk text) missed and a keyless rerun failed instead of reproducing the results.
describe('normalizeNewlines', () => {
  it('turns CRLF and lone CR into LF and leaves LF text unchanged', () => {
    expect(normalizeNewlines('a\r\nb\r\n\r\nc\rd')).toBe('a\nb\n\nc\nd');
    expect(normalizeNewlines('a\nb\n\nc')).toBe('a\nb\n\nc');
  });
});
