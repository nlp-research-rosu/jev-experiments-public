// Rebuilds ablations/markdown-stripped/ai/ from ../ai/: every AI sample containing markdown
// (bold, headings, bullets, numbered items) with that markup removed and the words kept.
// Used to test whether markdown, which only the AI samples contain, drives the detector.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.join(here, '..', 'ai');
const out = path.join(here, 'markdown-stripped', 'ai');
fs.mkdirSync(out, { recursive: true });
let n = 0;
for (const f of fs.readdirSync(src).filter((f) => f.endsWith('.txt'))) {
  const t = fs.readFileSync(path.join(src, f), 'utf8');
  if (!/\*\*|^#{1,4} |^[-*] /m.test(t)) continue;
  const s = t.replace(/\*\*(.+?)\*\*/g, '$1').replace(/^#{1,4} +/gm, '').replace(/^[-*] +/gm, '').replace(/^\d+\. +/gm, '');
  fs.writeFileSync(path.join(out, f), s);
  n++;
}
console.log(`wrote ${n} stripped files to ${out}`);
