#!/usr/bin/env node
// eval/collect-human.mjs
//
// Fetches real human-written text over HTTP from sources where authorship is
// certain to be human, because the text was published before ChatGPT existed
// (November 2022). Writes verbatim excerpts (only whitespace/citation-bracket/
// HTML-entity cleanup applied) to data/ai-text-detection-v1/human/*.txt and records
// provenance in data/ai-text-detection-v1/human/SOURCES.tsv.
//
// No dependencies: plain Node 22+, built-in fetch. Re-run any time to refresh
// the corpus; failed fetches are skipped (never substituted with anything
// written by hand or by an AI).
//
// Usage: node eval/collect-human.mjs

import { writeFile, mkdir, readdir, unlink } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, "human");
const SOURCES_TSV = path.join(OUT_DIR, "SOURCES.tsv");

const UA = "SlopAlarmCalibration/1.0 (internal eval corpus; contact: damian.ovidiu27@gmail.com)";
const MIN_WORDS = 150;
const MAX_WORDS = 400;

// ---------- generic helpers ----------

function wordCount(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

function slugify(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
}

function decodeEntities(s) {
  return s
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n))
    .replace(/&nbsp;/g, " ")
    .replace(/&ndash;/g, "–")
    .replace(/&mdash;/g, "—")
    .replace(/&rsquo;/g, "’")
    .replace(/&lsquo;/g, "‘")
    .replace(/&rdquo;/g, "”")
    .replace(/&ldquo;/g, "“")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function stripTags(html) {
  return html.replace(/<[^>]+>/g, "");
}

// Strip a whole HTML document down to plausible prose <p> blocks.
function extractParagraphsFromHtml(html) {
  let h = html;
  h = h.replace(/<script[\s\S]*?<\/script>/gi, "");
  h = h.replace(/<style[\s\S]*?<\/style>/gi, "");
  h = h.replace(/<sup[^>]*>[\s\S]*?<\/sup>/gi, ""); // citation markers
  h = h.replace(/<table[\s\S]*?<\/table>/gi, ""); // infoboxes / nav tables
  h = h.replace(/<nav[\s\S]*?<\/nav>/gi, "");
  h = h.replace(/<header[\s\S]*?<\/header>/gi, "");
  h = h.replace(/<footer[\s\S]*?<\/footer>/gi, "");
  const pMatches = [...h.matchAll(/<p[^>]*>([\s\S]*?)<\/p>/gi)];
  return pMatches
    .map((m) => cleanParagraph(decodeEntities(stripTags(m[1]))))
    .filter((t) => t.length > 0);
}

function cleanParagraph(text) {
  return text
    .replace(/\[\d+\]/g, "") // citation brackets [12]
    .replace(/\[citation needed\]/gi, "")
    .replace(/\[edit\]/gi, "")
    .replace(/Template:[A-Za-z0-9_\-]+/g, "") // unrendered template residue
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/ *\n */g, "\n")
    .trim();
}

function looksLikeJunk(text) {
  if (!text) return true;
  if (text.length < 20) return true;
  if (/Jump to navigation|Jump to search|Wayback Machine has not archived|cookie/i.test(text)) return true;
  if (text.includes("{") || text.includes("Template:")) return true;
  return false;
}

async function fetchWithRetry(url, opts = {}, retries = 2, timeoutMs = 20000) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": UA, ...(opts.headers || {}) },
        signal: ctrl.signal,
        redirect: "follow",
        ...opts,
      });
      clearTimeout(t);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      return res;
    } catch (err) {
      clearTimeout(t);
      if (attempt === retries) throw err;
      await sleep(800 * (attempt + 1));
    }
  }
  throw new Error("unreachable");
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Take paragraphs (in order) until word count is in [MIN_WORDS, MAX_WORDS].
// Returns null if we can't hit MIN_WORDS without junk, or the best-effort
// text plus its word count.
function assembleFromParagraphs(paragraphs, { minWords = MIN_WORDS, maxWords = MAX_WORDS, skipFirst = 0 } = {}) {
  const good = paragraphs.filter((p) => !looksLikeJunk(p));
  const usable = good.slice(skipFirst);
  let acc = [];
  let words = 0;
  for (const p of usable) {
    const w = wordCount(p);
    if (words + w > maxWords && words >= minWords) break;
    acc.push(p);
    words += w;
    if (words >= minWords && words <= maxWords) {
      // keep going a little only if next paragraph would still fit; loop handles that
    }
    if (words >= maxWords) break;
  }
  if (words < minWords) return null;
  return { text: acc.join("\n\n"), words };
}

// ---------- results collector ----------

const results = []; // { file, url, date, license, words, category }
const failures = []; // { label, reason }

async function saveSample({ source, slug, text, url, date, license, category }) {
  const words = wordCount(text);
  if (words < MIN_WORDS || words > MAX_WORDS + 40) {
    failures.push({ label: `${source}-${slug}`, reason: `word count ${words} out of range` });
    return false;
  }
  if (looksLikeJunk(text)) {
    failures.push({ label: `${source}-${slug}`, reason: "looks like junk after cleanup" });
    return false;
  }
  const filename = `${source}-${slug}.txt`;
  await writeFile(path.join(OUT_DIR, filename), text.trim() + "\n", "utf8");
  results.push({ file: filename, url, date, license, words, category });
  console.log(`  saved ${filename} (${words}w)`);
  return true;
}

// ---------- (a) Wikipedia, revisions at/before 2019-12-31 ----------

const WIKI_ARTICLES = [
  "Bologna",
  "Ganges",
  "Marie_Curie",
  "Siemens",
  "Glastonbury_Festival",
  "Poutine",
  "Office_for_National_Statistics",
  "Golden_Gate_Bridge",
  "Cricket",
  "Tuberculosis",
  "Radiohead",
  "Fall_of_the_Berlin_Wall",
  "Kyoto",
  "Mount_Kilimanjaro",
  "Nikola_Tesla",
  "Sourdough",
  "Ukulele",
  "Great_Barrier_Reef",
];

async function collectWikipedia() {
  console.log(`\n=== (a) Wikipedia (pre-2020 revisions), ${WIKI_ARTICLES.length} candidates ===`);
  let saved = 0;
  for (const title of WIKI_ARTICLES) {
    if (saved >= 15) break;
    try {
      const revUrl = `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(
        title
      )}&prop=revisions&rvlimit=1&rvstart=2019-12-31T00:00:00Z&rvdir=older&rvprop=ids|timestamp&format=json&formatversion=2`;
      const revRes = await fetchWithRetry(revUrl);
      const revData = await revRes.json();
      const page = revData.query?.pages?.[0];
      const rev = page?.revisions?.[0];
      if (!rev) {
        failures.push({ label: `wiki-${title}`, reason: "no revision found before 2019-12-31" });
        continue;
      }
      const parseUrl = `https://en.wikipedia.org/w/api.php?action=parse&oldid=${rev.revid}&prop=text&format=json&formatversion=2`;
      const parseRes = await fetchWithRetry(parseUrl);
      const parseData = await parseRes.json();
      const html = parseData.parse?.text;
      if (!html) {
        failures.push({ label: `wiki-${title}`, reason: "no parse text returned" });
        continue;
      }
      const paragraphs = extractParagraphsFromHtml(html);
      // Skip the very first paragraph sometimes (pronunciation guides / IPA residue);
      // assembleFromParagraphs also filters junk paragraphs out.
      const assembled = assembleFromParagraphs(paragraphs, { skipFirst: 0 });
      if (!assembled) {
        failures.push({ label: `wiki-${title}`, reason: "could not assemble 150-400 words of clean prose" });
        continue;
      }
      const ok = await saveSample({
        source: "wikipedia",
        slug: slugify(title),
        text: assembled.text,
        url: `https://en.wikipedia.org/wiki/${title} (revision ${rev.revid})`,
        date: rev.timestamp.slice(0, 10),
        license: "CC BY-SA 3.0/4.0",
        category: "wikipedia",
      });
      if (ok) saved++;
      await sleep(300);
    } catch (err) {
      failures.push({ label: `wiki-${title}`, reason: String(err.message || err) });
    }
  }
}

// ---------- (b) Stack Exchange answers 2012-2019 ----------

const SE_SITES = ["cooking", "travel", "english", "workplace", "diy", "academia", "parenting", "outdoors"];
const SE_FROM = Math.floor(new Date("2012-01-01T00:00:00Z").getTime() / 1000);
const SE_TO = Math.floor(new Date("2019-12-31T23:59:59Z").getTime() / 1000);

function stripSeHtml(bodyHtml) {
  let h = bodyHtml;
  h = h.replace(/<pre[\s\S]*?<\/pre>/gi, ""); // drop code blocks
  h = h.replace(/<code[\s\S]*?<\/code>/gi, "");
  h = h.replace(/<blockquote[\s\S]*?<\/blockquote>/gi, (m) => stripTags(m)); // keep quote text
  h = h.replace(/<a [^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi, "$2"); // links -> anchor text
  const paras = [...h.matchAll(/<p>([\s\S]*?)<\/p>/gi)].map((m) =>
    cleanParagraph(decodeEntities(stripTags(m[1])))
  );
  const listItems = [...h.matchAll(/<li>([\s\S]*?)<\/li>/gi)].map((m) =>
    cleanParagraph(decodeEntities(stripTags(m[1])))
  );
  return { paras: paras.filter((p) => p.length > 0), hasList: listItems.length > 0 };
}

async function collectStackExchange() {
  console.log(`\n=== (b) Stack Exchange answers 2012-2019, sites: ${SE_SITES.join(", ")} ===`);
  let saved = 0;
  for (const site of SE_SITES) {
    if (saved >= 20) break;
    try {
      const url = `https://api.stackexchange.com/2.3/answers?site=${site}&fromdate=${SE_FROM}&todate=${SE_TO}&order=desc&sort=votes&pagesize=15&filter=withbody`;
      const res = await fetchWithRetry(url);
      const data = await res.json();
      const items = data.items || [];
      let takenFromSite = 0;
      for (const item of items) {
        if (takenFromSite >= 3 || saved >= 20) break;
        const { paras } = stripSeHtml(item.body);
        const assembled = assembleFromParagraphs(paras, { minWords: 150, maxWords: 400 });
        if (!assembled) continue;
        const ok = await saveSample({
          source: "stackexchange",
          slug: `${site}-${item.answer_id}`,
          text: assembled.text,
          url: item.link || `https://${site}.stackexchange.com/a/${item.answer_id}`,
          date: new Date(item.creation_date * 1000).toISOString().slice(0, 10),
          license: "CC BY-SA (per Stack Exchange ToS)",
          category: "stackexchange",
        });
        if (ok) {
          saved++;
          takenFromSite++;
        }
      }
      if (takenFromSite === 0) {
        failures.push({ label: `se-${site}`, reason: "no answer in date range yielded 150-400 clean words" });
      }
      await sleep(300);
    } catch (err) {
      failures.push({ label: `se-${site}`, reason: String(err.message || err) });
    }
  }
}

// ---------- (c) Project Gutenberg essays/letters ----------

const GUTENBERG_BOOKS = [
  { id: 2945, title: "Essays-Second-Series", author: "Ralph Waldo Emerson" },
  { id: 2944, title: "Essays-First-Series", author: "Ralph Waldo Emerson" },
  { id: 575, title: "Essays", author: "Francis Bacon" },
  { id: 1022, title: "Walking", author: "Henry David Thoreau" },
  { id: 2048, title: "The-Sketch-Book", author: "Washington Irving" },
  { id: 1377, title: "Essays-of-Elia", author: "Charles Lamb" },
];

async function collectGutenberg() {
  console.log(`\n=== (c) Project Gutenberg essays, ${GUTENBERG_BOOKS.length} candidates ===`);
  let saved = 0;
  for (const book of GUTENBERG_BOOKS) {
    if (saved >= 8) break;
    try {
      const url = `https://www.gutenberg.org/cache/epub/${book.id}/pg${book.id}.txt`;
      const res = await fetchWithRetry(url);
      const raw = await res.text();
      const startMatch = raw.match(/\*\*\* START OF[^\n]*\*\*\*/i);
      const endMatch = raw.match(/\*\*\* END OF[^\n]*\*\*\*/i);
      const bodyStart = startMatch ? raw.indexOf(startMatch[0]) + startMatch[0].length : 0;
      const bodyEnd = endMatch ? raw.indexOf(endMatch[0]) : raw.length;
      const body = raw.slice(bodyStart, bodyEnd);
      // Split into paragraphs on blank lines, join wrapped lines within a paragraph.
      const rawParas = body
        .split(/\r?\n\s*\r?\n/)
        .map((p) => p.replace(/\r?\n/g, " ").replace(/\s+/g, " ").trim())
        .filter((p) => p.length > 200 && !/^(CONTENTS|CHAPTER|ESSAY [IVXLC]+|[IVXLC]+\.)$/i.test(p));
      // Take two non-adjacent excerpts from well into the book (skip TOC/title matter).
      const usable = rawParas.slice(Math.floor(rawParas.length * 0.1));
      let takenFromBook = 0;
      let idx = 0;
      while (idx < usable.length && takenFromBook < 2 && saved < 8) {
        // group consecutive paragraphs starting at idx until 150-400 words
        let acc = [];
        let words = 0;
        let j = idx;
        while (j < usable.length && words < MAX_WORDS) {
          const w = wordCount(usable[j]);
          if (words + w > MAX_WORDS && words >= MIN_WORDS) break;
          acc.push(usable[j]);
          words += w;
          j++;
        }
        if (words >= MIN_WORDS && words <= MAX_WORDS) {
          const ok = await saveSample({
            source: "gutenberg",
            slug: `${slugify(book.title)}-${takenFromBook + 1}`,
            text: acc.join("\n\n"),
            url: `https://www.gutenberg.org/ebooks/${book.id}`,
            date: "public domain (pre-1929 publication)",
            license: "Public Domain",
            category: "gutenberg",
          });
          if (ok) {
            saved++;
            takenFromBook++;
          }
        }
        idx = j + 10; // jump ahead so the two excerpts aren't adjacent
      }
      if (takenFromBook === 0) {
        failures.push({ label: `gutenberg-${book.title}`, reason: "could not assemble 150-400 word excerpt" });
      }
      await sleep(300);
    } catch (err) {
      failures.push({ label: `gutenberg-${book.title}`, reason: String(err.message || err) });
    }
  }
}

// ---------- Wayback helpers: discover real snapshots via the CDX API ----------
//
// Rather than guessing a Wayback timestamp for a URL (which 404s whenever that
// exact minute wasn't captured), ask the CDX API which snapshots actually
// exist in a date range, then fetch those exact (timestamp, url) pairs. This
// is what makes fetches of gov/blog pages reliable instead of guesswork.

async function cdxSearch(urlPattern, { from = "20190101", to = "20191231", limit = 40 } = {}) {
  const cdxUrl = `https://web.archive.org/cdx/search/cdx?url=${encodeURIComponent(
    urlPattern
  )}&from=${from}&to=${to}&filter=statuscode:200&filter=mimetype:text/html&collapse=urlkey&limit=${limit}&output=json`;
  const res = await fetchWithRetry(cdxUrl, {}, 3, 20000);
  const data = await res.json();
  if (!Array.isArray(data) || data.length < 2) return [];
  const [, ...rows] = data; // first row is the header
  return rows.map((r) => ({ timestamp: r[1], original: r[2], length: Number(r[6]) || 0 }));
}

async function fetchWaybackSnapshot(targetUrl, timestamp) {
  const waybackUrl = `https://web.archive.org/web/${timestamp}id_/${targetUrl}`;
  const res = await fetchWithRetry(waybackUrl, {}, 3, 25000);
  const html = await res.text();
  if (/Wayback Machine has not archived that URL|This page does not exist on the web/i.test(html.slice(0, 2000))) {
    throw new Error("no snapshot available");
  }
  return html;
}

// URL-path fragments that mark listing/index/utility pages rather than
// article prose, so we skip them when picking candidates from a CDX listing.
const NON_ARTICLE_PATH_HINTS = /\/(index\.html?|tag|tags|category|categories|search|page\/\d+|feed|amp)\/?($|\?|#)|\?/i;

function pickArticleCandidates(rows, { minLen = 6000, maxLen = 90000, limit = 12 } = {}) {
  const seen = new Set();
  const out = [];
  for (const r of rows) {
    if (r.length < minLen || r.length > maxLen) continue;
    if (NON_ARTICLE_PATH_HINTS.test(r.original)) continue;
    if (seen.has(r.original)) continue;
    seen.add(r.original);
    out.push(r);
    if (out.length >= limit) break;
  }
  return out;
}

// ---------- (d) US federal gov plain-language pages, pinned via Wayback ----------

const GOV_CDX_QUERIES = [
  "nps.gov/yell/learn/historyculture/*",
  "nps.gov/yell/learn/nature/*",
  "nps.gov/grca/learn/nature/*",
  "nps.gov/grca/learn/historyculture/*",
  "nps.gov/zion/learn/nature/*",
  "nps.gov/ever/learn/nature/*",
  "nps.gov/care/learn/historyculture/*",
  "nps.gov/deva/learn/nature/*",
  "cdc.gov/museum/*",
  "noaa.gov/education/explainers/*",
  "usgs.gov/special-topics/*",
];

async function collectGov() {
  console.log(`\n=== (d) US federal gov pages, discovered via Wayback CDX ===`);
  let saved = 0;
  for (const query of GOV_CDX_QUERIES) {
    if (saved >= 10) break;
    try {
      const rows = await cdxSearch(query);
      const candidates = pickArticleCandidates(rows);
      let takenFromQuery = 0;
      for (const cand of candidates) {
        if (takenFromQuery >= 2 || saved >= 10) break;
        try {
          const html = await fetchWaybackSnapshot(cand.original, cand.timestamp);
          const paragraphs = extractParagraphsFromHtml(html);
          const assembled = assembleFromParagraphs(paragraphs, { minWords: 150, maxWords: 400 });
          if (!assembled) continue;
          const label = slugify(new URL(cand.original).pathname);
          const ok = await saveSample({
            source: "govweb",
            slug: label,
            text: assembled.text,
            url: cand.original,
            date: `${cand.timestamp.slice(0, 4)}-${cand.timestamp.slice(4, 6)}-${cand.timestamp.slice(6, 8)} (Wayback snapshot)`,
            license: "US Government Work (public domain)",
            category: "govweb",
          });
          if (ok) {
            saved++;
            takenFromQuery++;
          }
          await sleep(300);
        } catch (err) {
          failures.push({ label: `gov-${cand.original}`, reason: String(err.message || err) });
        }
      }
      if (takenFromQuery === 0) {
        failures.push({ label: `gov-query-${query}`, reason: "no candidate yielded 150-400 clean words" });
      }
    } catch (err) {
      failures.push({ label: `gov-query-${query}`, reason: String(err.message || err) });
    }
  }
}

// ---------- (e) pre-2022 blogs/forum-style text via Wayback ----------

const BLOG_CDX_QUERIES = [
  { pattern: "blog.mozilla.org/*", license: "Mozilla blog (CC BY-SA per site footer)" },
  { pattern: "blog.wikimedia.org/*", license: "CC BY-SA (Wikimedia blog)" },
  { pattern: "blogs.loc.gov/loc/*", license: "US Government Work (public domain)" },
  { pattern: "blogs.loc.gov/headlinesandheroes/*", license: "US Government Work (public domain)" },
  { pattern: "blog.digitalpreservation.gov/*", license: "US Government Work (public domain)" },
];

async function collectBlogs() {
  console.log(`\n=== (e) Pre-2022 blog posts, discovered via Wayback CDX ===`);
  let saved = 0;
  for (const { pattern, license } of BLOG_CDX_QUERIES) {
    if (saved >= 8) break;
    try {
      const rows = await cdxSearch(pattern, { limit: 60 });
      const candidates = pickArticleCandidates(rows, { minLen: 8000, maxLen: 60000 });
      let takenFromQuery = 0;
      for (const cand of candidates) {
        if (takenFromQuery >= 2 || saved >= 8) break;
        try {
          const html = await fetchWaybackSnapshot(cand.original, cand.timestamp);
          const paragraphs = extractParagraphsFromHtml(html);
          const assembled = assembleFromParagraphs(paragraphs, { minWords: 150, maxWords: 400 });
          if (!assembled) continue;
          const label = slugify(new URL(cand.original).pathname);
          const ok = await saveSample({
            source: "blog",
            slug: label,
            text: assembled.text,
            url: cand.original,
            date: `${cand.timestamp.slice(0, 4)}-${cand.timestamp.slice(4, 6)}-${cand.timestamp.slice(6, 8)} (Wayback snapshot)`,
            license,
            category: "blog",
          });
          if (ok) {
            saved++;
            takenFromQuery++;
          }
          await sleep(300);
        } catch (err) {
          failures.push({ label: `blog-${cand.original}`, reason: String(err.message || err) });
        }
      }
      if (takenFromQuery === 0) {
        failures.push({ label: `blog-query-${pattern}`, reason: "no candidate yielded 150-400 clean words" });
      }
    } catch (err) {
      failures.push({ label: `blog-query-${pattern}`, reason: String(err.message || err) });
    }
  }
}

// ---------- main ----------

async function main() {
  await mkdir(OUT_DIR, { recursive: true });

  // Clean out previously generated samples so re-running doesn't accumulate stale files.
  const existing = await readdir(OUT_DIR);
  for (const f of existing) {
    if (f.endsWith(".txt")) await unlink(path.join(OUT_DIR, f));
  }

  await collectWikipedia();
  await collectStackExchange();
  await collectGutenberg();
  await collectGov();
  await collectBlogs();

  // Write SOURCES.tsv
  const header = "file\tsource_url\trevision_or_snapshot_date\tlicense\tword_count\n";
  const rows = results
    .map((r) => [r.file, r.url, r.date, r.license, r.words].join("\t"))
    .join("\n");
  await writeFile(SOURCES_TSV, header + rows + "\n", "utf8");

  console.log(`\n=== Summary ===`);
  const byCategory = {};
  for (const r of results) byCategory[r.category] = (byCategory[r.category] || 0) + 1;
  for (const [cat, count] of Object.entries(byCategory)) console.log(`  ${cat}: ${count}`);
  console.log(`  TOTAL saved: ${results.length}`);
  if (failures.length) {
    console.log(`\n=== Skipped/failed (${failures.length}) ===`);
    for (const f of failures) console.log(`  ${f.label}: ${f.reason}`);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
