#!/usr/bin/env python3
"""Render the comparison as a standalone HTML report."""
import datetime
import importlib.util
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
spec = importlib.util.spec_from_file_location("bt", HERE / "build_table.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
spec2 = importlib.util.spec_from_file_location("bt2", HERE / "build_table2.py")
bt2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(bt2)

OUT = DATA / "report.html"
LANE_LABEL = {"imp": "imp", "imp-obf": "imp-obf", "imp-swap": "imp-swap"}

CSS = """
:root{
  --paper:#f3f5f5; --card:#fbfcfc; --ink:#16201e; --muted:#5f6d6a;
  --line:#dde3e2; --accent:#0d6b5c; --accent-soft:#e2efec;
  --miss:#b3261e; --miss-soft:#fbeceb; --alarm:#a8751a; --alarm-soft:#fbf3e4;
  --good:#12695a;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --paper:#101615; --card:#161e1d; --ink:#e6edeb; --muted:#93a29e;
  --line:#26312f; --accent:#4fd1b5; --accent-soft:#16302b;
  --miss:#ff8a80; --miss-soft:#2d1917; --alarm:#e5b567; --alarm-soft:#2b2317;
  --good:#4fd1b5;
}}
:root[data-theme="dark"]{
  --paper:#101615; --card:#161e1d; --ink:#e6edeb; --muted:#93a29e;
  --line:#26312f; --accent:#4fd1b5; --accent-soft:#16302b;
  --miss:#ff8a80; --miss-soft:#2d1917; --alarm:#e5b567; --alarm-soft:#2b2317;
  --good:#4fd1b5;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  line-height:1.6;margin:0;padding:3rem 1.5rem 5rem}
.wrap{max-width:60rem;margin:0 auto;display:flex;flex-direction:column;gap:2.5rem}
.prose{max-width:38rem}
h1,h2,h3,.mono,table,code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
h1{font-size:1.65rem;line-height:1.25;margin:0;letter-spacing:-.02em;text-wrap:balance}
h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.14em;
  color:var(--accent);margin:0 0 .9rem;font-weight:600}
h3{font-size:.95rem;margin:1.6rem 0 .5rem;font-weight:600}
p{margin:0 0 .9rem}
.sub{color:var(--muted);font-size:1rem;margin-top:.6rem}
.meta{font-size:.75rem;color:var(--muted);letter-spacing:.04em;
  border-top:1px solid var(--line);padding-top:.8rem;margin-top:1.4rem}
section{border-top:1px solid var(--line);padding-top:1.8rem}
header{padding-bottom:.4rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-size:.8rem;width:max-content;min-width:100%;
  font-variant-numeric:tabular-nums}
th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--line);
  white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:nowrap}
thead th{font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;
  color:var(--muted);font-weight:600;border-bottom:1.5px solid var(--line)}
tbody tr:hover{background:var(--card)}
tr.jev td{background:var(--accent-soft)}
tr.jev:hover td{filter:brightness(.98)}
td.miss{color:var(--miss);background:var(--miss-soft);font-weight:600}
td.miss0{color:var(--good);font-weight:600}
td.alarm{color:var(--alarm);background:var(--alarm-soft);font-weight:600}
td.alarm0{color:var(--good);font-weight:600}
.n{color:var(--muted);font-size:.72rem}
pre{background:var(--card);border:1px solid var(--line);border-left:2px solid var(--accent);
  padding:.8rem 1rem;overflow-x:auto;font-size:.76rem;line-height:1.55;margin:.5rem 0}
code{background:var(--card);padding:.08em .35em;border:1px solid var(--line);
  font-size:.86em}
pre code{background:none;border:none;padding:0}
.tag{display:inline-block;font-size:.66rem;letter-spacing:.08em;padding:.12rem .45rem;
  border:1px solid currentColor;text-transform:uppercase;font-weight:600}
.tag.bad{color:var(--miss)} .tag.good{color:var(--good)}
.note{font-size:.82rem;color:var(--muted);border-left:2px solid var(--line);
  padding-left:.9rem;margin:.9rem 0}
ul{margin:.4rem 0 .9rem;padding-left:1.1rem} li{margin:.35rem 0}
strong{font-weight:600}
"""


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def metric_table(order, configs, invisible=False):
    head = ("<div class='scroll'><table><thead><tr>"
            "<th>judge / config</th><th>n</th><th>accuracy</th>"
            "<th>MISS<br><span class='n'>bad called good</span></th>"
            "<th>ALARM<br><span class='n'>good called bad</span></th>"
            + "".join(f"<th>{LANE_LABEL[l]}</th>" for l in bt.LANES)
            + "</tr></thead><tbody>")
    rows = []
    for o in order:
        s = bt.score(configs[o], prover_invisible_only=invisible)
        if not s:
            continue
        lanes = ""
        for l in bt.LANES:
            ls = bt.score(configs[o], l, prover_invisible_only=invisible)
            lanes += f"<td>{pct(ls['acc']) if ls else 'n/a'}</td>"
        mc = "miss0" if s["miss"] == 0 else "miss"
        ac = "alarm0" if s["alarm"] == 0 else "alarm"
        cls = " class='jev'" if o.startswith("jev") else ""
        rows.append(
            f"<tr{cls}><td>{o}</td><td class='n'>{s['n']}</td>"
            f"<td><strong>{pct(s['acc'])}</strong></td>"
            f"<td class='{mc}'>{pct(s['miss'])} "
            f"<span class='n'>{s['miss_n']}/{s['bad_n']}</span></td>"
            f"<td class='{ac}'>{pct(s['alarm'])} "
            f"<span class='n'>{s['alarm_n']}/{s['good_n']}</span></td>{lanes}</tr>")
    return head + "".join(rows) + "</tbody></table></div>"


def main():
    jev = bt.load_jev()
    agents = {j: bt.load_agent(j) for j in ("claude", "codex")}
    configs, perf, order = bt.build_configs(jev, agents)

    # --- fairness audit: tools vs no tools, same model, same criterion ---
    # Labels are written to stand alone in one table cell -- a narrow screen
    # may show only this column, so each label states judge, tool access,
    # and (where it varies within the table) run count in plain words.
    fair_configs = {
        "jev, 1 question": configs["jev, 1 question"],
        "claude, with tools": configs["claude (1 run)"],
        "claude, no tools": bt2.preds_notools("claude"),
        "codex, with tools": configs["codex (1 run)"],
        "codex, no tools": bt2.preds_notools("codex"),
    }
    fair_order = list(fair_configs)

    # --- semantics-trim ablation for Jev ---
    trim_configs = {
        "jev, full semantics": configs["jev, 3 questions (all must pass)"],
        "jev, rules only": bt2.preds_battery("rulesonly"),
        "jev, operators used only": bt2.preds_battery("relevant"),
    }
    trim_order = list(trim_configs)

    overhead_path = DATA / "notools_overhead.json"
    overhead = (json.loads(overhead_path.read_text())
               if overhead_path.exists() else {})

    # cost table
    cost_rows = ""
    for o in order:
        p = perf.get(o)
        if not p:
            continue
        cls = " class='jev'" if o.startswith("jev") else ""
        cost_rows += (f"<tr{cls}><td>{o}</td><td>{p['calls']:.0f}</td>"
                      f"<td>{p['secs']:.1f} s</td>"
                      f"<td>{p['tokens']:,.0f}</td></tr>")

    for j in ("claude", "codex"):
        data = list(bt2.load_notools(j).values())
        if data:
            cost_rows += (
                f"<tr><td>{j}, no tools (1 turn)</td><td>1</td>"
                f"<td>{statistics.mean(d['seconds'] for d in data):.1f} s</td>"
                f"<td>{statistics.mean(d['tokens'] for d in data if d.get('tokens')):,.0f}</td></tr>")

    variants = ["ref", "equiv", "vacuous", "weakened", "narrowed"]
    vhead = ("<div class='scroll'><table><thead><tr><th>judge / config</th>"
             + "".join(
                 f"<th>{v}<br><span class='n'>"
                 f"{'good' if v in ('ref','equiv') else 'bad'}</span></th>"
                 for v in variants) + "</tr></thead><tbody>")
    vrows = ""
    for o in order:
        cells = ""
        for v in variants:
            rows = [m for m in bt.MANIFEST
                    if m["variant"] == v and configs[o].get(m["case"])]
            if not rows:
                cells += "<td>n/a</td>"
                continue
            ok = sum(1 for m in rows if configs[o][m["case"]] == m["label"])
            cells += f"<td>{ok/len(rows):.0%}</td>"
        cls = " class='jev'" if o.startswith("jev") else ""
        vrows += f"<tr{cls}><td>{o}</td>{cells}</tr>"

    accepted = sum(1 for m in bt.MANIFEST
                   if m["label"] == "BAD" and bt.VALID.get(m["case"]))
    total_bad = sum(1 for m in bt.MANIFEST if m["label"] == "BAD")
    n_cases = len(bt.MANIFEST)
    n_good = sum(1 for m in bt.MANIFEST if m["label"] == "GOOD")
    today = datetime.date.today().isoformat()

    html = f"""<title>Judge Bakeoff</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Can Jev judge a proof?</h1>
  <p class="sub prose">A prover tells you a claim is true. It cannot tell you
  the claim says what you meant. Three judges, measured on that second
  question.</p>
  <div class="meta mono">{today} · {n_cases} cases ({n_good} good / {total_bad} bad)
  · jev-1.13.0 vs claude-sonnet-5 vs gpt-5.6-luna · labels exact by construction</div>
</header>

<section>
<h2>What is being judged</h2>
<div class="prose">
<p>Every KleverBench problem ships a prose <code>intent</code> and a gold
<code>reference-spec.k</code>. Mutating that gold spec produces cases whose
labels are exact without anyone hand-labelling them. Each judge sees the
program, the semantics, the proof vocabulary, and one candidate spec. None of
them sees the reference or the label.</p>
</div>
<pre><code><span class="tag good">good</span>  reference  &lt;state&gt; a |-&gt; A:Int  b |-&gt; B:Int  res |-&gt; (_ =&gt; absInt(A) *Int B) &lt;/state&gt;
<span class="tag bad">bad</span>   vacuous    ... requires A ==Int 1 andBool B ==Int 1     collapsed to one point
<span class="tag bad">bad</span>   narrowed   ... requires A &gt;Int 0 andBool A &lt;Int 100     bounded to a window
<span class="tag bad">bad</span>   weakened   res |-&gt; (_ =&gt; _)                             asserts nothing at all</code></pre>
<div class="prose">
<p>The three lanes are the same nine problems written in three languages.
<code>imp</code> reads the way you would expect. In <code>imp-swap</code> the
operators do not mean what they look like: <code>+</code> is subtraction and
<code>/</code> is multiplication. <code>imp-obf</code> keeps the honest
meanings but writes the operators as unfamiliar glyphs. Since only
<code>semantics.k</code> says what an operator does, the distance between the
imp and imp-swap columns shows whether a judge read the semantics or guessed
from the symbols.</p>
</div>
</section>

<section>
<h2>Results · all {n_cases} cases</h2>
{metric_table(order, configs)}
<div class="note">The two errors cost different things. A MISS lets through a
spec that proves but states the wrong theorem. An ALARM rejects correct work
and costs a reviewer some time. The first one is the expensive mistake.</div>
</section>

<section>
<h2>Results · the {accepted}/{total_bad} defects the prover cannot see</h2>
<div class="prose">
<p>Running every spec through the Prover showed that the
<code>weakened</code> mutants fail <code>kompile</code>. The deterministic
layer already rejects those, so counting them inflates every judge's score.
Removing them leaves {accepted} bad specs that compile and prove, which are
the ones worth measuring on.</p>
<p>A sample was run all the way to a finished proof against
<code>prover.intentcomputing.org</code>, so this rests on measurement rather
than on the assumption that a weaker claim must still close:</p>
</div>
<pre><code>imp/abs-times/ref             proved   10.5 s   <span class="tag good">good</span>
imp/abs-times/vacuous         proved   11.5 s   <span class="tag bad">bad</span>
imp/abs-times/narrowed        proved   11.8 s   <span class="tag bad">bad</span>
imp-swap/abs-times/narrowed   proved   12.2 s   <span class="tag bad">bad</span></code></pre>
<div class="prose">
<p>All four reach <code>#Top</code> against the same prover in about the same
time. Three of them state the wrong theorem.</p>
</div>
{metric_table(order, configs, invisible=True)}
</section>

<section>
<h2>Accuracy by variant</h2>
{vhead}{vrows}</tbody></table></div>
<div class="prose">
<h3>Where Jev goes wrong</h3>
<p>Every mistake Jev made sits in one cell of the grid. All five are correct
specs in <code>imp-swap</code> that it rejected: three <code>ref</code> and two
<code>equiv</code>. It did not accept a bad spec anywhere, in any lane. In all
five the question that fired was <code>relation_matches_intent</code>, scoring
between 0.24 and 0.45, so Jev was claiming the postcondition does not describe
the relation the intent asks for.</p>

<h3>Those five were checked, and the labels hold</h3>
<p>A judge disagreeing with the label on exactly one lane is what a mislabelled
lane would look like, so the five were worked through by hand. Under the swap
rules <code>-</code> is addition, so <code>array-sum</code>'s
<code>sum = sum - a[i]</code> accumulates and <code>i = i - 1</code> counts up.
<code>+</code> is subtraction and <code>/</code> is multiplication, so
<code>abs-times</code>'s <code>t = 0 + a</code> gives the absolute value and
<code>res = t / b</code> multiplies. Both reference specs say what their intent
says. The labels are right and the five are real errors.</p>

<h3>The agent judges hit the ceiling</h3>
<p>Claude and Codex both scored 100% on all 45 cases, which means this suite
cannot rank them against each other and cannot measure how much harder a
defect would have to be before they slipped. Codex at a single sample missed
two <code>weakened</code> specs and the three-sample <code>all_pass</code>
aggregation recovered both, which is the only evidence here that the sampling
does any work. Harder cases are needed to say anything more about the top of
the range.</p>
</div>
</section>

<section>
<h2>Is Jev actually reading the swapped semantics?</h2>
<div class="prose">
<p><code>imp/gcd-euclid</code> and <code>imp-swap/gcd-euclid</code> turned out
to be byte-identical: same program, same reference spec. The program uses only
<code>%</code>, <code>==</code>, <code>!</code> and assignment, none of which
the swap lane redefines, so the two cases are the same question asked twice
with a different <code>semantics.k</code> sitting next to them.</p>
</div>
<pre><code>imp/gcd-euclid/ref        relation_matches_intent  0.55   accepted
imp-swap/gcd-euclid/ref   relation_matches_intent  0.44   rejected</code></pre>
<div class="prose">
<p>Identical input, opposite verdict. That looked at first like contamination:
an irrelevant semantics file dragging the score down. Two follow-up checks
say otherwise.</p>
<h3>Trimming the file doesn't help</h3>
<p>If contamination were the cause, cutting the noise should fix it. Every
case was rerun twice more: once against just the rule lines with the
grammar and prose stripped out (26 KB down to about 5 KB), and once against
only the rules for operators that actually appear in that program, which is
roughly what a judge with a grep command would choose to read. Both trims
keep every swapped rule that matters. Neither is picked per case to flatter
the result.</p>
{metric_table(trim_order, trim_configs)}
<p>Accuracy does not recover. It drifts down slightly as more context is
removed. Whatever is going wrong in <code>imp-swap</code>, it survives having
the distraction taken away, which rules out noise as the explanation.</p>
<h3>gcd-euclid is a hard problem, not a swap-specific one</h3>
<p>Checking every lane on <code>relation_matches_intent</code> instead of
just the failing case tells a different story than the pair above suggested:</p>
<pre><code>imp/gcd-euclid/ref         0.55   imp-obf/gcd-euclid/ref     0.61
imp/gcd-euclid/equiv       0.56   imp-obf/gcd-euclid/equiv   0.57
imp-swap/gcd-euclid/ref    0.44   imp-swap/gcd-euclid/equiv  0.55</code></pre>
<p>Every one of these sits close enough to 0.5 to be a coin flip. gcd-euclid
has function calls, a stack, and a locals cell on top of the state cell the
other two problems use, and Jev seems to find that structure hard to reason
about regardless of which semantics file it is reading. The imp-swap instance
landing on the wrong side of that coin flip is closer to noise than to a
finding, and the claim that this pair proved contamination does not survive
the ablation. Treat it as retracted.</p>
<h3>The real pattern is abs-times and array-sum</h3>
<p>These two are a cleaner story. Confident and correct in <code>imp</code>
and <code>imp-obf</code>, then a sharp drop in <code>imp-swap</code> that
holds up under both trims:</p>
<pre><code>                 imp    imp-obf  imp-swap
abs-times/ref    0.87   0.73     0.45
abs-times/equiv  0.86   0.75     0.41
array-sum/ref    0.89   0.90     0.27
array-sum/equiv  0.91   0.90     0.24</code></pre>
<p>Even handed exactly the two rules it needs and nothing else, Jev does not
reliably work out that <code>t = 0 + a</code> under a semantics where
<code>+</code> means subtraction still computes the absolute value. That is a
real limit on simulating substituted arithmetic, not a side effect of a big
file. It also matches the earlier lane averages: <code>imp-swap</code> stayed
the weak column under every semantics trim.</p>
</div>
</section>

<section>
<h2>What happens with the tools taken away</h2>
<div class="prose">
<p>The agent judges have an advantage Jev never gets: they can read
<code>TASK.md</code> and <code>spec.k</code> first, then grep the semantics
file for just the two or three operators the program actually uses. Jev has to
take in all of it in a single pass. So some of the gap above could be a
harness difference, not a model difference.</p>
<p>"With tools" is also quietly multi-turn, worth spelling out since it is easy
to misread next to the row below. Pulling the full transcript for one case
shows Claude at <code>num_turns: 4</code>: it calls <code>Read</code>, gets a
result, calls again, and only then writes a verdict. Codex's own event log
calls the whole episode one "turn," but inside it sits an
<code>agent_message</code> stating a plan, a <code>command_execution</code>
running <code>sed</code> over all five files, and a second
<code>agent_message</code> with the verdict, conditioned on what the first
command returned. Both are real agentic loops, so "(1 run)" in the earlier
tables meant one of the three independent samples that get averaged into
"(3 runs, must agree)", never a turn count.</p>
<p>To separate tool access from turn count, Claude and Codex were each run a
third way: genuinely one turn, one model call in and one back, tool use
switched off outright rather than left unused (<code>claude --allowedTools
""</code>; Codex confined to an empty, read-only directory with nothing on
disk to open), given the same five fields Jev receives inlined into one
prompt, and asked Jev's own <code>faithful_overall</code> question word for
word. One call each, same shape as Jev, on all 45 cases.</p>
</div>
{metric_table(fair_order, fair_configs)}
<div class="prose">
<p>Neither model needed the tools. Both are still at or near 100% with the
files taken away, so the gap between them and Jev on this suite is
capability, not access.</p>
<p>This is not a clean zero-overhead comparison, and that is worth stating
plainly rather than glossing over. A completely empty prompt, no case content
at all, still costs {overhead.get('claude', 0):,} tokens through the Claude
CLI before a single word of the case is read. Jev has no equivalent tax: it is
a bare HTTP request with nothing but the state in the body, so its roughly
14,000 tokens per case are almost entirely the case itself. The accuracy
comparison above is fair, since it isolates tool access with everything else
held constant. The raw cost comparison further down is not, and is corrected
there.</p>
</div>
</section>

<section>
<h2>What this suggests doing</h2>
<div class="prose">
<p>Jev at 28% false alarms cannot replace the judge, since more than a quarter
of correct work would come back rejected. The zero miss rate points somewhere
else. On this suite every spec Jev accepted was genuinely correct, so its
acceptances can be taken at face value and only the specs it flags need the
expensive judge.</p>
<p>That routing keeps the agent judge's verdict on every case it currently
decides, skips 29% of the agent-judge calls, and brings the mean cost per case
from 138 s down to about 99 s. The saving is real but modest, and it rests on
a zero that was measured over 27 bad specs. A bigger and nastier corpus could
move it.</p>
</div>
</section>

<section>
<h2>Cost per case</h2>
<div class="scroll"><table><thead><tr><th>judge / config</th><th>calls</th>
<th>latency</th><th>tokens</th></tr></thead><tbody>{cost_rows}</tbody></table></div>
<div class="note">Read the CLI rows next to each other rather than against
Jev directly: {overhead.get('claude', 0):,} of Claude's tokens and
{overhead.get('codex', 0):,} of Codex's are the fixed harness tax measured
above, present in every row for that judge regardless of tool access. Jev
carries no such tax, so its number is closer to the true marginal cost of the
case than the others are.</div>
</section>

<section>
<h2>Reading this</h2>
<div class="prose">
<ul>
<li>The defects are synthetic. A mutation script fails in tidier ways than a
real agent does, so doing well here is necessary but not sufficient. The
obvious next corpus is real delivered <code>spec.k</code> files from the
3,296-trial run on <code>codex-pi-bare-kit-september</code>.</li>
<li>Two of the three judges scored 100%, so the suite has a ceiling and cannot
separate them. Read the agent columns as "did not fail here" rather than as a
measurement.</li>
<li>Nine problems across three lanes is a pilot. Each per-lane figure rests on
15 cases, so treat the columns as direction rather than measurement.</li>
<li>Jev returns a probability and no reasoning. On a spec it rejects, the agent
judges also say why, and that explanation is worth something this table does
not price.</li>
<li>Every threshold is fixed at 0.5 and was not tuned on this data. Half of
Jev's correct verdicts (20 of 40) rest on a question within 0.1 of that line,
so its margin is thinner than the accuracy column suggests and a small shift in
calibration would move a lot of cases.</li>
<li>The no-tools run asked Jev's single broad question, not its three-question
version, so it does not repeat the improvement that took Jev from 80% to 89%
earlier in this report. It answers a narrower question than the headline
table: with tools out of the picture, do Claude and Codex still beat Jev on
that one shared question? They do.</li>
<li>An earlier version of this report treated the gcd-euclid pair as proof
that irrelevant context was dragging Jev's score down. The semantics-trim
ablation does not support that: accuracy does not recover when the file is
cut down, and gcd-euclid turns out to be marginal in every lane, not just the
swapped one. That claim did not hold up and is corrected above rather than
carried forward.</li>
</ul>
</div>
</section>
</div>
"""
    OUT.write_text(html)
    print(f"report -> {OUT}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
