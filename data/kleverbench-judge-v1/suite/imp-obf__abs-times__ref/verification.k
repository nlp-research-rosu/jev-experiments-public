requires "semantics.k"

/*@ \section{Verification}
This module is not part of IMP: it is the vocabulary a \emph{claim} needs.
A claims module may contain no \texttt{syntax} or \texttt{rule} items of its
own, so everything a proof needs beyond the language --- the tokens a claim
names program variables with, the lemmas the prover cannot derive, and the
recursive summaries a loop invariant folds with --- is declared here, once
per language, where it is auditable.

\texttt{verification.k} is the file \texttt{kompile} is handed
(\texttt{--main-module VERIFICATION --syntax-module IMP-SYNTAX}); it
\texttt{requires} the semantics, so a language is still kompiled once, for
every problem.  A claims module starts \texttt{imports VERIFICATION}. */

module VERIFICATION
  imports IMP

/*@ \subsection{Program-variable tokens}
\K's generic \texttt{Id} lexer is not available inside rules and claims: a
bare lowercase word there is read as part of the rule grammar, not as a
program identifier, and a claim mentioning the program variable \texttt{n}
does not parse unless \texttt{n} is a declared token.  So every variable name
a claim may mention is declared below.  This is proof vocabulary, not
language --- an IMP \emph{program} may declare any identifier the generic
lexer accepts. */

  syntax Id ::= "n" [token] | "sum" [token] | "a" [token] | "b" [token]
              | "max" [token] | "r" [token]
  // KleverBench extension: more program variables for larger problems
  // (`k` deliberately omitted: it stays visually unambiguous next to the
  // <k> cell in claims)
  syntax Id ::= "i" [token] | "j" [token] | "x" [token] | "y" [token]
              | "z" [token] | "t" [token] | "m" [token] | "count" [token]
              | "res" [token] | "tmp" [token]
  // KleverBench extension: function names, parameters and function-locals
  syntax Id ::= "f" [token] | "g" [token] | "h" [token] | "s" [token]
              | "d" [token] | "p" [token] | "q" [token] | "u" [token]
              | "lo" [token] | "hi" [token] | "acc" [token]

/*@ \subsection{Lemmas about \texttt{size}}
An in-range write changes elements, never the length; and a length is never
negative.  Neither is derivable from \K's list hooks, and the prover reasons
about \texttt{size(A)} as an opaque integer, so both have to be said out
loud or every array proof splits on an impossible ``array of negative
length'' branch. */

  rule size(L:List [ I:Int <- _ ]) => size(L)
    requires 0 <=Int I andBool I <Int size(L)                    [simplification]

  rule 0 <=Int size(_L:List) => true                             [simplification]
  rule size(_L:List) >=Int 0 => true                             [simplification]
  rule size(_L:List) <Int 0 => false                             [simplification]
  rule 0 >Int size(_L:List) => false                             [simplification]

/*@ \subsection{Summary functions}
The recursive summaries an array loop needs to state its invariant.  Each
folds over a half-open index range \texttt{[I, N)} of a list, carrying the
accumulator a loop would carry; on an empty range each returns its
accumulator unchanged.

\texttt{sumRange(A, I, N, S)} is $S + \sum_{j=I}^{N-1} A[j]$;
\texttt{maxRange(A, I, N, M)} / \texttt{minRange(A, I, N, M)} are the largest
/ smallest of \texttt{M} and $A[I], \ldots, A[N-1]$;
\texttt{addRange(A, I, N, K)} is \texttt{A} with \texttt{K} added to every
element in the range; \texttt{setRange(A, I, N, V)} is \texttt{A} with every
element in the range replaced by \texttt{V}.  The two list-valued ones leave
every index outside \texttt{[I, N)} exactly as it was --- that is how a claim
frames the part of an array a loop does not touch.

Every step rule carries the same in-bounds guard the indexed statements do,
\texttt{0 <=Int I andBool I <Int size(L)}.  That is not decoration: \K's
list update \texttt{L[I <- V]} is partial, so an unguarded step would let a
\texttt{total} symbol rewrite to $\bot$ --- an inconsistency inside the one
block \texttt{kprove} trusts.  With the guard, a fold whose range leaves the
array is simply stuck: no rule applies, and the term stays an opaque (but
defined) value, which is what \texttt{total} claims and nothing more. */

  syntax Int ::= sumRange(List, Int, Int, Int)     [function, total, no-evaluators]
  rule sumRange(_, I, N, S) => S                 requires I >=Int N  [simplification]
  rule sumRange(L, I, N, S) => sumRange(L, I +Int 1, N, S +Int elem(L, I))
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)      [simplification]

  syntax Int ::= maxRange(List, Int, Int, Int)     [function, total, no-evaluators]
  rule maxRange(_, I, N, M) => M                 requires I >=Int N  [simplification]
  rule maxRange(L, I, N, M) => maxRange(L, I +Int 1, N, elem(L, I))
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)
     andBool M <Int elem(L, I)                                      [simplification]
  rule maxRange(L, I, N, M) => maxRange(L, I +Int 1, N, M)
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)
     andBool elem(L, I) <=Int M                                     [simplification]

  syntax Int ::= minRange(List, Int, Int, Int)     [function, total, no-evaluators]
  rule minRange(_, I, N, M) => M                 requires I >=Int N  [simplification]
  rule minRange(L, I, N, M) => minRange(L, I +Int 1, N, elem(L, I))
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)
     andBool elem(L, I) <Int M                                      [simplification]
  rule minRange(L, I, N, M) => minRange(L, I +Int 1, N, M)
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)
     andBool M <=Int elem(L, I)                                     [simplification]

  syntax List ::= addRange(List, Int, Int, Int)    [function, total, no-evaluators]
  rule addRange(L, I, N, _) => L                 requires I >=Int N  [simplification]
  rule addRange(L, I, N, K) => addRange(L [ I <- elem(L, I) +Int K ], I +Int 1, N, K)
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)      [simplification]

  syntax List ::= setRange(List, Int, Int, Int)    [function, total, no-evaluators]
  rule setRange(L, I, N, _) => L                 requires I >=Int N  [simplification]
  rule setRange(L, I, N, V) => setRange(L [ I <- V ], I +Int 1, N, V)
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)      [simplification]

// ---- scalar block problems ----

/*@ \subsection{Scalar-block spec functions}
The recursive spec functions of the scalar block, each transliterated from the
\emph{source's own} definition --- the \texttt{spec fn} / \texttt{function}
that the source's postcondition is stated against --- and nothing more.  A
source that writes its definition as a chain of \texttt{if}s becomes one
guarded rule per arm here, the way \texttt{maxRange} above splits;
that is a change of form, not of content, and
\texttt{tests/test\_scalar\_vocabulary.py} pins each one against the source
definition on concrete inputs.

Where a source states its function in the non-tail form, the first two carry
the accumulator their loop carries, exactly as the range folds above do, so
that the invariant's postcondition and one unrolling of the loop body are the
\emph{same term} rather than two associations of one sum; the source's own
function is then the accumulator started at its unit.  The last three need no
such treatment: their sources are already tail-recursive or their loop's
invariant is already absolute.

\texttt{countFactors(A, B, Idx, C)} is \texttt{C} plus the number of common
divisors of \texttt{A} and \texttt{B} in $1 \ldots \mathit{Idx}$ (VeriContest
\texttt{lc2427}'s \texttt{count\_factors} is \texttt{countFactors(a, b, idx,
0)});
\texttt{trailingZeroes(N, S)} is \texttt{S} plus $\lfloor N/5 \rfloor +
\lfloor N/25 \rfloor + \cdots$, the number of trailing zeroes of $N!$
(\texttt{lc172}'s \texttt{trailing\_zeroes\_spec} is
\texttt{trailingZeroes(n, 0)});
\texttt{reverseHelper(X, Acc)} pushes the decimal digits of \texttt{X} onto
\texttt{Acc} (ReForm-DafnyComp \texttt{DC-01}'s \texttt{reverseHelper}, whose
source definition is already accumulator-carrying);
\texttt{gcdSpec(A, B)} is Euclid's greatest common divisor
(ReForm-DafnyComp \texttt{DC-07}'s \texttt{gcd\_spec}, also already
tail-recursive);
\texttt{tribo(N)} is the $N$-th tribonacci number (\texttt{lc1137}'s
\texttt{tribo\_spec}), whose four cases are its four rules.

Every step rule carries the guard that makes the arithmetic defined ---
\texttt{\%Int} and \texttt{/Int} are partial at a zero divisor, and each
source states its own \texttt{decreases}/\texttt{requires} to the same effect
--- so a fold outside its domain is stuck rather than rewriting a
\texttt{total} symbol to $\bot$. */

  syntax Int ::= countFactors(Int, Int, Int, Int) [function, total, no-evaluators]
  rule countFactors(_, _, Idx, C) => C           requires Idx <=Int 0  [simplification]
  rule countFactors(A, B, Idx, C) => countFactors(A, B, Idx -Int 1, C +Int 1)
    requires Idx >Int 0
     andBool A %Int Idx ==Int 0 andBool B %Int Idx ==Int 0            [simplification]
  rule countFactors(A, B, Idx, C) => countFactors(A, B, Idx -Int 1, C)
    requires Idx >Int 0
     andBool notBool (A %Int Idx ==Int 0 andBool B %Int Idx ==Int 0)  [simplification]

  syntax Int ::= trailingZeroes(Int, Int)         [function, total, no-evaluators]
  rule trailingZeroes(N, S) => S                 requires N <=Int 0    [simplification]
  rule trailingZeroes(N, S) => trailingZeroes(N /Int 5, S +Int (N /Int 5))
    requires N >Int 0                                                 [simplification]

  syntax Int ::= reverseHelper(Int, Int)          [function, total, no-evaluators]
  rule reverseHelper(X, Acc) => Acc              requires X ==Int 0    [simplification]
  rule reverseHelper(X, Acc) => reverseHelper(X /Int 10, (Acc *Int 10) +Int (X %Int 10))
    requires X >Int 0                                                 [simplification]

  syntax Int ::= gcdSpec(Int, Int)                [function, total, no-evaluators]
  rule gcdSpec(A, B) => A                        requires B ==Int 0    [simplification]
  rule gcdSpec(A, B) => gcdSpec(B, A %Int B)     requires B >Int 0     [simplification]

  syntax Int ::= tribo(Int)                       [function, total, no-evaluators]
  rule tribo(N) => 0                             requires N <=Int 0    [simplification]
  rule tribo(N) => 1                             requires N ==Int 1    [simplification]
  rule tribo(N) => 1                             requires N ==Int 2    [simplification]
  rule tribo(N) => (tribo(N -Int 3) +Int tribo(N -Int 2)) +Int tribo(N -Int 1)
    requires N >Int 2                                                 [simplification]

/*@ \subsection{An encoding identity for \texttt{maxInt}}
The one rule pair here is not a fact about any problem's functions: it is one
of \K's own builtin \texttt{Int} operations, resolved under a known ordering
of its arguments.  \texttt{tribonacci}'s final loop counter really is
$\max(2, n)$ --- the loop leaves it at \texttt{n}, the two early-return
branches leave it at its initial \texttt{2} --- so its main claim says
\texttt{maxInt(2, N)}, and on the branch where the loop ran the prover has to
see that this is \texttt{N}.  It does not derive that from
$0 \le N \le 37 \wedge N \neq 0 \wedge N \neq 1$ on its own.  \texttt{minInt}
needs no counterpart: \texttt{common-factors} uses it in the same position and
closes without one, because there the branch condition compares the two
arguments directly. */

  rule maxInt(A, B) => B                         requires A <=Int B    [simplification]
  rule maxInt(A, B) => A                         requires B <=Int A    [simplification]

  // ---- array-input block A problems ----

/*@ \subsection{Prefix maximum and peek sum}
Vocabulary for \texttt{peek-sum} (DafnyBench \texttt{mPeekSum}).  Both fold
\emph{downward}, from the top index, which is the shape the source's own
recursive definitions have; a prefix-form loop invariant therefore steps with
one unrolling, the way the accumulator folds above do for a suffix-form one.

\texttt{prefixMax(A, I)} is the largest of $A[0], \ldots, A[I-1]$.  It is the
functional form of the source invariant's two conjuncts about the running
maximum --- \texttt{lmax in v[..i]} and
\texttt{forall k :: 0<=k<i ==> lmax>=v[k]} --- which together determine that
value and nothing else.

\texttt{peekSum(A, I)} is the source's \texttt{peekSum(v, i)}: the sum of the
elements of $A[0..I)$ that are at least as large as every element before them.
The source spells that test as the predicate \texttt{isPeek(v, j)},
\texttt{forall k :: 0<=k<j ==> v[j]>=v[k]}; here it is
\texttt{prefixMax(A, J) <=Int elem(A, J)} --- the same function of
\texttt{(A, J)}, on concrete inputs as well as symbolic ones
(\texttt{tests/test\_kexpr.py} evaluates both against the source definitions).

Each step rule carries the in-bounds guard the indexed statements do, for the
same reason the range folds above do: outside it the term is stuck, never
bottom. */

  syntax Int ::= prefixMax(List, Int)              [function, total, no-evaluators]
  rule prefixMax(L, I) => elem(L, 0)
    requires I <=Int 1 andBool 1 <=Int size(L)                      [simplification]
  rule prefixMax(L, I) => maxInt(elem(L, I -Int 1), prefixMax(L, I -Int 1))
    requires I >Int 1 andBool I <=Int size(L)                       [simplification]

  syntax Int ::= peekSum(List, Int)                [function, total, no-evaluators]
  rule peekSum(_, I) => 0                        requires I <=Int 0 [simplification]
  rule peekSum(L, I) => elem(L, I -Int 1) +Int peekSum(L, I -Int 1)
    requires I >Int 0 andBool I <=Int size(L)
     andBool prefixMax(L, I -Int 1) <=Int elem(L, I -Int 1)         [simplification]
  rule peekSum(L, I) => peekSum(L, I -Int 1)
    requires I >Int 0 andBool I <=Int size(L)
     andBool elem(L, I -Int 1) <Int prefixMax(L, I -Int 1)          [simplification]

/*@ \subsection{Array-input block B}
Vocabulary added for the problems translated from the array-input gold block
B (2026-09-02).  No new program-variable token was needed: the block's one
admitted problem uses \texttt{a} and \texttt{i}, both already declared above.

\texttt{swapEnds(A, I, N)} is \texttt{A} with, for every $j$ in
$[I, N)$, the elements at $j$ and at $\texttt{size}(A) - 1 - j$ exchanged;
outside those pairs every index keeps the value it had.  It is the
index-recursive form of the element-wise ``reversed'' postcondition an
in-place reverse states: with $N = \texttt{size}(A)/2$ and $I = 0$ it is the
reversal of \texttt{A}.  Like the other list-valued folds it carries no
accumulator beyond the list itself, and its step is guarded by the same
in-bounds condition the indexed statements carry --- the mirror index
$\texttt{size}(L) - 1 - I$ is in range whenever $I$ is.

One encoding lemma comes with it.  A loop whose bound is \texttt{length(a) /
2} puts \texttt{size(A) /Int 2} where the existing \texttt{size} lemmas
expect a bare \texttt{size(A)}, so ``a length is never negative'' no longer
reaches the term that needs it and a main claim cannot even discharge its
loop claim's \texttt{0 <=Int I andBool I <=Int size(A) /Int 2}.  The rule
below is the same statement one division on: \K's \texttt{/Int} is
t-division, so a non-negative numerator over a positive divisor is
non-negative.  It is an identity of the builtins, not a fact about any
problem. */

  rule 0 <=Int size(_L:List) /Int M:Int => true
    requires M >Int 0                                              [simplification]

  syntax List ::= swapEnds(List, Int, Int)        [function, total, no-evaluators]
  rule swapEnds(L, I, N) => L                    requires I >=Int N  [simplification]
  rule swapEnds(L, I, N) => swapEnds(L [ I <- elem(L, size(L) -Int 1 -Int I) ]
                                       [ size(L) -Int 1 -Int I <- elem(L, I) ],
                                     I +Int 1, N)
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)      [simplification]

// ---- array-written block problems ----
/*@ \subsection{Array-written block problems}
Vocabulary for the problems translated from gold verification artefacts whose
loops \emph{write} an array with a closed-form value.  One more program-variable
token, and one more array-valued fold.

\texttt{cubeRange(A, I, N)} is \texttt{A} with index $j$ replaced by $j^3$ for
every $j$ in $[I, N)$, every other index untouched.  It is the recursive form of
the quantified property \texttt{forall j :: I <= j < N ==> a[j] == j*j*j} --- a
change of form, nothing more --- and it folds exactly the way the loop it
summarises does, one index per step.  Same shape and the same in-bounds guard as
\texttt{setRange}: outside \texttt{0 <= I < size(L)} no rule applies and the
term is stuck rather than bottom. */

  syntax Id ::= "c" [token]

  syntax List ::= cubeRange(List, Int, Int)       [function, total, no-evaluators]
  rule cubeRange(L, I, N) => L                   requires I >=Int N  [simplification]
  rule cubeRange(L, I, N) => cubeRange(L [ I <- I *Int I *Int I ], I +Int 1, N)
    requires I <Int N andBool 0 <=Int I andBool I <Int size(L)      [simplification]

endmodule
