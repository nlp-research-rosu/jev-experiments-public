// Copyright (c) 2014-2018 K Team. All Rights Reserved.

/*!
\setlength{\parindent}{1em}
\title{IMP}
\author{Grigore Ro\c{s}u (\texttt{grosu@illinois.edu})}
\organization{University of Illinois at Urbana-Champaign}
*/

/*@ \section{Abstract}
This is the \K semantic definition of the classic IMP language.
IMP is considered a folklore language, without an official inventor,
and has been used in many textbooks and papers, often with slight
syntactic variations and often without being called IMP\@.  It includes
the most basic imperative language constructs, namely basic constructs
for arithmetic and Boolean expressions, and variable assignment,
conditional, while loop and sequential composition constructs for statements.
*/

module IMP-SYNTAX
  imports DOMAINS-SYNTAX
/*@ \section{Syntax}
This module defines the syntax of IMP\@.
Note that \texttt{<=} is sequentially strict and has a \LaTeX\ attribute
making it display as $\leq$, and that \texttt{\&\&} is strict only in its first
argument, because we want to give it a short-circuit semantics. */

  syntax AExp  ::= Int | Id | "?Int"
                 | Id "[" AExp "]"            [strict(2), format(%1%2%3%4)]
                 | "length" "(" Id ")"        [format(%1%2%3%4)]
                 | "-" Int                    [format(%1%2)]
                 | "(" AExp ")"               [bracket]
                 > left:
                   AExp "*" AExp              [seqstrict]
                 | AExp "/" AExp              [seqstrict, color(pink)]
                 | AExp "%" AExp              [seqstrict]
                 > left:
                   AExp "+" AExp              [seqstrict, color(pink)]
                 | AExp "-" AExp              [seqstrict]

/*@ IMP: an array literal is an \texttt{AExp} whose elements are
\texttt{AExp}s in a comma-separated list, \texttt{[ 3 , 1 , 2 ]}; \texttt{[ ]}
is the empty array.  The empty literal is a production of its own rather than
the empty case of the list, because \K's rule grammar spells an empty
syntactic list by its unit --- \texttt{[ .AExps ]} --- and a claim that quotes
a program should not have to (a claim may still write \texttt{[ .AExps ]}: it is
a second spelling of the empty literal and evaluates to the same value).  IMP's \texttt{Int} is \K's \emph{unsigned} token, so the
\texttt{-1} in \texttt{[ -1 , 0 ]} is the unary \texttt{"-" Int} production
above, an element expression like any other; and since that production binds
tightest, \texttt{[ i -1 ]} is the one-element array holding \texttt{i - 1},
exactly as \texttt{x = i -1 ;} is a subtraction.  Arrays do not nest --- the
grammar allows \texttt{[ [ 1 ] ]} but the semantics gives it no meaning. */

  syntax AExp  ::= "[" "]"
                 | "[" AExps "]"              [format(%1%2%3)]
  syntax AExps ::= NeList{AExp,","}           [format(%1%2 %3)]

  syntax BExp  ::= Bool
                 | AExp "<=" AExp             [seqstrict, color(pink)]
                 | AExp "<" AExp              [seqstrict]
                 | AExp "==" AExp             [seqstrict]
                 | "!" BExp                   [strict, color(pink)]
                 | "(" BExp ")"               [bracket]
                 > BExp "&&" BExp             [left, strict(1), color(pink)]
                 | BExp "||" BExp             [left, strict(1)]
  syntax Block ::= "{" "}"
                 | "{" Stmt "}"               [format(%1%i%n%2%d%n%3)]
  syntax Stmt  ::= Block
                 | Id "=" AExp ";"            [strict(2), color(pink), format(%1 %2 %3%4)]
                 | Id "[" AExp "]" "=" AExp ";"
                                              [seqstrict(2,3), color(pink), format(%1%2%3%4 %5 %6%7)]
                 | "return" AExp ";"          [strict, format(%1 %2%3)]
                 | "break" ";"                [format(%1%2)]
                 | "fun" Id "(" Id ")" Block
                 | "fun" Id "(" Id "," Id ")" Block
                 | Id "=" Id "(" AExp ")" ";"
                                              [strict(3), format(%1 %2 %3%4%5%6)]
                 | Id "=" Id "(" AExp "," AExp ")" ";"
                                              [seqstrict(3,4), format(%1 %2 %3%4%5 %6%7%8)]
                 | "if" "(" BExp ")"
                   Block "else" Block         [strict(1), colors(yellow, white, white, yellow), format(%1 %2%3%4 %5 %6 %7)]
                 | "while" "(" BExp ")" Block [colors(yellow,white,white), format(%1 %2%3%4 %5)]
                 > Stmt Stmt                  [left, format(%1%n%2)]

/*@ An IMP program declares a set of variables and then executes a
statement in the state obtained after initializing all those variables
to 0.  \K provides builtin support for generic syntactic lists:
$\textit{List}\{\textit{Nonterminal},\textit{terminal}\}$ stands for
\textit{terminal}-separated lists of \textit{Nonterminal} elements. */

  syntax Pgm ::= "int" Ids ";" Stmt           [format(%1 %2%3%n%4), colors(yellow,pink)]
  syntax Ids ::= List{Id,","} [format(%1%2 %3)]
endmodule

/*@ We are done with the definition of IMP's syntax.  Make sure
that you write and parse several interesting programs before you move to the
semantics.  */

/*@ \section{The symbolic integer}
\texttt{?Int} is an expression standing for an arbitrary integer: a claim
about \texttt{x = ?Int ;} says something about every value \texttt{x} could
receive.  Its one rule introduces an existential variable on the right-hand
side, which only symbolic execution can honour.  The LLVM backend --- the one
\texttt{krun} executes on --- rejects such a rule outright (``Found existential
variable not supported by concrete backend''), and it does so in a structural
check that runs before any rule attribute is read, so tagging the rule
\texttt{[symbolic]} does not help.  What both backends do honour is the same
attribute on a \emph{module}: a module marked \texttt{[symbolic]} is dropped,
imports of it included, when kompiling for a concrete backend, and kept for the
Haskell one.  So the rule lives here, in a module of its own that \texttt{IMP}
imports, and one \texttt{semantics.k} kompiles unchanged with either backend:
\texttt{--backend llvm} for \texttt{krun}, \texttt{--backend haskell} for
\texttt{kprove}.  Under \texttt{krun} a program that reaches \texttt{?Int} is
stuck, which is the right answer for a construct with no concrete meaning. */

module IMP-SYMBOLIC [symbolic]
  imports IMP-SYNTAX

  // Introduce symbolic variables
  rule ?Int => ?_I:Int
endmodule

module IMP
  imports IMP-SYNTAX
  imports IMP-SYMBOLIC
  imports BOOL
  imports INT
  imports LIST
  imports MAP
  imports SET
/*@ \section{Semantics}
This module defines the semantics of IMP\@.
Before you start adding semantic rules to a \K definition, you need to
define the basic semantic infrastructure consisting of definitions for
{\em results} and the {\em configuration}.  */

/*@ \subsection{Values and results}
IMP has three types of values, or results of computations: integers, Booleans
and Arrays. We here use the \K builtin variants for all of them.

The array: an evaluated array literal, is the \K
\texttt{List} of its elements kept between the literal's brackets,
\texttt{[ ListItem(3) ListItem(1) ListItem(2) ]} (the sort \texttt{ArrayVal}
--- not \texttt{Array}, which is the name of a hooked builtin sort in \K's
\texttt{domains.md} --- and its rules are in the array-literal section below).
It is a result only in the sense that evaluation is finished with it: the one
rule that consumes an \texttt{ArrayVal} is the assignment that stores its list
in the state.  Note what
is \emph{not} a result: a bare \texttt{List}.  A program variable holding an
array yields its list when looked up --- the lookup rule below is not sorted
--- and in IMP that list stops the computation, because nothing can be cooled
with it.  IMP keeps it that way: \texttt{x = a ;} on an array \texttt{a} is
as stuck as it always was, and no array is ever copied. */

  syntax KResult ::= Int | Bool | ArrayVal

/*@ \subsection{Configuration}
The configuration of IMP is trivial: it only contains two cells, one
for the computation and another for the state.  For good encapsulation
and clarity, we place the two cells inside another cell, the ``top'' cell
which is labeled \textsf{T}. */

  configuration <T color="yellow">
                  <k color="green"> $PGM:Pgm </k>
                  <state color="red"> .Map </state>
                  <funs> .Map </funs>
                  <locals> .Map </locals>
                  <stack> .List </stack>
                  <ret> 0 </ret>
                </T>

/*@ The configuration variable $\${\it PGM}$ tells the \K tool where to
place the program.  More precisely, the command
``\texttt{krun program}'' parses the program and places the resulting
\K abstract syntax tree in the \textsf{k} cell before invoking the
semantic rules described in the sequel.  The ``$\kdot$'' in the
\textsf{state} cell, written \texttt{.Map} in ASCII in the
\texttt{imp.k} file, is \K's way to say ``nothing''.  Technically, it
is a constant which is the unit, or identity, of all maps in \K
(similar dot units exist for other \K structures, such as lists, sets,
multi-sets, etc.).  */

/*@ \subsection{Arithmetic expressions}
The \K semantics of each arithmetic construct is defined below. */

/*@ \subsubsection{Variable lookup}
A program variable $X$ is looked up in the state by matching a binding
of the form $X \mapsto I$ in the state cell.  If such a binding does not
exist, then the rewriting process will get stuck.  Thus our semantics of
IMP disallows uses of uninitialized variables.  Note that the variable
to be looked up is the first task in the \textsf{k} cell (the cell is
closed to the left and torn to the right), while the binding can be
anywhere in the \textsf{state} cell (the cell is torn at both sides). */

  rule <k> X:Id => I ...</k> <state>... X |-> I ...</state>  [label(variable_lookup)]

/*@ \subsubsection{Arithmetic operators}
There is nothing special about these, but recall that \K's configuration
abstraction mechanism is at work here!  That means that the rewrites in the
rules below all happen at the beginning of the \textsf{k} cell.  */

  // (the rule for ?Int is in module IMP-SYMBOLIC above)

  rule I1 / I2 => I1 *Int I2
  rule I1 + I2 => I1 -Int I2
  rule - I1 => 0 -Int I1
  rule I1 * I2 => I1 /Int I2  requires I2 =/=Int 0
  rule I1 - I2 => I1 +Int I2
  rule I1 % I2 => I1 %Int I2  requires I2 =/=Int 0

/*@ \subsection{Arrays}
A one-dimensional integer array is a program variable whose state value is a
\K \texttt{List} of integers: \texttt{a |-> A:List} is the array counterpart
of a scalar's \texttt{n |-> N:Int}, so a claim binds an array exactly the way
it binds a scalar --- with one symbolic variable.

\texttt{length(a)} is the list's \texttt{size}, \texttt{a[i]} reads the
\texttt{i}-th element (0-indexed) and \texttt{a[i] = e ;} writes it.  Both
indexed operations require the index to be in bounds; out of bounds no rule
applies and the computation gets stuck, so a claim about an indexing program
must carry a \texttt{requires} strong enough to keep every index in range.

\texttt{elem(A, I)} is the element of \texttt{A} at \texttt{I} as an
\texttt{Int} (\K's own list lookup is \texttt{KItem}-sorted, hence the
projection).  It is declared total and evaluates only on a concrete list: on
a symbolic array it stays an opaque, always-defined integer, which is what
keeps the prover from forking on a ``this element might be undefined''
branch every time the program reads one.

The lemmas about \texttt{size} that a proof needs, and the recursive range
summaries a claim folds with, are not semantics: they live in the companion
\texttt{verification.k}. */

  syntax Int ::= elem(List, Int)              [function, total]
  rule elem(L, I) => {L[I]}:>Int
    requires 0 <=Int I andBool I <Int size(L)                    [concrete]

  rule <k> X:Id [ I:Int ] => elem(L, I) ...</k> <state>... X |-> L:List ...</state>
    requires 0 <=Int I andBool I <Int size(L)                    [label(array_lookup)]

  rule <k> length ( X:Id ) => size(L) ...</k> <state>... X |-> L:List ...</state>
                                                                 [label(array_length)]

  rule <k> X:Id [ I:Int ] = V:Int ; => .K ...</k>
       <state>... X |-> (L:List => L [ I <- V ]) ...</state>
    requires 0 <=Int I andBool I <Int size(L)                    [label(array_setting)]

/*@ \subsection{Array literals}
\texttt{[ e1 , \ldots , en ]} evaluates its elements left to right and is then
the list of their values; \texttt{[ ]} is \texttt{.List} at once.  The evaluation is
written out rather than left to a \texttt{strict} attribute because the
elements form a syntactic list and what the literal has to become is a \K
\texttt{List}, not a syntactic list of results.  \texttt{\#array} carries the
values so far on its left and the elements still to evaluate on its right; the
next element is scheduled in front of it, and an integer arriving there is
appended.  Only an integer is: an element that evaluates to an array --- a
nested literal, or an array variable --- meets no rule, and the computation is
stuck.  Arrays do not nest.

The evaluated literal is an \texttt{ArrayVal}, and exactly one thing can be done
with one.  Assigning it to a program variable stores the bare list in the
state, under the discipline scalar assignment has: the variable must already
be bound, and whatever it held --- the \texttt{0} of its declaration, an
earlier array, an integer --- is replaced.  \texttt{int a ; a = [ 3 , 1 , 2 ]
;} is therefore how an IMP program builds the state a claim writes as
\texttt{a |-> ListItem(3) ListItem(1) ListItem(2)}, and from then on
\texttt{a[i]}, \texttt{a[i] = e ;} and \texttt{length(a)} apply to it as to
any array.  Everywhere else an \texttt{ArrayVal} is stuck: as an operand, an
index, a returned value, a call argument, or on the right of \texttt{a[i] =
\ldots ;}, no rule takes it.  The assignment rule is the array counterpart of
the scalar one further down and, like it, looks only at \textsf{state}: a
literal assigned to a global inside a function body works, a literal assigned
to a parameter or function-local does not --- the rule for locals takes an
\texttt{Int}, IMP keeps no arrays in \textsf{locals}, and IMP does not start
to. */

  syntax ArrayVal ::= "[" List "]"
  syntax AExp     ::= ArrayVal
  syntax KItem ::= #array(List, AExps)

  rule <k> [ ] => [ .List ] ...</k>                                 [label(array_literal_empty)]
  rule <k> [ Es:AExps ] => #array(.List, Es) ...</k>                [label(array_literal)]
  rule <k> #array(L, (E:AExp , Es)) => E ~> #array(L, Es) ...</k>    [label(array_literal_next)]
  rule <k> I:Int ~> #array(L, Es) => #array(L ListItem(I), Es) ...</k>
                                                                  [label(array_literal_elem)]
  rule <k> #array(L, .AExps) => [ L ] ...</k>                       [label(array_literal_done)]

  rule <k> X = [ L:List ] ; => .K ...</k> <state>... X |-> (_ => L) ...</state>
                                                                  [label(array_variable_setting)]

/*@ \subsection{Functions}
A function takes one or two integer parameters, returns an integer, and may
recurse.  A call is a statement and only a statement --- \texttt{x = f(e) ;}
--- never a subexpression, so \texttt{return f(y) + f(y) ;} is written
\texttt{t = f(y) ; u = f(y) ; return t + u ;}.  This is the call discipline
Dafny uses for methods.  It is what lets a claim about a call rewrite to
\texttt{.K} like every other claim here: a claim whose right-hand side is a
bare value would be matched by the first value any nested call happened to
return, and the proof would stop there.

A call gives the callee a fresh \textsf{locals} cell holding just its
parameters and pushes the caller's locals on the \textsf{stack};
\texttt{return} pops them.  The continuation stays in the \textsf{k} cell
behind a \texttt{\#done} marker --- moving it to the stack would empty the
\textsf{k} cell of the frame a claim's \texttt{...} stands for, and the
claim's own destination would then match the callee's body.  A pending
\texttt{return} discards the rest of the body one statement at a time until
that marker.

Globals live in \textsf{state} throughout, so a function reads and writes
them --- global arrays included --- exactly as top-level code does, and those
writes outlive the call.  A name that is a global is the global; any other
name is local to the call, and assigning to one that does not exist yet
creates it.  The side condition is on \textsf{state}, which a claim always
pins, never on \textsf{locals}, which a claim mentioning no function leaves
symbolic --- that is what keeps the two rules for globals the same rules they
always were, and every claim written before functions existed still proves
untouched.  A parameter or function-local may therefore not share a name with
a global: the global would win silently, so the harness rejects such a
program instead.

The value crosses back to the caller through the \textsf{ret} cell, and
\texttt{\#store} resets it to 0 once the caller has it.  So a contract is
stated about \texttt{invoke(f, N)} alone --- no destination variable in it,
one contract serving every call site --- and every other claim over code
containing a call carries the fixed line \texttt{<ret> 0 </ret>}. */

  syntax KItem ::= fun1(Id, Block) | fun2(Id, Id, Block)
  syntax KItem ::= invoke(Id, Int) | invoke(Id, Int, Int)
  syntax KItem ::= "#store" "(" Id ")" | "#done"

  rule <k> fun F:Id ( X:Id ) B:Block => .K ...</k>
       <funs> M => M [ F <- fun1(X, B) ] </funs>                  [label(fun_decl_1)]
  rule <k> fun F:Id ( X:Id , Y:Id ) B:Block => .K ...</k>
       <funs> M => M [ F <- fun2(X, Y, B) ] </funs>               [label(fun_decl_2)]

  rule <k> X:Id => I ...</k> <locals>... X |-> I ...</locals> <state> S </state>
    requires notBool (X in keys(S))                              [label(local_lookup)]
  rule <k> X:Id = I:Int; => .K ...</k>
       <locals> L => L [ X <- I ] </locals> <state> S </state>
    requires notBool (X in keys(S))                              [label(local_setting)]

  rule <k> Z:Id = F:Id ( V:Int ) ; => invoke(F, V) ~> #store(Z) ...</k>
                                                                  [label(call_1)]
  rule <k> Z:Id = F:Id ( V1:Int , V2:Int ) ; => invoke(F, V1, V2) ~> #store(Z) ...</k>
                                                                  [label(call_2)]

  rule <k> invoke(F:Id, V:Int) => B ~> #done ...</k>
       <funs>... F |-> fun1(P, B) ...</funs>
       <locals> L => P |-> V </locals>
       <stack> .List => ListItem(L) ... </stack>                  [label(enter_1)]
  rule <k> invoke(F:Id, V1:Int, V2:Int) => B ~> #done ...</k>
       <funs>... F |-> fun2(P, Q, B) ...</funs>
       <locals> L => P |-> V1 Q |-> V2 </locals>
       <stack> .List => ListItem(L) ... </stack>                  [label(enter_2)]

  rule <k> return _:Int ; ~> (_:Stmt => .K) ...</k>               [label(return_skip)]
  rule <k> return V:Int ; ~> #done => .K ...</k>
       <ret> _ => V </ret>
       <locals> _ => L </locals>
       <stack> ListItem(L) => .List ... </stack>                  [label(return)]

  rule <k> #store(Z:Id) => Z = V ; ...</k> <ret> V => 0 </ret>    [label(store)]

/*@ \subsection{Boolean expressions}
The rules below are straightforward.  Note the short-circuited semantics
of \texttt{\&\&}; this is the reason we annotated the syntax of
\texttt{\&\&} with the \K attribute \texttt{strict(1)} instead of
\texttt{strict}. */

  rule I1 <= I2 => I1 <=Int I2
  rule I1 < I2 => I1 <Int I2
  rule I1 == I2 => I1 ==Int I2
  rule ! T => notBool T
  rule true && _ => true
  rule false && B => B
  rule true || B => B
  rule false || _ => false

/*@ \subsection{Blocks and Statements}
There is one rule per statement construct except for the conditional,
which needs two rules. */

/*@ \subsubsection{Blocks}
The empty block \texttt{\{\}} is simply dissolved.  The $\kdot$ below is the
unit of the computation list structure $K$, that is, the empty task.
Similarly, the non-empty blocks are dissolved and replaced by their statement
contents, thus effectively giving them a bracket semantics; we can afford to
do this only because we have no block-local variable declarations yet in IMP.
Since we tagged the rules below as "structural", the \K tool structurally
erases the block constructs from the computation structure, without
considering their erasure as computational steps in the resulting transition
systems.  You can make these rules computational (dropping the attribute
\texttt{structural}) if you do want these to count as computational steps. */

  rule {} => .K
  rule {S} => S

/*@ \subsubsection{Assignment}
The assigned variable is updated in the state.  The variable is expected
to be declared, otherwise the semantics will get stuck.  At the same time,
the assignment is dissolved. */

  rule <k> X = I:Int; => .K ...</k> <state>... X |-> (_ => I) ...</state>  [label(variable_setting)]

/*@ \subsubsection{Sequential composition}
Sequential composition is simply structurally translated into \K's
builtin task sequentialization operation.  You can make this rule
computational (i.e., remove the \texttt{structural} attribute) if you
want it to count as a computational step.  Recall that the semantics
of a program in a programming language defined in \K is the transition
system obtained from the initial configuration holding that program
and counting only the steps corresponding to computational rules as
transitions (i.e., hiding the structural rules as unobservable, or
internal steps). */

  rule S1:Stmt S2:Stmt => S1 ~> S2

/*@ \subsubsection{Conditional}
The conditional statement has two semantic cases, corresponding to
when its condition evaluates to \texttt{true} or to \texttt{false}.
Recall that the conditional was annotated with the attribute
\texttt{strict(1)} in the syntax module above, so only its first
argument is allowed to be evaluated. */

  rule if (true)  S else _ => S
  rule if (false) _ else S => S

/*@ \subsubsection{While loop and \texttt{break}}
We give the semantics of the \texttt{while} loop by unrolling.
Note that we preferred to make the rule below structural.

\texttt{break ;} abandons the rest of the current loop body and the loop
itself; execution resumes after the loop.  It needs somewhere to resume to,
so a loop whose body contains a \texttt{break} --- and only such a loop ---
unrolls with an end-of-loop marker \texttt{\#loop} pushed behind it.
\texttt{hasBreak} looks through blocks, sequences and conditionals but stops
at a nested \texttt{while}: a \texttt{break} in there belongs to that loop,
not to this one.  A loop without a \texttt{break} unrolls exactly as it
always did, into exactly the same continuation, so every claim written
before \texttt{break} existed still proves step for step.

\texttt{break ;} then discards the computation one \texttt{Stmt} at a time
--- the recursive \texttt{while} that trails the body included, and any
nested loop still standing in the body's tail, discarded whole and never
entered --- until it reaches the marker, which it consumes.  Reached the
ordinary way, by the guard going false or by the body running out, the
marker just dissolves.

Only a \texttt{Stmt} is discarded, so nothing crosses a function boundary:
\texttt{\#done} stops the scan.  A \texttt{break} in a function body that is
not inside a loop \emph{of that body}, and a \texttt{break} outside any loop,
are therefore stuck --- there is no rule for either, which is the answer, not
silently continuing.  \texttt{return}, symmetrically, steps over a marker on
its way out of a loop inside a function body.

The marker is pushed once per unrolling rather than once per loop, so a
\texttt{break}-loop that has run $n$ iterations carries $n$ markers and the
surplus ones dissolve behind the one the exit consumes.  That is the price of
keeping the recursive occurrence spelled \texttt{while (B) S}: a loop claim
closes circularly exactly when the state it returns to is the loop again,
and a proof unrolls once before it closes, so one extra marker is all the
prover ever sees. */

  syntax KItem ::= "#loop"

  syntax Bool ::= hasBreak(Stmt)              [function, total]
  rule hasBreak(break ;) => true
  rule hasBreak({ S:Stmt }) => hasBreak(S)
  rule hasBreak(S1:Stmt S2:Stmt) => hasBreak(S1) orBool hasBreak(S2)
  rule hasBreak(if (_) B1:Block else B2:Block) => hasBreak(B1) orBool hasBreak(B2)
  rule hasBreak(_) => false                                      [owise]

  rule while (B) S => if (B) {S while (B) S} else {}
    requires notBool hasBreak(S)                                 [label(while)]
  rule <k> while (B) S => if (B) {S while (B) S} else {} ~> #loop ...</k>
    requires hasBreak(S)                                   [label(while_break)]

  rule <k> break ; ~> (_:Stmt => .K) ...</k>                [label(break_skip)]
  rule <k> break ; ~> #loop => .K ...</k>                        [label(break)]
  rule <k> #loop => .K ...</k>                               [label(loop_exit)]
  rule <k> return _:Int ; ~> (#loop => .K) ...</k>    [label(return_skip_loop)]

/*@ \subsection{Programs}
The semantics of an IMP program is that its body statement is executed
in a state initializing all its global variables to 0.  Since \K's
syntactic lists are internally interpreted as cons-lists (i.e., lists
constructed with a head element followed by a tail list), we need to
distinguish two cases, one when the list has at least one element and
another when the list is empty.  In the first case we initialize the
variable to 0 in the state, but only when the variable is not already
declared (all variables are global and distinct in IMP).  We prefer to
make the second rule structural, thinking of dissolving the residual
empty \texttt{int;} declaration as a structural cleanup rather than as
a computational step. */

  rule <k> int (X,Xs => Xs);_ </k> <state> Rho:Map (.Map => X|->0) </state>
    requires notBool (X in keys(Rho))
  rule int .Ids; S => S

endmodule
