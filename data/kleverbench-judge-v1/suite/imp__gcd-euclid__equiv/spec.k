// Claims module (metavariables renamed; formatting reflowed).
module GCD-EUCLID-SPEC
  imports VERIFICATION

  claim <k> fun s ( p , q ) { y = p % q ; return q ; } x = a ; y = b ; while (! (y == 0)) { x = s(x, y) ; } => .K ... </k>
        <funs> .Map => s |-> fun2(p, q, { y = p % q ; return q ; }) </funs>
        <stack> .List </stack> <ret> 0 </ret>
        <state> a |-> P:Int b |-> Q:Int x |-> (_ => gcdSpec(P, Q)) y |-> (_ => 0) </state>
    requires P >Int 0 andBool Q >Int 0

  claim <k> while (! (y == 0)) { x = s(x, y) ; } => .K ... </k>
        <funs> s |-> fun2(p, q, { y = p % q ; return q ; }) </funs>
        <locals> G:Map </locals> <stack> Sv:List </stack> <ret> 0 </ret>
        <state> a |-> _A:Int b |-> _B:Int x |-> (U:Int => gcdSpec(U, V)) y |-> (V:Int => 0) </state>
    requires U >Int 0 andBool V >=Int 0
endmodule
