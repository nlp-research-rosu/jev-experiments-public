module GCD-EUCLID-SPEC
  imports VERIFICATION
  claim <k> fun s ( p , q ) { y = p % q ; return q ; } x = a ; y = b ; while (! (y == 0)) { x = s(x, y) ; } => .K ... </k>
        <funs> .Map => s |-> fun2(p, q, { y = p % q ; return q ; }) </funs>
        <stack> .List </stack> <ret> 0 </ret>
        <state> a |-> A:Int b |-> B:Int x |-> (_ => _) y |-> (_ => _) </state>
    requires A >Int 0 andBool B >Int 0
  claim <k> while (! (y == 0)) { x = s(x, y) ; } => .K ... </k>
        <funs> s |-> fun2(p, q, { y = p % q ; return q ; }) </funs>
        <locals> L:Map </locals> <stack> St:List </stack> <ret> 0 </ret>
        <state> a |-> _A:Int b |-> _B:Int x |-> (X:Int => gcdSpec(X, Y)) y |-> (Y:Int => 0) </state>
    requires X >Int 0 andBool Y >=Int 0
endmodule
