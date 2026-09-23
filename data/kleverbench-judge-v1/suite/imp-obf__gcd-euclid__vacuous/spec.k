module GCD-EUCLID-SPEC
  imports VERIFICATION
  claim <k> 𐕀 s ( p , q ) { y 𐔵 p 𐔴 q ; 𐕁 q ; } x 𐔵 a ; y 𐔵 b ; 𐔾 (𐔹 (y 𐔸 0)) { x 𐔵 s(x, y) ; } => .K ... </k>
        <funs> .Map => s |-> fun2(p, q, { y 𐔵 p 𐔴 q ; 𐕁 q ; }) </funs>
        <stack> .List </stack> <ret> 0 </ret>
        <state> a |-> A:Int b |-> B:Int x |-> (_ => gcdSpec(A, B)) y |-> (_ => 0) </state>
    requires A >Int 0 andBool B >Int 0
     andBool A ==Int 1 andBool B ==Int 1
  claim <k> 𐔾 (𐔹 (y 𐔸 0)) { x 𐔵 s(x, y) ; } => .K ... </k>
        <funs> s |-> fun2(p, q, { y 𐔵 p 𐔴 q ; 𐕁 q ; }) </funs>
        <locals> L:Map </locals> <stack> St:List </stack> <ret> 0 </ret>
        <state> a |-> _A:Int b |-> _B:Int x |-> (X:Int => gcdSpec(X, Y)) y |-> (Y:Int => 0) </state>
    requires X >Int 0 andBool Y >=Int 0
endmodule
