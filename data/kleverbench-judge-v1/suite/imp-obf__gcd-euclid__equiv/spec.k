// Claims module (metavariables renamed; formatting reflowed).
module GCD-EUCLID-SPEC
  imports VERIFICATION

  claim <k> 𐕀 s ( p , q ) { y 𐔵 p 𐔴 q ; 𐕁 q ; } x 𐔵 a ; y 𐔵 b ; 𐔾 (𐔹 (y 𐔸 0)) { x 𐔵 s(x, y) ; } => .K ... </k>
        <funs> .Map => s |-> fun2(p, q, { y 𐔵 p 𐔴 q ; 𐕁 q ; }) </funs>
        <stack> .List </stack> <ret> 0 </ret>
        <state> a |-> P:Int b |-> Q:Int x |-> (_ => gcdSpec(P, Q)) y |-> (_ => 0) </state>
    requires P >Int 0 andBool Q >Int 0

  claim <k> 𐔾 (𐔹 (y 𐔸 0)) { x 𐔵 s(x, y) ; } => .K ... </k>
        <funs> s |-> fun2(p, q, { y 𐔵 p 𐔴 q ; 𐕁 q ; }) </funs>
        <locals> G:Map </locals> <stack> Sv:List </stack> <ret> 0 </ret>
        <state> a |-> _A:Int b |-> _B:Int x |-> (U:Int => gcdSpec(U, V)) y |-> (V:Int => 0) </state>
    requires U >Int 0 andBool V >=Int 0
endmodule
