// Claims module (metavariables renamed; formatting reflowed).
module ARRAY-SUM-SPEC
  imports VERIFICATION

  claim <k> sum 𐔵 0 ; i 𐔵 0 ; 𐔾 (i 𐔶 𐔿(a)) { sum 𐔵 sum 𐔰 a[i] ; i 𐔵 i 𐔰 1 ; } => .K ... </k>
        <state> a |-> P:List i |-> (_ => size(P)) sum |-> (_ => sumRange(P, 0, size(P), 0)) </state>

  claim <k> 𐔾 (i 𐔶 𐔿(a)) { sum 𐔵 sum 𐔰 a[i] ; i 𐔵 i 𐔰 1 ; } => .K ... </k>
        <state> a |-> P:List i |-> (J:Int => size(P)) sum |-> (T:Int => sumRange(P, J, size(P), T)) </state>
    requires 0 <=Int J andBool J <=Int size(P)
endmodule
