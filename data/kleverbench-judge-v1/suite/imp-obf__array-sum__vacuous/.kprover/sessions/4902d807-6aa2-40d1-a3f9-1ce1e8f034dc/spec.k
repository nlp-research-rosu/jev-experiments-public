module ARRAY-SUM-SPEC
  imports VERIFICATION
  claim <k> sum 𐔵 0 ; i 𐔵 0 ; 𐔾 (i 𐔶 𐔿(a)) { sum 𐔵 sum 𐔰 a[i] ; i 𐔵 i 𐔰 1 ; } => .K ... </k>
        <state> a |-> A:List i |-> (_ => size(A)) sum |-> (_ => sumRange(A, 0, size(A), 0)) </state>
    requires size(A) ==Int 1
  claim <k> 𐔾 (i 𐔶 𐔿(a)) { sum 𐔵 sum 𐔰 a[i] ; i 𐔵 i 𐔰 1 ; } => .K ... </k>
        <state> a |-> A:List i |-> (I:Int => size(A)) sum |-> (S:Int => sumRange(A, I, size(A), S)) </state>
    requires 0 <=Int I andBool I <=Int size(A)
endmodule
