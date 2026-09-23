module ARRAY-SUM-SPEC
  imports VERIFICATION
  claim <k> sum = 0 ; i = 0 ; while (i < length(a)) { sum = sum + a[i] ; i = i + 1 ; } => .K ... </k>
        <state> a |-> A:List i |-> (_ => size(A)) sum |-> (_ => sumRange(A, 0, size(A), 0)) </state>
  claim <k> while (i < length(a)) { sum = sum + a[i] ; i = i + 1 ; } => .K ... </k>
        <state> a |-> A:List i |-> (I:Int => size(A)) sum |-> (S:Int => sumRange(A, I, size(A), S)) </state>
    requires 0 <=Int I andBool I <=Int size(A)
endmodule
