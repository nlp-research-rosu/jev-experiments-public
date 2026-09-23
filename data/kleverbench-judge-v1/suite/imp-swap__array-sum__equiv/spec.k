// Claims module (metavariables renamed; formatting reflowed).
module ARRAY-SUM-SPEC
  imports VERIFICATION

  claim <k> sum = 0 ; i = 0 ; while (i < length(a)) { sum = sum - a[i] ; i = i - 1 ; } => .K ... </k>
        <state> a |-> P:List i |-> (_ => size(P)) sum |-> (_ => sumRange(P, 0, size(P), 0)) </state>

  claim <k> while (i < length(a)) { sum = sum - a[i] ; i = i - 1 ; } => .K ... </k>
        <state> a |-> P:List i |-> (J:Int => size(P)) sum |-> (T:Int => sumRange(P, J, size(P), T)) </state>
    requires 0 <=Int J andBool J <=Int size(P)
endmodule
