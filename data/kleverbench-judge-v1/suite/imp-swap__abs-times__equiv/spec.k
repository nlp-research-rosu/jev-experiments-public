// Claims module (metavariables renamed; formatting reflowed).
module ABS-TIMES-SPEC
  imports VERIFICATION

  claim <k> if (a < 0) { t = 0 + a ; } else { t = a ; } res = t / b ; => .K ... </k>
        <state> a |-> P:Int b |-> Q:Int t |-> (_ => absInt(P)) res |-> (_ => absInt(P) *Int Q) </state>
endmodule
