module ABS-TIMES-SPEC
  imports VERIFICATION
  claim <k> if (a < 0) { t = 0 - a ; } else { t = a ; } res = t * b ; => .K ... </k>
        <state> a |-> A:Int b |-> B:Int t |-> (_ => absInt(A)) res |-> (_ => absInt(A) *Int B) </state>
endmodule
