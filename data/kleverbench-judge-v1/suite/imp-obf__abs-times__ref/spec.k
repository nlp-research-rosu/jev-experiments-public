module ABS-TIMES-SPEC
  imports VERIFICATION
  claim <k> 𐔼 (a 𐔶 0) { t 𐔵 0 𐔱 a ; } 𐔽 { t 𐔵 a ; } res 𐔵 t 𐔲 b ; => .K ... </k>
        <state> a |-> A:Int b |-> B:Int t |-> (_ => absInt(A)) res |-> (_ => absInt(A) *Int B) </state>
endmodule
