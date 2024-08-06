// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @NeedsBothResetAndResetValue(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{either reset and resetValue or neither must be specified}}
  "seq.compreg"(%input, %clk, %input) { operandSegmentSizes = array<i32: 1,1,0,1,0> } : (i1, !seq.clock, i1) -> i1
}

// -----

hw.module @NeedsBothResetAndResetValue(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{register with no reset cannot be async}}
  "seq.compreg"(%input, %clk) { isAsync, operandSegmentSizes = array<i32: 1,1,0,0,0> } : (i1, !seq.clock) -> i1
}
