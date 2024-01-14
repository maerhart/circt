// RUN: circt-opt %s --split-input-file --verify-diagnostics

func.func @extraction(%arg0: !smt.bv<32>) {
  // expected-error @below {{slice too big}}
  %36 = smt.bv.extract %arg0 from 20 {smt.some_attr} : (!smt.bv<32>) -> !smt.bv<16>
  return
}
