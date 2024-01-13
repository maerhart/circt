// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @types
// CHECK-SAME:  (%{{.*}}: !smt.bool)
func.func @types(%arg0: !smt.bool) {
  // CHECK: %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32> {smt.some_attr}
  %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32> {smt.some_attr}
  return
}
