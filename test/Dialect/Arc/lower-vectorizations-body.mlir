// RUN: circt-opt %s --arc-lower-vectorizations=mode=body -split-input-file | FileCheck %s

// hw.module @vectorize_body_already_lowered(%in0: i1, %in1: i1, %in2: i1, %in3: i1, %in4: vector<4xi1>, %in5: vector<4xi1>) -> (out0: i1, out1: i1, out2: vector<4xi1>) {
hw.module @vectorize_body_already_lowered(%in0: i1, %in1: i1, %in2: i1, %in3: i1, %in4: vector<4xi1>, %in5: vector<4xi1>) -> (out0: i1, out1: i1) {
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }

  // %1 = arc.vectorize (%in4), (%in5) : (vector<4xi1>, vector<4xi1>) -> vector<4xi1> {
  // ^bb0(%arg0: i1, %arg1: i1):
  //   %1 = arith.andi %arg0, %arg1 : i1
  //   arc.vectorize.return %1 :i1
  // }

  // hw.output %0#0, %0#1, %1 : i1, i1, vector<4xi1>
  hw.output %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: hw.module @vectorize_body_already_lowered
//       CHECK: arc.vectorize ({{.*}}, {{.*}}), ({{.*}}, {{.*}})
//       CHECK: ^bb0([[ARG0:%.+]]: i2, [[ARG1:%.+]]: i2):
//       CHECK:   [[V0:%.+]] = comb.and [[ARG0]], [[ARG1]] : i2
//       CHECK:   arc.vectorize.return [[V0]] : i2
