// RUN: circt-opt %s --arc-vectorize | FileCheck %s

hw.module @vectorize(%clk: i1, %in0: i1, %in1: i4, %in2: i1, %in3: i1) -> (out0: i5, out1: i5, out2: i5, out3: i5, out4: i5, out5: i5, out6: i5, out7: i5, out8: i5) {
  %0:3 = arc.clock_domain (%in0, %in1, %in2, %in3) clock %clk : (i1, i4, i1, i1) -> (i5, i5, i5) {
  ^bb0(%arg0: i1, %arg1: i4, %arg2: i1, %arg3: i1):
    %0 = arc.state @dummyArc(%arg0, %arg1) lat 1 : (i1, i4) -> i5
    %1 = arc.state @dummyArc(%arg2, %arg1) lat 1 : (i1, i4) -> i5
    %2 = arc.state @dummyArc(%arg3, %arg1) lat 1 : (i1, i4) -> i5
    arc.output %0, %1, %2 : i5, i5, i5
  }
  %1:3 = arc.clock_domain (%in0, %in1, %in2, %in3) clock %clk : (i1, i4, i1, i1) -> (i5, i5, i5) {
  ^bb0(%arg0: i1, %arg1: i4, %arg2: i1, %arg3: i1):
    %1 = arc.state @dummyArc(%arg0, %arg1) lat 1 : (i1, i4) -> i5
    %2 = arc.state @dummyArc(%arg2, %arg1) enable %arg2 lat 2 : (i1, i4) -> i5
    %3 = arc.state @dummyArc(%arg3, %arg1) reset %arg0 lat 3 : (i1, i4) -> i5
    arc.output %1, %2, %3 : i5, i5, i5
  }
  %2 = arc.state @dummyArc(%in0, %in1) clock %clk lat 1 : (i1, i4) -> i5
  %3 = arc.state @dummyArc(%in2, %in1) clock %clk lat 1 : (i1, i4) -> i5
  %4 = arc.state @dummyArc(%in3, %in1) clock %clk lat 1 : (i1, i4) -> i5
  hw.output %0#0, %0#1, %0#2, %1#0, %1#1, %1#2, %2, %3, %4  : i5, i5, i5, i5, i5, i5, i5, i5, i5
}
arc.define @dummyArc(%arg0: i1, %arg1: i4) -> i5 {
  %0 = comb.concat %arg0, %arg1 : i1, i4
  arc.output %0 : i5
}

// CHECK-LABEL: hw.module @vectorize
//  CHECK-SAME: (%clk: i1, %in0: i1, %in1: i4, %in2: i1, %in3: i1) -> (out0: i5, out1: i5, out2: i5, out3: i5, out4: i5, out5: i5, out6: i5, out7: i5, out8: i5)
//       CHECK:   %0:3 = arc.clock_domain (%in0, %in1, %in2, %in3) clock %clk : (i1, i4, i1, i1) -> (i5, i5, i5) {
//       CHECK:   ^bb0(%arg0: i1, %arg1: i4, %arg2: i1, %arg3: i1):
//       CHECK:     %5:3 = arc.vectorize (%arg0, %arg2, %arg3), (%arg1, %arg1, %arg1) : (i1, i1, i1, i4, i4, i4) -> (i5, i5, i5) {
//       CHECK:     ^bb0(%arg4: i1, %arg5: i4):
//       CHECK:       %6 = arc.state @dummyArc(%arg4, %arg5) lat 1 : (i1, i4) -> i5
//       CHECK:       arc.vectorize.return %6 : i5
//       CHECK:     }
//       CHECK:     arc.output %5#0, %5#1, %5#2 : i5, i5, i5
//       CHECK:   }
//       CHECK:   %1:3 = arc.clock_domain (%in0, %in1, %in2, %in3) clock %clk : (i1, i4, i1, i1) -> (i5, i5, i5) {
//       CHECK:   ^bb0(%arg0: i1, %arg1: i4, %arg2: i1, %arg3: i1):
//       CHECK:     %5:3 = arc.vectorize (%arg0, %arg2, %arg3), (%arg1, %arg1, %arg1) : (i1, i1, i1, i4, i4, i4) -> (i5, i5, i5) {
//       CHECK:     ^bb0(%arg4: i1, %arg5: i4):
//       CHECK:       %6 = arc.state @dummyArc(%arg4, %arg5) lat 1 : (i1, i4) -> i5
//       CHECK:       arc.vectorize.return %6 : i5
//       CHECK:     }
//       CHECK:     arc.output %5#0, %5#1, %5#2 : i5, i5, i5
//       CHECK:   }
//       CHECK:   %2 = arc.state @dummyArc(%in0, %in1) clock %clk lat 1 : (i1, i4) -> i5
//       CHECK:   %3 = arc.state @dummyArc(%in2, %in1) clock %clk lat 1 : (i1, i4) -> i5
//       CHECK:   %4 = arc.state @dummyArc(%in3, %in1) clock %clk lat 1 : (i1, i4) -> i5
//       CHECK:   hw.output %0#0, %0#1, %0#2, %1#0, %1#1, %1#2, %2, %3, %4 : i5, i5, i5, i5, i5, i5, i5, i5, i5
