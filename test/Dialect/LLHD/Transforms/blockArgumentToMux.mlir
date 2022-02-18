// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: circt-opt %s -llhd-block-argument-to-mux | FileCheck %s

// CHECK-LABEL:   llhd.proc @check_batm(
// CHECK-SAME:                          %[[VAL_0:.*]] : !llhd.sig<i1>) -> (
// CHECK-SAME:                          %[[VAL_1:.*]] : !llhd.sig<i32>) {
// CHECK:           %[[VAL_2:.*]] = llhd.constant_time <0ns, 1d, 0e>
// CHECK:           %[[VAL_3:.*]] = llhd.prb %[[VAL_0]] : !llhd.sig<i1>
// CHECK:           %[[VAL_4:.*]] = hw.constant 4 : i32
// CHECK:           %[[VAL_5:.*]] = hw.constant 3 : i32
// CHECK:           cf.cond_br %[[VAL_3]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           llhd.drv %[[VAL_1]], %[[VAL_6]] after %[[VAL_2]] : !llhd.sig<i32>
// CHECK:           llhd.halt
// CHECK:         }
llhd.proc @check_batm(%cond : !llhd.sig<i1>) -> (%out : !llhd.sig<i32>) {
  %del = llhd.constant_time <0ns, 1d, 0e>
  %cond_prb = llhd.prb %cond : !llhd.sig<i1>
  %c1 = hw.constant 4 : i32
  %c2 = hw.constant 3 : i32
  cf.cond_br %cond_prb, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%c1 : i32)
^bb2:
  cf.br ^bb3(%c2 : i32)
^bb3(%res : i32):
  llhd.drv %out, %res after %del : !llhd.sig<i32>
  llhd.halt
}
