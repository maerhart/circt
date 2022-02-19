// RUN: circt-opt %s -llhd-block-argument-to-mux | FileCheck %s

// CHECK-LABEL: @check_batm
// CHECK-NEXT:   %[[VAL_2:.*]] = llhd.constant_time <0ns, 1d, 0e>
// CHECK-NEXT:   %[[VAL_3:.*]] = llhd.prb %arg0 : !llhd.sig<i1>
// CHECK-NEXT:   %[[VAL_4:.*]] = hw.constant 4 : i32
// CHECK-NEXT:   %[[VAL_5:.*]] = hw.constant 3 : i32
// CHECK-NEXT:   cf.cond_br %[[VAL_3]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb3
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   cf.br ^bb3
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   %true = hw.constant true
// CHECK-NEXT:   %[[VAL_6:.*]] = comb.and %true, %[[VAL_3]] : i1
// CHECK-NEXT:   %[[VAL_7:.*]] = comb.mux %[[VAL_6]], %c4_i32, %c3_i32 : i32
// CHECK-NEXT:   %[[VAL_8:.*]] = comb.mux %[[VAL_6]], %c3_i32, %c4_i32 : i32
// CHECK-NEXT:   %[[VAL_9:.*]] = comb.mux %[[VAL_6]], %c4_i32, %c3_i32 : i32
// CHECK-NEXT:   llhd.drv %arg1, %[[VAL_7]] after %[[VAL_2]] : !llhd.sig<i32>
// CHECK-NEXT:   llhd.drv %arg1, %[[VAL_8]] after %[[VAL_2]] : !llhd.sig<i32>
// CHECK-NEXT:   llhd.drv %arg1, %[[VAL_9]] after %[[VAL_2]] : !llhd.sig<i32>
// CHECK-NEXT:   llhd.halt
llhd.proc @check_batm(%cond : !llhd.sig<i1>) -> (%out : !llhd.sig<i32>) {
  %del = llhd.constant_time <0ns, 1d, 0e>
  %cond_prb = llhd.prb %cond : !llhd.sig<i1>
  %c1 = hw.constant 4 : i32
  %c2 = hw.constant 3 : i32
  cf.cond_br %cond_prb, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%c1, %c2, %c1 : i32, i32, i32)
^bb2:
  cf.br ^bb3(%c2, %c1, %c2 : i32, i32, i32)
^bb3(%res: i32, %res2: i32, %res3: i32):
  llhd.drv %out, %res after %del : !llhd.sig<i32>
  llhd.drv %out, %res2 after %del : !llhd.sig<i32>
  llhd.drv %out, %res3 after %del : !llhd.sig<i32>
  llhd.halt
}

// CHECK-LABEL: @check_batm2
// CHECK-NEXT:   %[[VAL_2:.*]] = llhd.constant_time <0ns, 1d, 0e>
// CHECK-NEXT:   %[[VAL_3:.*]] = llhd.prb %arg0 : !llhd.sig<i1>
// CHECK-NEXT:   %c4_i32 = hw.constant 4 : i32
// CHECK-NEXT:   cf.cond_br %[[VAL_3]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb3(%c4_i32 : i32)
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   %c3_i32 = hw.constant 3 : i32
// CHECK-NEXT:   cf.br ^bb3(%c3_i32 : i32)
// CHECK-NEXT: ^bb3(%[[ARG_1:.*]]: i32):
// CHECK-NEXT:   llhd.drv %arg1, %[[ARG_1]] after %[[VAL_2]] : !llhd.sig<i32>
// CHECK-NEXT:   llhd.halt
llhd.proc @check_batm2(%cond : !llhd.sig<i1>) -> (%out : !llhd.sig<i32>) {
  %del = llhd.constant_time <0ns, 1d, 0e>
  %cond_prb = llhd.prb %cond : !llhd.sig<i1>
  %c1 = hw.constant 4 : i32
  cf.cond_br %cond_prb, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%c1 : i32)
^bb2:
  %c2 = hw.constant 3 : i32
  cf.br ^bb3(%c2 : i32)
^bb3(%res : i32):
  llhd.drv %out, %res after %del : !llhd.sig<i32>
  llhd.halt
}

// CHECK-LABEL: @check_batm3
// CHECK-NEXT:   %[[VAL_2:.*]] = llhd.constant_time <0ns, 1d, 0e>
// CHECK-NEXT:   %[[VAL_3:.*]] = llhd.prb %arg0 : !llhd.sig<i1>
// CHECK-NEXT:   %c4_i32 = hw.constant 4 : i32
// CHECK-NEXT:   %c3_i32 = hw.constant 3 : i32
// CHECK-NEXT:   %[[VAL_6:.*]] = comb.mux %[[VAL_3]], %c4_i32, %c3_i32 : i32
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   llhd.drv %arg1, %[[VAL_6]] after %[[VAL_2]] : !llhd.sig<i32>
// CHECK-NEXT:   llhd.halt
llhd.proc @check_batm3(%cond : !llhd.sig<i1>) -> (%out : !llhd.sig<i32>) {
  %del = llhd.constant_time <0ns, 1d, 0e>
  %cond_prb = llhd.prb %cond : !llhd.sig<i1>
  %c4_i32 = hw.constant 4 : i32
  %c3_i32 = hw.constant 3 : i32
  cf.cond_br %cond_prb, ^bb1(%c4_i32 : i32), ^bb1(%c3_i32 : i32)
^bb1(%res : i32):
  llhd.drv %out, %res after %del : !llhd.sig<i32>
  llhd.halt
}

// CHECK-LABEL: @check_batm4
// CHECK-NEXT:   %c3_i32 = hw.constant 3 : i32
// CHECK-NEXT:   cf.br ^bb1(%c3_i32 : i32)
// CHECK-NEXT: ^bb1(%[[ARG_1:.*]]: i32):
// CHECK-NEXT:   cf.br ^bb1(%[[ARG_1]] : i32)
llhd.proc @check_batm4() -> () {
  %c3_i32 = hw.constant 3 : i32
  cf.br ^bb1(%c3_i32 : i32)
^bb1(%res : i32):
  cf.br ^bb1(%res : i32)
}
