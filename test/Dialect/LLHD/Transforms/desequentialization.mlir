// RUN: circt-opt %s -llhd-desequentialize | FileCheck %s

// CHECK-LABEL: @check_deseq_rise_enable
llhd.proc @check_deseq_rise_enable(%arg0 : !llhd.sig<i1>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i32>) {
  cf.br ^bb2
 ^bb2:
  %del = llhd.constant_time <0ns, 1d, 0e>
  %prb1 = llhd.prb %arg0 : !llhd.sig<i1>
  %c1 = hw.constant 4 : i32
  %true = hw.constant true
  llhd.wait ^bb1
^bb1:
  %prb2 = llhd.prb %arg0 : !llhd.sig<i1>
  %1 = comb.xor %prb1, %true : i1
  %prb_en = llhd.prb %arg1 : !llhd.sig<i1>
  %2 = comb.and %1, %prb2, %prb_en : i1
  llhd.drv %arg2, %c1 after %del if %2 : !llhd.sig<i32>
  cf.br ^bb2
}

// CHECK-LABEL: @check_deseq_fall
llhd.proc @check_deseq_fall(%arg0 : !llhd.sig<i1>) -> (%arg1 : !llhd.sig<i32>) {
  cf.br ^bb2
 ^bb2:
  %del = llhd.constant_time <0ns, 1d, 0e>
  %prb1 = llhd.prb %arg0 : !llhd.sig<i1>
  %c1 = hw.constant 4 : i32
  %true = hw.constant true
  llhd.wait ^bb1
^bb1:
  %prb2 = llhd.prb %arg0 : !llhd.sig<i1>
  %1 = comb.xor %prb2, %true : i1
  %2 = comb.and %1, %prb1 : i1
  llhd.drv %arg1, %c1 after %del if %2 : !llhd.sig<i32>
  cf.br ^bb2
}

// CHECK-LABEL: @check_deseq_low
llhd.proc @check_deseq_low(%arg0 : !llhd.sig<i1>) -> (%arg1 : !llhd.sig<i32>) {
  cf.br ^bb2
 ^bb2:
  %del = llhd.constant_time <0ns, 1d, 0e>
  %c1 = hw.constant 4 : i32
  %true = hw.constant true
  llhd.wait ^bb1
^bb1:
  %prb = llhd.prb %arg0 : !llhd.sig<i1>
  %1 = comb.xor %prb, %true : i1
  llhd.drv %arg1, %c1 after %del if %1 : !llhd.sig<i32>
  cf.br ^bb2
}

// CHECK-LABEL: @check_deseq_high
llhd.proc @check_deseq_high(%arg0 : !llhd.sig<i1>) -> (%arg1 : !llhd.sig<i32>) {
  cf.br ^bb2
 ^bb2:
  %del = llhd.constant_time <0ns, 1d, 0e>
  %c1 = hw.constant 4 : i32
  llhd.wait ^bb1
^bb1:
  %prb = llhd.prb %arg0 : !llhd.sig<i1>
  llhd.drv %arg1, %c1 after %del if %prb : !llhd.sig<i32>
  cf.br ^bb2
}

// CHECK-LABEL: @check_deseq_high2
llhd.proc @check_deseq_high2(%arg0 : !llhd.sig<i1>) -> (%arg1 : !llhd.sig<i32>) {
  cf.br ^bb2
 ^bb2:
  %del = llhd.constant_time <0ns, 1d, 0e>
  %c1 = hw.constant 4 : i32
  llhd.wait ^bb1
^bb1:
  %prb1 = llhd.prb %arg0 : !llhd.sig<i1>
  %prb2 = llhd.prb %arg0 : !llhd.sig<i1>
  %true = hw.constant true
  %1 = comb.xor %prb1, %true : i1
  %2 = comb.and %1, %prb2 : i1
  llhd.drv %arg1, %c1 after %del if %2 : !llhd.sig<i32>
  cf.br ^bb2
}

// CHECK-LABEL: @check_deseq_error
llhd.proc @check_deseq_error(%arg0 : !llhd.sig<i1>) -> (%arg1 : !llhd.sig<i32>) {
  cf.br ^bb2
 ^bb2:
  %del = llhd.constant_time <0ns, 1d, 0e>
  %c1 = hw.constant 4 : i32
  %prb = llhd.prb %arg0 : !llhd.sig<i1>
  llhd.wait ^bb1
^bb1:
  // llhd.drv %arg1, %c1 after %del if %prb : !llhd.sig<i32>
  cf.br ^bb2
}