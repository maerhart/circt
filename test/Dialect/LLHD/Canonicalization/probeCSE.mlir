// RUN: circt-opt %s -cse | FileCheck %s

// CHECK-LABEL: @checkPrbCseAndDce
llhd.entity @checkPrbCseAndDce(%sig : !llhd.sig<i32>) -> () {
  // CHECK-NEXT: llhd.constant_time
  %time = llhd.constant_time <0ns, 1d, 0e>

  // CHECK-NEXT: llhd.prb
  %1 = llhd.prb %sig : !llhd.sig<i32>
  %2 = llhd.prb %sig : !llhd.sig<i32>
  %3 = llhd.prb %sig : !llhd.sig<i32>

  // CHECK-NEXT: llhd.drv
  // CHECK-NEXT: llhd.drv
  llhd.drv %sig, %1 after %time : !llhd.sig<i32>
  llhd.drv %sig, %2 after %time : !llhd.sig<i32>
}

// CHECK-LABEL: @checkPrbDce
llhd.proc @checkPrbDce(%sig : !llhd.sig<i32>) -> () {
  // CHECK-NEXT: llhd.constant_time
  %time = llhd.constant_time <0ns, 1d, 0e>

  // CHECK-NEXT: llhd.prb
  %1 = llhd.prb %sig : !llhd.sig<i32>
  // CHECK-NEXT: llhd.wait
  llhd.wait for %time, ^bb1
// CHECK-NEXT: ^bb1:
^bb1:
  // CHECK-NEXT: llhd.prb
  %2 = llhd.prb %sig : !llhd.sig<i32>
  %3 = llhd.prb %sig : !llhd.sig<i32>

  // CHECK-NEXT: llhd.drv
  // CHECK-NEXT: llhd.drv
  llhd.drv %sig, %1 after %time : !llhd.sig<i32>
  llhd.drv %sig, %2 after %time : !llhd.sig<i32>
  llhd.halt
}
