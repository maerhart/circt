// RUN: circt-opt %s -llhd-process-lowering -split-input-file -verify-diagnostics | FileCheck %s

// no inputs and outputs
// CHECK-LABEL: hw.module @empty
// CHECK-SAME: ()
llhd.process @empty() {
  // CHECK-NEXT: hw.output
  llhd.halt
}

// check that input and output signals are transferred correctly
// CHECK-LABEL: hw.module @inputAndOutput
// CHECK-SAME: (inout %{{.*}} : i64, inout %{{.*}} : i1, inout %{{.*}} : i1)
llhd.process @inputAndOutput(inout %arg0 : i64, inout %arg1 : i1, inout %arg2 : i1) {
  // CHECK-NEXT: hw.output
  llhd.halt
}

// check wait suspended process
// CHECK-LABEL: hw.module @simpleWait
// CHECK-SAME: ()
llhd.process @simpleWait() {
  // CHECK-NEXT: hw.output
  cf.br ^bb1
^bb1:
  llhd.wait ^bb1
}

// Check wait with observing probed signals
// CHECK-LABEL: hw.module @prbAndWait
// CHECK-SAME: (inout %{{.*}} : i64)
llhd.process @prbAndWait(inout %arg0 : i64) {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !hw.inout<i64>
  llhd.wait (%arg0 : !hw.inout<i64>), ^bb1
}

// Check wait with observing probed signals
// CHECK-LABEL: hw.module @prbAndWaitMoreObserved
// CHECK-SAME: (inout %{{.*}} : i64, inout %{{.*}} : i64)
llhd.process @prbAndWaitMoreObserved(inout %arg0 : i64, inout %arg1 : i64) {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !hw.inout<i64>
  llhd.wait (%arg0, %arg1 : !hw.inout<i64>, !hw.inout<i64>), ^bb1
}

// CHECK-LABEL: hw.module @muxedSignal
llhd.process @muxedSignal(inout %arg0 : i64, inout %arg1 : i64) {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
  %0 = llhd.prb %sig : !hw.inout<i64>
  llhd.wait (%arg0, %arg1 : !hw.inout<i64>, !hw.inout<i64>), ^bb1
}

// CHECK-LABEL: hw.module @muxedSignal2
llhd.process @muxedSignal2(inout %arg0 : i64, inout %arg1 : i64) {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !hw.inout<i64>
  %0 = llhd.prb %sig : !hw.inout<i64>
  llhd.wait (%sig : !hw.inout<i64>), ^bb1
}

// CHECK-LABEL: hw.module @partialSignal
llhd.process @partialSignal(inout %arg0 : i64) {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = llhd.sig.extract
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: hw.output
  %c = hw.constant 16 : i6
  %sig = llhd.sig.extract %arg0 from %c : (!hw.inout<i64>) -> !hw.inout<i32>
  %0 = llhd.prb %sig : !hw.inout<i32>
  llhd.wait (%arg0 : !hw.inout<i64>), ^bb1
}
