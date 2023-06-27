//RUN: circt-opt %s | circt-opt | FileCheck %s

// Testing Objectives:
// * inst can only be used in entities
// * inst must always refer to a valid proc or entity (match symbol name, input and output operands)
// * syntax: no inputs and outputs, one input zero outputs, zero inputs one output, multiple inputs and outputs
// * check that number of inputs and number of outputs are verified separately


// CHECK-LABEL: @empty_proc
llhd.proc @empty_proc() -> () {
  llhd.halt
}

// CHECK-LABEL: @one_input_proc
llhd.proc @one_input_proc(%arg : !hw.inout<i32>) -> () {
  llhd.halt
}

// CHECK-LABEL: @one_output_proc
llhd.proc @one_output_proc() -> (%arg : !hw.inout<i32>) {
  llhd.halt
}

// CHECK-LABEL: @proc
llhd.proc @proc(%arg0 : !hw.inout<i32>, %arg1 : !hw.inout<i16>) -> (%out0 : !hw.inout<i8>, %out1 : !hw.inout<i4>) {
  llhd.halt
}

// CHECK: hw.module @caller (%[[ARG0:.*]]: !hw.inout<i32>, %[[ARG1:.*]]: !hw.inout<i16>, %[[OUT0:.*]]: !hw.inout<i8>, %[[OUT1:.*]]: !hw.inout<i4>) {
hw.module @caller(%arg0 : !hw.inout<i32>, %arg1 : !hw.inout<i16>, %out0 : !hw.inout<i8>, %out1 : !hw.inout<i4>) {
  // CHECK-NEXT: llhd.inst "empty_proc" @empty_proc() -> () : () -> ()
  "llhd.inst"() {callee=@empty_proc, operand_segment_sizes=array<i32: 0,0>, name="empty_proc"} : () -> ()
  // CHECK-NEXT: llhd.inst "one_in_proc" @one_input_proc(%[[ARG0]]) -> () : (!hw.inout<i32>) -> ()
  "llhd.inst"(%arg0) {callee=@one_input_proc, operand_segment_sizes=array<i32: 1,0>, name="one_in_proc"} : (!hw.inout<i32>) -> ()
  // CHECK-NEXT: llhd.inst "one_out_proc" @one_output_proc() -> (%[[ARG0]]) : () -> !hw.inout<i32>
  "llhd.inst"(%arg0) {callee=@one_output_proc, operand_segment_sizes=array<i32: 0,1>, name="one_out_proc"} : (!hw.inout<i32>) -> ()
  // CHECK-NEXT: llhd.inst "proc" @proc(%[[ARG0]], %[[ARG1]]) -> (%[[OUT0]], %[[OUT1]]) : (!hw.inout<i32>, !hw.inout<i16>) -> (!hw.inout<i8>, !hw.inout<i4>)
  "llhd.inst"(%arg0, %arg1, %out0, %out1) {callee=@proc, operand_segment_sizes=array<i32: 2,2>, name="proc"} : (!hw.inout<i32>, !hw.inout<i16>, !hw.inout<i8>, !hw.inout<i4>) -> ()
  // CHECK-NEXT: }
}
