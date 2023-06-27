// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
//RUN: circt-opt %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL:   llvm.func @convert_empty(
// CHECK-SAME:                             %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                             %[[VAL_1:.*]]: !llvm.ptr<struct<()>>,
// CHECK-SAME:                             %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr, i64, i64, i64)>>) {
// CHECK:           llvm.return
// CHECK:         }
hw.module @convert_empty() {}

// CHECK-LABEL:   llvm.func @convert_one_input(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !llvm.ptr<struct<()>>,
// CHECK-SAME:                                 %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<(ptr, i64, i64, i64)>>, i32) -> !llvm.ptr<struct<(ptr, i64, i64, i64)>>
// CHECK:           llvm.return
// CHECK:         }
hw.module @convert_one_input(%in0: !hw.inout<i1>, %in1: i32) -> (out0: i32) {
  hw.output %in1 : i32
}
