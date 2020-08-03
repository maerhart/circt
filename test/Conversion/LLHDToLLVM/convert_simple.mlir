// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s


// CHECK-LABEL:   llvm.func @drive_signal(!llvm<"i8*">, !llvm<"{ i8*, i64, i64, i64 }*">, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64)

// CHECK-LABEL:   llvm.func @Foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                   %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                   %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_4]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm<"i8**">
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm<"i64*">
// CHECK:           %[[VAL_12:.*]] = llvm.bitcast %[[VAL_9]] : !llvm<"i8*"> to !llvm<"i16*">
// CHECK:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm<"i16*">
// CHECK:           %[[VAL_14:.*]] = llvm.trunc %[[VAL_11]] : !llvm.i64 to !llvm.i16
// CHECK:           %[[VAL_15:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_14]] : !llvm.i16
// CHECK:           %[[VAL_16:.*]] = llvm.trunc %[[VAL_15]] : !llvm.i16 to !llvm.i1
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(true) : !llvm.i1
// CHECK:           %[[VAL_18:.*]] = llvm.xor %[[VAL_16]], %[[VAL_17]] : !llvm.i1
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(dense<[1000, 0, 0]> : vector<3xi64>) : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_22:.*]] = llvm.alloca %[[VAL_21]] x !llvm.i1 {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"i1*">
// CHECK:           llvm.store %[[VAL_18]], %[[VAL_22]] : !llvm<"i1*">
// CHECK:           %[[VAL_23:.*]] = llvm.bitcast %[[VAL_22]] : !llvm<"i1*"> to !llvm<"i8*">
// CHECK:           %[[VAL_24:.*]] = llvm.extractvalue %[[VAL_19]][0 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_25:.*]] = llvm.extractvalue %[[VAL_19]][1 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_26:.*]] = llvm.extractvalue %[[VAL_19]][2 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_27:.*]] = llvm.call @drive_signal(%[[VAL_0]], %[[VAL_5]], %[[VAL_23]], %[[VAL_20]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]]) : (!llvm<"i8*">, !llvm<"{ i8*, i64, i64, i64 }*">, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.void
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @Foo () -> () {
  %0 = llhd.const 0 : i1
  %toggle = llhd.sig "toggle" %0 : i1
  %1 = llhd.prb %toggle : !llhd.sig<i1>
  %2 = llhd.not %1 : i1
  %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  llhd.drv %toggle, %2 after %dt : !llhd.sig<i1>
}
