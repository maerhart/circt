// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @drive_signal(!llvm<"i8*">, !llvm<"{ i8*, i64, i64, i64 }*">, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64)

// CHECK-LABEL:   llvm.func @convert_sig(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                           %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.undef : !llvm<"[4 x i1]">
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_4]][0 : i32] : !llvm<"[4 x i1]">
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_5]][1 : i32] : !llvm<"[4 x i1]">
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_6]][2 : i32] : !llvm<"[4 x i1]">
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_7]][3 : i32] : !llvm<"[4 x i1]">
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_9]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_11]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           llvm.return
// CHECK:         }

llhd.entity @convert_sig () -> () {
  %init = llhd.const 0 : i1
  %initArr = llhd.array_uniform %init : !llhd.array<4xi1>
  %s0 = llhd.sig "sig0" %init : i1
  %s1 = llhd.sig "sig1" %initArr : !llhd.array<4xi1>
}

// CHECK-LABEL:   llvm.func @convert_prb(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                           %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_5]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_7]], %[[VAL_7]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm<"i8**">
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm<"i64*">
// CHECK:           %[[VAL_13:.*]] = llvm.bitcast %[[VAL_10]] : !llvm<"i8*"> to !llvm<"i16*">
// CHECK:           %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm<"i16*">
// CHECK:           %[[VAL_15:.*]] = llvm.trunc %[[VAL_12]] : !llvm.i64 to !llvm.i16
// CHECK:           %[[VAL_16:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_15]] : !llvm.i16
// CHECK:           %[[VAL_17:.*]] = llvm.trunc %[[VAL_16]] : !llvm.i16 to !llvm.i1
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_18]], %[[VAL_18]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK:           %[[VAL_21:.*]] = llvm.load %[[VAL_20]] : !llvm<"i8**">
// CHECK:           %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_18]], %[[VAL_19]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK:           %[[VAL_23:.*]] = llvm.load %[[VAL_22]] : !llvm<"i64*">
// CHECK:           %[[VAL_24:.*]] = llvm.bitcast %[[VAL_21]] : !llvm<"i8*"> to !llvm<"[3 x i5]*">
// CHECK:           %[[VAL_25:.*]] = llvm.load %[[VAL_24]] : !llvm<"[3 x i5]*">
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_prb (%sI1 : !llhd.sig<i1>, %sArr : !llhd.sig<!llhd.array<3xi5>>) -> () {
  %p0 = llhd.prb %sI1 : !llhd.sig<i1>
  %p1 = llhd.prb %sArr : !llhd.sig<!llhd.array<3xi5>>
}

// CHECK-LABEL:   llvm.func @convert_drv(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                           %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_5]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(0 : i5) : !llvm.i5
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.undef : !llvm<"[3 x i5]">
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_9]][0 : i32] : !llvm<"[3 x i5]">
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_10]][1 : i32] : !llvm<"[3 x i5]">
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_11]][2 : i32] : !llvm<"[3 x i5]">
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(dense<[1000, 0, 0]> : vector<3xi64>) : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_16:.*]] = llvm.alloca %[[VAL_15]] x !llvm.i1 {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"i1*">
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_16]] : !llvm<"i1*">
// CHECK:           %[[VAL_17:.*]] = llvm.bitcast %[[VAL_16]] : !llvm<"i1*"> to !llvm<"i8*">
// CHECK:           %[[VAL_18:.*]] = llvm.extractvalue %[[VAL_13]][0 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_19:.*]] = llvm.extractvalue %[[VAL_13]][1 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_20:.*]] = llvm.extractvalue %[[VAL_13]][2 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_21:.*]] = llvm.call @drive_signal(%[[VAL_0]], %[[VAL_4]], %[[VAL_17]], %[[VAL_14]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]]) : (!llvm<"i8*">, !llvm<"{ i8*, i64, i64, i64 }*">, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.void
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.null : !llvm<"[3 x i5]*">
// CHECK:           %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_22]]] : (!llvm<"[3 x i5]*">, !llvm.i32) -> !llvm<"[3 x i5]*">
// CHECK:           %[[VAL_25:.*]] = llvm.ptrtoint %[[VAL_24]] : !llvm<"[3 x i5]*"> to !llvm.i64
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_27:.*]] = llvm.alloca %[[VAL_26]] x !llvm<"[3 x i5]"> {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"[3 x i5]*">
// CHECK:           llvm.store %[[VAL_12]], %[[VAL_27]] : !llvm<"[3 x i5]*">
// CHECK:           %[[VAL_28:.*]] = llvm.bitcast %[[VAL_27]] : !llvm<"[3 x i5]*"> to !llvm<"i8*">
// CHECK:           %[[VAL_29:.*]] = llvm.extractvalue %[[VAL_13]][0 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_30:.*]] = llvm.extractvalue %[[VAL_13]][1 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_31:.*]] = llvm.extractvalue %[[VAL_13]][2 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_32:.*]] = llvm.call @drive_signal(%[[VAL_0]], %[[VAL_6]], %[[VAL_28]], %[[VAL_25]], %[[VAL_29]], %[[VAL_30]], %[[VAL_31]]) : (!llvm<"i8*">, !llvm<"{ i8*, i64, i64, i64 }*">, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.void
// CHECK:           llvm.return
// CHECK:         }

llhd.entity @convert_drv (%sI1 : !llhd.sig<i1>, %sArr : !llhd.sig<!llhd.array<3xi5>>) -> () {
  %cI1 = llhd.const 0 : i1
  %cI5 = llhd.const 0 : i5
  %cArr = llhd.array_uniform %cI5 : !llhd.array<3xi5>
  %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  llhd.drv %sI1, %cI1 after %t : !llhd.sig<i1>
  llhd.drv %sArr, %cArr after %t : !llhd.sig<!llhd.array<3xi5>>
}

// CHECK-LABEL:   llvm.func @convert_drv_enable(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                                  %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(dense<[1000, 0, 0]> : vector<3xi64>) : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : i16) : !llvm.i1
// CHECK:           %[[VAL_9:.*]] = llvm.icmp "eq" %[[VAL_5]], %[[VAL_8]] : !llvm.i1
// CHECK:           llvm.cond_br %[[VAL_9]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_11:.*]] = llvm.alloca %[[VAL_10]] x !llvm.i1 {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"i1*">
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_11]] : !llvm<"i1*">
// CHECK:           %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : !llvm<"i1*"> to !llvm<"i8*">
// CHECK:           %[[VAL_13:.*]] = llvm.extractvalue %[[VAL_6]][0 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_6]][1 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_15:.*]] = llvm.extractvalue %[[VAL_6]][2 : i32] : !llvm<"[3 x i64]">
// CHECK:           %[[VAL_16:.*]] = llvm.call @drive_signal(%[[VAL_0]], %[[VAL_4]], %[[VAL_12]], %[[VAL_7]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]]) : (!llvm<"i8*">, !llvm<"{ i8*, i64, i64, i64 }*">, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.void
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_drv_enable (%sI1 : !llhd.sig<i1>) -> () {
    %cI1 = llhd.const 0 : i1
    %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %sI1, %cI1 after %t if %cI1 : !llhd.sig<i1>
}
