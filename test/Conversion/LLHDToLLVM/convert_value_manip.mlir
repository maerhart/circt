//RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: @convert_const
llvm.func @convert_const() {
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(true) : !llvm.i1
  %0 = llhd.const 1 : i1

  // CHECK-NEXT %{{.*}} = llvm.mlir.constant(0 : i32) : !llvm.i32
  %1 = llhd.const 0 : i32

  // this gets erased
  %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time

  // CHECK-NEXT %{{.*}} = llvm.mlir.constant(123 : i64) : !llvm.i64
  %3 = llhd.const 123 : i64

  llvm.return
}

// CHECK-LABEL: @convert_extract_slice_int
// CHECK-SAME: %[[CI32:.*]]: !llvm.i32
// CHECK-SAME: %[[CI100:.*]]: !llvm.i100
func @convert_extract_slice_int(%cI32 : i32, %cI100 : i100) {
  // CHECK-NEXT: %[[CIND0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK-NEXT: %[[ADJUST0:.*]] = llvm.trunc %[[CIND0]] : !llvm.i64 to !llvm.i32
  // CHECK-NEXT: %[[SHR0:.*]] = llvm.lshr %[[CI32]], %[[ADJUST0]] : !llvm.i32
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR0]] : !llvm.i32 to !llvm.i10
  %0 = llhd.extract_slice %cI32, 0 : i32 -> i10
  // CHECK-NEXT: %[[CIND2:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK-NEXT: %[[ADJUST1:.*]] = llvm.zext %[[CIND2]] : !llvm.i64 to !llvm.i100
  // CHECK-NEXT: %[[SHR1:.*]] = llvm.lshr %[[CI100]], %[[ADJUST1]] : !llvm.i100
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR1]] : !llvm.i100 to !llvm.i10
  %2 = llhd.extract_slice %cI100, 0 : i100 -> i10

  return
}

// CHECK-LABEL:   llvm.func @convert_extract_slice_sig(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                                         %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                                         %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm<"i8**">
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm<"i64*">
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_12]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm<"i64*">
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_13]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm<"i64*">
// CHECK:           %[[VAL_18:.*]] = llvm.add %[[VAL_11]], %[[VAL_5]] : !llvm.i64
// CHECK:           %[[VAL_19:.*]] = llvm.ptrtoint %[[VAL_9]] : !llvm<"i8*"> to !llvm.i64
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK:           %[[VAL_21:.*]] = llvm.udiv %[[VAL_18]], %[[VAL_20]] : !llvm.i64
// CHECK:           %[[VAL_22:.*]] = llvm.add %[[VAL_19]], %[[VAL_21]] : !llvm.i64
// CHECK:           %[[VAL_23:.*]] = llvm.inttoptr %[[VAL_22]] : !llvm.i64 to !llvm<"i8*">
// CHECK:           %[[VAL_24:.*]] = llvm.urem %[[VAL_18]], %[[VAL_20]] : !llvm.i64
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.undef : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_25]][0 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_26]][1 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_27]][2 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_28]][3 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_31:.*]] = llvm.alloca %[[VAL_30]] x !llvm<"{ i8*, i64, i64, i64 }"> {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           llvm.store %[[VAL_29]], %[[VAL_31]] : !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_extract_slice_sig (%sI32 : !llhd.sig<i32>) -> () {
  %0 = llhd.extract_slice %sI32, 0 : !llhd.sig<i32> -> !llhd.sig<i10>
}
