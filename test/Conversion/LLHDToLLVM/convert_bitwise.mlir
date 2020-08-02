//RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: convert_bitwise_i1
// CHECK-SAME: %[[LHS:.*]]: !llvm.i1,
// CHECK-SAME: %[[RHS:.*]]: !llvm.i1
func @convert_bitwise_i1(%lhs : i1, %rhs : i1) {
  // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(true) : !llvm.i1
  // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[MASK]] : !llvm.i1
  %0 = llhd.not %lhs : i1
  // CHECK-NEXT: %{{.*}} = llvm.and %[[LHS]], %[[RHS]] : !llvm.i1
  %1 = llhd.and %lhs, %rhs : i1
  // CHECK-NEXT: %{{.*}} = llvm.or %[[LHS]], %[[RHS]] : !llvm.i1
  %2 = llhd.or %lhs, %rhs : i1
  // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[RHS]] : !llvm.i1
  %3 = llhd.xor %lhs, %rhs : i1

  return
}

// CHECK-LABEL: convert_bitwise_i32
// CHECK-SAME: %[[LHS:.*]]: !llvm.i32,
// CHECK-SAME: %[[RHS:.*]]: !llvm.i32
func @convert_bitwise_i32(%lhs : i32, %rhs : i32) {
  // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
  // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[MASK]] : !llvm.i32
  llhd.not %lhs : i32
  // CHECK-NEXT: %{{.*}} = llvm.and %[[LHS]], %[[RHS]] : !llvm.i32
  llhd.and %lhs, %rhs : i32
  // CHECK-NEXT: %{{.*}} = llvm.or %[[LHS]], %[[RHS]] : !llvm.i32
  llhd.or %lhs, %rhs : i32
  // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[RHS]] : !llvm.i32
  llhd.xor %lhs, %rhs : i32

  return
}

// CHECK-LABEL: convert_shl_i5_i2_i2
// CHECK-SAME: %[[BASE:.*]]: !llvm.i5,
// CHECK-SAME: %[[HIDDEN:.*]]: !llvm.i2,
// CHECK-SAME: %[[AMOUNT:.*]]: !llvm.i2
func @convert_shl_i5_i2_i2(%base : i5, %hidden : i2, %amount : i2) {
  // CHECK-NEXT: %[[ZEXTB:.*]] = llvm.zext %[[BASE]] : !llvm.i5 to !llvm.i7
  // CHECK-NEXT: %[[ZEXTH:.*]] = llvm.zext %[[HIDDEN]] : !llvm.i2 to !llvm.i7
  // CHECK-NEXT: %[[ZEXTA:.*]] = llvm.zext %[[AMOUNT]] : !llvm.i2 to !llvm.i7
  // CHECK-NEXT: %[[HDNW:.*]] = llvm.mlir.constant(2 : i7) : !llvm.i7
  // CHECK-NEXT: %[[SHB:.*]] = llvm.shl %[[ZEXTB]], %[[HDNW]] : !llvm.i7
  // CHECK-NEXT: %[[COMB:.*]] = llvm.or %[[SHB]], %[[ZEXTH]] : !llvm.i7
  // CHECK-NEXT: %[[SA:.*]] = llvm.sub %[[HDNW]], %[[ZEXTA]] : !llvm.i
  // CHECK-NEXT: %[[SH:.*]] = llvm.lshr %[[COMB]], %[[SA]] : !llvm.i7
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SH]] : !llvm.i7 to !llvm.i5
  %0 = llhd.shl %base, %hidden, %amount : (i5, i2, i2) -> i5

  return
}

// CHECK-LABEL: convert_shr_i5_i2_i2
// CHECK-SAME: %[[BASE:.*]]: !llvm.i5,
// CHECK-SAME: %[[HIDDEN:.*]]: !llvm.i2,
// CHECK-SAME: %[[AMOUNT:.*]]: !llvm.i2
func @convert_shr_i5_i2_i2(%base : i5, %hidden : i2, %amount : i2) {
  // CHECK-NEXT: %[[ZEXTB:.*]] = llvm.zext %[[BASE]] : !llvm.i5 to !llvm.i7
  // CHECK-NEXT: %[[ZEXTH:.*]] = llvm.zext %[[HIDDEN]] : !llvm.i2 to !llvm.i7
  // CHECK-NEXT: %[[ZEXTA:.*]] = llvm.zext %[[AMOUNT]] : !llvm.i2 to !llvm.i7
  // CHECK-NEXT: %[[BASEW:.*]] = llvm.mlir.constant(5 : i7) : !llvm.i7
  // CHECK-NEXT: %[[SHH:.*]] = llvm.shl %[[ZEXTH]], %[[BASEW]] : !llvm.i7
  // CHECK-NEXT: %[[COMB:.*]] = llvm.or %[[SHH]], %[[ZEXTB]] : !llvm.i7
  // CHECK-NEXT: %[[SH:.*]] = llvm.lshr %[[COMB]], %[[ZEXTA]] : !llvm.i7
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SH]] : !llvm.i7 to !llvm.i5
  %0 = llhd.shr %base, %hidden, %amount : (i5, i2, i2) -> i5

  return
}

// CHECK-LABEL:   llvm.func @convert_shr_sig(
// CHECK-SAME:                               %[[VAL_0:.*]]: !llvm<"i8*">,
// CHECK-SAME:                               %[[VAL_1:.*]]: !llvm<"{}*">,
// CHECK-SAME:                               %[[VAL_2:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(8 : i32) : !llvm.i32
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
// CHECK:           %[[VAL_18:.*]] = llvm.zext %[[VAL_5]] : !llvm.i32 to !llvm.i64
// CHECK:           %[[VAL_19:.*]] = llvm.add %[[VAL_11]], %[[VAL_18]] : !llvm.i64
// CHECK:           %[[VAL_20:.*]] = llvm.ptrtoint %[[VAL_9]] : !llvm<"i8*"> to !llvm.i64
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK:           %[[VAL_22:.*]] = llvm.udiv %[[VAL_19]], %[[VAL_21]] : !llvm.i64
// CHECK:           %[[VAL_23:.*]] = llvm.add %[[VAL_20]], %[[VAL_22]] : !llvm.i64
// CHECK:           %[[VAL_24:.*]] = llvm.inttoptr %[[VAL_23]] : !llvm.i64 to !llvm<"i8*">
// CHECK:           %[[VAL_25:.*]] = llvm.urem %[[VAL_19]], %[[VAL_21]] : !llvm.i64
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.undef : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_26]][0 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_25]], %[[VAL_27]][1 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_28]][2 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_29]][3 : i32] : !llvm<"{ i8*, i64, i64, i64 }">
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_32:.*]] = llvm.alloca %[[VAL_31]] x !llvm<"{ i8*, i64, i64, i64 }"> {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           llvm.store %[[VAL_30]], %[[VAL_32]] : !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_shr_sig (%sI32 : !llhd.sig<i32>) -> () {
  %0 = llhd.const 8 : i32
  %1 = llhd.shr %sI32, %sI32, %0 : (!llhd.sig<i32>, !llhd.sig<i32>, i32) -> !llhd.sig<i32>
}
