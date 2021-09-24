// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @convert_bitcast
// CHECK-NEXT: %[[ONE1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE1]] x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr<i32>
// CHECK-NEXT: llvm.store %arg0, %[[A1]] : !llvm.ptr<i32>
// CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[A1]] : !llvm.ptr<i32> to !llvm.ptr<array<4 x i8>>
// CHECK-NEXT: llvm.load %[[B1]] : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE2]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT: llvm.store %arg1, %[[A2]] : !llvm.ptr<array<2 x i32>>
// CHECK-NEXT: %[[B2:.*]] = llvm.bitcast %[[A2]] : !llvm.ptr<array<2 x i32>> to !llvm.ptr<i64>
// CHECK-NEXT: llvm.load %[[B2]] : !llvm.ptr<i64>
// CHECK-NEXT: %[[ONE3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[A3:.*]] = llvm.alloca %[[ONE3]] x !llvm.struct<(i32, i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: llvm.store %arg2, %[[A3]] : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: %[[B3:.*]] = llvm.bitcast %[[A3]] : !llvm.ptr<struct<(i32, i32)>> to !llvm.ptr<i64>
// CHECK-NEXT: llvm.load %[[B3]] : !llvm.ptr<i64>
// CHECK-NEXT: llvm.return
func @convert_bitcast(%arg0 : i32,
                      %arg1: !hw.array<2xi32>,
                      %arg2: !hw.struct<foo: i32, bar: i32>) {

    %0 = hw.bitcast %arg0 : (i32) -> !hw.array<4xi8>
    %1 = hw.bitcast %arg1 : (!hw.array<2xi32>) -> i64
    %2 = hw.bitcast %arg2 : (!hw.struct<foo: i32, bar: i32>) -> i64

    return
}
