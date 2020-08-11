// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @test_array(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.i1,
// CHECK-SAME:                          %[[VAL_1:.*]]: !llvm.i32) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.undef : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][0 : i32] : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][1 : i32] : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_4]][2 : i32] : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.undef : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_6]][0 : i32] : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_7]][1 : i32] : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_8]][2 : i32] : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][3 : i32] : !llvm<"[4 x i32]">
// CHECK:           llvm.return
// CHECK:         }
func @test_array(%ci1 : i1, %ci32 : i32) {
  %0 = llhd.array %ci1, %ci1, %ci1 : !llhd.array<3xi1>
  %1 = llhd.array %ci32, %ci32, %ci32, %ci32 : !llhd.array<4xi32>

  return
}

// CHECK-LABEL:   llvm.func @test_array_uniform(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !llvm.i32) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.undef : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][0 : i32] : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][1 : i32] : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_4]][2 : i32] : !llvm<"[3 x i1]">
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.undef : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_6]][0 : i32] : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_7]][1 : i32] : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_8]][2 : i32] : !llvm<"[4 x i32]">
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][3 : i32] : !llvm<"[4 x i32]">
// CHECK:           llvm.return
// CHECK:         }
func @test_array_uniform(%ci1 : i1, %ci32 : i32) {
  %0 = llhd.array_uniform %ci1 : !llhd.array<3xi1>
  %1 = llhd.array_uniform %ci32 : !llhd.array<4xi32>

  return
}