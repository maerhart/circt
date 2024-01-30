// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @arrayOperations
// CHECK-SAME:  ([[A0:%.+]]: !smt.bool, [[A1:%.+]]: !smt.int)
func.func @arrayOperations(%arg0: !smt.bool, %arg1: !smt.int) {
  // CHECK-NEXT: [[V0:%.+]] = smt.array.broadcast [[A0]] {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>
  %0 = smt.array.broadcast %arg0 {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>

  // CHECK-NEXT: [[V1:%.+]] = smt.array.select [[V0]][[[A1]]] {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>
  %1 = smt.array.select %0[%arg1] {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>

  // CHECK-NEXT: [[V2:%.+]] = smt.array.store [[V0]][[[A1]]], [[A0]] {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>
  %2 = smt.array.store %0[%arg1], %arg0 {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>

  // CHECK-NEXT: smt.array.default [[V2]] {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>
  %3 = smt.array.default %2 {smt.some_attr} : !smt.array<[!smt.int -> !smt.bool]>

  return
}
