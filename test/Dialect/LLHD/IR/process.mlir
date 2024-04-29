// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: llhd.process @empty() {
llhd.process @empty() {
  // CHECK: llhd.halt
  // CHECK-NEXT: }
  llhd.halt
}

// CHECK-NEXT: llhd.process @inout(inout %{{.*}} : i64, inout %{{.*}} : i64, inout %{{.*}} : i64) {
llhd.process @inout(inout %arg0 : i64, inout %arg1 : i64, inout %out0 : i64) {
  // CHECK-NEXT: llhd.halt
  // CHECK-NEXT: }
  llhd.halt
}
