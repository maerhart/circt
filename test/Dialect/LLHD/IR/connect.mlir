// RUN: circt-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @connect_ports
// CHECK-SAME: (%[[IN:.+]]: [[TYPE:.+]], %[[OUT:.+]]: [[TYPE]])
hw.module @connect_ports(%in: !hw.inout<i32>, %out: !hw.inout<i32>) {
// CHECK-NEXT: llhd.con %[[IN]], %[[OUT]] : [[TYPE]]
  llhd.con %in, %out : !hw.inout<i32>
}
