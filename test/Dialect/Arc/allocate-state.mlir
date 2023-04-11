// RUN: circt-opt %s --arc-allocate-state | FileCheck %s

// CHECK-LABEL: arc.model "test"
arc.model "test" {
^bb0:
  // CHECK-NEXT: ([[PTR:%.+]]: !arc.storage<5724>):

  // CHECK-NEXT: arc.alloc_storage [[PTR]][0] : (!arc.storage<5724>) -> !arc.storage<1143>
  // CHECK-NEXT: arc.passthrough {
  arc.passthrough {
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][0] : !arc.storage<5724> -> !arc.storage<1143>
    %0 = arc.alloc_state !arc.state<i1>
    arc.alloc_state !arc.state<i8>
    arc.alloc_state !arc.state<i16>
    arc.alloc_state !arc.state<i32>
    arc.alloc_state !arc.state<i64>
    arc.alloc_state !arc.state<i9001>
    %1 = arc.alloc_state !arc.state<i1>
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 0 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 1 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 2 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 4 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 8 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 16 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 1142 : i32}
    // CHECK-NEXT: scf.execute_region {
    scf.execute_region {
      arc.state_read %0 : <i1>
      // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][0] : !arc.storage<5724> -> !arc.storage<1143>
      // CHECK-NEXT: [[STATE:%.+]] = arc.storage.get [[SUBPTR]][0] : !arc.storage<1143> -> !arc.state<i1>
      // CHECK-NEXT: arc.state_read [[STATE]] : <i1>
      arc.state_read %1 : <i1>
      // CHECK-NEXT: [[STATE:%.+]] = arc.storage.get [[SUBPTR]][1142] : !arc.storage<1143> -> !arc.state<i1>
      // CHECK-NEXT: arc.state_read [[STATE]] : <i1>
      scf.yield
      // CHECK-NEXT: scf.yield
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: arc.alloc_storage [[PTR]][1144] : (!arc.storage<5724>) -> !arc.storage<4577>
  // CHECK-NEXT: arc.passthrough {
  arc.passthrough {
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][1144] : !arc.storage<5724> -> !arc.storage<4577>
    arc.alloc_memory !arc.memory<4 x i1>
    arc.alloc_memory !arc.memory<4 x i8>
    arc.alloc_memory !arc.memory<4 x i16>
    arc.alloc_memory !arc.memory<4 x i32>
    arc.alloc_memory !arc.memory<4 x i64>
    arc.alloc_memory !arc.memory<4 x i9001>
    arc.alloc_state !arc.state<i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 0 : i32, stride = 1 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i1, 1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 4 : i32, stride = 1 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i8, 1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 8 : i32, stride = 2 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i16, 2>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 16 : i32, stride = 4 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i32, 4>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 32 : i32, stride = 8 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i64, 8>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 64 : i32, stride = 1128 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i9001, 1128>
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 4576 : i32}
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: arc.alloc_storage [[PTR]][5722] : (!arc.storage<5724>) -> !arc.storage<2>
  // CHECK-NEXT: arc.passthrough {
  arc.passthrough {
    arc.root_input "x" !arc.state<i1>
    arc.root_output "y" !arc.state<i1>
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][5722] : !arc.storage<5724> -> !arc.storage<2>
    // CHECK-NEXT: arc.root_input "x", [[SUBPTR]] {offset = 0 : i32}
    // CHECK-NEXT: arc.root_output "y", [[SUBPTR]] {offset = 1 : i32}
  }
  // CHECK-NEXT: }
}
