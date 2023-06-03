// RUN: circt-opt %s --lower-arc-to-llvm --verify-diagnostics --split-input-file

func.func @stateOp(%arg0: i32, %arg1: i1) -> i32 {
  // expected-error @below {{failed to legalize operation 'arc.state' that was explicitly marked illegal}}
  %0 = arc.state @dummyCallee(%arg0) clock %arg1 lat 1 : (i32) -> i32
  return %0 : i32
}
arc.define @dummyCallee(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}

// -----

func.func @StateAllocation(%arg0: !arc.storage<10>) -> !arc.state<i3> {
  // expected-error @below {{failed to legalize}}
  %0 = arc.alloc_state %arg0 {offset = 2.0 : f32} : (!arc.storage<10>) -> !arc.state<i3>
  // expected-note @below {{existing live user here}}
  return %0 : !arc.state<i3>
}
