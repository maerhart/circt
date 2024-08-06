// RUN: circt-opt %s | FileCheck %s --strict-whitespace

hw.module @foo(in %clk: !seq.clock, in %ce: i1, in %i: i32) {
  // CHECK: seq.compreg %{{[^,]*}} clock %{{[^ ]*}} : i32
  seq.compreg %i clock %clk : i32
  // CHECK: seq.compreg.ce %{{[^,]*}} clock %{{[^,]*}}, %{{[^ ]*}} : i32
  seq.compreg.ce %i clock %clk, %ce : i32
  // CHECK: seq.shiftreg[3] %{{[^,]*}}, %{{[^,]*}}, %{{[^ ]*}} : i32
  seq.shiftreg[3] %i, %clk, %ce : i32
}
