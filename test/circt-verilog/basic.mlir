// RUN: circt-verilog %s | FileCheck %s

// CHECK: hw.module @Foo() {
// CHECK: }
moore.module @Foo() {
}
