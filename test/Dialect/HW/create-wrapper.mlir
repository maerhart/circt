// RUN: circt-opt --create-wrapper=wrappedModule=B

hw.module @A (%in0: i32) -> (out2: i32) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: hw.module.extern @B(%in0: i32, %inout: !hw.inout<i32>) -> (out2: i32)
// CHECK-NEXT: hw.module @BWrapper(%in0: i32, %inout: !hw.inout<i32>) -> (out2: i32) {
// CHECK-NEXT:   %B.out2 = hw.instance "B" @B(in0: %in0: i32, inout: %inout: !hw.inout<i32>) -> (out2: i32)
// CHECK-NEXT:   hw.output %B.out2 : i32
// CHECK-NEXT: }
hw.module @B (%in0: i32, %inout: !hw.inout<i32>) -> (out2: i32) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

hw.module @C () -> () {}
