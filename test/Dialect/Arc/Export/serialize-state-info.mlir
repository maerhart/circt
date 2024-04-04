// RUN: circt-translate %s --export-arc-model-info | FileCheck %s

// CHECK-LABEL: "name": "Foo"
// CHECK-NEXT: "numStateBytes": 17
arc.model @Foo {
^bb0(%arg0: !arc.layout<@FooLayout>):
}

arc.layout @FooLayout {
  // CHECK:      "name": "a"
  // CHECK-NEXT: "offset": 0
  // CHECK-NEXT: "numBits": 19
  // CHECK-NEXT: "type": "input"
  arc.entry input @a : i19
  // CHECK:      "name": "b"
  // CHECK-NEXT: "offset": 4
  // CHECK-NEXT: "numBits": 42
  // CHECK-NEXT: "type": "output"
  arc.entry output @b : i42
  // CHECK:      "name": "c"
  // CHECK-NEXT: "offset": 16
  // CHECK-NEXT: "numBits": 8
  // CHECK-NEXT: "type": "inout"
  arc.entry inout @c : i8
}

// CHECK-LABEL: "name": "Bar"
// CHECK-NEXT: "numStateBytes": 240
arc.model @Bar {
^bb0(%arg0: !arc.layout<@BarLayout>):
}

arc.layout @BarLayout {
  // CHECK-NOT: "name": "pad"
  arc.entry padding @pad : i192
  // CHECK:      "name": "x"
  // CHECK-NEXT: "offset": 32
  // CHECK-NEXT: "numBits": 63
  // CHECK-NEXT: "type": "register"
  arc.entry register @x : i63
  // CHECK:      "name": "y"
  // CHECK-NEXT: "offset": 40
  // CHECK-NEXT: "numBits": 17
  // CHECK-NEXT: "type": "memory"
  // CHECK-NEXT: "stride": 4
  // CHECK-NEXT: "depth": 5
  arc.entry memory @y : !arc.memory<5 x i17, i3>
  // CHECK:      "name": "z"
  // CHECK-NEXT: "offset": 64
  // CHECK-NEXT: "numBits": 1337
  // CHECK-NEXT: "type": "wire"
  arc.entry wire @z : i1337
}
