// RUN: llhd-sim %s -n 10 -r Foo | FileCheck %s

// CHECK: 0ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 1000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 2000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 3000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 4000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 5000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 6000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 7000ps 0d 0e  Foo/toggle  0x01
// CHECK-NEXT: 8000ps 0d 0e  Foo/toggle  0x00
// CHECK-NEXT: 9000ps 0d 0e  Foo/toggle  0x01
llhd.entity @Foo () -> () {
  %0 = llhd.const 0 : i1
  %toggle = llhd.sig "toggle" %0 : i1
  %1 = llhd.prb %toggle : !llhd.sig<i1>
  %2 = llhd.not %1 : i1
  %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  llhd.drv %toggle, %2 after %dt : !llhd.sig<i1>
}
