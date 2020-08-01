// RUN: llhd-sim %s | FileCheck %s

// CHECK: 0ns 0d 0e  root/proc/s1  0x00000000
// CHECK-NEXT: 0ns 0d 0e  root/s1  0x00000000
// CHECK-NEXRT: 0ns 0d 0e  root/proc/s2  0x00000000
// CHECK-NEXRT: 0ns 0d 0e  root/s2  0x00000000
// CHECK-NEXRT: 0ns 0d 2e  root/proc/s1  0x00000001
// CHECK-NEXRT: 0ns 0d 2e  root/s1  0x00000001
// CHECK-NEXRT: 0ns 0d 3e  root/proc/s2  0x00000001
// CHECK-NEXRT: 0ns 0d 3e  root/s2  0x00000001
// CHECK-NEXRT: 0ns 0d 4e  root/proc/s2  0x00000002
// CHECK-NEXRT: 0ns 0d 4e  root/s2  0x00000002
// CHECK-NEXRT: 0ns 0d 5e  root/proc/s2  0x00000003
// CHECK-NEXRT: 0ns 0d 5e  root/s2  0x00000003
// CHECK-NEXRT: 0ns 0d 8e  root/proc/s1  0x00000004
// CHECK-NEXRT: 0ns 0d 8e  root/s1  0x00000004
// CHECK-NEXRT: 0ns 0d 8e  root/proc/s2  0x00000004
// CHECK-NEXRT: 0ns 0d 8e  root/s2  0x00000004
// CHECK-NEXRT: 0ns 0d 10e  root/proc/s2  0x00000005
// CHECK-NEXRT: 0ns 0d 10e  root/s2  0x00000005
llhd.entity @root () -> () {
  %0 = llhd.const 0 : i32
  %1 = llhd.sig "s1" %0 : i32
  %2 = llhd.sig "s2" %0 : i32
  llhd.inst "proc" @proc () -> (%1, %2) : () -> (!llhd.sig<i32>, !llhd.sig<i32>)
}

llhd.proc @proc () -> (%a : !llhd.sig<i32>, %b : !llhd.sig<i32>) {
  br ^timed
^timed:
  %t1 = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
  %t2 = llhd.const #llhd.time<0ns, 0d, 2e>: !llhd.time
  llhd.wait for %t1, ^observe
^observe:
  %c0 = llhd.const 1 : i32
  %p0 = llhd.prb %b : !llhd.sig<i32>
  %a0 = addi %c0, %p0 : i32
  llhd.drv %a, %a0 after %t1 : !llhd.sig<i32>
  llhd.drv %b, %a0 after %t2 : !llhd.sig<i32>
  llhd.wait (%b : !llhd.sig<i32>), ^timed_observe
^timed_observe:
  %p1 = llhd.prb %b : !llhd.sig<i32>
  %a1 = addi %c0, %p1 : i32
  llhd.drv %b, %a1 after %t1 : !llhd.sig<i32>
  llhd.wait for %t2, (%b : !llhd.sig<i32>), ^overlap_invalidated
^overlap_invalidated:
  %p2 = llhd.prb %b : !llhd.sig<i32>
  %a2 = addi %c0, %p2 : i32
  llhd.drv %b, %a2 after %t1 : !llhd.sig<i32>
  llhd.wait for %t2, ^observe_both
^observe_both:
  %p3 = llhd.prb %b : !llhd.sig<i32>
  %a3 = addi %c0, %p3 : i32
  llhd.drv %a, %a3 after %t2 : !llhd.sig<i32>
  llhd.drv %b, %a3 after %t2 : !llhd.sig<i32>
  llhd.wait (%a, %b : !llhd.sig<i32>, !llhd.sig<i32>), ^end
^end:
  %p4 = llhd.prb %b : !llhd.sig<i32>
  %a4 = addi %c0, %p4 : i32
  llhd.drv %b, %a4 after %t2 : !llhd.sig<i32>
  llhd.halt
}
