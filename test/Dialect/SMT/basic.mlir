// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @types
// CHECK-SAME:  (%{{.*}}: !smt.bool, %{{.*}}: !smt.solver)
func.func @types(%arg0: !smt.bool, %arg1: !smt.solver) {
  // CHECK: %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32> {smt.some_attr}
  %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32> {smt.some_attr}

  // CHECK: %{{.*}} = smt.bv.neg %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %0 = smt.bv.neg %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.add %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %1 = smt.bv.add %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.sub %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %2 = smt.bv.sub %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.mul %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %3 = smt.bv.mul %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.urem %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %4 = smt.bv.urem %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.srem %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %5 = smt.bv.srem %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.umod %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %6 = smt.bv.umod %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.smod %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %7 = smt.bv.smod %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.shl %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %8 = smt.bv.shl %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.lshr %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %9 = smt.bv.lshr %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.ashr %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %10 = smt.bv.ashr %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.udiv %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %11 = smt.bv.udiv %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.sdiv %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %12 = smt.bv.sdiv %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.bv.not %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %13 = smt.bv.not %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.and %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %14 = smt.bv.and %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.or %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %15 = smt.bv.or %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.xor %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %16 = smt.bv.xor %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.nand %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %17 = smt.bv.nand %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.nor %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %18 = smt.bv.nor %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.xnor %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %19 = smt.bv.xnor %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>

  // CHECK: [[BOOL:%.+]] = smt.declare_const "a" {smt.some_attr} : !smt.bool
  %20 = smt.declare_const "a" {smt.some_attr} : !smt.bool
  // CHECK: %{{.*}} = smt.declare_const "b" {smt.some_attr} : !smt.bv<32>
  %21 = smt.declare_const "b" {smt.some_attr} : !smt.bv<32>

  // CHECK: [[SOLVER:%.+]] = smt.solver_create "solver" {smt.some_attr}
  %22 = smt.solver_create "solver" {smt.some_attr}
  // CHECK: smt.assert [[SOLVER]], [[BOOL]] {smt.some_attr}
  smt.assert %22, %20 {smt.some_attr}
  // CHECK: smt.check_sat [[SOLVER]] {smt.some_attr}
  %res = smt.check_sat %22 {smt.some_attr}

  return
}
