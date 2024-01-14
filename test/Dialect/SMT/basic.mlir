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

  // CHECK: %{{.*}} = smt.eq %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %23 = smt.eq %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.distinct %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>, !smt.bv<32>
  %24 = smt.distinct %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>, !smt.bv<32>

  // CHECK: %{{.*}} = smt.eq %arg0, %arg0 {smt.some_attr} : !smt.bool
  %25 = smt.eq %arg0, %arg0 {smt.some_attr} : !smt.bool
  // CHECK: %{{.*}} = smt.distinct %arg0, %arg0, %arg0 {smt.some_attr} : !smt.bool, !smt.bool, !smt.bool
  %26 = smt.distinct %arg0, %arg0, %arg0 {smt.some_attr} : !smt.bool, !smt.bool, !smt.bool

  // CHECK: %{{.*}} = smt.bv.cmp slt %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %27 = smt.bv.cmp slt %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp sle %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %28 = smt.bv.cmp sle %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp sgt %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %29 = smt.bv.cmp sgt %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp sge %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %30 = smt.bv.cmp sge %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp ult %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %31 = smt.bv.cmp ult %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp ule %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %32 = smt.bv.cmp ule %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp ugt %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %33 = smt.bv.cmp ugt %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.cmp uge %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %34 = smt.bv.cmp uge %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.bv.concat %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>, !smt.bv<32>
  %35 = smt.bv.concat %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>, !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.extract %c0_bv32 from 8 {smt.some_attr} : (!smt.bv<32>) -> !smt.bv<16>
  %36 = smt.bv.extract %c0_bv32 from 8 {smt.some_attr} : (!smt.bv<32>) -> !smt.bv<16>
  // CHECK: %{{.*}} = smt.bv.repeat 2 times %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %37 = smt.bv.repeat 2 times %c0_bv32 {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.ite %arg0, %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>
  %38 = smt.ite %arg0, %c0_bv32, %c0_bv32 {smt.some_attr} : !smt.bv<32>

  // CEHCK: %{{.*}} = smt.not %arg0 {smt.some_attr}
  %39 = smt.not %arg0 {smt.some_attr}
  // CHECK: %{{.*}} = smt.and %arg0, %arg0, %arg0 {smt.some_attr}
  %40 = smt.and %arg0, %arg0, %arg0 {smt.some_attr}
  // CHECK: %{{.*}} = smt.or %arg0, %arg0, %arg0 {smt.some_attr}
  %41 = smt.or %arg0, %arg0, %arg0 {smt.some_attr}
  // CHECK: %{{.*}} = smt.xor %arg0, %arg0 {smt.some_attr}
  %42 = smt.xor %arg0, %arg0 {smt.some_attr}
  // CHECK: %{{.*}} = smt.implies %arg0, %arg0 {smt.some_attr}
  %43 = smt.implies %arg0, %arg0 {smt.some_attr}

  return
}
