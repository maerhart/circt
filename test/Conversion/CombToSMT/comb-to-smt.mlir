// RUN: circt-opt %s --convert-comb-to-smt | FileCheck %s

// CHECK-LABEL: func @test
func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>, %a2: !smt.bv<32>, %a3: !smt.bv<32>, %a4: !smt.bv<1>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a1 : !smt.bv<32> to i32
  %arg2 = builtin.unrealized_conversion_cast %a2 : !smt.bv<32> to i32
  %arg3 = builtin.unrealized_conversion_cast %a3 : !smt.bv<32> to i32
  %arg4 = builtin.unrealized_conversion_cast %a4 : !smt.bv<1> to i1

  // CHECK: smt.bv.sdiv %arg0, %arg1 : !smt.bv<32>
  %0 = comb.divs %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.udiv %arg0, %arg1 : !smt.bv<32>
  %1 = comb.divu %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.srem %arg0, %arg1 : !smt.bv<32>
  %2 = comb.mods %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.urem %arg0, %arg1 : !smt.bv<32>
  %3 = comb.modu %arg0, %arg1 : i32

  // CHECK-NEXT: smt.bv.sub %arg0, %arg1 : !smt.bv<32>
  %7 = comb.sub %arg0, %arg1 : i32

  // CHECK-NEXT: [[A1:%.+]] = smt.bv.add %arg0, %arg1 : !smt.bv<32>
  // CHECK-NEXT: [[A2:%.+]] = smt.bv.add [[A1]], %arg2 : !smt.bv<32>
  // CHECK-NEXT: smt.bv.add [[A2]], %arg3 : !smt.bv<32>
  %8 = comb.add %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[B1:%.+]] = smt.bv.mul %arg0, %arg1 : !smt.bv<32>
  // CHECK-NEXT: [[B2:%.+]] = smt.bv.mul [[B1]], %arg2 : !smt.bv<32>
  // CHECK-NEXT: smt.bv.mul [[B2]], %arg3 : !smt.bv<32>
  %9 = comb.mul %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[C1:%.+]] = smt.bv.and %arg0, %arg1 : !smt.bv<32>
  // CHECK-NEXT: [[C2:%.+]] = smt.bv.and [[C1]], %arg2 : !smt.bv<32>
  // CHECK-NEXT: smt.bv.and [[C2]], %arg3 : !smt.bv<32>
  %10 = comb.and %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[D1:%.+]] = smt.bv.or %arg0, %arg1 : !smt.bv<32>
  // CHECK-NEXT: [[D2:%.+]] = smt.bv.or [[D1]], %arg2 : !smt.bv<32>
  // CHECK-NEXT: smt.bv.or [[D2]], %arg3 : !smt.bv<32>
  %11 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[E1:%.+]] = smt.bv.xor %arg0, %arg1 : !smt.bv<32>
  // CHECK-NEXT: [[E2:%.+]] = smt.bv.xor [[E1]], %arg2 : !smt.bv<32>
  // CHECK-NEXT: smt.bv.xor [[E2]], %arg3 : !smt.bv<32>
  %12 = comb.xor %arg0, %arg1, %arg2, %arg3 : i32

  // CHECK-NEXT: [[CONST1:%.+]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
  // CHECK-NEXT: [[COND:%.+]] = smt.eq %arg4, [[CONST1]] : !smt.bv<1>
  // CHECK-NEXT: smt.ite [[COND]], %arg0, %arg1 : !smt.bv<32>
  %13 = comb.mux %arg4, %arg0, %arg1 : i32

  // CHECK-NEXT: smt.eq %arg0, %arg1 : !smt.bv<32>
  %14 = comb.icmp eq %arg0, %arg1 : i32
  // CHECK-NEXT: smt.distinct %arg0, %arg1 : !smt.bv<32>, !smt.bv<32>
  %15 = comb.icmp ne %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sle %arg0, %arg1 : !smt.bv<32>
  %20 = comb.icmp sle %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp slt %arg0, %arg1 : !smt.bv<32>
  %21 = comb.icmp slt %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ule %arg0, %arg1 : !smt.bv<32>
  %22 = comb.icmp ule %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ult %arg0, %arg1 : !smt.bv<32>
  %23 = comb.icmp ult %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sge %arg0, %arg1 : !smt.bv<32>
  %24 = comb.icmp sge %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sgt %arg0, %arg1 : !smt.bv<32>
  %25 = comb.icmp sgt %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp uge %arg0, %arg1 : !smt.bv<32>
  %26 = comb.icmp uge %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ugt %arg0, %arg1 : !smt.bv<32>
  %27 = comb.icmp ugt %arg0, %arg1 : i32

  // CHECK-NEXT: smt.bv.extract %arg0 from 5 : (!smt.bv<32>) -> !smt.bv<16>
  %28 = comb.extract %arg0 from 5 : (i32) -> i16
  // CHECK-NEXT: smt.bv.concat %arg0, %arg1 : !smt.bv<32>, !smt.bv<32>
  %29 = comb.concat %arg0, %arg1 : i32, i32
  // CHECK-NEXT: smt.bv.repeat 32 times %arg4 : !smt.bv<1>
  %30 = comb.replicate %arg4 : (i1) -> i32

  // CHECK-NEXT: %{{.*}} = smt.bv.shl %arg0, %arg1 : !smt.bv<32>
  %32 = comb.shl %arg0, %arg1 : i32
  // CHECK-NEXT: %{{.*}} = smt.bv.ashr %arg0, %arg1 : !smt.bv<32>
  %33 = comb.shrs %arg0, %arg1 : i32
  // CHECK-NEXT: %{{.*}} = smt.bv.lshr %arg0, %arg1 : !smt.bv<32>
  %34 = comb.shru %arg0, %arg1 : i32

  return
}
