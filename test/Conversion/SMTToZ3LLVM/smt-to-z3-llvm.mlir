// RUN: circt-opt %s --lower-smt-to-z3-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @test
func.func @test() {
  // CHECK-NEXT: [[V0:%.+]] = llvm.mlir.addressof @ctx : !llvm.ptr
  // CHECK-NEXT: [[CTX:%.+]] = llvm.load [[V0]] : !llvm.ptr -> !llvm.ptr

  // CHECK-NEXT: [[FOUR:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
  // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<4 x i8>
  // CHECK-NEXT: [[ZERO:%.+]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[ZERO]], [[A0]][0] : !llvm.array<4 x i8>
  // CHECK-NEXT: [[ZERO:%.+]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[ZERO]], [[A1]][1] : !llvm.array<4 x i8>
  // CHECK-NEXT: [[ZERO:%.+]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: [[A3:%.+]] = llvm.insertvalue [[ZERO]], [[A2]][2] : !llvm.array<4 x i8>
  // CHECK-NEXT: [[ZERO:%.+]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: [[A4:%.+]] = llvm.insertvalue [[ZERO]], [[A3]][3] : !llvm.array<4 x i8>
  // CHECK-NEXT: llvm.store [[A4]], [[STORAGE]] : !llvm.array<4 x i8>, !llvm.ptr
  // CHECK-NEXT: [[C0:%.+]] = llvm.call @Z3_mk_bv_numeral([[CTX]], [[FOUR]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<4>

  // CHECK-NEXT: llvm.call @Z3_mk_bvneg([[CTX]], [[C0]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %0 = smt.bv.neg %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvadd([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %1 = smt.bv.add %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsub([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %2 = smt.bv.sub %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvmul([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %3 = smt.bv.mul %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvurem([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %4 = smt.bv.urem %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsrem([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %5 = smt.bv.srem %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvumod([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %6 = smt.bv.umod %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsmod([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %7 = smt.bv.smod %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvudiv([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %8 = smt.bv.udiv %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsdiv([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %9 = smt.bv.sdiv %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvshl([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %10 = smt.bv.shl %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvlshr([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %11 = smt.bv.lshr %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvashr([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %12 = smt.bv.ashr %c0_bv32, %c0_bv32 : !smt.bv<4>

  // CHECK-NEXT: llvm.call @Z3_mk_bvnot([[CTX]], [[C0]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %13 = smt.bv.not %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvand([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %14 = smt.bv.and %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvor([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %15 = smt.bv.or %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvxor([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %16 = smt.bv.xor %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvnand([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %17 = smt.bv.nand %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvnor([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %18 = smt.bv.nor %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvxnor([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %19 = smt.bv.xnor %c0_bv32, %c0_bv32 : !smt.bv<4>

  // CHECK-NEXT: llvm.call @Z3_mk_concat([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %20 = smt.bv.concat %c0_bv32, %c0_bv32 : !smt.bv<4>, !smt.bv<4>
  // CHECK-NEXT: [[TWO:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT: llvm.call @Z3_mk_extract([[CTX]], [[THREE]], [[TWO]], [[C0]]) : (!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
  %21 = smt.bv.extract %c0_bv32 from 2 : (!smt.bv<4>) -> !smt.bv<2>
  // CHECK-NEXT: [[TWO:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call @Z3_mk_repeat([[CTX]], [[TWO]], [[C0]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %22 = smt.bv.repeat 2 times %c0_bv32 : !smt.bv<4>

  // CHECK-NEXT: llvm.call @Z3_mk_bvslt([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %23 = smt.bv.cmp slt %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsle([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %24 = smt.bv.cmp sle %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsgt([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %25 = smt.bv.cmp sgt %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvsge([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %26 = smt.bv.cmp sge %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvult([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %27 = smt.bv.cmp ult %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvule([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %28 = smt.bv.cmp ule %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvugt([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %29 = smt.bv.cmp ugt %c0_bv32, %c0_bv32 : !smt.bv<4>
  // CHECK-NEXT: llvm.call @Z3_mk_bvuge([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %30 = smt.bv.cmp uge %c0_bv32, %c0_bv32 : !smt.bv<4>

  // CHECK-NEXT: [[EQ:%.+]] = llvm.call @Z3_mk_eq([[CTX]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %31 = smt.eq %c0_bv32, %c0_bv32 : !smt.bv<4>

  // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
  // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[C0]], [[A0]][0] : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[C0]], [[A1]][1] : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A3:%.+]] = llvm.insertvalue [[C0]], [[A2]][2] : !llvm.array<3 x ptr>
  // CHECK-NEXT: llvm.store [[A3]], [[STORAGE]] : !llvm.array<3 x ptr>, !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_distinct([[CTX]], [[THREE]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %32 = smt.distinct %c0_bv32, %c0_bv32, %c0_bv32 : !smt.bv<4>, !smt.bv<4>, !smt.bv<4>

  // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
  // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[EQ]], [[A0]][0] : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[EQ]], [[A1]][1] : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A3:%.+]] = llvm.insertvalue [[EQ]], [[A2]][2] : !llvm.array<3 x ptr>
  // CHECK-NEXT: llvm.store [[A3]], [[STORAGE]] : !llvm.array<3 x ptr>, !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_and([[CTX]], [[THREE]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %33 = smt.and %31, %31, %31

  // CHECK-NEXT: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca [[ONE]] x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
  // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[EQ]], [[A0]][0] : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A2:%.+]] = llvm.insertvalue [[EQ]], [[A1]][1] : !llvm.array<3 x ptr>
  // CHECK-NEXT: [[A3:%.+]] = llvm.insertvalue [[EQ]], [[A2]][2] : !llvm.array<3 x ptr>
  // CHECK-NEXT: llvm.store [[A3]], [[STORAGE]] : !llvm.array<3 x ptr>, !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_or(%1, [[THREE]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %34 = smt.or %31, %31, %31

  // CHECK-NEXT: llvm.call @Z3_mk_ite([[CTX]], [[EQ]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %35 = smt.ite %31, %c0_bv32, %c0_bv32 : !smt.bv<4>

  // CHECK-NEXT: llvm.call @Z3_mk_not([[CTX]], [[EQ]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %36 = smt.not %31
  // CHECK-NEXT: llvm.call @Z3_mk_xor([[CTX]], [[EQ]], [[EQ]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %37 = smt.xor %31, %31
  // CHECK-NEXT: llvm.call @Z3_mk_implies([[CTX]], [[EQ]], [[EQ]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %38 = smt.implies %31, %31

  // CHECK-NEXT: [[FOUR:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[SORT:%.+]] = llvm.call @Z3_mk_bv_sort([[CTX]], [[FOUR]]) : (!llvm.ptr, i32) -> !llvm.ptr
  // CHECK-NEXT: [[STR:%.+]] = llvm.mlir.addressof @a : !llvm.ptr
  // CHECK-NEXT: [[SYM:%.+]] = llvm.call @Z3_mk_string_symbol([[CTX]], [[STR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_const([[CTX]], [[SYM]], [[SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %39 = smt.declare_const "a" : !smt.bv<4>

  // CHECK-NEXT: [[SOLVER:%.+]] = llvm.call @Z3_mk_solver([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  %s = smt.solver_create "s"

  // CHECK-NEXT: llvm.call @Z3_solver_assert([[CTX]], [[SOLVER]], [[EQ]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  smt.assert %s, %31

  // CHECK-NEXT: llvm.call @Z3_solver_check([[CTX]], [[SOLVER]]) : (!llvm.ptr, !llvm.ptr) -> i32
  smt.check_sat %s

  // CHECK-NEXT: [[FOUR:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[BV_SORT:%.+]] = llvm.call @Z3_mk_bv_sort([[CTX]], [[FOUR]]) : (!llvm.ptr, i32) -> !llvm.ptr
  // CHECK-NEXT: [[ARR:%.+]] = llvm.call @Z3_mk_const_array([[CTX]], [[BV_SORT]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %40 = smt.array.broadcast %c0_bv32 : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

  // CHECK-NEXT: llvm.call @Z3_mk_select([[CTX]], [[ARR]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %41 = smt.array.select %40[%c0_bv32] : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

  // CHECK-NEXT: llvm.call @Z3_mk_store([[CTX]], [[ARR]], [[C0]], [[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %42 = smt.array.store %40[%c0_bv32], %c0_bv32 : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

  // CHECK-NEXT: llvm.call @Z3_mk_array_default([[CTX]], [[ARR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %43 = smt.array.default %40 : !smt.array<[!smt.bv<4> -> !smt.bv<4>]>

  // CHECK-NEXT: llvm.return
  return
}

// CHECK-LABEL: llvm.mlir.global internal @ctx()
llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
// CHECK-DAG: llvm.mlir.global private constant @a("a\00") {addr_space = 0 : i32}

// CHECK-DAG: llvm.func @Z3_mk_bv_numeral(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvneg(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvadd(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsub(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvmul(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvurem(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsrem(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvumod(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsmod(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvudiv(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsdiv(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvshl(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvlshr(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvashr(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvnot(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvand(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvor(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvxor(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvnand(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvnor(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvxnor(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_concat(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_extract(!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_repeat(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvslt(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsle(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsgt(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvsge(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvult(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvule(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvugt(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bvuge(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_eq(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_distinct(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_and(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_or(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bv_sort(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_string_symbol(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_const(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_solver(!llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_solver_assert(!llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @Z3_solver_check(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @Z3_mk_const_array(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_select(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_store(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_array_default(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
