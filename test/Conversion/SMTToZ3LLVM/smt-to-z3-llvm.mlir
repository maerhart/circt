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

  // CHECK-NEXT: llvm.call @Z3_mk_true([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  %true = smt.constant true
  // CHECK-NEXT: llvm.call @Z3_mk_false([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  %false = smt.constant false

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

  // CHECK-NEXT: [[STR123:%.+]] = llvm.mlir.addressof @"123" : !llvm.ptr
  // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[NUMERAL:%.+]] = llvm.call @Z3_mk_numeral([[CTX]], [[STR123]], [[INT_SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[C123:%.+]] = llvm.call @Z3_mk_unary_minus([[CTX]], [[NUMERAL]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %44 = smt.int.constant -123

  // CHECK: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[ALLOCA:%.+]] = llvm.alloca
  // CHECK: llvm.call @Z3_mk_add([[CTX]], [[THREE]], [[ALLOCA]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %45 = smt.int.add %44, %44, %44

  // CHECK: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[ALLOCA:%.+]] = llvm.alloca
  // CHECK: llvm.call @Z3_mk_mul([[CTX]], [[THREE]], [[ALLOCA]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %46 = smt.int.mul %44, %44, %44

  // CHECK: [[THREE:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[ALLOCA:%.+]] = llvm.alloca
  // CHECK: llvm.call @Z3_mk_sub([[CTX]], [[THREE]], [[ALLOCA]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %47 = smt.int.sub %44, %44, %44

  // CHECK-NEXT: llvm.call @Z3_mk_div([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %48 = smt.int.div %44, %44
  // CHECK-NEXT: llvm.call @Z3_mk_mod([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %49 = smt.int.mod %44, %44
  // CHECK-NEXT: llvm.call @Z3_mk_rem([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %50 = smt.int.rem %44, %44
  // CHECK-NEXT: llvm.call @Z3_mk_power([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %51 = smt.int.pow %44, %44

  // CHECK-NEXT: llvm.call @Z3_mk_le([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %52 = smt.int.cmp le %44, %44
  // CHECK-NEXT: llvm.call @Z3_mk_lt([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %53 = smt.int.cmp lt %44, %44
  // CHECK-NEXT: llvm.call @Z3_mk_ge([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %54 = smt.int.cmp ge %44, %44
  // CHECK-NEXT: llvm.call @Z3_mk_gt([[CTX]], [[C123]], [[C123]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %55 = smt.int.cmp gt %44, %44

  // CHECK-NEXT: [[STR0:%.+]] = llvm.mlir.addressof @"0" : !llvm.ptr
  // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_numeral([[CTX]], [[STR0]], [[INT_SORT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %56 = smt.int.constant 0

  // CHECK-NEXT: [[TRUE:%.+]] = llvm.call @Z3_mk_true([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[NUM_PATTERNS:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[A0:%.+]] = llvm.mlir.undef : !llvm.array<1 x ptr>
  // CHECK-NEXT: [[A1:%.+]] = llvm.insertvalue [[TRUE]], [[A0]][0] : !llvm.array<1 x ptr>
  // CHECK-NEXT: llvm.store [[A1]], [[STORAGE]] : !llvm.array<1 x ptr>, !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_pattern([[CTX]], [[NUM_PATTERNS]], [[STORAGE]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %57 = smt.pattern_create {
    %58 = smt.constant true
    smt.yield %58 : !smt.bool
  }

  // CHECK-NEXT: [[WEIGHT:%.+]] = llvm.mlir.constant(42 : i32) : i32
  // CHECK-NEXT: [[NUM_PATTERNS:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[PATTERN_STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[PA0:%.+]] = llvm.mlir.undef : !llvm.array<1 x ptr>
  // CHECK-NEXT: [[PA1:%.+]] = llvm.insertvalue {{.*}}, [[PA0]][0] : !llvm.array<1 x ptr>
  // CHECK-NEXT: llvm.store [[PA1]], [[PATTERN_STORAGE]] : !llvm.array<1 x ptr>, !llvm.ptr
  // CHECK-NEXT: [[NUM_BOUND_VARS:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[SORT_STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[NAME_STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[SA0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[NA0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[A_IDX:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[A:%.+]] = llvm.call @Z3_mk_bound([[CTX]], [[A_IDX]], [[INT_SORT]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[SA1:%.+]] = llvm.insertvalue [[INT_SORT]], [[SA0]][0] : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[A_ADDR:%.+]] = llvm.mlir.addressof @a : !llvm.ptr
  // CHECK-NEXT: [[A_SYM:%.+]] = llvm.call @Z3_mk_string_symbol([[CTX]], [[A_ADDR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[NA1:%.+]] = llvm.insertvalue [[A_SYM]], [[NA0]][0] : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[B_IDX:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[B:%.+]] = llvm.call @Z3_mk_bound([[CTX]], [[B_IDX]], [[INT_SORT]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[SA2:%.+]] = llvm.insertvalue [[INT_SORT]], [[SA1]][1] : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[B_ADDR:%.+]] = llvm.mlir.addressof @b : !llvm.ptr
  // CHECK-NEXT: [[B_SYM:%.+]] = llvm.call @Z3_mk_string_symbol([[CTX]], [[B_ADDR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[NA2:%.+]] = llvm.insertvalue [[B_SYM]], [[NA1]][1] : !llvm.array<2 x ptr>
  // CHECK-NEXT: llvm.store [[SA2]], [[SORT_STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
  // CHECK-NEXT: llvm.store [[NA2]], [[NAME_STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
  // CHECK-NEXT: [[BODY:%.+]] = llvm.call @Z3_mk_eq([[CTX]], [[A]], [[B]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_forall([[CTX]], [[WEIGHT]], [[NUM_PATTERNS]], [[PATTERN_STORAGE]], [[NUM_BOUND_VARS]], [[SORT_STORAGE]], [[NAME_STORAGE]], [[BODY]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %58 = smt.forall ["a", "b"] patterns(%57) weight 42 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %59 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %59 : !smt.bool
  }

  // CHECK-NEXT: [[WEIGHT:%.+]] = llvm.mlir.constant(42 : i32) : i32
  // CHECK-NEXT: [[NUM_PATTERNS:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[PATTERN_STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[PA0:%.+]] = llvm.mlir.undef : !llvm.array<1 x ptr>
  // CHECK-NEXT: [[PA1:%.+]] = llvm.insertvalue {{.*}}, [[PA0]][0] : !llvm.array<1 x ptr>
  // CHECK-NEXT: llvm.store [[PA1]], [[PATTERN_STORAGE]] : !llvm.array<1 x ptr>, !llvm.ptr
  // CHECK-NEXT: [[NUM_BOUND_VARS:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[SORT_STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[NAME_STORAGE:%.+]] = llvm.alloca
  // CHECK-NEXT: [[SA0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[NA0:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[A_IDX:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[A:%.+]] = llvm.call @Z3_mk_bound([[CTX]], [[A_IDX]], [[INT_SORT]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[SA1:%.+]] = llvm.insertvalue [[INT_SORT]], [[SA0]][0] : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[A_ADDR:%.+]] = llvm.mlir.addressof @a : !llvm.ptr
  // CHECK-NEXT: [[A_SYM:%.+]] = llvm.call @Z3_mk_string_symbol([[CTX]], [[A_ADDR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[NA1:%.+]] = llvm.insertvalue [[A_SYM]], [[NA0]][0] : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[INT_SORT:%.+]] = llvm.call @Z3_mk_int_sort([[CTX]]) : (!llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[B_IDX:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[B:%.+]] = llvm.call @Z3_mk_bound([[CTX]], [[B_IDX]], [[INT_SORT]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[SA2:%.+]] = llvm.insertvalue [[INT_SORT]], [[SA1]][1] : !llvm.array<2 x ptr>
  // CHECK-NEXT: [[B_ADDR:%.+]] = llvm.mlir.addressof @b : !llvm.ptr
  // CHECK-NEXT: [[B_SYM:%.+]] = llvm.call @Z3_mk_string_symbol([[CTX]], [[B_ADDR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: [[NA2:%.+]] = llvm.insertvalue [[B_SYM]], [[NA1]][1] : !llvm.array<2 x ptr>
  // CHECK-NEXT: llvm.store [[SA2]], [[SORT_STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
  // CHECK-NEXT: llvm.store [[NA2]], [[NAME_STORAGE]] : !llvm.array<2 x ptr>, !llvm.ptr
  // CHECK-NEXT: [[BODY:%.+]] = llvm.call @Z3_mk_eq([[CTX]], [[A]], [[B]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK-NEXT: llvm.call @Z3_mk_exists([[CTX]], [[WEIGHT]], [[NUM_PATTERNS]], [[PATTERN_STORAGE]], [[NUM_BOUND_VARS]], [[SORT_STORAGE]], [[NAME_STORAGE]], [[BODY]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %59 = smt.exists ["a", "b"] patterns(%57) weight 42 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %60 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %60 : !smt.bool
  }

  // CHECK: %204 = llvm.call @Z3_mk_bool_sort(%97) : (!llvm.ptr) -> !llvm.ptr
  // CHECK: %205 = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK: %206 = llvm.call @Z3_mk_bool_sort(%97) : (!llvm.ptr) -> !llvm.ptr
  // CHECK: %207 = llvm.insertvalue %206, %205[0] : !llvm.array<2 x ptr>
  // CHECK: %208 = llvm.call @Z3_mk_bool_sort(%97) : (!llvm.ptr) -> !llvm.ptr
  // CHECK: %209 = llvm.insertvalue %208, %207[1] : !llvm.array<2 x ptr>
  // CHECK: %210 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %211 = llvm.alloca %210 x !llvm.ptr : (i32) -> !llvm.ptr
  // CHECK: llvm.store %209, %211 : !llvm.array<2 x ptr>, !llvm.ptr
  // CHECK: %212 = llvm.mlir.addressof @func1 : !llvm.ptr
  // CHECK: %213 = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %214 = llvm.call @Z3_mk_fresh_func_decl(%97, %212, %213, %211, %204) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %60 = smt.declare_func "func1" fresh : !smt.func<(!smt.bool, !smt.bool) -> !smt.bool>

  // CHECK: %215 = llvm.call @Z3_mk_bool_sort(%97) : (!llvm.ptr) -> !llvm.ptr
  // CHECK: %216 = llvm.mlir.undef : !llvm.array<1 x ptr>
  // CHECK: %217 = llvm.mlir.addressof @uninterpreted_sort : !llvm.ptr
  // CHECK: %218 = llvm.call @Z3_mk_string_symbol(%97, %217) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK: %219 = llvm.call @Z3_mk_uninterpreted_sort(%97, %218) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK: %220 = llvm.insertvalue %219, %216[0] : !llvm.array<1 x ptr>
  // CHECK: %221 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %222 = llvm.alloca %221 x !llvm.ptr : (i32) -> !llvm.ptr
  // CHECK: llvm.store %220, %222 : !llvm.array<1 x ptr>, !llvm.ptr
  // CHECK: %223 = llvm.mlir.addressof @func2 : !llvm.ptr
  // CHECK: %224 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %225 = llvm.call @Z3_mk_string_symbol(%97, %223) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK: %226 = llvm.call @Z3_mk_func_decl(%97, %225, %224, %222, %215) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %61 = smt.declare_func "func2" : !smt.func<(!smt.sort<"uninterpreted_sort">) -> !smt.bool>

  // CHECK: %227 = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK: %228 = llvm.insertvalue %15, %227[0] : !llvm.array<2 x ptr>
  // CHECK: %229 = llvm.insertvalue %16, %228[1] : !llvm.array<2 x ptr>
  // CHECK: %230 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %231 = llvm.alloca %230 x !llvm.ptr : (i32) -> !llvm.ptr
  // CHECK: llvm.store %229, %231 : !llvm.array<2 x ptr>, !llvm.ptr
  // CHECK: %232 = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %233 = llvm.call @Z3_mk_app(%97, %214, %232, %231) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %62 = smt.apply_func %60(%true, %false) : !smt.func<(!smt.bool, !smt.bool) -> !smt.bool>

  // CHECK-NEXT: llvm.return
  return
}

// CHECK-LABEL: llvm.mlir.global internal @ctx()
llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
// CHECK-DAG: llvm.mlir.global private constant @a("a\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global private constant @"0"("0\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global private constant @"123"("123\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global private constant @b("b\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global private constant @func1("func1\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global private constant @func2("func2\00") {addr_space = 0 : i32}

// CHECK-DAG: llvm.func @Z3_mk_bv_numeral(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_true(!llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_false(!llvm.ptr) -> !llvm.ptr
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
// CHECK-DAG: llvm.func @Z3_mk_int_sort(!llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_numeral(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_unary_minus(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_add(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_mul(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_sub(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_div(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_mod(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_rem(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_power(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_le(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_lt(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_ge(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_gt(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_pattern(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_bound(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_forall(!llvm.ptr, i32, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_exists(!llvm.ptr, i32, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_func_decl(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_fresh_func_decl(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @Z3_mk_app(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
