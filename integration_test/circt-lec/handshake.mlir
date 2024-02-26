//  RUN: circt-lec %s -c1=passthrough -c2=sinkSource | FileCheck %s --check-prefix=CHECK_SOURCE_SINK

// Those two circuits are not equivalent because 'passthrough' always produces
// the same amount of tokens as it gets in, while 'sinkSource' can produce more
// or less than it gets as input.

llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @Z3_mk_config() -> !llvm.ptr
llvm.func @Z3_del_config(!llvm.ptr)
llvm.func @Z3_mk_context(!llvm.ptr) -> !llvm.ptr

llvm.func @main(%argc: i32, %argv: !llvm.ptr) -> i32 {
  %0 = func.call @entry() : () -> i32
  llvm.return %0 : i32
}

func.func @e2() -> i32 {
  // TODO: the smt IR is in this separate function because smt constants would be
  // moved above the context initialization calls and thus fail.
  %solver = smt.solver_create "solver"
  %cond_br_true = smt.declare_func "cond_br_true" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<32>]>) -> !smt.array<[!smt.int -> !smt.bv<32>]>>
  %cond_br_false = smt.declare_func "cond_br_false" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<32>]>) -> !smt.array<[!smt.int -> !smt.bv<32>]>>
  %join_none_none = smt.declare_func "join_none_none" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.sort<"none">]>, !smt.array<[!smt.int -> !smt.sort<"none">]>) -> !smt.array<[!smt.int -> !smt.sort<"none">]>>
  %join_i1_none = smt.declare_func "join_i1_none" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.sort<"none">]>) -> !smt.array<[!smt.int -> !smt.sort<"none">]>>
  %not = smt.declare_func "not" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
  %constant_c1i1 = smt.declare_func "constant_c1i1" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.sort<"none">]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
  %source = smt.declare_func "source" fresh : !smt.func<() -> !smt.array<[!smt.int -> !smt.sort<"none">]>>
  %xor_i1 = smt.declare_func "xor_i1" fresh : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>

  // cond_br_true(c, d) == cond_br_false(not(c), d)
  %0 = smt.forall ["cond", "data"] patterns() weight 0 {
  ^bb0(%cond: !smt.array<[!smt.int -> !smt.bv<1>]>, %data: !smt.array<[!smt.int -> !smt.bv<32>]>):
    %1 = smt.apply_func %cond_br_true(%cond, %data) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<32>]>) -> !smt.array<[!smt.int -> !smt.bv<32>]>>
    %2 = smt.apply_func %not(%cond) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
    %3 = smt.apply_func %cond_br_false(%2, %data) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<32>]>) -> !smt.array<[!smt.int -> !smt.bv<32>]>>
    %4 = smt.eq %1, %3 : !smt.array<[!smt.int -> !smt.bv<32>]>
    smt.yield %4 : !smt.bool
  }
  smt.assert %solver, %0

  // not(not(x)) == x
  %c2 = smt.forall ["data"] patterns() weight 0 {
  ^bb0(%data: !smt.array<[!smt.int -> !smt.bv<1>]>):
    %1 = smt.apply_func %not(%data) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
    %2 = smt.apply_func %not(%1) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
    %4 = smt.eq %data, %2 : !smt.array<[!smt.int -> !smt.bv<1>]>
    smt.yield %4 : !smt.bool
  }
  smt.assert %solver, %c2

  // not(x) == ~x
  %c3 = smt.forall ["data", "idx"] patterns() weight 0 {
  ^bb0(%data: !smt.array<[!smt.int -> !smt.bv<1>]>, %idx: !smt.int):
    %1 = smt.apply_func %not(%data) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
    %2 = smt.array.select %1[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %3 = smt.array.select %data[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %4 = smt.bv.not %3 : !smt.bv<1>
    %5 = smt.eq %4, %2 : !smt.bv<1>
    smt.yield %5 : !smt.bool
  }
  smt.assert %solver, %c3

  // xor(x, y) == x ^ y
  %c4 = smt.forall ["lhs", "rhs", "idx"] patterns() weight 0 {
  ^bb0(%lhs: !smt.array<[!smt.int -> !smt.bv<1>]>, %rhs: !smt.array<[!smt.int -> !smt.bv<1>]>, %idx: !smt.int):
    %1 = smt.apply_func %xor_i1(%lhs, %rhs) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<1>]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
    %2 = smt.array.select %1[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %3 = smt.array.select %lhs[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %4 = smt.array.select %rhs[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %5 = smt.bv.xor %3, %4 : !smt.bv<1>
    %6 = smt.eq %5, %2 : !smt.bv<1>
    smt.yield %6 : !smt.bool
  }
  smt.assert %solver, %c4

  // not(x) == ~x
  %c5 = smt.forall ["trigger", "idx"] patterns() weight 0 {
  ^bb0(%trigger: !smt.array<[!smt.int -> !smt.sort<"none">]>, %idx: !smt.int):
    %1 = smt.apply_func %constant_c1i1(%trigger) : !smt.func<(!smt.array<[!smt.int -> !smt.sort<"none">]>) -> !smt.array<[!smt.int -> !smt.bv<1>]>>
    %2 = smt.array.select %1[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %3 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
    %4 = smt.eq %3, %2 : !smt.bv<1>
    smt.yield %4 : !smt.bool
  }
  smt.assert %solver, %c5

  %1 = smt.declare_const "data" : !smt.array<[!smt.int -> !smt.bv<32>]>
  %2 = smt.declare_const "cond" : !smt.array<[!smt.int -> !smt.bv<1>]>
  %3 = smt.declare_const "xor" : !smt.array<[!smt.int -> !smt.bv<1>]>
  %5 = smt.forall ["idx"] patterns() weight 0 {
  ^bb0(%idx: !smt.int):
    %6 = smt.array.select %2[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %7 = smt.array.select %3[%idx] : !smt.array<[!smt.int -> !smt.bv<1>]>
    %8 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
    %9 = smt.bv.xor %8, %6 : !smt.bv<1>
    %10 = smt.eq %7, %9 : !smt.bv<1>
    smt.yield %10 : !smt.bool
  }
  smt.assert %solver, %5

  %4 = smt.apply_func %cond_br_false(%2, %1) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<32>]>) -> !smt.array<[!smt.int -> !smt.bv<32>]>>
  %6 = smt.apply_func %cond_br_true(%3, %1) : !smt.func<(!smt.array<[!smt.int -> !smt.bv<1>]>, !smt.array<[!smt.int -> !smt.bv<32>]>) -> !smt.array<[!smt.int -> !smt.bv<32>]>>
  %7 = smt.distinct %4, %6 : !smt.array<[!smt.int -> !smt.bv<32>]>, !smt.array<[!smt.int -> !smt.bv<32>]>
  smt.assert %solver, %7

  %res = smt.check_sat %solver
  return %res : i32
}

func.func @entry() -> i32 {
  %config = llvm.call @Z3_mk_config() : () -> !llvm.ptr
  %ctx = llvm.call @Z3_mk_context(%config) : (!llvm.ptr) -> !llvm.ptr
  %global = llvm.mlir.addressof @ctx : !llvm.ptr
  llvm.store %ctx, %global {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.call @Z3_del_config(%config) : (!llvm.ptr) -> ()
  %0 = func.call @e2() : () -> i32
  return %0 : i32
}

func.func @entry() -> i32 {
  %config = llvm.call @Z3_mk_config() : () -> !llvm.ptr
  %ctx = llvm.call @Z3_mk_context(%config) : (!llvm.ptr) -> !llvm.ptr
  %global = llvm.mlir.addressof @ctx : !llvm.ptr
  llvm.store %ctx, %global {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.call @Z3_del_config(%config) : (!llvm.ptr) -> ()

  %solver = smt.solver_create "solver"
  %cond_br_true = smt.declare_func "cond_br_true" fresh : !smt.func<(!smt.sort<"token_stream_bv1">, !smt.sort<"token_stream_bv32">) -> !smt.sort<"token_stream_bv32">>
  %cond_br_false = smt.declare_func "cond_br_false" fresh : !smt.func<(!smt.sort<"token_stream_bv1">, !smt.sort<"token_stream_bv32">) -> !smt.sort<"token_stream_bv32">>
  %not = smt.declare_func "not" fresh : !smt.func<(!smt.sort<"token_stream_bv1">) -> !smt.sort<"token_stream_bv1">>

  // cond_br_true(c, d) == cond_br_false(not(c), d)
  %0 = smt.forall ["cond", "data"] patterns() weight 0 {
  ^bb0(%cond: !smt.sort<"token_stream_bv1">, %data: !smt.sort<"token_stream_bv32">):
    %1 = smt.apply_func %cond_br_true(%cond, %data) : !smt.func<(!smt.sort<"token_stream_bv1">, !smt.sort<"token_stream_bv32">) -> !smt.sort<"token_stream_bv32">>
    %2 = smt.apply_func %not(%cond) : !smt.func<(!smt.sort<"token_stream_bv1">) -> !smt.sort<"token_stream_bv1">>
    %3 = smt.apply_func %cond_br_false(%2, %data) : !smt.func<(!smt.sort<"token_stream_bv1">, !smt.sort<"token_stream_bv32">) -> !smt.sort<"token_stream_bv32">>
    %4 = smt.eq %1, %3 : !smt.sort<"token_stream_bv32">
    smt.yield %4 : !smt.bool
  }
  smt.assert %solver, %0

  // not(not(x)) == x
  %c2 = smt.forall ["data"] patterns() weight 0 {
  ^bb0(%data: !smt.sort<"token_stream_bv1">):
    %1 = smt.apply_func %not(%data) : !smt.func<(!smt.sort<"token_stream_bv1">) -> !smt.sort<"token_stream_bv1">>
    %2 = smt.apply_func %not(%1) : !smt.func<(!smt.sort<"token_stream_bv1">) -> !smt.sort<"token_stream_bv1">>
    %4 = smt.eq %data, %2 : !smt.sort<"token_stream_bv1">
    smt.yield %4 : !smt.bool
  }
  smt.assert %solver, %c2

  %1 = smt.declare_const "data" : !smt.sort<"token_stream_bv32">
  %2 = smt.declare_const "cond" : !smt.sort<"token_stream_bv1">
  %3 = smt.apply_func %cond_br_false(%2, %1) : !smt.func<(!smt.sort<"token_stream_bv1">, !smt.sort<"token_stream_bv32">) -> !smt.sort<"token_stream_bv32">>
  // %4 = smt.apply_func %not(%2) : !smt.func<(!smt.sort<"token_stream_bv1">) -> !smt.sort<"token_stream_bv1">>
  %5 = smt.apply_func %cond_br_true(%2, %1) : !smt.func<(!smt.sort<"token_stream_bv1">, !smt.sort<"token_stream_bv32">) -> !smt.sort<"token_stream_bv32">>
  %6 = smt.distinct %3, %5 : !smt.sort<"token_stream_bv32">, !smt.sort<"token_stream_bv32">
  smt.assert %solver, %6

  %res = smt.check_sat %solver

  return %res : i32
}

// handshake.func @passthrough2to1(%arg0: none, %arg1: none) -> none {
//   return %arg0 : none
// }

// handshake.func @passthrough1to1(%arg0: none) -> none {
//   return %arg0 : none
// }

// handshake.func @buffer(%arg0: i32) -> i32 {
//   %0 = handshake.buffer [1000] fifo %arg0 : i32
//   return %0 : i32
// }

// handshake.func @passthrough1to1int(%arg0: i32) -> i32 {
//   return %arg0 : i32
// }

// handshake.func @const0(%arg0: none) -> i32 {
//   %0 = handshake.constant %arg0 {value = 0 : i32} : i32
//   return %0 : i32
// }

// handshake.func @const1(%arg0: none) -> i32 {
//   %0 = handshake.constant %arg0 {value = 1 : i32} : i32
//   return %0 : i32
// }

// handshake.func @const0FromSource(%arg0: none) -> i32 {
//   handshake.sink %arg0 : none
//   %0 = handshake.source
//   %1 = handshake.constant %0 {value = 1 : i32} : i32
//   return %1 : i32
// }

// handshake.func @sinkSource(%arg0: none) -> none {
//   handshake.sink %arg0 : none
//   %0 = handshake.source
//   handshake.return %0 : none
// }

// // // -----
// // //  RUN: circt-lec %s -c1=forkLhs -c2=forkRhs | FileCheck %s --check-prefix=CHECK_FORK_COMMUTATIVE

// // // Those two circuits are equivalent because the number of output tokens and
// // // their values always match, only the exact delay differs (which we don't care
// // // about in latency-insensitive proofs).

// // Join has very bad performance, can only prove this function for token
// // sequences of up to 2 in reasonable time.
// handshake.func @forkJoin(%arg0: none) -> none {
//   %0:2 = handshake.fork [2] %arg0 : none
//   %1 = handshake.join %0#0, %0#1 : none, none
//   return %1 : none
// }

// handshake.func @forkRhs(%arg0: i32) -> (i32, i32, i32) {
//   %0:2 = handshake.fork [2] %arg0 : i32
//   %1:2 = handshake.fork [2] %0#1 : i32
//   return %0#0, %1#0, %1#1 : i32, i32, i32
// }

// // -----
// //  RUN: circt-lec %s -c1=joinLhs -c2=joinRhs | FileCheck %s --check-prefix=CHECK_JOIN_TREE

// handshake.func @join1(%arg0: none, %arg1: none) -> none {
//   %0 = handshake.join %arg0, %arg1 : none, none
//   return %0 : none
// }

// handshake.func @join2(%arg0: none, %arg1: none) -> none {
//   %0 = handshake.join %arg0, %arg1 : none, none
//   return %0 : none
// }

// // Those are equivalent.

// // handshake.func @joinLhs(%arg0: i32, %arg1: i32, %arg2: i32) -> none {
// //   // Values: %1 = constant none thus holds trivially
// //   // Length:
// //   // len(%0) == min(len(%arg0), len(%arg1))
// //   // len(%1) == min(len(%0), len(%arg2))
// //   %0 = handshake.join %arg0, %arg1 : i32, i32
// //   %1 = handshake.join %0, %arg2 : none, i32
// //   return %1 : none
// // }

// // handshake.func @joinRhs(%arg0: i32, %arg1: i32, %arg2: i32) -> none {
// //   // Values: %1 = constant none  thus holds trivially
// //   // Length:
// //   // len(%0) == min(len(%arg1), len(%arg2))
// //   // len(%1) == min(len(%arg0), len(%0))
// //   %0 = handshake.join %arg1, %arg2 : i32, i32
// //   %1 = handshake.join %arg0, %0 : i32, none
// //   return %1 : none
// // }

// // -----
// //  RUN: circt-lec %s -c1=joinLhs -c2=joinRhs | FileCheck %s --check-prefix=CHECK_JOIN_TREE

// handshake.func @branchFork(%cond: i1, %arg1: i32) -> (i32, i32, i32) {
//   %a, %b = handshake.cond_br %cond, %arg1 : i32
//   %c, %d = handshake.fork [2] %b : i32
//   return %a, %c, %d : i32, i32, i32
// }

// handshake.func @doubleBranchFork(%cond: i1, %arg1: i32) -> (i32, i32, i32) {
//   %c1, %c2 = handshake.fork [2] %cond : i1
//   %a1, %a2 = handshake.fork [2] %arg1 : i32
//   %a, %b = handshake.cond_br %c1, %a1 : i32
//   %c, %d = handshake.cond_br %c2, %a2 : i32
//   handshake.sink %c : i32
//   return %a, %b, %d : i32, i32, i32
// }

// handshake.func @doubleBranchForkWrong(%cond: i1, %arg1: i32) -> (i32, i32, i32) {
//   %c1, %c2 = handshake.fork [2] %cond : i1
//   %a1, %a2 = handshake.fork [2] %arg1 : i32
//   %a, %b = handshake.cond_br %c1, %a1 : i32
//   %c, %d = handshake.cond_br %c2, %a2 : i32
//   handshake.sink %d : i32
//   return %a, %b, %c : i32, i32, i32
// }

// handshake.func @branch(%cond: i1, %arg1: i32) -> (i32, i32) {
//   %a, %b = handshake.cond_br %cond, %arg1 : i32
//   return %a, %b : i32, i32
// }

// handshake.func @negBranch(%cond: i1, %arg1: i32) -> (i32, i32) {
//   %0 = handshake.source
//   %1 = handshake.join %cond, %0 : i1, none
//   %true = handshake.constant %1 {value = 1 : i1} : i1
//   %neg_cond = comb.xor %cond, %true : i1
//   %a, %b = handshake.cond_br %neg_cond, %arg1 : i32
//   return %b, %a : i32, i32
// }

// // This is ~16x slower to prove equivalence with @branch than @negBranch is!
// handshake.func @negBranch2(%cond: i1, %arg1: i32) -> (i32, i32) {
//   %1 = handshake.join %cond, %cond : i1, i1
//   %true = handshake.constant %1 {value = 1 : i1} : i1
//   %neg_cond = comb.xor %cond, %true : i1
//   %a, %b = handshake.cond_br %neg_cond, %arg1 : i32
//   return %b, %a : i32, i32
// }

// // This is ~86x slower to prove equivalence with @branch than @negBranch is!
// handshake.func @negBranch3(%cond: i1, %arg1: i32) -> (i32, i32) {
//   %0 = handshake.source
//   %true = handshake.constant %0 {value = 1 : i1} : i1
//   %neg_cond = comb.xor %cond, %true : i1
//   %a, %b = handshake.cond_br %neg_cond, %arg1 : i32
//   return %b, %a : i32, i32
// }

// handshake.func @negNeg(%arg0: i32) -> i32 {
//   %0 = handshake.join %arg0, %arg0 : i32, i32
//   // %0 = handshake.source
//   %c-1 = handshake.constant %0 {value = -1 : i32} : i32
//   %neg = comb.xor %arg0, %c-1 : i32
//   %res = comb.xor %neg, %c-1 : i32
//   return %res : i32
// }

// handshake.func @divAndMulBy2(%arg0: i32) -> i32 {
//   %ctrl = handshake.join %arg0, %arg0 : i32, i32
//   %c1 = handshake.constant %ctrl {value = 1 : i32} : i32
//   %c2 = handshake.constant %ctrl {value = 2 : i32} : i32
//   %0 = comb.shrs %arg0, %c1 : i32
//   %1 = comb.mul %0, %c2 : i32
//   return %1 : i32
// }

// handshake.func @mulAndDivBy2(%arg0: i32) -> i32 {
//   %ctrl = handshake.join %arg0, %arg0 : i32, i32
//   %c1 = handshake.constant %ctrl {value = 1 : i32} : i32
//   %c2 = handshake.constant %ctrl {value = 2 : i32} : i32
//   %0 = comb.mul %arg0, %c2 : i32
//   %1 = comb.shrs %0, %c1 : i32
//   return %1 : i32
// }
