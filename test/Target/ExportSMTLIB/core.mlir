// RUN: circt-translate --export-smtlib %s | FileCheck %s

%s = smt.solver_create "solver"

%c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
%true = smt.constant true
%false = smt.constant false

// CHECK: (declare-const b (BitVec 32))
// CHECK: (assert (= #x00000000 b))
%21 = smt.declare_const "b" : !smt.bv<32>
%23 = smt.eq %c0_bv32, %21 : !smt.bv<32>
smt.assert %s, %23

// CHECK: (assert (distinct #x00000000 #x00000000))
%24 = smt.distinct %c0_bv32, %c0_bv32 : !smt.bv<32>, !smt.bv<32>
smt.assert %s, %24

// CHECK: (declare-const a Bool)
// CHECK: (assert (= #x00000000 (ite a #x00000000 b)))
%20 = smt.declare_const "a" : !smt.bool
%38 = smt.ite %20, %c0_bv32, %21 : !smt.bv<32>
%4 = smt.eq %c0_bv32, %38 : !smt.bv<32>
smt.assert %s, %4

// CHECK: (assert (=> (xor (not true) (or (and (not true) true false) (not true) true)) false))
%39 = smt.not %true
%40 = smt.and %39, %true, %false
%41 = smt.or %40, %39, %true
%42 = smt.xor %39, %41
%43 = smt.implies %42, %false
smt.assert %s, %43

// CHECK: (declare-fun func1 (Bool Bool) Bool)
// CHECK: (assert (func1 true false))
%44 = smt.declare_func "func1" : !smt.func<(!smt.bool, !smt.bool) -> !smt.bool>
%45 = smt.apply_func %44(%true, %false) : !smt.func<(!smt.bool, !smt.bool) -> !smt.bool>
smt.assert %s, %45

%0 = smt.pattern_create {
  %1 = smt.constant true
  smt.yield %1 : !smt.bool
}

%1 = smt.forall ["a", "b"] patterns() weight 0 {
^bb0(%arg2: !smt.int, %arg3: !smt.int):
  %2 = smt.eq %arg2, %arg3 : !smt.int
  smt.yield %2 : !smt.bool
}
smt.assert %s, %1

%2 = smt.exists ["a", "b"] patterns() weight 0 {
^bb0(%arg2: !smt.int, %arg3: !smt.int):
  %3 = smt.eq %arg2, %arg3 : !smt.int
  smt.yield %3 : !smt.bool
}
smt.assert %s, %2
