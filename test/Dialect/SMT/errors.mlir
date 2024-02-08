// RUN: circt-opt %s --split-input-file --verify-diagnostics

func.func @extraction(%arg0: !smt.bv<32>) {
  // expected-error @below {{slice too big}}
  %36 = smt.bv.extract %arg0 from 20 {smt.some_attr} : (!smt.bv<32>) -> !smt.bv<16>
  return
}

// -----

func.func @patternCreateHasAtLeastOnePattern() {
  // expected-error @below {{must yield at least one expression}}
  %0 = smt.pattern_create {
    smt.yield
  }
  return
}

// -----

func.func @patternCreateHasNoBlockArguments() {
  // expected-error @below {{must have zero block arguments}}
  %0 = smt.pattern_create {
  ^bb0(%arg0: !smt.int):
    smt.yield %arg0 : !smt.int
  }
  return
}

// -----

func.func @forallNumberOfDeclNamesMustMatchNumArgs() {
  // expected-error @below {{number of bound variable names must match number of block arguments}}
  %1 = smt.forall ["a"] patterns() weight 0 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  return
}

// -----

func.func @existsNumberOfDeclNamesMustMatchNumArgs() {
  // expected-error @below {{number of bound variable names must match number of block arguments}}
  %1 = smt.exists ["a"] patterns() weight 0 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  return
}

// -----

func.func @forallYieldMustHaveExactlyOneBoolValue() {
  // expected-error @below {{yielded value must be of '!smt.bool' type}}
  %1 = smt.forall ["a", "b"] patterns() weight 0 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.int.add %arg2, %arg3
    smt.yield %2 : !smt.int
  }
  return
}

// -----

func.func @forallYieldMustHaveExactlyOneBoolValue() {
  // expected-error @below {{must have exactly one yielded value}}
  %1 = smt.forall ["a", "b"] patterns() weight 0 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    smt.yield
  }
  return
}

// -----

func.func @existsYieldMustHaveExactlyOneBoolValue() {
  // expected-error @below {{yielded value must be of '!smt.bool' type}}
  %1 = smt.exists ["a", "b"] patterns() weight 0 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.int.add %arg2, %arg3
    smt.yield %2 : !smt.int
  }
  return
}

// -----

func.func @existsYieldMustHaveExactlyOneBoolValue() {
  // expected-error @below {{must have exactly one yielded value}}
  %1 = smt.exists ["a", "b"] patterns() weight 0 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    smt.yield
  }
  return
}
