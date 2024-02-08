// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @quantifiers
// CHECK-SAME:  (%{{.*}}: !smt.bool, %{{.*}}: !smt.solver)
func.func @quantifiers(%arg0: !smt.bool, %arg1: !smt.solver) {
  // CHECK-NEXT: [[PATTERN:%.+]] = smt.pattern_create attributes {smt.some_attr} {
  // CHECK-NEXT:   smt.constant true
  // CHECK-NEXT:   smt.yield {{.*}} : !smt.bool {smt.some_attr}
  // CHECK-NEXT: }
  %0 = smt.pattern_create attributes {smt.some_attr} {
    %1 = smt.constant true
    smt.yield %1 : !smt.bool {smt.some_attr}
  }

  // CHECK-NEXT: smt.forall ["a", "b"] patterns([[PATTERN]]) weight 0 attributes {smt.some_attr} {
  // CHECK-NEXT: ^bb0({{.*}}: !smt.int, {{.*}}: !smt.int):
  // CHECK-NEXT:   smt.eq
  // CHECK-NEXT:   smt.yield {{.*}} : !smt.bool {smt.some_attr}
  // CHECK-NEXT: }
  %1 = smt.forall ["a", "b"] patterns(%0) weight 0 attributes {smt.some_attr} {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool {smt.some_attr}
  }

  // CHECK-NEXT: smt.exists ["a", "b"] patterns([[PATTERN]]) weight 0 attributes {smt.some_attr} {
  // CHECK-NEXT: ^bb0({{.*}}: !smt.int, {{.*}}: !smt.int):
  // CHECK-NEXT:   smt.eq
  // CHECK-NEXT:   smt.yield {{.*}} : !smt.bool {smt.some_attr}
  // CHECK-NEXT: }
  %2 = smt.exists ["a", "b"] patterns(%0) weight 0 attributes {smt.some_attr} {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %3 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %3 : !smt.bool {smt.some_attr}
  }

  return
}
