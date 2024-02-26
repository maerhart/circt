// RUN: circt-translate --export-smtlib %s | FileCheck %s

%s = smt.solver_create "solver"
%0 = smt.int.constant 5
%1 = smt.int.constant 10

// CHECK: (assert (= (+ 5 5 5) 10))
%2 = smt.int.add %0, %0, %0
%a2 = smt.eq %2, %1 : !smt.int
smt.assert %s, %a2

// CHECK: (assert (= (* 5 5 5) 10))
%3 = smt.int.mul %0, %0, %0
%a3 = smt.eq %3, %1 : !smt.int
smt.assert %s, %a3

// CHECK: (assert (= (- 5 5 5) 10))
%4 = smt.int.sub %0, %0, %0
%a4 = smt.eq %4, %1 : !smt.int
smt.assert %s, %a4

// CHECK: (assert (= (div 5 5) 10))
%5 = smt.int.div %0, %0
%a5 = smt.eq %5, %1 : !smt.int
smt.assert %s, %a5

// CHECK: (assert (= (mod 5 5) 10))
%6 = smt.int.mod %0, %0
%a6 = smt.eq %6, %1 : !smt.int
smt.assert %s, %a6

// CHECK: (assert (= (rem 5 5) 10))
%7 = smt.int.rem %0, %0
%a7 = smt.eq %7, %1 : !smt.int
smt.assert %s, %a7

// CHECK: (assert (= (^ 5 5) 10))
%8 = smt.int.pow %0, %0
%a8 = smt.eq %8, %1 : !smt.int
smt.assert %s, %a8

// CHECK: (assert (<= 5 5))
%9 = smt.int.cmp le %0, %0
smt.assert %s, %9

// CHECK: (assert (< 5 5))
%10 = smt.int.cmp lt %0, %0
smt.assert %s, %10

// CHECK: (assert (>= 5 5))
%11 = smt.int.cmp ge %0, %0
smt.assert %s, %11

// CHECK: (assert (> 5 5))
%12 = smt.int.cmp gt %0, %0
smt.assert %s, %12
