// RUN: circt-translate --export-smtlib %s | FileCheck %s

%s = smt.solver_create "solver"

%c = smt.int.constant 0
%true = smt.constant true

// CHECK: (assert (select (store ((as const (Array Int Bool)) true) 0 true) 0))
%0 = smt.array.broadcast %true : !smt.array<[!smt.int -> !smt.bool]>
%1 = smt.array.store %0[%c], %true : !smt.array<[!smt.int -> !smt.bool]>
%2 = smt.array.select %1[%c] : !smt.array<[!smt.int -> !smt.bool]>
smt.assert %s, %2
