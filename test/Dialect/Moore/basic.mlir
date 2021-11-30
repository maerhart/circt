// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
    // CHECK-NEXT: [[CONST:%.*]] = moore.mir.constant 5 : !moore.sv.int
    // CHECK-NEXT: [[VAR:%.*]] = moore.mir.vardecl "varname" = 3 : !moore.sv.int
    // CHECK-NEXT: moore.mir.blocking_assign [[VAR]] = [[CONST]] : !moore.sv.int
    %0 = moore.mir.constant 5 : !moore.sv.int
    %1 = moore.mir.vardecl "varname" = 3 : !moore.sv.int
    moore.mir.blocking_assign %1 = %0 : !moore.sv.int
}

// CHECK-LABEL: llhd.entity @test2
llhd.entity @test2() -> () {
    %c = moore.mir.constant 5 : !moore.sv.int
    %0 = moore.mir.not %c : !moore.sv.int
    %1 = moore.mir.and %c, %c : !moore.sv.int
    %2 = moore.mir.or %c, %c : !moore.sv.int
    %3 = moore.mir.xor %c, %c : !moore.sv.int

    %4 = moore.mir.and_reduce %c : !moore.sv.int
    %5 = moore.mir.or_reduce %c : !moore.sv.int
    %6 = moore.mir.xor_reduce %c : !moore.sv.int

    %7 = moore.mir.neg %c : !moore.sv.int
    %8 = moore.mir.add %c, %c : !moore.sv.int
    %9 = moore.mir.sub %c, %c : !moore.sv.int
    %10 = moore.mir.mul %c, %c : !moore.sv.int
    %11 = moore.mir.div %c, %c : !moore.sv.int
    %12 = moore.mir.pow %c, %c : !moore.sv.int
    %13 = moore.mir.mod %c, %c : !moore.sv.int

    %14 = moore.mir.eq %c, %c  : !moore.sv.int
    %15 = moore.mir.neq %c, %c : !moore.sv.int
    %16 = moore.mir.lt %c, %c  : !moore.sv.int
    %17 = moore.mir.leq %c, %c : !moore.sv.int
    %18 = moore.mir.gt %c, %c  : !moore.sv.int
    %19 = moore.mir.geq %c, %c : !moore.sv.int

    %20 = moore.mir.shll %c, %c : !moore.sv.int, !moore.sv.int
    %21 = moore.mir.shrl %c, %c : !moore.sv.int, !moore.sv.int
    %22 = moore.mir.shla %c, %c : !moore.sv.int, !moore.sv.int
    %23 = moore.mir.shra %c, %c : !moore.sv.int, !moore.sv.int

    %24 = moore.mir.ternary %14 then %c else %c : !moore.sv.int

    %var = moore.mir.vardecl "var" = 0 : !moore.sv.int
    %25 = moore.mir.assign_get_old %var = %c : !moore.sv.int
    %26 = moore.mir.assign_get_new %var = %c : !moore.sv.int
}
