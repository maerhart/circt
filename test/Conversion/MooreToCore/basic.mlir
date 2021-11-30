// RUN: circt-opt %s -convert-moore-to-core -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
    // CHECK-NEXT: %c5_i32 = hw.constant 5 : i32
    %0 = moore.mir.constant 5 : !moore.sv.int
    // CHECK-NEXT: %c3_i32 = hw.constant 3 : i32
    // CHECK-NEXT: [[SIG:%.*]] = llhd.sig "varname" %c3_i32 : i32
    %1 = moore.mir.vardecl "varname" = 3 : !moore.sv.int
    // CHECK-NEXT: [[TIME:%.*]] = llhd.constant_time #llhd.time<0s, 0d, 1e>
    // CHECK-NEXT: llhd.drv [[SIG]], %c5_i32 after [[TIME]] : !llhd.sig<i32>
    moore.mir.blocking_assign %1 = %0 : !moore.sv.int
}

// CHECK-LABEL: llhd.entity @test2
llhd.entity @test2() -> () {
    // CHECK-NEXT: %c5_i32 = hw.constant 5 : i32
    %c = moore.mir.constant 5 : !moore.sv.int
    // CHECK-NEXT: %c-1_i32 = hw.constant -1 : i32
    // CHECK-NEXT: {{%.*}} = comb.xor %c5_i32, %c-1_i32 : i32
    %0 = moore.mir.not %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.and %c5_i32, %c5_i32 : i32
    %1 = moore.mir.and %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.or %c5_i32, %c5_i32 : i32
    %2 = moore.mir.or %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.xor %c5_i32, %c5_i32 : i32
    %3 = moore.mir.xor %c, %c : !moore.sv.int

    // CHECK-NEXT: %c-1_i32_0 = hw.constant -1 : i32
    // CHECK-NEXT: {{%.*}} = comb.icmp eq %c5_i32, %c-1_i32_0 : i32
    %4 = moore.mir.and_reduce %c : !moore.sv.int
    // CHECK-NEXT: %c0_i32 = hw.constant 0 : i32
    // CHECK-NEXT: {{%.*}} = comb.icmp ne %c5_i32, %c0_i32 : i32
    %5 = moore.mir.or_reduce %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.parity %c5_i32 : i32
    %6 = moore.mir.xor_reduce %c : !moore.sv.int

    // CHECK-NEXT: %c0_i32_1 = hw.constant 0 : i32
    // CHECK-NEXT: {{%.*}} = comb.sub %c0_i32_1, %c5_i32 : i32
    %7 = moore.mir.neg %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.add %c5_i32, %c5_i32 : i32
    %8 = moore.mir.add %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.sub %c5_i32, %c5_i32 : i32
    %9 = moore.mir.sub %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.mul %c5_i32, %c5_i32 : i32
    %10 = moore.mir.mul %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.divu %c5_i32, %c5_i32 : i32
    %11 = moore.mir.div %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.divu %c5_i32, %c5_i32 : i32
    %12 = moore.mir.pow %c, %c : !moore.sv.int
    // CHECK-NEXT: {{%.*}} = comb.modu %c5_i32, %c5_i32 : i32
    %13 = moore.mir.mod %c, %c : !moore.sv.int

    // %14 = moore.mir.eq %c, %c  : !moore.sv.int
    // %15 = moore.mir.neq %c, %c : !moore.sv.int
    // %16 = moore.mir.lt %c, %c  : !moore.sv.int
    // %17 = moore.mir.leq %c, %c : !moore.sv.int
    // %18 = moore.mir.gt %c, %c  : !moore.sv.int
    // %19 = moore.mir.geq %c, %c : !moore.sv.int

    // %20 = moore.mir.shll %c, %c : !moore.sv.int, !moore.sv.int
    // %21 = moore.mir.shrl %c, %c : !moore.sv.int, !moore.sv.int
    // %22 = moore.mir.shla %c, %c : !moore.sv.int, !moore.sv.int
    // %23 = moore.mir.shra %c, %c : !moore.sv.int, !moore.sv.int

    // %24 = moore.mir.ternary %14 then %c else %c : !moore.sv.int

    // %var = moore.mir.vardecl "var" = 0 : !moore.sv.int
    // %25 = moore.mir.assign_get_old %var = %c : !moore.sv.int
    // %26 = moore.mir.assign_get_new %var = %c : !moore.sv.int
}
