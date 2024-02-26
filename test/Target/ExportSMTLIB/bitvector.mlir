// RUN: circt-translate --export-smtlib %s | FileCheck %s

%s = smt.solver_create "solver"

%c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>

// CHECK: (assert (= (bvneg #x00000000) #x00000000))
%0 = smt.bv.neg %c0_bv32 : !smt.bv<32>
%a0 = smt.eq %0, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a0
// CHECK: (assert (= (bvadd #x00000000 #x00000000) #x00000000))
%1 = smt.bv.add %c0_bv32, %c0_bv32 : !smt.bv<32>
%a1 = smt.eq %1, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a1
// CHECK: (assert (= (bvsub #x00000000 #x00000000) #x00000000))
%2 = smt.bv.sub %c0_bv32, %c0_bv32 : !smt.bv<32>
%a2 = smt.eq %2, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a2
// CHECK: (assert (= (bvmul #x00000000 #x00000000) #x00000000))
%3 = smt.bv.mul %c0_bv32, %c0_bv32 : !smt.bv<32>
%a3 = smt.eq %3, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a3
// CHECK: (assert (= (bvurem #x00000000 #x00000000) #x00000000))
%4 = smt.bv.urem %c0_bv32, %c0_bv32 : !smt.bv<32>
%a4 = smt.eq %4, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a4
// CHECK: (assert (= (bvsrem #x00000000 #x00000000) #x00000000))
%5 = smt.bv.srem %c0_bv32, %c0_bv32 : !smt.bv<32>
%a5 = smt.eq %5, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a5
// CHECK: (assert (= (bvsmod #x00000000 #x00000000) #x00000000))
%7 = smt.bv.smod %c0_bv32, %c0_bv32 : !smt.bv<32>
%a7 = smt.eq %7, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a7
// CHECK: (assert (= (bvshl #x00000000 #x00000000) #x00000000))
%8 = smt.bv.shl %c0_bv32, %c0_bv32 : !smt.bv<32>
%a8 = smt.eq %8, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a8
// CHECK: (assert (= (bvlshr #x00000000 #x00000000) #x00000000))
%9 = smt.bv.lshr %c0_bv32, %c0_bv32 : !smt.bv<32>
%a9 = smt.eq %9, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a9
// CHECK: (assert (= (bvashr #x00000000 #x00000000) #x00000000))
%10 = smt.bv.ashr %c0_bv32, %c0_bv32 : !smt.bv<32>
%a10 = smt.eq %10, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a10
// CHECK: (assert (= (bvudiv #x00000000 #x00000000) #x00000000))
%11 = smt.bv.udiv %c0_bv32, %c0_bv32 : !smt.bv<32>
%a11 = smt.eq %11, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a11
// CHECK: (assert (= (bvsdiv #x00000000 #x00000000) #x00000000))
%12 = smt.bv.sdiv %c0_bv32, %c0_bv32 : !smt.bv<32>
%a12 = smt.eq %12, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a12

// CHECK: (assert (= (bvnot #x00000000) #x00000000))
%13 = smt.bv.not %c0_bv32 : !smt.bv<32>
%a13 = smt.eq %13, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a13
// CHECK: (assert (= (bvand #x00000000 #x00000000) #x00000000))
%14 = smt.bv.and %c0_bv32, %c0_bv32 : !smt.bv<32>
%a14 = smt.eq %14, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a14
// CHECK: (assert (= (bvor #x00000000 #x00000000) #x00000000))
%15 = smt.bv.or %c0_bv32, %c0_bv32 : !smt.bv<32>
%a15 = smt.eq %15, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a15
// CHECK: (assert (= (bvxor #x00000000 #x00000000) #x00000000))
%16 = smt.bv.xor %c0_bv32, %c0_bv32 : !smt.bv<32>
%a16 = smt.eq %16, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a16
// CHECK: (assert (= (bvnand #x00000000 #x00000000) #x00000000))
%17 = smt.bv.nand %c0_bv32, %c0_bv32 : !smt.bv<32>
%a17 = smt.eq %17, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a17
// CHECK: (assert (= (bvnor #x00000000 #x00000000) #x00000000))
%18 = smt.bv.nor %c0_bv32, %c0_bv32 : !smt.bv<32>
%a18 = smt.eq %18, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a18
// CHECK: (assert (= (bvxnor #x00000000 #x00000000) #x00000000))
%19 = smt.bv.xnor %c0_bv32, %c0_bv32 : !smt.bv<32>
%a19 = smt.eq %19, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a19

// CHECK: (assert (bvslt #x00000000 #x00000000))
%27 = smt.bv.cmp slt %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %27
// CHECK: (assert (bvsle #x00000000 #x00000000))
%28 = smt.bv.cmp sle %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %28
// CHECK: (assert (bvsgt #x00000000 #x00000000))
%29 = smt.bv.cmp sgt %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %29
// CHECK: (assert (bvsge #x00000000 #x00000000))
%30 = smt.bv.cmp sge %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %30
// CHECK: (assert (bvult #x00000000 #x00000000))
%31 = smt.bv.cmp ult %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %31
// CHECK: (assert (bvule #x00000000 #x00000000))
%32 = smt.bv.cmp ule %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %32
// CHECK: (assert (bvugt #x00000000 #x00000000))
%33 = smt.bv.cmp ugt %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %33
// CHECK: (assert (bvuge #x00000000 #x00000000))
%34 = smt.bv.cmp uge %c0_bv32, %c0_bv32 : !smt.bv<32>
smt.assert %s, %34

// CHECK: (assert (= ((_ repeat 2) ((_ extract 23 8) (concat #x00000000 #x00000000))) #x00000000))
%35 = smt.bv.concat %c0_bv32, %c0_bv32 : !smt.bv<32>, !smt.bv<32>
%36 = smt.bv.extract %35 from 8 : (!smt.bv<64>) -> !smt.bv<16>
%37 = smt.bv.repeat 2 times %36 : !smt.bv<16>
%a37 = smt.eq %37, %c0_bv32 : !smt.bv<32>
smt.assert %s, %a37
