// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// This test checks correct simulation of the following operations and ensures
// that the endianess semantics as described in the rational are followed.
// * hw.struct_create
// * hw.struct_extract
// * hw.struct_inject

// CHECK: 0ps 0d 0e  root/ext  0x00
// CHECK-NEXT: 0ps 0d 0e  root/struct  0x00000000
// CHECK-NEXT: 1000ps 0d 0e  root/ext  0xff
// CHECK-NEXT: 1000ps 0d 0e  root/struct  0xffffff00
llhd.entity @root () -> () {
    %allset = hw.constant -1 : i8
    %zero = hw.constant 0 : i8
    %init = hw.constant 0 : i32

    %struct = hw.struct_create (%allset, %zero, %allset, %zero) : !hw.struct<a: i8, b: i8, c: i8, d: i8>
    %ext = hw.struct_extract %struct["c"] : !hw.struct<a: i8, b: i8, c: i8, d: i8>
    %inj = hw.struct_inject %struct["b"], %allset : !hw.struct<a: i8, b: i8, c: i8, d: i8>

    %structsig = llhd.sig "struct" %init : i32
    %extsig = llhd.sig "ext" %zero : i8

    %0 = hw.bitcast %inj : (!hw.struct<a: i8, b: i8, c: i8, d: i8>) -> i32

    %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

    llhd.drv %structsig, %0 after %time : !llhd.sig<i32>
    llhd.drv %extsig, %ext after %time : !llhd.sig<i8>
}
