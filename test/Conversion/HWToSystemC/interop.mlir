// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

module attributes {hw.backend_choice=#hw.backend_choice<1>} { // ExportVerilog
  hw.module @Bar (%a: i32, %b: i32) -> (c: i32) {
    %0 = comb.add %a, %b : i32
    hw.output %0 : i32
  }
}

module attributes {hw.backend_choice=#hw.backend_choice<0>} { // ExportSystemC
  hw.module.extern @Bar (%a: i32, %b: i32) -> (c: i32)

  hw.module @Foo (%x: i32) -> (y: i32) {
    %1 = systemc.model.verilated @Bar (["a", "b"]: %x, %x) -> (["c"]) : (i32, i32) -> i32
    hw.output %1 : i32
  }
}


module attributes {hw.backend_choice=#hw.backend_choice<2>} { // llhd-sim
  llhd.entity @Bar (%a: !llhd.sig<i32>, %b: !llhd.sig<i32>) -> (%c: !llhd.sig<i32>) {
    %time = llhd.constant_time <0ns, 1d, 0e>
    %a_prb = llhd.prb %a : !llhd.sig<i32>
    %b_prb = llhd.prb %b : !llhd.sig<i32>
    %0 = comb.add %a_prb, %b_prb : i32
    llhd.drv %c, %0 after %time : !llhd.sig<i32>
  }
}

module attributes {hw.backend_choice=#hw.backend_choice<0>} { // ExportSystemC
  llhd.entity.extern @Bar (%a: !llhd.sig<i32>, %b: !llhd.sig<i32>) -> (%c: !llhd.sig<i32>)

  hw.module @Foo (%x: i32) -> (y: i32) {
    %1 = systemc.model.llhdsim @Bar (["a", "b"]: %x, %x) -> (["c"]) : (i32, i32) -> i32
    hw.output %1 : i32
  }
}


module attributes {hw.backend_choice=#hw.backend_choice<2>} { // ExportVerilog
  hw.module @Bar (%a: i32, %b: i32) -> (c: i32) {
    %0 = comb.add %a, %b : i32
    hw.output %0 : i32
  }
}

module attributes {hw.backend_choice=#hw.backend_choice<1>} { // llhd-sim
  hw.module.extern @Bar (%a: i32, %b: i32) -> (%c: i32)

  llhd.entity @Root () -> () {
    %time = llhd.constant_time <0ns, 1d, 0e>
    %0 = hw.constant 3 : i32
    %1 = hw.constant 5 : i32
    %2 = systemc.model.verilated @Bar (["a", "b"]: %0, %1) -> (["c"]) : (i32, i32) -> i32
    %3 = llhd.sig "c" %0 : i32
    llhd.drv %3, %2 after %time : !llhd.sig<i32>
  }
}


// emitc.include <"systemc.h">
// emitc.include "VBar.h"

// systemc.module @Foo2 (%x: !systemc.in<i32>, %y: !systemc.out<i32>) {
//   %vbar = systemc.cpp.variable : !emitc.ptr<!emitc.opaque<"VBar">>

//   systemc.ctor {
//     %0 = systemc.cpp.new () : () -> !emitc.ptr<!emitc.opaque<"VBar">>
//     systemc.cpp.assign %vbar = %0 : !emitc.ptr<!emitc.opaque<"VBar">>

//     systemc.method %func0
//   }

//   %func0 = systemc.func {
//     %0 = systemc.signal.read %x : !systemc.in<i32>

//     %1 = systemc.cpp.member_access %vbar["a"] deref : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
//     systemc.cpp.assign %1 = %0 : i32
//     %2 = systemc.cpp.member_access %vbar["eval"] deref : (!emitc.ptr<!emitc.opaque<"VBar">>) -> !systemc.func_handle
//     systemc.call %2
//     %3 = systemc.cpp.member_access %vbar["b"] deref : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32

//     systemc.signal.write %y, %3 : !systemc.out<i32>
//   }

//   systemc.cpp.destructor {
//     systemc.cpp.delete %vbar : !emitc.ptr<!emitc.opaque<"VBar">>
//   }
// }