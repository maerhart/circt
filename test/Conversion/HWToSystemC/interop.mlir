// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

hw.module @Bar (%a: i32) -> (b: i32) {
  hw.output %a : i32
}

hw.module @Foo (%x: i32) -> (y: i32) {
  %1 = systemc.model.verilated @Bar (["a"]: %x) -> (["b"]) : (i32) -> i32
  hw.output %1 : i32
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