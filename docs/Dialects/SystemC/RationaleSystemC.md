# SystemC Dialect Rationale

This document describes various design points of the SystemC dialect, why they
are the way they are, and current status. This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [SystemC Dialect Rationale](#systemc-dialect-rationale)
  - [Introduction](#introduction)
  - [Lowering](#lowering)
  - [Q&A](#qa)


## Introduction

[SystemC](https://en.wikipedia.org/wiki/SystemC) is a library written in C++
to allow functional modeling of systems. The included event-driven simulation
kernel can then be used to simulate a system modeled entirely in SystemC.
Additionally, SystemC is a standard (IEEE Std 1666-2011) supported by several
tools (e.g., Verilator) and can thus be used as an interface to such tools as
well as between multiple systems that are internally using custom
implementations.

Enabling CIRCT to emit SystemC code provides another way (next to Verilog
emission) to interface with the outside-world and at the same time
provides another way to simulate systems compiled with CIRCT.

## Lowering

In a first step, lowering from [HW](https://circt.llvm.org/docs/Dialects/HW/)
to the SystemC dialect will be implemented. A tool called ExportSystemC,
which is analogous to ExportVerilog, will then take these SystemC and
[Comb](https://circt.llvm.org/docs/Dialects/Comb/) operations to emit proper
SystemC-C++ code that can be compiled using clang, GCC, or any other
C++-compiler to produce the simulator binary. In the long run support for more
dialects can be added, such as LLHD and SV.

As a simple example we take a look at the following HW module which just adds
two numbers together:

```mlir
hw.module @adder (%a: i32, %b: i32) -> (c: i32) {
    %sum = comb.add %a, %b : i32
    hw.output %sum : i32
}
```

It will then be lowered to the following SystemC IR to make code emission
easier for ExportSystemC:

```mlir
systemc.module @adder(%a: i32, %b: i32) -> (%c: i32) {
    systemc.ctor {
        systemc.method @add
    }
    systemc.func @add() {
        // c = a + b
        %res = comb.add %a, %b : i32
        systemc.con %c, %res : i32
    }
}
```

ExportSystemC will then emit the following C++ code to be compiled by clang or
another C++-compiler:

```cpp
#ifndef ADDER_H
#define ADDER_H

#include <systemc.h>

SC_MODULE(adder) {
    sc_in<sc_uint<32>> a;
    sc_in<sc_uint<32>> b;
    sc_out<sc_uint<32>> c;

    SC_CTOR(adder) {
        SC_METHOD(add);
    }

    void add() {
        c = a + b;
    }
};

#endif // ADDER_H
```

## Interop

Given a design in the core dialects (HW, Seq, Comb) we want to be able to use
different backends for various parts/instances in the design. For example,
verilate an instance and export a wrapper around the verilated module in SystemC
or SystemVerilog, or simulate a design in a native CIRCT simulator and use
Verilator for nested black-box modules provided in Verilog, etc.

* SystemC <=> Verilator
* SystemVerilog <=> Verilator
* LLHD testbench <=> Verilator
* SystemC <=> ARC <=> Verilator

We start with a simple HW module called 'Foo' that just wraps another HW module
called 'Bar'.

```mlir
hw.module @Bar (%a: i32) -> (b: i32) {...}

hw.module @Foo (%x: i32) -> (y: i32) {
  %1 = hw.instance @Bar (a: %x: i32) -> (b: i32)
  hw.output %1 : i32
}
```

A pass can then mark this instance of `@Bar` to be, e.g.,
verilated, by replacing the `hw.instance` with `model.verilated`.

```mlir
hw.module @Foo (%x: i32) -> (y: i32) {
  %1 = systemc.model.verilated @Bar (a: %x: i32) -> (b: i32)
  hw.output %1 : i32
}
```

The wrapper is then lowered by the desired backend, in this example
`HWToSystemC`. Note that `@Bar` stays a `hw.module`. `HWToSystemC` handles the
`model.verilated` opaquely, but `systemc.module` has to implement an interface
for procedural interop (later there can also be a structural interop interface)
that
1. tells the wrapped instance which interop mechanisms it supports, e.g.,
   C foreign functions, C++, etc. It is important that these interop mechanisms
   have a strong type system that both parties use in the same way.
2. provides builders to create operations to store state, initialize the state,
   update the state, and deallocate the state given an interop mechanism.

```mlir
emitc.include <"systemc.h">

systemc.module @Foo (%x: !systemc.in<i32>, %y: !systemc.out<i32>) {
  systemc.ctor {
    systemc.method %func0
  }

  %func0 = systemc.func {
    %0 = systemc.signal.read %x : !systemc.in<i32>
    %1 = systemc.model.verilated @Bar (a: %0: i32) -> (b: i32)
    systemc.signal.write %y, %1 : !systemc.out<i32>
  }
}
```

Afterwards, the interop lowering pass is called which lowers all `model.*`
operations. Next to the interface implemented by `systemc.module`, it is
important that the `model.verilated` implements a complementary interface
that
1. provides an ordered list of interop machanisms it supports in the same way as
   the interface mentioned above
2. takes the ordered list of interop mechanisms the `systemc.module`
   understands and picks the first in the ordered list they both have in common
3. provides callbacks that take the builders of the above mentioned interface
   as arguments and use them to build operations according to the selected
   interop mechanism.

The result of this lowering pass will then look like the following.

```mlir
emitc.include <"systemc.h">
// The interop lowering pass can always insert new global operations. This also
// includes external C functions, etc.
emitc.include "VBar.h"

systemc.module @Foo (%x: !systemc.in<i32>, %y: !systemc.out<i32>) {
  // The interface implementation of systemc.module provided a builder with
  // this insertion point and the interface implementation of model.verilated
  // created this operation to store state.
  %vbar = systemc.cpp.variable : !emitc.ptr<!emitc.opaque<"VBar">>

  systemc.ctor {
    // The interface implementation of systemc.module provided a builder with
    // this insertion point and the interface implementation of model.verilated
    // created these two operations to initialize the state.
    %0 = systemc.cpp.new (optional args) : !emitc.ptr<!emitc.opaque<"VBar">>
    systemc.cpp.assign %vbar, %0 : !emitc.ptr<!emitc.opaque<"VBar">>

    systemc.method %func0
  }

  %func0 = systemc.func {
    %0 = systemc.signal.read %x : !systemc.in<i32>

    // State update logic created by the model.verilated callback. The
    // interface implementation of systemc.module just provided a new builder
    // with this location as insertion point (and the arguments).
    %1 = systemc.cpp.member_access ptr %vbar["a"] : !emitc.ptr<!emitc.opaque<"VBar">> -> i32
    systemc.cpp.assign %1, %0 : i32
    %2 = systemc.cpp.member_access ptr %vbar["eval"] : !emitc.ptr<!emitc.opaque<"VBar">> -> !systemc.func_handle
    systemc.call %2 ()
    %3 = systemc.cpp.member_access ptr %vbar["b"] : !emitc.ptr<!emitc.opaque<"VBar">> -> i32

    systemc.signal.write %y, %3 : !systemc.out<i32>
  }

  // Destructor built by the interface implementation of systemc.module to allow
  // embedding of deallocation logic. This would look different if the
  // model.verilated would have chosen C foreign functions as interface.
  systemc.cpp.destructor {
    // State deallocation logic created in the model.verilated callback
    // for C++ interop.
    systemc.cpp.delete %vbar : !emitc.ptr<!emitc.opaque<"VBar">>
  }
}
```

All the operations in the above code block are supported in `ExportSystemC` and
thus can be printed to the following:

```cpp
#ifndef HEADER_GUARD
#define HEADER_GUARD

#include <systemc.h>
#include "VBar.h"

SC_MODULE(Foo) {
  sc_in<sc_uint<32>> x;
  sc_out<sc_uint<32>> y;
  VBar* vbar;

  SC_CTOR(Foo) {
    vbar = new VBar();
    SC_METHOD(func0);
  }

  void func0() {
    vbar->a = x.read();
    vbar->eval();
    y.write(vbar->b);
  }

  ~Foo() {
    delete vbar;
  }
};

#endif // HEADER_GUARD
```

SV lowering example

```mlir
hw.module @Foo (%x: i32) -> (y: i32) {
  %1 = systemc.model.verilated @Bar (a: %x: i32) -> (b: i32)
  hw.output %1 : i32
}
```

SV should implement its own module operation.

```mlir
hw.module @Foo (%x: i32) -> (y: i32) {
  %vbar = sv.reg : !sv.chandle
  sv.initial {
    %0 = sv.dpi.call @bar_new() : !sv.chandle
    sv.bassign %vbar, %0 : !sv.chandle
  }
  %1 = sv.reg : i32
  sv.always_comb {
    %0 = sv.dpi.call @bar_eval(%vbar: !sv.chandle, %a: i32) -> i32
    sv.passign %b, %0 : i32
  }
  sv.initial {
    sv.dpi.call @bar_delete(%vbar) : (!sv.chandle) -> !sv.void
  }
  hw.output %1 : i32
}
```

LLHD lowering example

```mlir
hw.module @Foo (%x: i32) -> (y: i32) {
  %1 = systemc.model.verilated @Bar (a: %x: i32) -> (b: i32)  // @Bar is a hw.module
  hw.output %1 : i32
}
```

```mlir
llhd.entity @Foo (%x: !llhd.sig<i32>) -> (%y: !llhd.sig<i32>) {
  %vbar = llhd.sig "vbar" %0 : !emitc.ptr<!emitc.opaque<"VBar">>
  llhd.inst "new_initial" @new_initial () -> (%vbar) : () -> (!llhd.sig<!emitc.ptr<!emitc.opaque<"VBar">>>)
  llhd.inst "always_comb" @always_comb (%vbar, %x) -> (%y) : (!llhd.sig<!emitc.ptr<!emitc.opaque<"VBar">>>, !llhd.sig<i32>) -> (!llhd.sig<i32>)
  llhd.inst "delete_initial" @new_initial (%vbar) -> () : (!llhd.sig<!emitc.ptr<!emitc.opaque<"VBar">>>) -> ()
}

llhd.proc @new_initial () -> () {
  
}

llhd.proc @always_comb () -> () {

}

llhd.proc @delete_initial () -> () {

}
```

## Q&A

**Q: Why implement a custom module operation rather than using `hw.module`?**

In SystemC we want to model module outputs as arguments such that the SSA value
is already defined from the beginning which we can then assign to and reference.

**Q: Why implement a custom func operation rather than using `func.func`?**

An important difference compared to the `func.func` operation is that it
represents a member function (method) of a SystemC module, i.e., a C++ struct.
This leads to some implementation differences:
* Not isolated from above: we need to be able to access module fields such as
  the modules inputs, outputs, and signals
* Verified to have no arguments and void return type: this is a restriction
  from SystemC for the function to be passed to SC_METHOD, etc. This could 
  also be achieved with `func.func`, but would require us to write the verifier
  in `systemc.module` instead.
* Region with only a single basic block (structured control flow) and no
  terminator: using structured control-flow leads to easier code emission

**Q: How much of C++ does the SystemC dialect aim to model?**

As much as necessary, as little as possible. Completely capturing C++ in a
dialect would be a huge undertaking and way too much to 'just' achieve SystemC
emission. At the same time, it is not possible to not model any C++ at all,
because when only modeling SystemC specific constructs, the gap for
ExportSystemC to bridge would be too big (we want the printer to be as simple
as possible).

**Q: Why does `systemc.module` have a graph region rather than a SSACFG region?**

It contains a single graph region to allow flexible positioning of the fields,
constructor and methods to support different ordering styles (fields at top
or bottom, methods to be registered with SC_METHOD positioned after the
constructor, etc.) without requiring any logic in ExportSystemC. Program code
to change the emission style can thus be written as part of the lowering from
HW, as a pre-emission transformation, or anywhere else.
