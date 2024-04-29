//===- StructureOps.td - Process and Entity definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the LLHD Process, Entity and control flow MLIR ops.
//
//===----------------------------------------------------------------------===//

def ProcessOp : LLHD_Op<"process", [
    DeclareOpInterfaceMethods<PortList>,
    DeclareOpInterfaceMethods<HWModuleLike>,
    IsolatedFromAbove,
    Symbol, InnerSymbolTable, HasParent<"mlir::ModuleOp">
  ]> {
  let summary = "Create a process";
  let description = [{
    A `llhd.processess` represents control-flow in a timed fashion. It allows a
    procedural description of how a circuit's output signals change in
    reaction to changing input signals. It has a region with arbitrarily
    many basic blocks. The first block is the entry block and cannot be
    targeted by the terminators. It uses `llhd.wait` as a terminator to add
    timed control-flow. Immediate control-flow with `br` or `cond_br` is
    also possible. Every process must either contain an infinite loop or
    terminate with the `llhd.halt` terminator.

    How does a process compare to functions and entities?

    | Unit     | Paradigm     | Timing    | Models                         |
    |----------|--------------|-----------|--------------------------------|
    | Function | control-flow | immediate | Computation in zero time       |
    | Process  | control-flow | timed     | Behavioral circuit description |
    | Module   | data-flow    | timed     | Structural circuit description |

    Examples:

    ```mlir
    // out port direction not supported (yet)
    llhd.processess @example(in %in0 : i64, inout %out : i1) {
      br ^bb1
    ^bb1:
      llhd.halt
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<ModuleType>:$module_type,
                       OptionalAttr<DictArrayAttr>:$per_port_attrs,
                       OptionalAttr<LocationArrayAttr>:$result_locs,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<StrAttr>:$comment);
  let regions = (region MinSizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<hw::PortInfo>":$ports,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes,
                   CArg<"StringAttr", "{}">:$comment)>,
    OpBuilder<(ins "StringAttr":$name, "const hw::ModulePortInfo &":$ports,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes,
                   CArg<"StringAttr", "{}">:$comment,
		   CArg<"bool", "true">:$shouldEnsureTerminator)>,
  ];

  let extraClassDeclaration = [{
    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    void setPortSymbolAttr(size_t portIndex, ::circt::hw::InnerSymAttr sym);

    StringAttr getArgName(size_t index) {
      return getHWModuleType().getInputNameAttr(index);
    }
  }];

  let extraClassDefinition = [{
    hw::ModuleType $cppClass::getHWModuleType() {
      return getModuleType();
    }

    void $cppClass::setPortSymbolAttr(size_t portIndex, ::circt::hw::InnerSymAttr sym) {
      auto portSymAttr = StringAttr::get(getContext(), getPortSymbolAttrName());
      setPortAttr(portIndex, portSymAttr, sym);
    }

    size_t $cppClass::getNumPorts() {
      auto modty = getHWModuleType();
      return modty.getNumPorts();
    }

    size_t $cppClass::getNumInputPorts() {
      auto modty = getHWModuleType();
      return modty.getNumInputs();
    }

    size_t $cppClass::getNumOutputPorts() {
      auto modty = getHWModuleType();
      return modty.getNumOutputs();
    }

    size_t $cppClass::getPortIdForInputId(size_t idx) {
      auto modty = getHWModuleType();
      return modty.getPortIdForInputId(idx);
    }

    size_t $cppClass::getPortIdForOutputId(size_t idx) {
      auto modty = getHWModuleType();
      return modty.getPortIdForOutputId(idx);
    }

    SmallVector<::circt::hw::PortInfo> $cppClass::getPortList() {
      return ::getPortList(*this);
    }

    ::circt::hw::PortInfo $cppClass::getPort(size_t idx) {
      return ::getPort(*this, idx);
    }

  }];


  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLHD_ConnectOp : LLHD_Op<"con", [
    SameTypeOperands,
    HasParent<"hw::HWModuleOp">
  ]> {
  let summary = "Connect two signals.";
  let description = [{
    The `con` instruction connects two signals such that they essentially become
    one signal. All driven values on one signal will be reflected on the other.
  }];

  let arguments = (ins InOutType:$lhs,
                       InOutType:$rhs);

  let assemblyFormat = [{
    operands attr-dict `:` qualified(type($lhs))
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
//=== Control Flow Operations
//===----------------------------------------------------------------------===//

def LLHD_WaitOp : LLHD_Op<"wait", [
    Terminator,
    AttrSizedOperandSegments,
    HasParent<"ProcessOp">,
    DeclareOpInterfaceMethods<BranchOpInterface>
  ]> {
  let summary = "Suspends execution of a process.";
  let description = [{
    The `wait` instruction suspends execution of a process until any of the
    observed signals change or a fixed time interval has passed. Execution
    resumes at the specified basic block with the passed arguments.
    * This is a terminator instruction.
    * This instruction is only allowed in processes (`llhd.process`).

    Example:

    ```mlir
    llhd.wait ^bb1
    llhd.wait for %time, ^bb1(%time : !llhd.time)
    llhd.wait (%0, %1 : !hw.inout<i64>, !hw.inout<i1>), ^bb1(%1 : !hw.inout<i1>)
    llhd.wait for %time, (%0, %1 : !hw.inout<i64>, !hw.inout<i1>),
      ^bb1(%1, %0 : !hw.inout<i1>, !hw.inout<i64>)
    ```
  }];

  let arguments = (ins Variadic<InOutType>:$obs,
                       Optional<LLHD_TimeType>:$time,
                       Variadic<AnyType>:$destOps);

  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    (`for` $time^ `,`)? (`(`$obs^ `:` qualified(type($obs))`)` `,`)?
    $dest (`(` $destOps^ `:` qualified(type($destOps)) `)`)? attr-dict
  }];
}

def LLHD_HaltOp : LLHD_Op<"halt", [Terminator, HasParent<"ProcessOp">]> {
  let summary = "Terminates execution of a process.";
  let description = [{
    The `halt` instruction terminates execution of a process. All processes
    must halt eventually or consist of an infinite loop.

    * This is a terminator instruction
    * This instruction is only allowed in processes (`llhd.process`).

    Syntax:

    ```
    halt-op ::= `llhd.halt`
    ```

    Example:

    ```mlir
    llhd.halt
    ```
  }];

  let assemblyFormat = "attr-dict";
}
