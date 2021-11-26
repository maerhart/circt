//===- MIRDialect.td - Moore MIR dialect definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Moore MIR dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIR_MIRDIALECT
#define CIRCT_DIALECT_MOORE_MIR_MIRDIALECT

def MIRDialect : Dialect {
  let name = "moore";

  let summary = "Types and operations for Moore MIR dialect";
  let description = [{
    This dialect defines the `moore.mir` dialect, which represents various
    SystemVerilog-specific constructs without ambiguities and all types
    resolved.
  }];
  let cppNamespace = "::circt::moore::mir";
  let extraClassDeclaration = [{
    /// Register all Moore MIR types.
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_MOORE_MIR_MIRDIALECT
