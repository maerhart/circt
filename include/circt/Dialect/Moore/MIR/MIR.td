//===- SV.td - SystemVerilog dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIR_MIR
#define CIRCT_DIALECT_MOORE_MIR_MIR

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/Moore/MIR/MIRDialect.td"

// Base class for the operation in this dialect.
class MIROp<string mnemonic, list<OpTrait> traits = []> :
    Op<MIRDialect, mnemonic, traits>;

include "circt/Dialect/Moore/MIR/MIRTypesImpl.td"
include "circt/Dialect/Moore/MIR/MIRTypes.td"
include "circt/Dialect/Moore/MIR/MIRExpressions.td"
include "circt/Dialect/Moore/MIR/MIRStatements.td"

#endif // CIRCT_DIALECT_MOORE_MIR_MIR
