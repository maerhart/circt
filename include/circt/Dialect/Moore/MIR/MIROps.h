//===- MIROps.h - Declare Moore MIR dialect operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Moore MIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIR_OPS_H
#define CIRCT_DIALECT_MOORE_MIR_OPS_H

#include "circt/Dialect/Moore/MIR/MIRDialect.h"
#include "circt/Dialect/Moore/MIR/MIRTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Moore/MIR/MIREnums.h.inc"
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/Moore/MIR/MIR.h.inc"
#include "circt/Dialect/Moore/MIR/MIRStructs.h.inc"

#endif // CIRCT_DIALECT_MOORE_MIR_OPS_H
