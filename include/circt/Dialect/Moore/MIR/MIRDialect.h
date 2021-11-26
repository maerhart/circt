//===- MIRDialect.h - Moore MIR dialect declaration -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a Moore MIR MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIR_DIALECT_H
#define CIRCT_DIALECT_MOORE_MIR_DIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringSet.h"

// Pull in the dialect definition.
#include "circt/Dialect/Moore/MIR/MIRDialect.h.inc"

#endif // CIRCT_DIALECT_MOORE_MIR_DIALECT_H
