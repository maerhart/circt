//===- MIRTypes.h - Declare Moore MIR dialect types --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the Moore MIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIR_TYPES_H
#define CIRCT_DIALECT_MOORE_MIR_TYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MIR/MIRTypes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MIR_TYPES_H
