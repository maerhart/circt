//===- InteropOpInterfaces.h - Declare interop op interfaces ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for dialect interop.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES_H
#define CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

namespace circt {
namespace systemc {
enum class InteropMechanism { CFFI, CPP };
} // namespace systemc
} // namespace circt

#include "circt/Dialect/SystemC/InteropOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES_H
