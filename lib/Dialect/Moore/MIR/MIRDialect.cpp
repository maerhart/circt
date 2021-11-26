//===- MIRDialect.cpp - Implement the Moore MIR dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore MIR dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MIR/MIRDialect.h"
#include "circt/Dialect/Moore/MIR/MIROps.h"
#include "circt/Dialect/Moore/MIR/MIRTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ManagedStatic.h"

using namespace circt;
using namespace circt::moore::mir;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void MIRDialect::initialize() {
  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Moore/MIR/MIR.cpp.inc"
      >();
}

#include "circt/Dialect/Moore/MIR/MIRDialect.cpp.inc"
