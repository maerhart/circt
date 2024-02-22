//===- SMTTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTTypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace smt;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SMT/SMTTypes.cpp.inc"

void SMTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SMT/SMTTypes.cpp.inc"
      >();
}
