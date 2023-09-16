//===- VectorizationInterfaces.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/VectorizationInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;

arc::VectorizationBuilder::VectorizationBuilder(mlir::Operation *op)
    : lBuilder(op), gBuilder(OpBuilder::atBlockBegin(
                        op->getParentOfType<ModuleOp>().getBody())) {}

#include "circt/Dialect/Arc/VectorizationInterfaces.cpp.inc"
