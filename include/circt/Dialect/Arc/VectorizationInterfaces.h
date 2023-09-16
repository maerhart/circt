//===- VectorizationInterfaces.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides registration functions for all external interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VECTORIZATIONINTERFACES_H
#define CIRCT_DIALECT_VECTORIZATIONINTERFACES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/PointerUnion.h"

// Forward declarations.
namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace circt {
namespace arc {

//===----------------------------------------------------------------------===//
// VectorizeOpInterface
//===----------------------------------------------------------------------===//

struct VectorizationOptions {
  enum class VectorizationKind { SIMD, Scalar };

  VectorizationKind kind;
};

class VectorizationBuilder {
public:
  explicit VectorizationBuilder(mlir::Operation *op);
  //  : localBuilder(op),
  //  globalBuilder(OpBuilder::atBlockBegin(&op.getParentOfType<ModuleOp>().getBody().front()))
  //  { }

  mlir::OpBuilder &localBuilder() { return lBuilder; }
  mlir::OpBuilder &globalBuilder() { return gBuilder; }

private:
  mlir::OpBuilder lBuilder;
  mlir::OpBuilder gBuilder;
};

} // namespace arc
} // namespace circt

#include "circt/Dialect/Arc/VectorizationInterfaces.h.inc"

#endif // CIRCT_DIALECT_VECTORIZATIONINTERFACES_H
