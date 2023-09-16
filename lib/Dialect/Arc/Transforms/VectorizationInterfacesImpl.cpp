//===- VectorizationInterfacesImpl.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Transforms/VectorizationInterfacesImpl.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/VectorizationInterfaces.h"

using namespace circt;
using namespace arc;

namespace {
///
struct VectorizeReturnOpInterface
    : public arc::VectorizeOpInterface::ExternalModel<
          VectorizeReturnOpInterface, VectorizeReturnOp> {
  Operation *vectorize(Operation *op, VectorizationKind kind,
                       SymbolTableCollection &symbolTable,
                       VectorizationBuilder &builder) const {
    auto returnOp = cast<VectorizeReturnOp>(op);
    return builder.localBuilder().create<VectorizeReturnOp>(
        op->getLoc(), returnOp.getOperand());
  }
};

///
struct StateOpInterface
    : public arc::VectorizeOpInterface::ExternalModel<StateOpInterface,
                                                      StateOp> {

  bool isVectorizable(Operation *op, VectorizationKind kind,
                      SymbolTableCollection &symbolTable) const {
    Operation *arcOp =
        cast<mlir::CallOpInterface>(op).resolveCallable(&symbolTable);
    auto arcVecOp = dyn_cast<VectorizeOpInterface>(arcOp);
    if (!arcVecOp)
      return false;
    return arcVecOp.isVectorizable(kind, symbolTable);
  }
  Operation *vectorize(Operation *op, VectorizationKind kind,
                       SymbolTableCollection &symbolTable,
                       VectorizationBuilder &builder) const {
    auto stateOp = cast<StateOp>(op);
    Operation *arcOp =
        cast<mlir::CallOpInterface>(op).resolveCallable(&symbolTable);
    auto arcVecOp = dyn_cast<VectorizeOpInterface>(arcOp);
    arcVecOp.vectorize(kind, symbolTable, builder);
    return builder.localBuilder().create<VectorizeReturnOp>(
        op->getLoc(), returnOp.getOperand());
  }
};

// TODO: create VectorizeOptions struct and make vectorization kind a field in
// it.
// TODO: pass OpAdaptor with new operands to use
// TODO: implement a caching mechanism for global symbol ops
// TODO: can' we pass a simple builder?

} // namespace

void circt::arc::registerVectorizeOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ArcDialect *dialect) {
    VectorizeReturnOp::attachInterface<VectorizeReturnOpInterface>(*ctx);
  });
}