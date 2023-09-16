//===- VectorizationInterfacesImpl.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/Transforms/VectorizationInterfacesImpl.h"
#include "circt/Dialect/Arc/VectorizationInterfaces.h"
#include "circt/Dialect/Comb/CombOps.h"

using namespace circt;
using namespace comb;
using namespace arc;

namespace {
///
template <typename BitwiseOp>
struct VariadicBitwiseOpInterface
    : public arc::VectorizeOpInterface::ExternalModel<
          VariadicBitwiseOpInterface<BitwiseOp>, BitwiseOp> {

  bool isVectorizable(Operation *op, VectorizationKind kind,
                      SymbolTableCollection &symbolTable) const {
    return kind == VectorizationKind::Scalar;
  }
  Operation *vectorize(Operation *op, VectorizationKind kind,
                       SymbolTableCollection &symbolTable,
                       VectorizationBuilder &builder) const {
    auto bitwiseOp = cast<BitwiseOp>(op);
    return builder.localBuilder().create<BitwiseOp>(
        op->getLoc(), bitwiseOp.getInputs(), bitwiseOp.getTwoState());
  }
};

} // namespace

void circt::comb::registerVectorizeOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CombDialect *dialect) {
    AndOp::attachInterface<VariadicBitwiseOpInterface<AndOp>>(*ctx);
    OrOp::attachInterface<VariadicBitwiseOpInterface<OrOp>>(*ctx);
    XorOp::attachInterface<VariadicBitwiseOpInterface<XorOp>>(*ctx);
    AddOp::attachInterface<VariadicBitwiseOpInterface<AddOp>>(*ctx);
    MulOp::attachInterface<VariadicBitwiseOpInterface<MulOp>>(*ctx);
  });
}