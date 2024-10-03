//===- ResolveDynamicSignalAliasesPass.cpp - Implement RDSA Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to convert dynamic signal accessing operations to static
// variants.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-rdsa"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_RESOLVEDYNAMICSIGNALALIASES
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {
struct RDSAPass
    : public circt::llhd::impl::ResolveDynamicSignalAliasesBase<RDSAPass> {
  void runOnOperation() override;
};
} // namespace

static void handle(llhd::SigArrayGetOp op) {
  if (auto *defOp = op.getIndex().getDefiningOp();
      defOp && defOp->hasTrait<OpTrait::ConstantLike>())
    return;

  // TODO: add support
  if (std::distance(op->getUsers().begin(), op->getUsers().end()) != 1)
    return;

  auto drvOp = dyn_cast<llhd::DrvOp>(*op->getUsers().begin());
  if (!drvOp)
    return;

  OpBuilder builder(drvOp);
  Location loc = drvOp->getLoc();
  for (unsigned i = 0, e = op.getArrayType().getNumElements(); i < e; ++i) {
    Value idx = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerAttr(op.getIndex().getType(), i));
    Value sig = builder.create<llhd::SigArrayGetOp>(loc, op.getInput(), idx);
    Value isTheOne = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq,
                                                  op.getIndex(), idx);
    builder.create<llhd::DrvOp>(loc, sig, drvOp.getValue(), drvOp.getTime(),
                                isTheOne);
  }

  LLVM_DEBUG(llvm::dbgs() << "Resolved: " << op << "\n");

  drvOp->erase();
  op->erase();
}

static void handle(llhd::SigExtractOp op) {
  if (!cast<hw::InOutType>(op.getType()).getElementType().isSignlessInteger(1))
    return;

  if (auto *defOp = op.getLowBit().getDefiningOp();
      defOp && defOp->hasTrait<OpTrait::ConstantLike>())
    return;

  // TODO: add support
  if (std::distance(op->getUsers().begin(), op->getUsers().end()) != 1)
    return;

  auto drvOp = dyn_cast<llhd::DrvOp>(*op->getUsers().begin());
  if (!drvOp)
    return;

  OpBuilder builder(drvOp);
  Location loc = drvOp->getLoc();
  for (unsigned i = 0, e = op.getInputWidth(); i < e; ++i) {
    Value idx = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerAttr(op.getLowBit().getType(), i));
    Value sig = builder.create<llhd::SigExtractOp>(loc, op.getType(),
                                                   op.getInput(), idx);
    Value isTheOne = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq,
                                                  op.getLowBit(), idx);
    builder.create<llhd::DrvOp>(loc, sig, drvOp.getValue(), drvOp.getTime(),
                                isTheOne);
  }

  LLVM_DEBUG(llvm::dbgs() << "Resolved: " << op << "\n");

  drvOp->erase();
  op->erase();
}

void RDSAPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();

  SmallVector<llhd::SigArrayGetOp> getOps;
  SmallVector<llhd::SigExtractOp> extOps;
  module.walk([&](Operation *op) {
    if (auto sigOp = dyn_cast<llhd::SigArrayGetOp>(op)) {
      getOps.push_back(sigOp);
      return;
    }
    if (auto sigOp = dyn_cast<llhd::SigExtractOp>(op)) {
      extOps.push_back(sigOp);
      return;
    }
  });

  for (auto op : getOps)
    handle(op);
  for (auto op : extOps)
    handle(op);
}
