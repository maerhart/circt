//===- LoopUnrollingPass.cpp - Implement Loop Unrolling Pass --------------===//
//
// Implement Pass to completely unroll ForOp loops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace {

/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct InstanceInliningPass : public llhd::InstanceInliningBase<InstanceInliningPass> {
  void runOnOperation() override;
};

class InlinePattern : public OpRewritePattern<llhd::InstOp> {
  using OpRewritePattern<llhd::InstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(llhd::InstOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = cast<ModuleOp>(op.getParentOp()->getParentOp());
    llhd::EntityOp toInline =
        cast<llhd::EntityOp>(module.lookupSymbol(op.callee()));
    BlockAndValueMapping mapping;
    mapping.map(toInline.getArguments().take_front(toInline.ins().getZExtValue()), op.inputs());
    mapping.map(
        toInline.getArguments().take_back(toInline.getNumFuncArguments() -
                                          toInline.ins().getZExtValue()),
        op.outputs());
    // op.getParentRegion()->front().getTerminator()->dropAllReferences();
    // op.getParentRegion()->front().getTerminator()->erase();
    rewriter.cloneRegionBefore(toInline.body(), *op.getParentRegion(),
                               op.getParentRegion()->end());
    // ValueRange replacements(op.getArgOperands());
    op.getParentRegion()->front().getTerminator()->erase();
    rewriter.mergeBlocks(&op.getParentRegion()->back(),
                         &op.getParentRegion()->front(), op.getArgOperands());

    op.getOperation()->dropAllReferences();
    op.erase();
    // If it was the last reference to this entity in the module, delete the
    // entity
    // toInline.getSymbolUses(module);
    // if (toInline.symbolKnownUseEmpty(module)) {
    //   toInline.getOperation()->dropAllReferences();
    //   toInline.getOperation()->dropAllDefinedValueUses();
    //   toInline.erase();
    // }
    
    return success();
  }

  bool hasBoundedRewriteRecursion() const final { return true; }
};

} // end anonymous namespace

void InstanceInliningPass::runOnOperation() {
  // First add the parent entity's symbol name as a prefix to the signal names
  // in this entity
  ModuleOp module = getOperation();
  OpBuilder builder(module);
  module.walk([&](llhd::SigOp sig) {
    sig.setAttr("name", builder.getStringAttr(sig.getParentOfType<llhd::EntityOp>().getName().str() + "/" + sig.name().str()));
  });

  OwningRewritePatternList patterns;
  patterns.insert<InlinePattern>(getOperation().getContext());
  applyPatternsAndFoldGreedily(getOperation(), patterns);

  module.walk([](llhd::EntityOp entity) {
  // Move all regs and drives to the bottom
  std::vector<llhd::DrvOp> drives;
  std::vector<llhd::RegOp> regs;

  // Collect them
  entity.walk([&](llhd::DrvOp drv) { drives.push_back(drv); });
  entity.walk([&](llhd::RegOp reg) { regs.push_back(reg); });

  // Move them to the bottom
  for (auto drv : drives) {
    drv.getOperation()->moveBefore(entity.getBody().front().getTerminator());
  }
  for (auto reg : regs) {
    reg.getOperation()->moveBefore(entity.getBody().front().getTerminator());
  }
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createInstanceInliningPass() {
  return std::make_unique<InstanceInliningPass>();
}
