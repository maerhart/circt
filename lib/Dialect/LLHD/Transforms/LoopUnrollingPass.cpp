//===- LoopUnrollingPass.cpp - Implement Loop Unrolling Pass --------------===//
//
// Implement Pass to completely unroll ForOp loops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;

static Optional<APInt> getConstantIntegerValue(Value value) {
  if (auto constOp = value.getDefiningOp<ConstantOp>())
    return constOp.value().cast<IntegerAttr>().getValue();
  if (auto constOp = value.getDefiningOp<llhd::ConstOp>())
    return constOp.value().cast<IntegerAttr>().getValue();
  return None;
}

static Optional<uint64_t> getConstantTripCount(llhd::ForOp loop) {
  Optional<APInt> lowerBound = getConstantIntegerValue(loop.lowerBound());
  Optional<APInt> upperBound = getConstantIntegerValue(loop.upperBound());
  Optional<APInt> step = getConstantIntegerValue(loop.step());

  if (!(lowerBound && upperBound && step))
    return None;

  if (upperBound.getValue().ule(lowerBound.getValue()))
    return 0;

  if (step->isNullValue())
    return None;

  APInt difference = upperBound.getValue() - lowerBound.getValue();
  return llvm::APIntOps::RoundingUDiv(difference, step.getValue(),
                                      APInt::Rounding::UP)
      .getZExtValue();
}

namespace {

/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct LoopUnrollingPass : public llhd::LoopUnrollingBase<LoopUnrollingPass> {
  void runOnOperation() override;
};

struct UnrollLoopPattern : public OpRewritePattern<llhd::ForOp> {
  using OpRewritePattern<llhd::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(llhd::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (getConstantTripCount(op).hasValue()) {
      uint64_t tripCount = getConstantTripCount(op).getValue();
      if (tripCount == 0) {
        // Replace result values by input iter_args
        for (auto result : llvm::zip(op.results(), op.getIterOperands())) {
          std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
        }

        // Delete ForOp
        rewriter.eraseOp(op);
        return success();
      }
      Block *beforeBlock = op.getOperation()->getBlock();
      Block *forBlock = beforeBlock->splitBlock(op);
      BlockAndValueMapping mapping;
      OpBuilder builder = rewriter.atBlockEnd(beforeBlock);
      // Create constant for induction var equal to lower bound
      Value newConst = builder.create<llhd::ConstOp>(
          op.getLoc(), op.getInductionVar().getType(),
          builder.getIntegerAttr(
              op.getInductionVar().getType(),
              getConstantIntegerValue(op.lowerBound())->getZExtValue()));
      // Map induction var to above created replacement
      mapping.map(op.getInductionVar(), newConst);
      // Map iter_args to their correspondants in the parent region
      mapping.map(op.getRegionIterArgs(), op.getIterOperands());
      rewriter.cloneRegionBefore(op.getLoopBody(), *op.getParentRegion(),
                                 forBlock->getIterator(), mapping);
      // Update the lowerBound of the for loop to old lowerBound + step
      Value newLowerBound = builder.create<llhd::ConstOp>(
          op.getLoc(), op.getInductionVar().getType(),
          builder.getIntegerAttr(
              op.getInductionVar().getType(),
              getConstantIntegerValue(op.lowerBound())->getZExtValue() +
                  getConstantIntegerValue(op.step())->getZExtValue()));
      op.setLowerBound(newLowerBound);
      // Add Block arguments to the block containing the for loop to replace the
      // iter_args with the values returned by the yield terminators now outside
      // of the loop body and replace the yields with branches
      for (BlockArgument arg : op.getRegionIterArgs()) {
        forBlock->addArgument(arg.getType());
      }
      // Replace iter_args with the block arguments
      op.initArgsMutable().assign(forBlock->getArguments());
      // Collect yield ops created in procOp
      SmallVector<llhd::YieldOp, 4> yields;
      for (auto iter = op.getParentRegion()->op_begin();
           iter != op.getParentRegion()->op_end(); ++iter) {
        if (auto op = dyn_cast<llhd::YieldOp>(*iter))
          yields.push_back(op);
      }
      // Replace yields whith Branches
      while (!yields.empty()) {
        llhd::YieldOp yield = yields.pop_back_val();
        rewriter.setInsertionPointToEnd(yield.getOperation()->getBlock());
        rewriter.create<BranchOp>(yield.getLoc(), forBlock, yield.results());
        rewriter.eraseOp(yield);
      }

      // Add a BranchOp Terminator to the block created by the split.
      builder.create<BranchOp>(op.getLoc(), beforeBlock->getNextNode());
      return success();
    }
    return failure();
  }

  bool hasBoundedRewriteRecursion() const final { return true; }
};

} // end anonymous namespace

void LoopUnrollingPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<UnrollLoopPattern>(getOperation().getContext());
  llhd::ProcOp proc = getOperation();
  // proc.walk([&](llhd::ForOp loop) -> WalkResult {
  //   applyOpPatternsAndFold(loop, patterns);
  //   return failure();
  // });
  applyPatternsAndFoldGreedily(getOperation(), patterns);
  bool hasLoop = false;
  do {
    hasLoop = false;
    proc.walk([&](llhd::ForOp loop) { hasLoop = true; });
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  } while (hasLoop);
}

std::unique_ptr<OperationPass<llhd::ProcOp>>
mlir::llhd::createLoopUnrollingPass() {
  return std::make_unique<LoopUnrollingPass>();
}
